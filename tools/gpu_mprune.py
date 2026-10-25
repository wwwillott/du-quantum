"""GPU warm-start MOMENT pruning for exact bonus circuits (unitary, no ancillas).

Representation: M moments; moment m = CZ mask over edge list E + U3 on every qubit not in a CZ that moment.
Candidate moves (all batched on the GPU):
  * del: delete moment m (drop its CZs; its 1q gates merged into the qubit's next free slot)
  * mov: delete moment m but relocate each of its CZs to m-1 or m+1 where both qubits are CZ-free
Each candidate is re-optimised from the warm start (float32 Adam), best ones polished in float64 (LBFGS);
accepted if infidelity < 1e-12. Repeats until no candidate works. Saves data/bonus_pruned/{N}_{n}_m{M}.json.

usage (dqgpu env): python tools/gpu_mprune.py circuit.json [steps=800] [maxrounds=200]
circuit.json produced by tools/export_bonus.py (fields: N, n, qubits, outputs, moments=[[('cz',a,b)|('u',q,[a,t,b])]])
"""
import sys, os, json, time, math, itertools
import numpy as np
import torch

dev = torch.device('cuda')


def dicke_vec(N, n):
    psi = np.zeros(2 ** N)
    for c in itertools.combinations(range(N), n):
        psi[sum(1 << (N - 1 - q) for q in c)] = 1
    return psi / np.linalg.norm(psi)


class Circ:
    def __init__(self, K, edges, masks, P):
        self.K, self.edges = K, edges          # edges: list of (a,b)
        self.masks = masks                     # (M, E) int
        self.P = P                             # (M, K, 3) float (alpha, theta, beta); identity = zeros

    def copy(self):
        return Circ(self.K, self.edges, self.masks.copy(), self.P.copy())

    def busy(self, m):
        b = set()
        for e in np.nonzero(self.masks[m])[0]:
            b.update(self.edges[e])
        return b


def u3_mat(p):
    a, t, b = p
    rz = lambda x: np.diag([np.exp(-0.5j * x), np.exp(0.5j * x)])
    c, s = math.cos(t / 2), math.sin(t / 2)
    return rz(b) @ np.array([[c, -s], [s, c]]) @ rz(a)


def zyz(U):
    U = U / np.sqrt(np.linalg.det(U))
    t = 2 * np.arctan2(abs(U[1, 0]), abs(U[0, 0]))
    if abs(U[1, 0]) < 1e-12:
        return np.array([0.0, t, 2 * np.angle(U[1, 1])])
    if abs(U[1, 1]) < 1e-12:
        return np.array([0.0, t, 2 * np.angle(U[1, 0])])
    s, d = np.angle(U[1, 1]), np.angle(U[1, 0])
    return np.array([s - d, t, s + d])


def merge_into(c, m, q, U, direction):
    """multiply U into qubit q's gate at the nearest CZ-free moment after (direction=+1) / before m."""
    j = m + direction
    while 0 <= j < c.masks.shape[0]:
        if q not in c.busy(j):
            V = u3_mat(c.P[j, q])
            c.P[j, q] = zyz(V @ U if direction > 0 else U @ V)
            return True
        j += direction
    return False


def delete_moment(c, m, relocate):
    c = c.copy()
    czs = [c.edges[e] for e in np.nonzero(c.masks[m])[0]]
    if relocate:
        for (a, b) in czs:
            placed = False
            for j in (m - 1, m + 1):
                if 0 <= j < c.masks.shape[0] and a not in c.busy(j) and b not in c.busy(j):
                    e = c.edges.index((a, b))
                    c.masks[j, e] = 1
                    c.P[j, a] = 0; c.P[j, b] = 0
                    placed = True
                    break
            if not placed:
                return None
    for q in range(c.K):
        if q in {x for e in czs for x in e}:
            continue
        U = u3_mat(c.P[m, q])
        if not merge_into(c, m, q, U, +1):
            merge_into(c, m, q, U, -1)
    c.masks = np.delete(c.masks, m, axis=0)
    c.P = np.delete(c.P, m, axis=0)
    return c


def build_torch(K, edges, masks_batch, dtype):
    """returns diag tensor (B, M, 2^K) and free mask (B, M, K)"""
    idx = np.arange(2 ** K)
    bits = (idx[:, None] >> (K - 1 - np.arange(K))) & 1
    AB = np.array([bits[:, a] & bits[:, b] for a, b in edges], dtype=np.float32)  # (E, 2^K)
    AB_t = torch.tensor(AB, device=dev)
    Mb = torch.tensor(np.array(masks_batch), dtype=torch.float32, device=dev)          # (B, M, E)
    par = torch.einsum('bme,es->bms', Mb, AB_t) % 2
    diag = (1 - 2 * par).to(dtype)
    busy = np.zeros(Mb.shape[:2] + (K,), dtype=bool)
    for bi, mk in enumerate(masks_batch):
        for m in range(mk.shape[0]):
            for e in np.nonzero(mk[m])[0]:
                busy[bi, m, list(edges[e])] = True
    return diag, torch.tensor(~busy, device=dev)


def forward(P, diag, free, K, dtype):
    """P: (B, M, K, 3) real; returns (real, imag) of final state. Gates built once, applied as batched matmuls."""
    B, M = P.shape[0], P.shape[1]
    cd = torch.complex64 if dtype == torch.float32 else torch.complex128
    f = free.to(dtype)
    al, t, be = P[..., 0] * f, P[..., 1] * f, P[..., 2] * f
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    ea, eb = torch.exp(-0.5j * al.to(cd)), torch.exp(-0.5j * be.to(cd))
    # U = Rz(b) Ry(t) Rz(a): [[eb*c*ea, -eb*s*conj(ea)], [conj(eb)*s*ea, conj(eb)*c*conj(ea)]]
    U = torch.stack([torch.stack([eb * c * ea, -eb * s * ea.conj()], -1),
                     torch.stack([eb.conj() * s * ea, eb.conj() * c * ea.conj()], -1)], -2)  # (B,M,K,2,2)
    psi = torch.zeros((B, 2 ** K), dtype=cd, device=dev); psi[:, 0] = 1
    dg = diag.to(cd)
    for m in range(M):
        for q in range(K):
            v = psi.view(B, 2 ** q, 2, -1)
            psi = torch.einsum('bij,bajr->bair', U[:, m, q], v).reshape(B, -1)
        psi = psi * dg[:, m]
    return psi.real, psi.imag


def infid(P, diag, free, K, tgt, dtype):
    pr, pi = forward(P, diag, free, K, dtype)
    ar, ai = pr @ tgt, pi @ tgt
    return 1 - (ar ** 2 + ai ** 2)


def optimise(cands, K, edges, tgt_np, steps, lr=0.01):
    diag, free = build_torch(K, edges, [c.masks for c in cands], torch.float32)
    tgt = torch.tensor(tgt_np, dtype=torch.float32, device=dev)
    P = torch.tensor(np.array([c.P for c in cands]), dtype=torch.float32, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([P], lr=lr)
    for it in range(steps):
        opt.zero_grad()
        L = infid(P, diag, free, K, tgt, torch.float32)
        L.sum().backward()
        opt.step()
    with torch.no_grad():
        L = infid(P, diag, free, K, tgt, torch.float32)
    return P.detach().cpu().numpy(), L.cpu().numpy()


def polish64(c, K, edges, tgt_np, iters=400):
    diag, free = build_torch(K, edges, [c.masks], torch.float64)
    tgt = torch.tensor(tgt_np, dtype=torch.float64, device=dev)
    P = torch.tensor(c.P[None], dtype=torch.float64, device=dev).requires_grad_(True)
    opt = torch.optim.LBFGS([P], lr=1, max_iter=iters, tolerance_grad=1e-16, tolerance_change=1e-20,
                            history_size=50, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        L = infid(P, diag, free, K, tgt, torch.float64).sum()
        L.backward()
        return L

    for _ in range(4):
        opt.step(closure)
    with torch.no_grad():
        L = float(infid(P, diag, free, K, tgt, torch.float64)[0])
    c = c.copy(); c.P = P.detach().cpu().numpy()[0]
    return c, L


def save(c, meta, path):
    moments = []
    for m in range(c.masks.shape[0]):
        busy = c.busy(m)
        ops = [['cz', int(a), int(b)] for e, (a, b) in enumerate(c.edges) if c.masks[m, e]]
        ops += [['u', q, [float(x) for x in c.P[m, q]]] for q in range(c.K) if q not in busy and np.any(np.abs(c.P[m, q]) > 1e-12)]
        moments.append(ops)
    json.dump(dict(meta, moments=moments, M=len(moments)), open(path, 'w'))


if __name__ == '__main__':
    src = json.load(open(sys.argv[1]))
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    maxrounds = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    N, n, K = src['N'], src['n'], len(src['qubits'])
    assert src['outputs'] == list(range(N)) and K == N, 'expects no-ancilla circuits with outputs 0..N-1'
    edges = sorted({tuple(sorted((op[1], op[2]))) for mo in src['moments'] for op in mo if op[0] == 'cz'})
    M = len(src['moments'])
    masks = np.zeros((M, len(edges)), dtype=int)
    P = np.zeros((M, K, 3))
    for m, mo in enumerate(src['moments']):
        for op in mo:
            if op[0] == 'cz':
                masks[m, edges.index(tuple(sorted((op[1], op[2]))))] = 1
            else:
                P[m, op[1]] = op[2]
    cur = Circ(K, edges, masks, P)
    tgt = dicke_vec(N, n)
    _, L0 = optimise([cur], K, edges, tgt, 0)
    print(f'D({N},{n}): start moments={M}, start infidelity(float32)={L0[0]:.2e}', flush=True)
    os.makedirs('data/bonus_pruned', exist_ok=True)
    t0 = time.time()
    for rd in range(maxrounds):
        Mc = cur.masks.shape[0]
        cands, labels = [], []
        for m in range(Mc):
            for rel in (False, True):
                cc = delete_moment(cur, m, rel)
                if cc is not None:
                    cands.append(cc); labels.append((m, rel))
        best = None
        for i0 in range(0, len(cands), 64):
            chunk = cands[i0:i0 + 64]
            Popt, L = optimise(chunk, K, edges, tgt, steps)
            for j in np.argsort(L)[:3]:
                if L[j] < 1e-4:
                    cc = chunk[j].copy(); cc.P = Popt[j]
                    cc, L64 = polish64(cc, K, edges, tgt)
                    if L64 < 1e-12:
                        best = (L64, cc, labels[i0 + j]); break
            if best:
                break
        if best is None:
            print(f'  no further deletion works at {Mc} moments ({time.time()-t0:.0f}s)', flush=True)
            break
        cur = best[1]
        print(f'  round {rd}: removed moment {best[2]} -> {cur.masks.shape[0]} moments, infid={best[0]:.1e} ({time.time()-t0:.0f}s)', flush=True)
        save(cur, dict(N=N, n=n, qubits=src['qubits'], outputs=src['outputs']), f'data/bonus_pruned/{N}_{n}_m{cur.masks.shape[0]}.json')
    print(f'FINAL D({N},{n}): {M} -> {cur.masks.shape[0]} moments', flush=True)
