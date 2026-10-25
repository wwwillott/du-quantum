"""GPU (PyTorch) massive-restart brickwork search for exact Dicke states; float32 Adam on GPU, then the best
candidates are handed to the float64 CPU polisher (tools/brick_polish.py).

usage (dqgpu env): python tools/gpu_brick.py N n R C d [mode=real|complex] [batch=512] [rounds=4] [steps=2500] [anc=0]
Writes logs/gpu_cand/{N}_{n}_{R}x{C}_d{d}_{mode}.npz with the top candidates (params, losses).
"""
import sys, os, time, itertools, math
import numpy as np
import torch

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dicke_vec(N, n):
    psi = np.zeros(2 ** N)
    for c in itertools.combinations(range(N), n):
        psi[sum(1 << (N - 1 - q) for q in c)] = 1
    return psi / np.linalg.norm(psi)


def patch(N, R, C, full=False):
    cells = [(r, c) for r in range(R) for c in range(C)]
    if not full:
        cells = cells[:N]
    idx = {rc: i for i, rc in enumerate(cells)}
    classes = [[], [], [], []]
    for (r, c), i in idx.items():
        if (r, c + 1) in idx:
            classes[0 if c % 2 == 0 else 2].append((i, idx[(r, c + 1)]))
        if (r + 1, c) in idx:
            classes[1 if r % 2 == 0 else 3].append((i, idx[(r + 1, c)]))
    return cells, [cl for cl in classes if cl]


def diags(K, classes):
    idx = np.arange(2 ** K)
    bits = (idx[:, None] >> (K - 1 - np.arange(K))) & 1
    out = []
    for cl in classes:
        d = np.ones(2 ** K)
        for a, b in cl:
            d = d * (1 - 2 * (bits[:, a] & bits[:, b]))
        out.append(d)
    return np.array(out)


def apply_ry(psi, q, th, K):
    # psi: (B, 2^K) real; th: (B,)
    B = psi.shape[0]
    p = psi.view(B, 2 ** q, 2, 2 ** (K - q - 1))
    c = torch.cos(th / 2).view(B, 1, 1)
    s = torch.sin(th / 2).view(B, 1, 1)
    a, b = p[:, :, 0, :], p[:, :, 1, :]
    return torch.stack([c * a - s * b, s * a + c * b], dim=2).reshape(B, -1)


def apply_u3(pr, pi, q, P, K):
    # complex as (real, imag) pair; P: (B, 3) angles (alpha, theta, beta): U = Rz(b) Ry(t) Rz(a)
    B = pr.shape[0]
    al, t, be = P[:, 0].view(B, 1, 1), P[:, 1].view(B, 1, 1), P[:, 2].view(B, 1, 1)
    vr, vi = pr.view(B, 2 ** q, 2, 2 ** (K - q - 1)), pi.view(B, 2 ** q, 2, 2 ** (K - q - 1))

    def rz(xr, xi, ang):
        # diag(e^{-i ang/2}, e^{+i ang/2})
        c, s = torch.cos(ang / 2), torch.sin(ang / 2)
        r0 = xr[:, :, 0, :] * c + xi[:, :, 0, :] * s
        i0 = xi[:, :, 0, :] * c - xr[:, :, 0, :] * s
        r1 = xr[:, :, 1, :] * c - xi[:, :, 1, :] * s
        i1 = xi[:, :, 1, :] * c + xr[:, :, 1, :] * s
        return torch.stack([r0, r1], 2), torch.stack([i0, i1], 2)

    vr, vi = rz(vr, vi, al)
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    r0 = c * vr[:, :, 0, :] - s * vr[:, :, 1, :]
    r1 = s * vr[:, :, 0, :] + c * vr[:, :, 1, :]
    i0 = c * vi[:, :, 0, :] - s * vi[:, :, 1, :]
    i1 = s * vi[:, :, 0, :] + c * vi[:, :, 1, :]
    vr, vi = torch.stack([r0, r1], 2), torch.stack([i0, i1], 2)
    vr, vi = rz(vr, vi, be)
    return vr.reshape(B, -1), vi.reshape(B, -1)


def run(N, n, R, C, d, mode='real', batch=512, rounds=4, steps=2500, anc=False, lr=0.05, seed=0, keep=16):
    cells, classes = patch(N, R, C, full=anc)
    K = len(cells)
    D = torch.tensor(diags(K, classes), dtype=torch.float32, device=dev)
    tgt = torch.tensor(dicke_vec(N, n), dtype=torch.float32, device=dev)
    seq = [i % len(classes) for i in range(d)]
    npg = 1 if mode == 'real' else 3
    g = torch.Generator(device=dev).manual_seed(seed)
    allP, allL = [], []
    for rd in range(rounds):
        P = ((torch.rand((batch, d + 1, K, npg), generator=g, device=dev) * 2 - 1) * math.pi).requires_grad_(True)
        opt = torch.optim.Adam([P], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
        for it in range(steps):
            opt.zero_grad()
            if mode == 'real':
                psi = torch.zeros((batch, 2 ** K), device=dev)
                psi[:, 0] = 1
                for L in range(d + 1):
                    if L > 0:
                        psi = psi * D[seq[L - 1]]
                    for q in range(K):
                        psi = apply_ry(psi, q, P[:, L, q, 0], K)
                amp = psi.view(batch, 2 ** N, -1)
                ov = torch.einsum('s,bsa->ba', tgt, amp)
                loss = 1 - (ov ** 2).sum(1)
            else:
                pr = torch.zeros((batch, 2 ** K), device=dev); pr[:, 0] = 1
                pi = torch.zeros((batch, 2 ** K), device=dev)
                for L in range(d + 1):
                    if L > 0:
                        pr, pi = pr * D[seq[L - 1]], pi * D[seq[L - 1]]
                    for q in range(K):
                        pr, pi = apply_u3(pr, pi, q, P[:, L, q, :], K)
                ar = torch.einsum('s,bsa->ba', tgt, pr.view(batch, 2 ** N, -1))
                ai = torch.einsum('s,bsa->ba', tgt, pi.view(batch, 2 ** N, -1))
                loss = 1 - (ar ** 2 + ai ** 2).sum(1)
            loss.sum().backward()
            opt.step()
            sched.step()
        Lf = loss.detach().cpu().numpy()
        order = np.argsort(Lf)[:keep]
        allP.append(P.detach().cpu().numpy()[order]); allL.append(Lf[order])
        print(f'  round {rd}: best {Lf[order[:5]]}', flush=True)
    P = np.concatenate(allP); L = np.concatenate(allL)
    o = np.argsort(L)[:keep]
    return P[o], L[o], cells, classes, seq


if __name__ == '__main__':
    N, n, R, C, d = map(int, sys.argv[1:6])
    mode = sys.argv[6] if len(sys.argv) > 6 else 'real'
    batch = int(sys.argv[7]) if len(sys.argv) > 7 else 512
    rounds = int(sys.argv[8]) if len(sys.argv) > 8 else 4
    steps = int(sys.argv[9]) if len(sys.argv) > 9 else 2500
    anc = len(sys.argv) > 10 and sys.argv[10] == '1'
    t0 = time.time()
    P, L, cells, classes, seq = run(N, n, R, C, d, mode, batch, rounds, steps, anc, seed=d * 1000 + N)
    os.makedirs('logs/gpu_cand', exist_ok=True)
    fn = f'logs/gpu_cand/{N}_{n}_{R}x{C}{"a" if anc else ""}_d{d}_{mode}.npz'
    np.savez(fn, P=P, L=L, cells=np.array(cells), seq=np.array(seq), N=N, n=n, R=R, C=C, d=d, mode=mode, anc=anc,
             classes=np.array([np.array(c) for c in classes], dtype=object))
    print(f'D({N},{n}) {R}x{C} d={d} {mode}: top losses {L[:6]}  ({time.time()-t0:.0f}s, {dev}) -> {fn}', flush=True)
