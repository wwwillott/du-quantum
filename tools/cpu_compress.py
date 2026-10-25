"""CPU-parallel global CZ compression of exact Dicke circuits (adjoint gradient + L-BFGS).

Same circuit form and file formats as tools/gpu_compress.py:
    U3 on every node, then for each CZ j=(a_j,b_j): CZ, U3(a_j), U3(b_j).   U3 = Rz(b) Ry(t) Rz(a)
Gradients use the adjoint method, recomputing earlier states by running the (unitary) circuit
backwards, so a gradient costs ~2 circuit passes. Each round tries deleting every CZ (refit from the
warm start) in a process pool and accepts the first exact refit (1 - F < 1e-11, checked independently).

usage (main python): python tools/cpu_compress.py SRC.json [--minutes 240] [--workers 6] [--outdir data/peep_gpu]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, json, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_compress import (zyz, u3_np, ops_to_slots, slots_to_ops, params_to_slots, slots_to_params,
                          delete_cz, dicke_vec, infid_np, load_ops, save_ops)

STOP = 'logs/gpu_compress/STOP'  # create this file to stop all runs after their current round
Zh = np.diag([-0.5j, 0.5j])  # d/da Rz(a) = Rz(a) @ Zh


def u3_and_grads(th):
    """th (S,3) -> U (S,2,2), dU (S,3,2,2)"""
    a, t, b = th[:, 0], th[:, 1], th[:, 2]
    c, s = np.cos(t / 2), np.sin(t / 2)
    ep, em = np.exp(-0.5j * (a + b)), np.exp(0.5j * (a - b))
    U = np.empty((len(th), 2, 2), complex)
    U[:, 0, 0], U[:, 0, 1] = c * ep, -s * em
    U[:, 1, 0], U[:, 1, 1] = s * em.conj(), c * ep.conj()
    dU = np.empty((len(th), 3, 2, 2), complex)
    dU[:, 0] = U @ Zh                      # a
    dU[:, 2] = Zh @ U                      # b
    dt = np.empty((len(th), 2, 2), complex)  # d/dt: replace c->-s/2, s->c/2
    dt[:, 0, 0], dt[:, 0, 1] = -0.5 * s * ep, -0.5 * c * em
    dt[:, 1, 0], dt[:, 1, 1] = 0.5 * c * em.conj(), -0.5 * s * ep.conj()
    dU[:, 1] = dt
    return U, dU


class Circ:
    def __init__(self, N, n, k, czs):
        self.N, self.n, self.k, self.czs = N, n, k, list(czs)
        idx = np.arange(2 ** k)
        bits = (idx[:, None] >> (k - 1 - np.arange(k))) & 1
        self.D = [(1 - 2 * (bits[:, a] & bits[:, b])).astype(float) for a, b in czs]
        self.t = dicke_vec(N, n)

    def _ap1(self, psi, q, U):
        return np.matmul(U, psi.reshape(2 ** q, 2, -1)).reshape(-1)

    def _env1(self, phi, w, q):
        """E[i,j] = sum conj(phi)[..i..] w[..j..] over the other qubits"""
        return np.tensordot(phi.reshape(2 ** q, 2, -1).conj(), w.reshape(2 ** q, 2, -1), axes=([0, 2], [0, 2]))

    def loss_grad(self, theta):
        k, m = self.k, len(self.czs)
        th = theta.reshape(-1, 3)
        U, dU = u3_and_grads(th)
        cols = U[:k, :, 0]
        psi = cols[0]
        for q in range(1, k):
            psi = np.outer(psi, cols[q]).reshape(-1)
        for j, (a, b) in enumerate(self.czs):
            psi = psi * self.D[j]
            psi = self._ap1(psi, a, U[k + 2 * j])
            psi = self._ap1(psi, b, U[k + 2 * j + 1])
        P = psi.reshape(2 ** self.N, -1)
        A = self.t @ P
        L = 1 - float(np.sum(np.abs(A) ** 2))
        phi = np.outer(self.t, A).reshape(-1)
        g = np.zeros_like(th)
        for j in range(m - 1, -1, -1):
            a, b = self.czs[j]
            Ua, Ub = U[k + 2 * j], U[k + 2 * j + 1]
            # psi = (Ub on b)(Ua on a) CZ chi
            w = self._ap1(psi, b, Ub.conj().T)          # = (Ua on a) CZ chi
            Eb = self._env1(phi, w, b)  # env of Ub with input w
            g[k + 2 * j + 1] = -2 * np.real(np.einsum('pij,ij->p', dU[k + 2 * j + 1], Eb))
            # w = Ua x ; x = CZ chi
            x = self._ap1(w, a, Ua.conj().T)
            phib = self._ap1(phi, b, Ub.conj().T)
            Ea = self._env1(phib, x, a)
            g[k + 2 * j] = -2 * np.real(np.einsum('pij,ij->p', dU[k + 2 * j], Ea))
            psi = x * self.D[j]
            phi = self._ap1(phib, a, Ua.conj().T) * self.D[j]
        # initial product layer
        for q in range(k):
            # environment vector e[i] = <phi| (|i> on q) (x) other columns>
            v = phi.reshape(2 ** q, 2, -1).conj()
            left = np.ones(1, complex)
            for r in range(q):
                left = np.outer(left, cols[r]).reshape(-1)
            right = np.ones(1, complex)
            for r in range(q + 1, k):
                right = np.outer(right, cols[r]).reshape(-1)
            e = np.einsum('xiy,x,y->i', v, left, right)
            g[q] = -2 * np.real(dU[q][:, :, 0] @ e)
        return L, g.reshape(-1)


GIVEUP = [(150, 3e-2), (400, 1e-3), (1000, 1e-5), (3000, 1e-9)]


def refit(N, n, k, czs, theta, maxiter=4000, giveup=GIVEUP):
    from scipy.optimize import minimize
    c = Circ(N, n, k, czs)
    it = [0]

    def cb(intermediate_result):
        it[0] += 1
        if any(it[0] >= gi and intermediate_result.fun > gl for gi, gl in giveup):
            raise StopIteration

    r = minimize(c.loss_grad, theta, jac=True, method='L-BFGS-B', callback=cb,
                 options=dict(maxiter=maxiter, maxcor=50, ftol=1e-30, gtol=1e-16, maxfun=maxiter * 2))
    return r.x, float(r.fun)


def _finish(N, n, k, czs2, th, L):
    """long continuation for near misses; returns (init, post, F) if exact else None"""
    if 1e-11 <= L < 1e-5:
        th, L = refit(N, n, k, czs2, th, maxiter=40000, giveup=[(20000, 1e-9)])
    if L < 1e-11:
        i2, p2 = params_to_slots(k, czs2, th)
        F = infid_np(N, n, k, czs2, i2, p2)
        if abs(F) < 1e-11:
            return i2, p2, F
    return None, None, L


def _worker(args):
    N, n, k, czs, init, post, j, seed, restarts = args
    rng = np.random.default_rng(seed)
    czs2, init2, post2 = delete_cz(k, czs, init, post, j)
    th0 = slots_to_params(k, czs2, init2, post2)
    best = (None, 1.0)
    for r in range(restarts):
        th = th0 if r == 0 else th0 + rng.normal(0, 0.05 * r, th0.shape)
        th, L = refit(N, n, k, czs2, th)
        if L < best[1]:
            best = (th, L)
        if L < 1e-13:
            break
    i2, p2, F = _finish(N, n, k, czs2, *best)
    if i2 is not None:
        return j, czs2, i2, p2, F
    return j, None, None, None, F


def _move_worker(args):
    """re-route CZ j onto edge e (same CZ count) and refit"""
    N, n, k, czs, init, post, j, e, seed = args
    czs2, init2, post2 = delete_cz(k, czs, init, post, j)
    czs2 = czs2[:j] + [tuple(e)] + czs2[j:]
    post2 = post2[:j] + [[np.eye(2, dtype=complex), np.eye(2, dtype=complex)]] + post2[j:]
    th0 = slots_to_params(k, czs2, init2, post2)
    th, L = refit(N, n, k, czs2, th0)
    i2, p2, F = _finish(N, n, k, czs2, th, L)
    if i2 is not None:
        return (j, e), czs2, i2, p2, F
    return (j, e), None, None, None, F


def _first_success(ex, fn, jobs, t_end):
    futs = [ex.submit(fn, a) for a in jobs]
    hit, losses = None, []
    for f in as_completed(futs):
        res = f.result()
        if res[1] is not None:
            hit = res
            break
        losses.append(res[4])
        if time.time() > t_end or os.path.exists(STOP):
            break
    for f in futs:
        f.cancel()
    return hit, losses


def compress(N, n, k, czs, init, post, minutes, workers, outdir, tag, restarts, log, seed=0, edges=None,
             max_moves=6, moves_per_phase=None):
    t_end = time.time() + 60 * minutes
    log(f'D({N},{n}) k={k} start CZ={len(czs)} infid={infid_np(N, n, k, czs, init, post):.2e}')
    rng = np.random.default_rng(seed)
    edges = sorted({tuple(sorted(e)) for e in (edges or [])} | {tuple(sorted(c)) for c in czs})
    log(f'  move edge set: {len(edges)} edges')
    moves_left = max_moves
    with ProcessPoolExecutor(workers) as ex:
        while time.time() < t_end and czs and not os.path.exists(STOP):
            m = len(czs)
            t0 = time.time()
            jobs = [(N, n, k, czs, init, post, int(j), int(rng.integers(1 << 30)), restarts) for j in rng.permutation(m)]
            hit, losses = _first_success(ex, _worker, jobs, t_end)
            if hit is not None:
                j, c2, i2, p2, F = hit
                log(f'  m={m}: delete cz {j} {czs[j]} -> CZ={len(c2)} infid={F:.1e} ({time.time() - t0:.0f}s, '
                    f'{len(losses)} failed first)')
                czs, init, post = c2, i2, p2
                save_ops(os.path.join(outdir, f'{N}_{n}_{tag}_cz{len(czs)}.json'), N, n, k, czs, init, post, tag, F)
                moves_left = max_moves
                continue
            log(f'  m={m}: no deletion ({time.time() - t0:.0f}s); best losses {np.array(sorted(losses)[:4])}')
            if moves_left <= 0:
                break
            moves_left -= 1
            t0 = time.time()
            cand = [(j, e) for j in range(m) for e in edges if e != tuple(sorted(czs[j]))]
            pick = rng.permutation(len(cand))[:moves_per_phase or 4 * workers]
            jobs = [(N, n, k, czs, init, post, cand[i][0], cand[i][1], int(rng.integers(1 << 30))) for i in pick]
            hit, losses = _first_success(ex, _move_worker, jobs, t_end)
            if hit is None:
                log(f'  move phase failed ({time.time() - t0:.0f}s)')
                continue
            (j, e), c2, i2, p2, F = hit
            log(f'  move cz {j} {czs[j]} -> {e} accepted ({time.time() - t0:.0f}s)')
            czs, init, post = c2, i2, p2
    log(f'FINAL D({N},{n}) CZ={len(czs)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--minutes', type=float, default=240)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--restarts', type=int, default=1)
    ap.add_argument('--tag', default='cpu')
    ap.add_argument('--outdir', default='data/peep_gpu')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    N, n, k, ops = load_ops(a.src)
    czs, init, post = ops_to_slots(k, ops)
    os.makedirs('logs/gpu_compress', exist_ok=True)
    lf = open(f'logs/gpu_compress/{N}_{n}_{a.tag}.log', 'a')

    def log(s):
        print(s, flush=True)
        lf.write(s + '\n')
        lf.flush()

    d = json.load(open(a.src))
    edges = None
    if 'coords' in d:  # Willow couplers are grid neighbours: allow moves onto any neighbouring node pair
        co = d['coords']
        edges = [(i, j) for i in range(k) for j in range(i + 1, k)
                 if abs(co[i][0] - co[j][0]) + abs(co[i][1] - co[j][1]) == 1]
    compress(N, n, k, czs, init, post, a.minutes, a.workers, a.outdir, a.tag, a.restarts, log, a.seed, edges=edges)
