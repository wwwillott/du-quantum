"""CZ-pruning search: start from long random CZ sequences that reach the Dicke state, then repeatedly try
deleting single CZs (batched, warm-started Adam) until no deletion keeps fidelity 1.

usage: python tools/prune.py N n shape [L0] [nstart] [mode]
shapes: path, sq (4-cycle + tail), g2xK (2 x ceil(N/2) ladder), star, plus (cross)
"""
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from synth import make_loss, polish, save_template, random_seqs

jax.config.update('jax_enable_x64', True)


def shape_edges(shape, N):
    if shape == 'path':
        return [(i, i + 1) for i in range(N - 1)]
    if shape == 'sq':
        return [(0, 1), (1, 2), (2, 3), (3, 0)] + [(i, i + 1) for i in range(3, N - 1)]
    if shape.startswith('g'):
        # grid R x C, nodes row-major, take first N nodes
        R, C = map(int, shape[1:].split('x'))
        idx = {}
        for r in range(R):
            for c in range(C):
                if len(idx) < N:
                    idx[(r, c)] = len(idx)
        E = []
        for (r, c), i in idx.items():
            for dr, dc in ((0, 1), (1, 0)):
                j = idx.get((r + dr, c + dc))
                if j is not None:
                    E.append((i, j))
        return E
    if shape == 'plus':
        E = [(0, i) for i in range(1, min(N, 5))]
        for i in range(5, N):
            E.append((i - 4, i))
        return E
    raise ValueError(shape)


def run_adam(loss, params0, seqs, steps, lr=0.03):
    vg = jax.jit(jax.vmap(jax.value_and_grad(loss), in_axes=(0, 0)))
    b1, b2, eps = 0.9, 0.999, 1e-12

    @jax.jit
    def step(p, m, v, t, s):
        val, g = vg(p, s)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        return p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps), m, v, val

    p = jnp.asarray(params0)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)
    s = jnp.asarray(np.asarray(seqs, dtype=np.int32))
    for t in range(1, steps + 1):
        p, m, v, val = step(p, m, v, t, s)
    val, _ = vg(p, s)
    return np.array(val), np.array(p)


def prune(N, n, edges, seq, params, mode, rng, steps=1500, restarts=2, thr=1e-8):
    L = len(seq)
    while L > 1:
        loss, _, npg = make_loss(N, n, L - 1, edges, mode)
        cands, P0 = [], []
        for t in range(L):
            s = seq[:t] + seq[t + 1:]
            # warm start: drop layer t+1 (merge approx)
            p = np.concatenate([params[:t + 1], params[t + 2:]], axis=0)
            for r in range(restarts):
                cands.append(s)
                P0.append(p + (0 if r == 0 else rng.normal(0, 0.3, p.shape)))
        val, P = run_adam(loss, np.array(P0), cands, steps)
        best = int(np.argmin(val))
        print(f'   L={L} round: best val {val[best]:.2e}, #<1e-3: {int(np.sum(val<1e-3))}', flush=True)
        if val[best] > thr:
            # try polishing the best couple before giving up
            ok = False
            for b in np.argsort(val)[:3]:
                if val[b] < 3e-2:
                    fv, pp = polish(loss, P[b], cands[b], iters=3000)
                    if fv < thr:
                        best, ok = b, True
                        P[b] = pp
                        break
            if not ok:
                return seq, params
        seq, params = cands[best], P[best]
        L -= 1
        print(f'   pruned to L={L} (loss {val[best]:.2e}) seq={seq}', flush=True)
    return seq, params


if __name__ == '__main__':
    N, n, shape = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    L0 = int(sys.argv[4]) if len(sys.argv) > 4 else 3 * N * n
    nstart = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    mode = sys.argv[6] if len(sys.argv) > 6 else 'real'
    edges = shape_edges(shape, N)
    rng = np.random.default_rng(int(time.time()) % 1000)
    loss, _, npg = make_loss(N, n, L0, edges, mode)
    seqs = random_seqs(edges, L0, nstart, rng)
    P0 = rng.uniform(-np.pi, np.pi, (len(seqs), L0 + 1, N, npg))
    t0 = time.time()
    val, P = run_adam(loss, P0, seqs, 3000)
    order = np.argsort(val)
    print(f'N={N} n={n} {shape} edges={edges} L0={L0}: start losses {val[order[:5]]} ({time.time()-t0:.0f}s)', flush=True)
    bestL, best = 10 ** 9, None
    for b in order[:4]:
        if val[b] > 1e-6:
            fv, pp = polish(loss, P[b], seqs[b], iters=3000)
            if fv > 1e-8:
                continue
            P[b] = pp
        s, p = prune(N, n, edges, list(seqs[b]), P[b], mode, rng)
        lossf, _, _ = make_loss(N, n, len(s), edges, mode)
        fv, p = polish(lossf, p, s, iters=3000)
        print(f'  start {b}: final L={len(s)} fid-loss={fv:.2e} seq={[edges[e] for e in s]}', flush=True)
        if fv < 1e-11 and len(s) < bestL:
            bestL, best = len(s), (s, p)
            save_template(N, n, shape, edges, s, p, mode, fv, tag='_pr')
    print(f'BEST N={N} n={n} {shape}: L={bestL}  ({time.time()-t0:.0f}s)', flush=True)
