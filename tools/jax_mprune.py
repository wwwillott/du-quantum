"""JAX (CPU, jit) version of the warm-start moment pruner (same moves as gpu_mprune.py, which is too launch-bound
on Windows GPUs). usage: python tools/jax_mprune.py export.json [steps=400] [maxrounds=200]"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
import gpu_mprune as G   # pure-numpy helpers: Circ, delete_moment, zyz, u3_mat, dicke_vec, save
jax.config.update('jax_enable_x64', True)


def make_fns(K, edges, tgt):
    idx = np.arange(2 ** K)
    bits = (idx[:, None] >> (K - 1 - np.arange(K))) & 1
    AB = jnp.asarray(np.array([bits[:, a] & bits[:, b] for a, b in edges], dtype=np.float64))
    T = jnp.asarray(tgt)

    def state(P, mask, free):
        # P (M,K,3), mask (M,E), free (M,K)
        diag = 1 - 2 * jnp.mod(mask @ AB, 2)          # (M, 2^K)
        Pm = P * free[..., None]

        def gate(p):
            a, t, b = p[0], p[1], p[2]
            c, s = jnp.cos(t / 2), jnp.sin(t / 2)
            ea, eb = jnp.exp(-0.5j * a), jnp.exp(-0.5j * b)
            return jnp.array([[eb * c * ea, -eb * s * jnp.conj(ea)], [jnp.conj(eb) * s * ea, jnp.conj(eb) * c * jnp.conj(ea)]])

        def body(psi, x):
            Pk, dg = x
            psi = psi.reshape((2,) * K)
            for q in range(K):
                psi = jnp.moveaxis(jnp.tensordot(gate(Pk[q]), psi, axes=([1], [q])), 0, q)
            return psi.reshape(-1) * dg, None

        psi0 = jnp.zeros(2 ** K, dtype=jnp.complex128).at[0].set(1.0)
        psi, _ = jax.lax.scan(body, psi0, (Pm, diag))
        return psi

    def loss(P, mask, free):
        return 1.0 - jnp.abs(jnp.vdot(T, state(P, mask, free))) ** 2

    vg = jax.jit(jax.vmap(jax.value_and_grad(loss), in_axes=(0, 0, 0)))
    f1 = jax.jit(loss)
    g1 = jax.jit(jax.grad(loss))
    return vg, f1, g1


def free_of(c):
    fr = np.ones((c.masks.shape[0], c.K))
    for m in range(c.masks.shape[0]):
        for q in c.busy(m):
            fr[m, q] = 0
    return fr


def adam(vg, cands, steps, lr=0.01):
    P = jnp.asarray(np.array([c.P for c in cands]))
    Mk = jnp.asarray(np.array([c.masks for c in cands], dtype=np.float64))
    Fr = jnp.asarray(np.array([free_of(c) for c in cands]))
    m = jnp.zeros_like(P); v = jnp.zeros_like(P)

    @jax.jit
    def step(P, m, v, t):
        val, g = vg(P, Mk, Fr)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        return P - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-14), m, v, val

    for t in range(1, steps + 1):
        P, m, v, val = step(P, m, v, t)
    val, _ = vg(P, Mk, Fr)
    return np.array(P), np.array(val)


def polish(f1, g1, c):
    Mk = jnp.asarray(c.masks.astype(np.float64)); Fr = jnp.asarray(free_of(c)); sh = c.P.shape
    r = minimize(lambda x: float(f1(jnp.asarray(x.reshape(sh)), Mk, Fr)), c.P.ravel(),
                 jac=lambda x: np.asarray(g1(jnp.asarray(x.reshape(sh)), Mk, Fr)).ravel(), method='BFGS',
                 options=dict(maxiter=5000, gtol=1e-16))
    cc = c.copy(); cc.P = r.x.reshape(sh)
    return cc, float(r.fun)


if __name__ == '__main__':
    src = json.load(open(sys.argv[1]))
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    maxrounds = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    N, n, K = src['N'], src['n'], len(src['qubits'])
    assert K == N
    edges = sorted({tuple(sorted((op[1], op[2]))) for mo in src['moments'] for op in mo if op[0] == 'cz'})
    M = len(src['moments'])
    masks = np.zeros((M, len(edges)), dtype=int); P = np.zeros((M, K, 3))
    for m, mo in enumerate(src['moments']):
        for op in mo:
            if op[0] == 'cz':
                masks[m, edges.index(tuple(sorted((op[1], op[2]))))] = 1
            else:
                P[m, op[1]] = op[2]
    cur = G.Circ(K, edges, masks, P)
    tgt = G.dicke_vec(N, n)
    t0 = time.time()
    print(f'D({N},{n}): start moments={M}', flush=True)
    os.makedirs('data/bonus_pruned', exist_ok=True)
    for rd in range(maxrounds):
        Mc = cur.masks.shape[0]
        vg, f1, g1 = make_fns(K, edges, tgt)   # shapes depend on Mc-1 only via inputs; jit retraces per shape
        cands, labels = [], []
        for m in range(Mc):
            for rel in (True, False):
                cc = G.delete_moment(cur, m, rel)
                if cc is not None:
                    cands.append(cc); labels.append((m, rel))
        Popt, L = adam(vg, cands, steps)
        best = None
        for j in np.argsort(L)[:4]:
            if L[j] < 1e-3:
                cc = cands[j].copy(); cc.P = Popt[j]
                cc, L64 = polish(f1, g1, cc)
                if L64 < 1e-13:
                    best = (L64, cc, labels[j]); break
        if best is None:
            print(f'  stuck at {Mc} moments; best adam loss {np.min(L):.2e} ({time.time()-t0:.0f}s)', flush=True)
            break
        cur = best[1]
        print(f'  round {rd}: removed moment {best[2]} -> {cur.masks.shape[0]} moments, infid={best[0]:.1e} ({time.time()-t0:.0f}s)', flush=True)
        G.save(cur, dict(N=N, n=n, qubits=src['qubits'], outputs=src['outputs']), f'data/bonus_pruned/{N}_{n}_m{cur.masks.shape[0]}.json')
        jax.clear_caches()
    print(f'FINAL D({N},{n}): {M} -> {cur.masks.shape[0]} moments', flush=True)
