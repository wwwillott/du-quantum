"""Depth-minimal exact Dicke synthesis on a fully parallel 2D brickwork (for the bonus: only moments count).

Patch: R x C grid (first N cells row-major). CZ layers cycle through edge colour classes
(H-even, V-even, H-odd, V-odd) restricted to the patch; a full 1q layer (U3, or Ry in real mode) sits before,
between and after the CZ layers -> moments = 2*d + 1. Batched Adam over random restarts, BFGS polish.

usage: python tools/brick.py N n R C dmin dmax [mode=real] [restarts=32] [steps=3000]
Saves data/brick/{N}_{n}_{R}x{C}_d{d}.json on success (loss < 1e-12).
"""
import sys, os, json, time, itertools
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update('jax_enable_x64', True)


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
    classes = [cl for cl in classes if cl]
    return cells, classes


def layer_diags(N, classes):
    idx = np.arange(2 ** N)
    bits = (idx[:, None] >> (N - 1 - np.arange(N))) & 1
    out = []
    for cl in classes:
        d = np.ones(2 ** N)
        for a, b in cl:
            d = d * (1 - 2 * (bits[:, a] & bits[:, b]))
        out.append(d)
    return np.array(out)


def make_loss(N, n, d, classes, mode, order=None, K=None):
    """K = total qubits (outputs 0..N-1 first, ancillas N..K-1)."""
    Nout = N
    N = N if K is None else K
    D = jnp.asarray(layer_diags(N, classes))
    target = jnp.asarray(dicke_vec(Nout, n))
    na = N - Nout
    seq = [i % len(classes) for i in range(d)] if order is None else order

    def gate(p):
        if mode == 'real':
            c, s = jnp.cos(p[0] / 2), jnp.sin(p[0] / 2)
            return jnp.array([[c, -s], [s, c]])
        al, t, b = p
        c, s = jnp.cos(t / 2), jnp.sin(t / 2)
        ryy = jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)
        return jnp.exp(-0.5j * b * jnp.array([1., -1.]))[:, None] * ryy * jnp.exp(-0.5j * al * jnp.array([1., -1.]))[None, :]

    def layer(psi, P):
        psi = psi.reshape((2,) * N)
        for q in range(N):
            psi = jnp.moveaxis(jnp.tensordot(gate(P[q]), psi, axes=([1], [q])), 0, q)
        return psi.reshape(-1)

    def state(P):
        psi = jnp.zeros(2 ** N, dtype=jnp.float64 if mode == 'real' else jnp.complex128).at[0].set(1.0)
        psi = layer(psi, P[0])
        for t in range(d):
            psi = layer(psi * D[seq[t]], P[t + 1])
        return psi

    def loss(P):
        if na == 0:
            return 1.0 - jnp.abs(jnp.vdot(target, state(P))) ** 2
        amp = target @ state(P).reshape(2 ** Nout, 2 ** na)
        return 1.0 - jnp.sum(jnp.abs(amp) ** 2)

    def residual(P):
        psi = state(P)
        ov = jnp.vdot(target, psi)
        ph = ov / jnp.abs(ov)
        r = psi * jnp.conj(ph) - target
        return jnp.concatenate([jnp.real(r), jnp.imag(r)]) if mode != 'real' else r

    loss.residual = residual
    return loss, seq


def lm_refine(residual, f, x0, shape, iters=200):
    from scipy.optimize import least_squares
    rf = jax.jit(lambda x: residual(x.reshape(shape)))
    jf = jax.jit(jax.jacfwd(lambda x: residual(x.reshape(shape))))
    try:
        r = least_squares(lambda x: np.asarray(rf(x)), x0, jac=lambda x: np.asarray(jf(x)), method='lm',
                          xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=iters * 10)
        x = r.x
    except Exception:
        x = x0
    return x, float(f(x.reshape(shape)))


def search(N, n, R, C, d, mode='real', restarts=32, steps=3000, lr=0.05, seed=0, anc=False):
    cells, classes = patch(N, R, C, full=anc)
    K = len(cells)
    loss, seq = make_loss(N, n, d, classes, mode, K=K)
    loss_res = loss.residual
    npg = 1 if mode == 'real' else 3
    key = jax.random.PRNGKey(seed)
    P = jax.random.uniform(key, (restarts, d + 1, K, npg), minval=-np.pi, maxval=np.pi)
    vg = jax.jit(jax.vmap(jax.value_and_grad(loss)))
    m = jnp.zeros_like(P); v = jnp.zeros_like(P)

    @jax.jit
    def step(P, m, v, t):
        val, g = vg(P)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        return P - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-12), m, v, val

    for t in range(1, steps + 1):
        P, m, v, val = step(P, m, v, t)
    val = np.array(val)
    order = np.argsort(val)
    best = None
    f = jax.jit(loss)
    g = jax.jit(jax.grad(loss))
    shape = P.shape[1:]
    for b in order[:int(os.environ.get('NPOL', '4'))]:
        if val[b] > 5e-2:
            break
        r = minimize(lambda x: float(f(x.reshape(shape))), np.array(P[b]).ravel(),
                     jac=lambda x: np.asarray(g(x.reshape(shape))).ravel(), method='BFGS',
                     options=dict(maxiter=int(os.environ.get('POLIT', '4000')), gtol=1e-16))
        x, fun = r.x, r.fun
        if 1e-13 < fun < 1e-2 and K == N:
            x, fun = lm_refine(loss_res, f, x, shape)
        if best is None or fun < best[0]:
            best = (fun, x.reshape(shape))
        if r.fun < 1e-13:
            break
    return val[order[:5]], best, cells, classes, seq


if __name__ == '__main__':
    N, n, R, C, dmin, dmax = map(int, sys.argv[1:7])
    mode = sys.argv[7] if len(sys.argv) > 7 else 'real'
    restarts = int(sys.argv[8]) if len(sys.argv) > 8 else 32
    steps = int(sys.argv[9]) if len(sys.argv) > 9 else 3000
    anc = len(sys.argv) > 10 and sys.argv[10] == 'anc'
    for d in range(dmin, dmax + 1):
        t0 = time.time()
        vals, best, cells, classes, seq = search(N, n, R, C, d, mode, restarts, steps, anc=anc)
        bl = best[0] if best else None
        print(f'D({N},{n}) {R}x{C} {mode} d={d} (moments {2*d+1}): adam best {vals[:3]}, polished {bl} ({time.time()-t0:.0f}s)', flush=True)
        if best and best[0] < 1e-12:
            os.makedirs('data/brick', exist_ok=True)
            json.dump(dict(N=N, n=n, R=R, C=C, d=d, mode=mode, cells=cells, classes=classes, seq=seq, anc=anc,
                           params=best[1].tolist(), loss=float(best[0])),
                      open(f'data/brick/{N}_{n}_{R}x{C}{"a" if anc else ""}_d{d}.json', 'w'))
            print('  SAVED', flush=True)
            break
