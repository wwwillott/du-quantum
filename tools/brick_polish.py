"""Polish GPU brickwork candidates (logs/gpu_cand/*.npz) to exactness in float64 (JAX + SciPy BFGS/LM).
Saves exact ones (loss < 1e-12) to data/brick/*.json for the submission builder.
usage: python tools/brick_polish.py [glob=logs/gpu_cand/*.npz] [thr=5e-3]
"""
import sys, os, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from brick import make_loss, lm_refine

jax.config.update('jax_enable_x64', True)


def polish_file(fn, thr):
    z = np.load(fn, allow_pickle=True)
    N, n, d, mode, anc = int(z['N']), int(z['n']), int(z['d']), str(z['mode']), bool(z['anc'])
    cells = [tuple(c) for c in z['cells']]
    classes = [[tuple(e) for e in np.array(cl).tolist()] for cl in z['classes']]
    K = len(cells)
    loss, seq = make_loss(N, n, d, classes, mode, K=K)
    f = jax.jit(loss)
    g = jax.jit(jax.grad(loss))
    best = None
    for P0, L0 in zip(z['P'], z['L']):
        if L0 > thr:
            break
        shape = P0.shape
        x0 = np.asarray(P0, dtype=np.float64).ravel()
        r = minimize(lambda x: float(f(x.reshape(shape))), x0, jac=lambda x: np.asarray(g(x.reshape(shape))).ravel(),
                     method='BFGS', options=dict(maxiter=20000, gtol=1e-16))
        x, fun = r.x, r.fun
        if 1e-13 < fun < 1e-3 and K == N:
            x, fun = lm_refine(loss.residual, f, x, shape)
        print(f'  {os.path.basename(fn)} cand L32={L0:.2e} -> {fun:.2e}', flush=True)
        if best is None or fun < best[0]:
            best = (fun, x.reshape(shape))
        if fun < 1e-13:
            break
    if best and best[0] < 1e-12:
        os.makedirs('data/brick', exist_ok=True)
        R, C = int(z['R']), int(z['C'])
        out = f'data/brick/{N}_{n}_{R}x{C}{"a" if anc else ""}_d{d}_{mode}.json'
        json.dump(dict(N=N, n=n, R=R, C=C, d=d, mode=mode, cells=cells, classes=classes, seq=list(map(int, seq)),
                       anc=anc, params=best[1].tolist(), loss=float(best[0])), open(out, 'w'))
        print(f'EXACT D({N},{n}) d={d} moments={2*d+1} -> {out}', flush=True)
        return True
    return False


if __name__ == '__main__':
    pat = sys.argv[1] if len(sys.argv) > 1 else 'logs/gpu_cand/*.npz'
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 5e-3
    for fn in sorted(glob.glob(pat)):
        polish_file(fn, thr)
