"""Prune CZs starting from an EXACT structured circuit (support-pruned B&E path, or any template).
usage: python tools/prune_be.py N n [restarts] [steps]
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp
import dicke_core as dc
from synth import make_loss, polish, save_template
from prune import prune, run_adam

jax.config.update('jax_enable_x64', True)


def zyz(U):
    """U = e^{i g} Rz(b) Ry(t) Rz(a)  -> (a, t, b)"""
    U = U / np.sqrt(np.linalg.det(U))
    t = 2 * np.arctan2(abs(U[1, 0]), abs(U[0, 0]))
    s = np.angle(U[1, 1]) if abs(U[1, 1]) > 1e-12 else 0.0  # (a+b)/2
    d = np.angle(U[1, 0]) if abs(U[1, 0]) > 1e-12 else 0.0  # (b-a)/2
    if abs(U[1, 1]) < 1e-12:
        s = -np.angle(-U[0, 1]) + 2 * d if False else np.angle(U[1, 0])  # degenerate: put all in b
        a, b = 0.0, 2 * np.angle(U[1, 0])
        return a, t, b
    if abs(U[1, 0]) < 1e-12:
        return 0.0, t, 2 * s
    a, b = s - d, s + d
    return a, t, b


def template_to_params(t):
    k = t['k']
    edges, seq, layers = [], [], []
    pend = [np.eye(2, dtype=complex) for _ in range(k)]
    eidx = {}
    for op in t['ops']:
        if op[0] == 'u':
            pend[op[1]] = op[2] @ pend[op[1]]
        else:
            e = (min(op[1], op[2]), max(op[1], op[2]))
            if e not in eidx:
                eidx[e] = len(edges)
                edges.append(e)
            layers.append(pend)
            pend = [np.eye(2, dtype=complex) for _ in range(k)]
            seq.append(eidx[e])
    layers.append(pend)
    P = np.array([[zyz(U) for U in L] for L in layers])
    return edges, seq, P


if __name__ == '__main__':
    if os.environ.get('PRUNE_OK') != '1':
        sys.exit(0)
    N, n = int(sys.argv[1]), int(sys.argv[2])
    restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    start = sys.argv[5] if len(sys.argv) > 5 else 'be'
    t = dc.be_path_template_opt(N, n) if start == 'be' else dc.dc2_template(N, n, int(start[3:]) if len(start) > 3 else None)
    edges, seq, P = template_to_params(t)
    loss, _, _ = make_loss(N, n, len(seq), edges, 'complex')
    l0 = float(loss(jnp.asarray(P), jnp.asarray(np.array(seq, dtype=np.int32))))
    print(f'N={N} n={n} start L={len(seq)} edges={edges} loss={l0:.2e}', flush=True)
    assert l0 < 1e-9
    rng = np.random.default_rng(0)
    t0 = time.time()
    s, p = prune(N, n, edges, list(seq), P, 'complex', rng, steps=steps, restarts=restarts)
    lossf, _, _ = make_loss(N, n, len(s), edges, 'complex')
    fv, p = polish(lossf, p, s, iters=3000)
    print(f'FINAL N={N} n={n} L={len(s)} loss={fv:.2e} t={time.time()-t0:.0f}s seq={[edges[e] for e in s]}', flush=True)
    if fv < 1e-11:
        save_template(N, n, 'path' if start == 'be' else 'dc2', edges, s, p, 'complex', fv, tag='_' + start)
