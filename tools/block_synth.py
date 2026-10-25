"""Phase-relaxed Dicke-unitary block synthesis.

Block U(a, J): on a path of a qubits, maps sorted unary |0^{a-j} 1^j> (ones at the END) to e^{i phi_j} D(a, j)
for every j in J (phases phi_j free: they are compensated for free by Z-rotations on the seed's unary bits).
Start from the exact support-pruned B&E block (dicke_core.be_ops), prune CZs one at a time with warm-started
batched Adam + BFGS polish. Saves data/blocks/{a}_{jmin}_{jmax}_L{L}.json.

usage: python tools/block_synth.py a jmin jmax [restarts] [steps] [extra]
  extra = 'ladder' adds rungs of a 2-row ladder (not used by default)
"""
import sys, os, time, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import dicke_core as dc
from synth import cz_diags, dicke_vec
from prune_be import template_to_params

jax.config.update('jax_enable_x64', True)


def sorted_in(a, j):
    v = np.zeros(2 ** a)
    v[(1 << j) - 1] = 1.0   # last j qubits (LSBs, qubit 0 = MSB) are ones
    return v


def make_block_loss(a, J, edges):
    D = jnp.asarray(cz_diags(a, edges))
    ins = jnp.asarray(np.array([sorted_in(a, j) for j in J], dtype=complex))
    outs = jnp.asarray(np.array([dicke_vec(a, j) for j in J], dtype=complex))
    K = len(J)

    def gate(p):
        al, t, b = p[0], p[1], p[2]
        c, s = jnp.cos(t / 2), jnp.sin(t / 2)
        ryy = jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)
        return jnp.exp(-0.5j * b * jnp.array([1., -1.]))[:, None] * ryy * jnp.exp(-0.5j * al * jnp.array([1., -1.]))[None, :]

    def layer(psi, P):
        psi = psi.reshape((K,) + (2,) * a)
        for q in range(a):
            psi = jnp.moveaxis(jnp.tensordot(gate(P[q]), psi, axes=([1], [q + 1])), 0, q + 1)
        return psi.reshape(K, -1)

    def loss(params, seq):
        psi = layer(ins, params[0])

        def body(psi, x):
            P, e = x
            return layer(psi * D[e][None, :], P), None

        psi, _ = jax.lax.scan(body, psi, (params[1:], seq))
        ov = jnp.sum(jnp.conj(outs) * psi, axis=1)
        return 1.0 - jnp.mean(jnp.abs(ov) ** 2)

    return loss


def run_adam(loss, P0, seqs, steps, lr=0.03):
    vg = jax.jit(jax.vmap(jax.value_and_grad(loss), in_axes=(0, 0)))
    b1, b2, eps = 0.9, 0.999, 1e-12

    @jax.jit
    def step(p, m, v, t, s):
        val, g = vg(p, s)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        return p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps), m, v, val

    p = jnp.asarray(P0); m = jnp.zeros_like(p); v = jnp.zeros_like(p)
    s = jnp.asarray(np.asarray(seqs, dtype=np.int32))
    for t in range(1, steps + 1):
        p, m, v, val = step(p, m, v, t, s)
    val, _ = vg(p, s)
    return np.array(val), np.array(p)


def polish(loss, params, seq, iters=3000):
    shape = params.shape
    sj = jnp.asarray(np.asarray(seq, dtype=np.int32))
    f = jax.jit(lambda x: loss(x.reshape(shape), sj))
    g = jax.jit(jax.grad(lambda x: loss(x.reshape(shape), sj)))
    r = minimize(lambda x: float(f(x)), params.ravel(), jac=lambda x: np.asarray(g(x)), method='BFGS',
                 options=dict(maxiter=iters, gtol=1e-15))
    return float(r.fun), r.x.reshape(shape)


def prune_block(a, J, edges, seq, params, rng, steps, restarts, thr=1e-9):
    L = len(seq)
    while L > 1:
        loss = make_block_loss(a, J, edges)
        cands, P0 = [], []
        for t in range(L):
            s = seq[:t] + seq[t + 1:]
            p = np.concatenate([params[:t + 1], params[t + 2:]], axis=0)
            for r in range(restarts):
                cands.append(s)
                P0.append(p + (0 if r == 0 else rng.normal(0, 0.3, p.shape)))
        val, P = run_adam(loss, np.array(P0), cands, steps)
        order = np.argsort(val)
        print(f'   L={L}: best {val[order[0]]:.2e}  #<3e-2: {int(np.sum(val < 3e-2))}', flush=True)
        ok = False
        for b in order[:4]:
            if val[b] < thr:
                ok = True
            elif val[b] < 3e-2:
                fv, pp = polish(loss, P[b], cands[b])
                if fv < thr:
                    P[b] = pp
                    ok = True
            if ok:
                seq, params = cands[b], P[b]
                L -= 1
                print(f'   -> L={L}', flush=True)
                break
        if not ok:
            break
    return seq, params


def save_block(a, J, edges, seq, params, fv):
    os.makedirs('data/blocks', exist_ok=True)
    d = dict(a=a, J=list(J), edges=[list(e) for e in edges], seq=[int(x) for x in seq],
             params=np.asarray(params).tolist(), mode='complex', infidelity=float(fv), L=len(seq))
    fn = f'data/blocks/{a}_{J[0]}_{J[-1]}_L{len(seq)}.json'
    with open(fn, 'w') as f:
        json.dump(d, f)
    return fn


if __name__ == '__main__':
    a, jmin, jmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    restarts = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    J = list(range(jmin, jmax + 1))
    ops = dc.be_ops(list(range(a)), jmax, jmin, jmax)
    t = dict(k=a, ops=ops)
    if not any(o[0] == 'cz' for o in ops):
        print('trivial block'); sys.exit()
    edges, seq, P = template_to_params(t)
    loss = make_block_loss(a, J, edges)
    l0 = float(loss(jnp.asarray(P), jnp.asarray(np.array(seq, dtype=np.int32))))
    print(f'block a={a} J={J} start L={len(seq)} loss={l0:.2e}', flush=True)
    assert l0 < 1e-9, l0
    rng = np.random.default_rng(0)
    t0 = time.time()
    s, p = prune_block(a, J, edges, list(seq), P, rng, steps, restarts)
    fv, p = polish(make_block_loss(a, J, edges), p, s)
    print(f'FINAL block a={a} J={J} L={len(s)} (from {len(seq)}) loss={fv:.2e} t={time.time()-t0:.0f}s', flush=True)
    if fv < 1e-11:
        print('saved', save_block(a, J, edges, s, p, fv))
