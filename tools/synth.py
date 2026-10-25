"""Batched variational search for CZ-minimal exact Dicke-state circuits (JAX).

Template: |0..0> -> L0 -> [CZ(e_1) -> L_1] -> ... -> [CZ(e_L) -> L_L]
where each L_t is a layer of single-qubit gates on all qubits (Ry only in 'real' mode; Rz.Ry.Rz in 'complex').
The CZ edge sequence is data (index into a table of diagonal phase vectors), so one jit serves many
sequences; we vmap over (sequence, random init) pairs and run Adam, then polish the best with BFGS.
"""
import itertools
import math
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


def dicke_vec(N, n):
    psi = np.zeros(2 ** N)
    for c in itertools.combinations(range(N), n):
        psi[sum(1 << (N - 1 - q) for q in c)] = 1
    return psi / np.linalg.norm(psi)


def cz_diags(N, edges):
    idx = np.arange(2 ** N)
    bits = (idx[:, None] >> (N - 1 - np.arange(N))) & 1
    return np.stack([1 - 2 * (bits[:, a] & bits[:, b]) for a, b in edges]).astype(np.float64)


def make_loss(N, n, L, edges, mode='real'):
    target = jnp.asarray(dicke_vec(N, n))
    D = jnp.asarray(cz_diags(N, edges))
    npg = 1 if mode == 'real' else 3

    def gate(p):
        if mode == 'real':
            c, s = jnp.cos(p[0] / 2), jnp.sin(p[0] / 2)
            return jnp.array([[c, -s], [s, c]])
        a, t, b = p
        c, s = jnp.cos(t / 2), jnp.sin(t / 2)
        rz = lambda x: jnp.array([jnp.exp(-0.5j * x), jnp.exp(0.5j * x)])
        ry = jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)
        return rz(b)[:, None] * ry * rz(a)[None, :]

    def layer(psi, P):
        psi = psi.reshape((2,) * N)
        for q in range(N):
            U = gate(P[q])
            psi = jnp.moveaxis(jnp.tensordot(U, psi, axes=([1], [q])), 0, q)
        return psi.reshape(-1)

    def state(params, seq):
        # params: (L+1, N, npg); seq: (L,) int
        psi = jnp.zeros(2 ** N, dtype=jnp.float64 if mode == 'real' else jnp.complex128).at[0].set(1.0)
        psi = layer(psi, params[0])

        def body(psi, x):
            P, e = x
            psi = psi * D[e]
            return layer(psi, P), None

        psi, _ = jax.lax.scan(body, psi, (params[1:], seq))
        return psi

    def loss(params, seq):
        psi = state(params, seq)
        return 1.0 - jnp.abs(jnp.vdot(target, psi)) ** 2

    return loss, state, npg


def adam_search(N, n, L, edges, seqs, restarts=4, steps=2000, lr=0.05, mode='real', seed=0, tol=1e-9):
    loss, state, npg = make_loss(N, n, L, edges, mode)
    seqs = np.asarray(seqs, dtype=np.int32)
    S = len(seqs)
    allseq = np.repeat(seqs, restarts, axis=0)
    key = jax.random.PRNGKey(seed)
    params = jax.random.uniform(key, (len(allseq), L + 1, N, npg), minval=-np.pi, maxval=np.pi)
    vg = jax.jit(jax.vmap(jax.value_and_grad(loss), in_axes=(0, 0)))
    m = jnp.zeros_like(params); v = jnp.zeros_like(params)
    b1, b2, eps = 0.9, 0.999, 1e-12

    @jax.jit
    def step(params, m, v, t, seqs):
        val, g = vg(params, seqs)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        return params - lr * mh / (jnp.sqrt(vh) + eps), m, v, val

    seqj = jnp.asarray(allseq)
    for t in range(1, steps + 1):
        params, m, v, val = step(params, m, v, t, seqj)
    val = np.asarray(val)
    return val, np.asarray(params), allseq, loss


def polish(loss, params, seq, iters=500):
    from scipy.optimize import minimize
    shape = params.shape
    f = jax.jit(lambda x: loss(x.reshape(shape), jnp.asarray(seq)))
    g = jax.jit(jax.grad(lambda x: loss(x.reshape(shape), jnp.asarray(seq))))
    r = minimize(lambda x: float(f(x)), params.ravel(), jac=lambda x: np.asarray(g(x)), method='BFGS',
                 options=dict(maxiter=iters, gtol=1e-14))
    return r.fun, r.x.reshape(shape)


def random_seqs(edges, L, count, rng):
    E = len(edges)
    out = set()
    tries = 0
    while len(out) < count and tries < count * 50:
        tries += 1
        s = [int(rng.integers(E))]
        while len(s) < L:
            e = int(rng.integers(E))
            if e == s[-1]:
                continue
            s.append(e)
        out.add(tuple(s))
    return [list(s) for s in out]


def all_seqs(edges, L):
    E = len(edges)
    for s in itertools.product(range(E), repeat=L):
        if all(s[i] != s[i + 1] for i in range(L - 1)):
            yield list(s)


def save_template(N, n, shape, edges, seq, params, mode, fv, tag=''):
    import json, os
    os.makedirs('data/templates', exist_ok=True)
    d = dict(N=N, n=n, shape=shape, edges=[list(e) for e in edges], seq=[int(x) for x in seq],
             params=np.asarray(params).tolist(), mode=mode, infidelity=float(fv), L=len(seq))
    with open(f'data/templates/{N}_{n}_{shape}{tag}_L{len(seq)}.json', 'w') as f:
        json.dump(d, f)


if __name__ == '__main__':
    N, n = int(sys.argv[1]), int(sys.argv[2])
    shape = sys.argv[3]
    Lmin, Lmax = int(sys.argv[4]), int(sys.argv[5])
    mode = sys.argv[6] if len(sys.argv) > 6 else 'real'
    shapes = {
        'path': [(i, i + 1) for i in range(N - 1)],
        'sq': [(0, 1), (1, 2), (2, 3), (3, 0)] + [(i, i + 1) for i in range(3, N - 1)],
        'star': [(0, i) for i in range(1, min(N, 5))] + [(i, i + 4) for i in range(1, N - 4)],
    }
    edges = shapes[shape]
    rng = np.random.default_rng(0)
    for L in range(Lmin, Lmax + 1):
        t0 = time.time()
        seqs = list(all_seqs(edges, L))
        if len(seqs) > 3000:
            seqs = random_seqs(edges, L, 3000, rng)
        val, params, allseq, loss = adam_search(N, n, L, edges, seqs, restarts=3, steps=1500, mode=mode)
        best = np.argsort(val)[:5]
        print(f'N={N} n={n} {shape} L={L}: nseq={len(seqs)} best={val[best]} t={time.time()-t0:.1f}s', flush=True)
        if val[best[0]] < 1e-6:
            fv, p = polish(loss, params[best[0]], allseq[best[0]])
            print('  polished', fv, 'seq', [edges[e] for e in allseq[best[0]]], flush=True)
            if fv < 1e-12:
                save_template(N, n, shape, edges, allseq[best[0]], p, mode, fv)
                break
