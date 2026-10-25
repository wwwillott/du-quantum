"""Minimal-CZ search for partial isometries ('gadgets') on a few qubits: U|in_i> = |out_i> for listed pairs,
with one common global phase. Batched over CZ sequences with JAX Adam (complex U3 layers).

usage: python tools/gadget.py <name> <Lmin> <Lmax>
"""
import sys
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from synth import cz_diags, all_seqs, random_seqs

jax.config.update('jax_enable_x64', True)


def basis(N, bits):
    v = np.zeros(2 ** N, dtype=complex)
    v[int(bits, 2)] = 1
    return v


def gadget_scs(theta):
    """qubits (c, a, b) on a path c-a-b. B&E-NN sweep gadget."""
    N = 3
    pairs = [('000', basis(N, '000')),
             ('001', np.cos(theta) * basis(N, '001') + np.sin(theta) * basis(N, '010')),
             ('011', basis(N, '011')),
             ('101', basis(N, '110')),
             ('111', basis(N, '111'))]
    return N, [(0, 1), (1, 2)], pairs


def make_loss(N, edges, pairs, L, mode='complex'):
    D = jnp.asarray(cz_diags(N, edges))
    ins = jnp.asarray(np.array([basis(N, b) for b, _ in pairs]))
    outs = jnp.asarray(np.array([o for _, o in pairs]))
    K = len(pairs)

    def gate(p):
        a, t, b = p
        c, s = jnp.cos(t / 2), jnp.sin(t / 2)
        ryy = jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)
        return jnp.exp(-0.5j * b * jnp.array([1., -1.]))[:, None] * ryy * jnp.exp(-0.5j * a * jnp.array([1., -1.]))[None, :]

    def layer(psi, P):
        psi = psi.reshape((K,) + (2,) * N)
        for q in range(N):
            psi = jnp.moveaxis(jnp.tensordot(gate(P[q]), psi, axes=([1], [q + 1])), 0, q + 1)
        return psi.reshape(K, -1)

    def loss(params, seq):
        psi = layer(ins, params[0])

        def body(psi, x):
            P, e = x
            return layer(psi * D[e][None, :], P), None

        psi, _ = jax.lax.scan(body, psi, (params[1:], seq))
        ov = jnp.sum(jnp.conj(outs) * psi)
        return 1.0 - jnp.abs(ov) ** 2 / K ** 2

    return loss


def search(N, edges, pairs, L, nseq=2000, restarts=4, steps=2500, lr=0.05, seed=0):
    from prune import run_adam
    rng = np.random.default_rng(seed)
    seqs = list(all_seqs(edges, L))
    if len(seqs) > nseq:
        seqs = random_seqs(edges, L, nseq, rng)
    seqs = [s for s in seqs for _ in range(restarts)]
    loss = make_loss(N, edges, pairs, L)
    P0 = rng.uniform(-np.pi, np.pi, (len(seqs), L + 1, N, 3))
    val, P = run_adam(loss, P0, seqs, steps, lr)
    return val, P, seqs, loss


if __name__ == '__main__':
    name = sys.argv[1]
    Lmin, Lmax = int(sys.argv[2]), int(sys.argv[3])
    theta = 0.7
    N, edges, pairs = {'scs': gadget_scs}[name](theta)
    for L in range(Lmin, Lmax + 1):
        val, P, seqs, loss = search(N, edges, pairs, L)
        o = np.argsort(val)
        print(L, val[o[:4]], [[edges[e] for e in seqs[i]] for i in o[:2]], flush=True)
        if val[o[0]] < 1e-9:
            np.save(f'data/gadget_{name}_L{L}.npy', P[o[0]])
            print('seq', seqs[o[0]])
            break
