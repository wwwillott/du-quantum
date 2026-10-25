"""State-aware peephole re-synthesis of exact Dicke circuits.

Given an exact template (ops on k nodes), repeatedly pick a window = (connected node set Q, contiguous op range
with no CZ crossing the boundary of Q). The window sub-circuit W only has to satisfy
    (W (x) I_env) |psi_in> = |psi_out>,
which only constrains W on the Schmidt support of psi_in on Q. With psi_in = sum_s lam_s |l_s>|r_s>,
the target images are l'_s = (I (x) <r_s|) psi_out / lam_s and the loss is
    1 - | sum_s lam_s^2 <l'_s| W |l_s> |^2      (tiny: 2^|Q| x rank).
We prune CZs of W (warm start) and splice the shorter W back. Final circuit is re-verified exactly.

usage: python tools/peephole.py N n [construction=dc2|path] [wmax=5] [rounds=200] [seed]
"""
import sys, os, time, json, math, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import dicke_core as dc
from synth import cz_diags, dicke_vec
from prune_be import zyz

jax.config.update('jax_enable_x64', True)


def apply_ops(psi, ops, k):
    psi = psi.reshape((2,) * k)
    for op in ops:
        if op[0] == 'u':
            psi = np.moveaxis(np.tensordot(op[2], psi, axes=([1], [op[1]])), 0, op[1])
        else:
            sl = [slice(None)] * k
            sl[op[1]] = 1
            sl[op[2]] = 1
            psi = psi.copy() if not psi.flags.writeable else psi
            psi[tuple(sl)] *= -1
    return psi.reshape(-1)


def local_params(sub, Q):
    """window ops (global nodes) -> local edges/seq/params (complex zyz), local node i = Q[i]"""
    loc = {q: i for i, q in enumerate(Q)}
    w = len(Q)
    edges, eidx, seq, layers = [], {}, [], []
    pend = [np.eye(2, dtype=complex) for _ in range(w)]
    for op in sub:
        if op[0] == 'u':
            i = loc[op[1]]
            pend[i] = op[2] @ pend[i]
        else:
            e = tuple(sorted((loc[op[1]], loc[op[2]])))
            if e not in eidx:
                eidx[e] = len(edges); edges.append(e)
            layers.append(pend); pend = [np.eye(2, dtype=complex) for _ in range(w)]
            seq.append(eidx[e])
    layers.append(pend)
    return edges, seq, np.array([[zyz(U) for U in L] for L in layers])


def make_window_loss(w, edges_all, Lmat, Rmat, wts):
    """Lmat: (r, 2^w) input Schmidt vectors; Rmat: (r, 2^w) target images; wts: lam^2 (sum 1)."""
    D = jnp.asarray(cz_diags(w, edges_all))
    ins = jnp.asarray(Lmat)
    outs = jnp.asarray(Rmat)
    W8 = jnp.asarray(wts)
    r = Lmat.shape[0]

    def gate(p):
        al, t, b = p[0], p[1], p[2]
        c, s = jnp.cos(t / 2), jnp.sin(t / 2)
        ryy = jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)
        return jnp.exp(-0.5j * b * jnp.array([1., -1.]))[:, None] * ryy * jnp.exp(-0.5j * al * jnp.array([1., -1.]))[None, :]

    def layer(psi, P):
        psi = psi.reshape((r,) + (2,) * w)
        for q in range(w):
            psi = jnp.moveaxis(jnp.tensordot(gate(P[q]), psi, axes=([1], [q + 1])), 0, q + 1)
        return psi.reshape(r, -1)

    def loss(params, seq):
        psi = layer(ins, params[0])

        def body(psi, x):
            P, e = x
            return layer(psi * D[e][None, :], P), None

        psi, _ = jax.lax.scan(body, psi, (params[1:], seq))
        ov = jnp.sum(W8 * jnp.sum(jnp.conj(outs) * psi, axis=1))
        return 1.0 - jnp.abs(ov) ** 2

    return loss


_cache = {}


def get_vg(w, L, edges, key):
    k = (w, L, tuple(edges), key)
    return k


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


def polish(loss, params, seq, iters=2000):
    shape = params.shape
    sj = jnp.asarray(np.asarray(seq, dtype=np.int32))
    f = jax.jit(lambda x: loss(x.reshape(shape), sj))
    g = jax.jit(jax.grad(lambda x: loss(x.reshape(shape), sj)))
    r = minimize(lambda x: float(f(x)), params.ravel(), jac=lambda x: np.asarray(g(x)), method='BFGS',
                 options=dict(maxiter=iters, gtol=1e-15))
    return float(r.fun), r.x.reshape(shape)


def try_window(ops, k, Q, i1, i2, rng, steps=600, restarts=2, thr=1e-12):
    """returns new op list (or None) with fewer CZs in the window."""
    Qs = set(Q)
    sub = [op for op in ops[i1:i2] if all(x in Qs for x in op[1:3] if isinstance(x, (int, np.integer)))]
    rest = [op for op in ops[i1:i2] if not all(x in Qs for x in op[1:3] if isinstance(x, (int, np.integer)))]
    ncz = sum(1 for op in sub if op[0] == 'cz')
    if ncz < 2:
        return None
    psi_in = apply_ops(np.eye(1, 2 ** k, 0, dtype=complex).ravel(), ops[:i1], k)
    psi_out = apply_ops(psi_in.copy(), sub, k)
    # bring window qubits to front: reshape (2^w, 2^(k-w))
    perm = list(Q) + [q for q in range(k) if q not in Qs]
    w = len(Q)
    A = np.transpose(psi_in.reshape((2,) * k), perm).reshape(2 ** w, -1)
    B = np.transpose(psi_out.reshape((2,) * k), perm).reshape(2 ** w, -1)
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    keep = S > 1e-12
    U, S, Vh = U[:, keep], S[keep], Vh[keep]
    Lmat = U.T                                  # (r, 2^w)
    Rmat = (B @ Vh.conj().T / S[None, :]).T     # (r, 2^w)  images
    wts = S ** 2
    edges, seq, P = local_params(sub, Q)
    loss = make_window_loss(w, edges, Lmat, Rmat, wts)
    l0 = float(loss(jnp.asarray(P), jnp.asarray(np.array(seq, dtype=np.int32))))
    if l0 > 1e-9:
        return None
    best = None
    L = len(seq)
    while L > 1:
        cands, P0 = [], []
        for t in range(L):
            s = seq[:t] + seq[t + 1:]
            p = np.concatenate([P[:t + 1], P[t + 2:]], axis=0)
            for r_ in range(restarts):
                cands.append(s)
                P0.append(p + (0 if r_ == 0 else rng.normal(0, 0.3, p.shape)))
        val, PP = run_adam(loss, np.array(P0), cands, steps)
        order = np.argsort(val)
        ok = False
        succ = []
        for b in order[:4]:
            if val[b] < 3e-2:
                fv, pp = polish(loss, PP[b], cands[b])
                if fv < thr:
                    succ.append((seq_depth(cands[b], edges), cands[b], pp))
        if succ:
            succ.sort(key=lambda x: x[0])
            _, seq, P = succ[0]
            ok = True
        if not ok:
            break
        L -= 1
        best = (seq, P)
    if best is None:
        return None
    seq, P = best
    new = []
    for t in range(len(seq) + 1):
        for i in range(w):
            a_, t_, b_ = P[t][i]
            U1 = dc.rz(b_) @ dc.ry(t_) @ dc.rz(a_)
            if not dc.is_identity_up_to_phase(U1):
                new.append(('u', Q[i], U1))
        if t < len(seq):
            e = edges[seq[t]]
            new.append(('cz', Q[e[0]], Q[e[1]]))
    return ops[:i1] + new + rest + ops[i2:], ncz - len(seq)


def seq_depth(seq, edges):
    """CZ-layer depth of a CZ sequence (1q gates between CZs on the same qubit force order)"""
    t = {}
    d = 0
    for e in seq:
        a, b = edges[e]
        L = max(t.get(a, 0), t.get(b, 0)) + 1
        t[a] = t[b] = L
        d = max(d, L)
    return d


def template_graph(t):
    adj = {i: set() for i in range(t['k'])}
    for op in t['ops']:
        if op[0] == 'cz':
            adj[op[1]].add(op[2]); adj[op[2]].add(op[1])
    return adj


def random_connected(adj, w, rng):
    start = int(rng.integers(len(adj)))
    Q = [start]
    while len(Q) < w:
        fr = sorted({v for u in Q for v in adj[u]} - set(Q))
        if not fr:
            break
        Q.append(int(rng.choice(fr)))
    return Q


def windows_for(ops, Q):
    """maximal op ranges where no CZ crosses Q's boundary"""
    Qs = set(Q)
    out, start = [], 0
    for i, op in enumerate(ops):
        if op[0] == 'cz' and ((op[1] in Qs) != (op[2] in Qs)):
            if i - start > 0:
                out.append((start, i))
            start = i + 1
    if len(ops) - start > 0:
        out.append((start, len(ops)))
    return out


def fidelity(ops, k, N, n):
    psi = apply_ops(np.eye(1, 2 ** k, 0, dtype=complex).ravel(), ops, k)
    return abs(np.vdot(dicke_vec(N, n), psi)) ** 2


def save(N, n, k, ops, edges_graph, tag):
    os.makedirs('data/peep', exist_ok=True)
    enc = []
    for op in ops:
        if op[0] == 'u':
            enc.append(['u', int(op[1]), [[complex(x).real, complex(x).imag] for x in np.asarray(op[2]).ravel()]])
        else:
            enc.append(['cz', int(op[1]), int(op[2])])
    ncz = sum(1 for op in ops if op[0] == 'cz')
    fn = f'data/peep/{N}_{n}_{tag}_cz{ncz}.json'
    json.dump(dict(N=N, n=n, k=k, ops=enc, tag=tag, ncz=ncz), open(fn, 'w'))
    return fn


if __name__ == '__main__':
    N, n = int(sys.argv[1]), int(sys.argv[2])
    cons = sys.argv[3] if len(sys.argv) > 3 else 'dc2'
    wmax = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    rng = np.random.default_rng(seed)
    if cons.startswith('lib:'):
        from evaluate import load_template
        t = load_template(cons[4:])
    elif cons.startswith('peep:'):
        d = json.load(open(cons[5:]))
        t = dict(k=d['k'], ops=[('cz', o[1], o[2]) if o[0] == 'cz' else
                                ('u', o[1], np.array([complex(r, i) for r, i in o[2]]).reshape(2, 2)) for o in d['ops']])
    else:
        t = dc.dc2_template(N, n) if cons == 'dc2' else dc.be_path_template_opt(N, n)
    cons = cons.split(':')[0]
    ops, k = list(t['ops']), t['k']
    adj = template_graph(t)
    ncz0 = sum(1 for op in ops if op[0] == 'cz')
    print(f'N={N} n={n} {cons}: start CZ={ncz0} F={fidelity(ops, k, N, n):.12f}', flush=True)
    t0 = time.time()
    fails = 0
    for rd in range(rounds):
        w = int(rng.integers(3, wmax + 1))
        Q = random_connected(adj, w, rng)
        wins = windows_for(ops, Q)
        wins = [x for x in wins if sum(1 for op in ops[x[0]:x[1]] if op[0] == 'cz' and op[1] in Q and op[2] in Q) >= 2]
        if not wins:
            continue
        i1, i2 = wins[int(rng.integers(len(wins)))]
        res = try_window(ops, k, Q, i1, i2, rng)
        jax.clear_caches()
        if res is None:
            fails += 1
            continue
        new_ops, saved = res
        F = fidelity(new_ops, k, N, n)
        if abs(F - 1) < 1e-10:
            ops = new_ops
            ncz = sum(1 for op in ops if op[0] == 'cz')
            print(f'  round {rd}: Q={Q} saved {saved} -> CZ={ncz}  F={F:.13f}  ({time.time()-t0:.0f}s)', flush=True)
            save(N, n, k, ops, adj, cons)
        else:
            print(f'  round {rd}: rejected (F={F})', flush=True)
    ncz = sum(1 for op in ops if op[0] == 'cz')
    print(f'FINAL N={N} n={n} {cons}: CZ {ncz0} -> {ncz} ({time.time()-t0:.0f}s)', flush=True)
