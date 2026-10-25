# ==================================================================== main entry (inlined into submission.py)
import time as _time

T1_EST = 68e-6
TPHI_EST = 120e-6
LIBRARY = {}   # (N, n) -> list of template dicts {edges, seq, params, mode}; filled by build_submission.py
PEEP = {}


def _lib_template(d, N, n):
    edges, seq, P, mode = d['edges'], d['seq'], d['params'], d['mode']
    ops = []

    def layer(Pl):
        for q in range(N):
            if mode == 'real':
                ops.append(('u', q, ry(Pl[q][0])))
            else:
                a, t, b = Pl[q]
                ops.append(('u', q, rz(b) @ ry(t) @ rz(a)))

    layer(P[0])
    for t, e in enumerate(seq):
        a, b = edges[e]
        ops.append(('cz', a, b))
        layer(P[t + 1])
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=n)


def estimate_error(circuit, state_qubits, n, cz_err, sq_err, zero_err, one_err, dd=True):
    """Crude log-infidelity proxy for choosing between exact candidates."""
    czp = {}
    for pair, val in cz_err.items():
        czp[frozenset(pair)] = float(val[0])
    N = len(state_qubits)
    e = 0.0
    t = 0.0
    first = {}
    for mi, m in enumerate(circuit):
        has2 = any(len(op.qubits) == 2 for op in m.operations)
        dt = 42e-9 if has2 else 25e-9
        for op in m.operations:
            for q in op.qubits:
                first.setdefault(q, t)
            if len(op.qubits) == 2:
                e += 0.8 * czp[frozenset(op.qubits)]
            else:
                e += 0.67 * float(sq_err[op.qubits[0]][0])
        t += dt
    T = t
    expo = _exposure(circuit) if len(circuit.all_qubits()) <= 14 else None
    if expo is None:
        # excitation exposure: n excitations from (roughly) the first gate on any state qubit to the end
        expo = n * (T - min(first.values())) if first else 0.0
    e += expo / T1_EST
    # dephasing exposure of live qubits (DD suppresses much of it)
    e += (0.1 if dd else 0.2) * sum(T - first[q] for q in first) / TPHI_EST
    for q in state_qubits:
        e += readout_cost(q, n / N, zero_err, one_err, x_weight=0.5)
    # small-M penalty: readout-asymmetry spread -> systematic shell non-uniformity (raises chi2)
    M = math.comb(N, n)
    if 1 < M <= 200:
        d = [float(one_err.get(q, (0, 0))[0]) - float(zero_err.get(q, (0, 0))[0]) for q in state_qubits]
        var = float(np.var(d))
        delta2 = n * (N - n) / (N - 1) * var
        K = 1e4 * math.exp(-e)
        e += math.log(1 + K * delta2 / (M - 1))
    return e


def _exposure(circuit):
    """sum over moments of (sum_q P(q=1)) * duration, from the exact noiseless state."""
    order = sorted(circuit.all_qubits())
    idx = {q: i for i, q in enumerate(order)}
    k = len(order)
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1
    tot = 0.0
    for mo in circuit:
        two = False
        for op in mo.operations:
            if len(op.qubits) == 1:
                a = idx[op.qubits[0]]
                psi = np.moveaxis(np.tensordot(cirq.unitary(op), psi, axes=([1], [a])), 0, a)
            else:
                two = True
                a, b = idx[op.qubits[0]], idx[op.qubits[1]]
                sl = [slice(None)] * k
                sl[a] = 1
                sl[b] = 1
                psi[tuple(sl)] *= -1
        p = np.abs(psi) ** 2
        tot += sum(float(p.take(1, axis=i).sum()) for i in range(k)) * (42e-9 if two else 25e-9)
    return tot


def _verify(circuit, state_qubits, n, atol=1e-6):
    sq = sorted(state_qubits)
    anc = sorted(set(circuit.all_qubits()) - set(sq))
    N = len(sq)
    if len(sq) + len(anc) > 18:
        return True  # too large to check here; constructions are proven exact
    psi = cirq.final_state_vector(circuit, qubit_order=sq + anc, dtype=np.complex128).reshape(2 ** N, -1)
    tgt = np.zeros(2 ** N)
    for c in itertools.combinations(range(N), n):
        tgt[sum(1 << (N - 1 - i) for i in c)] = 1
    tgt /= np.linalg.norm(tgt)
    F = float(np.sum(np.abs(tgt @ psi) ** 2))
    return abs(F - 1) < atol


def _diverse(states, k):
    """top-k embeddings with distinct qubit sets"""
    out, seen = [], set()
    for st in states:
        key = frozenset(st[1].values())
        if key in seen:
            continue
        seen.add(key)
        out.append(st)
        if len(out) >= k:
            break
    return out


def _circuit_to_template(c, outs, N, n):
    """unitary cirq circuit (CZ + 1q only) -> template with state nodes 0..N-1 (outputs first)."""
    order = sorted(outs) + sorted(set(c.all_qubits()) - set(outs))
    idx = {q: i for i, q in enumerate(order)}
    ops = []
    for m in c:
        for op in m.operations:
            if len(op.qubits) == 2:
                ops.append(('cz', idx[op.qubits[0]], idx[op.qubits[1]]))
            else:
                ops.append(('u', idx[op.qubits[0]], cirq.unitary(op)))
    return dict(k=len(order), ops=ops, state_nodes=list(range(N)), N=N, n=n)


def _bonus_templates(N, n):
    out = []
    try:
        srcs = []
        try:
            srcs.append(_bonus_mod.create_dicke_state_adaptive(N, n))
        except Exception:
            pass
        if (N, n) in BRICK:
            srcs.append(_brick_circuit(BRICK[(N, n)]))
        for c, qs in srcs:
            ok = all(isinstance(op.gate, (cirq.CZPowGate, cirq.PhasedXZGate)) or
                     (len(op.qubits) == 1 and cirq.has_unitary(op) and not cirq.is_measurement(op))
                     for op in c.all_operations())
            ok = ok and not any(cirq.is_measurement(op) or isinstance(op, cirq.ClassicallyControlledOperation)
                                for op in c.all_operations())
            ok = ok and all(op.gate == cirq.CZ for op in c.all_operations() if len(op.qubits) == 2)
            if ok and len(c.all_qubits()) == N:
                out.append(_circuit_to_template(c, qs, N, n))
    except Exception:
        pass
    return out


def _candidates(N, n, adj, cz_err, sq_err, zero_err, one_err, deadline):
    qcost = lambda q: readout_cost(q, n / N, zero_err, one_err) + 2 * float(sq_err[q][0])
    out = []
    good = sorted(adj, key=lambda q: qcost(q) + min(adj[q].values(), default=1))
    # --- W trees
    if n == 1 and N >= 2:
        for root in good[:10]:
            for lam in (0.0005, 0.003, 0.02):
                if _time.time() > deadline:
                    break
                patch = grow_patch(adj, N, root, qcost, lam=lam)
                if patch is None:
                    continue
                rounds = broadcast_tree(adj, patch, root)
                t = w_tree_template(patch, rounds)
                out.append(('wtree', t, {i: q for i, q in enumerate(patch)}))
    # --- library templates (numerically synthesised, exact)
    for d in LIBRARY.get((N, n), []):
        t = _lib_template(d, N, n)
        nw = node_weights(t)
        res = embed(t, adj, lambda node, q: qcost(q) + nw[node] * 0.0, beam=2000, return_all=True)
        if res:
            for r in _diverse(res, 6 if math.comb(N, n) <= 200 else 2):
                out.append(('lib', t, r[1]))
    # --- peephole-optimised exact op lists (state-aware re-synthesis, offline)
    for k, pops in PEEP.get((N, n), []):
        ops = [('u', o[1], rz(o[2][2]) @ ry(o[2][1]) @ rz(o[2][0])) if o[0] == 'u' else o for o in pops]
        t = dict(k=k, ops=ops, state_nodes=list(range(N)), N=N, n=n)
        res = embed(t, adj, lambda node, q: qcost(q), beam=1500, return_all=True)
        if res:
            for r in _diverse(res, 4 if math.comb(N, n) <= 200 else 2):
                out.append(('peep', t, r[1]))
    # --- depth-optimised bonus circuits (measurement-free ones are legal here): re-placed on calibrated qubits
    if N <= 15 and _time.time() < deadline - 20:
        for t in _bonus_templates(N, n):
            res = embed(t, adj, lambda node, q: qcost(q), beam=1500, return_all=True)
            if res:
                for r in _diverse(res, 3):
                    out.append(('bonus', t, r[1]))
    # --- divide & conquer (weight-distribution seed + two parallel B&E blocks)
    if n >= 2 and N >= 2 * n:
        splits = sorted({N // 2, (N + 1) // 2}, key=lambda a: abs(a - N / 2))
        for a in splits:
            if a < n or N - a < n or _time.time() > deadline:
                continue
            t = dc2_template(N, n, a)
            res = embed(t, adj, lambda node, q: qcost(q), beam=600 if N > 12 else 1500)
            if res is not None:
                out.append(('dc2', t, res[1]))
    # --- B&E nearest-neighbour path (support-pruned)
    if N >= 2:
        t = be_path_template_opt(N, n)
        nw = node_weights(t)
        res = embed(t, adj, lambda node, q: qcost(q), beam=600 if N > 12 else 2000)
        if res is not None:
            out.append(('bepath', t, res[1]))
    return out


def create_dicke_state(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate,
                       zero_errors, one_errors):
    t0 = _time.time()
    deadline = t0 + 35.0
    N, n = int(num_qubits), int(num_excitations)
    if N < 1 or n < 0 or n > N:
        raise NotImplementedError
    flip = False
    if n > N - n:
        n, flip = N - n, True
    adj = hw_graph(cz_pauli_error_rate, sq_pauli_error_rate)
    if n == 0:
        qs = sorted(adj, key=lambda q: readout_cost(q, 1.0 if flip else 0.0, zero_errors, one_errors))[:N]
        if flip:
            c = cirq.Circuit([cirq.Moment([cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)(q)
                                           for q in qs])])
        else:
            c = cirq.Circuit([cirq.Moment([cirq.PhasedXZGate(x_exponent=0, z_exponent=0, axis_phase_exponent=0)(q)
                                           for q in qs])])
        return c, set(qs)
    cands = _candidates(N, n, adj, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors, deadline)
    scored = []
    for name, t, mapping in cands:
        for dd in (True, False):
            c = build_circuit(t, mapping, dd=dd)
            if flip:
                c = _append_flip(c, [mapping[i] for i in t['state_nodes']])
            qs = {mapping[i] for i in t['state_nodes']}
            e = estimate_error(c, qs, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate,
                               zero_errors, one_errors, dd=dd)
            scored.append((e, name, c, qs))
    scored.sort(key=lambda x: x[0])
    for e, name, c, qs in scored:
        if len(c) > 1750:
            continue
        if _verify(c, qs, num_excitations):
            return c, qs
    raise NotImplementedError


def _append_flip(c, qubits):
    c = c.copy()
    last = c[-1] if len(c) else None
    if last is not None and all(len(op.qubits) == 1 for op in last.operations):
        ops = {op.qubits[0]: op for op in last.operations}
        new = []
        for q in qubits:
            U = cirq.unitary(ops[q]) if q in ops else _I2
            new.append(cirq.PhasedXZGate.from_matrix(_X @ U).on(q))
        for q, op in ops.items():
            if q not in set(qubits):
                new.append(op)
        c[-1] = cirq.Moment(new)
    else:
        c.append(cirq.Moment([cirq.PhasedXZGate.from_matrix(_X).on(q) for q in qubits]))
    return c
