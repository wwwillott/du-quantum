"""Core machinery for the DuQuantum Dicke-state submission (cirq / cirq_google / numpy / stdlib only).

A *template* is an abstract circuit on nodes 0..k-1:
    ops: list of ('u', node, 2x2 unitary) | ('cz', i, j)
    state_nodes: nodes that carry the Dicke state (others are ancillas)
Pipeline: template -> noise-aware embedding onto the hardware graph -> layered schedule
(1q / CZ moments alternate, ALAP so qubits stay in |0> as long as possible) -> dynamical decoupling
on idle superposed qubits -> PhasedXZ conversion.
"""
import math
import itertools

import numpy as np
import cirq

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


def is_identity_up_to_phase(U, atol=1e-9):
    ph = U[0, 0] if abs(U[0, 0]) > 0.5 else 1.0
    return abs(abs(U[0, 0]) - 1) < atol and np.allclose(U / ph * abs(ph), _I2, atol=atol) and \
        np.allclose(U, U[0, 0] * _I2, atol=atol)


# ------------------------------------------------------------------ hardware graph
def hw_graph(cz_err, sq_err):
    """adjacency dict: q -> {nbr: cz_error}; only qubits with single-qubit gates allowed."""
    adj = {q: {} for q in sq_err}
    for pair, val in cz_err.items():
        a, b = tuple(pair)
        p = float(val[0])
        if a in adj and b in adj:
            adj[a][b] = p
            adj[b][a] = p
    return adj


def readout_cost(q, n_frac, zero_err, one_err, x_weight=0.5):
    e0 = float(zero_err.get(q, (0.05, 0))[0])
    e1 = float(one_err.get(q, (0.05, 0))[0])
    z = (1 - n_frac) * e0 + n_frac * e1
    x = 0.5 * (e0 + e1)
    return z + x_weight * x


# ------------------------------------------------------------------ embedding
def template_edges(t):
    cnt = {}
    for op in t['ops']:
        if op[0] == 'cz':
            k = (min(op[1], op[2]), max(op[1], op[2]))
            cnt[k] = cnt.get(k, 0) + 1
    return cnt


def node_weights(t):
    """number of 1q gates per node (for sq error weighting) and lifetime proxy."""
    k = t['k']
    n1 = [0] * k
    for op in t['ops']:
        if op[0] == 'u':
            n1[op[1]] += 1
    return n1


def embed(t, adj, node_cost, edge_weight=1.0, beam=3000, exhaustive_limit=200000, return_all=False):
    """Find an injective map template-node -> hardware qubit s.t. every template edge is a hardware edge,
    minimising sum(edge_weight * count * p_cz) + sum(node_cost[node](q)).
    node_cost: function(node, q) -> float.
    Beam search over nodes in BFS order of the template graph."""
    k = t['k']
    edges = template_edges(t)
    nbrs = {i: set() for i in range(k)}
    for (a, b) in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    # node order: start from max-degree node, BFS, prefer nodes with most already-placed neighbours
    order = [max(range(k), key=lambda i: len(nbrs[i]))]
    while len(order) < k:
        cand = [i for i in range(k) if i not in order]
        best = max(cand, key=lambda i: (len(nbrs[i] & set(order)), len(nbrs[i])))
        order.append(best)
    qubits = sorted(adj)
    states = []  # (cost, assignment dict)
    first = order[0]
    for q in qubits:
        if len(adj[q]) >= len(nbrs[first]):
            states.append((node_cost(first, q), {first: q}))
    for node in order[1:]:
        placed_nbrs = [m for m in nbrs[node] if True]
        new = []
        for cost, asg in states:
            used = set(asg.values())
            pn = [m for m in nbrs[node] if m in asg]
            if pn:
                cands = set(adj[asg[pn[0]]])
                for m in pn[1:]:
                    cands &= set(adj[asg[m]])
            else:
                cands = set(qubits)
            for q in cands:
                if q in used:
                    continue
                c = cost + node_cost(node, q)
                for m in pn:
                    c += edge_weight * edges[(min(node, m), max(node, m))] * adj[q][asg[m]]
                d = dict(asg)
                d[node] = q
                new.append((c, d))
        if not new:
            return None
        new.sort(key=lambda x: x[0])
        # diversity: dedupe identical qubit sets
        states = new[:beam]
    return states[0] if not return_all else states


# ------------------------------------------------------------------ scheduling
def build_circuit(t, mapping, dd=True, align='alap', eject_z=True):
    """Turn a template + mapping into a layered cirq.Circuit of PhasedXZ and CZ moments."""
    k = t['k']
    q = [mapping[i] for i in range(k)]
    # 1) collapse consecutive 1q gates; produce sequence of 'blocks' per node between CZs
    ops = t['ops']
    # layered ASAP schedule: each CZ gets a CZ-layer index; 1q gates live in 1q-layers between CZ layers.
    # 1q-layer index L means before CZ layer L (layer L=numCZlayers is the final 1q layer).
    pend = [_I2.copy() for _ in range(k)]   # accumulated unitary since last CZ per node
    has_pend = [False] * k
    node_layer = [0] * k                    # next CZ layer index available per node
    cz_layers = []                          # list of list of (i,j)
    oneq = {}                               # (layer, node) -> unitary applied in 1q layer before CZ layer `layer`
    for op in ops:
        if op[0] == 'u':
            pend[op[1]] = op[2] @ pend[op[1]]
            has_pend[op[1]] = True
        else:
            i, j = op[1], op[2]
            L = max(node_layer[i], node_layer[j])
            # 1q gates pending on i/j must be placed in 1q layer L (just before)
            for a in (i, j):
                if has_pend[a]:
                    oneq[(L, a)] = pend[a] @ oneq.get((L, a), _I2)
                    pend[a] = _I2.copy()
                    has_pend[a] = False
            while len(cz_layers) <= L:
                cz_layers.append([])
            cz_layers[L].append((i, j))
            node_layer[i] = node_layer[j] = L + 1
    nL = len(cz_layers)
    for a in range(k):
        if has_pend[a]:
            oneq[(nL, a)] = pend[a] @ oneq.get((nL, a), _I2)
    # remove identities
    oneq = {key: U for key, U in oneq.items() if not is_identity_up_to_phase(U)}
    if align == 'alap':
        cz_layers, oneq = _alap(k, cz_layers, oneq)
    if eject_z:
        oneq = _eject_z(k, nL, oneq)
    # DD
    if dd:
        oneq = _add_dd(k, cz_layers, oneq)
    # emit
    moments = []
    for L in range(nL + 1):
        m1 = [cirq.PhasedXZGate.from_matrix(oneq[(L, a)]).on(q[a]) for a in range(k) if (L, a) in oneq]
        if m1:
            moments.append(cirq.Moment(m1))
        if L < nL and cz_layers[L]:
            moments.append(cirq.Moment([cirq.CZ(q[i], q[j]) for (i, j) in cz_layers[L]]))
    return cirq.Circuit(moments)


def _split_z(U):
    """U = Rz-type diag D @ W with W = Z^a X^x Z^-a (no trailing Z). Returns (W, D)."""
    g = cirq.PhasedXZGate.from_matrix(U)
    W = cirq.unitary(cirq.PhasedXZGate(x_exponent=g.x_exponent, z_exponent=0, axis_phase_exponent=g.axis_phase_exponent))
    D = cirq.unitary(cirq.ZPowGate(exponent=g.z_exponent))
    return W, D


def _eject_z(k, nL, oneq):
    """Push the Z part of every 1q gate forward through CZs (they commute) into the node's next gate.
    The leftover Z after the node's last gate stays in the final 1q layer (it matters for X-basis readout)."""
    out = {}
    for a in range(k):
        slots = sorted(L for (L, b) in oneq if b == a)
        carry = _I2
        for L in slots:
            U = oneq[(L, a)] @ carry
            W, D = _split_z(U)
            carry = D
            if not is_identity_up_to_phase(W):
                out[(L, a)] = W
        if not is_identity_up_to_phase(carry):
            if (nL, a) in out:
                out[(nL, a)] = carry @ out[(nL, a)]
            else:
                out[(nL, a)] = carry
    return out


def _alap(k, cz_layers, oneq):
    """Push CZ layers as late as possible (reverse ASAP), keeping per-node order."""
    nL = len(cz_layers)
    # per node ordered list of events: ('1q', L) and ('cz', L, idx)
    # reverse scheduling: process CZs from last layer to first, assign new layer = min(next layers of both) - 1
    next_free = [nL] * k  # new layer index bound (exclusive) per node
    new_layers = [[] for _ in range(nL)]
    remap = {}
    for L in range(nL - 1, -1, -1):
        for (i, j) in cz_layers[L]:
            nl = min(next_free[i], next_free[j]) - 1
            new_layers[nl].append((i, j))
            remap[(L, i, j)] = nl
            next_free[i] = next_free[j] = nl
    # 1q gates: a 1q gate at layer L on node a sits before the node's first CZ at layer >= L (old indexing).
    # new position = new layer of that CZ (or nL if none).
    node_cz = {a: [] for a in range(k)}
    for L in range(nL):
        for (i, j) in cz_layers[L]:
            node_cz[i].append((L, remap[(L, i, j)]))
            node_cz[j].append((L, remap[(L, i, j)]))
    newq = {}
    for (L, a), U in oneq.items():
        nxt = [nl for (ol, nl) in node_cz[a] if ol >= L]
        nl = nxt[0] if nxt else nL
        newq[(nl, a)] = U @ newq.get((nl, a), _I2) if (nl, a) in newq else U
    return new_layers, newq


def _add_dd(k, cz_layers, oneq):
    """Insert X-X echo pairs on idle windows of qubits that are already 'live' (had a gate).
    Time slots: 1q-layer L sits before CZ layer L. A node idle over CZ layers [L1, L2) with L2-L1 >= 2
    gets X at 1q-layer L1+? ... we use: X in the first free 1q slot after going idle and X in the last
    free 1q slot before the next gate (merged into that gate if one exists)."""
    nL = len(cz_layers)
    busy = [set() for _ in range(k)]
    for L in range(nL):
        for (i, j) in cz_layers[L]:
            busy[i].add(L)
            busy[j].add(L)
    out = dict(oneq)
    for a in range(k):
        events = sorted(set([L for L in busy[a]] + [L for (L, b) in oneq if b == a]))
        if not events:
            continue
        start = min(events)
        # walk CZ layers after `start`: find maximal idle stretches of CZ layers (no CZ on a)
        L = start
        while L < nL:
            if L in busy[a]:
                L += 1
                continue
            s = L
            while L < nL and L not in busy[a]:
                L += 1
            e = L  # idle over CZ layers [s, e); 1q slots s..e available (slot e is before next CZ or final)
            d = e - s
            if d >= 2:
                # echo: X at slot s+a and s+b with a + (d-b) = b - a  (equal free-evolution either side)
                aa = int(round(d / 4))
                bb = aa + int(round(d / 2))
                x1, x2 = s + aa, s + bb
                # slot x1 may hold the gate that preceded the idle window (only if aa == 0): X goes after it
                out[(x1, a)] = _X @ out[(x1, a)] if (x1, a) in out else _X.copy()
                # slot x2 may hold the gate that ends the window (only if bb == d): X goes before it
                out[(x2, a)] = out[(x2, a)] @ _X if (x2, a) in out else _X.copy()
        # nothing else
    return {key: U for key, U in out.items() if not is_identity_up_to_phase(U)}


# ------------------------------------------------------------------ validation helpers
def check_allowed(circuit, cz_err, sq_err):
    allowed = set(frozenset(p) for p in cz_err)
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.CZPowGate):
            assert op.gate.exponent == 1 and frozenset(op.qubits) in allowed, op
        elif isinstance(op.gate, cirq.PhasedXZGate):
            assert op.qubits[0] in sq_err, op
        elif isinstance(op.gate, cirq.ResetChannel):
            pass
        else:
            raise AssertionError(f'bad op {op}')


# ------------------------------------------------------------------ Bartschi-Eidenbenz, nearest-neighbour path version
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def _cnot(ops, a, b):
    ops += [('u', b, _H), ('cz', a, b), ('u', b, _H)]


def _gadget2(ops, a, b, th):
    """(a,b): 00->00, 01->cos 01 + sin 10, 11->11 (input 10 unconstrained)."""
    _cnot(ops, a, b)
    ops += [('u', a, ry(-th)), ('cz', b, a), ('u', a, ry(th))]
    _cnot(ops, a, b)


def _gadget3(ops, c, a, b, th):
    """path c-a-b. c=0: (a,b) 00->00, 01->cos 01+sin 10, 11->11;  c=1: 01->10, 11->11."""
    bp = (th + math.pi / 2) / 2
    ap = (th - math.pi / 2) / 2
    _cnot(ops, a, b)
    ops += [('u', a, ry(-bp)), ('cz', c, a), ('u', a, ry(-ap)), ('cz', b, a),
            ('u', a, ry(ap)), ('cz', c, a), ('u', a, ry(bp))]
    _cnot(ops, a, b)


def _gadget3b(ops, c, a, b, th, skip_first_cnot=False):
    """path c-a-b, valid when input 111 is absent: c=0: 00->00, 01->cos 01+sin 10, 11->11; c=1: 01->10.
    4 CZ (3 if a is always 0 on the support, i.e. 011 also absent)."""
    if not skip_first_cnot:
        _cnot(ops, a, b)
    g1 = th                      # b-controlled known-zero rotation: |0> -> Ry(2 th)|0>
    g2 = th + math.pi / 2        # c-controlled: Ry(2 th)|0> -> |1>
    ops += [('u', a, ry(-g1)), ('cz', b, a), ('u', a, ry(g1)),
            ('u', a, ry(-g2)), ('cz', c, a), ('u', a, ry(g2))]
    _cnot(ops, a, b)


def be_path_template(N, n):
    """D(N,n) on path nodes 0..N-1 (node p-1 = B&E position p). Exact, ~5 CZ per gadget."""
    ops = []
    for p in range(N - n, N):
        ops.append(('u', p, _X.copy()))
    for m in range(N, 1, -1):
        k = min(n, m - 1)
        w = [m - k - 1 + i for i in range(k + 1)]  # 0-indexed nodes of window
        for i in range(k):
            th = math.acos(math.sqrt((k - i) / m))
            if i == 0:
                _gadget2(ops, w[0], w[1], th)
            else:
                _gadget3(ops, w[i - 1], w[i], w[i + 1], th)
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=n)


def template_state(t):
    """noiseless statevector of a template (node 0 = MSB)."""
    k = t['k']
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1
    for op in t['ops']:
        if op[0] == 'u':
            psi = np.moveaxis(np.tensordot(op[2], psi, axes=([1], [op[1]])), 0, op[1])
        else:
            sl = [slice(None)] * k
            sl[op[1]] = 1
            sl[op[2]] = 1
            psi[tuple(sl)] *= -1
    return psi.reshape(-1)


# ---- support-aware simplification of the B&E-NN construction
def _givens_ops(a, b, th):
    """exact 2-CZ real Givens on (a,b): 01 -> cos 01 + sin 10, 10 -> -sin 01 + cos 10 (node a = first bit)."""
    c, s = math.cos(th), math.sin(th)
    G = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=complex)
    qa, qb = cirq.LineQubit(0), cirq.LineQubit(1)
    cops = cirq.two_qubit_matrix_to_cz_operations(qa, qb, G, allow_partial_czs=False)
    ops = []
    for op in cops:
        idx = [0 if q == qa else 1 for q in op.qubits]
        if len(idx) == 2:
            ops.append(('cz', a, b))
        else:
            ops.append(('u', a if idx[0] == 0 else b, cirq.unitary(op)))
    return ops


def _classical_synth(support, edges3):
    """support: dict input_bits(tuple)->output_bits(tuple) on 3 local bits (c,a,b). Find min-CNOT circuit
    (CNOTs along edges3 in either direction, X free). Returns list of ('x',q) / ('cnot',ctl,tgt)."""
    from collections import deque
    ins = list(support)
    target = tuple(support[i] for i in ins)
    start = tuple(ins)
    nb = len(ins[0])
    moves = []
    for (u, v) in edges3:
        moves.append(('cnot', u, v))
        moves.append(('cnot', v, u))

    def app(state, mv):
        out = []
        for s in state:
            s = list(s)
            if mv[0] == 'x':
                s[mv[1]] ^= 1
            else:
                s[mv[2]] ^= s[mv[1]]
            out.append(tuple(s))
        return tuple(out)

    def xfix(state):
        # X layer needed to map state->target, if it is a pure X difference
        diff = None
        for s, t in zip(state, target):
            d = tuple(x ^ y for x, y in zip(s, t))
            if diff is None:
                diff = d
            elif d != diff:
                return None
        return diff

    seen = {start: []}
    dq = deque([start])
    while dq:
        st = dq.popleft()
        path = seen[st]
        d = xfix(st)
        if d is not None:
            return path + [('x', q) for q in range(nb) if d[q]]
        if len(path) >= 6:
            continue
        # X moves are free but we fold them in by also exploring X-then-CNOT
        for xs in itertools.product([0, 1], repeat=nb):
            st2 = st
            pre = []
            for q in range(nb):
                if xs[q]:
                    st2 = app(st2, ('x', q))
                    pre.append(('x', q))
            for mv in moves:
                st3 = app(st2, mv)
                if st3 not in seen:
                    seen[st3] = path + pre + [mv]
                    dq.append(st3)
    raise RuntimeError('classical synth failed')


def be_path_template_opt(N, n):
    """Support-aware B&E nearest-neighbour construction on a path 0..N-1."""
    ops = []
    for p in range(N - n, N):
        ops.append(('u', p, _X.copy()))
    for m in range(N, 1, -1):
        k = min(n, m - 1)
        w = [m - k - 1 + i for i in range(k + 1)]
        J = [j for j in range(max(0, n - (N - m)), min(n, m) + 1)]
        for i in range(k):
            # local input support, per branch: c-branch / s-branch patterns
            S = set()
            for j in J:
                p = k - j
                if p < 0:
                    S.add((1, 1, 1) if i >= 1 else (1, 1))
                    continue
                if i < p:
                    S.add((0, 0, 0) if i >= 1 else (0, 0))
                elif i == p:
                    S.add((0, 0, 1) if i >= 1 else (0, 1))
                elif i == p + 1:
                    S.add((0, 1, 1))
                    if m - j > 0:
                        S.add((1, 0, 1))
                else:
                    S.add((1, 1, 1))
                    if m - j > 0:
                        S.add((1, 0, 1))
            th = math.acos(math.sqrt((k - i) / m))
            if i == 0:
                if (0, 1) not in S:
                    continue
                if S == {(0, 1)}:
                    ops += [('u', w[0], ry(2 * th))]
                    _cnot(ops, w[0], w[1])
                else:
                    ops += _givens_ops(w[0], w[1], th)
                continue
            P, Q = (0, 0, 1) in S, (1, 0, 1) in S
            c, a, b = w[i - 1], w[i], w[i + 1]
            if P and Q:
                if (1, 1, 1) not in S:
                    _gadget3b(ops, c, a, b, th, skip_first_cnot=(0, 1, 1) not in S)
                else:
                    _gadget3(ops, c, a, b, th)
            elif P:
                if S <= {(0, 0, 1)}:
                    ops += [('u', a, ry(2 * th))]
                    _cnot(ops, a, b)
                else:
                    ops += _givens_ops(a, b, th)
            elif Q:
                sup = {}
                for s in S:
                    sup[s] = (1, 1, 0) if s == (1, 0, 1) else s
                circ = _classical_synth(sup, [(0, 1), (1, 2)])
                loc = [c, a, b]
                for g in circ:
                    if g[0] == 'x':
                        ops.append(('u', loc[g[1]], _X.copy()))
                    else:
                        _cnot(ops, loc[g[1]], loc[g[2]])
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=n)


# ------------------------------------------------------------------ patches & W trees on hardware
def grow_patch(adj, N, root, qcost, lam=0.002, forbid=()):
    """Greedy best-first patch of N qubits from root. qcost(q)->float. Compactness via BFS distance."""
    from collections import deque
    dist = {root: 0}
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    patch = [root]
    inP = {root}
    while len(patch) < N:
        best, bc = None, 1e9
        for u in patch:
            for v, p in adj[u].items():
                if v in inP or v in forbid:
                    continue
                c = qcost(v) + p + lam * dist.get(v, 99) ** 2
                if c < bc:
                    best, bc = v, c
        if best is None:
            return None
        patch.append(best)
        inP.add(best)
    return patch


def broadcast_tree(adj, patch, root, prefer_good=True):
    """Rounds of parent->child splits inside patch (each node in <=1 split per round).
    Returns list of rounds, each a list of (parent, child)."""
    P = set(patch)
    informed = {root}
    rounds = []
    # remaining-depth heuristic: farther-from-informed nodes first
    while len(informed) < len(P):
        busy = set()
        rnd = []
        # candidate edges informed->uninformed, prefer child with many uninformed neighbours (spreads), low error
        cands = []
        for u in informed:
            for v, p in adj[u].items():
                if v in P and v not in informed:
                    fan = sum(1 for w in adj[v] if w in P and w not in informed)
                    cands.append((-fan, p if prefer_good else 0, u, v))
        cands.sort(key=lambda x: (x[0], x[1]))
        for _, _, u, v in cands:
            if u in busy or v in busy:
                continue
            rnd.append((u, v))
            busy.add(u)
            busy.add(v)
        for u, v in rnd:
            informed.add(v)
        rounds.append(rnd)
    return rounds


def w_tree_template(patch, rounds):
    """Exact W state on patch via tree splits (2N-3 CZ)."""
    idx = {q: i for i, q in enumerate(patch)}
    N = len(patch)
    children = {q: [] for q in patch}
    for rnd in rounds:
        for u, v in rnd:
            children[u].append(v)
    size = {}

    def sz(q):
        if q not in size:
            size[q] = 1 + sum(sz(c) for c in children[q])
        return size[q]

    root = rounds[0][0][0]
    sz(root)
    remaining = {q: size[q] for q in patch}  # mass still held by node (itself + undistributed subtrees)
    ops = []
    first = True
    for rnd in rounds:
        for u, v in rnd:
            R = remaining[u]
            s = size[v]
            th = math.asin(math.sqrt(s / R))   # amplitude fraction moved to child
            a, b = idx[u], idx[v]
            if first:
                # |00> -> cos th |10> + sin th |01>
                ops.append(('u', a, ry(2 * th)))  # a: cos|0> + sin|1>
                _cnot(ops, a, b)
                ops.append(('u', a, _X.copy()))
                first = False
            else:
                eta = (math.pi - 2 * th) / 2
                ops += [('u', b, ry(eta))]
                _cnot(ops, a, b)
                ops += [('u', b, ry(-eta))]
                _cnot(ops, b, a)
            remaining[u] -= s
    if N == 1:
        ops.append(('u', 0, _X.copy()))
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=1)


def be_ops(nodes, k, jmin, jmax):
    """B&E-NN Dicke unitary on path `nodes` (len m0): maps sorted |0..01^j> (ones at the END of `nodes`)
    to D(m0, j) for every j in [jmin, jmax] (j <= k), with consistent phases. Support-pruned."""
    ops = []
    m0 = len(nodes)
    for m in range(m0, 1, -1):
        kk = min(k, m - 1)
        w = [nodes[m - kk - 1 + i] for i in range(kk + 1)]
        J = [j for j in range(max(0, jmin - (m0 - m)), min(jmax, m) + 1)]
        for i in range(kk):
            S = set()
            for j in J:
                p = kk - j
                if p < 0:
                    S.add((1, 1, 1) if i >= 1 else (1, 1))
                    continue
                if i < p:
                    S.add((0, 0, 0) if i >= 1 else (0, 0))
                elif i == p:
                    S.add((0, 0, 1) if i >= 1 else (0, 1))
                elif i == p + 1:
                    S.add((0, 1, 1))
                    if m - j > 0:
                        S.add((1, 0, 1))
                else:
                    S.add((1, 1, 1))
                    if m - j > 0:
                        S.add((1, 0, 1))
            th = math.acos(math.sqrt((kk - i) / m))
            if i == 0:
                if (0, 1) not in S:
                    continue
                if S == {(0, 1)}:
                    ops += [('u', w[0], ry(2 * th))]
                    _cnot(ops, w[0], w[1])
                else:
                    ops += _givens_ops(w[0], w[1], th)
                continue
            P, Q = (0, 0, 1) in S, (1, 0, 1) in S
            c, a, b = w[i - 1], w[i], w[i + 1]
            if P and Q:
                if (1, 1, 1) not in S:
                    _gadget3b(ops, c, a, b, th, skip_first_cnot=(0, 1, 1) not in S)
                else:
                    _gadget3(ops, c, a, b, th)
            elif P:
                if S <= {(0, 0, 1)}:
                    ops += [('u', a, ry(2 * th))]
                    _cnot(ops, a, b)
                else:
                    ops += _givens_ops(a, b, th)
            elif Q:
                sup = {s: ((1, 1, 0) if s == (1, 0, 1) else s) for s in S}
                for g in _classical_synth(sup, [(0, 1), (1, 2)]):
                    loc = [c, a, b]
                    if g[0] == 'x':
                        ops.append(('u', loc[g[1]], _X.copy()))
                    else:
                        _cnot(ops, loc[g[1]], loc[g[2]])
    return ops


def dc2_template(N, n, a=None):
    """Divide & conquer: D(N,n) = sum_j sqrt(C(a,j)C(b,n-j)/C(N,n)) D(a,j) D(b,n-j).
    Node layout: A path = nodes 0..a-1 (tail at a-1), B path = nodes a..N-1 with B position r -> node a+r
    (tail at N-1). Rungs: A[a-r] <-> B[b-n+r-1] for r=1..n (need a>=n, b>=n).
    Returns template incl. 'extra_edges' rungs (they are CZ edges used)."""
    if a is None:
        a = N // 2
    b = N - a
    assert b >= n and a >= 1
    A = list(range(a))
    B = [a + r for r in range(b)]
    p = [math.comb(a, j) * math.comb(b, n - j) / math.comb(N, n) for j in range(n + 1)]
    jmin = min(j for j in range(n + 1) if p[j] > 0)
    jmax = max(j for j in range(n + 1) if p[j] > 0)
    T = [sum(p[r:]) for r in range(n + 2)]
    ops = []
    # unary u_r on A[a-r], r=1..jmax : u_r = 1 iff j >= r
    for r in range(1, jmax + 1):
        q = A[a - r]
        if r <= jmin:
            ops.append(('u', q, _X.copy()))
            continue
        cond = T[r] / T[r - 1]  # P(j>=r | j>=r-1)
        phi = 2 * math.asin(math.sqrt(cond))
        if r == 1 or r - 1 <= jmin and False:
            ops.append(('u', q, ry(phi)))
        elif r - 1 <= jmin:
            ops.append(('u', q, ry(phi)))   # previous bit is deterministically 1
        else:
            # known-zero controlled Ry(phi), control A[a-r+1]
            eta = (math.pi - phi) / 2
            ops.append(('u', q, ry(eta)))
            _cnot(ops, A[a - r + 1], q)
            ops.append(('u', q, ry(-eta)))
    # B starts with n ones at its tail, subtract j: CNOT A[a-r] -> B[b-n+r-1]
    for r in range(1, n + 1):
        bq = B[b - n + r - 1]
        if r <= jmin:
            continue  # u_r = 1 deterministically: X then X cancels -> leave 0
        ops.append(('u', bq, _X.copy()))
        if r <= jmax:
            _cnot(ops, A[a - r], bq)
    ops += be_ops(A, n, jmin, jmax)
    ops += be_ops(B, n, n - jmax, n - jmin)
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=n)
