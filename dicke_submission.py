# DuQuantum team portfolio: Astra/Claude, ColumnForge, and peer contributions.
# Submit this file alone. Public APIs: create_dicke_state and create_dicke_state_adaptive.
# Numeric templates were independently verified; ideal fidelity is checked by tests.
# No hardware scores are claimed. No external data files or runtime fitting are used.
"""DuQuantum exact Dicke-state submission: Astra and Claude collaboration.

Hardware: create_dicke_state(N,k,cz_rates,sq_rates,zero_errors,one_errors).
Bonus: create_dicke_state_adaptive(N,k), deterministic and measurement-free.
Optional hardware echo variant: create_dicke_state_echo (paired native X pulses).

Preparation families are analytically constructed or independently numerically validated; online selection
changes only circuit family, qubit placement and gates that cancel noiselessly.
Numerical noise predictions are ranking heuristics, not Willow hardware scores.
This file needs only Cirq/Cirq-Google, their dependencies, and Python stdlib.
No external files, network access, runtime fitting or training are required.

Foundational sources: arXiv:1904.07358 and arXiv:2112.12435.
See the accompanying prompt log, derivations, validation and score report.
"""


# Source: baseline_v3_snapshot.py; SHA256: 9e4ea6ccffb14f18aa09037e2e39818a828dadf1b52677abf69af56c133f0a2d
"""Exact Dicke circuits on a linear chain, using only full CZ and PhasedXZ.

The split/cyclic-shift construction follows Baertschi and Eidenbenz,
arXiv:1904.07358. A reverse sweep of the local blocks in Fig. 2 of
Aktar et al., arXiv:2112.12435v2 shifts a unary register one site at each
recursion without restoring swaps. No ancillas or measurements are used.
"""
import math
import functools
import cirq
import numpy as np

def _reflect3(a, b, c, n, ell):
    """Five local CNOTs implementing the reverse-sweep restricted isometry.

    On the only reachable basis states:
      000 -> 000; 100 -> sqrt(ell/n)100+sqrt(1-ell/n)010;
      101 -> 011; 110 -> 110; 111 -> 111.
    """
    alpha = 0.5 * math.asin(math.sqrt(ell / n))
    yield cirq.CNOT(b, a)
    yield cirq.ry(alpha)(b)
    yield cirq.CNOT(c, b)
    yield cirq.ry(alpha)(b)
    yield cirq.CNOT(a, b)
    yield cirq.ry(-alpha)(b)
    yield cirq.CNOT(c, b)
    yield cirq.ry(-alpha)(b)
    yield cirq.CNOT(b, a)

def raw_circuit(n: int, k: int, *, use_width2=True, use_top_row=True) -> cirq.Circuit:
    """Sliding unary register; reflect/swap blocks avoid restoring swaps."""
    if not isinstance(n, int) or not isinstance(k, int) or n < 1 or (not 0 <= k <= n):
        raise ValueError('Require integer n >= 1 and 0 <= k <= n')
    q = cirq.LineQubit.range(n)
    complement = k > n // 2
    weight = n - k if complement else k
    ops = list(cirq.X.on_each(q[:max(0, weight - 1)]))
    for start in range(n - 1):
        remaining = n - start
        if use_width2 and weight == 2 and (start > 0):
            if remaining == 3:
                ops.extend(_unary_three_weight_two(q[start:start + 3]))
                break
            if remaining >= 4:
                ops.extend(_width_two_row(q[start:start + 3], remaining))
                continue
        width = min(weight, remaining - 1)
        if not width:
            break
        co = math.sqrt(width / remaining)
        si = math.sqrt((remaining - width) / remaining)
        givens = np.array([[1, 0, 0, 0], [0, co, si, 0], [0, -si, co, 0], [0, 0, 0, 1]])
        top_ell = width - 1
        if use_top_row and weight >= 3 and (start > 0) and (remaining > weight):
            ops.extend(_combined_top_row(q[start + width - 2:start + width + 1], remaining, width))
            top_ell = width - 2
        elif start == 0:
            a, b = q[width - 1:width + 1]
            ops.extend([cirq.ry(2 * math.acos(co))(a), cirq.CNOT(a, b), cirq.X(a)])
        else:
            ops.append(cirq.MatrixGate(givens)(q[start + width - 1], q[start + width]))
        for ell in range(top_ell, 0, -1):
            a, b, c = q[start + ell - 1:start + ell + 2]
            if ell < weight - start:
                ops.extend([cirq.CNOT(b, a), cirq.CNOT(a, b), cirq.X(a)])
            else:
                ops.extend(_reflect3(a, b, c, remaining, ell))
    if complement:
        ops.extend(cirq.X.on_each(q))
    if not ops:
        ops = list(cirq.I.on_each(q))
    return cirq.Circuit(ops)

def _unary_three_weight_two(q):
    """Proved six-CZ isometry, valid ONLY on unary weights 0, 1, and 2.

    Analytic construction: research/exact_u3.py. All 24 promised amplitudes
    were proved symbolically in research/verify_exact_u3.py. Input 111 is
    deliberately outside this isometry's supported domain.
    """
    if len(q) != 3:
        raise ValueError('Expected exactly three line qubits')
    a = math.asin(1 / math.sqrt(3))
    angles = [-math.pi / 2, math.pi, a - math.pi, -math.pi / 2, a - math.pi, math.pi / 2 + a, a, -math.pi / 2, -math.pi / 2, -math.pi / 2, 0, -math.pi / 4, math.pi / 4, math.pi / 2, math.pi / 2]
    edges = [(1, 2), (0, 1), (1, 2), (0, 1), (1, 2), (1, 2)]
    ops = [cirq.ry(angles[i])(q[i]) for i in range(3)]
    for j, (u, v) in enumerate(edges):
        ops.append(cirq.CZ(q[u], q[v]))
        if angles[3 + 2 * j]:
            ops.append(cirq.ry(angles[3 + 2 * j])(q[u]))
        if angles[4 + 2 * j]:
            ops.append(cirq.ry(angles[4 + 2 * j])(q[v]))
    return ops

def _unary_three_full(q):
    """Proved eight-CZ full unary isometry, including input 111.

    research/verify_exact_u3_full.py proves all 32 amplitudes of this analytic
    construction, with common global phase -1 in the Ry representation.
    """
    if len(q) != 3:
        raise ValueError('Expected exactly three line qubits')
    u = math.atan(math.sqrt(3) - 1)
    v = math.atan((math.sqrt(3) - 1) / 2)
    a = math.acos(1 / (3 + math.sqrt(3)))
    b = math.acos(-1 / (3 - math.sqrt(3)))
    angles = [-math.pi / 2, math.pi / 2, math.pi / 2, u - math.pi / 2, math.pi - v, -math.pi, 0, a, b, math.pi / 2, -v, -math.pi / 2, u, -math.pi / 2, math.pi / 6, -math.pi / 2, math.pi / 3, math.pi / 2, math.pi / 2]
    edges = [(1, 2), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
    ops = [cirq.ry(angles[i])(q[i]) for i in range(3)]
    for j, (u, v) in enumerate(edges):
        ops.append(cirq.CZ(q[u], q[v]))
        if angles[3 + 2 * j]:
            ops.append(cirq.ry(angles[3 + 2 * j])(q[u]))
        if angles[4 + 2 * j]:
            ops.append(cirq.ry(angles[4 + 2 * j])(q[v]))
    return ops

def _combined_top_row(q, n, width):
    """Proved four-CZ replacement for the top Givens and reflect blocks.

    Its supported columns are 000, 100, and 110. The entire active unary
    register must have a trailing zero, so callers require n>max_weight.
    Input111 is deliberately unsupported. The proof uses independent
    half-angle variables in research/prove_upper_pair_four.py. All three
    promised columns have the same positive phase in this Ry representation.
    """
    if len(q) != 3 or not 2 <= width < n:
        raise ValueError('Require three ordered qubits and 2<=width<n')
    a = math.acos(math.sqrt(width / n))
    b = math.asin(math.sqrt((width - 1) / n))
    p = math.pi
    return [cirq.ry(-3 * p / 2)(q[0]), cirq.ry(a - p)(q[2]), cirq.CZ(q[1], q[2]), cirq.ry(p / 2)(q[1]), cirq.ry(p - a)(q[2]), cirq.CZ(q[0], q[1]), cirq.ry(b + p / 2)(q[0]), cirq.ry(p - b)(q[1]), cirq.CZ(q[1], q[2]), cirq.ry(-p / 2)(q[1]), cirq.CZ(q[0], q[1]), cirq.ry(p / 2)(q[0]), cirq.ry(p / 2)(q[1])]

def _width_two_row(q, n):
    return _combined_top_row(q, n, 2)

def _unary_four_weight_two(q):
    """Proved twelve-CZ unary isometry on weights zero through two only.

    research/verify_exact_u4.py checks all 48 amplitudes symbolically. The
    action on unary inputs of weight three and four is not specified.
    """
    if len(q) != 4:
        raise ValueError('Expected exactly four line qubits')
    p = math.pi
    a = math.asin(1 / math.sqrt(3))
    angles = [p / 2, 0, 3 * p / 4, a, 0, p / 4, -p / 2, p / 3, 2 * p / 3, 0, p / 2, 5 * p / 6, -5 * p / 6, 0, -p / 2, 0, p / 2 + a, a, p / 2, -a, -p / 2, -p / 4, p / 2, -p, -p / 2, p / 4, p / 2, 0]
    edges = [(1, 2), (0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (2, 3), (1, 2), (2, 3), (1, 2), (2, 3), (2, 3)]
    ops = [cirq.ry(angles[i])(q[i]) for i in range(4) if angles[i]]
    for j, (u, v) in enumerate(edges):
        ops.append(cirq.CZ(q[u], q[v]))
        if angles[4 + 2 * j]:
            ops.append(cirq.ry(angles[4 + 2 * j])(q[u]))
        if angles[5 + 2 * j]:
            ops.append(cirq.ry(angles[5 + 2 * j])(q[v]))
    return ops

def unary_to_dicke_circuit(n: int, max_weight: int, *, native=True, use_full_u3=True, use_u4=False, use_width2=True, use_top_row=True) -> cirq.Circuit:
    """Coherently map |1^k 0^(n-k)> -> |D(n,k)> for every k<=max_weight.

    No preparation gates are included. Relative phases between different k are
    preserved, so this can be used after a coherent unary-weight distribution.
    """
    if n < 1 or not 0 <= max_weight <= n:
        raise ValueError('Require n>=1 and 0<=max_weight<=n')
    q = cirq.LineQubit.range(n)
    ops = []
    for start in range(n - 1):
        remaining = n - start
        if remaining == 4 and max_weight == 2 and use_u4:
            ops.extend(_unary_four_weight_two(q[start:start + 4]))
            break
        if remaining == 3 and max_weight == 2:
            ops.extend(_unary_three_weight_two(q[start:start + 3]))
            break
        if remaining == 3 and max_weight >= 3 and use_full_u3:
            ops.extend(_unary_three_full(q[start:start + 3]))
            break
        if remaining >= 4 and max_weight == 2 and use_width2:
            ops.extend(_width_two_row(q[start:start + 3], remaining))
            continue
        width = min(max_weight, remaining - 1)
        if not width:
            break
        co = math.sqrt(width / remaining)
        si = math.sqrt((remaining - width) / remaining)
        givens = np.array([[1, 0, 0, 0], [0, co, si, 0], [0, -si, co, 0], [0, 0, 0, 1]])
        if use_top_row and max_weight >= 3 and (remaining > max_weight):
            ops.extend(_combined_top_row(q[start + width - 2:start + width + 1], remaining, width))
            top_ell = width - 2
        else:
            ops.append(cirq.MatrixGate(givens)(q[start + width - 1], q[start + width]))
            top_ell = width - 1
        for ell in range(top_ell, 0, -1):
            ops.extend(_reflect3(*q[start + ell - 1:start + ell + 2], remaining, ell))
    result = cirq.Circuit(ops)
    if native:
        result = native_compile(result)
    return result

def native_compile(raw: cirq.AbstractCircuit) -> cirq.Circuit:
    """Exact local compilation retaining the pipelined LNN schedule.

    Full-circuit target compilation puts additional dependencies between
    disjoint blocks. Local compilation followed by EARLIEST repacking gives
    the same CZ count and substantially fewer moments and single-qubit gates.
    """
    ops = []
    for op in raw.all_operations():
        if op.gate == cirq.CNOT:
            a, b = op.qubits
            ops.extend([cirq.H(b), cirq.CZ(a, b), cirq.H(b)])
        elif len(op.qubits) == 1 or op.gate == cirq.CZ:
            ops.append(op)
        else:
            ops.extend(cirq.two_qubit_matrix_to_cz_operations(*op.qubits, cirq.unitary(op), allow_partial_czs=False))
    result = cirq.Circuit(ops)
    result = cirq.merge_single_qubit_gates_to_phxz(result, atol=1e-14)
    result = cirq.drop_negligible_operations(result, atol=1e-14)
    return cirq.Circuit(result.all_operations())

@functools.lru_cache(maxsize=256)
def _logical_frozen(n: int, k: int) -> cirq.FrozenCircuit:
    raw = raw_circuit(n, k)
    native = native_compile(raw)
    return cirq.FrozenCircuit(native)

def logical_circuit(n: int, k: int) -> cirq.Circuit:
    """Return a fresh exact Dicke(n,k) circuit on LineQubit.range(n)."""
    return cirq.Circuit(_logical_frozen(n, k))


# Source: placement.py; SHA256: 9ef6c91ae55a7ab0e34f389608235e5954bc28b27f60b6f1ce4dc8c421bb2a86
"""Deterministic, calibration-aware path placement; no extra dependencies."""
import math
from collections import Counter
import cirq

def rate(value, uncertainty_weight=0.25):
    """Small uncertainty penalty discourages selecting noisy calibration outliers."""
    if isinstance(value, (tuple, list)):
        return max(0.0, min(0.49, float(value[0]) + uncertainty_weight * float(value[1])))
    try:
        return max(0.0, min(0.49, float(value[0]) + uncertainty_weight * float(value[1])))
    except (TypeError, IndexError):
        return max(0.0, min(0.49, float(value)))

def calibration_graph(cz_rates, sq_rates):
    nodes = sorted(sq_rates)
    graph = {q: {} for q in nodes}
    for (a, b), value in cz_rates.items():
        if a == b or a not in graph or b not in graph:
            continue
        p = rate(value)
        p = max(p, graph[a].get(b, 0.0))
        graph[a][b] = graph[b][a] = p
    return graph

def logical_weights(circuit, n):
    sq = [0] * n
    edges = [0] * max(0, n - 1)
    for op in circuit.all_operations():
        ids = [q.x for q in op.qubits]
        if len(ids) == 1:
            sq[ids[0]] += 1
        elif len(ids) == 2 and abs(ids[0] - ids[1]) == 1:
            edges[min(ids)] += 1
        else:
            raise ValueError(f'Expected line-neighbor circuit, got {op!r}')
    return (sq, edges)

def path_candidates(n, k, cz_rates, sq_rates, zero_errors, one_errors, sq_weights=None, edge_weights=None, width=768, count=12):
    """Beam-search paths, weighted by the exact logical gate-use profile.

    Readout dispersion penalty is the first-order shell nonuniformity term:
    Var(sum of k selected d_i) = k(N-k)/(N(N-1)) * sum((d_i-mean d)^2),
    where d_i = log(1-e1_i)-log(1-e0_i). It is a ranking heuristic;
    independent noisy simulation is required to establish an actual score.
    """
    if n < 1 or k < 0 or k > n:
        raise ValueError('Require 1 <= N and 0 <= k <= N')
    graph = calibration_graph(cz_rates, sq_rates)
    if n > len(graph):
        raise NotImplementedError('Not enough calibrated qubits')
    sq_weights = sq_weights if sq_weights is not None else [3.0] * n
    edge_weights = edge_weights if edge_weights is not None else [2.0] * (n - 1)
    if len(sq_weights) != n or len(edge_weights) != n - 1:
        raise ValueError('Incorrect gate-use profile size')
    frac = k / n
    readout, asym, singles = ({}, {}, {})
    for q in graph:
        e0 = rate(zero_errors.get(q, (0.01, 0.0)))
        e1 = rate(one_errors.get(q, (0.02, 0.0)))
        readout[q] = -((1 - frac) * math.log1p(-e0) + frac * math.log1p(-e1))
        asym[q] = math.log1p(-e1) - math.log1p(-e0)
        singles[q] = rate(sq_rates[q])
    variance_scale = 10000.0 / math.comb(n, k) * k * (n - k) / (n * max(1, n - 1))

    def node_cost(q, i):
        return readout[q] + sq_weights[i] * singles[q]
    states = [(node_cost(q, 0), (q,), node_cost(q, 0), asym[q], asym[q] ** 2) for q in graph]
    for position in range(1, n):
        expanded = []
        for _, path, additive, ds, d2 in states:
            for q, p in graph[path[-1]].items():
                if q in path:
                    continue
                cost = additive + node_cost(q, position) + edge_weights[position - 1] * p
                new_ds, new_d2 = (ds + asym[q], d2 + asym[q] ** 2)
                dispersion = max(0.0, new_d2 - new_ds ** 2 / (position + 1))
                expanded.append((cost + variance_scale * dispersion, path + (q,), cost, new_ds, new_d2))
        if not expanded:
            raise NotImplementedError('No calibrated path found for requested N')
        expanded.sort(key=lambda item: (item[0], item[1]))
        states, per_end = ([], Counter())
        cap = max(8, width // max(1, len(graph)) * 3)
        for state in expanded:
            end = state[1][-1]
            if per_end[end] < cap:
                states.append(state)
                per_end[end] += 1
                if len(states) >= width:
                    break
    return [(list(s[1]), s[0]) for s in sorted(states, key=lambda item: (item[0], item[1]))[:count]]

def place_line(circuit, n, k, cz_rates, sq_rates, zero_errors, one_errors):
    sq, edges = logical_weights(circuit, n)
    path, _ = path_candidates(n, k, cz_rates, sq_rates, zero_errors, one_errors, sq_weights=sq, edge_weights=edges, count=1)[0]
    mapping = {cirq.LineQubit(i): path[i] for i in range(n)}
    return (circuit.transform_qubits(mapping), set(path))

def line_placement_candidates(circuit, n, k, cz_rates, sq_rates, zero_errors, one_errors, count=2):
    sq, edges = logical_weights(circuit, n)
    paths = path_candidates(n, k, cz_rates, sq_rates, zero_errors, one_errors, sq_weights=sq, edge_weights=edges, count=count)
    result = []
    for path, cost in paths:
        mapping = {cirq.LineQubit(i): path[i] for i in range(n)}
        result.append((cost, circuit.transform_qubits(mapping), set(path)))
    return result


# Source: evaluation/bell_optimized.py; SHA256: bfc97a89efeddf88253a137ac91c0ae73c8ba328fec2bd83043e7a0358b54fff
"""Standalone, calibration-aware exact Dicke(2,1) preparation.

Only Cirq and the Python standard library are required. This optimizer ranks
all calibrated couplers in both orientations using an analytic Pauli/readout
proxy for the expected official score denominator, including finite shots.
It returns exactly one CZ and three PhasedXZ gates in three moments.
"""
import cirq

def _rate(mapping, key):
    value = mapping[key]
    return float(value[0] if isinstance(value, (tuple, list)) else value)

def bell_proxy(control, target, pair, cz_rates, sq_rates, zero_errors, one_errors, shots=10000, basis_rotation_noise=True):
    """Analytic independent-Pauli proxy; not a hardware prediction.

    Returns both Z/X probabilities, ordered control then target in big endian,
    and a deterministic objective. E[1/T] for retained count T is expanded
    through O((S*f)^-3); with S=10,000 and normal readout, omitted corrections
    are negligible compared with model mismatch. No-noise score is still
    random; expected denominator is a robust surrogate, not E[1/denominator].
    """
    pa, pb, pc = (_rate(sq_rates, control), _rate(sq_rates, target), _rate(cz_rates, pair))
    sa, sb, sc = (1 - 4 * pa / 3, 1 - 4 * pb / 3, 1 - 16 * pc / 15)
    zz = -sb * sb * sc
    xx = sa * sb * sc
    if basis_rotation_noise:
        xx *= sa * sb
    da = _rate(one_errors, control) - _rate(zero_errors, control)
    db = _rate(one_errors, target) - _rate(zero_errors, target)
    ra = 1 - _rate(one_errors, control) - _rate(zero_errors, control)
    rb = 1 - _rate(one_errors, target) - _rate(zero_errors, target)
    zcorr, xcorr = (da * db + ra * rb * zz, da * db + ra * rb * xx)

    def probabilities(correlation):
        return tuple(((1 + s * da + t * db + s * t * correlation) / 4 for s, t in ((1, 1), (1, -1), (-1, 1), (-1, -1))))
    f = (1 - zcorr) / 2
    if f <= 0:
        expected_z = float('inf')
    else:
        delta = (da - db) / (2 * f)
        mu = shots * f
        inverse_retained = 1 / mu + (1 - f) / mu ** 2 + (1 - f) * (2 - f) / mu ** 3
        expected_z = shots / 2 * (delta * delta + (1 - delta * delta) * inverse_retained)
    xmean1 = (da + db) / 2
    expected_x = xmean1 * xmean1 + (1 - xcorr) ** 2 + ((1 + xcorr) / 2 - xmean1 * xmean1 + 1 - xcorr * xcorr) / shots
    return {'denominator': expected_z + expected_x, 'expected_z': expected_z, 'expected_x': expected_x, 'shell_probability': f, 'z_probs': probabilities(zcorr), 'x_probs': probabilities(xcorr)}

def ranked_pairs(cz_rates, sq_rates, zero_errors, one_errors):
    available = set(sq_rates) & set(zero_errors) & set(one_errors)
    rows = []
    for pair in cz_rates:
        if len(pair) != 2 or not set(pair) <= available:
            continue
        for control, target in (tuple(pair), tuple(reversed(pair))):
            proxy = bell_proxy(control, target, pair, cz_rates, sq_rates, zero_errors, one_errors)
            rows.append((proxy['denominator'], control, target, pair, proxy))
    return sorted(rows, key=lambda row: (row[0], row[1], row[2]))

def bell_circuit(control, target):
    h = cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H))
    hx = cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H) @ cirq.unitary(cirq.X))
    return cirq.Circuit(cirq.Moment([h(control), hx(target)]), cirq.Moment([cirq.CZ(control, target)]), cirq.Moment([h(target)]))

def _create_bell(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    if (num_qubits, num_excitations) != (2, 1):
        raise NotImplementedError('This specialized circuit prepares Dicke(2,1)')
    rows = ranked_pairs(cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    if not rows:
        raise NotImplementedError('No calibrated coupler available')
    _, control, target, _, _ = rows[0]
    return (bell_circuit(control, target), {control, target})


# Source: research/exact4.py; SHA256: 15c99748b863e20c22054fa5f53d7067ced8f1225d6126556541ddd788cd90cf
"""Analytic five-CZ, seven-moment square preparation of D(4,2).

Discovered by offline topology search; the final circuit has exact closed-form
angles and an independent symbolic amplitude proof in verify_exact4.py.
The four qubits must be ordered clockwise around a fully coupled square.
No numerical optimization is performed when constructing this circuit.
"""
import math
import cirq

def dicke4_square(qubits):
    """Return D(4,2) on a,b,c,d with grid edges ab,bc,cd,da."""
    if len(qubits) != 4 or len(set(qubits)) != 4:
        raise ValueError('Need four distinct qubits around a square.')
    a, b, c, d = qubits

    def ry(q, theta):
        return cirq.PhasedXZGate(x_exponent=theta / math.pi, z_exponent=0, axis_phase_exponent=0.5).on(q)
    return cirq.Circuit([cirq.Moment([ry(a, math.acos(-1 / 3)), ry(b, math.pi / 4), ry(c, math.pi / 2), ry(d, -math.pi / 2)]), cirq.Moment([cirq.CZ(a, b), cirq.CZ(c, d)]), cirq.Moment([ry(b, math.pi / 4), cirq.CZ(d, a)]), cirq.Moment([cirq.CZ(b, c), ry(d, math.pi / 2), ry(a, -math.pi / 2)]), cirq.Moment([ry(b, math.pi / 2)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([ry(a, math.pi / 2)])])


# Source: small_density.py; SHA256: 35dbe8603a260d390f8c690220dcad04b4edb9b057a1c0e03aee74ae052e31b4
"""Exact small native-circuit density evolution for the calibration proxy.

Uniform nonidentity-Pauli noise on a d-dimensional subsystem is
  (1 - d*d*p/(d*d-1))*rho + d*p/(d*d-1)*I*partial_trace(rho).
This is algebraically the same channel as cirq.depolarize(p, n_qubits=m).
The partial-trace form avoids enumerating all 15 two-qubit Pauli branches.
"""
import cirq
import numpy as np

def native_density(circuit, qubits, cz_rates, sq_rates):
    n = len(qubits)
    positions = {q: i for i, q in enumerate(qubits)}
    rates = {}
    for pair, value in cz_rates.items():
        key = frozenset(pair)
        rates[key] = max(rates.get(key, 0.0), float(value[0]))
    rho = np.zeros((2,) * (2 * n), dtype=np.complex128)
    rho[(0,) * (2 * n)] = 1.0
    layouts = {}
    for op in circuit.all_operations():
        targets = tuple(sorted((positions[q] for q in op.qubits)))
        if len(targets) == 1 and isinstance(op.gate, cirq.PhasedXZGate):
            q = targets[0]
            unitary = cirq.unitary(op.gate)
            rho = np.moveaxis(np.tensordot(unitary, rho, axes=(1, q)), 0, q)
            rho = np.moveaxis(np.tensordot(unitary.conj(), rho, axes=(1, n + q)), 0, n + q)
            p = float(sq_rates[op.qubits[0]][0])
        elif len(targets) == 2 and op.gate == cirq.CZ:
            shape = [1] * (2 * n)
            for q in targets:
                shape[q] = 2
            diagonal = np.array([[1.0, 1.0], [1.0, -1.0]])
            rho *= diagonal.reshape(shape)
            shape = [1] * (2 * n)
            for q in targets:
                shape[n + q] = 2
            rho *= diagonal.reshape(shape)
            p = rates[frozenset(op.qubits)]
        else:
            raise ValueError('Proxy accepts native PhasedXZ and full CZ gates only')
        p = min(1.0, max(0.0, p))
        if p == 0:
            continue
        if targets not in layouts:
            rest = tuple((i for i in range(n) if i not in targets))
            perm = targets + rest + tuple((n + i for i in targets)) + tuple((n + i for i in rest))
            inverse = tuple(np.argsort(perm))
            layouts[targets] = (perm, inverse, 1 << len(targets), 1 << len(rest))
        perm, inverse, d, remainder = layouts[targets]
        block = rho.transpose(perm).reshape(d, remainder, d, remainder)
        partial = np.einsum('iaib->ab', block)
        block = (1 - d * d * p / (d * d - 1)) * block
        for i in range(d):
            block[i, :, i, :] += d * p / (d * d - 1) * partial
        rho = block.reshape((2,) * (2 * n)).transpose(inverse)
    return rho.reshape(1 << n, 1 << n)


# Source: small_exact.py; SHA256: 7821be6f5b8576f2b99478b92c8d10ff748d32c44d42100e09cd31497ad4fa83
"""Exact analytic D(4,2) with calibration-aware square selection."""
import itertools
import math
import cirq
import numpy as np

def _actual(mapping, key):
    value = mapping[key]
    return float(value[0])

def square_candidates(cz, sq, zero, one):
    graph = calibration_graph(cz, sq)
    candidates = []
    for a in graph:
        for b in graph[a]:
            for c in graph[b]:
                if c == a:
                    continue
                for d in graph[c]:
                    if d in (a, b) or a not in graph[d]:
                        continue
                    qs = (a, b, c, d)
                    score = sum((w * rate(sq[q]) for q, w in zip(qs, (3, 3, 1, 2))))
                    score += 2 * graph[a][b] + graph[b][c] + graph[c][d] + graph[d][a]
                    score += sum(((rate(zero.get(q, (0.01, 0.0))) + rate(one.get(q, (0.02, 0.0)))) / 2 for q in qs))
                    ds = [math.log1p(-rate(one.get(q, (0.02, 0.0)))) - math.log1p(-rate(zero.get(q, (0.01, 0.0)))) for q in qs]
                    score += 10000 / 18 * sum(((d - sum(ds) / 4) ** 2 for d in ds))
                    candidates.append((score, qs))
    return sorted(candidates, key=lambda row: (row[0], row[1]))

class _SmallNoise(cirq.NoiseModel):

    def __init__(self, cz, sq):
        self.sq = sq
        self.cz = {}
        for pair, v in cz.items():
            key = frozenset(pair)
            self.cz[key] = max(self.cz.get(key, 0.0), float(v[0]))

    def noisy_operation(self, op):
        p = _actual(self.sq, op.qubits[0]) if len(op.qubits) == 1 else self.cz[frozenset(op.qubits)]
        return [op, cirq.depolarize(p=min(1.0, max(0.0, p)), n_qubits=len(op.qubits)).on(*op.qubits)]

def _readout(probabilities, qs, zero, one, basis_noise=None):
    p = probabilities.reshape((2,) * len(qs))
    for i, q in enumerate(qs):
        e0, e1 = (float(zero.get(q, (0.01, 0.0))[0]), float(one.get(q, (0.02, 0.0))[0]))
        if basis_noise is not None:
            extra = 2 * float(basis_noise[q][0]) / 3 * (1 - e0 - e1)
            e0, e1 = (e0 + extra, e1 + extra)
        m = np.array([[1 - e0, e1], [e0, 1 - e1]])
        p = np.moveaxis(np.tensordot(m, p, axes=(1, i)), 0, i)
    return p.reshape(-1)

def small_proxy(circuit, qs, k, cz, sq, zero, one):
    """Expected official denominator under a small independent-Pauli model."""
    n = len(qs)
    zrho = native_density(circuit, qs, cz, sq)
    xrho = zrho.reshape((2,) * (2 * n))
    for axis in range(2 * n):
        a, b = (np.take(xrho, 0, axis=axis), np.take(xrho, 1, axis=axis))
        xrho = np.stack(((a + b) / math.sqrt(2), (a - b) / math.sqrt(2)), axis=axis)
    xrho = xrho.reshape((2 ** n, 2 ** n))
    z = _readout(np.real(np.diag(zrho)), qs, zero, one)
    x = _readout(np.real(np.diag(xrho)), qs, zero, one, basis_noise=sq)
    indices = [i for i in range(2 ** n) if i.bit_count() == k]
    f = float(sum((z[i] for i in indices)))
    if f <= 0:
        return float('inf')
    m = math.comb(n, k)
    delta = m * sum(((z[i] / f) ** 2 for i in indices)) - 1
    shots = 10000
    inv = 1 / (shots * f) + (1 - f) / (shots * f) ** 2 + (1 - f) * (2 - f) / (shots * f) ** 3
    expected_z = shots / m * (delta + (m - 1 - delta) * inv)
    expected_x = 0
    for ell in range(1, n + 1):

        def comb(a, b):
            return math.comb(a, b) if 0 <= b <= a else 0
        expected = comb(ell, ell // 2) * comb(n - ell, k - ell // 2) / m if ell % 2 == 0 else 0.0
        vals = np.array([sum(((-1) ** j * comb(i.bit_count(), j) * comb(n - i.bit_count(), ell - j) for j in range(ell + 1))) / comb(n, ell) for i in range(2 ** n)])
        mean = float(np.dot(x, vals))
        var = max(0.0, float(np.dot(x, vals ** 2)) - mean ** 2)
        expected_x += ((expected - mean) ** 2 + var / shots) / (expected ** 2 if expected else 1.0)
    return float(expected_z + expected_x)

def create_dicke4(cz, sq, zero, one, refine=64):
    ranked = square_candidates(cz, sq, zero, one)
    if not ranked:
        raise NotImplementedError('No intact calibrated square')
    best = None
    for _, qs in ranked[:refine]:
        circuit = dicke4_square(qs)
        proxy = small_proxy(circuit, qs, 2, cz, sq, zero, one)
        if best is None or proxy < best[0]:
            best = (proxy, circuit, set(qs))
    return (best[1], best[2])


# Source: decoupling.py; SHA256: 3cd49a543c4de4cb354dd757c29c8622572e4d8cd8289ac3dfc776088e63f8c1
"""Conservative exact echo pairs, preserving every original moment."""
import cirq

def echo_idle_qubits(circuit, min_idle_moments=18, single_qubit_moments_only=True):
    """Insert paired X pulses inside long idle intervals after first entanglement.

    Both pulses lie between consecutive operations on that qubit, so the
    ideal transformation is exactly unchanged. Candidate times approximate
    one-quarter and three-quarters of the interval's physical duration.
    Does not assume an unavailable Cirq-Google decoupling tag. This helps
    quasistatic dephasing, not Markovian dephasing, and adds pulse error.
    """
    moments = [list(m.operations) for m in circuit]
    durations = [42 if any((len(op.qubits) == 2 for op in m)) else 25 for m in moments]
    x = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)
    for q in sorted(circuit.all_qubits()):
        uses = [i for i, m in enumerate(circuit) if m.operates_on([q])]
        entanglers = [i for i, m in enumerate(circuit) if any((q in op.qubits and len(op.qubits) > 1 for op in m))]
        if not entanglers:
            continue
        first_entangler = entanglers[0]
        for left, right in zip(uses, uses[1:] + [len(circuit)]):
            if left < first_entangler or right - left - 1 < min_idle_moments:
                continue
            candidates = [i for i in range(left + 1, right) if not single_qubit_moments_only or durations[i] == 25]
            if len(candidates) < 2:
                continue
            centers = {}
            total = 0
            for i in range(left + 1, right):
                centers[i] = total + durations[i] / 2
                total += durations[i]
            first = min(candidates, key=lambda i: abs(centers[i] - total / 4))
            second = min((i for i in candidates if i > first), key=lambda i: abs(centers[i] - 3 * total / 4), default=None)
            if second is None:
                continue
            moments[first].append(x(q))
            moments[second].append(x(q))
    return cirq.Circuit((cirq.Moment(m) for m in moments))


# Source: template_data.py; SHA256: f65348eb5da86b0753db837aa8be79fff32d2a13b46d32b3355ba4309b33762c
"""Frozen native compilations of exact analytic constructions."""
BALANCED_TEMPLATES = {6: [{'qubits': [(4, 3), (4, 4), (5, 3), (5, 4), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.6309898804344547, 0.0, 0.5], ['sq', 1, 0.12337585721442501, -1.0, -0.5], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.8040867239846965, 0.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 2], ['sq', 1, 0.12337585721442501, -1.0, -0.5]], [['sq', 0, 0.5, 0.0, 0.0], ['cz', 1, 3], ['sq', 2, 0.5, -1.0, 0.5]], [['cz', 2, 4], ['sq', 3, 0.0, -1.0, -0.5], ['cz', 0, 1]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.8040867239846965, 0.0, -0.5], ['sq', 0, 0.5, 0.0, 0.0], ['sq', 1, 0.25, 0.0, 0.9999999999999991]], [['cz', 3, 2], ['cz', 0, 1]], [['sq', 2, 0.19591327601530373, 0.0, 0.5], ['sq', 3, 0.6959132760153035, 0.0, 0.5], ['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.24999999999999978, 0.5000000000000004, -4.440892098500626e-16]], [['cz', 2, 4]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 2]], [['cz', 2, 4], ['sq', 3, 0.5, 0.0, -0.5]], [['sq', 2, 0.25, 0.0, -0.5], ['sq', 4, 0.25, 0.0, 0.5]], [['cz', 2, 4]], [['sq', 2, 0.0, -1.0, -0.5], ['sq', 4, 0.0, -1.0, -0.5]], [['cz', 3, 2]], [['cz', 3, 1]], [['sq', 3, 0.5, -1.0, -0.5]], [['cz', 5, 3]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 5, 4]], [['cz', 3, 1], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 5, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 3, 2]], [['cz', 1, 0], ['sq', 2, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (5, 3), (5, 4), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.6309898804344547, 0.0, 0.5], ['sq', 1, 0.12337585721442501, -1.0, -0.5], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.8040867239846965, 0.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 2], ['sq', 1, 0.12337585721442501, -1.0, -0.5]], [['sq', 0, 0.5, 0.0, 0.0], ['cz', 1, 3], ['sq', 2, 0.5, -1.0, 0.5]], [['sq', 1, 0.5, -1.1102230246251565e-16, -4.440892098500626e-16], ['cz', 2, 4], ['sq', 3, 0.0, -1.0, -0.5]], [['cz', 0, 1], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.8040867239846965, 0.0, -0.5]], [['sq', 0, 0.25, 0.0, -1.1102230246251565e-16], ['sq', 1, 0.25, 0.0, -1.0], ['cz', 3, 2]], [['cz', 0, 1], ['sq', 2, 0.19591327601530373, 0.0, 0.5], ['sq', 3, 0.6959132760153035, 0.0, 0.5]], [['cz', 2, 4], ['sq', 0, 0.5, 0.5, -0.5], ['sq', 1, 0.5, 0.5, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 2]], [['cz', 2, 4], ['sq', 3, 1.0, 0.0, -1.0]], [['sq', 2, 0.25, 0.0, -0.5], ['sq', 4, 0.25, 0.0, 0.5], ['cz', 5, 3]], [['cz', 2, 4], ['sq', 3, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 4, 0.0, -1.0, -0.5], ['cz', 3, 1]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5]], [['cz', 2, 3]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5]], [['cz', 3, 2]], [['sq', 2, 0.5, -1.0, -0.5]], [['cz', 2, 0]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 2, 4]], [['sq', 2, 1.0, 0.0, -1.0], ['sq', 4, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (5, 3), (5, 4), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.6309898804344547, 0.0, 0.5], ['sq', 1, 0.12337585721442501, -1.0, -0.5], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.8040867239846965, 0.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 2], ['sq', 1, 0.12337585721442501, -1.0, -0.5]], [['sq', 0, 0.5, 0.0, 0.0], ['cz', 1, 3], ['sq', 2, 0.5, -1.0, 0.5]], [['cz', 2, 4], ['sq', 3, 0.0, -1.0, -0.5], ['cz', 0, 1]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.8040867239846965, 0.0, -0.5], ['sq', 0, 0.5, 0.0, 0.0], ['sq', 1, 0.25, 0.0, 0.9999999999999991]], [['cz', 3, 2], ['cz', 0, 1]], [['sq', 2, 0.19591327601530373, 0.0, 0.5], ['sq', 3, 0.6959132760153035, 0.0, 0.5], ['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.24999999999999978, 0.5000000000000004, -4.440892098500626e-16]], [['cz', 2, 4]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 2]], [['cz', 2, 4], ['sq', 3, 0.5, 0.0, -0.5]], [['sq', 2, 0.25, 0.0, -0.5], ['sq', 4, 0.25, 0.0, 0.5]], [['cz', 2, 4]], [['sq', 2, 0.0, -1.0, -0.5], ['sq', 4, 0.0, -1.0, -0.5]], [['cz', 3, 2]], [['cz', 3, 1]], [['sq', 3, 0.5, -1.0, -0.5]], [['cz', 5, 3]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 5, 4]], [['cz', 3, 1], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 5, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 3, 2]], [['cz', 1, 0], ['sq', 2, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (5, 3), (5, 4), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.6309898804344547, 0.0, 0.5], ['sq', 1, 0.12337585721442501, -1.0, -0.5], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.8040867239846965, 0.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 2], ['sq', 1, 0.12337585721442501, -1.0, -0.5]], [['sq', 0, 0.5, 0.0, 0.0], ['cz', 1, 3], ['sq', 2, 0.5, -1.0, 0.5]], [['sq', 1, 0.5, -1.1102230246251565e-16, -4.440892098500626e-16], ['cz', 2, 4], ['sq', 3, 0.0, -1.0, -0.5]], [['cz', 0, 1], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.8040867239846965, 0.0, -0.5]], [['sq', 0, 0.25, 0.0, -1.1102230246251565e-16], ['sq', 1, 0.25, 0.0, -1.0], ['cz', 3, 2]], [['cz', 0, 1], ['sq', 2, 0.19591327601530373, 0.0, 0.5], ['sq', 3, 0.6959132760153035, 0.0, 0.5]], [['cz', 2, 4], ['sq', 0, 0.5, 0.5, -0.5], ['sq', 1, 0.5, 0.5, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 2]], [['cz', 2, 4], ['sq', 3, 1.0, 0.0, -1.0]], [['sq', 2, 0.25, 0.0, -0.5], ['sq', 4, 0.25, 0.0, 0.5], ['cz', 5, 3]], [['cz', 2, 4], ['sq', 3, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 4, 0.0, -1.0, -0.5], ['cz', 3, 1]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5]], [['cz', 2, 3]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5]], [['cz', 3, 2]], [['sq', 2, 0.5, -1.0, -0.5]], [['cz', 2, 0]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 2, 4]], [['sq', 2, 1.0, 0.0, -1.0], ['sq', 4, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 4), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6)], 'moments': [[['sq', 0, 0.7048327646991335, 0.0, -0.5], ['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5]], [['cz', 0, 1]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2]], [['cz', 0, 3], ['cz', 1, 4], ['sq', 2, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, 0.0, 0.0], ['sq', 3, 0.5, -0.5, -1.0], ['sq', 1, 0.5, 0.0, -4.440892098500626e-16], ['cz', 2, 5], ['sq', 4, 0.49999999999999956, -0.5000000000000009, -1.0]], [['cz', 0, 1], ['sq', 2, 0.5, 0.0, -4.440892098500626e-16], ['sq', 5, 0.49999999999999956, -0.5000000000000009, -1.0], ['cz', 3, 4]], [['sq', 0, 0.30408672398469627, 0.0, 0.0], ['sq', 1, 0.30408672398469627, 0.0, -1.0], ['sq', 3, 0.30408672398469627, 0.0, 0.0], ['sq', 4, 0.30408672398469627, 0.0, -1.0]], [['cz', 0, 1], ['cz', 3, 4]], [['sq', 0, 0.5, 0.0, -1.0], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 1, 2], ['cz', 4, 5]], [['sq', 1, 0.25, 0.0, -1.1102230246251565e-16], ['sq', 2, 0.25, 0.0, -1.0], ['sq', 4, 0.25, 0.0, -1.1102230246251565e-16], ['sq', 5, 0.25, 0.0, -1.0]], [['cz', 1, 2], ['cz', 4, 5]], [['sq', 1, 0.5, 0.0, -1.0], ['sq', 2, 0.5, 0.0, -1.0], ['sq', 4, 0.5, 0.0, -1.0], ['sq', 5, 0.5, 0.0, -1.0]], [['cz', 0, 1], ['cz', 3, 4]], [['cz', 1, 2], ['cz', 4, 5]]]}, {'qubits': [(4, 4), (4, 5), (5, 4), (5, 5), (6, 4), (6, 5)], 'moments': [[['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.6959132760153035, 0.0, 0.5], ['sq', 2, 0.7048327646991335, 0.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.6959132760153035, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5]], [['cz', 2, 3]], [['sq', 2, 0.5, 0.0, -0.5], ['cz', 3, 5]], [['cz', 1, 3], ['cz', 2, 0], ['sq', 5, 0.5, 0.0, -0.5]], [['sq', 0, 0.5, 0.0, -0.5], ['cz', 2, 4], ['sq', 3, 0.75, 0.0, -0.5]], [['cz', 0, 1], ['sq', 2, 0.5, 0.0, -0.5], ['cz', 4, 5]], [['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.6959132760153035, 0.0, 0.5], ['sq', 4, 0.6959132760153035, 0.0, 0.5], ['sq', 5, 0.75, 0.0, -0.5]], [['cz', 1, 3], ['cz', 4, 5]], [['cz', 0, 1], ['cz', 2, 4], ['sq', 3, 0.75, 0.0, 0.5], ['sq', 5, 0.75, 0.0, 0.5]], [['sq', 1, 0.5, 0.0, 0.0], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 0, 0.5, -0.5, -0.5], ['sq', 2, 0.5, -0.5, -0.5]], [['cz', 1, 3], ['cz', 4, 5]], [['sq', 1, 0.5, 0.0, -1.0], ['sq', 4, 0.5, 0.0, -1.0]], [['cz', 0, 1], ['cz', 2, 4]], [['sq', 1, 0.0, 0.5, 0.0], ['sq', 4, 0.0, 0.5, 0.0]]]}], 8: [{'qubits': [(4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.7804574844601513, 0.0, 0.5], ['sq', 1, 0.22421722475922912, -1.0, -0.5], ['sq', 2, 0.08945618751103335, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, 0.0], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 3], ['sq', 1, 0.22421722475922912, -1.0, -0.5]], [['sq', 0, 0.5, 0.0, -0.5], ['cz', 1, 2], ['sq', 3, 0.5, 0.5, 0.0]], [['cz', 1, 4], ['sq', 2, 0.08945618751103357, -1.0, -0.5], ['cz', 3, 6]], [['sq', 1, 0.5, 0.0, 0.5], ['cz', 2, 5], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 6, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 2, 0.5, 0.0, 0.5], ['cz', 3, 6]], [['cz', 1, 2], ['sq', 3, 0.5, 0.0, 0.0], ['sq', 6, 0.5, 0.0, -1.0]], [['sq', 1, 0.2988554271499826, 0.0, -0.5], ['sq', 2, 0.8883116146610162, 0.0, 0.5], ['cz', 3, 4]], [['cz', 1, 2], ['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, -1.0, -0.5]], [['sq', 1, 1.0, 0.0, 0.5], ['cz', 6, 3]], [['cz', 0, 1], ['sq', 3, 0.125, 0.0, -0.5000000000000002]], [['sq', 0, 0.43222219976770426, 0.0, 0.5], ['sq', 1, 0.789232625404525, 0.0, 0.5], ['cz', 4, 3]], [['cz', 1, 2], ['sq', 3, 0.125, 0.0, 0.5], ['sq', 4, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, 0.5], ['sq', 2, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 6, 3]], [['cz', 0, 1], ['sq', 3, 0.375, -1.0, -0.5], ['sq', 6, 0.5, 0.0, 0.5]], [['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.2011445728500172, 0.0, 0.5], ['cz', 3, 4]], [['cz', 1, 2], ['sq', 4, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.16666666666666674, 0.0, 0.5], ['cz', 4, 5]], [['cz', 0, 1], ['sq', 4, 0.41666666666666674, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 0, 1.0, 0.0, -1.0], ['cz', 3, 4], ['sq', 1, 0.5, -0.33333333333333304, -1.0]], [['sq', 4, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 1, 2]], [['cz', 5, 4], ['sq', 1, 0.4999999999999998, -0.49999999999999933, 0.49999999999999956], ['sq', 2, 0.0, 0.5000000000000002, -2.220446049250313e-16]], [['sq', 4, 0.08333333333333326, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5], ['cz', 1, 0]], [['cz', 3, 4], ['sq', 1, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.41666666666666674, -1.0, -0.5]], [['cz', 3, 6], ['cz', 4, 5]], [['sq', 3, 0.2988554271499826, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 6, 0.8883116146610162, 0.0, 0.5]], [['cz', 3, 6]], [['sq', 3, 1.0, 0.0, 0.5]], [['cz', 4, 3]], [['sq', 3, 0.789232625404525, 0.0, 0.5], ['sq', 4, 0.43222219976770426, 0.0, 0.5]], [['cz', 3, 6]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 6, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 4, 3]], [['sq', 3, 0.2011445728500172, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 6]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 6, 0.16666666666666674, 0.0, 0.5]], [['cz', 4, 3]], [['sq', 3, 0.33333333333333326, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 6], ['cz', 4, 5]], [['sq', 3, 0.0, -1.0, -0.5], ['sq', 6, 0.0, -1.0, -0.5]], [['cz', 4, 3]], [['cz', 4, 1]], [['sq', 4, 0.5, -1.0, -0.5]], [['cz', 7, 4]], [['sq', 4, 0.5, -1.0, -0.5], ['cz', 7, 6]], [['cz', 4, 1], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 4, 3]], [['cz', 1, 0], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 4, 5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5), (6, 3), (6, 5)], 'moments': [[['sq', 0, 0.7804574844601513, 0.0, 0.5], ['sq', 1, 0.22421722475922912, -1.0, -0.5], ['sq', 2, 0.08945618751103335, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.16666666666666674, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 3], ['sq', 1, 0.22421722475922912, -1.0, -0.5]], [['sq', 0, 0.5, 0.0, -0.5], ['cz', 1, 2], ['sq', 3, 0.5, -1.0, -0.5]], [['cz', 1, 4], ['sq', 2, 0.08945618751103357, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, 0.5], ['cz', 2, 5], ['sq', 4, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['cz', 4, 3]], [['cz', 1, 2], ['sq', 3, 1.0, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5]], [['sq', 1, 0.2988554271499826, 0.0, -0.5], ['sq', 2, 0.8883116146610162, 0.0, 0.5], ['cz', 3, 6]], [['cz', 1, 2], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 6, 0.16666666666666674, 0.0, 0.5]], [['sq', 1, 1.0, 0.0, 0.5], ['cz', 4, 3]], [['cz', 0, 1], ['sq', 3, 0.25, 0.0, -0.5], ['sq', 4, 0.75, 0.0, 0.5]], [['sq', 0, 0.43222219976770426, 0.0, 0.5], ['sq', 1, 0.789232625404525, 0.0, 0.5], ['cz', 3, 6]], [['cz', 1, 2], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 6, 0.5, 0.0, 0.5]], [['sq', 1, 0.5, 0.0, 0.5], ['sq', 2, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 4, 3]], [['cz', 0, 1], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5]], [['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.2011445728500172, 0.0, 0.5], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 4, 0.41666666666666674, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.16666666666666674, 0.0, 0.5], ['cz', 3, 4]], [['cz', 0, 1], ['sq', 4, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 1, 0.33333333333333326, 0.0, 0.5], ['cz', 5, 4], ['sq', 0, 1.0, 0.0, -1.0]], [['cz', 1, 2], ['sq', 4, 0.08333333333333326, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 3, 4], ['sq', 1, 0.0, -1.0, -0.5], ['sq', 2, 0.0, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.41666666666666674, -1.0, -0.5]], [['cz', 3, 6], ['cz', 4, 5]], [['sq', 3, 0.2988554271499826, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 6, 0.8883116146610162, 0.0, 0.5], ['cz', 7, 5]], [['cz', 3, 6], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 3, 1.0, 0.0, 0.5], ['cz', 5, 2]], [['cz', 4, 3], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 3, 0.789232625404525, 0.0, 0.5], ['sq', 4, 0.43222219976770426, 0.0, 0.5]], [['cz', 3, 6]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 6, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 4, 3]], [['sq', 3, 0.2011445728500172, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 6]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 6, 0.16666666666666674, 0.0, 0.5]], [['cz', 4, 3]], [['sq', 3, 0.33333333333333326, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 6], ['cz', 4, 5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.0, -1.0, -0.5]], [['cz', 5, 4]], [['sq', 4, 0.5, -1.0, -0.5]], [['cz', 4, 1]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 4, 0.5, -1.0, -0.5]], [['cz', 3, 4]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.5, -1.0, -0.5]], [['cz', 4, 3]], [['sq', 3, 0.5, -1.0, -0.5]], [['cz', 3, 0]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 3, 6]], [['sq', 3, 1.0, 0.0, -1.0], ['sq', 6, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.7804574844601513, 0.0, 0.5], ['sq', 1, 0.22421722475922912, -1.0, -0.5], ['sq', 2, 0.08945618751103335, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.8333333333333333, 0.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 3], ['sq', 1, 0.22421722475922912, -1.0, -0.5]], [['sq', 0, 0.5, 0.0, -0.5], ['cz', 1, 2], ['sq', 3, 0.5, -1.0, -0.5]], [['cz', 1, 4], ['sq', 2, 0.08945618751103357, -1.0, -0.5], ['cz', 3, 6]], [['sq', 1, 0.5, 0.0, 0.5], ['cz', 2, 5], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 1.0, 0.0, -1.0], ['sq', 6, 0.8333333333333333, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['cz', 4, 3]], [['cz', 1, 2], ['sq', 3, 0.75, 0.0, 0.5], ['sq', 4, 0.75, 0.0, 0.5]], [['sq', 1, 0.2988554271499826, 0.0, -0.5], ['sq', 2, 0.8883116146610162, 0.0, 0.5], ['cz', 3, 6]], [['cz', 1, 2], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, 0.5]], [['sq', 1, 1.0, 0.0, 0.5], ['cz', 4, 3]], [['cz', 0, 1], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5]], [['sq', 0, 0.43222219976770426, 0.0, 0.5], ['sq', 1, 0.789232625404525, 0.0, 0.5], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 4, 0.41666666666666674, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, 0.5], ['sq', 2, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 3, 4]], [['cz', 0, 1], ['sq', 4, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.2011445728500172, 0.0, 0.5], ['cz', 5, 4]], [['cz', 1, 2], ['sq', 4, 0.08333333333333326, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.16666666666666674, 0.0, 0.5], ['cz', 3, 4]], [['cz', 0, 1], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.41666666666666674, -1.0, -0.5]], [['sq', 0, 1.0, 0.0, -1.0], ['cz', 3, 6], ['cz', 4, 5], ['sq', 1, 0.5, -0.33333333333333304, -1.0]], [['sq', 3, 0.2988554271499826, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 6, 0.8883116146610162, 0.0, 0.5], ['cz', 1, 2]], [['cz', 3, 6], ['sq', 1, 0.4999999999999998, -0.49999999999999933, 0.49999999999999956], ['sq', 2, 0.0, 0.5000000000000002, -2.220446049250313e-16]], [['sq', 3, 1.0, 0.0, 0.5], ['cz', 1, 0]], [['cz', 4, 3], ['sq', 1, 0.5, -1.0, -0.5]], [['sq', 3, 0.789232625404525, 0.0, 0.5], ['sq', 4, 0.43222219976770426, 0.0, 0.5]], [['cz', 3, 6]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 6, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 4, 3]], [['sq', 3, 0.2011445728500172, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 6]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 6, 0.16666666666666674, 0.0, 0.5]], [['cz', 4, 3]], [['sq', 3, 0.33333333333333326, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 6], ['cz', 4, 5]], [['sq', 3, 0.0, -1.0, -0.5], ['sq', 6, 0.0, -1.0, -0.5]], [['cz', 4, 3]], [['cz', 4, 1]], [['sq', 4, 0.5, -1.0, -0.5]], [['cz', 7, 4]], [['sq', 4, 0.5, -1.0, -0.5], ['cz', 7, 6]], [['cz', 4, 1], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 4, 3]], [['cz', 1, 0], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 4, 5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5), (6, 3), (6, 5)], 'moments': [[['sq', 0, 0.7804574844601513, 0.0, 0.5], ['sq', 1, 0.22421722475922912, -1.0, -0.5], ['sq', 2, 0.08945618751103335, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.8333333333333333, 0.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 3], ['sq', 1, 0.22421722475922912, -1.0, -0.5]], [['sq', 0, 0.5, 0.0, -0.5], ['cz', 1, 2], ['sq', 3, 0.5, -1.0, -0.5]], [['cz', 1, 4], ['sq', 2, 0.08945618751103357, -1.0, -0.5], ['cz', 3, 6]], [['sq', 1, 0.5, 0.0, 0.5], ['cz', 2, 5], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 1.0, 0.0, -1.0], ['sq', 6, 0.8333333333333333, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['cz', 4, 3]], [['cz', 1, 2], ['sq', 3, 0.75, 0.0, 0.5], ['sq', 4, 0.75, 0.0, 0.5]], [['sq', 1, 0.2988554271499826, 0.0, -0.5], ['sq', 2, 0.8883116146610162, 0.0, 0.5], ['cz', 3, 6]], [['cz', 1, 2], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, 0.5]], [['sq', 1, 1.0, 0.0, 0.5], ['cz', 4, 3]], [['cz', 0, 1], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5]], [['sq', 0, 0.43222219976770426, 0.0, 0.5], ['sq', 1, 0.789232625404525, 0.0, 0.5], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 4, 0.41666666666666674, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, 0.5], ['sq', 2, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 3, 4]], [['cz', 0, 1], ['sq', 4, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.2011445728500172, 0.0, 0.5], ['cz', 5, 4]], [['cz', 1, 2], ['sq', 4, 0.08333333333333326, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.16666666666666674, 0.0, 0.5], ['cz', 3, 4]], [['cz', 0, 1], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.41666666666666674, -1.0, -0.5]], [['sq', 1, 0.33333333333333326, 0.0, 0.5], ['cz', 3, 6], ['cz', 4, 5], ['sq', 0, 1.0, 0.0, -1.0]], [['cz', 1, 2], ['sq', 3, 0.2988554271499826, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 6, 0.8883116146610162, 0.0, 0.5], ['cz', 7, 5]], [['cz', 3, 6], ['sq', 1, 0.0, -1.0, -0.5], ['sq', 2, 0.0, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 3, 1.0, 0.0, 0.5], ['cz', 5, 2]], [['cz', 4, 3], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 3, 0.789232625404525, 0.0, 0.5], ['sq', 4, 0.43222219976770426, 0.0, 0.5]], [['cz', 3, 6]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 6, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 4, 3]], [['sq', 3, 0.2011445728500172, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 6]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 6, 0.16666666666666674, 0.0, 0.5]], [['cz', 4, 3]], [['sq', 3, 0.33333333333333326, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 6], ['cz', 4, 5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.0, -1.0, -0.5]], [['cz', 5, 4]], [['sq', 4, 0.5, -1.0, -0.5]], [['cz', 4, 1]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 4, 0.5, -1.0, -0.5]], [['cz', 3, 4]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.5, -1.0, -0.5]], [['cz', 4, 3]], [['sq', 3, 0.5, -1.0, -0.5]], [['cz', 3, 0]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 3, 6]], [['sq', 3, 1.0, 0.0, -1.0], ['sq', 6, 0.5, -1.0, -0.5]]]}], 10: [{'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (5, 3), (5, 4), (5, 5), (5, 6), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.8723282232918721, 0.0, 0.5], ['sq', 1, 0.3050177709578634, -1.0, -0.5], ['sq', 2, 0.17004943144282958, -1.0, -0.5], ['sq', 3, 0.070024348046551, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 4], ['sq', 1, 0.3050177709578634, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 4, 0.0, -1.0, -0.5]], [['cz', 1, 5], ['sq', 2, 0.17004943144282958, -1.0, -0.5], ['cz', 4, 8]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 4, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 8, 0.14758361765043326, 0.0, 0.5]], [['cz', 2, 6], ['sq', 3, 0.070024348046551, -1.0, -0.5], ['cz', 4, 8]], [['sq', 2, 0.5, 0.0, -1.0], ['cz', 3, 7], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 8, 0.5, 0.0, 0.5]], [['sq', 3, 0.5, 0.0, 0.0], ['cz', 4, 5]], [['cz', 2, 3], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 2, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 8, 4]], [['cz', 2, 3], ['sq', 4, 0.14102355421224377, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, 0.0, 0.0], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 5, 4]], [['cz', 2, 1], ['sq', 4, 0.14102355421224377, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.375, -1.0, -0.5], ['cz', 8, 4]], [['cz', 3, 2], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['sq', 8, 0.5, 0.0, 0.0]], [['sq', 2, 0.125, 0.0, -0.5000000000000002], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.125, 0.0, 0.5], ['cz', 5, 6]], [['cz', 3, 2], ['sq', 5, 0.3910235542122438, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.375, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 4, 5]], [['cz', 2, 1], ['sq', 5, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 6, 5]], [['cz', 1, 0], ['sq', 5, 0.10897644578775623, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.41666666666666674, -1.0, -0.5], ['cz', 4, 5]], [['cz', 2, 1], ['sq', 4, 0.5, 0.0, -1.0], ['sq', 5, 0.3910235542122438, -1.0, -0.5]], [['sq', 1, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 4, 8], ['cz', 5, 6]], [['cz', 0, 1], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.08333333333333326, 0.0, 0.5], ['cz', 4, 8], ['cz', 6, 7]], [['cz', 2, 1], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 6, 0.4262081911747835, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, 0.0, -1.0]], [['sq', 1, 0.41666666666666674, -1.0, -0.5], ['sq', 2, 0.5, 0.0, 0.5], ['cz', 5, 6]], [['cz', 1, 0], ['cz', 2, 3], ['sq', 6, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.2988554271499826, 0.0, -0.5], ['sq', 3, 0.8883116146610162, 0.0, 0.5], ['cz', 7, 6]], [['cz', 2, 3], ['sq', 6, 0.07379180882521652, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 1.0, 0.0, 0.5], ['cz', 5, 6]], [['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.4262081911747835, -1.0, -0.5]], [['sq', 1, 0.43222219976770426, 0.0, 0.5], ['sq', 2, 0.789232625404525, 0.0, 0.5], ['cz', 4, 5], ['cz', 6, 7]], [['cz', 2, 3], ['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 3, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 8, 4]], [['cz', 1, 2], ['sq', 4, 0.125, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.2011445728500172, 0.0, 0.5], ['cz', 5, 4]], [['cz', 2, 3], ['sq', 4, 0.125, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.16666666666666674, 0.0, 0.5], ['cz', 8, 4]], [['cz', 1, 2], ['sq', 4, 0.375, -1.0, -0.5], ['sq', 8, 0.5, 0.0, 0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['cz', 4, 5], ['sq', 2, 0.49999999999999956, 0.3333333333333328, -1.1102230246251565e-16]], [['sq', 5, 0.5, -1.0, -0.5], ['cz', 2, 3]], [['cz', 5, 6], ['sq', 2, 0.4999999999999998, -1.0, 0.9999999999999998], ['sq', 3, 0.0, -0.4999999999999991, 0.9999999999999993]], [['sq', 5, 0.41666666666666674, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['cz', 1, 2]], [['cz', 4, 5], ['cz', 1, 0]], [['sq', 5, 0.08333333333333348, 0.0, -0.5000000000000002], ['sq', 1, 0.5, -1.0, -0.5]], [['cz', 6, 5]], [['sq', 5, 0.08333333333333326, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 4, 5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.41666666666666674, -1.0, -0.5]], [['cz', 4, 8], ['cz', 5, 6]], [['sq', 4, 0.2988554271499826, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.8883116146610162, 0.0, 0.5]], [['cz', 4, 8], ['cz', 6, 7]], [['sq', 4, 1.0, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 5, 4]], [['sq', 4, 0.789232625404525, 0.0, 0.5], ['sq', 5, 0.43222219976770426, 0.0, 0.5]], [['cz', 4, 8]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 8, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 5, 4]], [['sq', 4, 0.2011445728500172, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5]], [['cz', 4, 8]], [['sq', 4, 0.5, 0.0, -0.5], ['sq', 8, 0.16666666666666674, 0.0, 0.5]], [['cz', 5, 4]], [['sq', 4, 0.33333333333333326, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5]], [['cz', 4, 8], ['cz', 5, 6]], [['sq', 4, 0.0, -1.0, -0.5], ['sq', 8, 0.0, -1.0, -0.5]], [['cz', 5, 4]], [['cz', 5, 1]], [['sq', 5, 0.5, -1.0, -0.5]], [['cz', 9, 5]], [['sq', 5, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 5, 1], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 5, 4]], [['cz', 1, 0], ['sq', 4, 0.5, -1.0, -0.5], ['cz', 5, 6]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 6, 7]], [['cz', 2, 3], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (5, 3), (5, 4), (5, 5), (5, 6), (6, 3), (6, 6)], 'moments': [[['sq', 0, 0.8723282232918721, 0.0, 0.5], ['sq', 1, 0.3050177709578634, -1.0, -0.5], ['sq', 2, 0.17004943144282958, -1.0, -0.5], ['sq', 3, 0.070024348046551, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.14758361765043326, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 4], ['sq', 1, 0.3050177709578634, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 4, 0.5, -1.0, -0.5]], [['cz', 1, 5], ['sq', 2, 0.17004943144282958, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 2, 6], ['sq', 3, 0.070024348046551, -1.0, -0.5], ['cz', 5, 4]], [['sq', 2, 0.5, 0.0, -1.0], ['cz', 3, 7], ['sq', 4, 1.0, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5]], [['sq', 3, 0.5, 0.0, 0.0], ['cz', 4, 8]], [['cz', 2, 3], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 8, 0.14758361765043326, 0.0, 0.5]], [['sq', 2, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 5, 4]], [['cz', 2, 3], ['sq', 4, 0.28204710842448755, 0.0, -0.5], ['sq', 5, 0.7820471084244875, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, 0.0], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 4, 8]], [['cz', 2, 1], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 8, 0.5, 0.0, 0.0]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.375, -1.0, -0.5], ['cz', 5, 4]], [['cz', 3, 2], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5]], [['sq', 2, 0.125, 0.0, -0.5000000000000002], ['cz', 5, 6]], [['cz', 1, 2], ['sq', 5, 0.3910235542122438, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.125, 0.0, 0.5], ['cz', 4, 5]], [['cz', 3, 2], ['sq', 5, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 2, 0.375, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 6, 5]], [['cz', 2, 1], ['sq', 5, 0.10897644578775623, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 4, 5]], [['cz', 1, 0], ['sq', 4, 0.5, 0.0, -1.0], ['sq', 5, 0.3910235542122438, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.41666666666666674, -1.0, -0.5], ['cz', 4, 8], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 1, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 4, 8], ['cz', 6, 7]], [['cz', 0, 1], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 6, 0.4262081911747835, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, 0.0, -1.0]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.08333333333333326, 0.0, 0.5], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 6, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 1, 0.41666666666666674, -1.0, -0.5], ['sq', 2, 0.5, 0.0, 0.5], ['cz', 7, 6]], [['cz', 1, 0], ['cz', 2, 3], ['sq', 6, 0.07379180882521652, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.2988554271499826, 0.0, -0.5], ['sq', 3, 0.8883116146610162, 0.0, 0.5], ['cz', 5, 6]], [['cz', 2, 3], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.4262081911747835, -1.0, -0.5]], [['sq', 2, 1.0, 0.0, 0.5], ['cz', 4, 5], ['cz', 6, 7]], [['cz', 1, 2], ['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['cz', 9, 7]], [['sq', 1, 0.43222219976770426, 0.0, 0.5], ['sq', 2, 0.789232625404525, 0.0, 0.5], ['cz', 8, 4], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 2, 3], ['sq', 4, 0.125, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 3, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 5, 4]], [['cz', 1, 2], ['sq', 4, 0.125, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.2011445728500172, 0.0, 0.5], ['cz', 8, 4]], [['cz', 2, 3], ['sq', 4, 0.375, -1.0, -0.5], ['sq', 8, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.16666666666666674, 0.0, 0.5], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 2, 0.33333333333333326, 0.0, 0.5], ['cz', 5, 6], ['sq', 1, 1.0, 0.0, -1.0]], [['cz', 2, 3], ['sq', 5, 0.41666666666666674, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 4, 5], ['sq', 2, 0.0, -1.0, -0.5], ['sq', 3, 0.0, -1.0, -0.5]], [['sq', 5, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 7, 3]], [['cz', 6, 5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 5, 0.08333333333333326, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 4, 5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.41666666666666674, -1.0, -0.5]], [['cz', 4, 8], ['cz', 5, 6]], [['sq', 4, 0.2988554271499826, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.8883116146610162, 0.0, 0.5]], [['cz', 4, 8], ['cz', 6, 7]], [['sq', 4, 1.0, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 5, 4], ['cz', 7, 6]], [['sq', 4, 0.789232625404525, 0.0, 0.5], ['sq', 5, 0.43222219976770426, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 4, 8], ['cz', 6, 2]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 5, 4]], [['sq', 4, 0.2011445728500172, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5]], [['cz', 4, 8]], [['sq', 4, 0.5, 0.0, -0.5], ['sq', 8, 0.16666666666666674, 0.0, 0.5]], [['cz', 5, 4]], [['sq', 4, 0.33333333333333326, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5]], [['cz', 4, 8], ['cz', 5, 6]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.0, -1.0, -0.5]], [['cz', 6, 5]], [['sq', 5, 0.5, -1.0, -0.5]], [['cz', 5, 1]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 4, 5]], [['sq', 4, 0.5, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 5, 4]], [['sq', 4, 0.5, -1.0, -0.5]], [['cz', 4, 0]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 4, 8]], [['sq', 4, 1.0, 0.0, -1.0], ['sq', 8, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (5, 3), (5, 4), (5, 5), (5, 6), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.8723282232918721, 0.0, 0.5], ['sq', 1, 0.3050177709578634, -1.0, -0.5], ['sq', 2, 0.17004943144282958, -1.0, -0.5], ['sq', 3, 0.070024348046551, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.8524163823495667, 0.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 4], ['sq', 1, 0.3050177709578634, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 4, 0.5, -1.0, -0.5]], [['cz', 1, 5], ['sq', 2, 0.17004943144282958, -1.0, -0.5], ['cz', 4, 8]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 1.0, 0.0, -1.0], ['sq', 8, 0.8524163823495667, 0.0, 0.5]], [['cz', 2, 6], ['sq', 3, 0.070024348046551, -1.0, -0.5], ['cz', 5, 4]], [['sq', 2, 0.5, 0.0, -1.0], ['cz', 3, 7], ['sq', 4, 0.7179528915755125, 0.0, 0.5], ['sq', 5, 0.7820471084244875, 0.0, 0.5]], [['sq', 3, 0.5, 0.0, 0.0], ['cz', 4, 8]], [['cz', 2, 3], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, 0.0]], [['sq', 2, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 5, 4]], [['cz', 2, 3], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, 0.0], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 5, 0.3910235542122438, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.375, -1.0, -0.5], ['cz', 4, 5]], [['cz', 3, 2], ['sq', 5, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 2, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 5]], [['cz', 1, 2], ['sq', 5, 0.10897644578775623, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.125, 0.0, 0.5], ['cz', 4, 5]], [['cz', 3, 2], ['sq', 4, 0.5, 0.0, -1.0], ['sq', 5, 0.3910235542122438, -1.0, -0.5]], [['sq', 2, 0.375, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 4, 8], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 4, 8], ['cz', 6, 7]], [['cz', 1, 0], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 6, 0.4262081911747835, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, 0.0, -1.0]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.41666666666666674, -1.0, -0.5], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 6, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 1, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 7, 6]], [['cz', 0, 1], ['sq', 6, 0.07379180882521652, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.08333333333333326, 0.0, 0.5], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.4262081911747835, -1.0, -0.5]], [['sq', 1, 0.41666666666666674, -1.0, -0.5], ['sq', 2, 0.5, 0.0, 0.5], ['cz', 4, 5], ['cz', 6, 7]], [['cz', 1, 0], ['cz', 2, 3], ['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.2988554271499826, 0.0, -0.5], ['sq', 3, 0.8883116146610162, 0.0, 0.5], ['cz', 8, 4]], [['cz', 2, 3], ['sq', 4, 0.125, 0.0, -0.5000000000000002]], [['sq', 2, 1.0, 0.0, 0.5], ['cz', 5, 4]], [['cz', 1, 2], ['sq', 4, 0.125, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.43222219976770426, 0.0, 0.5], ['sq', 2, 0.789232625404525, 0.0, 0.5], ['cz', 8, 4]], [['cz', 2, 3], ['sq', 4, 0.375, -1.0, -0.5], ['sq', 8, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 3, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.2011445728500172, 0.0, 0.5], ['cz', 5, 6]], [['cz', 2, 3], ['sq', 5, 0.41666666666666674, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.16666666666666674, 0.0, 0.5], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 5, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, 0.0, -0.5], ['cz', 6, 5], ['sq', 2, 0.49999999999999956, 0.3333333333333328, -1.1102230246251565e-16]], [['sq', 5, 0.08333333333333326, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['cz', 2, 3]], [['cz', 4, 5], ['sq', 2, 0.4999999999999998, -1.0, 0.9999999999999998], ['sq', 3, 0.0, -0.4999999999999991, 0.9999999999999993]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.41666666666666674, -1.0, -0.5], ['cz', 1, 2]], [['cz', 4, 8], ['cz', 5, 6], ['cz', 1, 0]], [['sq', 4, 0.2988554271499826, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.8883116146610162, 0.0, 0.5], ['sq', 1, 0.5, -1.0, -0.5]], [['cz', 4, 8], ['cz', 6, 7]], [['sq', 4, 1.0, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 5, 4]], [['sq', 4, 0.789232625404525, 0.0, 0.5], ['sq', 5, 0.43222219976770426, 0.0, 0.5]], [['cz', 4, 8]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 8, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 5, 4]], [['sq', 4, 0.2011445728500172, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5]], [['cz', 4, 8]], [['sq', 4, 0.5, 0.0, -0.5], ['sq', 8, 0.16666666666666674, 0.0, 0.5]], [['cz', 5, 4]], [['sq', 4, 0.33333333333333326, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5]], [['cz', 4, 8], ['cz', 5, 6]], [['sq', 4, 0.0, -1.0, -0.5], ['sq', 8, 0.0, -1.0, -0.5]], [['cz', 5, 4]], [['cz', 5, 1]], [['sq', 5, 0.5, -1.0, -0.5]], [['cz', 9, 5]], [['sq', 5, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 5, 1], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 5, 4]], [['cz', 1, 0], ['sq', 4, 0.5, -1.0, -0.5], ['cz', 5, 6]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 6, 7]], [['cz', 2, 3], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (5, 3), (5, 4), (5, 5), (5, 6), (6, 3), (6, 6)], 'moments': [[['sq', 0, 0.8723282232918721, 0.0, 0.5], ['sq', 1, 0.3050177709578634, -1.0, -0.5], ['sq', 2, 0.17004943144282958, -1.0, -0.5], ['sq', 3, 0.070024348046551, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.8524163823495667, 0.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 4], ['sq', 1, 0.3050177709578634, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 4, 0.5, -1.0, -0.5]], [['cz', 1, 5], ['sq', 2, 0.17004943144282958, -1.0, -0.5], ['cz', 4, 8]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 1.0, 0.0, -1.0], ['sq', 8, 0.8524163823495667, 0.0, 0.5]], [['cz', 2, 6], ['sq', 3, 0.070024348046551, -1.0, -0.5], ['cz', 5, 4]], [['sq', 2, 0.5, 0.0, -1.0], ['cz', 3, 7], ['sq', 4, 0.7179528915755125, 0.0, 0.5], ['sq', 5, 0.7820471084244875, 0.0, 0.5]], [['sq', 3, 0.5, 0.0, 0.0], ['cz', 4, 8]], [['cz', 2, 3], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, 0.0]], [['sq', 2, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 5, 4]], [['cz', 2, 3], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, 0.0], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 5, 0.3910235542122438, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.375, -1.0, -0.5], ['cz', 4, 5]], [['cz', 3, 2], ['sq', 5, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 2, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 5]], [['cz', 1, 2], ['sq', 5, 0.10897644578775623, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.125, 0.0, 0.5], ['cz', 4, 5]], [['cz', 3, 2], ['sq', 4, 0.5, 0.0, -1.0], ['sq', 5, 0.3910235542122438, -1.0, -0.5]], [['sq', 2, 0.375, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 4, 8], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 4, 8], ['cz', 6, 7]], [['cz', 1, 0], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 6, 0.4262081911747835, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, 0.0, -1.0]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.41666666666666674, -1.0, -0.5], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 6, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 1, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 7, 6]], [['cz', 0, 1], ['sq', 6, 0.07379180882521652, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.08333333333333326, 0.0, 0.5], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.4262081911747835, -1.0, -0.5]], [['sq', 1, 0.41666666666666674, -1.0, -0.5], ['sq', 2, 0.5, 0.0, 0.5], ['cz', 4, 5], ['cz', 6, 7]], [['cz', 1, 0], ['cz', 2, 3], ['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['cz', 9, 7]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.2988554271499826, 0.0, -0.5], ['sq', 3, 0.8883116146610162, 0.0, 0.5], ['cz', 8, 4], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 2, 3], ['sq', 4, 0.125, 0.0, -0.5000000000000002]], [['sq', 2, 1.0, 0.0, 0.5], ['cz', 5, 4]], [['cz', 1, 2], ['sq', 4, 0.125, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.43222219976770426, 0.0, 0.5], ['sq', 2, 0.789232625404525, 0.0, 0.5], ['cz', 8, 4]], [['cz', 2, 3], ['sq', 4, 0.375, -1.0, -0.5], ['sq', 8, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 3, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.2011445728500172, 0.0, 0.5], ['cz', 5, 6]], [['cz', 2, 3], ['sq', 5, 0.41666666666666674, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.16666666666666674, 0.0, 0.5], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 5, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 2, 0.33333333333333326, 0.0, 0.5], ['cz', 6, 5], ['sq', 1, 1.0, 0.0, -1.0]], [['cz', 2, 3], ['sq', 5, 0.08333333333333326, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 4, 5], ['sq', 2, 0.0, -1.0, -0.5], ['sq', 3, 0.0, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.41666666666666674, -1.0, -0.5], ['cz', 7, 3]], [['cz', 4, 8], ['cz', 5, 6], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 4, 0.2988554271499826, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.8883116146610162, 0.0, 0.5]], [['cz', 4, 8], ['cz', 6, 7]], [['sq', 4, 1.0, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 5, 4], ['cz', 7, 6]], [['sq', 4, 0.789232625404525, 0.0, 0.5], ['sq', 5, 0.43222219976770426, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 4, 8], ['cz', 6, 2]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 5, 4]], [['sq', 4, 0.2011445728500172, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5]], [['cz', 4, 8]], [['sq', 4, 0.5, 0.0, -0.5], ['sq', 8, 0.16666666666666674, 0.0, 0.5]], [['cz', 5, 4]], [['sq', 4, 0.33333333333333326, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5]], [['cz', 4, 8], ['cz', 5, 6]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.0, -1.0, -0.5]], [['cz', 6, 5]], [['sq', 5, 0.5, -1.0, -0.5]], [['cz', 5, 1]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 4, 5]], [['sq', 4, 0.5, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 5, 4]], [['sq', 4, 0.5, -1.0, -0.5]], [['cz', 4, 0]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 4, 8]], [['sq', 4, 1.0, 0.0, -1.0], ['sq', 8, 0.5, -1.0, -0.5]]]}], 12: [{'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.9272924851969222, 0.0, 0.5], ['sq', 1, 0.3670787763403043, -1.0, -0.5], ['sq', 2, 0.24205984594556806, -1.0, -0.5], ['sq', 3, 0.1358157355171723, -1.0, -0.5], ['sq', 4, 0.057482058975516725, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 5], ['sq', 1, 0.3670787763403043, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 5, 0.0, -1.0, -0.5]], [['cz', 1, 6], ['sq', 2, 0.24205984594556806, -1.0, -0.5], ['cz', 5, 10]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 5, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 10, 0.13386023640061495, 0.0, 0.5]], [['cz', 2, 7], ['sq', 3, 0.1358157355171723, -1.0, -0.5], ['cz', 5, 10]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['cz', 3, 8], ['sq', 4, 0.0574820589755165, -1.0, -0.5], ['cz', 5, 6]], [['sq', 3, 0.5, 0.0, -0.5], ['cz', 4, 9], ['sq', 5, 0.34795663800765175, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 10, 5]], [['cz', 3, 4], ['sq', 5, 0.15204336199234825, 0.0, -0.5000000000000002]], [['sq', 3, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 4, 0.14758361765043326, 0.0, 0.5], ['cz', 6, 5]], [['cz', 3, 4], ['sq', 5, 0.15204336199234825, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 10, 5]], [['cz', 3, 2], ['sq', 5, 0.34795663800765175, -1.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3589764457877562, -1.0, -0.5], ['cz', 5, 6]], [['cz', 4, 3], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 3, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 2, 3], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.14102355421224377, 0.0, 0.5], ['cz', 5, 6]], [['cz', 4, 3], ['sq', 6, 0.125, 0.0, -0.5000000000000002]], [['sq', 3, 0.3589764457877562, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['cz', 7, 6]], [['cz', 3, 2], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.375, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.3910235542122438, -1.0, -0.5], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 5, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.14758361765043326, 0.0, 0.5]], [['sq', 2, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 5, 10], ['cz', 7, 8]], [['cz', 1, 2], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 7, 0.40204336199234825, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.10897644578775623, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 7, 0.09795663800765175, 0.0, -0.5000000000000002]], [['sq', 2, 0.3910235542122438, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 8, 7]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 7, 0.09795663800765197, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 6, 7]], [['cz', 1, 0], ['cz', 3, 4], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.40204336199234825, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.0], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 2, 1], ['sq', 5, 0.3589764457877562, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 1, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 8, 9], ['cz', 10, 5]], [['cz', 0, 1], ['sq', 5, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 8, 0.4330698817996925, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.07379180882521652, 0.0, 0.5], ['cz', 6, 5], ['cz', 7, 8]], [['cz', 2, 1], ['sq', 5, 0.14102355421224377, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 1, 0.4262081911747835, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['cz', 9, 8], ['cz', 10, 5]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 5, 0.3589764457877562, -1.0, -0.5], ['sq', 8, 0.06693011820030748, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.0]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4330698817996925, -1.0, -0.5]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.3910235542122438, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 5, 6]], [['cz', 4, 3], ['sq', 6, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 7, 6]], [['cz', 3, 2], ['sq', 6, 0.10897644578775623, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 5, 0.5, 0.0, -1.0], ['sq', 6, 0.3910235542122438, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.41666666666666674, -1.0, -0.5], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 2, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 5, 10], ['cz', 7, 8]], [['cz', 1, 2], ['sq', 5, 0.5, 0.0, 0.0], ['sq', 7, 0.4262081911747835, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.08333333333333326, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 7, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 2, 0.41666666666666674, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 8, 7]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 7, 0.07379180882521652, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.2988554271499826, 0.0, -0.5], ['sq', 4, 0.8883116146610162, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.4262081911747835, -1.0, -0.5]], [['sq', 3, 1.0, 0.0, 0.5], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 2, 3], ['sq', 5, 0.375, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.43222219976770426, 0.0, 0.5], ['sq', 3, 0.789232625404525, 0.0, 0.5], ['cz', 8, 9], ['cz', 10, 5]], [['cz', 3, 4], ['sq', 5, 0.125, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 6, 5]], [['cz', 2, 3], ['sq', 5, 0.125, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.2011445728500172, 0.0, 0.5], ['cz', 10, 5]], [['cz', 3, 4], ['sq', 5, 0.375, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.16666666666666674, 0.0, 0.5], ['cz', 5, 6]], [['cz', 2, 3], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['cz', 6, 7], ['sq', 3, 0.49999999999999956, 0.3333333333333328, -1.1102230246251565e-16]], [['sq', 6, 0.41666666666666674, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['cz', 3, 4]], [['cz', 5, 6], ['sq', 3, 0.4999999999999998, -1.0, 0.9999999999999998], ['sq', 4, 0.0, -0.4999999999999991, 0.9999999999999993]], [['sq', 6, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 2, 3]], [['cz', 7, 6], ['sq', 2, 0.5, -1.0, -0.5]], [['sq', 6, 0.08333333333333326, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['cz', 1, 2]], [['cz', 5, 6], ['cz', 1, 0]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 0.41666666666666674, -1.0, -0.5], ['sq', 1, 0.5, -1.0, -0.5]], [['cz', 5, 10], ['cz', 6, 7]], [['sq', 5, 0.2988554271499826, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.8883116146610162, 0.0, 0.5]], [['cz', 5, 10], ['cz', 7, 8]], [['sq', 5, 1.0, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 6, 5]], [['sq', 5, 0.789232625404525, 0.0, 0.5], ['sq', 6, 0.43222219976770426, 0.0, 0.5]], [['cz', 5, 10]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 10, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 6, 5]], [['sq', 5, 0.2011445728500172, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5]], [['cz', 5, 10]], [['sq', 5, 0.5, 0.0, -0.5], ['sq', 10, 0.16666666666666674, 0.0, 0.5]], [['cz', 6, 5]], [['sq', 5, 0.33333333333333326, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5]], [['cz', 5, 10], ['cz', 6, 7]], [['sq', 5, 0.0, -1.0, -0.5], ['sq', 10, 0.0, -1.0, -0.5]], [['cz', 6, 5]], [['cz', 6, 1]], [['sq', 6, 0.5, -1.0, -0.5]], [['cz', 11, 6]], [['sq', 6, 0.5, -1.0, -0.5], ['cz', 11, 10]], [['cz', 6, 1], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 11, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 6, 5]], [['cz', 1, 0], ['sq', 5, 0.5, -1.0, -0.5], ['cz', 6, 7]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 7, 8]], [['cz', 2, 3], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 8, 9]], [['cz', 3, 4], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 3), (6, 7)], 'moments': [[['sq', 0, 0.9272924851969222, 0.0, 0.5], ['sq', 1, 0.3670787763403043, -1.0, -0.5], ['sq', 2, 0.24205984594556806, -1.0, -0.5], ['sq', 3, 0.1358157355171723, -1.0, -0.5], ['sq', 4, 0.057482058975516725, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.13386023640061495, 0.0, 0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 5], ['sq', 1, 0.3670787763403043, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 1, 6], ['sq', 2, 0.24205984594556806, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 2, 7], ['sq', 3, 0.1358157355171723, -1.0, -0.5], ['cz', 6, 5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 5, 1.0, 0.0, 0.5], ['sq', 6, 0.5, 0.0, 0.5]], [['cz', 3, 8], ['sq', 4, 0.0574820589755165, -1.0, -0.5], ['cz', 5, 10]], [['sq', 3, 0.5, 0.0, -0.5], ['cz', 4, 9], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 10, 0.13386023640061495, 0.0, 0.5]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 6, 5]], [['cz', 3, 4], ['sq', 5, 0.3040867239846965, 0.0, -0.5], ['sq', 6, 0.8040867239846965, 0.0, 0.5]], [['sq', 3, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 4, 0.14758361765043326, 0.0, 0.5], ['cz', 5, 10]], [['cz', 3, 4], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 10, 0.5, 0.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 6, 5]], [['cz', 3, 2], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3589764457877562, -1.0, -0.5], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 5, 6]], [['cz', 2, 3], ['sq', 6, 0.125, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.14102355421224377, 0.0, 0.5], ['cz', 7, 6]], [['cz', 4, 3], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.3589764457877562, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['cz', 5, 6]], [['cz', 3, 2], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.375, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 2, 1], ['sq', 5, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.14758361765043326, 0.0, 0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.3910235542122438, -1.0, -0.5], ['cz', 5, 10], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 7, 0.40204336199234825, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['sq', 2, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 1, 2], ['sq', 7, 0.09795663800765175, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.10897644578775623, 0.0, 0.5], ['cz', 8, 7]], [['cz', 3, 2], ['sq', 7, 0.09795663800765197, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.3910235542122438, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 6, 7]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.40204336199234825, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 1, 0], ['cz', 3, 4], ['sq', 5, 0.3589764457877562, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.0], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 8, 9], ['cz', 10, 5]], [['cz', 2, 1], ['sq', 5, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 8, 0.4330698817996925, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 1, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 6, 5], ['cz', 7, 8]], [['cz', 0, 1], ['sq', 5, 0.14102355421224377, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.07379180882521652, 0.0, 0.5], ['cz', 9, 8], ['cz', 10, 5]], [['cz', 2, 1], ['sq', 5, 0.3589764457877562, -1.0, -0.5], ['sq', 8, 0.06693011820030748, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.0]], [['sq', 1, 0.4262081911747835, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4330698817996925, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 4, 3], ['sq', 6, 0.3910235542122438, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['cz', 11, 9]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 5, 6], ['sq', 9, 0.5, -1.0, -0.5]], [['cz', 2, 3], ['sq', 6, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 7, 6]], [['cz', 4, 3], ['sq', 6, 0.10897644578775623, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 5, 6]], [['cz', 3, 2], ['sq', 5, 0.5, 0.0, -1.0], ['sq', 6, 0.3910235542122438, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 2, 1], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.41666666666666674, -1.0, -0.5], ['cz', 5, 10], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.5, 0.0, 0.0], ['sq', 7, 0.4262081911747835, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, -1.0]], [['sq', 2, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 1, 2], ['sq', 7, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.08333333333333326, 0.0, 0.5], ['cz', 8, 7]], [['cz', 3, 2], ['sq', 7, 0.07379180882521652, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.41666666666666674, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 6, 7]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.4262081911747835, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.2988554271499826, 0.0, -0.5], ['sq', 4, 0.8883116146610162, 0.0, 0.5], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 3, 4], ['sq', 5, 0.375, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 1.0, 0.0, 0.5], ['cz', 10, 5]], [['cz', 2, 3], ['sq', 5, 0.125, 0.0, -0.5000000000000002]], [['sq', 2, 0.43222219976770426, 0.0, 0.5], ['sq', 3, 0.789232625404525, 0.0, 0.5], ['cz', 6, 5]], [['cz', 3, 4], ['sq', 5, 0.125, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 10, 5]], [['cz', 2, 3], ['sq', 5, 0.375, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.2011445728500172, 0.0, 0.5], ['cz', 5, 6]], [['cz', 3, 4], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.16666666666666674, 0.0, 0.5], ['cz', 6, 7]], [['cz', 2, 3], ['sq', 6, 0.41666666666666674, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.33333333333333326, 0.0, 0.5], ['cz', 5, 6], ['sq', 2, 1.0, 0.0, -1.0]], [['cz', 3, 4], ['sq', 6, 0.08333333333333348, 0.0, -0.5000000000000002]], [['cz', 7, 6], ['sq', 3, 0.0, -1.0, -0.5], ['sq', 4, 0.0, -1.0, -0.5]], [['sq', 6, 0.08333333333333326, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['cz', 9, 4]], [['cz', 5, 6], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 0.41666666666666674, -1.0, -0.5], ['cz', 8, 9]], [['cz', 5, 10], ['cz', 6, 7], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 5, 0.2988554271499826, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.8883116146610162, 0.0, 0.5], ['cz', 9, 8]], [['cz', 5, 10], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 5, 1.0, 0.0, 0.5], ['cz', 8, 3]], [['cz', 6, 5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 5, 0.789232625404525, 0.0, 0.5], ['sq', 6, 0.43222219976770426, 0.0, 0.5], ['cz', 7, 8]], [['cz', 5, 10], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 10, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 8, 7]], [['cz', 6, 5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 5, 0.2011445728500172, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5], ['cz', 7, 2]], [['cz', 5, 10], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 5, 0.5, 0.0, -0.5], ['sq', 10, 0.16666666666666674, 0.0, 0.5]], [['cz', 6, 5]], [['sq', 5, 0.33333333333333326, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5]], [['cz', 5, 10], ['cz', 6, 7]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.0, -1.0, -0.5]], [['cz', 7, 6]], [['sq', 6, 0.5, -1.0, -0.5]], [['cz', 6, 1]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 5, 6]], [['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 6, 5]], [['sq', 5, 0.5, -1.0, -0.5]], [['cz', 5, 0]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 5, 10]], [['sq', 5, 1.0, 0.0, -1.0], ['sq', 10, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.9272924851969222, 0.0, 0.5], ['sq', 1, 0.3670787763403043, -1.0, -0.5], ['sq', 2, 0.24205984594556806, -1.0, -0.5], ['sq', 3, 0.1358157355171723, -1.0, -0.5], ['sq', 4, 0.057482058975516725, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.866139763599385, 0.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 5], ['sq', 1, 0.3670787763403043, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 1, 6], ['sq', 2, 0.24205984594556806, -1.0, -0.5], ['cz', 5, 10]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 1.0, 0.0, -1.0], ['sq', 10, 0.866139763599385, 0.0, 0.5]], [['cz', 2, 7], ['sq', 3, 0.1358157355171723, -1.0, -0.5], ['cz', 6, 5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 5, 0.6959132760153035, 0.0, 0.5], ['sq', 6, 0.8040867239846965, 0.0, 0.5]], [['cz', 3, 8], ['sq', 4, 0.0574820589755165, -1.0, -0.5], ['cz', 5, 10]], [['sq', 3, 0.5, 0.0, -0.5], ['cz', 4, 9], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 6, 5]], [['cz', 3, 4], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 0.5, 0.0, 0.5]], [['sq', 3, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 4, 0.14758361765043326, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 5, 6]], [['cz', 3, 2], ['sq', 6, 0.125, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3589764457877562, -1.0, -0.5], ['cz', 7, 6]], [['cz', 4, 3], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 5, 6]], [['cz', 2, 3], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.375, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.14102355421224377, 0.0, 0.5], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 5, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.14758361765043326, 0.0, 0.5]], [['sq', 3, 0.3589764457877562, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['cz', 5, 10], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 7, 0.40204336199234825, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 6, 7]], [['cz', 2, 1], ['sq', 7, 0.09795663800765175, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.3910235542122438, -1.0, -0.5], ['cz', 8, 7]], [['cz', 3, 2], ['sq', 7, 0.09795663800765197, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.40204336199234825, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.10897644578775623, 0.0, 0.5], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.3589764457877562, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.3910235542122438, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 8, 9], ['cz', 10, 5]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 5, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 8, 0.4330698817996925, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 6, 5], ['cz', 7, 8]], [['cz', 1, 0], ['cz', 3, 4], ['sq', 5, 0.14102355421224377, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.0], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 9, 8], ['cz', 10, 5]], [['cz', 2, 1], ['sq', 5, 0.3589764457877562, -1.0, -0.5], ['sq', 8, 0.06693011820030748, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.0]], [['sq', 1, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 0, 1], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4330698817996925, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.07379180882521652, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 1], ['sq', 6, 0.3910235542122438, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 1, 0.4262081911747835, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['cz', 5, 6]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 6, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 7, 6]], [['cz', 4, 3], ['sq', 6, 0.10897644578775623, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 5, 6]], [['cz', 2, 3], ['sq', 5, 0.5, 0.0, -1.0], ['sq', 6, 0.3910235542122438, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 5, 10], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.5, 0.0, 0.0], ['sq', 7, 0.4262081911747835, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, -1.0]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 6, 7]], [['cz', 2, 1], ['sq', 7, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.41666666666666674, -1.0, -0.5], ['cz', 8, 7]], [['cz', 3, 2], ['sq', 7, 0.07379180882521652, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.4262081911747835, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.08333333333333326, 0.0, 0.5], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.375, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.41666666666666674, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 8, 9], ['cz', 10, 5]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 5, 0.125, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.2988554271499826, 0.0, -0.5], ['sq', 4, 0.8883116146610162, 0.0, 0.5], ['cz', 6, 5]], [['cz', 3, 4], ['sq', 5, 0.125, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 3, 1.0, 0.0, 0.5], ['cz', 10, 5]], [['cz', 2, 3], ['sq', 5, 0.375, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['sq', 2, 0.43222219976770426, 0.0, 0.5], ['sq', 3, 0.789232625404525, 0.0, 0.5], ['cz', 5, 6]], [['cz', 3, 4], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 2, 3], ['sq', 6, 0.41666666666666674, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.2011445728500172, 0.0, 0.5], ['cz', 5, 6]], [['cz', 3, 4], ['sq', 6, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.16666666666666674, 0.0, 0.5], ['cz', 7, 6]], [['cz', 2, 3], ['sq', 6, 0.08333333333333326, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['cz', 5, 6], ['sq', 3, 0.49999999999999956, 0.3333333333333328, -1.1102230246251565e-16]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 0.41666666666666674, -1.0, -0.5], ['cz', 3, 4]], [['cz', 5, 10], ['cz', 6, 7], ['sq', 3, 0.4999999999999998, -1.0, 0.9999999999999998], ['sq', 4, 0.0, -0.4999999999999991, 0.9999999999999993]], [['sq', 5, 0.2988554271499826, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.8883116146610162, 0.0, 0.5], ['cz', 2, 3]], [['cz', 5, 10], ['cz', 7, 8], ['sq', 2, 0.5, -1.0, -0.5]], [['sq', 5, 1.0, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['cz', 1, 2]], [['cz', 6, 5], ['cz', 1, 0]], [['sq', 5, 0.789232625404525, 0.0, 0.5], ['sq', 6, 0.43222219976770426, 0.0, 0.5], ['sq', 1, 0.5, -1.0, -0.5]], [['cz', 5, 10]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 10, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 6, 5]], [['sq', 5, 0.2011445728500172, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5]], [['cz', 5, 10]], [['sq', 5, 0.5, 0.0, -0.5], ['sq', 10, 0.16666666666666674, 0.0, 0.5]], [['cz', 6, 5]], [['sq', 5, 0.33333333333333326, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5]], [['cz', 5, 10], ['cz', 6, 7]], [['sq', 5, 0.0, -1.0, -0.5], ['sq', 10, 0.0, -1.0, -0.5]], [['cz', 6, 5]], [['cz', 6, 1]], [['sq', 6, 0.5, -1.0, -0.5]], [['cz', 11, 6]], [['sq', 6, 0.5, -1.0, -0.5], ['cz', 11, 10]], [['cz', 6, 1], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 11, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 6, 5]], [['cz', 1, 0], ['sq', 5, 0.5, -1.0, -0.5], ['cz', 6, 7]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 7, 8]], [['cz', 2, 3], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 8, 9]], [['cz', 3, 4], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 3), (6, 7)], 'moments': [[['sq', 0, 0.9272924851969222, 0.0, 0.5], ['sq', 1, 0.3670787763403043, -1.0, -0.5], ['sq', 2, 0.24205984594556806, -1.0, -0.5], ['sq', 3, 0.1358157355171723, -1.0, -0.5], ['sq', 4, 0.057482058975516725, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.866139763599385, 0.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 5], ['sq', 1, 0.3670787763403043, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 1, 6], ['sq', 2, 0.24205984594556806, -1.0, -0.5], ['cz', 5, 10]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 1.0, 0.0, -1.0], ['sq', 10, 0.866139763599385, 0.0, 0.5]], [['cz', 2, 7], ['sq', 3, 0.1358157355171723, -1.0, -0.5], ['cz', 6, 5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 5, 0.6959132760153035, 0.0, 0.5], ['sq', 6, 0.8040867239846965, 0.0, 0.5]], [['cz', 3, 8], ['sq', 4, 0.0574820589755165, -1.0, -0.5], ['cz', 5, 10]], [['sq', 3, 0.5, 0.0, -0.5], ['cz', 4, 9], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 6, 5]], [['cz', 3, 4], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 0.5, 0.0, 0.5]], [['sq', 3, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 4, 0.14758361765043326, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 5, 6]], [['cz', 3, 2], ['sq', 6, 0.125, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3589764457877562, -1.0, -0.5], ['cz', 7, 6]], [['cz', 4, 3], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 5, 6]], [['cz', 2, 3], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.375, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.14102355421224377, 0.0, 0.5], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 5, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.14758361765043326, 0.0, 0.5]], [['sq', 3, 0.3589764457877562, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['cz', 5, 10], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 7, 0.40204336199234825, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 6, 7]], [['cz', 2, 1], ['sq', 7, 0.09795663800765175, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.3910235542122438, -1.0, -0.5], ['cz', 8, 7]], [['cz', 3, 2], ['sq', 7, 0.09795663800765197, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.40204336199234825, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.10897644578775623, 0.0, 0.5], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.3589764457877562, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.3910235542122438, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 8, 9], ['cz', 10, 5]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 5, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 8, 0.4330698817996925, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 6, 5], ['cz', 7, 8]], [['cz', 1, 0], ['cz', 3, 4], ['sq', 5, 0.14102355421224377, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.0], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 9, 8], ['cz', 10, 5]], [['cz', 2, 1], ['sq', 5, 0.3589764457877562, -1.0, -0.5], ['sq', 8, 0.06693011820030748, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.0]], [['sq', 1, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 0, 1], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4330698817996925, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.07379180882521652, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 1], ['sq', 6, 0.3910235542122438, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['cz', 11, 9]], [['sq', 1, 0.4262081911747835, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['cz', 5, 6], ['sq', 9, 0.5, -1.0, -0.5]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 6, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 7, 6]], [['cz', 4, 3], ['sq', 6, 0.10897644578775623, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 5, 6]], [['cz', 2, 3], ['sq', 5, 0.5, 0.0, -1.0], ['sq', 6, 0.3910235542122438, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 5, 10], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.5, 0.0, 0.0], ['sq', 7, 0.4262081911747835, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, -1.0]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 6, 7]], [['cz', 2, 1], ['sq', 7, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.41666666666666674, -1.0, -0.5], ['cz', 8, 7]], [['cz', 3, 2], ['sq', 7, 0.07379180882521652, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.4262081911747835, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.08333333333333326, 0.0, 0.5], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.375, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.41666666666666674, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 10, 5]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 5, 0.125, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.2988554271499826, 0.0, -0.5], ['sq', 4, 0.8883116146610162, 0.0, 0.5], ['cz', 6, 5]], [['cz', 3, 4], ['sq', 5, 0.125, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 3, 1.0, 0.0, 0.5], ['cz', 10, 5]], [['cz', 2, 3], ['sq', 5, 0.375, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['sq', 2, 0.43222219976770426, 0.0, 0.5], ['sq', 3, 0.789232625404525, 0.0, 0.5], ['cz', 5, 6]], [['cz', 3, 4], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 2, 3], ['sq', 6, 0.41666666666666674, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.2011445728500172, 0.0, 0.5], ['cz', 5, 6]], [['cz', 3, 4], ['sq', 6, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.16666666666666674, 0.0, 0.5], ['cz', 7, 6]], [['cz', 2, 3], ['sq', 6, 0.08333333333333326, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.33333333333333326, 0.0, 0.5], ['cz', 5, 6], ['sq', 2, 1.0, 0.0, -1.0]], [['cz', 3, 4], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 0.41666666666666674, -1.0, -0.5]], [['cz', 5, 10], ['cz', 6, 7], ['sq', 3, 0.0, -1.0, -0.5], ['sq', 4, 0.0, -1.0, -0.5]], [['sq', 5, 0.2988554271499826, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.8883116146610162, 0.0, 0.5], ['cz', 9, 4]], [['cz', 5, 10], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 5, 1.0, 0.0, 0.5], ['cz', 8, 9]], [['cz', 6, 5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 5, 0.789232625404525, 0.0, 0.5], ['sq', 6, 0.43222219976770426, 0.0, 0.5], ['cz', 9, 8]], [['cz', 5, 10], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 10, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 8, 3]], [['cz', 6, 5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 5, 0.2011445728500172, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5], ['cz', 7, 8]], [['cz', 5, 10], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 5, 0.5, 0.0, -0.5], ['sq', 10, 0.16666666666666674, 0.0, 0.5], ['cz', 8, 7]], [['cz', 6, 5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 5, 0.33333333333333326, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5], ['cz', 7, 2]], [['cz', 5, 10], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 10, 0.0, -1.0, -0.5], ['cz', 6, 7]], [['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 7, 6]], [['sq', 6, 0.5, -1.0, -0.5]], [['cz', 6, 1]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 5, 6]], [['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 6, 5]], [['sq', 5, 0.5, -1.0, -0.5]], [['cz', 5, 0]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 5, 10]], [['sq', 5, 1.0, 0.0, -1.0], ['sq', 10, 0.5, -1.0, -0.5]]]}], 14: [{'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.9593120124343613, 0.0, 0.5], ['sq', 1, 0.4124711229199527, -1.0, -0.5], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5], ['sq', 11, 0.5, 0.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5], ['sq', 13, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 6], ['sq', 1, 0.4124711229199527, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 6, 0.0, -1.0, -0.5]], [['cz', 1, 7], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['cz', 6, 12]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 6, 0.12337585721442501, 0.0, -0.5], ['sq', 12, 0.12337585721442501, 0.0, 0.5]], [['cz', 2, 8], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['cz', 6, 12]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['cz', 3, 9], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['cz', 6, 7]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 4, 5], ['sq', 6, 0.33975425899284417, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 4, 10], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['cz', 12, 6]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 5, 11], ['sq', 6, 0.16024574100715605, 0.0, -0.5000000000000002]], [['sq', 5, 0.5, 0.0, -0.5], ['cz', 7, 6]], [['cz', 4, 5], ['sq', 6, 0.16024574100715605, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 4, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 5, 0.13386023640061495, 0.0, 0.5], ['cz', 12, 6]], [['cz', 4, 5], ['sq', 6, 0.33975425899284417, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.34795663800765175, -1.0, -0.5], ['cz', 7, 8]], [['cz', 5, 4], ['sq', 7, 0.36359276291425235, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.15204336199234825, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 7, 0.13640723708574742, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.15204336199234825, 0.0, 0.5], ['cz', 8, 7]], [['cz', 5, 4], ['sq', 7, 0.13640723708574742, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.34795663800765175, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.3635927629142526, -1.0, -0.5000000000000002]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 6, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.13386023640061495, 0.0, 0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 4, 3], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.38640723708574765, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 7, 8]], [['cz', 2, 3], ['sq', 8, 0.11359276291425258, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 9, 8]], [['cz', 4, 3], ['sq', 8, 0.11359276291425258, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['cz', 7, 8]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.3864072370857474, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 5, 0.14758361765043326, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 3, 2], ['sq', 6, 0.15204336199234825, 0.0, -0.5000000000000002], ['sq', 9, 0.41024574100715605, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 2, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 1, 2], ['sq', 6, 0.15204336199234825, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.09795663800765197, 0.0, 0.5], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 3, 2], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.41024574100715605, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 1, 0], ['cz', 5, 4], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 10, 11]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 7, 0.125, 0.0, -0.5000000000000002], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 1, 0.06693011820030748, 0.0, -0.5000000000000002], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, 0.5], ['cz', 8, 7], ['cz', 9, 10]], [['cz', 0, 1], ['cz', 5, 4], ['sq', 7, 0.125, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, -0.5000000000000002]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.06693011820030748, 0.0, 0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.0], ['cz', 6, 7], ['cz', 11, 10]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, 0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 12], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 6, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 12, 0.14758361765043326, 0.0, 0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3910235542122438, -1.0, -0.5], ['cz', 6, 12], ['cz', 8, 9], ['cz', 10, 11]], [['cz', 4, 3], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 3, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 7, 8]], [['cz', 2, 3], ['sq', 8, 0.09795663800765175, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.10897644578775623, 0.0, 0.5], ['cz', 9, 8]], [['cz', 4, 3], ['sq', 8, 0.09795663800765197, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.3910235542122438, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 7, 8]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 5, 0.5, 0.0, -1.0], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 3, 2], ['sq', 6, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 9, 0.4330698817996925, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 2, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 1, 2], ['sq', 6, 0.14102355421224377, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.07379180882521652, 0.0, 0.5], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 3, 2], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.0]], [['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.4330698817996925, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.375, -1.0, -0.5], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 5, 4], ['sq', 7, 0.3910235542122438, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 4, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 10, 11]], [['cz', 3, 4], ['sq', 7, 0.10897644578775623, 0.0, -0.5000000000000002], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.125, 0.0, 0.5], ['cz', 8, 7]], [['cz', 5, 4], ['sq', 7, 0.10897644578775623, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 6, 0.5, 0.0, -1.0], ['sq', 7, 0.3910235542122438, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 6, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.41666666666666674, -1.0, -0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 4, 3], ['sq', 6, 0.5, 0.0, 0.0], ['sq', 8, 0.4262081911747835, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -1.0]], [['sq', 3, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 7, 8]], [['cz', 2, 3], ['sq', 8, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.08333333333333326, 0.0, 0.5], ['cz', 9, 8]], [['cz', 4, 3], ['sq', 8, 0.07379180882521652, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.41666666666666674, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 7, 8]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4262081911747835, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.2988554271499826, 0.0, -0.5], ['sq', 5, 0.8883116146610162, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 4, 5], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 4, 1.0, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 3, 4], ['sq', 6, 0.125, 0.0, -0.5000000000000002], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.43222219976770426, 0.0, 0.5], ['sq', 4, 0.789232625404525, 0.0, 0.5], ['cz', 7, 6]], [['cz', 4, 5], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 12, 6]], [['cz', 3, 4], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.2011445728500172, 0.0, 0.5], ['cz', 6, 7]], [['cz', 4, 5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.16666666666666674, 0.0, 0.5], ['cz', 7, 8]], [['cz', 3, 4], ['sq', 7, 0.41666666666666674, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, -0.5], ['cz', 6, 7], ['sq', 4, 0.49999999999999956, 0.3333333333333328, -1.1102230246251565e-16]], [['sq', 7, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 4, 5]], [['cz', 8, 7], ['sq', 4, 0.4999999999999998, -1.0, 0.9999999999999998], ['sq', 5, 0.0, -0.4999999999999991, 0.9999999999999993]], [['sq', 7, 0.08333333333333326, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5], ['cz', 3, 4]], [['cz', 6, 7], ['sq', 3, 0.5, -1.0, -0.5]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.41666666666666674, -1.0, -0.5], ['cz', 2, 3]], [['cz', 6, 12], ['cz', 7, 8], ['sq', 2, 0.5, -1.0, -0.5]], [['sq', 6, 0.2988554271499826, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.8883116146610162, 0.0, 0.5], ['cz', 1, 2]], [['cz', 6, 12], ['cz', 8, 9], ['cz', 1, 0]], [['sq', 6, 1.0, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 1, 0.5, -1.0, -0.5]], [['cz', 7, 6]], [['sq', 6, 0.789232625404525, 0.0, 0.5], ['sq', 7, 0.43222219976770426, 0.0, 0.5]], [['cz', 6, 12]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 12, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 7, 6]], [['sq', 6, 0.2011445728500172, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5]], [['cz', 6, 12]], [['sq', 6, 0.5, 0.0, -0.5], ['sq', 12, 0.16666666666666674, 0.0, 0.5]], [['cz', 7, 6]], [['sq', 6, 0.33333333333333326, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5]], [['cz', 6, 12], ['cz', 7, 8]], [['sq', 6, 0.0, -1.0, -0.5], ['sq', 12, 0.0, -1.0, -0.5]], [['cz', 7, 6]], [['cz', 7, 1]], [['sq', 7, 0.5, -1.0, -0.5]], [['cz', 13, 7]], [['sq', 7, 0.5, -1.0, -0.5], ['cz', 13, 12]], [['cz', 7, 1], ['sq', 12, 0.5, -1.0, -0.5], ['sq', 13, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 7, 6]], [['cz', 1, 0], ['sq', 6, 0.5, -1.0, -0.5], ['cz', 7, 8]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 10]], [['cz', 3, 4], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, -1.0, -0.5], ['cz', 10, 11]], [['cz', 4, 5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 5, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 3), (6, 8)], 'moments': [[['sq', 0, 0.9593120124343613, 0.0, 0.5], ['sq', 1, 0.4124711229199527, -1.0, -0.5], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5], ['sq', 11, 0.5, 0.0, -0.5], ['sq', 12, 0.12337585721442501, 0.0, 0.5], ['sq', 13, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 6], ['sq', 1, 0.4124711229199527, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 1, 7], ['sq', 2, 0.3046558254995755, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 2, 8], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['cz', 7, 6]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 6, 1.0, 0.0, 0.5], ['sq', 7, 0.5, 0.0, 0.5]], [['cz', 3, 9], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['cz', 6, 12]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 4, 5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 12, 0.12337585721442501, 0.0, 0.5]], [['cz', 4, 10], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['cz', 7, 6]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 5, 11], ['sq', 6, 0.3204914820143121, 0.0, -0.5], ['sq', 7, 0.8204914820143121, 0.0, 0.5]], [['sq', 5, 0.5, 0.0, -0.5], ['cz', 6, 12]], [['cz', 4, 5], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 4, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 5, 0.13386023640061495, 0.0, 0.5], ['cz', 7, 6]], [['cz', 4, 5], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.5, 0.0, 0.5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.36359276291425235, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.34795663800765175, -1.0, -0.5], ['cz', 6, 7]], [['cz', 5, 4], ['sq', 7, 0.13640723708574742, 0.0, -0.5000000000000002]], [['sq', 4, 0.15204336199234825, 0.0, -0.5000000000000002], ['cz', 8, 7]], [['cz', 3, 4], ['sq', 7, 0.13640723708574742, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.15204336199234825, 0.0, 0.5], ['cz', 6, 7]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.3635927629142526, -1.0, -0.5000000000000002]], [['sq', 4, 0.34795663800765175, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 6, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.13386023640061495, 0.0, 0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.38640723708574765, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 8, 0.11359276291425258, 0.0, -0.5000000000000002]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 9, 8]], [['cz', 2, 3], ['sq', 8, 0.11359276291425258, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.3864072370857474, -1.0, -0.5]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 5, 0.14758361765043326, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.15204336199234825, 0.0, -0.5000000000000002], ['sq', 9, 0.41024574100715605, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 6, 0.15204336199234825, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, -0.5000000000000002]], [['sq', 2, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 1, 2], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.09795663800765197, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.41024574100715605, -1.0, -0.5]], [['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['cz', 6, 7], ['cz', 10, 11]], [['cz', 1, 0], ['cz', 5, 4], ['sq', 7, 0.125, 0.0, -0.5000000000000002], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 8, 7], ['cz', 9, 10]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 7, 0.125, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, -0.5000000000000002]], [['sq', 1, 0.06693011820030748, 0.0, -0.5000000000000002], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, 0.5], ['cz', 6, 7], ['cz', 11, 10]], [['cz', 0, 1], ['cz', 5, 4], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, 0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.06693011820030748, 0.0, 0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.0], ['cz', 6, 12], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 6, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 12, 0.14758361765043326, 0.0, 0.5]], [['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 12], ['cz', 8, 9], ['cz', 10, 11]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5], ['cz', 13, 11]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3910235542122438, -1.0, -0.5], ['cz', 7, 8], ['sq', 11, 0.5, -1.0, -0.5]], [['cz', 4, 3], ['sq', 8, 0.09795663800765175, 0.0, -0.5000000000000002]], [['sq', 3, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 9, 8]], [['cz', 2, 3], ['sq', 8, 0.09795663800765197, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.10897644578775623, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5]], [['sq', 3, 0.3910235542122438, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 9, 0.4330698817996925, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 5, 0.5, 0.0, -1.0], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 6, 0.14102355421224377, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 2, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 1, 2], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.0]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.07379180882521652, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.4330698817996925, -1.0, -0.5]], [['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.3910235542122438, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.375, -1.0, -0.5], ['cz', 6, 7]], [['cz', 5, 4], ['sq', 7, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 4, 0.125, 0.0, -0.5000000000000002], ['cz', 8, 7]], [['cz', 3, 4], ['sq', 7, 0.10897644578775623, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.125, 0.0, 0.5], ['cz', 6, 7]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, -1.0], ['sq', 7, 0.3910235542122438, -1.0, -0.5]], [['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 6, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 6, 0.5, 0.0, 0.0], ['sq', 8, 0.4262081911747835, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -1.0]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.41666666666666674, -1.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 8, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 3, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 9, 8]], [['cz', 2, 3], ['sq', 8, 0.07379180882521652, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.08333333333333326, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4262081911747835, -1.0, -0.5]], [['sq', 3, 0.41666666666666674, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.2988554271499826, 0.0, -0.5], ['sq', 5, 0.8883116146610162, 0.0, 0.5], ['cz', 12, 6]], [['cz', 4, 5], ['sq', 6, 0.125, 0.0, -0.5000000000000002]], [['sq', 4, 1.0, 0.0, 0.5], ['cz', 7, 6]], [['cz', 3, 4], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.43222219976770426, 0.0, 0.5], ['sq', 4, 0.789232625404525, 0.0, 0.5], ['cz', 12, 6]], [['cz', 4, 5], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.2011445728500172, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 5], ['sq', 7, 0.41666666666666674, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.16666666666666674, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 7, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 4, 0.33333333333333326, 0.0, 0.5], ['cz', 8, 7], ['sq', 3, 1.0, 0.0, -1.0]], [['cz', 4, 5], ['sq', 7, 0.08333333333333326, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['cz', 6, 7], ['sq', 4, 0.0, -1.0, -0.5], ['sq', 5, 0.0, -1.0, -0.5]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.41666666666666674, -1.0, -0.5], ['cz', 11, 5]], [['cz', 6, 12], ['cz', 7, 8], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 6, 0.2988554271499826, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.8883116146610162, 0.0, 0.5], ['cz', 10, 11]], [['cz', 6, 12], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 6, 1.0, 0.0, 0.5], ['cz', 11, 10]], [['cz', 7, 6], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 6, 0.789232625404525, 0.0, 0.5], ['sq', 7, 0.43222219976770426, 0.0, 0.5], ['cz', 10, 4]], [['cz', 6, 12], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 12, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 9, 10]], [['cz', 7, 6], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 6, 0.2011445728500172, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5], ['cz', 10, 9]], [['cz', 6, 12], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 6, 0.5, 0.0, -0.5], ['sq', 12, 0.16666666666666674, 0.0, 0.5], ['cz', 9, 3]], [['cz', 7, 6], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 6, 0.33333333333333326, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5], ['cz', 8, 9]], [['cz', 6, 12], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 12, 0.0, -1.0, -0.5], ['cz', 9, 8]], [['sq', 8, 0.5, -1.0, -0.5]], [['cz', 8, 2]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['cz', 7, 8]], [['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['cz', 8, 7]], [['sq', 7, 0.5, -1.0, -0.5]], [['cz', 7, 1]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 6, 7]], [['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 7, 6]], [['sq', 6, 0.5, -1.0, -0.5]], [['cz', 6, 0]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 6, 12]], [['sq', 6, 1.0, 0.0, -1.0], ['sq', 12, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.9593120124343613, 0.0, 0.5], ['sq', 1, 0.4124711229199527, -1.0, -0.5], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5], ['sq', 11, 0.5, 0.0, -0.5], ['sq', 12, 0.876624142785575, 0.0, -0.5], ['sq', 13, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 6], ['sq', 1, 0.4124711229199527, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 1, 7], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['cz', 6, 12]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 1.0, 0.0, -1.0], ['sq', 12, 0.876624142785575, 0.0, 0.5]], [['cz', 2, 8], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['cz', 7, 6]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 6, 0.6795085179856879, 0.0, 0.5], ['sq', 7, 0.8204914820143121, 0.0, 0.5]], [['cz', 3, 9], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['cz', 6, 12]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 4, 5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['cz', 4, 10], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['cz', 7, 6]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 5, 11], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.5, 0.0, 0.5]], [['sq', 5, 0.5, 0.0, -0.5], ['cz', 7, 8]], [['cz', 4, 5], ['sq', 7, 0.36359276291425235, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 5, 0.13386023640061495, 0.0, 0.5], ['cz', 6, 7]], [['cz', 4, 5], ['sq', 7, 0.13640723708574742, 0.0, -0.5000000000000002]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 8, 7]], [['cz', 4, 3], ['sq', 7, 0.13640723708574742, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.34795663800765175, -1.0, -0.5], ['cz', 6, 7]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.3635927629142526, -1.0, -0.5000000000000002]], [['sq', 4, 0.15204336199234825, 0.0, -0.5000000000000002], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 3, 4], ['sq', 6, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.13386023640061495, 0.0, 0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.15204336199234825, 0.0, 0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.38640723708574765, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 4, 0.34795663800765175, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 8, 0.11359276291425258, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 3, 2], ['sq', 8, 0.11359276291425258, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.3864072370857474, -1.0, -0.5]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 4, 3], ['sq', 6, 0.15204336199234825, 0.0, -0.5000000000000002], ['sq', 9, 0.41024574100715605, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.15204336199234825, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 5, 0.14758361765043326, 0.0, 0.5], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.41024574100715605, -1.0, -0.5]], [['sq', 2, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 1, 2], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.09795663800765197, 0.0, 0.5], ['cz', 6, 7], ['cz', 10, 11]], [['cz', 3, 2], ['sq', 7, 0.125, 0.0, -0.5000000000000002], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 8, 7], ['cz', 9, 10]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.125, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['cz', 6, 7], ['cz', 11, 10]], [['cz', 1, 0], ['cz', 5, 4], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, 0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 6, 12], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 6, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 12, 0.14758361765043326, 0.0, 0.5]], [['sq', 1, 0.06693011820030748, 0.0, -0.5000000000000002], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, 0.5], ['cz', 6, 12], ['cz', 8, 9], ['cz', 10, 11]], [['cz', 0, 1], ['cz', 5, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.06693011820030748, 0.0, 0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.0], ['cz', 7, 8]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 8, 0.09795663800765175, 0.0, -0.5000000000000002]], [['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 8, 0.09795663800765197, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3910235542122438, -1.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5]], [['sq', 3, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.10897644578775623, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 4, 3], ['sq', 6, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 9, 0.4330698817996925, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 3, 0.3910235542122438, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.14102355421224377, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.0]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 5, 0.5, 0.0, -1.0], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.4330698817996925, -1.0, -0.5]], [['sq', 2, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 1, 2], ['sq', 7, 0.3910235542122438, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.07379180882521652, 0.0, 0.5], ['cz', 6, 7], ['cz', 10, 11]], [['cz', 3, 2], ['sq', 7, 0.10897644578775623, 0.0, -0.5000000000000002], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 8, 7]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.10897644578775623, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.375, -1.0, -0.5], ['cz', 6, 7]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, -1.0], ['sq', 7, 0.3910235542122438, -1.0, -0.5]], [['sq', 4, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 3, 4], ['sq', 6, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.125, 0.0, 0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, 0.0], ['sq', 8, 0.4262081911747835, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -1.0]], [['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 8, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 3, 2], ['sq', 8, 0.07379180882521652, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.41666666666666674, -1.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4262081911747835, -1.0, -0.5]], [['sq', 3, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.08333333333333326, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 4, 3], ['sq', 6, 0.125, 0.0, -0.5000000000000002], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.41666666666666674, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 7, 6]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.2988554271499826, 0.0, -0.5], ['sq', 5, 0.8883116146610162, 0.0, 0.5], ['cz', 12, 6]], [['cz', 4, 5], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 4, 1.0, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.43222219976770426, 0.0, 0.5], ['sq', 4, 0.789232625404525, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 5], ['sq', 7, 0.41666666666666674, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 7, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.2011445728500172, 0.0, 0.5], ['cz', 8, 7]], [['cz', 4, 5], ['sq', 7, 0.08333333333333326, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.16666666666666674, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.41666666666666674, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, -0.5], ['cz', 6, 12], ['cz', 7, 8], ['sq', 4, 0.49999999999999956, 0.3333333333333328, -1.1102230246251565e-16]], [['sq', 6, 0.2988554271499826, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.8883116146610162, 0.0, 0.5], ['cz', 4, 5]], [['cz', 6, 12], ['cz', 8, 9], ['sq', 4, 0.4999999999999998, -1.0, 0.9999999999999998], ['sq', 5, 0.0, -0.4999999999999991, 0.9999999999999993]], [['sq', 6, 1.0, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5], ['cz', 3, 4]], [['cz', 7, 6], ['sq', 3, 0.5, -1.0, -0.5]], [['sq', 6, 0.789232625404525, 0.0, 0.5], ['sq', 7, 0.43222219976770426, 0.0, 0.5], ['cz', 2, 3]], [['cz', 6, 12], ['sq', 2, 0.5, -1.0, -0.5]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 12, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 1, 2]], [['cz', 7, 6], ['cz', 1, 0]], [['sq', 6, 0.2011445728500172, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 1, 0.5, -1.0, -0.5]], [['cz', 6, 12]], [['sq', 6, 0.5, 0.0, -0.5], ['sq', 12, 0.16666666666666674, 0.0, 0.5]], [['cz', 7, 6]], [['sq', 6, 0.33333333333333326, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5]], [['cz', 6, 12], ['cz', 7, 8]], [['sq', 6, 0.0, -1.0, -0.5], ['sq', 12, 0.0, -1.0, -0.5]], [['cz', 7, 6]], [['cz', 7, 1]], [['sq', 7, 0.5, -1.0, -0.5]], [['cz', 13, 7]], [['sq', 7, 0.5, -1.0, -0.5], ['cz', 13, 12]], [['cz', 7, 1], ['sq', 12, 0.5, -1.0, -0.5], ['sq', 13, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 7, 6]], [['cz', 1, 0], ['sq', 6, 0.5, -1.0, -0.5], ['cz', 7, 8]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 10]], [['cz', 3, 4], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, -1.0, -0.5], ['cz', 10, 11]], [['cz', 4, 5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 5, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 3), (6, 8)], 'moments': [[['sq', 0, 0.9593120124343613, 0.0, 0.5], ['sq', 1, 0.4124711229199527, -1.0, -0.5], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5], ['sq', 11, 0.5, 0.0, -0.5], ['sq', 12, 0.876624142785575, 0.0, -0.5], ['sq', 13, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 6], ['sq', 1, 0.4124711229199527, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 1, 7], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['cz', 6, 12]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 1.0, 0.0, -1.0], ['sq', 12, 0.876624142785575, 0.0, 0.5]], [['cz', 2, 8], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['cz', 7, 6]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 6, 0.6795085179856879, 0.0, 0.5], ['sq', 7, 0.8204914820143121, 0.0, 0.5]], [['cz', 3, 9], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['cz', 6, 12]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 4, 5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['cz', 4, 10], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['cz', 7, 6]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 5, 11], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.5, 0.0, 0.5]], [['sq', 5, 0.5, 0.0, -0.5], ['cz', 7, 8]], [['cz', 4, 5], ['sq', 7, 0.36359276291425235, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 5, 0.13386023640061495, 0.0, 0.5], ['cz', 6, 7]], [['cz', 4, 5], ['sq', 7, 0.13640723708574742, 0.0, -0.5000000000000002]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 8, 7]], [['cz', 4, 3], ['sq', 7, 0.13640723708574742, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.34795663800765175, -1.0, -0.5], ['cz', 6, 7]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.3635927629142526, -1.0, -0.5000000000000002]], [['sq', 4, 0.15204336199234825, 0.0, -0.5000000000000002], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 3, 4], ['sq', 6, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.13386023640061495, 0.0, 0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.15204336199234825, 0.0, 0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.38640723708574765, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 4, 0.34795663800765175, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 8, 0.11359276291425258, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 3, 2], ['sq', 8, 0.11359276291425258, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.3864072370857474, -1.0, -0.5]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 4, 3], ['sq', 6, 0.15204336199234825, 0.0, -0.5000000000000002], ['sq', 9, 0.41024574100715605, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.15204336199234825, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 5, 0.14758361765043326, 0.0, 0.5], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.41024574100715605, -1.0, -0.5]], [['sq', 2, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 1, 2], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.09795663800765197, 0.0, 0.5], ['cz', 6, 7], ['cz', 10, 11]], [['cz', 3, 2], ['sq', 7, 0.125, 0.0, -0.5000000000000002], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 8, 7], ['cz', 9, 10]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.125, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['cz', 6, 7], ['cz', 11, 10]], [['cz', 1, 0], ['cz', 5, 4], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, 0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 6, 12], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 6, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 12, 0.14758361765043326, 0.0, 0.5]], [['sq', 1, 0.06693011820030748, 0.0, -0.5000000000000002], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, 0.5], ['cz', 6, 12], ['cz', 8, 9], ['cz', 10, 11]], [['cz', 0, 1], ['cz', 5, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5], ['cz', 13, 11]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.06693011820030748, 0.0, 0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.0], ['cz', 7, 8], ['sq', 11, 0.5, -1.0, -0.5]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 8, 0.09795663800765175, 0.0, -0.5000000000000002]], [['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 8, 0.09795663800765197, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3910235542122438, -1.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5]], [['sq', 3, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.10897644578775623, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 4, 3], ['sq', 6, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 9, 0.4330698817996925, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 3, 0.3910235542122438, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.14102355421224377, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.0]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 5, 0.5, 0.0, -1.0], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.4330698817996925, -1.0, -0.5]], [['sq', 2, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 1, 2], ['sq', 7, 0.3910235542122438, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.07379180882521652, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 7, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 8, 7]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.10897644578775623, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.375, -1.0, -0.5], ['cz', 6, 7]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, -1.0], ['sq', 7, 0.3910235542122438, -1.0, -0.5]], [['sq', 4, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 3, 4], ['sq', 6, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.125, 0.0, 0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, 0.0], ['sq', 8, 0.4262081911747835, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -1.0]], [['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 8, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 3, 2], ['sq', 8, 0.07379180882521652, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.41666666666666674, -1.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4262081911747835, -1.0, -0.5]], [['sq', 3, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.08333333333333326, 0.0, 0.5], ['cz', 12, 6]], [['cz', 4, 3], ['sq', 6, 0.125, 0.0, -0.5000000000000002]], [['sq', 3, 0.41666666666666674, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 7, 6]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.2988554271499826, 0.0, -0.5], ['sq', 5, 0.8883116146610162, 0.0, 0.5], ['cz', 12, 6]], [['cz', 4, 5], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 4, 1.0, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.43222219976770426, 0.0, 0.5], ['sq', 4, 0.789232625404525, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 5], ['sq', 7, 0.41666666666666674, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 7, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.2011445728500172, 0.0, 0.5], ['cz', 8, 7]], [['cz', 4, 5], ['sq', 7, 0.08333333333333326, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.16666666666666674, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.41666666666666674, -1.0, -0.5]], [['sq', 4, 0.33333333333333326, 0.0, 0.5], ['cz', 6, 12], ['cz', 7, 8], ['sq', 3, 1.0, 0.0, -1.0]], [['cz', 4, 5], ['sq', 6, 0.2988554271499826, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.8883116146610162, 0.0, 0.5]], [['cz', 6, 12], ['sq', 4, 0.0, -1.0, -0.5], ['sq', 5, 0.0, -1.0, -0.5]], [['sq', 6, 1.0, 0.0, 0.5], ['cz', 11, 5]], [['cz', 7, 6], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 6, 0.789232625404525, 0.0, 0.5], ['sq', 7, 0.43222219976770426, 0.0, 0.5], ['cz', 10, 11]], [['cz', 6, 12], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 12, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 11, 10]], [['cz', 7, 6], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 6, 0.2011445728500172, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5], ['cz', 10, 4]], [['cz', 6, 12], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 6, 0.5, 0.0, -0.5], ['sq', 12, 0.16666666666666674, 0.0, 0.5], ['cz', 9, 10]], [['cz', 7, 6], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 6, 0.33333333333333326, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5], ['cz', 10, 9]], [['cz', 6, 12], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 12, 0.0, -1.0, -0.5], ['cz', 9, 3]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['cz', 8, 9]], [['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['cz', 9, 8]], [['sq', 8, 0.5, -1.0, -0.5]], [['cz', 8, 2]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['cz', 7, 8]], [['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['cz', 8, 7]], [['sq', 7, 0.5, -1.0, -0.5]], [['cz', 7, 1]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 6, 7]], [['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 7, 6]], [['sq', 6, 0.5, -1.0, -0.5]], [['cz', 6, 0]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 6, 12]], [['sq', 6, 1.0, 0.0, -1.0], ['sq', 12, 0.5, -1.0, -0.5]]]}, {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 3), (6, 8)], 'moments': [[['sq', 0, 0.9593120124343613, 0.0, 0.5], ['sq', 1, 0.4124711229199527, -1.0, -0.5], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5], ['sq', 11, 0.5, 0.0, -0.5], ['sq', 12, 0.876624142785575, 0.0, -0.5], ['sq', 13, 0.5, 0.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 6], ['sq', 1, 0.4124711229199527, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 1, 7], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['cz', 6, 12]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 1.0, 0.0, -1.0], ['sq', 12, 0.876624142785575, 0.0, 0.5]], [['cz', 2, 8], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['cz', 7, 6]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 6, 0.6795085179856879, 0.0, 0.5], ['sq', 7, 0.8204914820143121, 0.0, 0.5]], [['cz', 3, 9], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['cz', 6, 12]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 4, 5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['cz', 4, 10], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['cz', 7, 6]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 5, 11], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.5, 0.0, 0.5]], [['sq', 5, 0.5, 0.0, -0.5], ['cz', 7, 8]], [['cz', 4, 5], ['sq', 7, 0.36359276291425235, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 5, 0.13386023640061495, 0.0, 0.5], ['cz', 6, 7]], [['cz', 4, 5], ['sq', 7, 0.13640723708574742, 0.0, -0.5000000000000002]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 8, 7]], [['cz', 4, 3], ['sq', 7, 0.13640723708574742, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.34795663800765175, -1.0, -0.5], ['cz', 6, 7]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.3635927629142526, -1.0, -0.5000000000000002]], [['sq', 4, 0.15204336199234825, 0.0, -0.5000000000000002], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 3, 4], ['sq', 6, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.13386023640061495, 0.0, 0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.15204336199234825, 0.0, 0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.38640723708574765, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 4, 0.34795663800765175, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 8, 0.11359276291425258, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 3, 2], ['sq', 8, 0.11359276291425258, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.3864072370857474, -1.0, -0.5]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 4, 3], ['sq', 6, 0.15204336199234825, 0.0, -0.5000000000000002], ['sq', 9, 0.41024574100715605, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.15204336199234825, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 5, 0.14758361765043326, 0.0, 0.5], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.41024574100715605, -1.0, -0.5]], [['sq', 2, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 1, 2], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.09795663800765197, 0.0, 0.5], ['cz', 6, 7], ['cz', 10, 11]], [['cz', 3, 2], ['sq', 7, 0.125, 0.0, -0.5000000000000002], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 8, 7], ['cz', 9, 10]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.125, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['cz', 6, 7], ['cz', 11, 10]], [['cz', 1, 0], ['cz', 5, 4], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, 0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 6, 12], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 6, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 12, 0.14758361765043326, 0.0, 0.5]], [['sq', 1, 0.06693011820030748, 0.0, -0.5000000000000002], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, 0.5], ['cz', 6, 12], ['cz', 8, 9], ['cz', 10, 11]], [['cz', 0, 1], ['cz', 5, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5], ['sq', 11, 0.0, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.06693011820030748, 0.0, 0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.0], ['cz', 7, 8]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 8, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 11, 13]], [['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 8], ['sq', 11, 0.5, 0.0, 0.5], ['sq', 13, 0.0, -1.0, -0.5]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 8, 0.09795663800765197, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3910235542122438, -1.0, -0.5], ['cz', 7, 8], ['sq', 0, 0.0, -1.0, -0.5]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5]], [['sq', 3, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.10897644578775623, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 4, 3], ['sq', 6, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 9, 0.4330698817996925, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 3, 0.3910235542122438, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.14102355421224377, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.0]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 5, 0.5, 0.0, -1.0], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 3, 2], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.4330698817996925, -1.0, -0.5]], [['sq', 2, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 1, 2], ['sq', 7, 0.3910235542122438, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -0.75, -0.7500000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.07379180882521652, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 7, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 8, 7]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.10897644578775623, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.375, -1.0, -0.5], ['cz', 6, 7], ['sq', 1, 0.0, -1.0, -0.5]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, -1.0], ['sq', 7, 0.3910235542122438, -1.0, -0.5]], [['sq', 4, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 3, 4], ['sq', 6, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.125, 0.0, 0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 5, 4], ['sq', 6, 0.5, 0.0, 0.0], ['sq', 8, 0.4262081911747835, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -1.0]], [['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 8, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 3, 2], ['sq', 8, 0.07379180882521652, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.41666666666666674, -1.0, -0.5], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4262081911747835, -1.0, -0.5]], [['sq', 3, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -0.75, -0.7500000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.08333333333333326, 0.0, 0.5], ['cz', 12, 6]], [['cz', 4, 3], ['sq', 6, 0.125, 0.0, -0.5000000000000002]], [['sq', 3, 0.41666666666666674, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 7, 6]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.2988554271499826, 0.0, -0.5], ['sq', 5, 0.8883116146610162, 0.0, 0.5], ['cz', 12, 6], ['sq', 2, 0.0, -1.0, -0.5]], [['cz', 4, 5], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 4, 1.0, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.43222219976770426, 0.0, 0.5], ['sq', 4, 0.789232625404525, 0.0, 0.5], ['cz', 7, 8]], [['cz', 4, 5], ['sq', 7, 0.41666666666666674, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.11168838533898384, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 7, 0.08333333333333348, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.2011445728500172, 0.0, 0.5], ['cz', 8, 7]], [['cz', 4, 5], ['sq', 7, 0.08333333333333326, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.16666666666666674, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.41666666666666674, -1.0, -0.5]], [['sq', 4, 0.33333333333333326, 0.0, 0.5], ['cz', 6, 12], ['cz', 7, 8], ['sq', 3, 1.0, 0.0, 0.5]], [['cz', 4, 5], ['sq', 6, 0.2988554271499826, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 12, 0.8883116146610162, 0.0, 0.5], ['sq', 8, 0.5, -0.75, -0.7500000000000002]], [['cz', 5, 11], ['cz', 6, 12]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 6, 1.0, 0.0, 0.5], ['sq', 11, 0.5, 0.0, -0.25]], [['cz', 7, 6], ['cz', 10, 11]], [['sq', 6, 0.789232625404525, 0.0, 0.5], ['sq', 7, 0.43222219976770426, 0.0, 0.5], ['sq', 10, 0.5, 0.0, -0.5], ['sq', 11, 0.5, 0.0, -0.25]], [['cz', 6, 12], ['cz', 10, 11]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 10, 0.5, -0.75, -0.5], ['sq', 11, 0.75, 0.75, -0.75], ['sq', 12, 0.11168838533898384, 0.0, -0.5000000000000002]], [['cz', 4, 10], ['cz', 7, 6]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 6, 0.2011445728500172, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.25]], [['cz', 6, 12], ['cz', 9, 10]], [['sq', 6, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.25], ['sq', 12, 0.16666666666666674, 0.0, 0.5]], [['cz', 7, 6], ['cz', 9, 10]], [['sq', 6, 0.33333333333333326, 0.0, 0.5], ['sq', 9, 0.5, -0.75, -0.5], ['sq', 10, 0.75, 0.75, -0.75], ['sq', 7, 0.5, -0.25, -0.25]], [['cz', 3, 9], ['cz', 6, 12]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 9, 0.5, 0.0, -0.25], ['sq', 6, 0.5, 0.25, 0.25]], [['cz', 8, 9]], [['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.25]], [['cz', 8, 9]], [['sq', 8, 0.5, -0.75, -0.5], ['sq', 9, 0.75, 0.75, -0.75]], [['cz', 2, 8]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 8, 0.5, 0.0, -0.25]], [['cz', 7, 8]], [['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.25]], [['cz', 7, 8]], [['sq', 7, 0.5, -0.75, -0.5], ['sq', 8, 0.75, 0.75, -0.75]], [['cz', 1, 7]], [['sq', 1, 0.5, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.25]], [['cz', 6, 7]], [['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.25]], [['cz', 6, 7]], [['sq', 6, 0.5, -0.75, -0.5], ['sq', 7, 0.75, 0.75, -0.75]], [['cz', 0, 6]], [['sq', 0, 0.5, 0.0, 0.5], ['cz', 6, 12]], [['sq', 6, 1.0, 0.0, -1.0], ['sq', 12, 0.5, 0.0, 0.5]]]}]}
W_TEMPLATES = {3: {'qubits': [(3, 4), (3, 5), (4, 4)], 'moments': [[['sq', 0, 0.6081734479693928, 0.0, 0.5], ['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.25, 0.0, -0.5]], [['cz', 1, 0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 0]], [['sq', 0, 0.5, 0.0, -0.5], ['sq', 2, 0.25, 0.0, 0.5]], [['cz', 2, 0]], [['sq', 0, 0.5, 0.0, 0.5]]]}, 4: {'qubits': [(3, 8), (3, 9), (4, 9), (5, 9)], 'moments': [[['sq', 1, 0.5000000000000001, 0.0, 0.5], ['sq', 2, 0.5, 1.0, -0.5]], [['cz', 1, 2]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 3, 0.25, 1.0, -0.5], ['sq', 0, 0.25, 1.0, -0.5]], [['cz', 3, 2], ['cz', 0, 1]], [['sq', 3, 0.25, 1.0, -0.5], ['sq', 2, 0.5, 1.0, -0.5], ['sq', 0, 0.25, 1.0, -0.5], ['sq', 1, 0.5, 1.0, -0.5]], [['cz', 3, 2], ['cz', 0, 1]], [['sq', 2, 0.5, 1.0, -0.5], ['sq', 1, 0.5, 1.0, -0.5]]]}, 5: {'qubits': [(5, 1), (5, 2), (6, 1), (6, 2), (6, 3)], 'moments': [[['sq', 0, 0.3040867239846965, 0.0, -0.5], ['sq', 1, 0.25, 0.0, -0.5], ['sq', 2, 0.5640942168489751, 0.0, 0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.25, 0.0, -0.5]], [['cz', 2, 3]], [['cz', 0, 2], ['sq', 3, 0.5, 0.0, -0.5]], [['sq', 0, 0.3040867239846965, 0.0, 0.5], ['sq', 2, 0.5, 0.0, -0.5], ['cz', 3, 4]], [['cz', 0, 2], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.25, 0.0, 0.5]], [['cz', 0, 1], ['sq', 2, 0.5, 0.0, 0.5], ['cz', 3, 4]], [['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.25, 0.0, 0.5], ['sq', 3, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['sq', 0, 0.5, 0.0, 0.5]]]}, 6: {'qubits': [(4, 10), (4, 11), (4, 12), (5, 10), (5, 11), (6, 11)], 'moments': [[['sq', 0, 0.5, 0.0, 0.5], ['sq', 1, 0.3040867239846965, 0.0, -0.5], ['sq', 2, 0.25, 0.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.3040867239846965, 0.0, -0.5], ['sq', 5, 0.25, 0.0, -0.5]], [['cz', 0, 3]], [['cz', 0, 1], ['sq', 3, 0.5, 0.0, -0.5]], [['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.3040867239846965, 0.0, 0.5], ['cz', 3, 4]], [['cz', 0, 1], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.3040867239846965, 0.0, 0.5]], [['sq', 0, 0.5, 0.0, 0.5], ['cz', 2, 1], ['cz', 3, 4]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.25, 0.0, 0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 4, 5]], [['cz', 2, 1], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.25, 0.0, 0.5]], [['sq', 1, 0.5, 0.0, 0.5], ['cz', 4, 5]], [['sq', 4, 0.5, 0.0, 0.5]]]}, 7: {'qubits': [(5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (7, 7)], 'moments': [[['sq', 0, 0.25, 0.0, -0.5], ['sq', 1, 0.25, 0.0, -0.5], ['sq', 2, 0.3040867239846965, 0.0, -0.5], ['sq', 3, 0.25, 0.0, -0.5], ['sq', 4, 0.5456289483429899, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.25, 0.0, -0.5]], [['cz', 4, 5]], [['cz', 1, 4], ['sq', 5, 0.5, 0.0, -0.5]], [['sq', 1, 0.25, 0.0, 0.5], ['cz', 2, 5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 1, 4], ['sq', 2, 0.3040867239846965, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5]], [['cz', 1, 0], ['cz', 2, 5], ['sq', 4, 0.5, 0.0, 0.5]], [['sq', 0, 0.25, 0.0, 0.5], ['sq', 1, 0.5, 0.0, -0.5], ['cz', 3, 2], ['cz', 4, 6], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 1, 0], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.25, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 6, 0.25, 0.0, 0.5]], [['sq', 1, 0.5, 0.0, 0.5], ['cz', 3, 2], ['cz', 4, 6]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5]]]}, 8: {'qubits': [(3, 9), (4, 9), (5, 7), (5, 8), (5, 9), (5, 10), (6, 8), (7, 8)], 'moments': [[['sq', 3, 0.5000000000000001, 0.0, 0.5], ['sq', 4, 0.5, 1.0, -0.5]], [['cz', 4, 3]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 6, 0.25, 1.0, -0.5], ['sq', 1, 0.25, 1.0, -0.5]], [['cz', 3, 6], ['cz', 4, 1]], [['sq', 6, 0.25, 1.0, -0.5], ['sq', 3, 0.5, 1.0, -0.5], ['sq', 1, 0.25, 1.0, -0.5], ['sq', 4, 0.5, 1.0, -0.5]], [['cz', 3, 6], ['cz', 4, 1]], [['sq', 3, 0.5, 1.0, -0.5], ['sq', 4, 0.5, 1.0, -0.5], ['sq', 5, 0.25, 1.0, -0.5], ['sq', 7, 0.25, 1.0, -0.5], ['sq', 2, 0.25, 1.0, -0.5], ['sq', 0, 0.25, 1.0, -0.5]], [['cz', 4, 5], ['cz', 6, 7], ['cz', 2, 3], ['cz', 0, 1]], [['sq', 5, 0.25, 1.0, -0.5], ['sq', 4, 0.5, 1.0, -0.5], ['sq', 7, 0.25, 1.0, -0.5], ['sq', 6, 0.5, 1.0, -0.5], ['sq', 2, 0.25, 1.0, -0.5], ['sq', 3, 0.5, 1.0, -0.5], ['sq', 0, 0.25, 1.0, -0.5], ['sq', 1, 0.5, 1.0, -0.5]], [['cz', 4, 5], ['cz', 6, 7], ['cz', 2, 3], ['cz', 0, 1]], [['sq', 4, 0.5, 1.0, -0.5], ['sq', 6, 0.5, 1.0, -0.5], ['sq', 3, 0.5, 1.0, -0.5], ['sq', 1, 0.5, 1.0, -0.5]]]}, 9: {'qubits': [(6, 8), (6, 10), (7, 8), (7, 9), (7, 10), (8, 8), (8, 9), (9, 8), (9, 9)], 'moments': [[['sq', 0, 0.25, 0.0, -0.5], ['sq', 1, 0.25, 0.0, -0.5], ['sq', 2, 0.25, 0.0, -0.5], ['sq', 3, 0.28204710842448755, 0.0, -0.5], ['sq', 4, 0.3040867239846965, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.53544094560246, 0.0, 0.5], ['sq', 7, 0.25, 0.0, -0.5], ['sq', 8, 0.25, 0.0, -0.5]], [['cz', 5, 6]], [['cz', 3, 6], ['sq', 5, 0.5, 0.0, -0.5]], [['sq', 3, 0.28204710842448755, 0.0, 0.5], ['cz', 5, 2], ['sq', 6, 0.5, 0.0, -0.5]], [['sq', 2, 0.25, 0.0, 0.5], ['cz', 3, 6], ['sq', 5, 0.5, 0.0, 0.5]], [['cz', 3, 4], ['cz', 5, 2], ['sq', 6, 0.5, 0.0, 0.5]], [['cz', 0, 2], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.3040867239846965, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5], ['cz', 6, 8]], [['sq', 0, 0.25, 0.0, 0.5], ['sq', 2, 0.5, 0.0, -0.5], ['cz', 3, 4], ['cz', 5, 7], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 8, 0.25, 0.0, 0.5]], [['cz', 0, 2], ['cz', 1, 4], ['sq', 3, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 6, 8], ['sq', 7, 0.25, 0.0, 0.5]], [['sq', 1, 0.25, 0.0, 0.5], ['sq', 2, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5], ['cz', 5, 7], ['sq', 6, 0.5, 0.0, 0.5]], [['cz', 1, 4], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, 0.5]]]}, 10: {'qubits': [(4, 5), (5, 3), (5, 4), (5, 5), (5, 6), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)], 'moments': [[['sq', 0, 0.25, 0.0, -0.5], ['sq', 1, 0.25, 0.0, -0.5], ['sq', 2, 0.25, 0.0, -0.5], ['sq', 3, 0.3040867239846965, 0.0, -0.5], ['sq', 4, 0.19591327601530373, 0.0, -0.5000000000000002], ['sq', 5, 0.25, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5640942168489751, 0.0, 0.5], ['sq', 8, 0.25, 0.0, -0.5], ['sq', 9, 0.25, 0.0, -0.5]], [['cz', 6, 7]], [['sq', 6, 0.5, 0.0, -0.5], ['cz', 8, 7]], [['cz', 2, 6], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.25, 0.0, 0.5]], [['sq', 2, 0.25, 0.0, 0.5], ['sq', 6, 0.5, 0.0, 0.5], ['cz', 8, 7]], [['cz', 2, 6], ['sq', 7, 0.5, 0.0, 0.5], ['cz', 8, 4]], [['cz', 2, 1], ['cz', 3, 7], ['sq', 4, 0.1959132760153035, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5]], [['sq', 1, 0.25, 0.0, 0.5], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.3040867239846965, 0.0, 0.5], ['cz', 6, 5], ['sq', 7, 0.5, 0.0, -0.5], ['cz', 8, 4]], [['cz', 2, 1], ['cz', 3, 7], ['sq', 5, 0.25, 0.0, 0.5], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.5, 0.0, 0.5]], [['cz', 0, 3], ['sq', 2, 0.5, 0.0, 0.5], ['cz', 6, 5], ['sq', 7, 0.5, 0.0, 0.5], ['cz', 8, 9]], [['sq', 0, 0.25, 0.0, 0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.25, 0.0, 0.5]], [['cz', 0, 3], ['cz', 8, 9]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 8, 0.5, 0.0, 0.5]]]}, 11: {'qubits': [(6, 7), (7, 6), (7, 7), (8, 5), (8, 6), (9, 5), (9, 6), (10, 6), (10, 7), (11, 6), (12, 6)], 'moments': [[['sq', 0, 0.25, 0.0, -0.5], ['sq', 1, 0.28204710842448755, 0.0, -0.5], ['sq', 2, 0.3040867239846965, 0.0, -0.5], ['sq', 3, 0.25, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.25, 0.0, -0.5], ['sq', 6, 0.5289772698358561, 0.0, 0.5], ['sq', 7, 0.3040867239846965, 0.0, -0.5], ['sq', 8, 0.25, 0.0, -0.5], ['sq', 9, 0.25, 0.0, -0.5], ['sq', 10, 0.25, 0.0, -0.5]], [['cz', 4, 6]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 6, 7]], [['cz', 1, 4], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.3040867239846965, 0.0, 0.5]], [['sq', 1, 0.28204710842448755, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 6, 7]], [['cz', 1, 4], ['sq', 6, 0.5, 0.0, 0.5], ['cz', 7, 9]], [['cz', 1, 2], ['sq', 4, 0.5, 0.0, -0.5], ['cz', 5, 6], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 9, 0.25, 0.0, 0.5]], [['sq', 1, 0.5, 0.0, -0.5], ['sq', 2, 0.3040867239846965, 0.0, 0.5], ['cz', 4, 3], ['sq', 5, 0.25, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5], ['cz', 7, 9]], [['cz', 1, 2], ['sq', 3, 0.25, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 5, 6], ['sq', 7, 0.5, 0.0, 0.5], ['cz', 9, 10]], [['cz', 0, 2], ['sq', 1, 0.5, 0.0, 0.5], ['cz', 4, 3], ['sq', 6, 0.5, 0.0, 0.5], ['cz', 7, 8], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.25, 0.0, 0.5]], [['sq', 0, 0.25, 0.0, 0.5], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.25, 0.0, 0.5], ['cz', 9, 10]], [['cz', 0, 2], ['cz', 7, 8], ['sq', 9, 0.5, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 7, 0.5, 0.0, 0.5]]]}, 12: {'qubits': [(8, 8), (8, 9), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (10, 7), (10, 8), (10, 9), (11, 7), (11, 9)], 'moments': [[['sq', 0, 0.25, 0.0, -0.5], ['sq', 1, 0.25, 0.0, -0.5], ['sq', 2, 0.25, 0.0, -0.5], ['sq', 3, 0.3040867239846965, 0.0, -0.5], ['sq', 4, 0.28204710842448755, 0.0, -0.5], ['sq', 5, 0.3040867239846965, 0.0, -0.5], ['sq', 6, 0.25, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5533003790381135, 0.0, 0.5], ['sq', 9, 0.27281447417149507, 0.0, -0.5], ['sq', 10, 0.25, 0.0, -0.5], ['sq', 11, 0.25, 0.0, -0.5]], [['cz', 8, 7]], [['sq', 7, 0.5, 0.0, -0.5], ['cz', 8, 9]], [['cz', 4, 7], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.27281447417149507, 0.0, 0.5]], [['sq', 4, 0.28204710842448755, 0.0, 0.5], ['sq', 7, 0.5, 0.0, 0.5], ['cz', 8, 9]], [['cz', 4, 7], ['cz', 6, 9], ['sq', 8, 0.5, 0.0, 0.5]], [['cz', 3, 4], ['cz', 5, 8], ['sq', 6, 0.25, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5]], [['sq', 3, 0.3040867239846965, 0.0, 0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.3040867239846965, 0.0, 0.5], ['cz', 6, 9], ['sq', 8, 0.5, 0.0, -0.5], ['cz', 10, 7]], [['cz', 1, 6], ['cz', 3, 4], ['cz', 5, 8], ['sq', 7, 0.5, 0.0, 0.5], ['sq', 9, 0.5, 0.0, 0.5], ['sq', 10, 0.25, 0.0, 0.5]], [['cz', 0, 5], ['sq', 1, 0.25, 0.0, 0.5], ['cz', 2, 3], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, 0.5], ['cz', 9, 11], ['cz', 10, 7]], [['sq', 0, 0.25, 0.0, 0.5], ['cz', 1, 6], ['sq', 2, 0.25, 0.0, 0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 11, 0.25, 0.0, 0.5]], [['cz', 0, 5], ['cz', 2, 3], ['sq', 6, 0.5, 0.0, 0.5], ['cz', 9, 11]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 9, 0.5, 0.0, 0.5]]]}, 13: {'qubits': [(5, 6), (6, 4), (6, 5), (6, 6), (7, 4), (7, 5), (7, 6), (8, 5), (8, 6), (8, 7), (8, 8), (9, 5), (10, 5)], 'moments': [[['sq', 0, 0.25, 0.0, -0.5], ['sq', 1, 0.25, 0.0, -0.5], ['sq', 2, 0.25, 0.0, -0.5], ['sq', 3, 0.3040867239846965, 0.0, -0.5], ['sq', 4, 0.25, 0.0, -0.5], ['sq', 5, 0.27281447417149507, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.25, 0.0, -0.5], ['sq', 8, 0.47549041230127687, 0.0, 0.5], ['sq', 9, 0.3040867239846965, 0.0, -0.5], ['sq', 10, 0.25, 0.0, -0.5], ['sq', 11, 0.3040867239846965, 0.0, -0.5], ['sq', 12, 0.25, 0.0, -0.5]], [['cz', 6, 8]], [['sq', 6, 0.5, 0.0, -0.5], ['cz', 8, 7]], [['cz', 6, 5], ['sq', 7, 0.25, 0.0, 0.5], ['sq', 8, 0.5, 0.0, -0.5]], [['sq', 5, 0.27281447417149507, 0.0, 0.5], ['sq', 6, 0.5, 0.0, 0.5], ['cz', 8, 7]], [['cz', 6, 5], ['sq', 8, 0.5, 0.0, 0.5], ['cz', 11, 7]], [['cz', 4, 5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['cz', 8, 9], ['sq', 11, 0.3040867239846965, 0.0, 0.5]], [['cz', 3, 6], ['sq', 4, 0.25, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.3040867239846965, 0.0, 0.5], ['cz', 11, 7]], [['sq', 3, 0.3040867239846965, 0.0, 0.5], ['cz', 4, 5], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.5, 0.0, 0.5], ['cz', 8, 9], ['cz', 11, 12]], [['cz', 1, 4], ['cz', 3, 6], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 8, 0.5, 0.0, 0.5], ['cz', 10, 9], ['sq', 11, 0.5, 0.0, -0.5], ['sq', 12, 0.25, 0.0, 0.5]], [['sq', 1, 0.25, 0.0, 0.5], ['cz', 2, 5], ['cz', 3, 0], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.25, 0.0, 0.5], ['cz', 11, 12]], [['sq', 0, 0.25, 0.0, 0.5], ['cz', 1, 4], ['sq', 2, 0.25, 0.0, 0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['cz', 10, 9], ['sq', 11, 0.5, 0.0, 0.5]], [['cz', 2, 5], ['cz', 3, 0], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 9, 0.5, 0.0, 0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5]]]}, 14: {'qubits': [(7, 7), (7, 8), (8, 7), (8, 8), (9, 6), (9, 7), (9, 8), (10, 7), (10, 8), (10, 9), (11, 7), (11, 8), (11, 9), (12, 8)], 'moments': [[['sq', 0, 0.25, 0.0, -0.5], ['sq', 1, 0.25, 0.0, -0.5], ['sq', 2, 0.25, 0.0, -0.5], ['sq', 3, 0.27281447417149507, 0.0, -0.5], ['sq', 4, 0.25, 0.0, -0.5], ['sq', 5, 0.3040867239846965, 0.0, -0.5], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 7, 0.3040867239846965, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.25, 0.0, -0.5], ['sq', 10, 0.25, 0.0, -0.5], ['sq', 11, 0.27281447417149507, 0.0, -0.5], ['sq', 12, 0.25, 0.0, -0.5], ['sq', 13, 0.25, 0.0, -0.5]], [['cz', 6, 8]], [['cz', 3, 6], ['sq', 8, 0.5, 0.0, -0.5]], [['sq', 3, 0.27281447417149507, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5], ['cz', 8, 11]], [['cz', 3, 6], ['sq', 8, 0.5, 0.0, 0.5], ['sq', 11, 0.27281447417149507, 0.0, 0.5]], [['cz', 3, 2], ['sq', 6, 0.5, 0.0, 0.5], ['cz', 8, 11]], [['sq', 2, 0.25, 0.0, 0.5], ['sq', 3, 0.5, 0.0, -0.5], ['cz', 6, 5], ['sq', 8, 0.5, 0.0, -0.5], ['cz', 11, 12]], [['cz', 3, 2], ['sq', 5, 0.3040867239846965, 0.0, 0.5], ['sq', 6, 0.5, 0.0, -0.5], ['cz', 8, 7], ['sq', 11, 0.5, 0.0, -0.5], ['sq', 12, 0.25, 0.0, 0.5]], [['cz', 0, 2], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 6, 5], ['sq', 7, 0.3040867239846965, 0.0, 0.5], ['sq', 8, 0.5, 0.0, 0.5], ['cz', 11, 12]], [['sq', 0, 0.25, 0.0, 0.5], ['sq', 2, 0.5, 0.0, -0.5], ['cz', 3, 1], ['cz', 4, 5], ['sq', 6, 0.5, 0.0, 0.5], ['cz', 8, 7], ['cz', 9, 12], ['sq', 11, 0.5, 0.0, 0.5]], [['cz', 0, 2], ['sq', 1, 0.25, 0.0, 0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.25, 0.0, 0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.25, 0.0, 0.5], ['cz', 10, 7], ['cz', 11, 13], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['cz', 3, 1], ['cz', 4, 5], ['sq', 7, 0.5, 0.0, -0.5], ['cz', 9, 12], ['sq', 10, 0.25, 0.0, 0.5], ['sq', 11, 0.5, 0.0, -0.5], ['sq', 13, 0.25, 0.0, 0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 10, 7], ['cz', 11, 13], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 7, 0.5, 0.0, 0.5], ['sq', 11, 0.5, 0.0, 0.5]]]}, 15: {'qubits': [(2, 7), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (5, 5), (5, 6), (5, 7), (6, 7)], 'moments': [[['sq', 0, 0.25, 0.0, -0.5], ['sq', 1, 0.25, 0.0, -0.5], ['sq', 2, 0.25, 0.0, -0.5], ['sq', 3, 0.25, 0.0, -0.5], ['sq', 4, 0.25, 0.0, -0.5], ['sq', 5, 0.25, 0.0, -0.5], ['sq', 6, 0.25, 0.0, -0.5], ['sq', 7, 0.5212364096070798, 0.0, 0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.27281447417149507, 0.0, -0.5], ['sq', 10, 0.25, 0.0, -0.5], ['sq', 11, 0.25, 0.0, -0.5], ['sq', 12, 0.25, 0.0, -0.5], ['sq', 13, 0.3040867239846965, 0.0, -0.5], ['sq', 14, 0.25, 0.0, -0.5]], [['cz', 8, 7]], [['cz', 2, 7], ['sq', 8, 0.5, 0.0, -0.5]], [['sq', 2, 0.25, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5], ['cz', 8, 9]], [['cz', 2, 7], ['sq', 8, 0.5, 0.0, 0.5], ['sq', 9, 0.27281447417149507, 0.0, 0.5]], [['cz', 3, 2], ['sq', 7, 0.5, 0.0, 0.5], ['cz', 8, 9]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.25, 0.0, 0.5], ['cz', 4, 9], ['cz', 7, 12], ['sq', 8, 0.5, 0.0, -0.5]], [['cz', 3, 2], ['sq', 4, 0.25, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5], ['cz', 8, 13], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 12, 0.25, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['cz', 3, 0], ['cz', 4, 9], ['cz', 7, 12], ['sq', 8, 0.5, 0.0, 0.5], ['sq', 13, 0.3040867239846965, 0.0, 0.5]], [['sq', 0, 0.25, 0.0, 0.5], ['cz', 1, 2], ['sq', 3, 0.5, 0.0, -0.5], ['cz', 4, 5], ['sq', 7, 0.5, 0.0, 0.5], ['cz', 8, 13], ['sq', 9, 0.5, 0.0, 0.5], ['cz', 11, 12]], [['sq', 1, 0.25, 0.0, 0.5], ['sq', 2, 0.5, 0.0, -0.5], ['cz', 3, 0], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.25, 0.0, 0.5], ['cz', 6, 7], ['sq', 8, 0.5, -1.0, -0.5], ['cz', 9, 10], ['sq', 11, 0.25, 0.0, 0.5], ['sq', 12, 0.5, 0.0, -0.5], ['cz', 13, 14]], [['cz', 1, 2], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 4, 5], ['sq', 6, 0.25, 0.0, 0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.25, 0.0, 0.5], ['cz', 11, 12], ['sq', 13, 0.5, 0.0, -0.5], ['sq', 14, 0.25, 0.0, 0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 6, 7], ['cz', 9, 10], ['sq', 12, 0.5, 0.0, 0.5], ['cz', 13, 14]], [['sq', 7, 0.5, 0.0, 0.5], ['sq', 9, 0.5, 0.0, 0.5], ['sq', 13, 0.5, 0.0, 0.5]]]}}


# Source: graph_placement.py; SHA256: cf87bb8ed4c21e0fb87f52f6b92eda403a33a870fc2db2b693e599658e1cf63b
"""Bounded calibration-aware embedding of a circuit's interaction graph.

Only the qubit labels change. Unlike rigid translations this also permits
flexible path arms to turn around missing or poorly calibrated couplers.
"""
import math
import time
from collections import Counter
import cirq

def rank_graph_embeddings(circuit, outputs, k, cz, sq, zero, one, width=1024, count=1, time_budget=3.0):
    started = time.monotonic()
    logical = sorted(outputs)
    n = len(logical)
    index = {q: i for i, q in enumerate(logical)}
    singles, pairs = (Counter(), Counter())
    neighbors = {i: set() for i in range(n)}
    for op in circuit.all_operations():
        ids = [index[q] for q in op.qubits]
        if len(ids) == 1:
            singles[ids[0]] += 1
        elif len(ids) == 2:
            a, b = sorted(ids)
            pairs[a, b] += 1
            neighbors[a].add(b)
            neighbors[b].add(a)
        else:
            raise ValueError('Embedding expects one- and two-qubit gates')
    graph = calibration_graph(cz, sq)
    if n > len(graph):
        return []
    weighted_degree = {i: sum((pairs[tuple(sorted((i, j)))] for j in neighbors[i])) for i in range(n)}
    order = []
    remaining = set(range(n))
    while remaining:
        chosen = max(remaining, key=lambda i: (len(neighbors[i].intersection(order)), len(neighbors[i]), weighted_degree[i], singles[i], -i))
        order.append(chosen)
        remaining.remove(chosen)
    position = {i: p for p, i in enumerate(order)}
    earlier = {i: sorted((j for j in neighbors[i] if position[j] < position[i])) for i in range(n)}
    readout, asymmetry, sq_cost = ({}, {}, {})
    for q in graph:
        e0 = rate(zero.get(q, (0.01, 0.0)))
        e1 = rate(one.get(q, (0.02, 0.0)))
        readout[q] = -(1 - k / n) * math.log1p(-e0) - k / n * math.log1p(-e1)
        asymmetry[q] = math.log1p(-e1) - math.log1p(-e0)
        sq_cost[q] = rate(sq[q])
    scale = 10000 / math.comb(n, k) * k * (n - k) / (n * max(1, n - 1))
    states = [(0.0, (), 0.0, 0.0, 0.0)]
    cap = max(8, width // max(1, len(graph)) * 3)
    for p, i in enumerate(order):
        expanded = []
        for _, assigned, additive, ds, d2 in states:
            prior = earlier[i]
            candidates = set(graph) if not prior else set(graph[assigned[position[prior[0]]]])
            for j in prior[1:]:
                candidates.intersection_update(graph[assigned[position[j]]])
            candidates.difference_update(assigned)
            for q in sorted(candidates):
                if len(graph[q]) < len(neighbors[i]):
                    continue
                cost = additive + readout[q] + singles[i] * sq_cost[q]
                cost += sum((pairs[tuple(sorted((i, j)))] * graph[q][assigned[position[j]]] for j in prior))
                new_ds = ds + asymmetry[q]
                new_d2 = d2 + asymmetry[q] ** 2
                score = cost + scale * max(0.0, new_d2 - new_ds ** 2 / (p + 1))
                expanded.append((score, assigned + (q,), cost, new_ds, new_d2))
        if not expanded:
            return []
        expanded.sort(key=lambda item: (item[0], item[1]))
        states, per_last = ([], Counter())
        for state in expanded:
            last = state[1][-1]
            if per_last[last] < cap:
                states.append(state)
                per_last[last] += 1
                if len(states) >= width:
                    break
        if time.monotonic() - started > time_budget and p + 1 < n:
            return []
    result = []
    for score, assigned, _, _, _ in states[:count]:
        mapping = {logical[i]: assigned[position[i]] for i in range(n)}
        result.append((score, circuit.transform_qubits(mapping), set(assigned)))
    return result


# Source: template_placement.py; SHA256: 60a708c4020e7f73a5e68064492048d64d7bf74f11de6b152713ed19679d9d46
"""Calibrated placement of exact graph-local templates over the Willow grid."""
import math
from collections import Counter
import cirq

def decode_template(n, variant=0):
    data = BALANCED_TEMPLATES[n]
    if isinstance(data, list):
        data = data[variant]
    return _decode_template_data(data)

def decode_w_template(n):
    return _decode_template_data(W_TEMPLATES[n])

def _decode_template_data(data):
    qs = [cirq.GridQubit(*q) for q in data['qubits']]
    moments = []
    for moment in data['moments']:
        ops = []
        for entry in moment:
            if entry[0] == 'cz':
                ops.append(cirq.CZ(qs[entry[1]], qs[entry[2]]))
            else:
                ops.append(cirq.PhasedXZGate(x_exponent=entry[2], z_exponent=entry[3], axis_phase_exponent=entry[4])(qs[entry[1]]))
        moments.append(cirq.Moment(ops))
    return (cirq.Circuit(moments), qs)

def decode_templates(n):
    data = BALANCED_TEMPLATES[n]
    return [decode_template(n, i) for i in range(len(data) if isinstance(data, list) else 1)]

def circuit_calibration_cost(circuit, outputs, k, cz, sq, zero, one):
    n = len(outputs)
    pairs = {}
    for pair, v in cz.items():
        key = frozenset(pair)
        pairs[key] = max(pairs.get(key, 0.0), rate(v))
    result = 0.0
    for op in circuit.all_operations():
        result += rate(sq[op.qubits[0]]) if len(op.qubits) == 1 else pairs[frozenset(op.qubits)]
    ds = []
    for q in outputs:
        e0, e1 = (rate(zero.get(q, (0.01, 0.0))), rate(one.get(q, (0.02, 0.0))))
        result -= (1 - k / n) * math.log1p(-e0) + k / n * math.log1p(-e1)
        ds.append(math.log1p(-e1) - math.log1p(-e0))
    scale = 10000 / math.comb(n, k) * k * (n - k) / (n * max(1, n - 1))
    mean = sum(ds) / n
    return result + scale * sum(((d - mean) ** 2 for d in ds))

def place_template(n, cz, sq, zero, one, count=4):
    preferred = []
    alternatives = []
    for c, qs in decode_templates(n):
        results = rank_template(c, qs, n // 2, cz, sq, zero, one, count)
        results.extend(rank_graph_embeddings(c, qs, n // 2, cz, sq, zero, one, count=count))
        distinct = {}
        for item in sorted(results, key=lambda item: item[0]):
            key = cirq.FrozenCircuit(item[1])
            if key not in distinct:
                distinct[key] = item
        unique = list(distinct.values())
        if unique:
            preferred.append(unique[0])
            alternatives.extend(unique[1:])
    preferred.sort(key=lambda item: item[0])
    alternatives.sort(key=lambda item: item[0])
    distinct = {}
    for item in preferred + alternatives:
        key = cirq.FrozenCircuit(item[1])
        if key not in distinct:
            distinct[key] = item
    return list(distinct.values())[:count]

def rank_template(c, qs, k, cz, sq, zero, one, count=4):
    qs = sorted(qs)
    n = len(qs)
    graph = calibration_graph(cz, sq)
    index = {q: i for i, q in enumerate(qs)}
    singles = Counter()
    edges = Counter()
    for op in c.all_operations():
        if len(op.qubits) == 1:
            singles[index[op.qubits[0]]] += 1
        else:
            edges[tuple((index[q] for q in op.qubits))] += 1
    candidates = []
    seen = set()
    for flip in (False, True):
        for sr, sc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            coords = [(sc * q.col, sr * q.row) if flip else (sr * q.row, sc * q.col) for q in qs]
            anchor = coords[0]
            for physical in graph:
                mapped = tuple((cirq.GridQubit(physical.row + r - anchor[0], physical.col + col - anchor[1]) for r, col in coords))
                if mapped in seen or not all((q in graph for q in mapped)):
                    continue
                seen.add(mapped)
                if not all((mapped[b] in graph[mapped[a]] for a, b in edges)):
                    continue
                cost = sum((w * rate(sq[mapped[i]]) for i, w in singles.items()))
                cost += sum((w * graph[mapped[a]][mapped[b]] for (a, b), w in edges.items()))
                ds = []
                for q in mapped:
                    e0, e1 = (rate(zero.get(q, (0.01, 0.0))), rate(one.get(q, (0.02, 0.0))))
                    cost -= (1 - k / n) * math.log1p(-e0) + k / n * math.log1p(-e1)
                    ds.append(math.log1p(-e1) - math.log1p(-e0))
                cost += 10000 / math.comb(n, k) * k * (n - k) / (n * max(1, n - 1)) * sum(((d - sum(ds) / n) ** 2 for d in ds))
                candidates.append((cost, mapped))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [(cost, c.transform_qubits(dict(zip(qs, mapped))), set(mapped)) for cost, mapped in candidates[:count]]


# Source: divide_snapshot.py; SHA256: f45dd956d2726b1ffc9a90f8e05df94ae8b8ce943ee160d2f426fd4373160bbe
"""Exact hypergeometric split on a ladder, followed by parallel unary decoding."""
import functools
import math
import cirq
import cirq_google

def logical_divide_circuit(n, k, *, use_full_u3=True, use_u4=False, use_width2=True, use_top_row=True):
    if not isinstance(n, int) or not isinstance(k, int) or n < 2 or (not 1 <= k <= n // 2):
        raise ValueError('Require n>=2 and 1<=k<=n/2')
    a, b = (n // 2, n - n // 2)
    top = [cirq.GridQubit(0, i) for i in range(a)]
    bottom = [cirq.GridQubit(1, k - 1 - i) for i in range(b)]
    weights = [math.comb(a, j) * math.comb(b, k - j) for j in range(k + 1)]
    suffix = [sum(weights[j:]) for j in range(k + 1)]
    ops = list(cirq.X.on_each(bottom[:k]))
    theta = 2 * math.acos(math.sqrt(weights[0] / suffix[0]))
    ops.append(cirq.ry(theta)(top[0]))
    for j in range(1, k):
        theta = 2 * math.acos(math.sqrt(weights[j] / suffix[j]))
        angle = math.pi / 2 - theta / 2
        ops.extend([cirq.ry(angle)(top[j]), cirq.CNOT(top[j - 1], top[j]), cirq.ry(-angle)(top[j])])
    ops.extend((cirq.CNOT(top[j], bottom[k - 1 - j]) for j in range(k)))
    upper = unary_to_dicke_circuit(a, k, native=False, use_full_u3=use_full_u3, use_u4=use_u4, use_width2=use_width2, use_top_row=use_top_row).transform_qubits(dict(zip(cirq.LineQubit.range(a), top)))
    lower = unary_to_dicke_circuit(b, k, native=False, use_full_u3=use_full_u3, use_u4=use_u4, use_width2=use_width2, use_top_row=use_top_row).transform_qubits(dict(zip(cirq.LineQubit.range(b), bottom)))
    raw = cirq.Circuit(ops, upper.all_operations(), lower.all_operations())
    return (native_compile(raw), top, bottom)

def embed_ladder_arms(n, k, available, pairs):
    """Embed the k-rung core and extend disjoint chains; labels need not be straight."""
    a, b = (n // 2, n - n // 2)
    available = set(available)
    graph = {q: set() for q in available}
    for x, y in pairs:
        if x in graph and y in graph:
            graph[x].add(y)
            graph[y].add(x)

    def paths(prefix, length, occupied):
        if len(prefix) == length:
            yield prefix
            return
        for nxt in sorted(graph[prefix[-1]] - occupied):
            yield from paths(prefix + [nxt], length, occupied | {nxt})
    middle = sorted(available, key=lambda q: ((q.row - 5.5) ** 2 + (q.col - 5.5) ** 2, q))
    for root in middle:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            for sr, sc in [(-dc, dr), (dc, -dr)]:
                top = [cirq.GridQubit(root.row + i * dr, root.col + i * dc) for i in range(k)]
                bottom = [cirq.GridQubit(root.row + (k - 1 - i) * dr + sr, root.col + (k - 1 - i) * dc + sc) for i in range(k)]
                core = set(top + bottom)
                if len(core) != 2 * k or not core <= available:
                    continue
                required = list(zip(top, top[1:])) + list(zip(bottom, bottom[1:])) + [(top[j], bottom[k - 1 - j]) for j in range(k)]
                if any((y not in graph[x] for x, y in required)):
                    continue
                for full_top in paths(top, a, core):
                    used = core | set(full_top)
                    full_bottom = next(paths(bottom, b, used), None)
                    if full_bottom is not None:
                        return (full_top, full_bottom)
    raise NotImplementedError('No embedded ladder with disjoint extension arms')

@functools.lru_cache(maxsize=128)
def _willow_frozen(n, k):
    c, top, bottom = logical_divide_circuit(n, k)
    device = cirq_google.Willow105.metadata
    pt, pb = embed_ladder_arms(n, k, device.qubit_set, device.qubit_pairs)
    mapped = c.transform_qubits(dict(zip(top + bottom, pt + pb)))
    cirq_google.Willow105.validate_circuit(mapped)
    return (cirq.FrozenCircuit(mapped), frozenset(pt + pb))

def willow_divide_circuit(n, k):
    c, qs = _willow_frozen(n, k)
    return (cirq.Circuit(c), set(qs))


# Source: wtree_snapshot.py; SHA256: 29c68b4b369a4255bfd8f9598a02f88a8ae868baa43f471f75f9c01297cd9e55
"""Exact, shallow W-state (and anti-W, D(N, N-1)) preparation on a branching Willow tree.

Only cirq.CZ (on allowed pairs) and cirq.PhasedXZGate are emitted.  Exact in the
noiseless limit: every amplitude is produced by a closed-form angle from subtree sizes.

Construction
------------
A rooted spanning tree on N hardware qubits plus a *broadcast schedule*: in every round
each qubit that already holds part of the excitation "splits" to at most one fresh
neighbour (child).  Invariant: after a node v received its share, the |1>_v component of
the global state means "the excitation lives somewhere in the not-yet-distributed part of
v's subtree" (size R_v), with amplitude sqrt(R_v / N).

* First split (pure root r, fresh child c):  Ry(beta)_r, CNOT(r->c), X_c.  1 CZ.
      |00> -> sqrt((N-S_c)/N) |10> + sqrt(S_c/N) |01>.
* Later split (parent p in superposition, fresh child c), 2 CZ:
      c: Ry(a), CNOT(p->c), c: Ry(-a), CNOT(c->p),   a = pi/2 - phi,  sin^2(phi) = S_c / R_p
  acts as |00> -> |00>, |10> -> cos(phi)|10> + sin(phi)|01> (both real, positive).
  A CRy onto a known |0> target needs only one CNOT; that is why the split costs 2 CZ.
  One CZ is impossible for this isometry (operator-Schmidt rank argument, see RESULTS.md).

Total CZ = 2N - 3.  With perfect doubling the depth is 2*(2R-1)+1 moments, R = ceil(log2 N).
"""
import math
import random
from dataclasses import dataclass, field
import cirq
import networkx as nx
import numpy as np
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)

def _ry(theta: float) -> np.ndarray:
    c, s = (math.cos(theta / 2), math.sin(theta / 2))
    return np.array([[c, -s], [s, c]], dtype=complex)

def hardware_graph(cz_pauli_error_rate=None, sq_pauli_error_rate=None, zero_errors=None, one_errors=None) -> nx.Graph:
    """Graph of usable qubits/couplers with error attributes.

    Edge attribute 'key' keeps the exact pair orientation from cz_pauli_error_rate so the
    emitted CZ uses a pair that is literally present in the allowed dictionary.
    If no error maps are given, the full Willow105 device with nominal errors is used.
    """
    G = nx.Graph()
    if cz_pauli_error_rate is None:
        import cirq_google
        md = cirq_google.Willow105.metadata
        for q in md.qubit_set:
            G.add_node(q, sq=0.0004, e0=0.005, e1=0.01)
        for pair in md.qubit_pairs:
            a, b = tuple(pair)
            G.add_edge(a, b, cz=0.0025, key=(a, b))
        return G

    def val(d, q, default):
        if d is None or q not in d:
            return default
        v = d[q]
        return float(v[0]) if isinstance(v, (tuple, list)) else float(v)
    for q in sq_pauli_error_rate:
        G.add_node(q, sq=val(sq_pauli_error_rate, q, 0.001), e0=val(zero_errors, q, 0.02), e1=val(one_errors, q, 0.04))
    for (a, b), v in cz_pauli_error_rate.items():
        if a in G and b in G:
            e = float(v[0]) if isinstance(v, (tuple, list)) else float(v)
            if G.has_edge(a, b):
                if e >= G.edges[a, b]['cz']:
                    continue
            G.add_edge(a, b, cz=e, key=(a, b))
    return G

@dataclass
class WTree:
    root: cirq.GridQubit
    calls: list
    nodes: list
    parent: dict = field(default_factory=dict)

    @property
    def rounds(self) -> int:
        return max((r for r, _, _ in self.calls), default=0)

    def children(self, v):
        return [c for _, p, c in self.calls if p == v]

    def subtree_sizes(self) -> dict:
        size = {v: 1 for v in self.nodes}
        for _, p, c in reversed(self.calls):
            size[p] += size[c]
        return size

def _check_tree(tree: WTree, G: nx.Graph | None=None):
    busy = {}
    informed_at = {tree.root: 0}
    for r, p, c in tree.calls:
        assert p in informed_at and informed_at[p] < r, 'parent must hold amplitude first'
        assert c not in informed_at, 'child must be fresh'
        assert (r, p) not in busy and (r, c) not in busy, 'one call per qubit per round'
        if r == 1:
            assert p == tree.root
        busy[r, p] = busy[r, c] = True
        informed_at[c] = r
        if G is not None:
            assert G.has_edge(p, c), f'no coupler {p}-{c}'
    assert set(informed_at) == set(tree.nodes)

def w_tree_circuit(tree: WTree, G: nx.Graph | None=None, complement: bool=False) -> tuple[cirq.Circuit, set]:
    """Build the exact CZ + PhasedXZ circuit for the broadcast tree.

    complement=True returns D(N, N-1) = X^{(x)N} W_N (X merged into the last 1q layer).
    Moment structure: alternating [1q layer, CZ layer, ..., CZ layer, 1q layer].
    """
    _check_tree(tree, G)
    N = len(tree.nodes)
    S = tree.subtree_sizes()
    remaining = dict(S)
    R = tree.rounds
    n_cz_layers = 0 if R == 0 else 2 * R - 1
    oneq = [dict() for _ in range(n_cz_layers + 1)]
    czs = [[] for _ in range(n_cz_layers)]

    def add1(layer, q, m):
        oneq[layer].setdefault(q, []).append(m)

    def cz_pair(p, c):
        if G is not None and G.has_edge(p, c):
            return G.edges[p, c]['key']
        return (p, c)
    if N == 1:
        add1(0, tree.root, _X)
    for r, p, c in tree.calls:
        Sc = S[c]
        if r == 1:
            keep = (N - Sc) / N
            beta = 2 * math.asin(math.sqrt(keep))
            add1(0, p, _ry(beta))
            add1(0, c, _H)
            czs[0].append(cz_pair(p, c))
            add1(1, c, _H)
            add1(1, c, _X)
        else:
            t = 2 * r - 3
            Rp = remaining[p]
            phi = math.asin(math.sqrt(Sc / Rp))
            a = math.pi / 2 - phi
            add1(t, c, _ry(a))
            add1(t, c, _H)
            czs[t].append(cz_pair(p, c))
            add1(t + 1, c, _H)
            add1(t + 1, c, _ry(-a))
            add1(t + 1, p, _H)
            czs[t + 1].append(cz_pair(p, c))
            add1(t + 2, p, _H)
        remaining[p] -= Sc
    if complement:
        for q in tree.nodes:
            add1(n_cz_layers, q, _X)
    moments = []
    for layer in range(n_cz_layers + 1):
        ops = []
        for q, mats in oneq[layer].items():
            m = _I2
            for mm in mats:
                m = mm @ m
            g = cirq.PhasedXZGate.from_matrix(m)
            if not cirq.equal_up_to_global_phase(m, _I2, atol=1e-12):
                ops.append(g.on(q))
        if ops:
            moments.append(cirq.Moment(ops))
        if layer < n_cz_layers:
            moments.append(cirq.Moment((cirq.CZ(*pr) for pr in czs[layer])))
    return (cirq.Circuit(moments), set(tree.nodes))

def _node_cost(G, q, N, n_exc=1):
    d = G.nodes[q]
    readout = (N - n_exc) / N * d['e0'] + n_exc / N * d['e1']
    return readout + 3 * d['sq']

def grow_tree(G: nx.Graph, N: int, root, rng: random.Random, jitter: float=0.0, allowed: set | None=None, n_exc: int=1):
    """Greedy round-by-round broadcast growth using a min-cost maximum matching per round."""
    informed = [root]
    inf_set = {root}
    calls = []
    r = 0
    while len(informed) < N:
        r += 1
        callers = [root] if r == 1 else list(informed)
        B = nx.Graph()
        for p in callers:
            for c in G.neighbors(p):
                if c in inf_set or (allowed is not None and c not in allowed):
                    continue
                usage = 1 if r == 1 else 2
                cost = usage * G.edges[p, c]['cz'] + _node_cost(G, c, N, n_exc)
                cost *= 1 + jitter * rng.random()
                B.add_edge(('p', p), ('c', c), weight=1.0 - cost)
        if B.number_of_edges() == 0:
            return None
        M = nx.max_weight_matching(B, maxcardinality=True)
        pairs = []
        for u, v in M:
            if u[0] == 'c':
                u, v = (v, u)
            pairs.append((B.edges[u, v]['weight'], u[1], v[1]))
        need = N - len(informed)
        pairs.sort(key=lambda x: -x[0])
        for _, p, c in pairs[:need]:
            calls.append((r, p, c))
            informed.append(c)
            inf_set.add(c)
    return WTree(root=root, calls=calls, nodes=informed, parent={c: p for _, p, c in calls})

def tree_cost(G, tree: WTree, n_exc: int=1, per_moment: float=0.0004) -> float:
    """Crude additive infidelity proxy for f (Hamming-weight retention)."""
    N = len(tree.nodes)
    c = sum((_node_cost(G, q, N, n_exc) for q in tree.nodes))
    for r, p, ch in tree.calls:
        c += (1 if r == 1 else 2) * G.edges[p, ch]['cz'] + (0 if r == 1 else 2) * G.nodes[p]['sq']
    R = tree.rounds
    moments = 2 * (2 * R - 1) + 1 if R else 1
    return c + per_moment * moments

def best_w_tree(G: nx.Graph, N: int, seed: int=0, restarts: int=4, time_budget: float=10.0, n_exc: int=1, per_moment: float=0.0004):
    import time
    rng = random.Random(seed)
    t0 = time.time()
    best, best_cost = (None, float('inf'))
    roots = sorted(G.nodes, key=lambda q: _node_cost(G, q, N, n_exc))
    for it in range(restarts):
        for root in roots:
            if time.time() - t0 > time_budget and best is not None:
                return (best, best_cost)
            tr = grow_tree(G, N, root, rng, jitter=0.0 if it == 0 else 0.3, n_exc=n_exc)
            if tr is None:
                continue
            c = tree_cost(G, tr, n_exc, per_moment)
            if c < best_cost:
                best, best_cost = (tr, c)
    if best is None:
        raise NotImplementedError(f'no tree with {N} qubits found')
    return (best, best_cost)

def create_w_state(num_qubits: int, cz_pauli_error_rate=None, sq_pauli_error_rate=None, zero_errors=None, one_errors=None, *, complement: bool=False, seed: int=0, time_budget: float=10.0):
    """D(N,1) (or D(N,N-1) if complement=True) on the best tree found on the error maps."""
    G = hardware_graph(cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    if num_qubits < 1 or num_qubits > G.number_of_nodes():
        raise NotImplementedError
    n_exc = num_qubits - 1 if complement else 1
    tree, _ = best_w_tree(G, num_qubits, seed=seed, time_budget=time_budget, n_exc=n_exc)
    return w_tree_circuit(tree, G, complement=complement)

def create_dicke_state_w(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    """Notebook-signature wrapper: handles n = 1 and n = N - 1 only."""
    N, n = (num_qubits, num_excitations)
    if n == 1:
        return create_w_state(N, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    if n == N - 1 and N >= 2:
        return create_w_state(N, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors, complement=True)
    raise NotImplementedError


# Source: depth_compile.py; SHA256: 577b482f005eb0688122fee365c0ffa705f87f621b31810d39e073abd78c5d8c
"""Exact Z/Pauli ejection and commuting-diagonal moment scheduling.

The pipeline is independently validated in evaluation/bonus_depth_review.json.
Only overlapping diagonal gates may change order. Other gates preserve their
dependency order, so the complete unitary is unchanged up to global phase.
"""
import cirq

def _depth_metrics(circuit):
    operations = list(circuit.all_operations())
    return (len(circuit), sum((len(op.qubits) == 2 for op in operations)), len(operations))

def _depth_normalize(circuit):
    c = cirq.Circuit(circuit.all_operations())
    c = cirq.merge_single_qubit_gates_to_phxz(c, atol=1e-14)
    c = cirq.drop_negligible_operations(c, atol=1e-14)
    operations = [op for op in c.all_operations() if op.qubits]
    if not all((op.gate == cirq.CZ or isinstance(op.gate, cirq.PhasedXZGate) for op in operations)):
        raise ValueError('Expected only full CZ and PhasedXZ gates')
    return cirq.Circuit(operations)

def _depth_diagonal(op):
    return op.gate == cirq.CZ or (isinstance(op.gate, cirq.PhasedXZGate) and op.gate.x_exponent % 2 == 0)

def _depth_schedule(circuit):
    operations = list(circuit.all_operations())
    predecessors = [set() for _ in operations]
    successors = [set() for _ in operations]
    last_barrier = {}
    pending = {}
    for i, op in enumerate(operations):
        for q in op.qubits:
            if q in last_barrier:
                predecessors[i].add(last_barrier[q])
            if _depth_diagonal(op):
                pending.setdefault(q, []).append(i)
            else:
                predecessors[i].update(pending.get(q, ()))
                pending[q] = []
                last_barrier[q] = i
        for previous in predecessors[i]:
            successors[previous].add(i)
    critical = [1] * len(operations)
    for i in range(len(operations) - 1, -1, -1):
        if successors[i]:
            critical[i] += max((critical[j] for j in successors[i]))
    outstanding = [len(deps) for deps in predecessors]
    ready = {i for i, degree in enumerate(outstanding) if degree == 0}
    moments = []
    emitted = 0
    while ready:
        order = sorted(ready, key=lambda i: (-critical[i], -len(operations[i].qubits), i))
        busy, chosen = (set(), [])
        for i in order:
            if not busy.intersection(operations[i].qubits):
                chosen.append(i)
                busy.update(operations[i].qubits)
        moments.append(cirq.Moment((operations[i] for i in sorted(chosen))))
        emitted += len(chosen)
        ready.difference_update(chosen)
        for i in chosen:
            for j in successors[i]:
                outstanding[j] -= 1
                if outstanding[j] == 0:
                    ready.add(j)
    if emitted != len(operations):
        raise ValueError('Invalid gate dependency graph')
    return cirq.Circuit(moments)

def optimize_native_depth(circuit):
    c = cirq.eject_z(circuit, atol=0)
    c = cirq.eject_phased_paulis(c, atol=0)
    c = cirq.eject_z(c, atol=0)
    c = _depth_normalize(c)
    candidate = _depth_normalize(_depth_schedule(c))
    return min((circuit, candidate), key=_depth_metrics)


# Source: evaluation/alap_prefix.py; SHA256: 78f917193ff8477c478e4f2d56584d49dbaff22560ef4927cc36a715c6eb8f49
"""Delay a qubit's one already-merged preparation gate to its first CZ.

The pass only changes scheduling. It preserves every operation, each wire's
operation order, all moment indices of entangling gates, and total depth.
It does not recompile, merge, fit, or remove gates.
"""
import cirq

def alap_leading_single(circuit):
    """Move a lone native leading SQ gate immediately before its first CZ.

    If a wire has multiple leading SQ gates, no entangler, or any unexpected
    operation, leave that wire alone. Every moved gate crosses only disjoint
    operations; consequently the complete unitary and independent per-gate
    local noise channel are unchanged. Apply after all EARLIEST repacking.
    Empty moments are retained, making the depth exactly invariant.
    """
    moments = [list(moment.operations) for moment in circuit]
    first_two = {}
    for index, moment in enumerate(circuit):
        for op in moment.operations:
            for q in op.qubits:
                seen = first_two.setdefault(q, [])
                if len(seen) < 2:
                    seen.append((index, op))
    for q, seen in first_two.items():
        if len(seen) != 2:
            continue
        (source, leading), (first_entangler, following) = seen
        if len(leading.qubits) != 1 or not isinstance(leading.gate, cirq.PhasedXZGate) or following.gate != cirq.CZ:
            continue
        destination = first_entangler - 1
        if destination <= source:
            continue
        moments[source].remove(leading)
        moments[destination].append(leading)
    return cirq.Circuit((cirq.Moment(ops) for ops in moments))


# Source: candidate_v6.py; SHA256: c9484492552594b38168e522d4d448bded1eb41500ff3cd075dbf944151b45f8
"""Portfolio: exact square, branching W, hypergeometric split, balanced and line."""
import time
import cirq
import cirq_google

def _hardware_portfolio(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    started = time.monotonic()
    n, k = (num_qubits, num_excitations)
    if not isinstance(n, int) or not isinstance(k, int) or n < 1 or (not 0 <= k <= n):
        raise ValueError('Require integer N>=1 and 0<=k<=N')
    if n > len(sq_pauli_error_rate):
        raise NotImplementedError('Insufficient calibrated qubits')
    if k in (0, n):
        errors = zero_errors if k == 0 else one_errors
        qs = sorted(sq_pauli_error_rate, key=lambda q: (rate(errors.get(q, (1.0, 0.0))), q))[:n]
        gate = cirq.PhasedXZGate(x_exponent=int(k == n), z_exponent=0, axis_phase_exponent=0)
        return (cirq.Circuit(cirq.Moment((gate(q) for q in qs))), set(qs))
    args = (cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    if n == 2 and k == 1:
        try:
            return _create_bell(n, k, *args)
        except NotImplementedError:
            pass
    if n == 4 and k == 2:
        try:
            return create_dicke4(*args)
        except NotImplementedError:
            pass
    choices = []
    if n in (6, 8, 10, 12, 14) and k == n // 2:
        choices = place_template(n, *args, count=4 if n <= 8 else 1)
    if k not in (1, n - 1) and n <= 24:
        try:
            dc, qs = willow_divide_circuit(n, min(k, n - k))
            if k > n // 2:
                dc = cirq.Circuit(cirq.merge_single_qubit_gates_to_phxz(dc + cirq.Circuit(cirq.X.on_each(qs))).all_operations())
            dc_choices = rank_template(dc, qs, k, *args, count=1)
            dc_choices.extend(rank_graph_embeddings(dc, qs, k, *args, count=1))
            choices.extend(sorted(dc_choices, key=lambda item: item[0])[:1])
        except NotImplementedError:
            pass
    if k in (1, n - 1) and n > 2:
        if n <= 15:
            template, tq = decode_w_template(n)
            if k > 1:
                template = cirq.Circuit(cirq.merge_single_qubit_gates_to_phxz(template + cirq.Circuit(cirq.X.on_each(tq))).all_operations())
            w_choices = rank_template(template, tq, k, *args, count=1)
            w_choices.extend(rank_graph_embeddings(template, tq, k, *args, count=1))
            choices.extend(w_choices)
        try:
            wc, wq = create_w_state(n, *args, complement=k > 1, time_budget=2.0)
            choices.append((circuit_calibration_cost(wc, wq, k, *args), wc, wq))
        except NotImplementedError:
            pass
    try:
        choices.extend(line_placement_candidates(logical_circuit(n, k), n, k, *args, count=2 if n <= 8 else 1))
    except NotImplementedError:
        pass
    if not choices:
        raise NotImplementedError('No valid calibrated placement found')
    choices = [item for item in choices if len(item[1]) <= 1750]
    if not choices:
        raise NotImplementedError('Exceeds1750-moment limit')
    distinct = {}
    for item in sorted(choices, key=lambda item: item[0]):
        key = cirq.FrozenCircuit(item[1])
        if key not in distinct:
            distinct[key] = item
    choices = list(distinct.values())
    if n <= 8 and len(choices) > 1:
        choices.sort(key=lambda item: item[0])
        refined = []
        for _, c, qs in choices[:8]:
            if refined and time.monotonic() - started > 30:
                break
            refined.append((small_proxy(c, sorted(qs), k, *args), c, qs))
        choices = refined
    score, circuit, outputs = min(choices, key=lambda item: item[0])
    improved = optimize_native_depth(circuit)
    if improved != circuit:
        if n <= 8 and time.monotonic() - started < 35:
            old_score = score if len(choices) > 1 else small_proxy(circuit, sorted(outputs), k, *args)
            if small_proxy(improved, sorted(outputs), k, *args) <= old_score:
                circuit = improved
        elif n > 8:
            old_cost = circuit_calibration_cost(circuit, outputs, k, *args)
            new_cost = circuit_calibration_cost(improved, outputs, k, *args)
            if new_cost <= old_cost + 1e-14:
                circuit = improved
    return (circuit, outputs)

def create_dicke_state_echo(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    c, q = create_dicke_state(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    return (echo_idle_qubits(c, min_idle_moments=32), q)

def _bonus_portfolio(num_qubits, num_excitations):
    n, k = (num_qubits, num_excitations)
    if not isinstance(n, int) or not isinstance(k, int) or n < 2 or (not 1 <= k <= n // 2):
        raise ValueError('Require integer N>=2 and 1<=k<=N/2')
    if (n, k) == (4, 2):
        qs = [cirq.GridQubit(4, 4), cirq.GridQubit(4, 5), cirq.GridQubit(5, 5), cirq.GridQubit(5, 4)]
        return (dicke4_square(qs), set(qs))
    qs = cirq_google.Willow105.metadata.qubit_set
    pairs = cirq_google.Willow105.metadata.qubit_pairs
    args = ({tuple(p): (0.0, 0.0) for p in pairs}, {q: (0.0, 0.0) for q in qs}, {q: (0.0, 0.0) for q in qs}, {q: (0.0, 0.0) for q in qs})
    if n == 2:
        return _create_bell(n, k, *args)
    if k == 1:
        if n <= 15:
            c, q = decode_w_template(n)
            return (c, set(q))
        c, q = create_w_state(n, *args, time_budget=2.0)
        return (optimize_native_depth(c), q)
    line, lq = place_line(logical_circuit(n, k), n, k, *args)
    choices = [(line, lq)]
    try:
        choices.append(willow_divide_circuit(n, k))
    except NotImplementedError:
        pass
    if n in (6, 8, 10, 12, 14) and k == n // 2:
        choices.extend(((template, set(tq)) for template, tq in decode_templates(n)))
    return min(((optimize_native_depth(c), q) for c, q in choices), key=lambda item: _depth_metrics(item[0]))


# Source: safe_api.py; SHA256: e8c3431c23176ee947c93124fb30c515662a026c067affcf15dbc778a040f3e1
"""Judge-facing entry points with literal coupler orientation and fallback."""
import cirq
import cirq_google

def _checked_hardware(circuit, outputs, n, cz, sq):
    outputs = set(outputs)
    if len(outputs) != n or not outputs.issubset(sq) or len(circuit) > 1750:
        raise NotImplementedError('Invalid output set or depth')
    moments = []
    for moment in circuit:
        ops = []
        for op in moment:
            if op.gate == cirq.CZ:
                pair = tuple(op.qubits)
                if pair not in cz:
                    pair = pair[::-1]
                    if pair not in cz:
                        raise NotImplementedError('Uncalibrated coupler')
                op = cirq.CZ(*pair)
            elif not isinstance(op.gate, cirq.PhasedXZGate):
                raise NotImplementedError('Unexpected non-native gate')
            if any((q not in sq for q in op.qubits)):
                raise NotImplementedError('Uncalibrated qubit')
            ops.append(op)
        moments.append(cirq.Moment(ops))
    return (alap_leading_single(cirq.Circuit(moments)), outputs)

def create_dicke_state(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    args = (cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    try:
        c, q = _hardware_portfolio(num_qubits, num_excitations, *args)
        return _checked_hardware(c, q, num_qubits, cz_pauli_error_rate, sq_pauli_error_rate)
    except Exception:
        try:
            c, q = place_line(logical_circuit(num_qubits, num_excitations), num_qubits, num_excitations, *args)
            return _checked_hardware(c, q, num_qubits, cz_pauli_error_rate, sq_pauli_error_rate)
        except Exception as error:
            raise NotImplementedError('No valid exact circuit: ' + type(error).__name__) from None

def create_dicke_state_adaptive(num_qubits, num_excitations):
    try:
        c, q = _bonus_portfolio(num_qubits, num_excitations)
        if len(q) != num_qubits or len(c) > 1750:
            raise ValueError('Invalid output set or depth')
        if not all((op.gate == cirq.CZ or isinstance(op.gate, cirq.PhasedXZGate) for op in c.all_operations())):
            raise ValueError('Unexpected non-native gate')
        c = alap_leading_single(c)
        cirq_google.Willow105.validate_circuit(c)
        return (c, set(q))
    except Exception as error:
        raise NotImplementedError('No valid exact bonus circuit: ' + type(error).__name__) from None

_FINAL_ASTRA_MAIN=create_dicke_state
_FINAL_ASTRA_BONUS=create_dicke_state_adaptive

_FINAL_EXTRA={(5, 2): [{'n': 5, 'k': 2, 'qubits': [[3, 5], [4, 5], [4, 6], [5, 5], [5, 6]], 'moments': [[['sq', 0, 0.1959132760153037, 5.183513872475822e-08, 0.5], ['sq', 1, 0.3333333333333333, 0.9999998307071643, 0.5], ['sq', 2, 0.5, 2.598822158415535e-17, 0.5], ['sq', 3, -0.29516723530086647, -1.0000000000000002, 0.5], ['sq', 4, -0.6666666666666666, 1.0, 0.5]], [['cz', 3, 4]], [['sq', 4, -0.33333333333333337, 21756.541190195523, 0.5], ['cz', 3, 1]], [['cz', 4, 2], ['sq', 3, 0.5, -32734.62973664393, 0.5], ['sq', 1, 0.6666666666666681, 0.9999998730303662, 0.5]], [['sq', 4, 0.5, 6.024557572515177e-17, 0.5], ['sq', 2, 0.5, -21753.958809931446, 0.5], ['cz', 1, 0]], [['cz', 3, 4], ['sq', 1, 0.6959132760153054, -1.0000001466118924, 0.5], ['sq', 0, 0.6959132760153038, -2.992703410520468e-08, 0.5]], [['sq', 4, 0.5, 32595.370263356075, 0.5], ['cz', 2, 1]], [['sq', 2, 0.5, -1.0, 0.5], ['sq', 1, 0.19591327601530364, 1.0000001795621705, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.7499999999999999, -0.4999999576767804, 0.5], ['sq', 0, 0.7500000000000011, 32596.370263271427, 0.5]], [['cz', 2, 1]], [['sq', 2, 0.5, 32596.87026331375, 0.5], ['sq', 1, 0.5, -1.0, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.49999999999999994, 32595.37026327143, 0.5]]], 'cz': 9, 'residual': 4.360526229350869e-12, 'source': 'local_prune\\candidate_5_2_9cz_sq.cirq.json'}, {'n': 5, 'k': 2, 'qubits': [[3, 5], [4, 5], [4, 6], [5, 5], [5, 6]], 'moments': [[['sq', 0, 0.5000000000000003, -0.30377017242489757, 0.5], ['sq', 1, 0.49999999999999994, 0.5, 0.5], ['sq', 2, 0.5000000000000001, -0.5, -0.5], ['sq', 3, 0.6641868814212597, 0.413629038091223, -0.5], ['sq', 4, 0.4293050623646759, 0.6320465085643596, -0.5]], [['cz', 3, 4]], [['sq', 3, 0.23432606472107978, -0.4094613009426271, -0.5], ['sq', 4, 0.5089084392942935, -0.7525196160326788, -0.5]], [['cz', 3, 1], ['cz', 4, 2]], [['sq', 3, 0.7500000000000001, 4.440892098500626e-16, 0.5], ['sq', 1, 0.38746446708156956, 0.02066505568974686, -0.5], ['sq', 4, 1.992244125879467e-17, 0.30474062330736273, -0.5000000000000002], ['sq', 2, 0.45875539218505157, 1.0, -0.5]], [['cz', 1, 0], ['cz', 3, 4]], [['sq', 1, 0.19591327601530364, 0.6959132760153022, 0.49999999999999994], ['sq', 0, 0.5168689922915881, -0.2380578403990523, 0.5], ['sq', 3, 0.25000000000000006, 0.5275556511206005, 0.5], ['sq', 4, 0.49999999999999994, 0.0, 0.5]], [['cz', 2, 1], ['cz', 3, 4]], [['sq', 2, 0.6959132760153038, 1.0, 0.5], ['sq', 1, 0.5000000000000001, -0.5000000000000018, 0.5], ['sq', 3, 1.0, 0.0, -0.3857942006584991], ['sq', 4, 0.5000000000000001, 0.7008559475624048, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.4999999999999998, -0.5000000000000004, 0.49999999999999994], ['sq', 0, 0.5, 0.4999999999999991, 0.5]], [['cz', 2, 1]], [['sq', 2, 0.5, 0.9629431985167436, 0.5], ['sq', 1, 0.24999999999999992, 1.0, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.5, -0.037056801483256185, 0.5], ['sq', 0, 0.5000000000000001, -0.037056801483257074, 0.5]]], 'cz': 10, 'residual': 9.379544120789983e-15, 'source': 'direct_search\\candidate_5_2_10cz.cirq.json'}], (6, 3): [{'n': 6, 'k': 3, 'qubits': [[4, 3], [4, 4], [5, 3], [5, 4], [6, 3], [6, 4]], 'moments': [[['sq', 0, 0.20978468837241684, -1.0, 0.5], ['sq', 1, 1.5212364096070798, 1.0000000000000002, 0.5], ['sq', 2, 0.5, -1.0, 0.5], ['sq', 3, 0.7038845844471661, 7.06948367233203e-17, 0.5], ['sq', 4, 0.5, -0.5000000000000001, 0.5], ['sq', 5, 0.5, 224.50000000000003, 0.5]], [['cz', 0, 1]], [['sq', 0, -0.2097846883724169, -15296.455169008472, 0.5], ['cz', 1, 3]], [['cz', 0, 2], ['sq', 1, -0.8040867239846965, -1.0, 0.5], ['sq', 3, 0.29611541555283416, -2.276376362177206e-16, 0.5]], [['sq', 2, 1.5, -15300.455169008472, 0.5], ['cz', 0, 1]], [['cz', 2, 4], ['sq', 0, -0.5, -0.5000000000000001, 0.5], ['sq', 1, -0.19591327601530348, -2.0, 0.5]], [['sq', 2, 0.14758361765043337, 0.5, 0.5]], [['cz', 3, 2]], [['sq', 3, 0.75, 2.9779112160266155e-17, 0.5], ['sq', 2, 0.5, -0.24999999999999997, 0.5]], [['cz', 2, 4], ['cz', 3, 1]], [['sq', 2, -0.5000000000000001, -1.0000000000000002, 0.5], ['sq', 4, -0.39758361765043326, -0.4999999999999998, 0.5], ['sq', 3, -0.24999999999999997, 0.9999999999999996, 0.5]], [['cz', 2, 4], ['cz', 5, 3]], [['sq', 3, 0.5, -0.5, 0.5], ['cz', 5, 4]], [['cz', 3, 1], ['sq', 5, -11.5, -5102.265083376326, 0.5], ['sq', 4, 0.49999999999999994, 5102.2349166236745, 0.5]], [['sq', 1, 0.5000000000000001, 0.5000000000000001, 0.5], ['cz', 3, 2]], [['cz', 1, 0], ['sq', 3, 0.49999999999999994, -5102.265083376326, 0.5], ['sq', 2, 0.5, -5101.765083376326, 0.5]], [['sq', 1, 0.5000000000000001, -5101.265083376326, 0.5], ['sq', 0, 0.5000000000000001, -5101.765083376326, 0.5]]], 'cz': 14, 'residual': 1.5791248240149163e-12, 'source': 'local_prune\\candidate_6_3_14cz_balanced_sq.cirq.json'}, {'n': 6, 'k': 3, 'qubits': [[4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7]], 'moments': [[['sq', 0, 5.304086723984696, 7.0, 0.5], ['sq', 1, 1.2048327646991335, -35.5, 0.5], ['sq', 2, 0.5, 2701.5, 0.5], ['sq', 3, -1.5, 1.5, 0.5], ['sq', 4, 0.5, -11.240955216597516, 0.5], ['sq', 5, 0.5, 2906.499999999999, 0.5]], [['cz', 3, 4]], [['cz', 4, 5]], [['sq', 4, 1.3040867239846965, -2.0, 0.5], ['sq', 5, -91.00904478340249, 2933.9999999999995, 0.5]], [['cz', 4, 1], ['cz', 5, 2]], [['sq', 4, 0.6959132760153036, -1.2500000000000002, 0.5], ['sq', 1, 12.666666666666668, -46.0, 0.5], ['sq', 5, 0.5000000000000001, -0.5, 0.5]], [['cz', 1, 0], ['cz', 5, 4]], [['sq', 1, 4.25, -16.0, 0.5], ['sq', 0, 2.3040867239846965, 39285.33685819926, 0.5], ['sq', 5, 0.5, 9666.471901354469, 0.5], ['cz', 3, 4]], [['cz', 1, 0], ['sq', 3, 0.5, -23544.5, 0.5], ['sq', 4, 0.5, -1440.5615082278216, 0.5]], [['sq', 1, -5.25, -39356.33685819926, 0.5], ['cz', 4, 3]], [['cz', 1, 2], ['sq', 4, 1.0, 1261.9357217251904, 0.5], ['sq', 3, -0.5, -23700.00277004699, 0.5]], [['sq', 1, 0.49999999999999994, -1.0, 0.5], ['sq', 2, 0.5, -1.0, 0.5], ['cz', 4, 5]], [['cz', 0, 1], ['sq', 5, -1.6880585753848136e-17, -6964.474671401455, 0.5]], [['sq', 0, 0.5, -0.002770046988151846, 0.5], ['sq', 1, 0.5000000000000001, -39270.83685819926, 0.5]], [['cz', 2, 1]], [['sq', 2, 0.5, 1.4972299530118482, 0.5], ['sq', 1, 0.5, -1.0, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.5, -1.0027700469881518, 0.5]]], 'cz': 14, 'residual': 6.424570543841875e-12, 'source': 'local_prune\\candidate_6_3_14cz_sq.cirq.json'}, {'n': 6, 'k': 3, 'qubits': [[4, 3], [4, 4], [5, 3], [5, 4], [6, 3], [6, 4]], 'moments': [[['sq', 0, 0.22643317490804352, -0.9093904344571828, -0.5], ['sq', 1, 0.6145975419752596, -0.8555022656948945, -0.5], ['sq', 2, 0.49999999999999983, 0.1583523210975275, 0.5], ['sq', 3, 0.29611541555283405, 1.1102230246251565e-16, 0.5], ['sq', 4, 0.7436321034001204, 0.00447095922801255, 0.5], ['sq', 5, 0.5, 0.7861025741172161, 0.5]], [['cz', 0, 1]], [['sq', 0, 0.8652947187757795, 0.9126662084493313, 0.5], ['sq', 1, 0.8684092687026346, 0.40916583944582063, 0.5]], [['cz', 0, 2], ['cz', 1, 3]], [['sq', 0, 0.49999999999999983, 0.5, 0.5], ['sq', 2, 0.4800199708301628, -0.46339300040243714, 0.5], ['sq', 1, 0.26661506024497816, 0.7430869083263505, 0.5], ['sq', 3, 0.703884584447166, 8.881784197001252e-16, 0.5]], [['cz', 0, 1], ['cz', 2, 4]], [['sq', 0, 0.7167329707440594, 1.0, 0.49999999999999994], ['sq', 1, 1.0, 0.0, 0.22990993953520478], ['sq', 2, 0.78706212165858, 0.9896134342228788, 0.5], ['sq', 4, 0.49999999999999983, 0.0, 0.5]], [['cz', 0, 1], ['cz', 3, 2]], [['sq', 0, 0.10354215668300573, -0.9999999999999998, -0.5], ['sq', 1, 0.26661506024497866, -0.04041354526095797, -0.5], ['sq', 3, 0.24999999999999983, 0.0, 0.5], ['sq', 2, 0.6794319595054809, -0.9870662498400764, 0.5]], [['cz', 2, 4], ['cz', 3, 1]], [['sq', 2, 0.10260454378043365, 0.9795386623993516, 0.5], ['sq', 4, 0.3155360350764457, 0.028353980937656398, 0.5], ['sq', 3, 0.2499999999999999, 0.0, 0.5], ['sq', 1, 2.078850232934214e-17, 0.7601927697856681, 0.5]], [['cz', 2, 4], ['cz', 5, 3]], [['sq', 2, 0.2727833953716869, 0.504469636713631, -0.5], ['sq', 4, 2.0475253503525847e-16, 0.4716460190623435, 0.5], ['sq', 5, 1.0, 0.0, 0.14305128705860798], ['sq', 3, 0.5, -0.7028784442207385, 0.5]], [['cz', 5, 4], ['cz', 3, 1]], [['sq', 5, 0.5, -0.3567809934716333, 0.5], ['sq', 4, 0.5000000000000001, 0.14321900652836694, 0.5], ['sq', 3, 1.1695680047042487e-16, -0.7971215557792615, 0.5], ['sq', 1, 0.5000000000000001, 0.5, 0.5]], [['cz', 3, 2], ['cz', 1, 0]], [['sq', 3, 0.5, -0.35678099347163306, 0.5], ['sq', 2, 0.5000000000000001, 0.14321900652836694, 0.5], ['sq', 1, 0.5, 0.6432190065283669, 0.5], ['sq', 0, 0.5000000000000001, 0.14321900652836694, 0.5]]], 'cz': 15, 'residual': 2.2603067459636836e-15, 'source': 'direct_search\\candidate_6_3_15cz_balanced.cirq.json'}, {'n': 6, 'k': 3, 'qubits': [[4, 3], [4, 4], [5, 3], [5, 4], [6, 3], [6, 4]], 'moments': [[['sq', 0, 1376.3040867239847, 6203.0, 0.5], ['sq', 1, -0.3040867239846963, 1.0, 0.5], ['sq', 2, -3326.6309898804348, 682.5, 0.5], ['sq', 3, 0.227185525828505, 4.180927621920264e-17, 0.5], ['sq', 4, 0.5, -7.742942702846523e-17, 0.5], ['sq', 5, 0.5, -0.5, 0.5]], [['cz', 0, 2]], [['sq', 0, -1764.3040867239847, -1590.0000000000002, 0.5], ['cz', 2, 3]], [['sq', 2, -0.5, -0.25000000000000006, 0.5], ['sq', 3, -0.227185525828505, 1.0, 0.5]], [['cz', 1, 3], ['cz', 2, 4]], [['sq', 1, 0.6959132760153038, -2.778751687005279e-17, 0.5], ['sq', 3, 0.2499999999999999, -1.0, 0.5], ['sq', 2, 0.5, 0.49999999999999994, 0.5]], [['cz', 0, 2], ['cz', 3, 1]], [['sq', 0, 0.5000000000000001, 0.5000000000000001, 0.5], ['sq', 2, -0.25, -2.5, 0.5], ['sq', 3, 0.7500000000000001, -1.0, 0.5]], [['cz', 2, 4], ['cz', 5, 3]], [['sq', 3, 0.5, 0.5000000000000001, 0.5], ['cz', 5, 4]], [['cz', 3, 1], ['sq', 5, 0.5, 4536.370029688775, 0.5], ['sq', 4, 0.5, 4536.870029688775, 0.5]], [['sq', 1, 0.5, 0.5, 0.5], ['cz', 3, 2]], [['cz', 1, 0], ['sq', 3, -0.49999999999999994, -4537.629970311225, 0.5], ['sq', 2, 3813.5, 2459.870029688775, 0.5]], [['sq', 1, 0.49999999999999994, 4537.370029688775, 0.5], ['sq', 0, 0.5, 4536.870029688775, 0.5]]], 'cz': 12, 'residual': 1.3081409952687938e-12, 'source': 'completion_search\\native_candidates\\candidate_6_3_12cz_sq.cirq.json'}], (7, 3): [{'n': 7, 'k': 3, 'qubits': [[3, 5], [4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7]], 'moments': [[['sq', 0, 0.23346263562123676, 0.09680310696012473, 0.5], ['sq', 1, 0.5000000000000001, -0.33519779061546107, 0.5], ['sq', 2, 0.75, 1.0, 0.5], ['sq', 3, 0.5, 1.0, 0.5], ['sq', 4, 0.5439241310515563, -0.3249105730274453, 0.49999999999999994], ['sq', 5, 0.36099330313764266, -0.8423735342902232, 0.5], ['sq', 6, 0.4652748742609469, 1.0, 0.5]], [['cz', 4, 5]], [['sq', 4, 0.2875635924879449, -0.904248117149197, 0.5], ['sq', 5, 0.532607423202116, -0.6849929712023104, 0.5]], [['cz', 4, 1], ['cz', 5, 6]], [['sq', 4, 1.0, 0.0, 0.6795120730799196], ['sq', 1, 0.6108838999403319, 0.06112364929646752, 0.5], ['sq', 5, 0.49999999999999994, -0.5965764022616762, 0.5], ['sq', 6, 0.3503967845281199, 0.791435554180036, 0.5]], [['cz', 6, 3], ['cz', 5, 4], ['cz', 1, 2]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 3, 0.5, -0.4702703427937607, 0.5], ['sq', 5, 0.24037987725671286, 0.08958177328159911, 0.5], ['sq', 4, 0.25000000000000006, 0.0, 0.5], ['sq', 1, 0.5000000000000001, 0.19591327601530395, 0.5], ['sq', 2, 0.6320277731639083, -0.4999999999999998, 0.5]], [['cz', 4, 5], ['cz', 2, 3]], [['sq', 4, 0.75, -0.9807150792042418, 0.5], ['sq', 5, 7.25822796650499e-17, -0.018238367464662897, 0.5], ['sq', 2, 0.5000000000000002, 0.8215907364977066, 0.5], ['sq', 3, 1.0, 0.0, 0.4256314950418236]], [['cz', 6, 5], ['cz', 1, 2]], [['sq', 6, 0.5000000000000001, -0.10674362401575421, 0.5], ['sq', 5, 0.5000000000000001, 0.0, 0.5], ['sq', 1, 0.6666666666666667, -0.354847437565305, 0.5], ['sq', 2, 0.8291885549916075, -0.7230611869234917, 0.5]], [['cz', 5, 4], ['cz', 3, 2]], [['sq', 5, 0.5000000000000001, 0.6703723052793857, 0.5], ['sq', 4, 1.0, 0.0, -0.6808027365194838], ['sq', 3, 0.5, 1.0, 0.5], ['sq', 2, 0.9718346440436038, -0.9325562035174695, 0.5]], [['cz', 5, 6], ['cz', 1, 2]], [['sq', 5, 2.3165180176245194e-17, -0.05126269911411119, -0.5], ['sq', 6, 1.0, 0.0, 0.5061829910747602], ['sq', 1, 0.4205497467660725, 0.4497803225509407, 0.5], ['sq', 2, 0.5000000000000001, -0.42620976487146733, 0.5]], [['cz', 1, 0], ['cz', 2, 3]], [['sq', 1, 0.6326561005067499, 0.10782350391620188, 0.5], ['sq', 0, 0.7010158865227026, 0.997457607891409, 0.49999999999999994], ['sq', 2, 1.0, 0.0, -0.46310488243573356], ['sq', 3, 0.5, 0.08901182746790326, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.7301934385056068, -0.6039279398383677, 0.5], ['sq', 0, 0.6488254987416722, 0.31982092378724314, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.5, 0.0, 0.5], ['sq', 2, 0.5, 0.0, 0.5]], [['cz', 0, 1]], [['sq', 0, 0.5, -0.47341502449765893, 0.5], ['sq', 1, 0.5, 0.7403481844396369, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.49999999999999994, 1.0, 0.5], ['sq', 2, 0.5, -0.910988172532097, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.5, -0.41098817253209674, 0.5], ['sq', 0, 1.3145880996945247e-17, 0.06242685196556241, 0.5]]], 'cz': 21, 'residual': 1.878558504603599e-15, 'source': 'direct_search\\candidate_7_3_21cz.cirq.json'}, {'n': 7, 'k': 3, 'qubits': [[3, 5], [4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7]], 'moments': [[['sq', 0, 2.98990646078402e-17, -0.14130179031352963, -0.5], ['sq', 1, 0.5000000000000001, -0.5721993096269484, 0.5], ['sq', 2, 0.75, 1.0, 0.5], ['sq', 3, 0.5, 1.0, 0.5], ['sq', 4, 0.5439241318645377, -0.3249105734692026, 0.5], ['sq', 5, 0.3609933037565794, -0.8423735388425244, 0.5], ['sq', 6, 0.46527487426094694, 0.9999999999999996, 0.5]], [['cz', 4, 5]], [['sq', 4, 0.2875635878708291, -0.545223962295414, 0.5], ['sq', 5, 0.5326074210506879, -0.6849929718414312, 0.5]], [['cz', 4, 1], ['cz', 5, 6]], [['sq', 4, 1.0, 0.0, -0.1409758450440428], ['sq', 1, 0.5319047913303867, 0.37217908897987595, 0.5], ['sq', 5, 0.5000000000000001, -0.5965764026111344, 0.5], ['sq', 6, 0.35039678452811995, 0.7914355565902798, 0.5]], [['cz', 6, 3], ['cz', 0, 1], ['cz', 5, 4]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 3, 0.5, -0.6190073583024396, 0.5], ['sq', 0, 0.23346263634960418, 0.096803106226659, 0.5], ['sq', 1, 0.24326790811419663, -0.36663325029496896, 0.5], ['sq', 5, 0.2403798762059917, 0.10782013429449944, 0.5], ['sq', 4, 0.24999999999999992, 0.0, 0.5]], [['cz', 4, 5], ['cz', 1, 2]], [['sq', 4, 0.75, 0.6576794517407847, 0.5], ['sq', 5, 1.097495530934548e-18, -0.0364767279976379, 0.5], ['sq', 1, 0.5000000000000002, 0.1959132760153035, 0.5], ['sq', 2, 0.632027772969383, -0.4999999999999998, 0.49999999999999994]], [['cz', 6, 5], ['cz', 2, 3]], [['sq', 6, 0.5, -0.09437765891219874, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 2, 0.5000000000000002, 0.821590736112018, 0.5], ['sq', 3, 1.0, 0.0, -0.6487370103845658]], [['cz', 5, 4], ['cz', 1, 2]], [['sq', 5, 0.5, 0.7216350050329243, 0.5], ['sq', 4, 1.0, 0.0, 0.13839452956603532], ['sq', 1, 0.6666666666666667, -0.3548474413474296, 0.5], ['sq', 2, 0.8291885548476134, -0.7230611801175955, 0.5]], [['cz', 5, 6], ['cz', 3, 2]], [['sq', 5, 4.354709451073785e-17, -0.10252539764163782, -0.5], ['sq', 6, 1.0, 0.0, 0.5123659742395437], ['sq', 3, 0.5000000000000001, 1.0, 0.5], ['sq', 2, 0.9718346440887352, -0.9325561962343139, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.4205497461326053, 0.4497803247436556, 0.5], ['sq', 2, 0.5000000000000002, -0.3524195578496536, 0.5]], [['cz', 1, 0], ['cz', 2, 3]], [['sq', 1, 0.6326561032694434, 0.10782350273907904, 0.5], ['sq', 0, 0.7010158862704285, 0.9974576091629359, 0.5], ['sq', 2, 1.0, 0.0, -0.4262097789248269], ['sq', 3, 0.5, 0.08901182644825756, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.730193438975439, -0.6039279414745997, 0.5], ['sq', 0, 0.6488254961482807, 0.3198209245413959, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.5000000000000001, 0.0, 0.5], ['sq', 2, 0.5, 0.0, 0.5]], [['cz', 0, 1]], [['sq', 0, 0.5, -0.5358418739675712, 0.5], ['sq', 1, 0.5, 0.7403481845821607, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.5, 1.0, 0.5], ['sq', 2, 0.5, -0.9109881735517429, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.5000000000000001, -0.4109881735517429, 0.5], ['sq', 0, 3.3510915571587396e-17, 0.12485370041582833, 0.5]]], 'cz': 22, 'residual': 2.089038344358335e-15, 'source': 'direct_search\\candidate_7_3_22cz.cirq.json'}, {'n': 7, 'k': 3, 'qubits': [[3, 5], [4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7]], 'moments': [[['sq', 0, 4.026550121215985e-17, -0.15157534350475665, -0.5], ['sq', 1, 0.5000000000000001, 0.482723358479761, 0.5], ['sq', 2, 0.23886579606389338, -0.9512380823916957, 0.5], ['sq', 3, 0.5, 1.0, 0.5], ['sq', 4, 0.5272951270854374, -0.3229505379680875, 0.5], ['sq', 5, 0.33101002956971337, -0.8450384888091407, 0.5], ['sq', 6, 0.46519921011652454, -0.9895203639014174, 0.5]], [['cz', 4, 5]], [['sq', 4, 0.27207095110843904, -0.06618040179862028, 0.5], ['sq', 5, 0.5556458868541129, 0.41510310269895445, 0.5]], [['cz', 4, 1], ['cz', 5, 6]], [['sq', 4, 1.0, 0.0, -0.07623350264259354], ['sq', 1, 0.3514838800411226, 0.09361201765563676, 0.5], ['sq', 5, 0.49999999999999983, 0.4231031295975568, -0.5], ['sq', 6, 0.35076045514547877, 0.7163715908662196, 0.5]], [['cz', 6, 3], ['cz', 1, 2], ['cz', 5, 4]], [['sq', 6, 0.49999999999999994, 0.0, 0.5], ['sq', 3, 0.5, -0.9166839853291973, 0.5], ['sq', 1, 0.9900041759732306, -0.4843137578171919, -0.5], ['sq', 2, 0.7711182708497809, 0.9337557392109144, 0.5], ['sq', 5, 0.3272429637449465, 0.07953304887928203, 0.5], ['sq', 4, 0.25000000000000006, 0.0, 0.5]], [['cz', 0, 1], ['cz', 4, 5]], [['sq', 0, 0.23697080170306567, 0.05770345374842967, 0.5], ['sq', 1, 0.32869747159644436, -0.26195256935187294, 0.5], ['sq', 4, 0.7500000000000001, -0.06549949150411205, 0.5], ['sq', 5, 5.085226484029666e-17, -0.04072545823755691, -0.5]], [['cz', 1, 2], ['cz', 6, 5]], [['sq', 1, 0.39753027189345563, 0.1516223944224312, 0.5], ['sq', 2, 0.7719260562178989, -0.4731997924689182, 0.5], ['sq', 6, 0.5000000000000001, -0.06961232067008205, 0.5], ['sq', 5, 0.5, 0.0, 0.5]], [['cz', 2, 3], ['cz', 5, 4]], [['sq', 2, 0.673420111688748, 0.6906617979582457, 0.5], ['sq', 3, 1.0, 0.0, -0.722739214485858], ['sq', 5, 0.5, 0.8239795872639557, 0.5], ['sq', 4, 1.0, 0.0, -0.16825483622922666]], [['cz', 1, 2], ['cz', 5, 6]], [['sq', 1, 0.6897931699087418, -0.3496653358174173, 0.5], ['sq', 2, 0.6530606501486816, 0.14261179475289953, 0.5], ['sq', 5, 8.978670961734652e-18, -0.09498976821829697, -0.5000000000000002], ['sq', 6, 1.0, 0.0, -0.42031125081221177]], [['cz', 3, 2]], [['sq', 3, 0.5000000000000001, 1.0, 0.5], ['sq', 2, 0.9325692279808523, 0.1523869309974868, -0.5]], [['cz', 1, 2]], [['sq', 1, 0.2739602599988857, 0.30591728890887104, 0.5], ['sq', 2, 0.4834388013723244, -0.20505524041168566, 0.5]], [['cz', 1, 0], ['cz', 2, 3]], [['sq', 1, 0.6653920436706233, 0.061100452762441915, 0.5], ['sq', 0, 0.6968176325830696, -0.9925451550285065, 0.5], ['sq', 2, 1.0, 0.0, -0.34465792705912524], ['sq', 3, 0.5, 0.20776190192448496, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.7439872562978573, -0.587046735374426, 0.49999999999999994], ['sq', 0, 0.6168537417567912, 0.3798697858654325, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.5000000000000001, 0.0, 0.5], ['sq', 2, 0.5, 0.0, 0.5]], [['cz', 0, 1]], [['sq', 0, 0.49999999999999994, -0.6605817466339126, 0.5], ['sq', 1, 0.5, 0.8259962754617947, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.5, 1.0, 0.5], ['sq', 2, 0.5, 0.958356468729674, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.5, -0.541643531270326, 0.5], ['sq', 0, 1.2305413739883816e-17, 0.1189382153635864, -0.5000000000000002]]], 'cz': 23, 'residual': 2.218076107525605e-15, 'source': 'direct_search\\candidate_7_3_23cz.cirq.json'}], (8, 4): [{'n': 8, 'k': 4, 'qubits': [[4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5], [6, 3], [6, 4]], 'moments': [[['sq', 0, 0.5000000000000001, 334.19900748501783, 0.5], ['sq', 1, 0.5000000000000002, -46.729634474536766, 0.5], ['sq', 2, 0.5, -1.0, 0.5], ['sq', 3, 0.38157135719845003, -2.6287036779257007e-16, 0.5], ['sq', 4, 0.5, -1.0, 0.5], ['sq', 5, 0.49999999999999994, -1.0, 0.5], ['sq', 6, 0.49999999999999994, -175.8646824293776, 0.5], ['sq', 7, 0.5, -0.5, 0.5]], [['cz', 0, 1]], [['sq', 0, 426.6734797534457, 1271.5290474887195, 0.5], ['cz', 1, 4]], [['cz', 0, 3], ['sq', 1, -1.0, -369.0919005639051, 0.5], ['sq', 4, 0.5, -2870.2052182402376, 0.5]], [['sq', 0, 0.5, 1.4105438124889669, 0.5], ['sq', 3, 1.5, -594.4742942203508, 0.5], ['cz', 1, 2]], [['cz', 3, 6], ['sq', 1, 0.5000000000000001, 1.800957539780115e-17, 0.5], ['sq', 2, 0.5, -1019.4001780899035, 0.5]], [['sq', 3, 0.32158739372375184, 0.7790991158233053, 0.5], ['sq', 6, 310.33457894421707, 528.9571789265273, 0.5], ['cz', 0, 1]], [['cz', 6, 3], ['sq', 0, 0.3040867239846963, 1470.8319969040817, 0.5], ['sq', 1, 0.8040867239846964, -1.0, 0.5]], [['cz', 1, 0], ['cz', 3, 4]], [['sq', 1, 0.5000000000000002, -0.5, 0.5], ['sq', 0, -1.0, 1471.0819969040815, 0.5], ['sq', 3, 0.34704221097538546, -27.250000000000004, 0.5], ['sq', 4, 0.5, -865.9017426483061, 0.5]], [['cz', 1, 2], ['cz', 5, 4]], [['sq', 1, 0.75, -1.0, 0.5], ['sq', 2, 0.5, 898.7377959146387, 0.5], ['sq', 5, -1.5, 865.9017426483061, 0.5], ['sq', 4, 0.5, -0.5, 0.5]], [['cz', 1, 2], ['cz', 3, 4]], [['sq', 1, 0.5, -0.49999999999999994, 0.5], ['sq', 2, -1.0000000000000002, 898.7377959146387, 0.5], ['sq', 4, 0.5, 2223.948355692352, 0.5], ['cz', 3, 6]], [['cz', 1, 0], ['cz', 4, 5], ['sq', 3, 0.49999999999999994, -2.5933950436548905e-17, 0.5], ['sq', 6, 1.2361429389567886, 28.227185525828506, 0.5]], [['sq', 1, 0.5, -1.0, 0.5], ['sq', 0, 0.5, 4.7390340801119747e-17, 0.5], ['sq', 4, -99.00000000000001, -2224.0516443076485, 0.5], ['sq', 5, 0.5000000000000001, -1.0, 0.5], ['cz', 3, 6]], [['sq', 3, 0.1959132760153036, 1.0, 0.5], ['sq', 6, -0.3040867239846964, 1.25, 0.5]], [['cz', 3, 4]], [['sq', 3, 0.49999999999999994, -2.0, 0.5], ['sq', 4, 24.25, 159.36704601079956, 0.5]], [['cz', 4, 3]], [['sq', 4, 1.0, 158.86704601079956, 0.5], ['cz', 6, 3]], [['sq', 6, 0.5, 0.45740660163505914, 0.5], ['sq', 3, 0.5, -1.0, 0.5]], [['cz', 3, 6]], [['sq', 6, 18.0, -0.9574066016350592, 0.5], ['cz', 4, 3]], [['sq', 4, 0.5, -13.965601167392037, 0.5]], [['cz', 7, 4]], [['sq', 7, -13.465601167392036, -546.5189596561949, 0.5], ['sq', 4, 0.5, -305.3675817638942, 0.5]], [['cz', 7, 6], ['cz', 4, 1]], [['sq', 7, -1.0, -547.8598484625082, 0.5], ['sq', 6, 0.5, 0.1591111936867182, 0.5], ['sq', 4, -1.0, -305.8675817638942, 0.5], ['sq', 1, 0.5, -0.5, 0.5]], [['cz', 4, 3], ['cz', 1, 0]], [['sq', 4, 0.5000000000000001, 1.0, 0.5], ['sq', 3, 0.5000000000000001, 0.15911119368671817, 0.5], ['sq', 1, 0.49999999999999994, 0.6591111936867182, 0.5], ['sq', 0, 0.5, 0.1591111936867182, 0.5]], [['cz', 4, 5], ['cz', 1, 2]], [['sq', 4, 0.5000000000000001, 0.6591111936867182, 0.5], ['sq', 5, 0.5, 0.1591111936867182, 0.5], ['sq', 2, 0.5, 0.15911119368671822, 0.5]]], 'cz': 29, 'residual': 9.027192022015129e-13, 'source': 'local_prune\\candidate_8_4_29cz_sq.cirq.json'}, {'n': 8, 'k': 4, 'qubits': [[4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5], [6, 3], [6, 4]], 'moments': [[['sq', 0, 1.0, -2.666666841446821, 0.5], ['sq', 1, 0.5000000000000001, 0.9006708950338593, 0.5], ['sq', 2, 0.5, -1.499999999999999, 0.5], ['sq', 3, 0.5, 0.5366082371567077, 0.5], ['sq', 4, 0.5, -0.9436285020575691, 0.5], ['sq', 5, 0.5, -1.0, 0.5], ['sq', 6, 0.5, -2.66244818818997, 0.5], ['sq', 7, 0.5000000000000001, 2.4236338191442246, 0.5]], [['cz', 0, 1]], [['sq', 0, 2.7815168817445013, -8.5, 0.5], ['cz', 1, 4]], [['cz', 0, 3], ['sq', 1, -0.5148399988034192, 4.4593318750267565, 0.5], ['sq', 4, 1.4666719890476991, 5.655122928890573, 0.5]], [['sq', 0, 1.1254015258841572, 0.5000000000000001, 0.5], ['cz', 3, 6], ['cz', 1, 2]], [['sq', 3, 0.35436630366582683, -0.6402188990037717, 0.5], ['sq', 6, 0.8131920051538619, -0.008772269935714757, 0.5], ['sq', 1, 0.6959132760153036, 1.1857804378766844e-17, 0.5], ['sq', 2, 1.9769532836069301, -4.500000000000001, 0.5]], [['cz', 6, 3], ['cz', 0, 1]], [['sq', 3, -0.5, 1.5, 0.5], ['sq', 0, 0.5, -2.519002068282071, 0.5], ['sq', 1, 0.3040867239846964, 0.5, 0.5]], [['cz', 1, 0], ['cz', 3, 4]], [['sq', 0, -2.0, -1.980997931717929, 0.5], ['sq', 3, 0.5000000000000001, -3.2500000000000004, 0.5], ['sq', 4, 0.49999999999999994, -3.0658480624808258, 0.5], ['cz', 1, 2]], [['cz', 5, 4], ['sq', 1, 0.75, -1.0000000000000002, 0.5], ['sq', 2, 0.5000000000000001, -6.91238022627052e-17, 0.5]], [['sq', 5, 0.5, 2.5658480624808258, 0.5], ['sq', 4, 0.8040867239846964, -0.5, 0.5], ['cz', 1, 2]], [['cz', 3, 4], ['sq', 1, 0.75, -0.5000000000000001, 0.5], ['sq', 2, -0.49999999999999994, 1.5846204109109305e-17, 0.5]], [['sq', 4, 0.5, -1.0, 0.5], ['cz', 1, 0], ['cz', 3, 6]], [['cz', 4, 5], ['sq', 1, 0.5, -6.721649171193046e-17, 0.5], ['sq', 0, 0.5, -0.5000000000000001, 0.5], ['sq', 3, -0.5, 1.0, 0.5], ['sq', 6, 1.5, 2.666666666666667, 0.5]], [['cz', 3, 6]], [['sq', 3, 1.5, 1.0, 0.5], ['sq', 6, 0.49999999999999994, -1.0, 0.5]], [['cz', 3, 4]], [['sq', 3, 0.5000000000000001, -1.0, 0.5], ['sq', 4, 0.75, -0.5000000000000001, 0.5]], [['cz', 4, 3]], [['cz', 6, 3]], [['sq', 6, 0.8040867239846964, -1.0571715100448908, 0.5], ['sq', 3, 0.5, -0.5, 0.5]], [['cz', 3, 6]], [['sq', 3, 1.25, -0.5, 0.5], ['sq', 6, 1.0000000000000002, -1.0571715100448908, 0.5]], [['cz', 4, 3]], [['sq', 4, -0.5, -1.8574995682636801, 0.5]], [['cz', 4, 1]], [['sq', 4, -1.0, -1.8574995682636801, 0.5]], [['cz', 7, 4]], [['sq', 4, 0.5, 0.5, 0.5], ['cz', 7, 6]], [['cz', 4, 1], ['sq', 7, -1.0, 1.0977607716695863, 0.5], ['sq', 6, 0.5000000000000001, -0.32587304747463824, 0.5]], [['sq', 4, 0.5000000000000001, 2.1585623275700458e-17, 0.5], ['sq', 1, 0.5, -0.32587304747463824, 0.5]], [['cz', 4, 3], ['cz', 1, 0]], [['sq', 4, 0.5000000000000001, 1.174126952525362, 0.5], ['sq', 3, 0.5, -0.32587304747463824, 0.5], ['sq', 0, 0.49999999999999994, -0.32587304747463824, 0.5], ['cz', 1, 2]], [['cz', 4, 5], ['sq', 2, 0.5, -0.32587304747463824, 0.5]], [['sq', 5, 0.5, -0.3258730474746382, 0.5]]], 'cz': 30, 'residual': 1.7249703312307386e-15, 'source': 'local_prune\\candidate_8_4_30cz_sq.cirq.json'}, {'n': 8, 'k': 4, 'qubits': [[3, 5], [3, 6], [4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7]], 'moments': [[['sq', 0, 0.5, 13.238772182139625, 0.5], ['sq', 1, 0.5000000000000001, -4.947137167066211, 0.5], ['sq', 2, 0.3373083465539407, 0.4999999999999998, 0.5], ['sq', 3, 0.24999999999999997, 1.0000000000000002, 0.5], ['sq', 4, 0.5000000000000001, -1.0, 0.5], ['sq', 5, 0.3648337925259496, 0.5000000000000001, 0.5], ['sq', 6, 0.39695342131456696, -1.0, 0.5], ['sq', 7, 0.6285082223277061, -1.0, 0.5]], [['cz', 5, 6]], [['sq', 6, 0.34719543335082964, -1.5, 0.5], ['cz', 5, 2]], [['cz', 6, 7], ['sq', 5, -1.5, -1.2260245808659396, 0.5], ['cz', 2, 3]], [['sq', 6, 0.5, 0.12907995580997594, 0.5], ['sq', 7, 0.2727792189844248, 32.74664207904849, 0.5], ['sq', 2, 0.5, 0.19591327601530362, 0.5], ['sq', 3, 0.5774503045752932, -0.49999999999999983, 0.5]], [['cz', 6, 5], ['cz', 7, 4]], [['sq', 6, 0.6909541236524172, -0.25, 0.5], ['sq', 5, 0.3823556079069398, 0.37337585721442496, 0.5], ['sq', 7, 0.5000000000000001, 0.5, 0.5], ['sq', 4, 0.5, -59.44425150035996, 0.5]], [['cz', 3, 4], ['cz', 5, 6]], [['sq', 3, 0.5, 1.4369335816434734, 0.5], ['sq', 4, -1.0, -27.197609421311473, 0.5], ['sq', 5, 0.5, -0.5, 0.5], ['cz', 7, 6]], [['cz', 2, 3], ['sq', 7, 0.8393092936803588, -1.0, 0.5], ['sq', 6, 0.5000000000000001, 0.5, 0.5]], [['sq', 2, 0.6666666666666666, 14.366399381754048, 0.5], ['sq', 3, 1.6359914822683863, 0.8337713376832331, 0.5], ['cz', 6, 5]], [['cz', 4, 3], ['sq', 5, 0.5, 4.091918200920406e-17, 0.5], ['cz', 6, 7]], [['sq', 4, 0.49999999999999994, -1.0, 0.5], ['sq', 3, 1.4570986387292764, 0.06188759519792074, 0.5]], [['cz', 2, 3]], [['sq', 3, 0.5, 1.5, 0.5], ['cz', 2, 0]], [['cz', 3, 4], ['sq', 2, 0.3040867239846963, 0.25, 0.5], ['sq', 0, 7.289482290752495, 18.564500924359287, 0.5]], [['sq', 3, -1.5, 1.5, 0.5], ['sq', 4, 0.5, -1.0, 0.5], ['cz', 2, 0]], [['cz', 4, 7], ['sq', 2, 0.5000000000000001, 19.809893615126242, 0.5], ['sq', 0, 0.5000000000000001, -20.809893615126242, 0.5]], [['sq', 4, -0.5000000000000001, -1.0, 0.5], ['cz', 2, 3]], [['sq', 2, 0.5, 3.122892804227693e-18, 0.5], ['sq', 3, 0.49999999999999994, -0.5, 0.5]], [['cz', 0, 2]], [['sq', 0, 0.5000000000000001, -1.0000000000000002, 0.5], ['sq', 2, 0.5, -20.809893615126242, 0.5]], [['cz', 2, 3]], [['sq', 2, 0.5, 1.0000000000000002, 0.5], ['sq', 3, 0.5000000000000001, -0.49999999999999983, 0.5]], [['cz', 2, 0], ['cz', 3, 6]], [['sq', 0, -0.5, -0.9999999999999999, 0.5], ['cz', 3, 4], ['cz', 2, 5]], [['sq', 3, 0.5000000000000001, -1.0, 0.5], ['sq', 2, 0.5, -16.97290179418562, 0.5]], [['cz', 0, 2]], [['sq', 0, 0.5, -1.0, 0.5], ['sq', 2, 4.05292212614211e-17, 15.97290179418562, 0.5]], [['cz', 1, 0]], [['sq', 0, 0.5, 0.03550983184268789, 0.5], ['cz', 1, 3]], [['cz', 0, 2], ['sq', 1, -1.0, -3.911627335223523, 0.5], ['sq', 3, 0.5, 0.035509831842687925, 0.5]], [['sq', 2, 0.5, 0.03550983184268791, 0.5], ['cz', 3, 4]], [['cz', 2, 5], ['sq', 4, 0.5, 0.03550983184268794, 0.5], ['cz', 3, 6]], [['sq', 5, 0.5, 0.035509831842687904, 0.5], ['cz', 4, 7], ['sq', 6, 0.5, 0.03550983184268792, 0.5]], [['sq', 7, 0.5, 0.0355098318426879, 0.5]]], 'cz': 33, 'residual': 7.229191963882663e-15, 'source': 'local_prune\\candidate_8_4_33cz_extension0_sq.cirq.json'}, {'n': 8, 'k': 4, 'qubits': [[4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5], [6, 3], [6, 4]], 'moments': [[['sq', 0, 0.21859111351161822, -0.5000000000000016, 0.5], ['sq', 1, 0.5, -5991.363961377644, 0.5], ['sq', 2, 0.5, -1.0, 0.5], ['sq', 3, 0.6459721624073824, -2.0, 0.5], ['sq', 4, 0.5, -2516.846488170638, 0.5], ['sq', 5, 0.5, -0.5, 0.5], ['sq', 6, 0.6959132760153036, -4.3401850930671934e-17, 0.5], ['sq', 7, 0.5, -0.5, 0.5]], [['cz', 0, 1]], [['sq', 0, -1.0038758162973733, -2.0000000000000013, 0.5], ['cz', 1, 4]], [['cz', 0, 3], ['sq', 1, 0.5228141721046287, -649.6702398776288, 0.5], ['sq', 4, 1614.5060204698057, 6550.344745910827, 0.5]], [['sq', 0, 0.34546860330888607, -1.0, 0.5], ['sq', 3, -1.153912167686454, 6822.407375740574, 0.5], ['cz', 1, 2]], [['cz', 6, 3], ['sq', 1, 0.7730855641139139, -0.12687635107644896, 0.5], ['sq', 2, 0.5, -649.9042862624343, 0.5]], [['sq', 6, 0.19591327601530362, 0.5000000000000001, 0.5], ['sq', 3, -1.0000000000000002, 6822.407375740574, 0.5], ['cz', 0, 1]], [['cz', 3, 4], ['sq', 0, 0.5, 0.5000000000000001, 0.5], ['sq', 1, 0.2858525318292365, -59.09692417965118, 0.5]], [['sq', 3, 0.5, 24901.491702879117, 0.5], ['sq', 4, 0.24063253699046155, 3670.7500000000005, 0.5], ['cz', 1, 0]], [['cz', 3, 4], ['sq', 0, 0.3213834548125703, 0.75, 0.5], ['cz', 1, 2]], [['sq', 3, -0.9999999999999996, 24901.741702879117, 0.5], ['sq', 1, 0.75, -1.0000000000000002, 0.5], ['sq', 2, -1.0, -1299.0932286993045, 0.5]], [['cz', 3, 6], ['cz', 2, 5]], [['sq', 3, 0.5000000000000001, -0.5, 0.5], ['sq', 6, 0.24999999999999992, -1.0000000000000002, 0.5], ['sq', 2, 0.5, 18021.71484241765, 0.5], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 4, 0.5, -1.0, 0.5], ['sq', 5, 0.5, -0.5, 0.5]], [['sq', 1, 0.5, -0.5, 0.5], ['sq', 2, -1.0, 18020.71484241765, 0.5], ['cz', 3, 4]], [['cz', 1, 0], ['sq', 3, 0.5, -1343.4996989341191, 0.5], ['sq', 4, 0.5, -0.5000000000000001, 0.5]], [['sq', 1, 0.5, -0.5, 0.5], ['sq', 0, 0.5, -1.7476160733215138e-17, 0.5], ['cz', 6, 3], ['cz', 7, 4]], [['sq', 6, 0.5, -0.5, 0.5], ['sq', 3, -1.0, -1342.4996989341191, 0.5], ['sq', 4, 0.5, -13358.71404856638, 0.5]], [['cz', 7, 6], ['cz', 4, 1]], [['sq', 7, 0.5, 0.6936596704594525, 0.5], ['sq', 6, 0.5, 0.19365967045945254, 0.5], ['sq', 4, -1.0, -13359.21404856638, 0.5], ['sq', 1, 0.5, -0.5, 0.5]], [['cz', 4, 3], ['cz', 1, 0]], [['sq', 4, 0.5, -0.30634032954054746, 0.5], ['sq', 3, 0.5, 0.19365967045945254, 0.5], ['sq', 1, 0.5, 0.6936596704594525, 0.5], ['sq', 0, 0.5, 0.19365967045945254, 0.5]], [['cz', 4, 5], ['cz', 1, 2]], [['sq', 5, 0.5, 0.19365967045945254, 0.5], ['sq', 2, 0.5, 0.19365967045945254, 0.5]]], 'cz': 24, 'residual': 3.87457269419548e-12, 'source': 'completion_search\\native_candidates\\candidate_8_4_24cz_sq.cirq.json'}]}

for _fk,_fv in _FINAL_EXTRA.items():
    if _fk[0] in BALANCED_TEMPLATES and _fk[1]*2==_fk[0]: BALANCED_TEMPLATES[_fk[0]].extend(_fv)

import heapq as _idea_heapq
import itertools as _idea_itertools
import math as _idea_math
import time as _idea_time
import cirq as _idea_cirq
import cirq_google as _idea_cirq_google
import networkx as _idea_nx
import numpy as _idea_np

def _idea_native_x(qubit: _idea_cirq.Qid):
    """Return a Pauli-X gate expressed as a Willow-native PhasedXZGate."""
    return _idea_cirq.PhasedXZGate(x_exponent=1.0, z_exponent=0.0, axis_phase_exponent=0.0)(qubit)

def _idea_native_h(qubit: _idea_cirq.Qid):
    """Return a Hadamard gate expressed as a Willow-native PhasedXZGate."""
    return _idea_cirq.PhasedXZGate(x_exponent=0.5, z_exponent=1.0, axis_phase_exponent=1.5)(qubit)

def _idea_native_ry(qubit: _idea_cirq.Qid, theta: float):
    """Return an Ry(theta) rotation expressed as a Willow-native PhasedXZGate."""
    return _idea_cirq.PhasedXZGate(x_exponent=theta / _idea_np.pi, z_exponent=0.0, axis_phase_exponent=0.5)(qubit)

def _idea_native_cnot(control: _idea_cirq.Qid, target: _idea_cirq.Qid):
    """Yield a CNOT decomposed into Willow-native CZ and PhasedXZ gates.

    CNOT(c, t) = (I x H_t) . CZ(c, t) . (I x H_t)
    """
    yield _idea_native_h(target)
    yield _idea_cirq.CZ(control, target)
    yield _idea_native_h(target)

def _idea_build_cry_core(q_ctrl: _idea_cirq.Qid, q_target: _idea_cirq.Qid, theta: float):
    """Decompose a controlled Ry(2 * theta) using two CNOTs and two rotations.

    When q_ctrl is 1, applies Ry(2 * theta) to q_target.
    """
    yield _idea_native_ry(q_target, theta)
    yield _idea_native_cnot(q_ctrl, q_target)
    yield _idea_native_ry(q_target, -theta)
    yield _idea_native_cnot(q_ctrl, q_target)

def _idea_build_gate_i(q_ctrl: _idea_cirq.Qid, q_target: _idea_cirq.Qid, theta: float):
    """Build gate (i) from BÃ¤rtschi and Eidenbenz.

    Applies controlled Ry(2 * theta) with q_target as the control and q_ctrl
    as the target, conjugated by CNOT(q_ctrl, q_target).
    """
    yield _idea_native_cnot(q_ctrl, q_target)
    yield _idea_build_cry_core(q_target, q_ctrl, theta)
    yield _idea_native_cnot(q_ctrl, q_target)

def _idea_build_ccry_core(q_ctrl1: _idea_cirq.Qid, q_ctrl2: _idea_cirq.Qid, q_target: _idea_cirq.Qid, theta: float):
    """Decompose CCRy(2*theta) using four Ry rotations and four CNOTs."""
    yield _idea_native_ry(q_target, theta / 2)
    yield _idea_native_cnot(q_ctrl2, q_target)
    yield _idea_native_ry(q_target, -theta / 2)
    yield _idea_native_cnot(q_ctrl1, q_target)
    yield _idea_native_ry(q_target, theta / 2)
    yield _idea_native_cnot(q_ctrl2, q_target)
    yield _idea_native_ry(q_target, -theta / 2)
    yield _idea_native_cnot(q_ctrl1, q_target)

def _idea_build_gate_ii(q_top: _idea_cirq.Qid, q_mid: _idea_cirq.Qid, q_bot: _idea_cirq.Qid, theta: float):
    """Build gate (ii)_l from Figure 3 of BÃ¤rtschi and Eidenbenz.

    Acts on the triple (m-l-1, m-l, m-1) = (q_top, q_mid, q_bot).
    """
    yield _idea_native_cnot(q_top, q_bot)
    yield _idea_build_ccry_core(q_ctrl1=q_mid, q_ctrl2=q_bot, q_target=q_top, theta=theta)
    yield _idea_native_cnot(q_top, q_bot)

def _idea_build_scs_block(path_qubits: list[_idea_cirq.Qid], m: int, k: int):
    """Construct the SCS_{m,k} block acting on path_qubits[:m]."""
    sub_q = path_qubits[:m]
    theta_i = _idea_np.arccos(1.0 / _idea_np.sqrt(m))
    yield _idea_build_gate_i(sub_q[m - 2], sub_q[m - 1], theta_i)
    for l in range(2, min(k, m - 1) + 1):
        q_top = sub_q[m - l - 1]
        q_mid = sub_q[m - l]
        q_bot = sub_q[m - 1]
        theta_l = _idea_np.arccos(_idea_np.sqrt(l / m))
        yield _idea_build_gate_ii(q_top, q_mid, q_bot, theta_l)

class _idea_MultiModePathfinder:
    """Find a low-error connected qubit chain on the Google Willow grid."""

    def __init__(self, mode: str='branch_and_bound'):
        self.mode = mode

    def calculate_cz_counts(self, N: int, n: int) -> dict:
        """Count the emitted CZ gates on each relative path coupler."""
        cz_counts = {i: 0 for i in range(N - 1)}
        logical_qubits = list(_idea_cirq.LineQubit.range(N))
        index_by_qubit = {qubit: index for index, qubit in enumerate(logical_qubits)}
        circuit = _idea_build_full_u_nk(logical_qubits, N, n)
        for operation in circuit.all_operations():
            if operation.gate != _idea_cirq.CZ:
                continue
            first, second = (index_by_qubit[qubit] for qubit in operation.qubits)
            if abs(first - second) != 1:
                raise ValueError(f'non-nearest CZ remained between {first} and {second}')
            cz_counts[min(first, second)] += 1
        return cz_counts

    def compute_path_cost(self, path: list, sq_errors: dict, cz_errors: dict, readout_errors: dict, cz_counts: dict) -> float:
        """Compute the total additive logarithmic noise cost of a path."""
        cost = 0.0
        for q in path:
            sq_e = sq_errors.get(q, 0.001)
            ro_e = readout_errors.get(q, 0.01)
            cost += -_idea_np.log(max(1.0 - sq_e, 1e-06))
            cost += -_idea_np.log(max(1.0 - ro_e, 1e-06))
        for i in range(len(path) - 1):
            u, v = (path[i], path[i + 1])
            cz_e = cz_errors.get((u, v), cz_errors.get((v, u), 0.01))
            activity = cz_counts.get(i, 1)
            cost += activity * -_idea_np.log(max(1.0 - cz_e, 1e-06))
        return cost

    def find_path(self, qubits: list, couplers: list, N: int, n: int, sq_errors: dict, cz_errors: dict, readout_errors: dict, override_mode: str | None=None):
        """Dispatch to the selected pathfinding strategy."""
        search_mode = override_mode or self.mode
        cz_counts = self.calculate_cz_counts(N, n)
        adjacency = {q: [] for q in qubits}
        for u, v in couplers:
            if u in adjacency and v in adjacency:
                adjacency[u].append(v)
                adjacency[v].append(u)
        search_args = (qubits, adjacency, N, sq_errors, cz_errors, readout_errors, cz_counts)
        if search_mode == 'branch_and_bound':
            return self._branch_and_bound(*search_args)
        if search_mode == 'dijkstra':
            return self._dijkstra(*search_args)
        if search_mode == 'geometric_snake':
            return self._geometric_snake(*search_args)
        raise ValueError(f'Unknown pathfinding mode: {search_mode}')

    def _branch_and_bound(self, qubits, adjacency, N, sq_errors, cz_errors, readout_errors, cz_counts):
        best_path = None
        best_cost = float('inf')

        def dfs(current_path, current_cost, visited):
            nonlocal best_path, best_cost
            if current_cost >= best_cost:
                return
            if len(current_path) == N:
                best_cost = current_cost
                best_path = list(current_path)
                return
            last_q = current_path[-1]
            step_idx = len(current_path) - 1
            neighbors = [neighbor for neighbor in adjacency[last_q] if neighbor not in visited]
            neighbors.sort(key=lambda neighbor: cz_errors.get((last_q, neighbor), cz_errors.get((neighbor, last_q), 0.01)))
            for neighbor in neighbors:
                sq_e = sq_errors.get(neighbor, 0.001)
                ro_e = readout_errors.get(neighbor, 0.01)
                cz_e = cz_errors.get((last_q, neighbor), cz_errors.get((neighbor, last_q), 0.01))
                step_cost = -_idea_np.log(max(1.0 - sq_e, 1e-06)) - _idea_np.log(max(1.0 - ro_e, 1e-06)) + cz_counts.get(step_idx, 1) * -_idea_np.log(max(1.0 - cz_e, 1e-06))
                visited.add(neighbor)
                current_path.append(neighbor)
                dfs(current_path, current_cost + step_cost, visited)
                current_path.pop()
                visited.remove(neighbor)
        for start_q in qubits:
            sq_e = sq_errors.get(start_q, 0.001)
            ro_e = readout_errors.get(start_q, 0.01)
            initial_cost = -_idea_np.log(max(1.0 - sq_e, 1e-06)) - _idea_np.log(max(1.0 - ro_e, 1e-06))
            dfs([start_q], initial_cost, {start_q})
        return (best_path, best_cost)

    def _dijkstra(self, qubits, adjacency, N, sq_errors, cz_errors, readout_errors, cz_counts):
        priority_queue = []
        path_counter = 0
        for start_q in qubits:
            sq_e = sq_errors.get(start_q, 0.001)
            ro_e = readout_errors.get(start_q, 0.01)
            initial_cost = -_idea_np.log(max(1.0 - sq_e, 1e-06)) - _idea_np.log(max(1.0 - ro_e, 1e-06))
            _idea_heapq.heappush(priority_queue, (initial_cost, path_counter, (start_q,), {start_q}))
            path_counter += 1
        while priority_queue:
            cost, _, path, visited = _idea_heapq.heappop(priority_queue)
            if len(path) == N:
                return (list(path), cost)
            last_q = path[-1]
            step_idx = len(path) - 1
            for neighbor in adjacency[last_q]:
                if neighbor in visited:
                    continue
                sq_e = sq_errors.get(neighbor, 0.001)
                ro_e = readout_errors.get(neighbor, 0.01)
                cz_e = cz_errors.get((last_q, neighbor), cz_errors.get((neighbor, last_q), 0.01))
                step_cost = -_idea_np.log(max(1.0 - sq_e, 1e-06)) - _idea_np.log(max(1.0 - ro_e, 1e-06)) + cz_counts.get(step_idx, 1) * -_idea_np.log(max(1.0 - cz_e, 1e-06))
                new_visited = set(visited)
                new_visited.add(neighbor)
                _idea_heapq.heappush(priority_queue, (cost + step_cost, path_counter, path + (neighbor,), new_visited))
                path_counter += 1
        return (None, float('inf'))

    def _geometric_snake(self, qubits, adjacency, N, sq_errors, cz_errors, readout_errors, cz_counts):
        best_path = None
        best_cost = float('inf')

        def search_snake(path, visited):
            nonlocal best_path, best_cost
            if len(path) == N:
                cost = self.compute_path_cost(path, sq_errors, cz_errors, readout_errors, cz_counts)
                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path)
                return
            last_q = path[-1]
            for neighbor in adjacency[last_q]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                search_snake(path, visited)
                path.pop()
                visited.remove(neighbor)
        for qubit in qubits:
            search_snake([qubit], {qubit})
        return (best_path, best_cost)

def _idea_build_unrouted_u_nk(path_qubits: list[_idea_cirq.Qid], N: int, n: int) -> _idea_cirq.Circuit:
    """Construct U_{N,n} before enforcing hardware connectivity."""
    circuit = _idea_cirq.Circuit()
    for qubit in path_qubits[N - n:]:
        circuit.append(_idea_native_x(qubit))
    for m in range(N, n, -1):
        circuit.append(_idea_build_scs_block(path_qubits, m=m, k=n))
    for m in range(n, 1, -1):
        circuit.append(_idea_build_scs_block(path_qubits, m=m, k=m - 1))
    return circuit

def _idea_build_full_u_nk(path_qubits: list[_idea_cirq.Qid], N: int, n: int) -> _idea_cirq.Circuit:
    """Construct and route U_{N,n} over the selected active path.

    RouteCQC tracks the evolving logical-to-physical permutation, avoiding
    the depth cost of undoing every remote interaction. Dicke states are
    permutation-invariant, so its final logical permutation needs no repair.
    """
    unrouted = _idea_build_unrouted_u_nk(path_qubits, N, n)
    path_graph = _idea_nx.Graph()
    path_graph.add_nodes_from(path_qubits)
    path_graph.add_edges_from(zip(path_qubits, path_qubits[1:]))
    initial_mapper = _idea_cirq.HardCodedInitialMapper({qubit: qubit for qubit in path_qubits})
    routed = _idea_cirq.RouteCQC(path_graph)(unrouted, initial_mapper=initial_mapper)
    return _idea_cirq.optimize_for_target_gateset(routed, gateset=_idea_cirq.CZTargetGateset(allow_partial_czs=False))

def _idea_create_dicke_state(num_qubits: int, num_excitations: int, cz_pauli_error_rate: dict[tuple[_idea_cirq.GridQubit, _idea_cirq.GridQubit], tuple[float, float]], sq_pauli_error_rate: dict[_idea_cirq.GridQubit, tuple[float, float]], zero_errors: dict[_idea_cirq.GridQubit, tuple[float, float]], one_errors: dict[_idea_cirq.GridQubit, tuple[float, float]]) -> tuple[_idea_cirq.Circuit, set[_idea_cirq.GridQubit]]:
    """Create a hardware-native circuit for an (N, n) Dicke state on Willow105.

    Returns:
        The compiled circuit and the physical GridQubits it uses.
    """
    cz_errors = {pair: error_rate for pair, (error_rate, _) in cz_pauli_error_rate.items()}
    sq_errors = {qubit: error_rate for qubit, (error_rate, _) in sq_pauli_error_rate.items()}
    readout_errors = {qubit: (zero_errors.get(qubit, (0.01, 0))[0] + one_errors.get(qubit, (0.01, 0))[0]) / 2.0 for qubit in sq_pauli_error_rate}
    active_qubits = list(sq_pauli_error_rate)
    active_couplers = list(cz_pauli_error_rate)
    pathfinder = _idea_MultiModePathfinder(mode='branch_and_bound')
    best_path, _ = pathfinder.find_path(qubits=active_qubits, couplers=active_couplers, N=num_qubits, n=num_excitations, sq_errors=sq_errors, cz_errors=cz_errors, readout_errors=readout_errors)
    if best_path is None:
        raise NotImplementedError(f'No valid connected 1D path found for N={num_qubits}')
    circuit = _idea_build_full_u_nk(path_qubits=best_path, N=num_qubits, n=num_excitations)
    circuit = _idea_cirq.drop_negligible_operations(circuit, atol=1e-08)
    return (circuit, set(best_path))

"""Willow105 Dicke preparation, verified to complex128 numerical precision.

Analytic recurrences are supplemented by independently synthesized, embedded
native seeds. No fitting, calibration maps, external files, or hardware
submission imports are needed at runtime. The public function returns a fresh
circuit and output set. See verify_bonus.py for full-state and all-branch checks.
"""
import functools as _other_functools
import collections as _other_collections
import math as _other_math
import random as _other_random
import cirq as _other_cirq
import cirq_google as _other_cirq_google
import numpy as _other_np
import sympy as _other_sympy
_other_H = _other_cirq.unitary(_other_cirq.H)
_other_X = _other_cirq.unitary(_other_cirq.X)
_other_I = _other_np.eye(2, dtype=_other_np.complex128)

def _other_ry(theta):
    c, s = (_other_math.cos(theta / 2), _other_math.sin(theta / 2))
    return _other_np.array([[c, -s], [s, c]], dtype=_other_np.complex128)

def _other__u(ops, q, matrix):
    ops.append((q, matrix))

def _other__cx(ops, a, b):
    ops.extend([(b, _other_H), (a, b), (b, _other_H)])

def _other__identity(matrix):
    return _other_np.linalg.norm(matrix - _other_np.trace(matrix) * _other_I / 2) < 2e-14

def _other__compile(ops):
    """Fuse local matrices between CZs and ASAP-pack mixed native moments.

    A CZ flushes only its two operands, so disjoint rotations and entanglers
    share moments.  Local identity up to phase can be dropped because these
    are unconditional operations.  Never apply that rule to feedforward gates.
    """
    pending = {}
    emitted = []

    def flush(q):
        matrix = pending.pop(q, None)
        if matrix is not None and (not _other__identity(matrix)):
            emitted.append(_other_cirq.PhasedXZGate.from_matrix(matrix).on(q))
    for a, b in ops:
        if isinstance(b, _other_np.ndarray):
            pending[a] = b @ pending.get(a, _other_I)
        else:
            flush(a)
            flush(b)
            emitted.append(_other_cirq.CZ(a, b))
    for q in list(pending):
        flush(q)
    return _other_cirq.Circuit(emitted, strategy=_other_cirq.InsertStrategy.EARLIEST)

def _other__optimize_frames(circuit):
    """Commute exact Pauli/Z frames through CZs, then refold local layers."""
    best = circuit
    for _ in range(3):
        c = _other_cirq.eject_z(best, atol=1e-13)
        c = _other_cirq.eject_phased_paulis(c, atol=1e-13)
        c = _other_cirq.merge_single_qubit_gates_to_phxz(c, atol=1e-13)
        c = _other_cirq.drop_negligible_operations(c, atol=1e-13)
        ops = []
        for op in c.all_operations():
            if not op.qubits:
                continue
            if len(op.qubits) == 1:
                ops.append((op.qubits[0], _other_cirq.unitary(op)))
            else:
                assert op.gate == _other_cirq.CZ
                ops.append(op.qubits)
        c = _other__compile(ops)
        if len(c) >= len(best):
            break
        best = c
    return best

def _other__schedule_commuting(circuit, trials=6):
    """Resource-constrained scheduling of an exact commutation DAG.

    Diagonal gates commute even when they share a wire. Every other shared
    wire ordering is retained, so this never approximates the unitary.
    """
    ops = list(circuit.all_operations())
    successors = [set() for _ in ops]
    predecessors = [set() for _ in ops]
    diagonal = []
    for op in ops:
        matrix = _other_cirq.unitary(op) if len(op.qubits) == 1 else None
        diagonal.append(matrix is None or abs(matrix[0, 1]) + abs(matrix[1, 0]) < 1e-13)
    by_qubit = {}
    for j, op in enumerate(ops):
        for q in op.qubits:
            for i in by_qubit.get(q, []):
                if not (diagonal[i] and diagonal[j]):
                    predecessors[j].add(i)
                    successors[i].add(j)
            by_qubit.setdefault(q, []).append(j)
    heights = [0] * len(ops)
    for i in range(len(ops) - 1, -1, -1):
        heights[i] = 1 + max((heights[j] for j in successors[i]), default=0)
    best = circuit
    for seed in range(trials):
        rng = _other_random.Random(seed)
        ranks = [rng.random() for _ in ops]
        counts = [len(p) for p in predecessors]
        ready = {i for i, n in enumerate(counts) if n == 0}
        moments = []
        while ready:
            busy, chosen = (set(), [])
            for i in sorted(ready, key=lambda i: (-heights[i], ranks[i])):
                if not busy.intersection(ops[i].qubits):
                    chosen.append(i)
                    busy.update(ops[i].qubits)
            moments.append(_other_cirq.Moment((ops[i] for i in chosen)))
            ready.difference_update(chosen)
            for i in chosen:
                for j in successors[i]:
                    counts[j] -= 1
                    if counts[j] == 0:
                        ready.add(j)
        candidate = _other_cirq.Circuit(moments)
        assert sum((len(m) for m in candidate)) == len(ops)
        if len(candidate) < len(best):
            best = candidate
    return best

def _other__optimize_depth(circuit):
    best = _other__optimize_frames(circuit)
    current = circuit
    for _ in range(4):
        current = _other_cirq.eject_z(current, atol=1e-13)
        current = _other_cirq.eject_phased_paulis(current, atol=1e-13)
        current = _other_cirq.merge_single_qubit_gates_to_phxz(current, atol=1e-13)
        current = _other_cirq.drop_negligible_operations(current, atol=1e-13)
        current = _other__schedule_commuting(current)
        if len(current) < len(best):
            best = current
    return best

@_other_functools.lru_cache(maxsize=1)
def _other__graph():
    metadata = _other_cirq_google.Willow105.metadata
    graph = {q: set() for q in metadata.qubit_set}
    for a, b in metadata.qubit_pairs:
        graph[a].add(b)
        graph[b].add(a)
    return graph

def _other__givens(ops, a, b, theta):
    """Full real excitation-preserving rotation: 10 -> cos*10 + sin*01.

    Conjugating opposite Ry rotations by H.H CZ gives this exact two-CZ
    formula, retaining the common phase on the vacuum and occupied branches.
    """
    ops.extend([(a, _other_H), (b, _other_H), (a, b), (a, _other_ry(theta)), (b, _other_ry(-theta)), (a, b), (a, _other_H), (b, _other_H)])

@_other_functools.lru_cache(maxsize=256)
def _other__givens_native(theta):
    c, s = (_other_math.cos(theta), _other_math.sin(theta))
    matrix = _other_np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]], dtype=complex)
    a, b = _other_cirq.LineQubit.range(2)
    return tuple(_other_cirq.two_qubit_matrix_to_cz_operations(a, b, matrix, allow_partial_czs=False, atol=1e-12))

def _other__reflect3(ops, a, b, c, m, ell):
    """Core five-CZ restricted isometry on path a-b-c.

    Reverse of the core module's c-a-b convention. Allowed inputs:
    000->000, 100->sqrt(ell/m)100+sqrt(1-ell/m)010,
    101->011, 110->110, 111->111. All phases are identical.
    """
    theta = _other_math.acos(_other_math.sqrt(ell / m))
    bp, ap = ((theta + _other_math.pi / 2) / 2, (theta - _other_math.pi / 2) / 2)
    _other__cx(ops, b, a)
    ops.extend([(b, _other_ry(-bp)), (c, b), (b, _other_ry(-ap)), (a, b), (b, _other_ry(ap)), (c, b), (b, _other_ry(bp))])
    _other__cx(ops, b, a)

def _other__unary3(ops, q):
    """Analytic six-CZ isometry, valid ONLY for input weights 0,1,2.

    Adapted read-only from astra/research/exact_u3.py; its domain restriction
    matters: this is not a U(3,3). The general routine handles weight three.
    """
    a = _other_math.asin(1 / _other_math.sqrt(3))
    angles = [-_other_math.pi / 2, _other_math.pi, a - _other_math.pi, -_other_math.pi / 2, a - _other_math.pi, _other_math.pi / 2 + a, a, -_other_math.pi / 2, -_other_math.pi / 2, -_other_math.pi / 2, 0, -_other_math.pi / 4, _other_math.pi / 4, _other_math.pi / 2, _other_math.pi / 2]
    edges = [(1, 2), (0, 1), (1, 2), (0, 1), (1, 2), (1, 2)]
    for i in range(3):
        _other__u(ops, q[i], _other_ry(angles[i]))
    for j, (u, v) in enumerate(edges):
        ops.append((q[u], q[v]))
        _other__u(ops, q[u], _other_ry(angles[3 + 2 * j]))
        _other__u(ops, q[v], _other_ry(angles[4 + 2 * j]))

def _other__unary_legacy(ops, q, k, small=True):
    """Coherently decode 1^j 0^(m-j) for every 0<=j<=k.

    Baertschi--Eidenbenz SCS with wire movement absorbed in the reverse
    staircase. No restoration swaps; disjoint blocks pipeline automatically.
    """
    if len(q) == 3 and k == 2 and small:
        _other__unary3(ops, q)
        return
    for start in range(len(q) - 1):
        m = len(q) - start
        if m == 3 and k == 2 and small:
            _other__unary3(ops, q[start:])
            break
        width = min(k, m - 1)
        if width == 0:
            break
        a, b = (q[start + width - 1], q[start + width])
        theta = _other_math.acos(_other_math.sqrt(width / m))
        if m > k + 1:
            eta = _other_math.pi / 2 - theta
            _other__u(ops, b, _other_ry(eta))
            _other__cx(ops, a, b)
            _other__u(ops, b, _other_ry(-eta))
            _other__cx(ops, b, a)
        else:
            _other__givens(ops, a, b, theta)
        for ell in range(width - 1, 0, -1):
            _other__reflect3(ops, *q[start + ell - 1:start + ell + 2], m, ell)

def _other__unary3_full_forward(ops, q):
    """Eight-CZ coherent U(3,3), independently synthesized and analyticized.

    All four unary columns, including 111, have the same phase.
    """
    a = _other_math.asin(1 / _other_math.sqrt(3))
    h = _other_math.pi / 2
    angles = [h, h, h, h + a, _other_math.pi - a, h, h + a / 2, -h + a / 2, h, 0, h + a / 2, -h + a / 2, -h, 3 * h, -3 * _other_math.pi / 4, h, -5 * _other_math.pi / 4, h, -h]
    for i in range(3):
        _other__u(ops, q[i], _other_ry(angles[i]))
    for j in range(8):
        u, v = (q[j % 2], q[1 + j % 2])
        ops.append((u, v))
        _other__u(ops, u, _other_ry(angles[3 + 2 * j]))
        _other__u(ops, v, _other_ry(angles[4 + 2 * j]))

def _other__unary3_full(ops, q):
    ops.extend(((x, _other_X) for x in q))
    _other__unary3_full_forward(ops, q[::-1])
    ops.extend(((x, _other_X) for x in q))

def _other__scs_pair(ops, q, m, ell):
    """Four-CZ fused SCS window, input columns 000,100,110.

    100 -> sqrt(ell/m)100 + sqrt(1-ell/m)010;
    110 -> sqrt((ell+1)/m)110 + sqrt(1-(ell+1)/m)011.
    Independently derived by real-isometry synthesis then analyticized.
    """
    a, b = (_other_math.asin(_other_math.sqrt(ell / m)), _other_math.asin(_other_math.sqrt((ell + 1) / m)))
    h = _other_math.pi / 2
    angles = [-h, _other_math.pi, h + b, -h, h + b, h + a, a - _other_math.pi, h, 0, -h, h]
    for i in range(3):
        _other__u(ops, q[i], _other_ry(angles[i]))
    for j, (u, v) in enumerate([(1, 2), (0, 1), (1, 2), (0, 1)]):
        ops.append((q[u], q[v]))
        _other__u(ops, q[u], _other_ry(angles[3 + 2 * j]))
        _other__u(ops, q[v], _other_ry(angles[4 + 2 * j]))

def _other__unary(ops, q, k, small=True):
    if small == 'fused':
        return _other__unary_fused(ops, q, k)
    if small == 'pruned':
        return _other__unary_pruned(ops, q, k)
    return _other__unary_legacy(ops, q, k, small)

def _other__unary_fused(ops, q, k):
    for start in range(len(q) - 1):
        m = len(q) - start
        if m == 3 and k >= 3:
            _other__unary3_full(ops, q[start:])
            break
        width = min(k, m - 1)
        if m > k and width >= 2:
            _other__scs_pair(ops, q[start + width - 2:start + width + 1], m, width - 1)
            for ell in range(width - 2, 0, -1):
                _other__reflect3(ops, *q[start + ell - 1:start + ell + 2], m, ell)
        else:
            a, b = (q[start + width - 1], q[start + width])
            _other__givens(ops, a, b, _other_math.acos(_other_math.sqrt(width / m)))
            for ell in range(width - 1, 0, -1):
                _other__reflect3(ops, *q[start + ell - 1:start + ell + 2], m, ell)

def _other__reflect4(ops, a, b, c, theta, skip=False):
    """Claude's four-CZ isometry in our reversed wire convention."""
    if not skip:
        _other__cx(ops, b, a)
    ops.extend([(b, _other_ry(-theta)), (a, b), (b, _other_ry(-_other_math.pi / 2)), (c, b), (b, _other_ry(theta + _other_math.pi / 2))])
    _other__cx(ops, b, a)

@_other_functools.lru_cache(maxsize=64)
def _other__classical_map(source, target):
    """Shortest adjacent-CX/X realization on a promised three-bit support."""
    queue = _other_collections.deque([(source, ())])
    seen = {source}
    gates = [(a, b) for a, b in [(0, 1), (1, 0), (1, 2), (2, 1)]]
    gates += [(a, a) for a in range(3)]
    while queue:
        state, path = queue.popleft()
        if state == target:
            return path
        for a, b in gates:
            nxt = tuple((s ^ (1 << 2 - b if a == b or s & 1 << 2 - a else 0) for s in state))
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + ((a, b),)))
    raise ValueError('No affine support realization')

def _other__unary_pruned(ops, q, k, jmin=0, jmax=None):
    """Track exact combinatorial support, with no amplitudes or state fitting."""
    jmax = min(k, len(q)) if jmax is None else jmax
    support = {tuple([1] * j + [0] * (len(q) - j)) for j in range(jmin, jmax + 1)}

    def apply(indices, mapping):
        nonlocal support
        changed = set()
        for state in support:
            local = tuple((state[i] for i in indices))
            for output in mapping.get(local, [local]):
                result = list(state)
                for i, v in zip(indices, output):
                    result[i] = v
                changed.add(tuple(result))
        support = changed
    for start in range(len(q) - 1):
        m = len(q) - start
        if m == 3:
            if k == 2:
                _other__unary3(ops, q[start:])
                break
            if k >= 3:
                _other__unary3_full(ops, q[start:])
                break
        width = min(k, m - 1)
        indices = [start + width - 1, start + width]
        a, b = [q[i] for i in indices]
        s = {tuple((state[i] for i in indices)) for state in support}
        theta = _other_math.acos(_other_math.sqrt(width / m))
        if (1, 0) in s:
            if s == {(1, 0)}:
                _other__u(ops, b, _other_ry(2 * theta))
                _other__cx(ops, b, a)
            elif (1, 1) not in s:
                eta = _other_math.pi / 2 - theta
                _other__u(ops, b, _other_ry(eta))
                _other__cx(ops, a, b)
                _other__u(ops, b, _other_ry(-eta))
                _other__cx(ops, b, a)
            else:
                _other__givens(ops, a, b, theta)
        apply(indices, {(1, 0): [(1, 0), (0, 1)]})
        for ell in range(width - 1, 0, -1):
            indices = list(range(start + ell - 1, start + ell + 2))
            a, b, c = [q[i] for i in indices]
            s = {tuple((state[i] for i in indices)) for state in support}
            theta = _other_math.acos(_other_math.sqrt(ell / m))
            p, r = ((1, 0, 0) in s, (1, 0, 1) in s)
            if p and r:
                if (1, 1, 1) not in s:
                    _other__reflect4(ops, a, b, c, theta, skip=(1, 1, 0) not in s)
                else:
                    _other__reflect3(ops, a, b, c, m, ell)
            elif p:
                _other__givens(ops, a, b, theta)
            elif r:
                source = tuple(sorted((4 * x + 2 * y + z for x, y, z in s)))
                target = tuple((3 if v == 5 else v for v in source))
                for u, v in _other__classical_map(source, target):
                    if u == v:
                        _other__u(ops, [a, b, c][u], _other_X)
                    else:
                        _other__cx(ops, [a, b, c][u], [a, b, c][v])
            apply(indices, {(1, 0, 0): [(1, 0, 0), (0, 1, 0)], (1, 0, 1): [(0, 1, 1)]})

def _other__ladder(a, b, k):
    """Find k rungs with two nonintersecting path extensions on Willow."""
    graph = _other__graph()

    def paths(prefix, length, occupied):
        if len(prefix) == length:
            yield prefix
        else:
            for nxt in sorted(graph[prefix[-1]] - occupied):
                yield from paths(prefix + [nxt], length, occupied | {nxt})
    roots = sorted(graph, key=lambda q: ((q.row - 6) ** 2 + (q.col - 7) ** 2, q))
    for root in roots:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            for sr, sc in [(-dc, dr), (dc, -dr)]:
                top = [_other_cirq.GridQubit(root.row + i * dr, root.col + i * dc) for i in range(k)]
                bottom = [_other_cirq.GridQubit(root.row + (k - 1 - i) * dr + sr, root.col + (k - 1 - i) * dc + sc) for i in range(k)]
                core = set(top + bottom)
                if len(core) != 2 * k or not core <= graph.keys():
                    continue
                edges = list(zip(top, top[1:])) + list(zip(bottom, bottom[1:]))
                edges += [(top[j], bottom[k - 1 - j]) for j in range(k)]
                if any((y not in graph[x] for x, y in edges)):
                    continue
                for full_top in paths(top, a, core):
                    full_bottom = next(paths(bottom, b, core | set(full_top)), None)
                    if full_bottom is not None:
                        return (full_top, full_bottom)
    raise RuntimeError('No Willow ladder found')

def _other__divide(n, k, a=None, small=True, pivot=0, left_first=False):
    a = n // 2 if a is None else a
    b = n - a
    top, bottom = _other__ladder(a, b, k)
    weights = [_other_math.comb(a, j) * _other_math.comb(b, k - j) for j in range(k + 1)]
    tail = [sum(weights[j:]) for j in range(k + 1)]
    ops = [(q, _other_X) for q in bottom[:k]]
    prefix = [sum(weights[:j + 1]) for j in range(k + 1)]
    _other__u(ops, top[pivot], _other_ry(2 * _other_math.asin(_other_math.sqrt(tail[pivot + 1] / tail[0]))))

    def expand_right():
        for j in range(pivot + 1, k):
            eta = _other_math.pi / 2 - _other_math.asin(_other_math.sqrt(tail[j + 1] / tail[j]))
            _other__u(ops, top[j], _other_ry(eta))
            _other__cx(ops, top[j - 1], top[j])
            _other__u(ops, top[j], _other_ry(-eta))

    def expand_left():
        for j in range(pivot - 1, -1, -1):
            eta = _other_math.pi / 2 - _other_math.asin(_other_math.sqrt(prefix[j] / prefix[j + 1]))
            _other__u(ops, top[j + 1], _other_X)
            _other__u(ops, top[j], _other_ry(eta))
            _other__cx(ops, top[j + 1], top[j])
            _other__u(ops, top[j], _other_ry(-eta))
            _other__u(ops, top[j + 1], _other_X)
            _other__u(ops, top[j], _other_X)
    if left_first:
        expand_left()
        expand_right()
    else:
        expand_right()
        expand_left()
    for j in range(k):
        _other__cx(ops, top[j], bottom[k - 1 - j])
    _other__unary(ops, top, k, small)
    _other__unary(ops, bottom, k, small)
    return (_other__compile(ops), set(top + bottom))

def _other__square():
    q = [_other_cirq.GridQubit(4, 4), _other_cirq.GridQubit(4, 5), _other_cirq.GridQubit(5, 5), _other_cirq.GridQubit(5, 4)]
    a = _other_math.acos(1 / _other_math.sqrt(3))
    ops = []
    layers = [[_other_ry(a - _other_math.pi / 2), _other_H, _other_ry(-a), _other_H], [_other_ry(_other_math.pi / 2), _other_ry(-_other_math.pi / 2) @ _other_H, _other_ry(3 * _other_math.pi / 2), _other_ry(-_other_math.pi - a) @ _other_H], [_other_I, _other_ry(-_other_math.pi), _other_ry(-_other_math.pi / 2), _other_ry(-_other_math.pi / 2)], [_other_ry(a - _other_math.pi), _other_ry(2 * a), _other_ry(a), _other_ry(a - _other_math.pi / 2)]]
    for i, layer in enumerate(layers):
        ops.extend(zip(q, layer))
        if i < 3:
            ops.extend([(q[0], q[1]), (q[2], q[3])] if i % 2 == 0 else [(q[1], q[2]), (q[3], q[0])])
    return (_other__compile(ops), set(q))
_other__NATIVE_SEEDS = {(5, 2): {'angles': [1.5901770943121394, -2.209960070660441, -0.7091601230882575, -0.3043315333387232, -2.598142427345299, -0.917169399671696, 0.41842715016396437, 1.7496392757141055, 2.1664957579788107, -3.741609447121856, -2.7336266023998292, 2.0446189031881654, -4.316243675422165, 3.110303759135296, 1.627362894082465, -2.4206903968601123, 1.701620864075715, -1.5707963267948968, -2.8410295874377143, -1.484634915615345, -2.607216994232227, 0.6798184144094802, -3.38439815282079e-16, -1.4656517427965308e-16, 3.313868466630986, 1.5707963267948966, 1.6596137066175907, -1.5707963267948966, -1.171515403577853, 2.8831554293690487, -1.5707963267948968, 3.1415926535897927, 3.141592653589794, -1.9700772500119397, -3.859892801317504e-17], 'coords': [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]], 'layers': [[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[2, 4], [0, 1]], [[0, 1], [2, 3]], [[0, 2], [1, 3]], [[2, 4], [0, 1]]]}, (6, 3): {'angles': [-1.1587325261959078, 4.054291264522096, -0.832318319265232, -0.05529000552069274, 1.945349057346249, 2.449356900559118, -1.2273715213816687, -3.0740608433392755, -0.22411234242881936, -1.218563474982311, -1.2682497606296208, 1.7028864809395576, -0.5583156428328178, -0.45406327603201657, 2.642354447205798, 1.3654962842988456, 0.7989304988580569, -0.696406337546169, -0.6296843788676697, 2.0042367834475625, 0.4465084180019968, 0.06506701652677492, 1.397654299316236, -1.0305224066812717, -1.0596798458932037, 0.9355058269456195, 4.71238898038469, 1.5707963267948966, -1.5707963267948963, 4.71238898038469, -1.5707963267948966, 1.5707963267948966, -9.244868821011682e-17, -3.141592653589793, 1.5707963267948968, -1.5707963267948966, 1.5707963267948968, -5.2202401900754905e-17, 1.5707963267948963, 3.141592653589793, 1.5707963267948966, 3.3919127285705133e-16], 'coords': [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], 'layers': [[[0, 2], [1, 3], [4, 5]], [[0, 1], [2, 3], [4, 5]], [[0, 2], [1, 3], [4, 5]], [[0, 1], [2, 4], [3, 5]], [[0, 2], [1, 3], [4, 5]], [[0, 1], [2, 3], [4, 5]]]}, (8, 3): {'angles': [1.5583187045626021, -3.1415926535897927, 5.208491071435463, 0.7818320292060451, 0.5313519625682895, -2.670820321126655, -3.139913987986567, -1.0149708664277908, 1.5583187045626021, -3.6039317335405306e-16, -4.405158511086932, -3.191424733971771, -1.0972527734774649, 3.0933092733224776, -1.9960076199541927, 2.3396128956288824, 3.094494000534572, 1.3625073655517543, 1.581929853746623, 2.4763086992746093, -3.175404704938736, -6.184957546348496, -2.8580372420390145, -0.9901588318738902, 2.8329988416935614, -2.0774516709018056, 1.750194454777398, -2.7238527237224184, 1.1504270471575575, -2.878694326002739, -0.5251095513406993, -0.5630258708367256, -8.473365917931746, -1.2031086345590554, 3.3840571385390796, 0.42989943402514597, -0.20514828979913344, 1.2792023071464371, 0.6655837745142688, 0.6547237454781646, 0.864662744791886, 2.3379665699681538, 3.1415926535897936, 1.1589338251128961e-16, -0.7054971262004875, -1.8171405310405533, 4.4930901681199105, -1.0374989643280963, 1.9547930623865346, 1.4022020809967617, 3.1415926535897936, 2.2113504909352253, 1.2158289201545691, -1.3746655614494785, -1.3944749412557864, -0.37269481654733194, -1.2131507666444596, -0.2929603566478166, -3.141592653589793, 2.7360054115101713, 1.462659293282678, -0.030978369796198283, -4.593270659377416, 2.146158857081619, 3.8432714931494187, 2.226107556515043, -3.1415926535897936, -0.7510787657431719, 2.818927827293758, -0.2796952785212738, 1.7159293330234586, -1.658158213158069, 2.0606115785894383, 1.6894075654110476, -2.0183266607644494e-17, -1.097561054985494e-17, -1.170027477180718, 3.141592653589793, -1.5250599141234784, 2.1893776337589133, 1.4686008662970813, 1.610009484128973, -3.141592653589793, -3.141592653589793, 1.0591793861554955, 3.141592653589793, 1.28697448178118, -0.5402432982433211, 0.9578020357652939, -0.5605245247508785, 3.141592653589793, -3.141592653589793, 0.11651114501944934, 3.141592653589793, -2.557858682256592, 2.787982139225623, -1.7589980729426002, 3.8518281175833535, -1.5707963267948966, -2.997539058395352, 1.6333908462806441, -2.999902776521562, -1.6668292415473602, -2.3416814810735547, -0.3190545626931916, -1.0022370355752628, -1.1758108535150003e-16, -1.5707963267948966, -1.49688668473229, 0.1092276667572707, -1.116635076589909, 0.7273158990165092, -3.5514854616344877e-16, -3.141592653589793, 1.5707963267948966, 1.5707963267948966, 1.3145263286586932, -1.1789709192162374, -3.2629847494128574, -1.6246852449622058], 'coords': [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]], 'layers': [[[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]], [[2, 4], [3, 5], [0, 1], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]], [[2, 4], [3, 5], [0, 1], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]], [[2, 4], [3, 5], [0, 1], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]], [[2, 4], [3, 5], [0, 1], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]]]}, (8, 4): {'angles': [1.342148489334907, -4.612114008872736, -0.5055213333918288, -2.442351980254898, 1.5707963267948963, -1.5707963267948966, 4.71238898038469, -1.5707963267948966, 5.5195889672377545, -3.532078850472962, -3.4582841638009136, 1.4180146845055248, 3.8087564804381864, 10.091941787617772, -3.746255035781058, 2.1754587089861612, 2.8032559155511527, 4.1695980573432525, 0.5168572971379731, 2.6134153217969067, 0.36603547559631594, -1.3742753164431483, 3.8746911286404777, 1.5707963267948968, -1.1067256021843812, 1.4773640088256055, 1.5707963267948968, 8.813522445852739e-17, 1.8299966339145368, -2.6841558389605695, 1.0017651766069884, -1.2710065158619162, 5.23695538276118, -2.005376070877285, 1.5707963267948963, 1.5707963267948972, -3.1415926535897936, -1.5707963267948968, 6.223319612899869, 1.5707963267948966, 2.043808448093259, -1.5707963267948968, -3.141592653589793, 1.5707963267948972, 9.902629713708122e-18, -2.75723953456677e-16, -0.1736976233072595, 5.0730971112734995, 1.5707963267948966, -1.5707963267948968, -1.1028159302606837, 1.4411394796797814, -3.141592653589793, -1.5707963267948966, -9.666891048842117e-17, -2.254966393059075e-17, 3.141592653589794, 0.2996808628723877, 3.0597028088390537, -1.2150944469848233, 3.9575497334626766e-16, -1.5707963267948966, -1.397098703487637, 1.2100881959060872, 4.712388980384929, -1.043298218363021, 3.141592653589793, 0.2701453725531202, -3.0722598631232356e-16, 2.156970577508139, 4.05288834663716, -1.8374115464844574, 1.5707963267948968, -0.0033703839439966905, -1.5707963267948966, -3.4566197594509463e-18, 6.298744508512385e-17, 2.172950900540702, 3.141592653589793, 0.9400908425031144, -4.712388980384929, 1.570796326794898, -4.71238898038469, -3.0523149439161976, 2.186276035465284, 1.5707963267948966, 0.19061272694848302, -3.4082078732793537, -1.5707963267948968, 9.138459623426163e-16, 9.415552633385868e-17, -6.93392713101534e-17, -1.5707963267948966, -1.3005854598447509e-17, -2.0844629258405236, 1.3876251559129937e-16, 1.941890635282795, 0.7327953705992574, -1.5707963267948966, -3.438080472071161, -3.1415926535897936, -1.7640017699633897e-16, -2.3561944901923453, -1.9817112991944992e-16, -3.1415926535897922, 4.2264584775420873e-17, 1.4655737820522332e-16, -1.5707963267948963, -5.699837118170861e-17, -2.819849061273205e-17, 0.5136665990456272, 5.496428860628118e-17, -1.1076322260409452e-16, -1.432987450681155e-16, -1.5707963267948966, 1.5707963267948963, 1.6102316138157857e-16, 4.3418420262843594e-16, 2.920790813687381e-16, -1.570796326794897], 'coords': [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]], 'layers': [[[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]], [[2, 4], [3, 5], [0, 1], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]], [[2, 4], [3, 5], [0, 1], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]], [[2, 4], [3, 5], [0, 1], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]], [[2, 4], [3, 5], [0, 1], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]]]}}
_other__THREE_ARM_SEEDS = {12: {'N': 6, 'angles': [-1.7753612101505012, -1.5031115642308233, -1.2974973838927273, 4.795027544633506, 1.5707963267948966, 1.5707963267948974, 0.5896530708477706, -2.258350892005411, -3.7687001756936302, 2.255551546582445, -0.10784931344318781, 2.699974722275354, -3.6945708360575424, 3.141592653589793, -1.5707963267948966, 3.141592653589793, -2.5921254088321657, 2.877531407632771e-17, -4.604551122942012e-17, -0.5494672447576275, -2.356194490192345, 2.356194490192345, 1.570796326794896, -5.497787143782138, -1.5707963267948966, 4.837678183550181, -3.9269908169872414, 1.5707963267948981, -6.3665077052936025, 1.5707963267948966, 1.5707963267948966, -9.166390104959884e-18, -3.141592653589793, 2.2979787286487854e-16, -1.5707963267948968, 1.1788520049107612e-16, -3.1415926535897936, -4.987980590979463e-19, 4.71238898038469, -3.141592653589793, 7.9373040320885, 3.1415926535897936], 'layers': [[[0, 2], [1, 3], [4, 5]], [[0, 1], [2, 4], [3, 5]], [[0, 2], [1, 3], [4, 5]], [[0, 1], [2, 4], [3, 5]], [[0, 2], [1, 3], [4, 5]], [[0, 1], [2, 3], [4, 5]]], 'n': 2, 'residual': 5.086359306078474e-14, 'sizes': [4, 4, 4], 'total': 12}, 15: {'N': 6, 'angles': [-1.440790703919356, -2.287067197151006, -1.255914435443566, 4.751237396643934, 1.5707963267948963, 1.5707963267948954, 0.184716530316489, -3.1861799918514793, -2.7628019666403714, 3.200721579119603, 0.2754057996269368, 2.302244212602228, -3.9299654461729254, 3.1415926535897936, -1.5707963267948968, 3.141592653589793, -2.5776500122291646, -5.4397938530322016e-17, 1.71019330737766e-17, -0.5639426413606287, -2.356194490192345, 2.356194490192345, 1.570796326794897, -5.497787143782138, -1.5707963267948968, 3.8214782192759196, -3.926990816987242, 1.5707963267948948, -9.17147783224243, 1.5707963267948968, 1.5707963267948966, -1.0650113191615857e-16, -3.141592653589793, -3.913842428212354e-16, -1.570796326794897, -1.2239910749985933e-16, -3.141592653589793, -3.587376619827427e-17, 4.71238898038469, -3.141592653589793, 10.742274159037327, 3.141592653589793], 'layers': [[[0, 2], [1, 3], [4, 5]], [[0, 1], [2, 4], [3, 5]], [[0, 2], [1, 3], [4, 5]], [[0, 1], [2, 4], [3, 5]], [[0, 2], [1, 3], [4, 5]], [[0, 1], [2, 3], [4, 5]]], 'n': 2, 'residual': 7.5645951984681e-14, 'sizes': [5, 5, 5], 'total': 15}}

def _other__three_arm(n):
    data = _other__THREE_ARM_SEEDS.get(n)
    if data is None:
        return None
    arms = [[_other_cirq.GridQubit(4 + i, 4 + j) for j in range(size)] for i, size in enumerate(data['sizes'])]
    roots = [q for arm in arms for q in arm[:2]]
    ops = []
    for layer in range(len(data['layers']) + 1):
        if layer:
            ops.extend(((roots[a], roots[b]) for a, b in data['layers'][layer - 1]))
        for i, q in enumerate(roots):
            _other__u(ops, q, _other_ry(data['angles'][6 * layer + i]))
    for arm in arms:
        _other__unary_fused(ops, arm, 2)
    return (_other__optimize_depth(_other__compile(ops)), {q for arm in arms for q in arm})

def _other__native_seed(n, k):
    data = _other__NATIVE_SEEDS.get((n, k))
    if data is None:
        return None
    q = [_other_cirq.GridQubit(4 + r, 4 + c) for r, c in data['coords']]
    ops = []
    for layer in range(len(data['layers']) + 1):
        if layer:
            ops.extend(((q[a], q[b]) for a, b in data['layers'][layer - 1]))
        for i in range(n):
            _other__u(ops, q[i], _other_ry(data['angles'][layer * n + i]))
    return (_other__optimize_depth(_other__compile(ops)), set(q))

def _other__subtract_balanced_seed(n, k):
    """Measure an isolated output diagonal, then complement the zero branch."""
    if n != 2 * k + 1 or (n + 1, k + 1) not in _other__NATIVE_SEEDS:
        return None
    circuit, outputs = _other__native_seed(n + 1, k + 1)
    physical = _other__graph()
    choices = [q for q in outputs if sum((x.row + x.col == q.row + q.col for x in outputs)) == 1]
    if not choices:
        return None
    q = min(choices)
    diagonal = sorted((x for x in physical if x.row + x.col == q.row + q.col))
    circuit.append(_other_cirq.Moment((_other_cirq.measure(x, key=f'subtract_{x.row}_{x.col}') for x in diagonal)))
    outputs.remove(q)
    condition = _other_cirq.SympyCondition(_other_sympy.Eq(_other_sympy.Symbol(f'subtract_{q.row}_{q.col}'), 0))
    circuit.append(_other_cirq.Moment((_other_cirq.X(x).with_classical_controls(condition) for x in outputs)))
    return (circuit, outputs)

def _other__w_from_tree(rounds):
    edges = [edge for rnd in rounds for edge in rnd]
    root = edges[0][0]
    children = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)
    sizes = {}

    def size(q):
        sizes[q] = 1 + sum((size(c) for c in children.get(q, [])))
        return sizes[q]
    size(root)
    remaining = dict(sizes)
    ops = []
    for i, (a, b) in enumerate(edges):
        theta = _other_math.asin(_other_math.sqrt(sizes[b] / remaining[a]))
        if i == 0:
            _other__u(ops, a, _other_ry(2 * theta))
            _other__cx(ops, a, b)
            _other__u(ops, a, _other_X)
        else:
            eta = _other_math.pi / 2 - theta
            _other__u(ops, b, _other_ry(eta))
            _other__cx(ops, a, b)
            _other__u(ops, b, _other_ry(-eta))
            _other__cx(ops, b, a)
        remaining[a] -= sizes[b]
    return (_other__compile(ops), set(sizes))

def _other__w(n):
    graph = _other__graph()
    rng = _other_random.Random(83419 + n)
    best = None
    finalists = []
    root = _other_cirq.GridQubit(6, 7)
    for trial in range(96):
        used = {root}
        rounds = []
        while len(used) < n:
            candidates = [(a, b) for a in sorted(used) for b in sorted(graph[a] - used)]
            rng.shuffle(candidates)
            busy, rnd = (set(), [])
            for a, b in candidates:
                if a in busy or b in busy:
                    continue
                rnd.append((a, b))
                busy.update((a, b))
                if len(used) + len(rnd) == n:
                    break
            rounds.append(rnd)
            used.update((b for _, b in rnd))
        candidate = _other__w_from_tree(rounds)
        candidate = (_other__optimize_frames(candidate[0]), candidate[1])
        finalists.append(candidate)
        if best is None or len(candidate[0]) < len(best[0]):
            best = candidate
    for candidate, outputs in sorted(finalists, key=lambda item: len(item[0]))[:16]:
        candidate = _other__optimize_depth(candidate)
        if len(candidate) < len(best[0]):
            best = (candidate, outputs)
    return best

@_other_functools.lru_cache(maxsize=64)
def _other__best(n, k):
    if k == 1:
        c, q = _other__w(n)
        method = 'parallel W tree'
    elif (n, k) == (4, 2):
        c, q = _other__square()
        method = 'analytic square'
    else:
        candidates = []
        for a in sorted({n // 2, (n + 1) // 2}):
            for small in ['fused', True, False, 'pruned'] if k == 2 else ['fused', False, 'pruned']:
                candidate, outputs = _other__divide(n, k, a, small)
                candidate = _other__optimize_depth(candidate)
                candidates.append((len(candidate), candidate, outputs, f'hypergeometric {a}+{n - a}' + (' / ' + small if isinstance(small, str) else ' / U3' if small else '')))
            if n % 2 == 0 and k >= 3:
                candidate, outputs = _other__divide(n, k, a, small='fused', pivot=k // 2)
                candidate = _other__optimize_depth(candidate)
                candidates.append((len(candidate), candidate, outputs, f'hypergeometric {a}+{n - a} / center-out seed'))
        _, c, q, method = min(candidates, key=lambda item: item[0])
        seed = _other__native_seed(n, k)
        if seed is not None and len(seed[0]) < len(c):
            c, q = seed
            method = 'independent native grid seed'
        arms = _other__three_arm(n) if k == 2 else None
        if arms is not None and len(arms[0]) < len(c):
            c, q = arms
            method = 'three-arm unary distribution + parallel decoders'
        subtraction = _other__subtract_balanced_seed(n, k)
        if subtraction is not None and len(subtraction[0]) < len(c):
            c, q = subtraction
            method = 'native balanced seed + deterministic measurement/complement'
        if n == 2 * k and n >= 6:
            adaptive = _other_balanced_candidate(n)
            if len(adaptive['circuit']) < len(c):
                c, q = (_other_cirq.Circuit(adaptive['circuit']), set(adaptive['outputs']))
                method = 'adaptive cat complement + even-site teleportation'
    for op in c.all_operations():
        _other_cirq_google.Willow105.validate_operation(op.without_classical_controls())
    if not c.has_measurements():
        _other_cirq_google.Willow105.validate_circuit(c)
    return (_other_cirq.FrozenCircuit(c), frozenset(q), method)

def _other_create_dicke_state_adaptive(num_qubits, num_excitations):
    """Return an exact D(N,n) for all confirmed bonus instances.

    The interface permits feedforward; the portfolio may choose a shallower
    unitary circuit. N=2..15, n=1..floor(N/2). Build time is bounded and cached.
    """
    if isinstance(num_qubits, bool) or isinstance(num_excitations, bool) or (not isinstance(num_qubits, int)) or (not isinstance(num_excitations, int)) or (not 2 <= num_qubits <= 15) or (not 1 <= num_excitations <= num_qubits // 2):
        raise NotImplementedError('Bonus domain: integer 2<=N<=15, 1<=n<=N//2')
    circuit, qubits, _ = _other__best(num_qubits, num_excitations)
    return (_other_cirq.Circuit(circuit), set(qubits))

def _other_method_for(num_qubits, num_excitations):
    return _other__best(num_qubits, num_excitations)[2]

def _other__layout(core_outputs):
    graph = _other__graph()
    core = set(core_outputs)
    odd = {q for q in graph if (q.row + q.col) % 2}
    even = set(graph) - odd
    external_cats = odd - core
    assignment = {}
    occupied = set(core)
    destinations = {q: q for q in core if q in even}
    for q in sorted(core & odd):
        candidates = [v for v in graph[q] - occupied if v in even and graph[v] & external_cats]
        if not candidates:
            raise ValueError('No fresh even teleportation destination')
        v = min(candidates, key=lambda v: (-len(graph[v] & external_cats), v))
        destinations[q] = v
        occupied.add(v)
    usage = _other_collections.Counter()
    for q, dest in sorted(destinations.items()):
        choices = graph[dest] & external_cats
        if not choices:
            raise ValueError('No outside cat neighbor for an even output')
        cat = min(choices, key=lambda c: (usage[c], c))
        usage[cat] += 1
        assignment[dest] = cat
    cat_graph = {q: {} for q in external_cats}
    for mediator in sorted(even - core):
        neighbors = sorted(graph[mediator] & external_cats)
        for a in neighbors:
            for b in neighbors:
                if a != b:
                    cat_graph[a].setdefault(b, mediator)
    terminals = set(assignment.values())
    best = None
    for root in sorted(cat_graph):
        parent = {root: None}
        distance = {root: 0}
        queue = _other_collections.deque([root])
        while queue:
            a = queue.popleft()
            for b in sorted(cat_graph[a]):
                if b not in parent:
                    parent[b] = a
                    distance[b] = distance[a] + 1
                    queue.append(b)
        if not terminals <= parent.keys():
            continue
        included = {root}
        for terminal in terminals:
            v = terminal
            while v not in included:
                included.add(v)
                v = parent[v]
        edges = [(parent[q], q, cat_graph[parent[q]][q]) for q in sorted(included - {root}, key=lambda q: (distance[q], q))]
        ops = [(root, _other_H)]
        for a, b, m in edges:
            _other__cx(ops, a, m)
            _other__cx(ops, m, b)
            _other__cx(ops, a, m)
        cat_circuit = _other__compile(ops)
        root_options = [(cat, target) for cat in included for target in sorted(graph[cat] - occupied) if target in even]
        if not root_options:
            continue
        root_cat, output_root = min(root_options, key=lambda ct: (usage[ct[0]], ct))
        candidate = (len(cat_circuit), len(included), cat_circuit, included, root_cat, output_root, edges)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError('External cat graph is disconnected')
    _, _, cat_circuit, cats, root_cat, output_root, edges = best
    return dict(destinations=destinations, assignment=assignment, cat_circuit=cat_circuit, cats=cats, root_cat=root_cat, root=output_root, cat_edges=edges)

def _other__key(q):
    return f'odd_{q.row}_{q.col}'

def _other__parity(keys):
    return _other_cirq.SympyCondition(_other_sympy.Eq(_other_sympy.Mod(sum((_other_sympy.Symbol(k) for k in keys)), 2), 1))

@_other_functools.lru_cache(maxsize=8)
def _other_balanced_candidate(n):
    if n % 2 or not 6 <= n <= 14:
        raise ValueError('Use even N=6..14')
    k = n // 2
    cores = []
    for a in sorted({(n - 1) // 2, n // 2}):
        for mode in [True, 'pruned', 'fused']:
            candidate, output = _other__divide(n - 1, k - 1, a=a, small=mode)
            cores.append((_other__optimize_depth(candidate), output))
    seed = _other__native_seed(n - 1, k - 1)
    if seed is not None:
        cores.append(seed)
    core, core_outputs = min(cores, key=lambda pair: len(pair[0]))
    layout = _other__layout(core_outputs)
    cat = layout['cat_circuit']
    assert not core.all_qubits() & cat.all_qubits()
    prefix = _other_cirq.Circuit(core.all_operations(), cat.all_operations())
    suffix_ops = []
    for old, dest in sorted(layout['destinations'].items()):
        if old != dest:
            _other__cx(suffix_ops, old, dest)
    for dest, control in sorted(layout['assignment'].items()):
        _other__cx(suffix_ops, control, dest)
    _other__cx(suffix_ops, layout['root_cat'], layout['root'])
    _other__u(suffix_ops, layout['root'], _other_X)
    odd = sorted((q for q in _other__graph() if (q.row + q.col) % 2))
    measured_active = layout['cats'] | {q for q in core_outputs if (q.row + q.col) % 2}
    for q in sorted(measured_active):
        _other__u(suffix_ops, q, _other_H)
    suffix_unitary = _other__compile(suffix_ops)
    prep_ops = []
    for op in _other_itertools_ops(prefix, suffix_unitary):
        if len(op.qubits) == 1:
            prep_ops.append((op.qubits[0], _other_cirq.unitary(op)))
        else:
            assert op.gate == _other_cirq.CZ
            prep_ops.append(op.qubits)
    circuit = _other__compile(prep_ops)
    circuit.append(_other_cirq.Moment((_other_cirq.measure(q, key=_other__key(q)) for q in odd)))
    corrections = []
    correction_keys = {}
    for old, dest in sorted(layout['destinations'].items()):
        if old != dest:
            keys = [_other__key(old)]
            corrections.append(_other_cirq.Z(dest).with_classical_controls(_other__key(old)))
            correction_keys[dest] = keys
    keys = sorted((_other__key(q) for q in measured_active))
    corrections.append(_other_cirq.Z(layout['root']).with_classical_controls(_other__parity(keys)))
    correction_keys[layout['root']] = keys
    circuit.append(_other_cirq.Moment(corrections))
    outputs = set(layout['destinations'].values()) | {layout['root']}
    assert len(outputs) == n and all(((q.row + q.col) % 2 == 0 for q in outputs))
    return dict(circuit=_other_cirq.FrozenCircuit(circuit), outputs=frozenset(outputs), core=_other_cirq.FrozenCircuit(core), core_outputs=frozenset(core_outputs), cat=_other_cirq.FrozenCircuit(cat), suffix=_other_cirq.FrozenCircuit(suffix_unitary), correction_keys=correction_keys, **layout)

def _other_itertools_ops(*circuits):
    for circuit in circuits:
        yield from circuit.all_operations()

def create_dicke_state(num_qubits: int, num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit,cirq.GridQubit],tuple[float,float]],
    sq_pauli_error_rate: dict[cirq.GridQubit,tuple[float,float]],
    zero_errors: dict[cirq.GridQubit,tuple[float,float]],
    one_errors: dict[cirq.GridQubit,tuple[float,float]]) -> tuple[cirq.Circuit,set[cirq.GridQubit]]:
    """Notebook API: return a native measurement-free circuit and its output set."""
    args=(cz_pauli_error_rate,sq_pauli_error_rate,zero_errors,one_errors)
    try:
        c,q=_FINAL_ASTRA_MAIN(num_qubits,num_excitations,*args)
    except NotImplementedError:
        c,q=_idea_create_dicke_state(num_qubits,num_excitations,*args)
    return _checked_hardware(c,q,num_qubits,cz_pauli_error_rate,sq_pauli_error_rate)

def create_dicke_state_adaptive(num_qubits: int,num_excitations: int) -> tuple[cirq.Circuit,set[cirq.GridQubit]]:
    """Notebook bonus API; only validated unitary candidates are selected."""
    if num_excitations in (0, num_qubits):
        return _boundary_product_state(num_qubits, num_excitations)
    c,q=_FINAL_ASTRA_BONUS(num_qubits,num_excitations)
    candidates=[]
    try:
        seed=_other__native_seed(num_qubits,num_excitations)
        if seed is not None:candidates.append(seed)
        if num_excitations==2:
            arms=_other__three_arm(num_qubits)
            if arms is not None:candidates.append(arms)
    except Exception:pass
    for data in _FINAL_EXTRA.get((num_qubits,num_excitations),[]):
        candidates.append(_decode_template_data(data))
    for candidate,outputs in candidates:
        candidate=cirq.Circuit(candidate)
        if len(candidate)>=len(c) or candidate.has_measurements() or len(outputs)!=num_qubits:continue
        if not all(op.gate==cirq.CZ or isinstance(op.gate,cirq.PhasedXZGate) for op in candidate.all_operations()):continue
        cirq_google.Willow105.validate_circuit(candidate)
        c,q=candidate,set(outputs)
    return c,set(q)

"""ColumnForge: first-column Dicke synthesis, extending the attributed Astra baseline.

Hardware: create_dicke_state(N,k,cz_rates,sq_rates,zero_errors,one_errors).
Bonus: create_dicke_state_adaptive(N,k), deterministic and measurement-free.
Optional hardware echo variant: create_dicke_state_echo (paired native X pulses).

Families include analytic constructions and offline numerical templates
verified to tight noiseless amplitude residuals; online selection
changes only circuit family, qubit placement and gates that cancel noiselessly.
Numerical noise predictions are ranking heuristics, not Willow hardware scores.
This file needs only Cirq/Cirq-Google, their dependencies, and Python stdlib.
No external files, network access, runtime fitting or training are required.

Foundational sources: arXiv:1904.07358 and arXiv:2112.12435.
See the accompanying prompt log, derivations, validation and score report.
"""
'Exact Dicke circuits on a linear chain, using only full CZ and PhasedXZ.\n\nThe split/cyclic-shift construction follows Baertschi and Eidenbenz,\narXiv:1904.07358. A reverse sweep of the local blocks in Fig. 2 of\nAktar et al., arXiv:2112.12435v2 shifts a unary register one site at each\nrecursion without restoring swaps. No ancillas or measurements are used.\n'
import math as _cf_math
import functools as _cf_functools
import cirq as _cf_cirq
import numpy as _cf_np

def _cf__reflect3(a, b, c, n, ell):
    """Five local CNOTs implementing the reverse-sweep restricted isometry.

    On the only reachable basis states:
      000 -> 000; 100 -> sqrt(ell/n)100+sqrt(1-ell/n)010;
      101 -> 011; 110 -> 110; 111 -> 111.
    """
    alpha = 0.5 * _cf_math.asin(_cf_math.sqrt(ell / n))
    yield _cf_cirq.CNOT(b, a)
    yield _cf_cirq.ry(alpha)(b)
    yield _cf_cirq.CNOT(c, b)
    yield _cf_cirq.ry(alpha)(b)
    yield _cf_cirq.CNOT(a, b)
    yield _cf_cirq.ry(-alpha)(b)
    yield _cf_cirq.CNOT(c, b)
    yield _cf_cirq.ry(-alpha)(b)
    yield _cf_cirq.CNOT(b, a)

def _cf_raw_circuit(n: int, k: int) -> _cf_cirq.Circuit:
    """Sliding unary register; reflect/swap blocks avoid restoring swaps."""
    if not isinstance(n, int) or not isinstance(k, int) or n < 1 or (not 0 <= k <= n):
        raise ValueError('Require integer n >= 1 and 0 <= k <= n')
    q = _cf_cirq.LineQubit.range(n)
    complement = k > n // 2
    weight = n - k if complement else k
    ops = list(_cf_cirq.X.on_each(q[:max(0, weight - 1)]))
    for start in range(n - 1):
        remaining = n - start
        width = min(weight, remaining - 1)
        if not width:
            break
        co = _cf_math.sqrt(width / remaining)
        si = _cf_math.sqrt((remaining - width) / remaining)
        givens = _cf_np.array([[1, 0, 0, 0], [0, co, si, 0], [0, -si, co, 0], [0, 0, 0, 1]])
        if start == 0:
            a, b = q[width - 1:width + 1]
            ops.extend([_cf_cirq.ry(2 * _cf_math.acos(co))(a), _cf_cirq.CNOT(a, b), _cf_cirq.X(a)])
        else:
            ops.append(_cf_cirq.MatrixGate(givens)(q[start + width - 1], q[start + width]))
        for ell in range(width - 1, 0, -1):
            a, b, c = q[start + ell - 1:start + ell + 2]
            if ell < weight - start:
                ops.extend([_cf_cirq.CNOT(b, a), _cf_cirq.CNOT(a, b), _cf_cirq.X(a)])
            else:
                ops.extend(_cf__reflect3(a, b, c, remaining, ell))
    if complement:
        ops.extend(_cf_cirq.X.on_each(q))
    if not ops:
        ops = list(_cf_cirq.I.on_each(q))
    return _cf_cirq.Circuit(ops)

def _cf__unary_three_weight_two(q):
    """Proved six-CZ isometry, valid ONLY on unary weights 0, 1, and 2.

    Analytic construction: research/exact_u3.py. All 24 promised amplitudes
    were proved symbolically in research/verify_exact_u3.py. Input 111 is
    deliberately outside this isometry's supported domain.
    """
    if len(q) != 3:
        raise ValueError('Expected exactly three line qubits')
    a = _cf_math.asin(1 / _cf_math.sqrt(3))
    angles = [-_cf_math.pi / 2, _cf_math.pi, a - _cf_math.pi, -_cf_math.pi / 2, a - _cf_math.pi, _cf_math.pi / 2 + a, a, -_cf_math.pi / 2, -_cf_math.pi / 2, -_cf_math.pi / 2, 0, -_cf_math.pi / 4, _cf_math.pi / 4, _cf_math.pi / 2, _cf_math.pi / 2]
    edges = [(1, 2), (0, 1), (1, 2), (0, 1), (1, 2), (1, 2)]
    ops = [_cf_cirq.ry(angles[i])(q[i]) for i in range(3)]
    for j, (u, v) in enumerate(edges):
        ops.append(_cf_cirq.CZ(q[u], q[v]))
        if angles[3 + 2 * j]:
            ops.append(_cf_cirq.ry(angles[3 + 2 * j])(q[u]))
        if angles[4 + 2 * j]:
            ops.append(_cf_cirq.ry(angles[4 + 2 * j])(q[v]))
    return ops

def _cf_unary_to_dicke_circuit(n: int, max_weight: int, *, native=True) -> _cf_cirq.Circuit:
    """Coherently map |1^k 0^(n-k)> -> |D(n,k)> for every k<=max_weight.

    No preparation gates are included. Relative phases between different k are
    preserved, so this can be used after a coherent unary-weight distribution.
    """
    if n < 1 or not 0 <= max_weight <= n:
        raise ValueError('Require n>=1 and 0<=max_weight<=n')
    q = _cf_cirq.LineQubit.range(n)
    ops = []
    for start in range(n - 1):
        remaining = n - start
        if remaining == 3 and max_weight == 2:
            ops.extend(_cf__unary_three_weight_two(q[start:start + 3]))
            break
        width = min(max_weight, remaining - 1)
        if not width:
            break
        co = _cf_math.sqrt(width / remaining)
        si = _cf_math.sqrt((remaining - width) / remaining)
        givens = _cf_np.array([[1, 0, 0, 0], [0, co, si, 0], [0, -si, co, 0], [0, 0, 0, 1]])
        ops.append(_cf_cirq.MatrixGate(givens)(q[start + width - 1], q[start + width]))
        for ell in range(width - 1, 0, -1):
            ops.extend(_cf__reflect3(*q[start + ell - 1:start + ell + 2], remaining, ell))
    result = _cf_cirq.Circuit(ops)
    if native:
        result = _cf_native_compile(result)
    return result

def _cf_native_compile(raw: _cf_cirq.AbstractCircuit) -> _cf_cirq.Circuit:
    """Exact local compilation retaining the pipelined LNN schedule.

    Full-circuit target compilation puts additional dependencies between
    disjoint blocks. Local compilation followed by EARLIEST repacking gives
    the same CZ count and substantially fewer moments and single-qubit gates.
    """
    ops = []
    for op in raw.all_operations():
        if op.gate == _cf_cirq.CNOT:
            a, b = op.qubits
            ops.extend([_cf_cirq.H(b), _cf_cirq.CZ(a, b), _cf_cirq.H(b)])
        elif len(op.qubits) == 1 or op.gate == _cf_cirq.CZ:
            ops.append(op)
        else:
            ops.extend(_cf_cirq.two_qubit_matrix_to_cz_operations(*op.qubits, _cf_cirq.unitary(op), allow_partial_czs=False))
    result = _cf_cirq.Circuit(ops)
    result = _cf_cirq.merge_single_qubit_gates_to_phxz(result, atol=1e-14)
    result = _cf_cirq.drop_negligible_operations(result, atol=1e-14)
    return _cf_cirq.Circuit(result.all_operations())

@_cf_functools.lru_cache(maxsize=256)
def _cf__logical_frozen(n: int, k: int) -> _cf_cirq.FrozenCircuit:
    raw = _cf_raw_circuit(n, k)
    native = _cf_native_compile(raw)
    return _cf_cirq.FrozenCircuit(native)

def _cf_logical_circuit(n: int, k: int) -> _cf_cirq.Circuit:
    """Return a fresh exact Dicke(n,k) circuit on LineQubit.range(n)."""
    return _cf_cirq.Circuit(_cf__logical_frozen(n, k))
'Deterministic, calibration-aware path placement; no extra dependencies.'
import math as _cf_math
from collections import Counter as _cf_Counter
import cirq as _cf_cirq

def _cf_rate(value, uncertainty_weight=0.25):
    """Small uncertainty penalty discourages selecting noisy calibration outliers."""
    if isinstance(value, (tuple, list)):
        return max(0.0, min(0.49, float(value[0]) + uncertainty_weight * float(value[1])))
    try:
        return max(0.0, min(0.49, float(value[0]) + uncertainty_weight * float(value[1])))
    except (TypeError, IndexError):
        return max(0.0, min(0.49, float(value)))

def _cf_calibration_graph(cz_rates, sq_rates):
    nodes = sorted(sq_rates)
    graph = {q: {} for q in nodes}
    for (a, b), value in cz_rates.items():
        if a == b or a not in graph or b not in graph:
            continue
        p = _cf_rate(value)
        p = max(p, graph[a].get(b, 0.0))
        graph[a][b] = graph[b][a] = p
    return graph

def _cf_logical_weights(circuit, n):
    sq = [0] * n
    edges = [0] * max(0, n - 1)
    for op in circuit.all_operations():
        ids = [q.x for q in op.qubits]
        if len(ids) == 1:
            sq[ids[0]] += 1
        elif len(ids) == 2 and abs(ids[0] - ids[1]) == 1:
            edges[min(ids)] += 1
        else:
            raise ValueError(f'Expected line-neighbor circuit, got {op!r}')
    return (sq, edges)

def _cf_path_candidates(n, k, cz_rates, sq_rates, zero_errors, one_errors, sq_weights=None, edge_weights=None, width=768, count=12):
    """Beam-search paths, weighted by the exact logical gate-use profile.

    Readout dispersion penalty is the first-order shell nonuniformity term:
    Var(sum of k selected d_i) = k(N-k)/(N(N-1)) * sum((d_i-mean d)^2),
    where d_i = log(1-e1_i)-log(1-e0_i). It is a ranking heuristic;
    independent noisy simulation is required to establish an actual score.
    """
    if n < 1 or k < 0 or k > n:
        raise ValueError('Require 1 <= N and 0 <= k <= N')
    graph = _cf_calibration_graph(cz_rates, sq_rates)
    if n > len(graph):
        raise NotImplementedError('Not enough calibrated qubits')
    sq_weights = sq_weights if sq_weights is not None else [3.0] * n
    edge_weights = edge_weights if edge_weights is not None else [2.0] * (n - 1)
    if len(sq_weights) != n or len(edge_weights) != n - 1:
        raise ValueError('Incorrect gate-use profile size')
    frac = k / n
    readout, asym, singles = ({}, {}, {})
    for q in graph:
        e0 = _cf_rate(zero_errors.get(q, (0.01, 0.0)))
        e1 = _cf_rate(one_errors.get(q, (0.02, 0.0)))
        readout[q] = -((1 - frac) * _cf_math.log1p(-e0) + frac * _cf_math.log1p(-e1))
        asym[q] = _cf_math.log1p(-e1) - _cf_math.log1p(-e0)
        singles[q] = _cf_rate(sq_rates[q])
    variance_scale = 10000.0 / _cf_math.comb(n, k) * k * (n - k) / (n * max(1, n - 1))

    def node_cost(q, i):
        return readout[q] + sq_weights[i] * singles[q]
    states = [(node_cost(q, 0), (q,), node_cost(q, 0), asym[q], asym[q] ** 2) for q in graph]
    for position in range(1, n):
        expanded = []
        for _, path, additive, ds, d2 in states:
            for q, p in graph[path[-1]].items():
                if q in path:
                    continue
                cost = additive + node_cost(q, position) + edge_weights[position - 1] * p
                new_ds, new_d2 = (ds + asym[q], d2 + asym[q] ** 2)
                dispersion = max(0.0, new_d2 - new_ds ** 2 / (position + 1))
                expanded.append((cost + variance_scale * dispersion, path + (q,), cost, new_ds, new_d2))
        if not expanded:
            raise NotImplementedError('No calibrated path found for requested N')
        expanded.sort(key=lambda item: (item[0], item[1]))
        states, per_end = ([], _cf_Counter())
        cap = max(8, width // max(1, len(graph)) * 3)
        for state in expanded:
            end = state[1][-1]
            if per_end[end] < cap:
                states.append(state)
                per_end[end] += 1
                if len(states) >= width:
                    break
    return [(list(s[1]), s[0]) for s in sorted(states, key=lambda item: (item[0], item[1]))[:count]]

def _cf_place_line(circuit, n, k, cz_rates, sq_rates, zero_errors, one_errors):
    sq, edges = _cf_logical_weights(circuit, n)
    path, _ = _cf_path_candidates(n, k, cz_rates, sq_rates, zero_errors, one_errors, sq_weights=sq, edge_weights=edges, count=1)[0]
    mapping = {_cf_cirq.LineQubit(i): path[i] for i in range(n)}
    return (circuit.transform_qubits(mapping), set(path))
'Standalone, calibration-aware exact Dicke(2,1) preparation.\n\nOnly Cirq and the Python standard library are required. This optimizer ranks\nall calibrated couplers in both orientations using an analytic Pauli/readout\nproxy for the expected official score denominator, including finite shots.\nIt returns exactly one CZ and three PhasedXZ gates in three moments.\n'
import cirq as _cf_cirq

def _cf__rate(mapping, key):
    value = mapping[key]
    return float(value[0] if isinstance(value, (tuple, list)) else value)

def _cf_bell_proxy(control, target, pair, cz_rates, sq_rates, zero_errors, one_errors, shots=10000, basis_rotation_noise=True):
    """Analytic independent-Pauli proxy; not a hardware prediction.

    Returns both Z/X probabilities, ordered control then target in big endian,
    and a deterministic objective. E[1/T] for retained count T is expanded
    through O((S*f)^-3); with S=10,000 and normal readout, omitted corrections
    are negligible compared with model mismatch. No-noise score is still
    random; expected denominator is a robust surrogate, not E[1/denominator].
    """
    pa, pb, pc = (_cf__rate(sq_rates, control), _cf__rate(sq_rates, target), _cf__rate(cz_rates, pair))
    sa, sb, sc = (1 - 4 * pa / 3, 1 - 4 * pb / 3, 1 - 16 * pc / 15)
    zz = -sb * sb * sc
    xx = sa * sb * sc
    if basis_rotation_noise:
        xx *= sa * sb
    da = _cf__rate(one_errors, control) - _cf__rate(zero_errors, control)
    db = _cf__rate(one_errors, target) - _cf__rate(zero_errors, target)
    ra = 1 - _cf__rate(one_errors, control) - _cf__rate(zero_errors, control)
    rb = 1 - _cf__rate(one_errors, target) - _cf__rate(zero_errors, target)
    zcorr, xcorr = (da * db + ra * rb * zz, da * db + ra * rb * xx)

    def probabilities(correlation):
        return tuple(((1 + s * da + t * db + s * t * correlation) / 4 for s, t in ((1, 1), (1, -1), (-1, 1), (-1, -1))))
    f = (1 - zcorr) / 2
    if f <= 0:
        expected_z = float('inf')
    else:
        delta = (da - db) / (2 * f)
        mu = shots * f
        inverse_retained = 1 / mu + (1 - f) / mu ** 2 + (1 - f) * (2 - f) / mu ** 3
        expected_z = shots / 2 * (delta * delta + (1 - delta * delta) * inverse_retained)
    xmean1 = (da + db) / 2
    expected_x = xmean1 * xmean1 + (1 - xcorr) ** 2 + ((1 + xcorr) / 2 - xmean1 * xmean1 + 1 - xcorr * xcorr) / shots
    return {'denominator': expected_z + expected_x, 'expected_z': expected_z, 'expected_x': expected_x, 'shell_probability': f, 'z_probs': probabilities(zcorr), 'x_probs': probabilities(xcorr)}

def _cf_ranked_pairs(cz_rates, sq_rates, zero_errors, one_errors):
    available = set(sq_rates) & set(zero_errors) & set(one_errors)
    rows = []
    for pair in cz_rates:
        if len(pair) != 2 or not set(pair) <= available:
            continue
        for control, target in (tuple(pair), tuple(reversed(pair))):
            proxy = _cf_bell_proxy(control, target, pair, cz_rates, sq_rates, zero_errors, one_errors)
            rows.append((proxy['denominator'], control, target, pair, proxy))
    return sorted(rows, key=lambda row: (row[0], row[1], row[2]))

def _cf_bell_circuit(control, target):
    h = _cf_cirq.PhasedXZGate.from_matrix(_cf_cirq.unitary(_cf_cirq.H))
    hx = _cf_cirq.PhasedXZGate.from_matrix(_cf_cirq.unitary(_cf_cirq.H) @ _cf_cirq.unitary(_cf_cirq.X))
    return _cf_cirq.Circuit(_cf_cirq.Moment([h(control), hx(target)]), _cf_cirq.Moment([_cf_cirq.CZ(control, target)]), _cf_cirq.Moment([h(target)]))

def _cf__create_bell(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    if (num_qubits, num_excitations) != (2, 1):
        raise NotImplementedError('This specialized circuit prepares Dicke(2,1)')
    rows = _cf_ranked_pairs(cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    if not rows:
        raise NotImplementedError('No calibrated coupler available')
    _, control, target, _, _ = rows[0]
    return (_cf_bell_circuit(control, target), {control, target})
'Analytic five-CZ, seven-moment square preparation of D(4,2).\n\nDiscovered by offline topology search; the final circuit has exact closed-form\nangles and an independent symbolic amplitude proof in verify_exact4.py.\nThe four qubits must be ordered clockwise around a fully coupled square.\nNo numerical optimization is performed when constructing this circuit.\n'
import math as _cf_math
import cirq as _cf_cirq

def _cf_dicke4_square(qubits):
    """Return D(4,2) on a,b,c,d with grid edges ab,bc,cd,da."""
    if len(qubits) != 4 or len(set(qubits)) != 4:
        raise ValueError('Need four distinct qubits around a square.')
    a, b, c, d = qubits

    def ry(q, theta):
        return _cf_cirq.PhasedXZGate(x_exponent=theta / _cf_math.pi, z_exponent=0, axis_phase_exponent=0.5).on(q)
    return _cf_cirq.Circuit([_cf_cirq.Moment([ry(a, _cf_math.acos(-1 / 3)), ry(b, _cf_math.pi / 4), ry(c, _cf_math.pi / 2), ry(d, -_cf_math.pi / 2)]), _cf_cirq.Moment([_cf_cirq.CZ(a, b), _cf_cirq.CZ(c, d)]), _cf_cirq.Moment([ry(b, _cf_math.pi / 4), _cf_cirq.CZ(d, a)]), _cf_cirq.Moment([_cf_cirq.CZ(b, c), ry(d, _cf_math.pi / 2), ry(a, -_cf_math.pi / 2)]), _cf_cirq.Moment([ry(b, _cf_math.pi / 2)]), _cf_cirq.Moment([_cf_cirq.CZ(a, b)]), _cf_cirq.Moment([ry(a, _cf_math.pi / 2)])])
'Exact analytic D(4,2) with calibration-aware square selection.'
import itertools as _cf_itertools
import math as _cf_math
import cirq as _cf_cirq
import numpy as _cf_np

def _cf__actual(mapping, key):
    value = mapping[key]
    return float(value[0])

def _cf_square_candidates(cz, sq, zero, one):
    graph = _cf_calibration_graph(cz, sq)
    candidates = []
    for a in graph:
        for b in graph[a]:
            for c in graph[b]:
                if c == a:
                    continue
                for d in graph[c]:
                    if d in (a, b) or a not in graph[d]:
                        continue
                    qs = (a, b, c, d)
                    score = sum((w * _cf_rate(sq[q]) for q, w in zip(qs, (3, 3, 1, 2))))
                    score += 2 * graph[a][b] + graph[b][c] + graph[c][d] + graph[d][a]
                    score += sum(((_cf_rate(zero.get(q, (0.01, 0.0))) + _cf_rate(one.get(q, (0.02, 0.0)))) / 2 for q in qs))
                    ds = [_cf_math.log1p(-_cf_rate(one.get(q, (0.02, 0.0)))) - _cf_math.log1p(-_cf_rate(zero.get(q, (0.01, 0.0)))) for q in qs]
                    score += 10000 / 18 * sum(((d - sum(ds) / 4) ** 2 for d in ds))
                    candidates.append((score, qs))
    return sorted(candidates, key=lambda row: (row[0], row[1]))

class _cf__SmallNoise(_cf_cirq.NoiseModel):

    def __init__(self, cz, sq):
        self.sq = sq
        self.cz = {}
        for pair, v in cz.items():
            key = frozenset(pair)
            self.cz[key] = max(self.cz.get(key, 0.0), float(v[0]))

    def noisy_operation(self, op):
        p = _cf__actual(self.sq, op.qubits[0]) if len(op.qubits) == 1 else self.cz[frozenset(op.qubits)]
        return [op, _cf_cirq.depolarize(p=min(1.0, max(0.0, p)), n_qubits=len(op.qubits)).on(*op.qubits)]

def _cf__readout(probabilities, qs, zero, one, basis_noise=None):
    p = probabilities.reshape((2,) * len(qs))
    for i, q in enumerate(qs):
        e0, e1 = (float(zero.get(q, (0.01, 0.0))[0]), float(one.get(q, (0.02, 0.0))[0]))
        if basis_noise is not None:
            extra = 2 * float(basis_noise[q][0]) / 3 * (1 - e0 - e1)
            e0, e1 = (e0 + extra, e1 + extra)
        m = _cf_np.array([[1 - e0, e1], [e0, 1 - e1]])
        p = _cf_np.moveaxis(_cf_np.tensordot(m, p, axes=(1, i)), 0, i)
    return p.reshape(-1)

def _cf_small_proxy(circuit, qs, k, cz, sq, zero, one):
    """Expected official denominator under a small independent-Pauli model."""
    n = len(qs)
    zrho = _cf_native_density(circuit, qs, cz, sq)
    xrho = zrho.reshape((2,) * (2 * n))
    for axis in range(2 * n):
        a, b = (_cf_np.take(xrho, 0, axis=axis), _cf_np.take(xrho, 1, axis=axis))
        xrho = _cf_np.stack(((a + b) / _cf_math.sqrt(2), (a - b) / _cf_math.sqrt(2)), axis=axis)
    xrho = xrho.reshape((2 ** n, 2 ** n))
    z = _cf__readout(_cf_np.real(_cf_np.diag(zrho)), qs, zero, one)
    x = _cf__readout(_cf_np.real(_cf_np.diag(xrho)), qs, zero, one, basis_noise=sq)
    indices = [i for i in range(2 ** n) if i.bit_count() == k]
    f = float(sum((z[i] for i in indices)))
    if f <= 0:
        return float('inf')
    m = _cf_math.comb(n, k)
    delta = m * sum(((z[i] / f) ** 2 for i in indices)) - 1
    shots = 10000
    inv = 1 / (shots * f) + (1 - f) / (shots * f) ** 2 + (1 - f) * (2 - f) / (shots * f) ** 3
    expected_z = shots / m * (delta + (m - 1 - delta) * inv)
    expected_x = 0
    for ell in range(1, n + 1):

        def comb(a, b):
            return _cf_math.comb(a, b) if 0 <= b <= a else 0
        expected = comb(ell, ell // 2) * comb(n - ell, k - ell // 2) / m if ell % 2 == 0 else 0.0
        vals = _cf_np.array([sum(((-1) ** j * comb(i.bit_count(), j) * comb(n - i.bit_count(), ell - j) for j in range(ell + 1))) / comb(n, ell) for i in range(2 ** n)])
        mean = float(_cf_np.dot(x, vals))
        var = max(0.0, float(_cf_np.dot(x, vals ** 2)) - mean ** 2)
        expected_x += ((expected - mean) ** 2 + var / shots) / (expected ** 2 if expected else 1.0)
    return float(expected_z + expected_x)

def _cf_create_dicke4(cz, sq, zero, one, refine=12):
    ranked = _cf_square_candidates(cz, sq, zero, one)
    if not ranked:
        raise NotImplementedError('No intact calibrated square')
    best = None
    for _, qs in ranked[:refine]:
        circuit = _cf_dicke4_square(qs)
        proxy = _cf_small_proxy(circuit, qs, 2, cz, sq, zero, one)
        if best is None or proxy < best[0]:
            best = (proxy, circuit, set(qs))
    return (best[1], best[2])
'Conservative exact echo pairs, preserving every original moment.'
import cirq as _cf_cirq

def _cf_echo_idle_qubits(circuit, min_idle_moments=18, single_qubit_moments_only=True):
    """Insert paired X pulses inside long idle intervals after first entanglement.

    Both pulses lie between consecutive operations on that qubit, so the
    ideal transformation is exactly unchanged. Candidate times approximate
    one-quarter and three-quarters of the interval's physical duration.
    Does not assume an unavailable Cirq-Google decoupling tag. This helps
    quasistatic dephasing, not Markovian dephasing, and adds pulse error.
    """
    moments = [list(m.operations) for m in circuit]
    durations = [42 if any((len(op.qubits) == 2 for op in m)) else 25 for m in moments]
    x = _cf_cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)
    for q in sorted(circuit.all_qubits()):
        uses = [i for i, m in enumerate(circuit) if m.operates_on([q])]
        entanglers = [i for i, m in enumerate(circuit) if any((q in op.qubits and len(op.qubits) > 1 for op in m))]
        if not entanglers:
            continue
        first_entangler = entanglers[0]
        for left, right in zip(uses, uses[1:] + [len(circuit)]):
            if left < first_entangler or right - left - 1 < min_idle_moments:
                continue
            candidates = [i for i in range(left + 1, right) if not single_qubit_moments_only or durations[i] == 25]
            if len(candidates) < 2:
                continue
            centers = {}
            total = 0
            for i in range(left + 1, right):
                centers[i] = total + durations[i] / 2
                total += durations[i]
            first = min(candidates, key=lambda i: abs(centers[i] - total / 4))
            second = min((i for i in candidates if i > first), key=lambda i: abs(centers[i] - 3 * total / 4), default=None)
            if second is None:
                continue
            moments[first].append(x(q))
            moments[second].append(x(q))
    return _cf_cirq.Circuit((_cf_cirq.Moment(m) for m in moments))
'Frozen native compilations of the exact balanced-complement construction.'
_cf_BALANCED_TEMPLATES = {6: {'qubits': [(4, 3), (4, 4), (5, 3), (5, 4), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.6309898804344547, 0.0, 0.5], ['sq', 1, 0.12337585721442501, -1.0, -0.5], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.8040867239846965, 0.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 2], ['sq', 1, 0.12337585721442501, -1.0, -0.5]], [['sq', 0, 0.5, 0.0, 0.0], ['cz', 1, 3], ['sq', 2, 0.5, -1.0, 0.5]], [['cz', 2, 4], ['sq', 3, 0.0, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.8040867239846965, 0.0, -0.5]], [['cz', 0, 1], ['cz', 3, 2]], [['sq', 0, 0.5, 0.0, 0.0], ['sq', 1, 0.25, 0.0, 0.9999999999999991], ['sq', 2, 0.19591327601530373, 0.0, 0.5], ['sq', 3, 0.6959132760153035, 0.0, 0.5]], [['cz', 0, 1], ['cz', 2, 4]], [['sq', 0, 0.5, 0.0, -0.5], ['sq', 1, 0.24999999999999978, 0.5000000000000004, -4.440892098500626e-16], ['sq', 2, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5]], [['cz', 3, 2]], [['cz', 2, 4], ['sq', 3, 0.5, 0.0, -0.5]], [['sq', 2, 0.25, 0.0, -0.5], ['sq', 4, 0.25, 0.0, 0.5]], [['cz', 2, 4]], [['sq', 2, 0.0, -1.0, -0.5], ['sq', 4, 0.0, -1.0, -0.5]], [['cz', 3, 2]], [['cz', 3, 1]], [['sq', 3, 0.5, -1.0, -0.5]], [['cz', 5, 3]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 5, 4]], [['cz', 3, 1], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 5, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 3, 2]], [['cz', 1, 0], ['sq', 2, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5]]]}, 8: {'qubits': [(4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.7804574844601513, 0.0, 0.5], ['sq', 1, 0.22421722475922912, -1.0, -0.5], ['sq', 2, 0.08945618751103335, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, 0.0], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 3], ['sq', 1, 0.22421722475922912, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 3, 0.5, 0.5, 0.0]], [['cz', 1, 4], ['sq', 2, 0.08945618751103357, -1.0, -0.5], ['cz', 3, 6]], [['sq', 1, 0.5, 0.0, -0.5], ['cz', 2, 5], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 6, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 2, 0.5, 0.0, -0.5], ['cz', 3, 6]], [['cz', 1, 2], ['sq', 3, 0.5, 0.0, 0.0], ['sq', 6, 0.5, 0.0, -1.0]], [['sq', 1, 0.19591327601530373, 0.0, -0.5000000000000002], ['sq', 2, 0.1959132760153035, 0.0, 0.5], ['cz', 3, 4]], [['cz', 1, 2], ['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, 0.0, 0.5], ['sq', 2, 0.5, 0.0, 0.5], ['cz', 6, 3]], [['cz', 1, 0], ['sq', 3, 0.125, 0.0, -0.5000000000000002]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.40204336199234825, -1.0, -0.5], ['cz', 4, 3]], [['cz', 2, 1], ['sq', 3, 0.125, 0.0, 0.5], ['sq', 4, 0.5, -1.0, -0.5]], [['sq', 1, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 6, 3]], [['cz', 0, 1], ['sq', 3, 0.375, -1.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.09795663800765197, 0.0, 0.5], ['cz', 3, 4]], [['cz', 2, 1], ['sq', 4, 0.5, -1.0, -0.5]], [['sq', 1, 0.40204336199234825, -1.0, -0.5], ['cz', 4, 5], ['sq', 2, 0.49999999999999956, -0.49999999999999956, 0.49999999999999956]], [['cz', 1, 0], ['sq', 4, 0.41666666666666674, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 3, 4], ['sq', 1, 0.5, 0.5, -0.5]], [['sq', 4, 0.08333333333333348, 0.0, -0.5000000000000002]], [['cz', 1, 2], ['cz', 5, 4]], [['sq', 1, 0.5, 0.0, 0.5], ['sq', 2, 0.25, 0.0, 0.0], ['sq', 4, 0.08333333333333326, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 1, 2], ['cz', 3, 4]], [['sq', 1, 0.5, -0.75, 0.75], ['sq', 2, 1.0, 0.0, 0.5], ['sq', 3, 0.5, 0.0, -0.5], ['sq', 4, 0.41666666666666674, -1.0, -0.5]], [['cz', 1, 0], ['cz', 3, 6], ['cz', 4, 5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.19591327601530373, 0.0, -0.5000000000000002], ['sq', 4, 0.5, -1.0, -0.5], ['sq', 6, 0.1959132760153035, 0.0, 0.5]], [['cz', 3, 6]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 6, 0.5, 0.0, 0.5]], [['cz', 3, 4]], [['sq', 3, 0.40204336199234825, -1.0, -0.5], ['sq', 4, 0.5, -1.0, -0.5]], [['cz', 6, 3]], [['sq', 3, 0.09795663800765175, 0.0, -0.5000000000000002]], [['cz', 4, 3]], [['sq', 3, 0.09795663800765197, 0.0, 0.5], ['sq', 4, 0.5, -1.0, -0.5]], [['cz', 6, 3]], [['sq', 3, 0.40204336199234825, -1.0, -0.5], ['sq', 6, 0.5, -1.1102230246251565e-16, -4.440892098500626e-16]], [['cz', 3, 4]], [['sq', 3, 0.5, 0.0, 0.0], ['sq', 4, 0.5, -1.0, -0.5]], [['cz', 3, 6], ['cz', 4, 5]], [['sq', 3, 0.25, 0.0, -1.1102230246251565e-16], ['sq', 6, 0.25, 0.0, -1.0]], [['cz', 3, 6]], [['sq', 3, 0.5, 0.5, -0.5], ['sq', 6, 0.5, 0.5, -0.5]], [['cz', 4, 3]], [['cz', 4, 1]], [['sq', 4, 0.5, -1.0, -0.5]], [['cz', 7, 4]], [['sq', 4, 0.5, -1.0, -0.5], ['cz', 7, 6]], [['cz', 4, 1], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 4, 3]], [['cz', 1, 0], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 4, 5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5]]]}, 10: {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (5, 3), (5, 4), (5, 5), (5, 6), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.8723282232918721, 0.0, 0.5], ['sq', 1, 0.3050177709578634, -1.0, -0.5], ['sq', 2, 0.17004943144282958, -1.0, -0.5], ['sq', 3, 0.070024348046551, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 4], ['sq', 1, 0.3050177709578634, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 4, 0.0, -1.0, -0.5]], [['cz', 1, 5], ['sq', 2, 0.17004943144282958, -1.0, -0.5], ['cz', 4, 8]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 4, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 8, 0.14758361765043326, 0.0, 0.5]], [['cz', 2, 6], ['sq', 3, 0.070024348046551, -1.0, -0.5], ['cz', 4, 8]], [['sq', 2, 0.5, 0.0, -1.0], ['cz', 3, 7], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 8, 0.5, 0.0, 0.5]], [['sq', 3, 0.5, 0.0, 0.0], ['cz', 4, 5]], [['cz', 2, 3], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 2, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 8, 4]], [['cz', 2, 3], ['sq', 4, 0.14102355421224377, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, 0.0, 0.0], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 5, 4]], [['cz', 2, 1], ['sq', 4, 0.14102355421224377, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.375, -1.0, -0.5], ['cz', 8, 4]], [['cz', 3, 2], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['sq', 8, 0.5, 0.0, 0.0]], [['sq', 2, 0.125, 0.0, -0.5000000000000002], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.125, 0.0, 0.5], ['cz', 5, 6]], [['cz', 3, 2], ['sq', 5, 0.3910235542122438, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.375, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['cz', 4, 5]], [['cz', 2, 1], ['sq', 5, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 6, 5]], [['cz', 1, 0], ['sq', 5, 0.10897644578775623, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.41666666666666674, -1.0, -0.5], ['cz', 4, 5]], [['cz', 2, 1], ['sq', 4, 0.5, 0.0, -1.0], ['sq', 5, 0.3910235542122438, -1.0, -0.5]], [['sq', 1, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 4, 8], ['cz', 5, 6]], [['cz', 0, 1], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.08333333333333326, 0.0, 0.5], ['cz', 4, 8], ['cz', 6, 7]], [['cz', 2, 1], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 6, 0.4262081911747835, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, 0.0, -1.0]], [['sq', 1, 0.41666666666666674, -1.0, -0.5], ['sq', 2, 0.5, 0.0, -0.5], ['cz', 5, 6]], [['cz', 1, 0], ['cz', 2, 3], ['sq', 6, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.19591327601530373, 0.0, -0.5000000000000002], ['sq', 3, 0.1959132760153035, 0.0, 0.5], ['cz', 7, 6]], [['cz', 2, 3], ['sq', 6, 0.07379180882521652, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 3, 0.5, 0.0, 0.5], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.4262081911747835, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.40204336199234825, -1.0, -0.5], ['cz', 4, 5], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 8, 4]], [['cz', 1, 2], ['sq', 4, 0.125, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.09795663800765197, 0.0, 0.5], ['cz', 5, 4]], [['cz', 3, 2], ['sq', 4, 0.125, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['sq', 2, 0.40204336199234825, -1.0, -0.5], ['cz', 8, 4], ['sq', 3, 0.5000000000000004, 0.5000000000000004, -0.5000000000000004]], [['cz', 2, 1], ['sq', 4, 0.375, -1.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 4, 5], ['sq', 2, 0.5, 0.5, -0.5]], [['sq', 5, 0.5, -1.0, -0.5]], [['cz', 2, 3], ['cz', 5, 6]], [['sq', 2, 0.5, 0.0, 0.5], ['sq', 3, 0.25, 0.0, -1.0], ['sq', 5, 0.41666666666666674, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 2, 3], ['cz', 4, 5]], [['sq', 2, 0.5, 0.25, -0.25], ['sq', 3, 0.0, -1.0, -0.5], ['sq', 5, 0.08333333333333348, 0.0, -0.5000000000000002]], [['cz', 1, 2], ['cz', 6, 5]], [['cz', 1, 0], ['sq', 5, 0.08333333333333326, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 4, 5]], [['sq', 4, 0.5, 0.0, -0.5], ['sq', 5, 0.41666666666666674, -1.0, -0.5]], [['cz', 4, 8], ['cz', 5, 6]], [['sq', 4, 0.19591327601530373, 0.0, -0.5000000000000002], ['sq', 5, 0.5, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.1959132760153035, 0.0, 0.5]], [['cz', 4, 8], ['cz', 6, 7]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.5, 0.0, 0.5]], [['cz', 4, 5]], [['sq', 4, 0.40204336199234825, -1.0, -0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 8, 4]], [['sq', 4, 0.09795663800765175, 0.0, -0.5000000000000002]], [['cz', 5, 4]], [['sq', 4, 0.09795663800765197, 0.0, 0.5], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 8, 4]], [['sq', 4, 0.40204336199234825, -1.0, -0.5], ['sq', 8, 0.5, -1.1102230246251565e-16, -4.440892098500626e-16]], [['cz', 4, 5]], [['sq', 4, 0.5, 0.0, 0.0], ['sq', 5, 0.5, -1.0, -0.5]], [['cz', 4, 8], ['cz', 5, 6]], [['sq', 4, 0.25, 0.0, -1.1102230246251565e-16], ['sq', 8, 0.25, 0.0, -1.0]], [['cz', 4, 8]], [['sq', 4, 0.5, 0.5, -0.5], ['sq', 8, 0.5, 0.5, -0.5]], [['cz', 5, 4]], [['cz', 5, 1]], [['sq', 5, 0.5, -1.0, -0.5]], [['cz', 9, 5]], [['sq', 5, 0.5, -1.0, -0.5], ['cz', 9, 8]], [['cz', 5, 1], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 5, 4]], [['cz', 1, 0], ['sq', 4, 0.5, -1.0, -0.5], ['cz', 5, 6]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 6, 7]], [['cz', 2, 3], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5]]]}, 12: {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.9272924851969222, 0.0, 0.5], ['sq', 1, 0.3670787763403043, -1.0, -0.5], ['sq', 2, 0.24205984594556806, -1.0, -0.5], ['sq', 3, 0.1358157355171723, -1.0, -0.5], ['sq', 4, 0.057482058975516725, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 5], ['sq', 1, 0.3670787763403043, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 5, 0.0, -1.0, -0.5]], [['cz', 1, 6], ['sq', 2, 0.24205984594556806, -1.0, -0.5], ['cz', 5, 10]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 5, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 10, 0.13386023640061495, 0.0, 0.5]], [['cz', 2, 7], ['sq', 3, 0.1358157355171723, -1.0, -0.5], ['cz', 5, 10]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['cz', 3, 8], ['sq', 4, 0.0574820589755165, -1.0, -0.5], ['cz', 5, 6]], [['sq', 3, 0.5, 0.0, -0.5], ['cz', 4, 9], ['sq', 5, 0.34795663800765175, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 10, 5]], [['cz', 3, 4], ['sq', 5, 0.15204336199234825, 0.0, -0.5000000000000002]], [['sq', 3, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 4, 0.14758361765043326, 0.0, 0.5], ['cz', 6, 5]], [['cz', 3, 4], ['sq', 5, 0.15204336199234825, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 10, 5]], [['cz', 3, 2], ['sq', 5, 0.34795663800765175, -1.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3589764457877562, -1.0, -0.5], ['cz', 5, 6]], [['cz', 4, 3], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 3, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 2, 3], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.14102355421224377, 0.0, 0.5], ['cz', 5, 6]], [['cz', 4, 3], ['sq', 6, 0.125, 0.0, -0.5000000000000002]], [['sq', 3, 0.3589764457877562, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['cz', 7, 6]], [['cz', 3, 2], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.375, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.3910235542122438, -1.0, -0.5], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 5, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.14758361765043326, 0.0, 0.5]], [['sq', 2, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 5, 10], ['cz', 7, 8]], [['cz', 1, 2], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 7, 0.40204336199234825, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.10897644578775623, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 7, 0.09795663800765175, 0.0, -0.5000000000000002]], [['sq', 2, 0.3910235542122438, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -1.0], ['cz', 8, 7]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 7, 0.09795663800765197, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 6, 7]], [['cz', 1, 0], ['cz', 3, 4], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.40204336199234825, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, 0.0, 0.0], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 2, 1], ['sq', 5, 0.3589764457877562, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 1, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 8, 9], ['cz', 10, 5]], [['cz', 0, 1], ['sq', 5, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 8, 0.4330698817996925, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.07379180882521652, 0.0, 0.5], ['cz', 6, 5], ['cz', 7, 8]], [['cz', 2, 1], ['sq', 5, 0.14102355421224377, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 8, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 1, 0.4262081911747835, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['cz', 9, 8], ['cz', 10, 5]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 5, 0.3589764457877562, -1.0, -0.5], ['sq', 8, 0.06693011820030748, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.0]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 4, 3], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4330698817996925, -1.0, -0.5]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 6, 0.3910235542122438, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 5, 6]], [['cz', 4, 3], ['sq', 6, 0.10897644578775623, 0.0, -0.5000000000000002]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['cz', 7, 6]], [['cz', 3, 2], ['sq', 6, 0.10897644578775623, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 5, 6]], [['cz', 2, 1], ['sq', 5, 0.5, 0.0, -1.0], ['sq', 6, 0.3910235542122438, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.41666666666666674, -1.0, -0.5], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 2, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 5, 10], ['cz', 7, 8]], [['cz', 1, 2], ['sq', 5, 0.5, 0.0, 0.0], ['sq', 7, 0.4262081911747835, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.08333333333333326, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 2], ['sq', 7, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 2, 0.41666666666666674, -1.0, -0.5], ['sq', 3, 0.5, 0.0, -0.5], ['cz', 8, 7]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 7, 0.07379180882521652, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.19591327601530373, 0.0, -0.5000000000000002], ['sq', 4, 0.1959132760153035, 0.0, 0.5], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.4262081911747835, -1.0, -0.5]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.5, 0.0, 0.5], ['cz', 5, 6], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 5, 0.375, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.40204336199234825, -1.0, -0.5], ['cz', 8, 9], ['cz', 10, 5]], [['cz', 4, 3], ['sq', 5, 0.125, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 6, 5]], [['cz', 2, 3], ['sq', 5, 0.125, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.09795663800765197, 0.0, 0.5], ['cz', 10, 5]], [['cz', 4, 3], ['sq', 5, 0.375, -1.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5]], [['sq', 3, 0.40204336199234825, -1.0, -0.5], ['cz', 5, 6], ['sq', 4, 0.5000000000000004, 0.5000000000000004, -0.5000000000000004]], [['cz', 3, 2], ['sq', 6, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 6, 7], ['sq', 3, 0.5, 0.5, -0.5]], [['sq', 6, 0.41666666666666674, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 3, 4], ['cz', 5, 6]], [['sq', 3, 0.5, 0.0, 0.5], ['sq', 4, 0.25, 0.0, -1.0], ['sq', 6, 0.08333333333333348, 0.0, -0.5000000000000002]], [['cz', 3, 4], ['cz', 7, 6]], [['sq', 3, 0.5, 0.25, -0.25], ['sq', 4, 0.0, -1.0, -0.5], ['sq', 6, 0.08333333333333326, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 2, 3], ['cz', 5, 6]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['sq', 6, 0.41666666666666674, -1.0, -0.5]], [['cz', 1, 2], ['cz', 5, 10], ['cz', 6, 7]], [['cz', 1, 0], ['sq', 5, 0.19591327601530373, 0.0, -0.5000000000000002], ['sq', 6, 0.5, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.1959132760153035, 0.0, 0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 5, 10], ['cz', 7, 8]], [['sq', 5, 0.5, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 10, 0.5, 0.0, 0.5]], [['cz', 5, 6]], [['sq', 5, 0.40204336199234825, -1.0, -0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 10, 5]], [['sq', 5, 0.09795663800765175, 0.0, -0.5000000000000002]], [['cz', 6, 5]], [['sq', 5, 0.09795663800765197, 0.0, 0.5], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 10, 5]], [['sq', 5, 0.40204336199234825, -1.0, -0.5], ['sq', 10, 0.5, -1.1102230246251565e-16, -4.440892098500626e-16]], [['cz', 5, 6]], [['sq', 5, 0.5, 0.0, 0.0], ['sq', 6, 0.5, -1.0, -0.5]], [['cz', 5, 10], ['cz', 6, 7]], [['sq', 5, 0.25, 0.0, -1.1102230246251565e-16], ['sq', 10, 0.25, 0.0, -1.0]], [['cz', 5, 10]], [['sq', 5, 0.5, 0.5, -0.5], ['sq', 10, 0.5, 0.5, -0.5]], [['cz', 6, 5]], [['cz', 6, 1]], [['sq', 6, 0.5, -1.0, -0.5]], [['cz', 11, 6]], [['sq', 6, 0.5, -1.0, -0.5], ['cz', 11, 10]], [['cz', 6, 1], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 11, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 6, 5]], [['cz', 1, 0], ['sq', 5, 0.5, -1.0, -0.5], ['cz', 6, 7]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 7, 8]], [['cz', 2, 3], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 8, 9]], [['cz', 3, 4], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, -1.0, -0.5]]]}, 14: {'qubits': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 3), (6, 4)], 'moments': [[['sq', 0, 0.9593120124343613, 0.0, 0.5], ['sq', 1, 0.4124711229199527, -1.0, -0.5], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.5, 0.0, -0.5], ['sq', 8, 0.5, 0.0, -0.5], ['sq', 9, 0.5, 0.0, -0.5], ['sq', 10, 0.5, 0.0, -0.5], ['sq', 11, 0.5, 0.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5], ['sq', 13, 0.5, -1.0, -0.5]], [['cz', 0, 1]], [['cz', 0, 6], ['sq', 1, 0.4124711229199527, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 6, 0.0, -1.0, -0.5]], [['cz', 1, 7], ['sq', 2, 0.3046558254995755, -1.0, -0.5], ['cz', 6, 12]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 2, 3], ['sq', 6, 0.12337585721442501, 0.0, -0.5], ['sq', 12, 0.12337585721442501, 0.0, 0.5]], [['cz', 2, 8], ['sq', 3, 0.1976120242455861, -1.0, -0.5], ['cz', 6, 12]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 3, 4], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['cz', 3, 9], ['sq', 4, 0.11265416171383746, -1.0, -0.5], ['cz', 6, 7]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 4, 5], ['sq', 6, 0.33975425899284417, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 4, 10], ['sq', 5, 0.048731945757486184, -1.0, -0.5], ['cz', 12, 6]], [['sq', 4, 0.5, 0.0, -0.5], ['cz', 5, 11], ['sq', 6, 0.16024574100715605, 0.0, -0.5000000000000002]], [['sq', 5, 0.5, 0.0, -0.5], ['cz', 7, 6]], [['cz', 4, 5], ['sq', 6, 0.16024574100715605, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 4, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 5, 0.13386023640061495, 0.0, 0.5], ['cz', 12, 6]], [['cz', 4, 5], ['sq', 6, 0.33975425899284417, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.34795663800765175, -1.0, -0.5], ['cz', 7, 8]], [['cz', 5, 4], ['sq', 7, 0.36359276291425235, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.15204336199234825, 0.0, -0.5000000000000002], ['cz', 6, 7]], [['cz', 3, 4], ['sq', 7, 0.13640723708574742, 0.0, -0.5000000000000002]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.15204336199234825, 0.0, 0.5], ['cz', 8, 7]], [['cz', 5, 4], ['sq', 7, 0.13640723708574742, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.34795663800765175, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.3635927629142526, -1.0, -0.5000000000000002]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 6, 0.13386023640061495, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.13386023640061495, 0.0, 0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.375, -1.0, -0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 4, 3], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.38640723708574765, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 3, 0.125, 0.0, -0.5000000000000002], ['cz', 7, 8]], [['cz', 2, 3], ['sq', 8, 0.11359276291425258, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.125, 0.0, 0.5], ['cz', 9, 8]], [['cz', 4, 3], ['sq', 8, 0.11359276291425258, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.375, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['cz', 7, 8]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.3864072370857474, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 5, 0.14758361765043326, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 3, 2], ['sq', 6, 0.15204336199234825, 0.0, -0.5000000000000002], ['sq', 9, 0.41024574100715605, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 2, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 1, 2], ['sq', 6, 0.15204336199234825, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.09795663800765197, 0.0, 0.5], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 3, 2], ['sq', 6, 0.34795663800765175, -1.0, -0.5], ['sq', 9, 0.08975425899284395, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 2, 0.40204336199234825, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.41024574100715605, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 1, 0], ['cz', 5, 4], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 10, 11]], [['cz', 2, 1], ['cz', 3, 4], ['sq', 7, 0.125, 0.0, -0.5000000000000002], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 1, 0.06693011820030748, 0.0, -0.5000000000000002], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.14102355421224377, 0.0, 0.5], ['cz', 8, 7], ['cz', 9, 10]], [['cz', 0, 1], ['cz', 5, 4], ['sq', 7, 0.125, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, -0.5000000000000002]], [['sq', 0, 0.5, -1.0, -0.5], ['sq', 1, 0.06693011820030748, 0.0, 0.5], ['sq', 4, 0.3589764457877562, -1.0, -0.5], ['sq', 5, 0.5, 0.0, 0.0], ['cz', 6, 7], ['cz', 11, 10]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.375, -1.0, -0.5], ['sq', 10, 0.061687928607212505, 0.0, 0.5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 1, 0.4330698817996925, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 12], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 1, 0], ['cz', 3, 2], ['sq', 6, 0.14758361765043326, 0.0, -0.5000000000000002], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.4383120713927875, -1.0, -0.5], ['sq', 12, 0.14758361765043326, 0.0, 0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.3910235542122438, -1.0, -0.5], ['cz', 6, 12], ['cz', 8, 9], ['cz', 10, 11]], [['cz', 4, 3], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['sq', 3, 0.10897644578775623, 0.0, -0.5000000000000002], ['cz', 7, 8]], [['cz', 2, 3], ['sq', 8, 0.09795663800765175, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.10897644578775623, 0.0, 0.5], ['cz', 9, 8]], [['cz', 4, 3], ['sq', 8, 0.09795663800765197, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.3910235542122438, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -1.0], ['cz', 7, 8]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.40204336199234825, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 4, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 5, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 1], ['cz', 4, 5], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 4, 0.5, 0.0, 0.0], ['sq', 5, 0.5, 0.0, -1.0], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 3, 2], ['sq', 6, 0.14102355421224377, 0.0, -0.5000000000000002], ['sq', 9, 0.4330698817996925, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 2, 0.07379180882521652, 0.0, -0.5000000000000002], ['cz', 7, 6], ['cz', 8, 9]], [['cz', 1, 2], ['sq', 6, 0.14102355421224377, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, -0.5000000000000002]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.07379180882521652, 0.0, 0.5], ['cz', 10, 9], ['cz', 12, 6]], [['cz', 3, 2], ['sq', 6, 0.3589764457877562, -1.0, -0.5], ['sq', 9, 0.06693011820030748, 0.0, 0.5], ['sq', 10, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.0]], [['sq', 2, 0.4262081911747835, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 2, 1], ['cz', 4, 3], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.4330698817996925, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.375, -1.0, -0.5], ['cz', 7, 8], ['cz', 9, 10]], [['cz', 5, 4], ['sq', 7, 0.3910235542122438, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 4, 0.125, 0.0, -0.5000000000000002], ['cz', 6, 7], ['cz', 10, 11]], [['cz', 3, 4], ['sq', 7, 0.10897644578775623, 0.0, -0.5000000000000002], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.125, 0.0, 0.5], ['cz', 8, 7]], [['cz', 5, 4], ['sq', 7, 0.10897644578775623, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 4, 0.375, -1.0, -0.5], ['sq', 5, 0.5, 0.0, -0.5], ['cz', 6, 7]], [['cz', 4, 3], ['sq', 6, 0.5, 0.0, -1.0], ['sq', 7, 0.3910235542122438, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 12], ['cz', 7, 8]], [['cz', 3, 2], ['sq', 6, 0.16666666666666674, 0.0, -1.1102230246251565e-16], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.16666666666666674, 0.0, -1.1102230246251565e-16]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.41666666666666674, -1.0, -0.5], ['cz', 6, 12], ['cz', 8, 9]], [['cz', 4, 3], ['sq', 6, 0.5, 0.0, 0.0], ['sq', 8, 0.4262081911747835, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -1.0]], [['sq', 3, 0.08333333333333348, 0.0, -0.5000000000000002], ['cz', 7, 8]], [['cz', 2, 3], ['sq', 8, 0.07379180882521652, 0.0, -0.5000000000000002]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.08333333333333326, 0.0, 0.5], ['cz', 9, 8]], [['cz', 4, 3], ['sq', 8, 0.07379180882521652, 0.0, 0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.41666666666666674, -1.0, -0.5], ['sq', 4, 0.5, 0.0, -0.5], ['cz', 7, 8]], [['cz', 3, 2], ['cz', 4, 5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.4262081911747835, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.19591327601530373, 0.0, -0.5000000000000002], ['sq', 5, 0.1959132760153035, 0.0, 0.5], ['cz', 6, 7], ['cz', 8, 9]], [['cz', 4, 5], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['cz', 9, 10], ['cz', 12, 6]], [['cz', 4, 3], ['sq', 6, 0.125, 0.0, -0.5000000000000002], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.40204336199234825, -1.0, -0.5], ['cz', 7, 6]], [['cz', 5, 4], ['sq', 6, 0.125, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 4, 0.09795663800765175, 0.0, -0.5000000000000002], ['cz', 12, 6]], [['cz', 3, 4], ['sq', 6, 0.375, -1.0, -0.5], ['sq', 12, 0.5, 0.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 4, 0.09795663800765197, 0.0, 0.5], ['cz', 6, 7]], [['cz', 5, 4], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 4, 0.40204336199234825, -1.0, -0.5], ['cz', 7, 8], ['sq', 5, 0.5000000000000004, 0.5000000000000004, -0.5000000000000004]], [['cz', 4, 3], ['sq', 7, 0.41666666666666674, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 6, 7], ['sq', 4, 0.5, 0.5, -0.5]], [['sq', 7, 0.08333333333333348, 0.0, -0.5000000000000002]], [['cz', 4, 5], ['cz', 8, 7]], [['sq', 4, 0.5, 0.0, 0.5], ['sq', 5, 0.25, 0.0, -1.0], ['sq', 7, 0.08333333333333326, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5]], [['cz', 4, 5], ['cz', 6, 7]], [['sq', 4, 0.5, 0.25, -0.25], ['sq', 5, 0.0, -1.0, -0.5], ['sq', 6, 0.5, 0.0, -0.5], ['sq', 7, 0.41666666666666674, -1.0, -0.5]], [['cz', 3, 4], ['cz', 6, 12], ['cz', 7, 8]], [['sq', 3, 0.5, -1.0, -0.5], ['sq', 6, 0.19591327601530373, 0.0, -0.5000000000000002], ['sq', 7, 0.5, -1.0, -0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.1959132760153035, 0.0, 0.5]], [['cz', 2, 3], ['cz', 6, 12], ['cz', 8, 9]], [['sq', 2, 0.5, -1.0, -0.5], ['sq', 6, 0.5, 0.0, 0.5], ['sq', 8, 0.5, -1.0, -0.5], ['sq', 12, 0.5, 0.0, 0.5]], [['cz', 1, 2], ['cz', 6, 7]], [['cz', 1, 0], ['sq', 6, 0.40204336199234825, -1.0, -0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 12, 6]], [['sq', 6, 0.09795663800765175, 0.0, -0.5000000000000002]], [['cz', 7, 6]], [['sq', 6, 0.09795663800765197, 0.0, 0.5], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 12, 6]], [['sq', 6, 0.40204336199234825, -1.0, -0.5], ['sq', 12, 0.5, -1.1102230246251565e-16, -4.440892098500626e-16]], [['cz', 6, 7]], [['sq', 6, 0.5, 0.0, 0.0], ['sq', 7, 0.5, -1.0, -0.5]], [['cz', 6, 12], ['cz', 7, 8]], [['sq', 6, 0.25, 0.0, -1.1102230246251565e-16], ['sq', 12, 0.25, 0.0, -1.0]], [['cz', 6, 12]], [['sq', 6, 0.5, 0.5, -0.5], ['sq', 12, 0.5, 0.5, -0.5]], [['cz', 7, 6]], [['cz', 7, 1]], [['sq', 7, 0.5, -1.0, -0.5]], [['cz', 13, 7]], [['sq', 7, 0.5, -1.0, -0.5], ['cz', 13, 12]], [['cz', 7, 1], ['sq', 12, 0.5, -1.0, -0.5], ['sq', 13, 1.0, 0.0, -1.0]], [['sq', 1, 0.5, -1.0, -0.5], ['cz', 7, 6]], [['cz', 1, 0], ['sq', 6, 0.5, -1.0, -0.5], ['cz', 7, 8]], [['sq', 0, 0.5, -1.0, -0.5], ['cz', 1, 2], ['sq', 8, 0.5, -1.0, -0.5]], [['sq', 2, 0.5, -1.0, -0.5], ['cz', 8, 9]], [['cz', 2, 3], ['sq', 9, 0.5, -1.0, -0.5]], [['sq', 3, 0.5, -1.0, -0.5], ['cz', 9, 10]], [['cz', 3, 4], ['sq', 10, 0.5, -1.0, -0.5]], [['sq', 4, 0.5, -1.0, -0.5], ['cz', 10, 11]], [['cz', 4, 5], ['sq', 11, 0.5, -1.0, -0.5]], [['sq', 5, 0.5, -1.0, -0.5]]]}}
'Calibrated placement of exact graph-local templates over the Willow grid.'
import math as _cf_math
from collections import Counter as _cf_Counter
import cirq as _cf_cirq

def _cf_decode_template(n):
    data = _cf_BALANCED_TEMPLATES[n]
    qs = [_cf_cirq.GridQubit(*q) for q in data['qubits']]
    moments = []
    for moment in data['moments']:
        ops = []
        for entry in moment:
            if entry[0] == 'cz':
                ops.append(_cf_cirq.CZ(qs[entry[1]], qs[entry[2]]))
            else:
                ops.append(_cf_cirq.PhasedXZGate(x_exponent=entry[2], z_exponent=entry[3], axis_phase_exponent=entry[4])(qs[entry[1]]))
        moments.append(_cf_cirq.Moment(ops))
    return (_cf_cirq.Circuit(moments), qs)

def _cf_circuit_calibration_cost(circuit, outputs, k, cz, sq, zero, one):
    n = len(outputs)
    pairs = {}
    for pair, v in cz.items():
        key = frozenset(pair)
        pairs[key] = max(pairs.get(key, 0.0), _cf_rate(v))
    result = 0.0
    for op in circuit.all_operations():
        result += _cf_rate(sq[op.qubits[0]]) if len(op.qubits) == 1 else pairs[frozenset(op.qubits)]
    ds = []
    for q in outputs:
        e0, e1 = (_cf_rate(zero.get(q, (0.01, 0.0))), _cf_rate(one.get(q, (0.02, 0.0))))
        result -= (1 - k / n) * _cf_math.log1p(-e0) + k / n * _cf_math.log1p(-e1)
        ds.append(_cf_math.log1p(-e1) - _cf_math.log1p(-e0))
    scale = 10000 / _cf_math.comb(n, k) * k * (n - k) / (n * max(1, n - 1))
    mean = sum(ds) / n
    return result + scale * sum(((d - mean) ** 2 for d in ds))

def _cf_place_template(n, cz, sq, zero, one, count=4):
    c, qs = _cf_decode_template(n)
    return _cf_rank_template(c, qs, n // 2, cz, sq, zero, one, count)

def _cf_rank_template(c, qs, k, cz, sq, zero, one, count=4):
    qs = sorted(qs)
    n = len(qs)
    graph = _cf_calibration_graph(cz, sq)
    index = {q: i for i, q in enumerate(qs)}
    singles = _cf_Counter()
    edges = _cf_Counter()
    for op in c.all_operations():
        if len(op.qubits) == 1:
            singles[index[op.qubits[0]]] += 1
        else:
            edges[tuple((index[q] for q in op.qubits))] += 1
    candidates = []
    seen = set()
    for flip in (False, True):
        for sr, sc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            coords = [(sc * q.col, sr * q.row) if flip else (sr * q.row, sc * q.col) for q in qs]
            anchor = coords[0]
            for physical in graph:
                mapped = tuple((_cf_cirq.GridQubit(physical.row + r - anchor[0], physical.col + col - anchor[1]) for r, col in coords))
                if mapped in seen or not all((q in graph for q in mapped)):
                    continue
                seen.add(mapped)
                if not all((mapped[b] in graph[mapped[a]] for a, b in edges)):
                    continue
                cost = sum((w * _cf_rate(sq[mapped[i]]) for i, w in singles.items()))
                cost += sum((w * graph[mapped[a]][mapped[b]] for (a, b), w in edges.items()))
                ds = []
                for q in mapped:
                    e0, e1 = (_cf_rate(zero.get(q, (0.01, 0.0))), _cf_rate(one.get(q, (0.02, 0.0))))
                    cost -= (1 - k / n) * _cf_math.log1p(-e0) + k / n * _cf_math.log1p(-e1)
                    ds.append(_cf_math.log1p(-e1) - _cf_math.log1p(-e0))
                cost += 10000 / _cf_math.comb(n, k) * k * (n - k) / (n * max(1, n - 1)) * sum(((d - sum(ds) / n) ** 2 for d in ds))
                candidates.append((cost, mapped))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [(cost, c.transform_qubits(dict(zip(qs, mapped))), set(mapped)) for cost, mapped in candidates[:count]]
'Exact hypergeometric split on a ladder, followed by parallel unary decoding.'
import functools as _cf_functools
import math as _cf_math
import cirq as _cf_cirq
import cirq_google as _cf_cirq_google

def _cf_logical_divide_circuit(n, k):
    if not isinstance(n, int) or not isinstance(k, int) or n < 2 or (not 1 <= k <= n // 2):
        raise ValueError('Require n>=2 and 1<=k<=n/2')
    a, b = (n // 2, n - n // 2)
    top = [_cf_cirq.GridQubit(0, i) for i in range(a)]
    bottom = [_cf_cirq.GridQubit(1, k - 1 - i) for i in range(b)]
    weights = [_cf_math.comb(a, j) * _cf_math.comb(b, k - j) for j in range(k + 1)]
    suffix = [sum(weights[j:]) for j in range(k + 1)]
    ops = list(_cf_cirq.X.on_each(bottom[:k]))
    theta = 2 * _cf_math.acos(_cf_math.sqrt(weights[0] / suffix[0]))
    ops.append(_cf_cirq.ry(theta)(top[0]))
    for j in range(1, k):
        theta = 2 * _cf_math.acos(_cf_math.sqrt(weights[j] / suffix[j]))
        angle = _cf_math.pi / 2 - theta / 2
        ops.extend([_cf_cirq.ry(angle)(top[j]), _cf_cirq.CNOT(top[j - 1], top[j]), _cf_cirq.ry(-angle)(top[j])])
    ops.extend((_cf_cirq.CNOT(top[j], bottom[k - 1 - j]) for j in range(k)))
    upper = _cf_unary_to_dicke_circuit(a, k, native=False).transform_qubits(dict(zip(_cf_cirq.LineQubit.range(a), top)))
    lower = _cf_unary_to_dicke_circuit(b, k, native=False).transform_qubits(dict(zip(_cf_cirq.LineQubit.range(b), bottom)))
    raw = _cf_cirq.Circuit(ops, upper.all_operations(), lower.all_operations())
    return (_cf_native_compile(raw), top, bottom)

def _cf_embed_ladder_arms(n, k, available, pairs):
    """Embed the k-rung core and extend disjoint chains; labels need not be straight."""
    a, b = (n // 2, n - n // 2)
    available = set(available)
    graph = {q: set() for q in available}
    for x, y in pairs:
        if x in graph and y in graph:
            graph[x].add(y)
            graph[y].add(x)

    def paths(prefix, length, occupied):
        if len(prefix) == length:
            yield prefix
            return
        for nxt in sorted(graph[prefix[-1]] - occupied):
            yield from paths(prefix + [nxt], length, occupied | {nxt})
    middle = sorted(available, key=lambda q: ((q.row - 5.5) ** 2 + (q.col - 5.5) ** 2, q))
    for root in middle:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            for sr, sc in [(-dc, dr), (dc, -dr)]:
                top = [_cf_cirq.GridQubit(root.row + i * dr, root.col + i * dc) for i in range(k)]
                bottom = [_cf_cirq.GridQubit(root.row + (k - 1 - i) * dr + sr, root.col + (k - 1 - i) * dc + sc) for i in range(k)]
                core = set(top + bottom)
                if len(core) != 2 * k or not core <= available:
                    continue
                required = list(zip(top, top[1:])) + list(zip(bottom, bottom[1:])) + [(top[j], bottom[k - 1 - j]) for j in range(k)]
                if any((y not in graph[x] for x, y in required)):
                    continue
                for full_top in paths(top, a, core):
                    used = core | set(full_top)
                    full_bottom = next(paths(bottom, b, used), None)
                    if full_bottom is not None:
                        return (full_top, full_bottom)
    raise NotImplementedError('No embedded ladder with disjoint extension arms')

@_cf_functools.lru_cache(maxsize=128)
def _cf__willow_frozen(n, k):
    c, top, bottom = _cf_logical_divide_circuit(n, k)
    device = _cf_cirq_google.Willow105.metadata
    pt, pb = _cf_embed_ladder_arms(n, k, device.qubit_set, device.qubit_pairs)
    mapped = c.transform_qubits(dict(zip(top + bottom, pt + pb)))
    _cf_cirq_google.Willow105.validate_circuit(mapped)
    return (_cf_cirq.FrozenCircuit(mapped), frozenset(pt + pb))

def _cf_willow_divide_circuit(n, k):
    c, qs = _cf__willow_frozen(n, k)
    return (_cf_cirq.Circuit(c), set(qs))
'Exact, shallow W-state (and anti-W, D(N, N-1)) preparation on a branching Willow tree.\n\nOnly cirq.CZ (on allowed pairs) and cirq.PhasedXZGate are emitted.  Exact in the\nnoiseless limit: every amplitude is produced by a closed-form angle from subtree sizes.\n\nConstruction\n------------\nA rooted spanning tree on N hardware qubits plus a *broadcast schedule*: in every round\neach qubit that already holds part of the excitation "splits" to at most one fresh\nneighbour (child).  Invariant: after a node v received its share, the |1>_v component of\nthe global state means "the excitation lives somewhere in the not-yet-distributed part of\nv\'s subtree" (size R_v), with amplitude sqrt(R_v / N).\n\n* First split (pure root r, fresh child c):  Ry(beta)_r, CNOT(r->c), X_c.  1 CZ.\n      |00> -> sqrt((N-S_c)/N) |10> + sqrt(S_c/N) |01>.\n* Later split (parent p in superposition, fresh child c), 2 CZ:\n      c: Ry(a), CNOT(p->c), c: Ry(-a), CNOT(c->p),   a = pi/2 - phi,  sin^2(phi) = S_c / R_p\n  acts as |00> -> |00>, |10> -> cos(phi)|10> + sin(phi)|01> (both real, positive).\n  A CRy onto a known |0> target needs only one CNOT; that is why the split costs 2 CZ.\n  One CZ is impossible for this isometry (operator-Schmidt rank argument, see RESULTS.md).\n\nTotal CZ = 2N - 3.  With perfect doubling the depth is 2*(2R-1)+1 moments, R = ceil(log2 N).\n'
import math as _cf_math
import random as _cf_random
from dataclasses import dataclass as _cf_dataclass, field as _cf_field
import cirq as _cf_cirq
import networkx as _cf_nx
import numpy as _cf_np
_cf__H = _cf_np.array([[1, 1], [1, -1]], dtype=complex) / _cf_math.sqrt(2)
_cf__X = _cf_np.array([[0, 1], [1, 0]], dtype=complex)
_cf__I2 = _cf_np.eye(2, dtype=complex)

def _cf__ry(theta: float) -> _cf_np.ndarray:
    c, s = (_cf_math.cos(theta / 2), _cf_math.sin(theta / 2))
    return _cf_np.array([[c, -s], [s, c]], dtype=complex)

def _cf_hardware_graph(cz_pauli_error_rate=None, sq_pauli_error_rate=None, zero_errors=None, one_errors=None) -> _cf_nx.Graph:
    """Graph of usable qubits/couplers with error attributes.

    Edge attribute 'key' keeps the exact pair orientation from cz_pauli_error_rate so the
    emitted CZ uses a pair that is literally present in the allowed dictionary.
    If no error maps are given, the full Willow105 device with nominal errors is used.
    """
    G = _cf_nx.Graph()
    if cz_pauli_error_rate is None:
        import cirq_google as _cf_cirq_google
        md = _cf_cirq_google.Willow105.metadata
        for q in md.qubit_set:
            G.add_node(q, sq=0.0004, e0=0.005, e1=0.01)
        for pair in md.qubit_pairs:
            a, b = tuple(pair)
            G.add_edge(a, b, cz=0.0025, key=(a, b))
        return G

    def val(d, q, default):
        if d is None or q not in d:
            return default
        v = d[q]
        return float(v[0]) if isinstance(v, (tuple, list)) else float(v)
    for q in sq_pauli_error_rate:
        G.add_node(q, sq=val(sq_pauli_error_rate, q, 0.001), e0=val(zero_errors, q, 0.02), e1=val(one_errors, q, 0.04))
    for (a, b), v in cz_pauli_error_rate.items():
        if a in G and b in G:
            e = float(v[0]) if isinstance(v, (tuple, list)) else float(v)
            if G.has_edge(a, b):
                if e >= G.edges[a, b]['cz']:
                    continue
            G.add_edge(a, b, cz=e, key=(a, b))
    return G

@_cf_dataclass
class _cf_WTree:
    root: _cf_cirq.GridQubit
    calls: list
    nodes: list
    parent: dict = _cf_field(default_factory=dict)

    @property
    def rounds(self) -> int:
        return max((r for r, _, _ in self.calls), default=0)

    def children(self, v):
        return [c for _, p, c in self.calls if p == v]

    def subtree_sizes(self) -> dict:
        size = {v: 1 for v in self.nodes}
        for _, p, c in reversed(self.calls):
            size[p] += size[c]
        return size

def _cf__check_tree(tree: _cf_WTree, G: _cf_nx.Graph | None=None):
    busy = {}
    informed_at = {tree.root: 0}
    for r, p, c in tree.calls:
        assert p in informed_at and informed_at[p] < r, 'parent must hold amplitude first'
        assert c not in informed_at, 'child must be fresh'
        assert (r, p) not in busy and (r, c) not in busy, 'one call per qubit per round'
        if r == 1:
            assert p == tree.root
        busy[r, p] = busy[r, c] = True
        informed_at[c] = r
        if G is not None:
            assert G.has_edge(p, c), f'no coupler {p}-{c}'
    assert set(informed_at) == set(tree.nodes)

def _cf_w_tree_circuit(tree: _cf_WTree, G: _cf_nx.Graph | None=None, complement: bool=False) -> tuple[_cf_cirq.Circuit, set]:
    """Build the exact CZ + PhasedXZ circuit for the broadcast tree.

    complement=True returns D(N, N-1) = X^{(x)N} W_N (X merged into the last 1q layer).
    Moment structure: alternating [1q layer, CZ layer, ..., CZ layer, 1q layer].
    """
    _cf__check_tree(tree, G)
    N = len(tree.nodes)
    S = tree.subtree_sizes()
    remaining = dict(S)
    R = tree.rounds
    n_cz_layers = 0 if R == 0 else 2 * R - 1
    oneq = [dict() for _ in range(n_cz_layers + 1)]
    czs = [[] for _ in range(n_cz_layers)]

    def add1(layer, q, m):
        oneq[layer].setdefault(q, []).append(m)

    def cz_pair(p, c):
        if G is not None and G.has_edge(p, c):
            return G.edges[p, c]['key']
        return (p, c)
    if N == 1:
        add1(0, tree.root, _cf__X)
    for r, p, c in tree.calls:
        Sc = S[c]
        if r == 1:
            keep = (N - Sc) / N
            beta = 2 * _cf_math.asin(_cf_math.sqrt(keep))
            add1(0, p, _cf__ry(beta))
            add1(0, c, _cf__H)
            czs[0].append(cz_pair(p, c))
            add1(1, c, _cf__H)
            add1(1, c, _cf__X)
        else:
            t = 2 * r - 3
            Rp = remaining[p]
            phi = _cf_math.asin(_cf_math.sqrt(Sc / Rp))
            a = _cf_math.pi / 2 - phi
            add1(t, c, _cf__ry(a))
            add1(t, c, _cf__H)
            czs[t].append(cz_pair(p, c))
            add1(t + 1, c, _cf__H)
            add1(t + 1, c, _cf__ry(-a))
            add1(t + 1, p, _cf__H)
            czs[t + 1].append(cz_pair(p, c))
            add1(t + 2, p, _cf__H)
        remaining[p] -= Sc
    if complement:
        for q in tree.nodes:
            add1(n_cz_layers, q, _cf__X)
    moments = []
    for layer in range(n_cz_layers + 1):
        ops = []
        for q, mats in oneq[layer].items():
            m = _cf__I2
            for mm in mats:
                m = mm @ m
            g = _cf_cirq.PhasedXZGate.from_matrix(m)
            if not _cf_cirq.equal_up_to_global_phase(m, _cf__I2, atol=1e-12):
                ops.append(g.on(q))
        if ops:
            moments.append(_cf_cirq.Moment(ops))
        if layer < n_cz_layers:
            moments.append(_cf_cirq.Moment((_cf_cirq.CZ(*pr) for pr in czs[layer])))
    return (_cf_cirq.Circuit(moments), set(tree.nodes))

def _cf__node_cost(G, q, N, n_exc=1):
    d = G.nodes[q]
    readout = (N - n_exc) / N * d['e0'] + n_exc / N * d['e1']
    return readout + 3 * d['sq']

def _cf_grow_tree(G: _cf_nx.Graph, N: int, root, rng: _cf_random.Random, jitter: float=0.0, allowed: set | None=None, n_exc: int=1):
    """Greedy round-by-round broadcast growth using a min-cost maximum matching per round."""
    informed = [root]
    inf_set = {root}
    calls = []
    r = 0
    while len(informed) < N:
        r += 1
        callers = [root] if r == 1 else list(informed)
        B = _cf_nx.Graph()
        for p in callers:
            for c in G.neighbors(p):
                if c in inf_set or (allowed is not None and c not in allowed):
                    continue
                usage = 1 if r == 1 else 2
                cost = usage * G.edges[p, c]['cz'] + _cf__node_cost(G, c, N, n_exc)
                cost *= 1 + jitter * rng.random()
                B.add_edge(('p', p), ('c', c), weight=1.0 - cost)
        if B.number_of_edges() == 0:
            return None
        M = _cf_nx.max_weight_matching(B, maxcardinality=True)
        pairs = []
        for u, v in M:
            if u[0] == 'c':
                u, v = (v, u)
            pairs.append((B.edges[u, v]['weight'], u[1], v[1]))
        need = N - len(informed)
        pairs.sort(key=lambda x: -x[0])
        for _, p, c in pairs[:need]:
            calls.append((r, p, c))
            informed.append(c)
            inf_set.add(c)
    return _cf_WTree(root=root, calls=calls, nodes=informed, parent={c: p for _, p, c in calls})

def _cf_tree_cost(G, tree: _cf_WTree, n_exc: int=1, per_moment: float=0.0004) -> float:
    """Crude additive infidelity proxy for f (Hamming-weight retention)."""
    N = len(tree.nodes)
    c = sum((_cf__node_cost(G, q, N, n_exc) for q in tree.nodes))
    for r, p, ch in tree.calls:
        c += (1 if r == 1 else 2) * G.edges[p, ch]['cz'] + (0 if r == 1 else 2) * G.nodes[p]['sq']
    R = tree.rounds
    moments = 2 * (2 * R - 1) + 1 if R else 1
    return c + per_moment * moments

def _cf_best_w_tree(G: _cf_nx.Graph, N: int, seed: int=0, restarts: int=4, time_budget: float=10.0, n_exc: int=1, per_moment: float=0.0004):
    import time as _cf_time
    rng = _cf_random.Random(seed)
    t0 = _cf_time.time()
    best, best_cost = (None, float('inf'))
    roots = sorted(G.nodes, key=lambda q: _cf__node_cost(G, q, N, n_exc))
    for it in range(restarts):
        for root in roots:
            if _cf_time.time() - t0 > time_budget and best is not None:
                return (best, best_cost)
            tr = _cf_grow_tree(G, N, root, rng, jitter=0.0 if it == 0 else 0.3, n_exc=n_exc)
            if tr is None:
                continue
            c = _cf_tree_cost(G, tr, n_exc, per_moment)
            if c < best_cost:
                best, best_cost = (tr, c)
    if best is None:
        raise NotImplementedError(f'no tree with {N} qubits found')
    return (best, best_cost)

def _cf_create_w_state(num_qubits: int, cz_pauli_error_rate=None, sq_pauli_error_rate=None, zero_errors=None, one_errors=None, *, complement: bool=False, seed: int=0, time_budget: float=10.0):
    """D(N,1) (or D(N,N-1) if complement=True) on the best tree found on the error maps."""
    G = _cf_hardware_graph(cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    if num_qubits < 1 or num_qubits > G.number_of_nodes():
        raise NotImplementedError
    n_exc = num_qubits - 1 if complement else 1
    tree, _ = _cf_best_w_tree(G, num_qubits, seed=seed, time_budget=time_budget, n_exc=n_exc)
    return _cf_w_tree_circuit(tree, G, complement=complement)

def _cf_create_dicke_state_w(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    """Notebook-signature wrapper: handles n = 1 and n = N - 1 only."""
    N, n = (num_qubits, num_excitations)
    if n == 1:
        return _cf_create_w_state(N, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    if n == N - 1 and N >= 2:
        return _cf_create_w_state(N, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors, complement=True)
    raise NotImplementedError
'Portfolio: exact square, branching W, hypergeometric split, balanced and line.'
import time as _cf_time
import cirq as _cf_cirq
import cirq_google as _cf_cirq_google

def _cf_create_dicke_state(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    started = _cf_time.monotonic()
    n, k = (num_qubits, num_excitations)
    if not isinstance(n, int) or not isinstance(k, int) or n < 1 or (not 0 <= k <= n):
        raise ValueError('Require integer N>=1 and 0<=k<=N')
    if n > len(sq_pauli_error_rate):
        raise NotImplementedError('Insufficient calibrated qubits')
    if k in (0, n):
        errors = zero_errors if k == 0 else one_errors
        qs = sorted(sq_pauli_error_rate, key=lambda q: (_cf_rate(errors.get(q, (1.0, 0.0))), q))[:n]
        gate = _cf_cirq.PhasedXZGate(x_exponent=int(k == n), z_exponent=0, axis_phase_exponent=0)
        return (_cf_cirq.Circuit(_cf_cirq.Moment((gate(q) for q in qs))), set(qs))
    args = (cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    if n == 2 and k == 1:
        try:
            return _cf__create_bell(n, k, *args)
        except NotImplementedError:
            pass
    if n == 4 and k == 2:
        try:
            return _cf_create_dicke4(*args)
        except NotImplementedError:
            pass
    choices = []
    if n in (6, 8, 10, 12, 14) and k == n // 2:
        choices = _cf_place_template(n, *args, count=2 if n <= 6 else 1)
    if k not in (1, n - 1) and n <= 20:
        try:
            dc, qs = _cf_willow_divide_circuit(n, min(k, n - k))
            if k > n // 2:
                dc = _cf_cirq.Circuit(_cf_cirq.merge_single_qubit_gates_to_phxz(dc + _cf_cirq.Circuit(_cf_cirq.X.on_each(qs))).all_operations())
            choices.extend(_cf_rank_template(dc, qs, k, *args, count=1))
        except NotImplementedError:
            pass
    if k in (1, n - 1) and n > 2:
        try:
            wc, wq = _cf_create_w_state(n, *args, complement=k > 1, time_budget=2.0)
            choices.append((_cf_circuit_calibration_cost(wc, wq, k, *args), wc, wq))
        except NotImplementedError:
            pass
    try:
        line, qs = _cf_place_line(_cf_logical_circuit(n, k), n, k, *args)
        choices.append((_cf_circuit_calibration_cost(line, qs, k, *args), line, qs))
    except NotImplementedError:
        pass
    if not choices:
        raise NotImplementedError('No valid calibrated placement found')
    choices = [item for item in choices if len(item[1]) <= 1750]
    if not choices:
        raise NotImplementedError('Exceeds1750-moment limit')
    if n <= 8 and len(choices) > 1:
        choices.sort(key=lambda item: item[0])
        refined = []
        for _, c, qs in choices[:6 if n <= 6 else 2]:
            if refined and _cf_time.monotonic() - started > 30:
                break
            refined.append((_cf_small_proxy(c, sorted(qs), k, *args), c, qs))
        choices = refined
    _, circuit, outputs = min(choices, key=lambda item: item[0])
    return (circuit, outputs)

def _cf_create_dicke_state_echo(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    c, q = _cf_create_dicke_state(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    return (_cf_echo_idle_qubits(c, min_idle_moments=32), q)

def _cf_create_dicke_state_adaptive(num_qubits, num_excitations):
    n, k = (num_qubits, num_excitations)
    if not isinstance(n, int) or not isinstance(k, int) or n < 2 or (not 1 <= k <= n // 2):
        raise ValueError('Require integer N>=2 and 1<=k<=N/2')
    if (n, k) == (4, 2):
        qs = [_cf_cirq.GridQubit(4, 4), _cf_cirq.GridQubit(4, 5), _cf_cirq.GridQubit(5, 5), _cf_cirq.GridQubit(5, 4)]
        return (_cf_dicke4_square(qs), set(qs))
    qs = _cf_cirq_google.Willow105.metadata.qubit_set
    pairs = _cf_cirq_google.Willow105.metadata.qubit_pairs
    args = ({tuple(p): (0.0, 0.0) for p in pairs}, {q: (0.0, 0.0) for q in qs}, {q: (0.0, 0.0) for q in qs}, {q: (0.0, 0.0) for q in qs})
    if n == 2:
        return _cf__create_bell(n, k, *args)
    if k == 1:
        return _cf_create_w_state(n, *args, time_budget=2.0)
    line, lq = _cf_place_line(_cf_logical_circuit(n, k), n, k, *args)
    choices = [(line, lq)]
    try:
        choices.append(_cf_willow_divide_circuit(n, k))
    except NotImplementedError:
        pass
    if n in (6, 8, 10, 12, 14) and k == n // 2:
        template, tq = _cf_decode_template(n)
        choices.append((template, set(tq)))
    return min(choices, key=lambda item: len(item[0]))
'Exact small native-circuit density evolution for the calibration proxy.\n\nUniform nonidentity-Pauli noise on a d-dimensional subsystem is\n  (1 - d*d*p/(d*d-1))*rho + d*p/(d*d-1)*I*partial_trace(rho).\nThis is algebraically the same channel as cirq.depolarize(p, n_qubits=m).\nThe partial-trace form avoids enumerating all 15 two-qubit Pauli branches.\n'
import cirq as _cf_cirq
import numpy as _cf_np

def _cf_native_density(circuit, qubits, cz_rates, sq_rates):
    n = len(qubits)
    positions = {q: i for i, q in enumerate(qubits)}
    rates = {}
    for pair, value in cz_rates.items():
        key = frozenset(pair)
        rates[key] = max(rates.get(key, 0.0), float(value[0]))
    rho = _cf_np.zeros((2,) * (2 * n), dtype=_cf_np.complex128)
    rho[(0,) * (2 * n)] = 1.0
    layouts = {}
    for op in circuit.all_operations():
        targets = tuple(sorted((positions[q] for q in op.qubits)))
        if len(targets) == 1 and isinstance(op.gate, _cf_cirq.PhasedXZGate):
            q = targets[0]
            unitary = _cf_cirq.unitary(op.gate)
            rho = _cf_np.moveaxis(_cf_np.tensordot(unitary, rho, axes=(1, q)), 0, q)
            rho = _cf_np.moveaxis(_cf_np.tensordot(unitary.conj(), rho, axes=(1, n + q)), 0, n + q)
            p = float(sq_rates[op.qubits[0]][0])
        elif len(targets) == 2 and op.gate == _cf_cirq.CZ:
            shape = [1] * (2 * n)
            for q in targets:
                shape[q] = 2
            diagonal = _cf_np.array([[1.0, 1.0], [1.0, -1.0]])
            rho *= diagonal.reshape(shape)
            shape = [1] * (2 * n)
            for q in targets:
                shape[n + q] = 2
            rho *= diagonal.reshape(shape)
            p = rates[frozenset(op.qubits)]
        else:
            raise ValueError('Proxy accepts native PhasedXZ and full CZ gates only')
        p = min(1.0, max(0.0, p))
        if p == 0:
            continue
        if targets not in layouts:
            rest = tuple((i for i in range(n) if i not in targets))
            perm = targets + rest + tuple((n + i for i in targets)) + tuple((n + i for i in rest))
            inverse = tuple(_cf_np.argsort(perm))
            layouts[targets] = (perm, inverse, 1 << len(targets), 1 << len(rest))
        perm, inverse, d, remainder = layouts[targets]
        block = rho.transpose(perm).reshape(d, remainder, d, remainder)
        partial = _cf_np.einsum('iaib->ab', block)
        block = (1 - d * d * p / (d * d - 1)) * block
        for i in range(d):
            block[i, :, i, :] += d * p / (d * d - 1) * partial
        rho = block.reshape((2,) * (2 * n)).transpose(inverse)
    return rho.reshape(1 << n, 1 << n)
'Offline/runtime first-order Pauli response for state-preparation placement.\n\nThis loss only observes the prepared state, never the unspecified columns.\nFull density evolution verifies the short list before selection. The response\nis a ranking approximation, not a claim about real Willow noise.\n'
import functools as _cf_functools
import math as _cf_math
import cirq as _cf_cirq
import numpy as _cf_np

def _cf__apply(a, u, q, n):
    shape = a.shape
    tensor = a.reshape((2,) * n + shape[1:])
    return _cf_np.moveaxis(_cf_np.tensordot(u, tensor, axes=(1, q)), 0, q).reshape(shape)

def _cf__hadamard(a, n):
    out = a.copy()
    for q in range(n):
        out = _cf__apply(out, _cf_np.array([[1.0, 1.0], [1.0, -1.0]]) / _cf_math.sqrt(2), q, n)
    return out

@_cf_functools.lru_cache(maxsize=24)
def _cf_fault_response(frozen, qubits):
    """Return exact single-fault probability derivatives at zero gate noise."""
    n, dim = (len(qubits), 1 << len(qubits))
    ids = {q: i for i, q in enumerate(qubits)}
    ops = list(frozen.all_operations())
    index = _cf_np.arange(dim)
    states = []
    v = _cf_np.zeros(dim, dtype=complex)
    v[0] = 1
    for op in ops:
        loc = tuple((ids[q] for q in op.qubits))
        if len(loc) == 1:
            v = _cf__apply(v, _cf_cirq.unitary(op), loc[0], n)
        else:
            assert op.gate == _cf_cirq.CZ
            mask = index >> n - 1 - loc[0] & index >> n - 1 - loc[1] & 1
            v = v * (1 - 2 * mask)
        states.append(v)
    pz, px = (abs(v) ** 2, abs(_cf__hadamard(v, n)) ** 2)
    response = {}
    suffix = _cf_np.eye(dim, dtype=complex)
    for j in range(len(ops) - 1, -1, -1):
        op, state = (ops[j], states[j])
        loc = tuple((ids[q] for q in op.qubits))
        variants = []
        for code in range(1, 4 ** len(loc)):
            flip, sign = (0, _cf_np.ones(dim))
            for t, q in enumerate(loc):
                p = code >> 2 * t & 3
                bit = 1 << n - 1 - q
                if p in (1, 2):
                    flip ^= bit
                if p in (2, 3):
                    sign *= 1 - 2 * (index & bit != 0)
            variants.append(state[index ^ flip] * sign)
        faulty = suffix @ _cf_np.stack(variants, axis=1)
        dz = _cf_np.mean(abs(faulty) ** 2, axis=1) - pz
        dx = _cf_np.mean(abs(_cf__hadamard(faulty, n)) ** 2, axis=1) - px
        key = tuple(sorted(loc))
        if key not in response:
            response[key] = [_cf_np.zeros(dim), _cf_np.zeros(dim)]
        response[key][0] += dz
        response[key][1] += dx
        if len(loc) == 1:
            suffix = _cf__apply(suffix.T, _cf_cirq.unitary(op).T, loc[0], n).T
        else:
            mask = index >> n - 1 - loc[0] & index >> n - 1 - loc[1] & 1
            suffix *= (1 - 2 * mask)[None, :]
    keys = tuple(response)
    return (pz, px, keys, _cf_np.stack([response[k][0] for k in keys]), _cf_np.stack([response[k][1] for k in keys]))

def _cf_rank_mappings(circuit, logical, mappings, k, cz, sq, zero, one):
    """Vectorized expected-denominator approximation including exact readout."""
    n = len(logical)
    pz, px, keys, dz, dx = _cf_fault_response(_cf_cirq.FrozenCircuit(circuit), tuple(logical))
    edges = {frozenset(pair): float(v[0]) for pair, v in cz.items()}
    rates = _cf_np.array([[float(sq[m[key[0]]][0]) if len(key) == 1 else edges[frozenset((m[key[0]], m[key[1]]))] for key in keys] for m in mappings])
    z = _cf_np.maximum(0, pz[None, :] + rates @ dz)
    x = _cf_np.maximum(0, px[None, :] + rates @ dx)
    z /= z.sum(axis=1, keepdims=True)
    x /= x.sum(axis=1, keepdims=True)
    index = _cf_np.arange(1 << n)
    for q in range(n):
        bit = 1 << n - 1 - q
        lo, hi = (index[index & bit == 0], index[index & bit != 0])
        e0 = _cf_np.array([float(zero.get(m[q], (0.01, 0))[0]) for m in mappings])[:, None]
        e1 = _cf_np.array([float(one.get(m[q], (0.02, 0))[0]) for m in mappings])[:, None]
        a, b = (z[:, lo].copy(), z[:, hi].copy())
        z[:, lo], z[:, hi] = ((1 - e0) * a + e1 * b, e0 * a + (1 - e1) * b)
        extra = _cf_np.array([2 * float(sq[m[q]][0]) / 3 for m in mappings])[:, None] * (1 - e0 - e1)
        e0, e1 = (e0 + extra, e1 + extra)
        a, b = (x[:, lo].copy(), x[:, hi].copy())
        x[:, lo], x[:, hi] = ((1 - e0) * a + e1 * b, e0 * a + (1 - e1) * b)
    weights = _cf_np.array([int(i).bit_count() for i in index])
    m = _cf_math.comb(n, k)
    f = z[:, weights == k].sum(axis=1)
    delta = m * (z[:, weights == k] ** 2).sum(axis=1) / f ** 2 - 1
    inv = 1 / (10000 * f) + (1 - f) / (10000 * f) ** 2 + (1 - f) * (2 - f) / (10000 * f) ** 3
    score = 10000 / m * (delta + (m - 1 - delta) * inv)

    def comb(a, b):
        return _cf_math.comb(a, b) if 0 <= b <= a else 0
    for ell in range(1, n + 1):
        target = comb(ell, ell // 2) * comb(n - ell, k - ell // 2) / m if ell % 2 == 0 else 0
        vals = _cf_np.array([sum(((-1) ** j * comb(w, j) * comb(n - w, ell - j) for j in range(ell + 1))) / comb(n, ell) for w in weights])
        mean = x @ vals
        var = _cf_np.maximum(0, x @ vals ** 2 - mean ** 2)
        score += ((mean - target) ** 2 + var / 10000) / (target ** 2 if target else 1)
    return score
_cf_COLUMNFORGE_TEMPLATES = {(5, 2): [{'n': 5, 'k': 2, 'qubits': [[3, 5], [4, 5], [4, 6], [5, 5], [5, 6]], 'moments': [[['sq', 0, 0.1959132760153037, 5.183513872475822e-08, 0.5], ['sq', 1, 0.3333333333333333, 0.9999998307071643, 0.5], ['sq', 2, 0.5, 2.598822158415535e-17, 0.5], ['sq', 3, -0.29516723530086647, -1.0000000000000002, 0.5], ['sq', 4, -0.6666666666666666, 1.0, 0.5]], [['cz', 3, 4]], [['sq', 4, -0.33333333333333337, 21756.541190195523, 0.5], ['cz', 3, 1]], [['cz', 4, 2], ['sq', 3, 0.5, -32734.62973664393, 0.5], ['sq', 1, 0.6666666666666681, 0.9999998730303662, 0.5]], [['sq', 4, 0.5, 6.024557572515177e-17, 0.5], ['sq', 2, 0.5, -21753.958809931446, 0.5], ['cz', 1, 0]], [['cz', 3, 4], ['sq', 1, 0.6959132760153054, -1.0000001466118924, 0.5], ['sq', 0, 0.6959132760153038, -2.992703410520468e-08, 0.5]], [['sq', 4, 0.5, 32595.370263356075, 0.5], ['cz', 2, 1]], [['sq', 2, 0.5, -1.0, 0.5], ['sq', 1, 0.19591327601530364, 1.0000001795621705, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.7499999999999999, -0.4999999576767804, 0.5], ['sq', 0, 0.7500000000000011, 32596.370263271427, 0.5]], [['cz', 2, 1]], [['sq', 2, 0.5, 32596.87026331375, 0.5], ['sq', 1, 0.5, -1.0, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.49999999999999994, 32595.37026327143, 0.5]]], 'cz': 9, 'residual': 4.360526229350869e-12, 'source': 'local_prune\\candidate_5_2_9cz_sq.cirq.json'}, {'n': 5, 'k': 2, 'qubits': [[3, 5], [4, 5], [4, 6], [5, 5], [5, 6]], 'moments': [[['sq', 0, 0.5000000000000003, -0.30377017242489757, 0.5], ['sq', 1, 0.49999999999999994, 0.5, 0.5], ['sq', 2, 0.5000000000000001, -0.5, -0.5], ['sq', 3, 0.6641868814212597, 0.413629038091223, -0.5], ['sq', 4, 0.4293050623646759, 0.6320465085643596, -0.5]], [['cz', 3, 4]], [['sq', 3, 0.23432606472107978, -0.4094613009426271, -0.5], ['sq', 4, 0.5089084392942935, -0.7525196160326788, -0.5]], [['cz', 3, 1], ['cz', 4, 2]], [['sq', 3, 0.7500000000000001, 4.440892098500626e-16, 0.5], ['sq', 1, 0.38746446708156956, 0.02066505568974686, -0.5], ['sq', 4, 1.992244125879467e-17, 0.30474062330736273, -0.5000000000000002], ['sq', 2, 0.45875539218505157, 1.0, -0.5]], [['cz', 1, 0], ['cz', 3, 4]], [['sq', 1, 0.19591327601530364, 0.6959132760153022, 0.49999999999999994], ['sq', 0, 0.5168689922915881, -0.2380578403990523, 0.5], ['sq', 3, 0.25000000000000006, 0.5275556511206005, 0.5], ['sq', 4, 0.49999999999999994, 0.0, 0.5]], [['cz', 2, 1], ['cz', 3, 4]], [['sq', 2, 0.6959132760153038, 1.0, 0.5], ['sq', 1, 0.5000000000000001, -0.5000000000000018, 0.5], ['sq', 3, 1.0, 0.0, -0.3857942006584991], ['sq', 4, 0.5000000000000001, 0.7008559475624048, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.4999999999999998, -0.5000000000000004, 0.49999999999999994], ['sq', 0, 0.5, 0.4999999999999991, 0.5]], [['cz', 2, 1]], [['sq', 2, 0.5, 0.9629431985167436, 0.5], ['sq', 1, 0.24999999999999992, 1.0, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.5, -0.037056801483256185, 0.5], ['sq', 0, 0.5000000000000001, -0.037056801483257074, 0.5]]], 'cz': 10, 'residual': 9.379544120789983e-15, 'source': 'direct_search\\candidate_5_2_10cz.cirq.json'}], (6, 3): [{'n': 6, 'k': 3, 'qubits': [[4, 3], [4, 4], [5, 3], [5, 4], [6, 3], [6, 4]], 'moments': [[['sq', 0, 0.20978468837241684, -1.0, 0.5], ['sq', 1, 1.5212364096070798, 1.0000000000000002, 0.5], ['sq', 2, 0.5, -1.0, 0.5], ['sq', 3, 0.7038845844471661, 7.06948367233203e-17, 0.5], ['sq', 4, 0.5, -0.5000000000000001, 0.5], ['sq', 5, 0.5, 224.50000000000003, 0.5]], [['cz', 0, 1]], [['sq', 0, -0.2097846883724169, -15296.455169008472, 0.5], ['cz', 1, 3]], [['cz', 0, 2], ['sq', 1, -0.8040867239846965, -1.0, 0.5], ['sq', 3, 0.29611541555283416, -2.276376362177206e-16, 0.5]], [['sq', 2, 1.5, -15300.455169008472, 0.5], ['cz', 0, 1]], [['cz', 2, 4], ['sq', 0, -0.5, -0.5000000000000001, 0.5], ['sq', 1, -0.19591327601530348, -2.0, 0.5]], [['sq', 2, 0.14758361765043337, 0.5, 0.5]], [['cz', 3, 2]], [['sq', 3, 0.75, 2.9779112160266155e-17, 0.5], ['sq', 2, 0.5, -0.24999999999999997, 0.5]], [['cz', 2, 4], ['cz', 3, 1]], [['sq', 2, -0.5000000000000001, -1.0000000000000002, 0.5], ['sq', 4, -0.39758361765043326, -0.4999999999999998, 0.5], ['sq', 3, -0.24999999999999997, 0.9999999999999996, 0.5]], [['cz', 2, 4], ['cz', 5, 3]], [['sq', 3, 0.5, -0.5, 0.5], ['cz', 5, 4]], [['cz', 3, 1], ['sq', 5, -11.5, -5102.265083376326, 0.5], ['sq', 4, 0.49999999999999994, 5102.2349166236745, 0.5]], [['sq', 1, 0.5000000000000001, 0.5000000000000001, 0.5], ['cz', 3, 2]], [['cz', 1, 0], ['sq', 3, 0.49999999999999994, -5102.265083376326, 0.5], ['sq', 2, 0.5, -5101.765083376326, 0.5]], [['sq', 1, 0.5000000000000001, -5101.265083376326, 0.5], ['sq', 0, 0.5000000000000001, -5101.765083376326, 0.5]]], 'cz': 14, 'residual': 1.5791248240149163e-12, 'source': 'local_prune\\candidate_6_3_14cz_balanced_sq.cirq.json'}, {'n': 6, 'k': 3, 'qubits': [[4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7]], 'moments': [[['sq', 0, 5.304086723984696, 7.0, 0.5], ['sq', 1, 1.2048327646991335, -35.5, 0.5], ['sq', 2, 0.5, 2701.5, 0.5], ['sq', 3, -1.5, 1.5, 0.5], ['sq', 4, 0.5, -11.240955216597516, 0.5], ['sq', 5, 0.5, 2906.499999999999, 0.5]], [['cz', 3, 4]], [['cz', 4, 5]], [['sq', 4, 1.3040867239846965, -2.0, 0.5], ['sq', 5, -91.00904478340249, 2933.9999999999995, 0.5]], [['cz', 4, 1], ['cz', 5, 2]], [['sq', 4, 0.6959132760153036, -1.2500000000000002, 0.5], ['sq', 1, 12.666666666666668, -46.0, 0.5], ['sq', 5, 0.5000000000000001, -0.5, 0.5]], [['cz', 1, 0], ['cz', 5, 4]], [['sq', 1, 4.25, -16.0, 0.5], ['sq', 0, 2.3040867239846965, 39285.33685819926, 0.5], ['sq', 5, 0.5, 9666.471901354469, 0.5], ['cz', 3, 4]], [['cz', 1, 0], ['sq', 3, 0.5, -23544.5, 0.5], ['sq', 4, 0.5, -1440.5615082278216, 0.5]], [['sq', 1, -5.25, -39356.33685819926, 0.5], ['cz', 4, 3]], [['cz', 1, 2], ['sq', 4, 1.0, 1261.9357217251904, 0.5], ['sq', 3, -0.5, -23700.00277004699, 0.5]], [['sq', 1, 0.49999999999999994, -1.0, 0.5], ['sq', 2, 0.5, -1.0, 0.5], ['cz', 4, 5]], [['cz', 0, 1], ['sq', 5, -1.6880585753848136e-17, -6964.474671401455, 0.5]], [['sq', 0, 0.5, -0.002770046988151846, 0.5], ['sq', 1, 0.5000000000000001, -39270.83685819926, 0.5]], [['cz', 2, 1]], [['sq', 2, 0.5, 1.4972299530118482, 0.5], ['sq', 1, 0.5, -1.0, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.5, -1.0027700469881518, 0.5]]], 'cz': 14, 'residual': 6.424570543841875e-12, 'source': 'local_prune\\candidate_6_3_14cz_sq.cirq.json'}, {'n': 6, 'k': 3, 'qubits': [[4, 3], [4, 4], [5, 3], [5, 4], [6, 3], [6, 4]], 'moments': [[['sq', 0, 0.22643317490804352, -0.9093904344571828, -0.5], ['sq', 1, 0.6145975419752596, -0.8555022656948945, -0.5], ['sq', 2, 0.49999999999999983, 0.1583523210975275, 0.5], ['sq', 3, 0.29611541555283405, 1.1102230246251565e-16, 0.5], ['sq', 4, 0.7436321034001204, 0.00447095922801255, 0.5], ['sq', 5, 0.5, 0.7861025741172161, 0.5]], [['cz', 0, 1]], [['sq', 0, 0.8652947187757795, 0.9126662084493313, 0.5], ['sq', 1, 0.8684092687026346, 0.40916583944582063, 0.5]], [['cz', 0, 2], ['cz', 1, 3]], [['sq', 0, 0.49999999999999983, 0.5, 0.5], ['sq', 2, 0.4800199708301628, -0.46339300040243714, 0.5], ['sq', 1, 0.26661506024497816, 0.7430869083263505, 0.5], ['sq', 3, 0.703884584447166, 8.881784197001252e-16, 0.5]], [['cz', 0, 1], ['cz', 2, 4]], [['sq', 0, 0.7167329707440594, 1.0, 0.49999999999999994], ['sq', 1, 1.0, 0.0, 0.22990993953520478], ['sq', 2, 0.78706212165858, 0.9896134342228788, 0.5], ['sq', 4, 0.49999999999999983, 0.0, 0.5]], [['cz', 0, 1], ['cz', 3, 2]], [['sq', 0, 0.10354215668300573, -0.9999999999999998, -0.5], ['sq', 1, 0.26661506024497866, -0.04041354526095797, -0.5], ['sq', 3, 0.24999999999999983, 0.0, 0.5], ['sq', 2, 0.6794319595054809, -0.9870662498400764, 0.5]], [['cz', 2, 4], ['cz', 3, 1]], [['sq', 2, 0.10260454378043365, 0.9795386623993516, 0.5], ['sq', 4, 0.3155360350764457, 0.028353980937656398, 0.5], ['sq', 3, 0.2499999999999999, 0.0, 0.5], ['sq', 1, 2.078850232934214e-17, 0.7601927697856681, 0.5]], [['cz', 2, 4], ['cz', 5, 3]], [['sq', 2, 0.2727833953716869, 0.504469636713631, -0.5], ['sq', 4, 2.0475253503525847e-16, 0.4716460190623435, 0.5], ['sq', 5, 1.0, 0.0, 0.14305128705860798], ['sq', 3, 0.5, -0.7028784442207385, 0.5]], [['cz', 5, 4], ['cz', 3, 1]], [['sq', 5, 0.5, -0.3567809934716333, 0.5], ['sq', 4, 0.5000000000000001, 0.14321900652836694, 0.5], ['sq', 3, 1.1695680047042487e-16, -0.7971215557792615, 0.5], ['sq', 1, 0.5000000000000001, 0.5, 0.5]], [['cz', 3, 2], ['cz', 1, 0]], [['sq', 3, 0.5, -0.35678099347163306, 0.5], ['sq', 2, 0.5000000000000001, 0.14321900652836694, 0.5], ['sq', 1, 0.5, 0.6432190065283669, 0.5], ['sq', 0, 0.5000000000000001, 0.14321900652836694, 0.5]]], 'cz': 15, 'residual': 2.2603067459636836e-15, 'source': 'direct_search\\candidate_6_3_15cz_balanced.cirq.json'}], (7, 3): [{'n': 7, 'k': 3, 'qubits': [[3, 5], [4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7]], 'moments': [[['sq', 0, 0.23346263562123676, 0.09680310696012473, 0.5], ['sq', 1, 0.5000000000000001, -0.33519779061546107, 0.5], ['sq', 2, 0.75, 1.0, 0.5], ['sq', 3, 0.5, 1.0, 0.5], ['sq', 4, 0.5439241310515563, -0.3249105730274453, 0.49999999999999994], ['sq', 5, 0.36099330313764266, -0.8423735342902232, 0.5], ['sq', 6, 0.4652748742609469, 1.0, 0.5]], [['cz', 4, 5]], [['sq', 4, 0.2875635924879449, -0.904248117149197, 0.5], ['sq', 5, 0.532607423202116, -0.6849929712023104, 0.5]], [['cz', 4, 1], ['cz', 5, 6]], [['sq', 4, 1.0, 0.0, 0.6795120730799196], ['sq', 1, 0.6108838999403319, 0.06112364929646752, 0.5], ['sq', 5, 0.49999999999999994, -0.5965764022616762, 0.5], ['sq', 6, 0.3503967845281199, 0.791435554180036, 0.5]], [['cz', 6, 3], ['cz', 5, 4], ['cz', 1, 2]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 3, 0.5, -0.4702703427937607, 0.5], ['sq', 5, 0.24037987725671286, 0.08958177328159911, 0.5], ['sq', 4, 0.25000000000000006, 0.0, 0.5], ['sq', 1, 0.5000000000000001, 0.19591327601530395, 0.5], ['sq', 2, 0.6320277731639083, -0.4999999999999998, 0.5]], [['cz', 4, 5], ['cz', 2, 3]], [['sq', 4, 0.75, -0.9807150792042418, 0.5], ['sq', 5, 7.25822796650499e-17, -0.018238367464662897, 0.5], ['sq', 2, 0.5000000000000002, 0.8215907364977066, 0.5], ['sq', 3, 1.0, 0.0, 0.4256314950418236]], [['cz', 6, 5], ['cz', 1, 2]], [['sq', 6, 0.5000000000000001, -0.10674362401575421, 0.5], ['sq', 5, 0.5000000000000001, 0.0, 0.5], ['sq', 1, 0.6666666666666667, -0.354847437565305, 0.5], ['sq', 2, 0.8291885549916075, -0.7230611869234917, 0.5]], [['cz', 5, 4], ['cz', 3, 2]], [['sq', 5, 0.5000000000000001, 0.6703723052793857, 0.5], ['sq', 4, 1.0, 0.0, -0.6808027365194838], ['sq', 3, 0.5, 1.0, 0.5], ['sq', 2, 0.9718346440436038, -0.9325562035174695, 0.5]], [['cz', 5, 6], ['cz', 1, 2]], [['sq', 5, 2.3165180176245194e-17, -0.05126269911411119, -0.5], ['sq', 6, 1.0, 0.0, 0.5061829910747602], ['sq', 1, 0.4205497467660725, 0.4497803225509407, 0.5], ['sq', 2, 0.5000000000000001, -0.42620976487146733, 0.5]], [['cz', 1, 0], ['cz', 2, 3]], [['sq', 1, 0.6326561005067499, 0.10782350391620188, 0.5], ['sq', 0, 0.7010158865227026, 0.997457607891409, 0.49999999999999994], ['sq', 2, 1.0, 0.0, -0.46310488243573356], ['sq', 3, 0.5, 0.08901182746790326, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.7301934385056068, -0.6039279398383677, 0.5], ['sq', 0, 0.6488254987416722, 0.31982092378724314, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.5, 0.0, 0.5], ['sq', 2, 0.5, 0.0, 0.5]], [['cz', 0, 1]], [['sq', 0, 0.5, -0.47341502449765893, 0.5], ['sq', 1, 0.5, 0.7403481844396369, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.49999999999999994, 1.0, 0.5], ['sq', 2, 0.5, -0.910988172532097, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.5, -0.41098817253209674, 0.5], ['sq', 0, 1.3145880996945247e-17, 0.06242685196556241, 0.5]]], 'cz': 21, 'residual': 1.878558504603599e-15, 'source': 'direct_search\\candidate_7_3_21cz.cirq.json'}, {'n': 7, 'k': 3, 'qubits': [[3, 5], [4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7]], 'moments': [[['sq', 0, 2.98990646078402e-17, -0.14130179031352963, -0.5], ['sq', 1, 0.5000000000000001, -0.5721993096269484, 0.5], ['sq', 2, 0.75, 1.0, 0.5], ['sq', 3, 0.5, 1.0, 0.5], ['sq', 4, 0.5439241318645377, -0.3249105734692026, 0.5], ['sq', 5, 0.3609933037565794, -0.8423735388425244, 0.5], ['sq', 6, 0.46527487426094694, 0.9999999999999996, 0.5]], [['cz', 4, 5]], [['sq', 4, 0.2875635878708291, -0.545223962295414, 0.5], ['sq', 5, 0.5326074210506879, -0.6849929718414312, 0.5]], [['cz', 4, 1], ['cz', 5, 6]], [['sq', 4, 1.0, 0.0, -0.1409758450440428], ['sq', 1, 0.5319047913303867, 0.37217908897987595, 0.5], ['sq', 5, 0.5000000000000001, -0.5965764026111344, 0.5], ['sq', 6, 0.35039678452811995, 0.7914355565902798, 0.5]], [['cz', 6, 3], ['cz', 0, 1], ['cz', 5, 4]], [['sq', 6, 0.5, 0.0, 0.5], ['sq', 3, 0.5, -0.6190073583024396, 0.5], ['sq', 0, 0.23346263634960418, 0.096803106226659, 0.5], ['sq', 1, 0.24326790811419663, -0.36663325029496896, 0.5], ['sq', 5, 0.2403798762059917, 0.10782013429449944, 0.5], ['sq', 4, 0.24999999999999992, 0.0, 0.5]], [['cz', 4, 5], ['cz', 1, 2]], [['sq', 4, 0.75, 0.6576794517407847, 0.5], ['sq', 5, 1.097495530934548e-18, -0.0364767279976379, 0.5], ['sq', 1, 0.5000000000000002, 0.1959132760153035, 0.5], ['sq', 2, 0.632027772969383, -0.4999999999999998, 0.49999999999999994]], [['cz', 6, 5], ['cz', 2, 3]], [['sq', 6, 0.5, -0.09437765891219874, 0.5], ['sq', 5, 0.5, 0.0, 0.5], ['sq', 2, 0.5000000000000002, 0.821590736112018, 0.5], ['sq', 3, 1.0, 0.0, -0.6487370103845658]], [['cz', 5, 4], ['cz', 1, 2]], [['sq', 5, 0.5, 0.7216350050329243, 0.5], ['sq', 4, 1.0, 0.0, 0.13839452956603532], ['sq', 1, 0.6666666666666667, -0.3548474413474296, 0.5], ['sq', 2, 0.8291885548476134, -0.7230611801175955, 0.5]], [['cz', 5, 6], ['cz', 3, 2]], [['sq', 5, 4.354709451073785e-17, -0.10252539764163782, -0.5], ['sq', 6, 1.0, 0.0, 0.5123659742395437], ['sq', 3, 0.5000000000000001, 1.0, 0.5], ['sq', 2, 0.9718346440887352, -0.9325561962343139, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.4205497461326053, 0.4497803247436556, 0.5], ['sq', 2, 0.5000000000000002, -0.3524195578496536, 0.5]], [['cz', 1, 0], ['cz', 2, 3]], [['sq', 1, 0.6326561032694434, 0.10782350273907904, 0.5], ['sq', 0, 0.7010158862704285, 0.9974576091629359, 0.5], ['sq', 2, 1.0, 0.0, -0.4262097789248269], ['sq', 3, 0.5, 0.08901182644825756, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.730193438975439, -0.6039279414745997, 0.5], ['sq', 0, 0.6488254961482807, 0.3198209245413959, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.5000000000000001, 0.0, 0.5], ['sq', 2, 0.5, 0.0, 0.5]], [['cz', 0, 1]], [['sq', 0, 0.5, -0.5358418739675712, 0.5], ['sq', 1, 0.5, 0.7403481845821607, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.5, 1.0, 0.5], ['sq', 2, 0.5, -0.9109881735517429, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.5000000000000001, -0.4109881735517429, 0.5], ['sq', 0, 3.3510915571587396e-17, 0.12485370041582833, 0.5]]], 'cz': 22, 'residual': 2.089038344358335e-15, 'source': 'direct_search\\candidate_7_3_22cz.cirq.json'}, {'n': 7, 'k': 3, 'qubits': [[3, 5], [4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7]], 'moments': [[['sq', 0, 4.026550121215985e-17, -0.15157534350475665, -0.5], ['sq', 1, 0.5000000000000001, 0.482723358479761, 0.5], ['sq', 2, 0.23886579606389338, -0.9512380823916957, 0.5], ['sq', 3, 0.5, 1.0, 0.5], ['sq', 4, 0.5272951270854374, -0.3229505379680875, 0.5], ['sq', 5, 0.33101002956971337, -0.8450384888091407, 0.5], ['sq', 6, 0.46519921011652454, -0.9895203639014174, 0.5]], [['cz', 4, 5]], [['sq', 4, 0.27207095110843904, -0.06618040179862028, 0.5], ['sq', 5, 0.5556458868541129, 0.41510310269895445, 0.5]], [['cz', 4, 1], ['cz', 5, 6]], [['sq', 4, 1.0, 0.0, -0.07623350264259354], ['sq', 1, 0.3514838800411226, 0.09361201765563676, 0.5], ['sq', 5, 0.49999999999999983, 0.4231031295975568, -0.5], ['sq', 6, 0.35076045514547877, 0.7163715908662196, 0.5]], [['cz', 6, 3], ['cz', 1, 2], ['cz', 5, 4]], [['sq', 6, 0.49999999999999994, 0.0, 0.5], ['sq', 3, 0.5, -0.9166839853291973, 0.5], ['sq', 1, 0.9900041759732306, -0.4843137578171919, -0.5], ['sq', 2, 0.7711182708497809, 0.9337557392109144, 0.5], ['sq', 5, 0.3272429637449465, 0.07953304887928203, 0.5], ['sq', 4, 0.25000000000000006, 0.0, 0.5]], [['cz', 0, 1], ['cz', 4, 5]], [['sq', 0, 0.23697080170306567, 0.05770345374842967, 0.5], ['sq', 1, 0.32869747159644436, -0.26195256935187294, 0.5], ['sq', 4, 0.7500000000000001, -0.06549949150411205, 0.5], ['sq', 5, 5.085226484029666e-17, -0.04072545823755691, -0.5]], [['cz', 1, 2], ['cz', 6, 5]], [['sq', 1, 0.39753027189345563, 0.1516223944224312, 0.5], ['sq', 2, 0.7719260562178989, -0.4731997924689182, 0.5], ['sq', 6, 0.5000000000000001, -0.06961232067008205, 0.5], ['sq', 5, 0.5, 0.0, 0.5]], [['cz', 2, 3], ['cz', 5, 4]], [['sq', 2, 0.673420111688748, 0.6906617979582457, 0.5], ['sq', 3, 1.0, 0.0, -0.722739214485858], ['sq', 5, 0.5, 0.8239795872639557, 0.5], ['sq', 4, 1.0, 0.0, -0.16825483622922666]], [['cz', 1, 2], ['cz', 5, 6]], [['sq', 1, 0.6897931699087418, -0.3496653358174173, 0.5], ['sq', 2, 0.6530606501486816, 0.14261179475289953, 0.5], ['sq', 5, 8.978670961734652e-18, -0.09498976821829697, -0.5000000000000002], ['sq', 6, 1.0, 0.0, -0.42031125081221177]], [['cz', 3, 2]], [['sq', 3, 0.5000000000000001, 1.0, 0.5], ['sq', 2, 0.9325692279808523, 0.1523869309974868, -0.5]], [['cz', 1, 2]], [['sq', 1, 0.2739602599988857, 0.30591728890887104, 0.5], ['sq', 2, 0.4834388013723244, -0.20505524041168566, 0.5]], [['cz', 1, 0], ['cz', 2, 3]], [['sq', 1, 0.6653920436706233, 0.061100452762441915, 0.5], ['sq', 0, 0.6968176325830696, -0.9925451550285065, 0.5], ['sq', 2, 1.0, 0.0, -0.34465792705912524], ['sq', 3, 0.5, 0.20776190192448496, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.7439872562978573, -0.587046735374426, 0.49999999999999994], ['sq', 0, 0.6168537417567912, 0.3798697858654325, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.5000000000000001, 0.0, 0.5], ['sq', 2, 0.5, 0.0, 0.5]], [['cz', 0, 1]], [['sq', 0, 0.49999999999999994, -0.6605817466339126, 0.5], ['sq', 1, 0.5, 0.8259962754617947, 0.5]], [['cz', 1, 2]], [['sq', 1, 0.5, 1.0, 0.5], ['sq', 2, 0.5, 0.958356468729674, 0.5]], [['cz', 1, 0]], [['sq', 1, 0.5, -0.541643531270326, 0.5], ['sq', 0, 1.2305413739883816e-17, 0.1189382153635864, -0.5000000000000002]]], 'cz': 23, 'residual': 2.218076107525605e-15, 'source': 'direct_search\\candidate_7_3_23cz.cirq.json'}], (8, 4): [{'n': 8, 'k': 4, 'qubits': [[4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5], [6, 3], [6, 4]], 'moments': [[['sq', 0, 0.5000000000000001, 334.19900748501783, 0.5], ['sq', 1, 0.5000000000000002, -46.729634474536766, 0.5], ['sq', 2, 0.5, -1.0, 0.5], ['sq', 3, 0.38157135719845003, -2.6287036779257007e-16, 0.5], ['sq', 4, 0.5, -1.0, 0.5], ['sq', 5, 0.49999999999999994, -1.0, 0.5], ['sq', 6, 0.49999999999999994, -175.8646824293776, 0.5], ['sq', 7, 0.5, -0.5, 0.5]], [['cz', 0, 1]], [['sq', 0, 426.6734797534457, 1271.5290474887195, 0.5], ['cz', 1, 4]], [['cz', 0, 3], ['sq', 1, -1.0, -369.0919005639051, 0.5], ['sq', 4, 0.5, -2870.2052182402376, 0.5]], [['sq', 0, 0.5, 1.4105438124889669, 0.5], ['sq', 3, 1.5, -594.4742942203508, 0.5], ['cz', 1, 2]], [['cz', 3, 6], ['sq', 1, 0.5000000000000001, 1.800957539780115e-17, 0.5], ['sq', 2, 0.5, -1019.4001780899035, 0.5]], [['sq', 3, 0.32158739372375184, 0.7790991158233053, 0.5], ['sq', 6, 310.33457894421707, 528.9571789265273, 0.5], ['cz', 0, 1]], [['cz', 6, 3], ['sq', 0, 0.3040867239846963, 1470.8319969040817, 0.5], ['sq', 1, 0.8040867239846964, -1.0, 0.5]], [['cz', 1, 0], ['cz', 3, 4]], [['sq', 1, 0.5000000000000002, -0.5, 0.5], ['sq', 0, -1.0, 1471.0819969040815, 0.5], ['sq', 3, 0.34704221097538546, -27.250000000000004, 0.5], ['sq', 4, 0.5, -865.9017426483061, 0.5]], [['cz', 1, 2], ['cz', 5, 4]], [['sq', 1, 0.75, -1.0, 0.5], ['sq', 2, 0.5, 898.7377959146387, 0.5], ['sq', 5, -1.5, 865.9017426483061, 0.5], ['sq', 4, 0.5, -0.5, 0.5]], [['cz', 1, 2], ['cz', 3, 4]], [['sq', 1, 0.5, -0.49999999999999994, 0.5], ['sq', 2, -1.0000000000000002, 898.7377959146387, 0.5], ['sq', 4, 0.5, 2223.948355692352, 0.5], ['cz', 3, 6]], [['cz', 1, 0], ['cz', 4, 5], ['sq', 3, 0.49999999999999994, -2.5933950436548905e-17, 0.5], ['sq', 6, 1.2361429389567886, 28.227185525828506, 0.5]], [['sq', 1, 0.5, -1.0, 0.5], ['sq', 0, 0.5, 4.7390340801119747e-17, 0.5], ['sq', 4, -99.00000000000001, -2224.0516443076485, 0.5], ['sq', 5, 0.5000000000000001, -1.0, 0.5], ['cz', 3, 6]], [['sq', 3, 0.1959132760153036, 1.0, 0.5], ['sq', 6, -0.3040867239846964, 1.25, 0.5]], [['cz', 3, 4]], [['sq', 3, 0.49999999999999994, -2.0, 0.5], ['sq', 4, 24.25, 159.36704601079956, 0.5]], [['cz', 4, 3]], [['sq', 4, 1.0, 158.86704601079956, 0.5], ['cz', 6, 3]], [['sq', 6, 0.5, 0.45740660163505914, 0.5], ['sq', 3, 0.5, -1.0, 0.5]], [['cz', 3, 6]], [['sq', 6, 18.0, -0.9574066016350592, 0.5], ['cz', 4, 3]], [['sq', 4, 0.5, -13.965601167392037, 0.5]], [['cz', 7, 4]], [['sq', 7, -13.465601167392036, -546.5189596561949, 0.5], ['sq', 4, 0.5, -305.3675817638942, 0.5]], [['cz', 7, 6], ['cz', 4, 1]], [['sq', 7, -1.0, -547.8598484625082, 0.5], ['sq', 6, 0.5, 0.1591111936867182, 0.5], ['sq', 4, -1.0, -305.8675817638942, 0.5], ['sq', 1, 0.5, -0.5, 0.5]], [['cz', 4, 3], ['cz', 1, 0]], [['sq', 4, 0.5000000000000001, 1.0, 0.5], ['sq', 3, 0.5000000000000001, 0.15911119368671817, 0.5], ['sq', 1, 0.49999999999999994, 0.6591111936867182, 0.5], ['sq', 0, 0.5, 0.1591111936867182, 0.5]], [['cz', 4, 5], ['cz', 1, 2]], [['sq', 4, 0.5000000000000001, 0.6591111936867182, 0.5], ['sq', 5, 0.5, 0.1591111936867182, 0.5], ['sq', 2, 0.5, 0.15911119368671822, 0.5]]], 'cz': 29, 'residual': 9.027192022015129e-13, 'source': 'local_prune\\candidate_8_4_29cz_sq.cirq.json'}, {'n': 8, 'k': 4, 'qubits': [[4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5], [6, 3], [6, 4]], 'moments': [[['sq', 0, 1.0, -2.666666841446821, 0.5], ['sq', 1, 0.5000000000000001, 0.9006708950338593, 0.5], ['sq', 2, 0.5, -1.499999999999999, 0.5], ['sq', 3, 0.5, 0.5366082371567077, 0.5], ['sq', 4, 0.5, -0.9436285020575691, 0.5], ['sq', 5, 0.5, -1.0, 0.5], ['sq', 6, 0.5, -2.66244818818997, 0.5], ['sq', 7, 0.5000000000000001, 2.4236338191442246, 0.5]], [['cz', 0, 1]], [['sq', 0, 2.7815168817445013, -8.5, 0.5], ['cz', 1, 4]], [['cz', 0, 3], ['sq', 1, -0.5148399988034192, 4.4593318750267565, 0.5], ['sq', 4, 1.4666719890476991, 5.655122928890573, 0.5]], [['sq', 0, 1.1254015258841572, 0.5000000000000001, 0.5], ['cz', 3, 6], ['cz', 1, 2]], [['sq', 3, 0.35436630366582683, -0.6402188990037717, 0.5], ['sq', 6, 0.8131920051538619, -0.008772269935714757, 0.5], ['sq', 1, 0.6959132760153036, 1.1857804378766844e-17, 0.5], ['sq', 2, 1.9769532836069301, -4.500000000000001, 0.5]], [['cz', 6, 3], ['cz', 0, 1]], [['sq', 3, -0.5, 1.5, 0.5], ['sq', 0, 0.5, -2.519002068282071, 0.5], ['sq', 1, 0.3040867239846964, 0.5, 0.5]], [['cz', 1, 0], ['cz', 3, 4]], [['sq', 0, -2.0, -1.980997931717929, 0.5], ['sq', 3, 0.5000000000000001, -3.2500000000000004, 0.5], ['sq', 4, 0.49999999999999994, -3.0658480624808258, 0.5], ['cz', 1, 2]], [['cz', 5, 4], ['sq', 1, 0.75, -1.0000000000000002, 0.5], ['sq', 2, 0.5000000000000001, -6.91238022627052e-17, 0.5]], [['sq', 5, 0.5, 2.5658480624808258, 0.5], ['sq', 4, 0.8040867239846964, -0.5, 0.5], ['cz', 1, 2]], [['cz', 3, 4], ['sq', 1, 0.75, -0.5000000000000001, 0.5], ['sq', 2, -0.49999999999999994, 1.5846204109109305e-17, 0.5]], [['sq', 4, 0.5, -1.0, 0.5], ['cz', 1, 0], ['cz', 3, 6]], [['cz', 4, 5], ['sq', 1, 0.5, -6.721649171193046e-17, 0.5], ['sq', 0, 0.5, -0.5000000000000001, 0.5], ['sq', 3, -0.5, 1.0, 0.5], ['sq', 6, 1.5, 2.666666666666667, 0.5]], [['cz', 3, 6]], [['sq', 3, 1.5, 1.0, 0.5], ['sq', 6, 0.49999999999999994, -1.0, 0.5]], [['cz', 3, 4]], [['sq', 3, 0.5000000000000001, -1.0, 0.5], ['sq', 4, 0.75, -0.5000000000000001, 0.5]], [['cz', 4, 3]], [['cz', 6, 3]], [['sq', 6, 0.8040867239846964, -1.0571715100448908, 0.5], ['sq', 3, 0.5, -0.5, 0.5]], [['cz', 3, 6]], [['sq', 3, 1.25, -0.5, 0.5], ['sq', 6, 1.0000000000000002, -1.0571715100448908, 0.5]], [['cz', 4, 3]], [['sq', 4, -0.5, -1.8574995682636801, 0.5]], [['cz', 4, 1]], [['sq', 4, -1.0, -1.8574995682636801, 0.5]], [['cz', 7, 4]], [['sq', 4, 0.5, 0.5, 0.5], ['cz', 7, 6]], [['cz', 4, 1], ['sq', 7, -1.0, 1.0977607716695863, 0.5], ['sq', 6, 0.5000000000000001, -0.32587304747463824, 0.5]], [['sq', 4, 0.5000000000000001, 2.1585623275700458e-17, 0.5], ['sq', 1, 0.5, -0.32587304747463824, 0.5]], [['cz', 4, 3], ['cz', 1, 0]], [['sq', 4, 0.5000000000000001, 1.174126952525362, 0.5], ['sq', 3, 0.5, -0.32587304747463824, 0.5], ['sq', 0, 0.49999999999999994, -0.32587304747463824, 0.5], ['cz', 1, 2]], [['cz', 4, 5], ['sq', 2, 0.5, -0.32587304747463824, 0.5]], [['sq', 5, 0.5, -0.3258730474746382, 0.5]]], 'cz': 30, 'residual': 1.7249703312307386e-15, 'source': 'local_prune\\candidate_8_4_30cz_sq.cirq.json'}, {'n': 8, 'k': 4, 'qubits': [[3, 5], [3, 6], [4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7]], 'moments': [[['sq', 0, 0.5, 13.238772182139625, 0.5], ['sq', 1, 0.5000000000000001, -4.947137167066211, 0.5], ['sq', 2, 0.3373083465539407, 0.4999999999999998, 0.5], ['sq', 3, 0.24999999999999997, 1.0000000000000002, 0.5], ['sq', 4, 0.5000000000000001, -1.0, 0.5], ['sq', 5, 0.3648337925259496, 0.5000000000000001, 0.5], ['sq', 6, 0.39695342131456696, -1.0, 0.5], ['sq', 7, 0.6285082223277061, -1.0, 0.5]], [['cz', 5, 6]], [['sq', 6, 0.34719543335082964, -1.5, 0.5], ['cz', 5, 2]], [['cz', 6, 7], ['sq', 5, -1.5, -1.2260245808659396, 0.5], ['cz', 2, 3]], [['sq', 6, 0.5, 0.12907995580997594, 0.5], ['sq', 7, 0.2727792189844248, 32.74664207904849, 0.5], ['sq', 2, 0.5, 0.19591327601530362, 0.5], ['sq', 3, 0.5774503045752932, -0.49999999999999983, 0.5]], [['cz', 6, 5], ['cz', 7, 4]], [['sq', 6, 0.6909541236524172, -0.25, 0.5], ['sq', 5, 0.3823556079069398, 0.37337585721442496, 0.5], ['sq', 7, 0.5000000000000001, 0.5, 0.5], ['sq', 4, 0.5, -59.44425150035996, 0.5]], [['cz', 3, 4], ['cz', 5, 6]], [['sq', 3, 0.5, 1.4369335816434734, 0.5], ['sq', 4, -1.0, -27.197609421311473, 0.5], ['sq', 5, 0.5, -0.5, 0.5], ['cz', 7, 6]], [['cz', 2, 3], ['sq', 7, 0.8393092936803588, -1.0, 0.5], ['sq', 6, 0.5000000000000001, 0.5, 0.5]], [['sq', 2, 0.6666666666666666, 14.366399381754048, 0.5], ['sq', 3, 1.6359914822683863, 0.8337713376832331, 0.5], ['cz', 6, 5]], [['cz', 4, 3], ['sq', 5, 0.5, 4.091918200920406e-17, 0.5], ['cz', 6, 7]], [['sq', 4, 0.49999999999999994, -1.0, 0.5], ['sq', 3, 1.4570986387292764, 0.06188759519792074, 0.5]], [['cz', 2, 3]], [['sq', 3, 0.5, 1.5, 0.5], ['cz', 2, 0]], [['cz', 3, 4], ['sq', 2, 0.3040867239846963, 0.25, 0.5], ['sq', 0, 7.289482290752495, 18.564500924359287, 0.5]], [['sq', 3, -1.5, 1.5, 0.5], ['sq', 4, 0.5, -1.0, 0.5], ['cz', 2, 0]], [['cz', 4, 7], ['sq', 2, 0.5000000000000001, 19.809893615126242, 0.5], ['sq', 0, 0.5000000000000001, -20.809893615126242, 0.5]], [['sq', 4, -0.5000000000000001, -1.0, 0.5], ['cz', 2, 3]], [['sq', 2, 0.5, 3.122892804227693e-18, 0.5], ['sq', 3, 0.49999999999999994, -0.5, 0.5]], [['cz', 0, 2]], [['sq', 0, 0.5000000000000001, -1.0000000000000002, 0.5], ['sq', 2, 0.5, -20.809893615126242, 0.5]], [['cz', 2, 3]], [['sq', 2, 0.5, 1.0000000000000002, 0.5], ['sq', 3, 0.5000000000000001, -0.49999999999999983, 0.5]], [['cz', 2, 0], ['cz', 3, 6]], [['sq', 0, -0.5, -0.9999999999999999, 0.5], ['cz', 3, 4], ['cz', 2, 5]], [['sq', 3, 0.5000000000000001, -1.0, 0.5], ['sq', 2, 0.5, -16.97290179418562, 0.5]], [['cz', 0, 2]], [['sq', 0, 0.5, -1.0, 0.5], ['sq', 2, 4.05292212614211e-17, 15.97290179418562, 0.5]], [['cz', 1, 0]], [['sq', 0, 0.5, 0.03550983184268789, 0.5], ['cz', 1, 3]], [['cz', 0, 2], ['sq', 1, -1.0, -3.911627335223523, 0.5], ['sq', 3, 0.5, 0.035509831842687925, 0.5]], [['sq', 2, 0.5, 0.03550983184268791, 0.5], ['cz', 3, 4]], [['cz', 2, 5], ['sq', 4, 0.5, 0.03550983184268794, 0.5], ['cz', 3, 6]], [['sq', 5, 0.5, 0.035509831842687904, 0.5], ['cz', 4, 7], ['sq', 6, 0.5, 0.03550983184268792, 0.5]], [['sq', 7, 0.5, 0.0355098318426879, 0.5]]], 'cz': 33, 'residual': 7.229191963882663e-15, 'source': 'local_prune\\candidate_8_4_33cz_extension0_sq.cirq.json'}]}
'ColumnForge native portfolio and state-sensitive calibration selection.\n\nAppended to a frozen, attributed baseline by build.py. No offline numerical\nsearch libraries are imported by the emitted submission.\n'
_cf__columnforge_baseline = _cf_create_dicke_state
_cf__columnforge_adaptive_baseline = _cf_create_dicke_state_adaptive

def _cf__rigid_mappings(circuit, logical, cz, sq):
    graph = _cf_calibration_graph(cz, sq)
    index = {q: i for i, q in enumerate(logical)}
    edges = {tuple((index[q] for q in op.qubits)) for op in circuit.all_operations() if len(op.qubits) == 2}
    seen = set()
    for flip in (False, True):
        for sr, sc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            coords = [(sc * q.col, sr * q.row) if flip else (sr * q.row, sc * q.col) for q in logical]
            ar, ac = coords[0]
            for p in graph:
                mapped = tuple((_cf_cirq.GridQubit(p.row + r - ar, p.col + c - ac) for r, c in coords))
                if mapped in seen or not all((q in graph for q in mapped)):
                    continue
                seen.add(mapped)
                if all((mapped[b] in graph[mapped[a]] for a, b in edges)):
                    yield mapped

def _cf__flexible_mappings(circuit, logical, k, cz, sq, zero, one, width=384, budget=2):
    """Beam search over graph embeddings, allowing arms to bend around defects."""
    started = _cf_time.monotonic()
    n = len(logical)
    ids = {q: i for i, q in enumerate(logical)}
    singles, pairs = (_cf_Counter(), _cf_Counter())
    neighbors = {i: set() for i in range(n)}
    for op in circuit.all_operations():
        loc = [ids[q] for q in op.qubits]
        if len(loc) == 1:
            singles[loc[0]] += 1
        else:
            a, b = sorted(loc)
            pairs[a, b] += 1
            neighbors[a].add(b)
            neighbors[b].add(a)
    graph = _cf_calibration_graph(cz, sq)
    remaining, order = (set(range(n)), [])
    while remaining:
        q = max(remaining, key=lambda i: (len(neighbors[i].intersection(order)), len(neighbors[i]), sum((pairs[tuple(sorted((i, j)))] for j in neighbors[i])), -i))
        order.append(q)
        remaining.remove(q)
    pos = {q: i for i, q in enumerate(order)}
    scale = 10000 / _cf_math.comb(n, k) * k * (n - k) / (n * max(1, n - 1))
    asym = {q: _cf_math.log1p(-_cf_rate(one.get(q, (0.02, 0)))) - _cf_math.log1p(-_cf_rate(zero.get(q, (0.01, 0)))) for q in graph}
    ro = {q: -(1 - k / n) * _cf_math.log1p(-_cf_rate(zero.get(q, (0.01, 0)))) - k / n * _cf_math.log1p(-_cf_rate(one.get(q, (0.02, 0)))) for q in graph}
    states = [(0.0, (), 0.0, 0.0, 0.0)]
    for i, q in enumerate(order):
        before = [j for j in neighbors[q] if pos[j] < i]
        expanded = []
        for _, assigned, add, ds, d2 in states:
            options = set(graph) if not before else set(graph[assigned[pos[before[0]]]])
            for j in before[1:]:
                options.intersection_update(graph[assigned[pos[j]]])
            options.difference_update(assigned)
            for p in sorted(options):
                if len(graph[p]) < len(neighbors[q]):
                    continue
                newadd = add + ro[p] + singles[q] * _cf_rate(sq[p]) + sum((pairs[tuple(sorted((q, j)))] * graph[p][assigned[pos[j]]] for j in before))
                nds, nd2 = (ds + asym[p], d2 + asym[p] ** 2)
                score = newadd + scale * max(0.0, nd2 - nds ** 2 / (i + 1))
                expanded.append((score, assigned + (p,), newadd, nds, nd2))
        if not expanded:
            return []
        expanded.sort(key=lambda x: (x[0], x[1]))
        states = expanded[:width]
        if _cf_time.monotonic() - started > budget and i + 1 < n:
            return []
    return [tuple((assigned[pos[i]] for i in range(n))) for _, assigned, _, _, _ in states]

def _cf__decode_columnforge(data, complement=False):
    qs = [_cf_cirq.GridQubit(*p) for p in data['qubits']]
    moments = []
    for moment in data['moments']:
        ops = []
        for item in moment:
            if item[0] == 'cz':
                ops.append(_cf_cirq.CZ(qs[item[1]], qs[item[2]]))
            else:
                ops.append(_cf_cirq.PhasedXZGate(x_exponent=item[2], z_exponent=item[3], axis_phase_exponent=item[4])(qs[item[1]]))
        moments.append(_cf_cirq.Moment(ops))
    c = _cf_cirq.Circuit(moments)
    if complement:
        c = _cf_cirq.merge_single_qubit_gates_to_phxz(c + _cf_cirq.Circuit(_cf_cirq.X.on_each(qs)))
    return (c, qs)

def _cf_create_dicke_state(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    started = _cf_time.monotonic()
    n, k = (num_qubits, num_excitations)
    args = (cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    try:
        baseline, outputs = _cf__columnforge_baseline(n, k, *args)
    except NotImplementedError:
        baseline, outputs = (None, set())
    if k in (0, n) or n <= 2 or n > 8:
        if baseline is None:
            raise NotImplementedError('No valid calibrated placement found')
        return (baseline, outputs)
    families = [] if baseline is None else [(baseline, sorted(outputs))]
    for data in _cf_COLUMNFORGE_TEMPLATES.get((n, min(k, n - k)), []):
        families.append(_cf__decode_columnforge(data, complement=k > n // 2))
    if n <= 8:
        try:
            c, q = _cf_willow_divide_circuit(n, min(k, n - k)) if k not in (1, n - 1) else (None, None)
            if c is not None:
                if k > n // 2:
                    c = _cf_cirq.merge_single_qubit_gates_to_phxz(c + _cf_cirq.Circuit(_cf_cirq.X.on_each(q)))
                families.append((c, sorted(q)))
        except NotImplementedError:
            pass
    if baseline is None:
        best = (float('inf'), None, set())
    else:
        best = (_cf_small_proxy(baseline, sorted(outputs), k, *args), baseline, outputs) if n <= 8 else (_cf_circuit_calibration_cost(baseline, outputs, k, *args), baseline, outputs)
    seen = set()
    for c, logical in families:
        if _cf_time.monotonic() - started > 36:
            break
        logical = sorted(logical)
        key = _cf_cirq.FrozenCircuit(c.transform_qubits(dict(zip(logical, _cf_cirq.LineQubit.range(n)))))
        if key in seen:
            continue
        seen.add(key)
        mappings = list(dict.fromkeys(list(_cf__rigid_mappings(c, logical, *args[:2])) + _cf__flexible_mappings(c, logical, k, *args)))
        if not mappings:
            continue
        if n <= 8:
            scores = _cf_rank_mappings(c, logical, mappings, k, *args)
            indices = _cf_np.argsort(scores)[:3 if n >= 8 else 6]
        else:
            costs = [_cf_circuit_calibration_cost(c.transform_qubits(dict(zip(logical, m))), m, k, *args) for m in mappings]
            indices = _cf_np.argsort(costs)[:1]
        for i in indices:
            if _cf_time.monotonic() - started > 45:
                break
            mapped = c.transform_qubits(dict(zip(logical, mappings[int(i)])))
            qs = set(mappings[int(i)])
            score = _cf_small_proxy(mapped, sorted(qs), k, *args) if n <= 8 else _cf_circuit_calibration_cost(mapped, qs, k, *args)
            if score < best[0] - 1e-12:
                best = (score, mapped, qs)
    if best[1] is None:
        raise NotImplementedError('No valid calibrated placement found')
    return (best[1], best[2])

def _cf_create_dicke_state_adaptive(num_qubits, num_excitations):
    """Reuse shorter verified native circuits for the measurement-free bonus."""
    c, qs = _cf__columnforge_adaptive_baseline(num_qubits, num_excitations)
    for data in _cf_COLUMNFORGE_TEMPLATES.get((num_qubits, num_excitations), []):
        candidate, outputs = _cf__decode_columnforge(data)
        if len(candidate) < len(c):
            _cf_cirq_google.Willow105.validate_circuit(candidate)
            c, qs = (candidate, set(outputs))
    return (c, qs)
_cf_COLUMNFORGE_TEMPLATES[6, 3] = _cf_COLUMNFORGE_TEMPLATES.get((6, 3), []) + [{'n': 6, 'k': 3, 'qubits': [[4, 3], [4, 4], [5, 3], [5, 4], [6, 3], [6, 4]], 'moments': [[['sq', 0, 1376.3040867239847, 6203.0, 0.5], ['sq', 1, -0.3040867239846963, 1.0, 0.5], ['sq', 2, -3326.6309898804348, 682.5, 0.5], ['sq', 3, 0.227185525828505, 4.180927621920264e-17, 0.5], ['sq', 4, 0.5, -7.742942702846523e-17, 0.5], ['sq', 5, 0.5, -0.5, 0.5]], [['cz', 0, 2]], [['sq', 0, -1764.3040867239847, -1590.0000000000002, 0.5], ['cz', 2, 3]], [['sq', 2, -0.5, -0.25000000000000006, 0.5], ['sq', 3, -0.227185525828505, 1.0, 0.5]], [['cz', 1, 3], ['cz', 2, 4]], [['sq', 1, 0.6959132760153038, -2.778751687005279e-17, 0.5], ['sq', 3, 0.2499999999999999, -1.0, 0.5], ['sq', 2, 0.5, 0.49999999999999994, 0.5]], [['cz', 0, 2], ['cz', 3, 1]], [['sq', 0, 0.5000000000000001, 0.5000000000000001, 0.5], ['sq', 2, -0.25, -2.5, 0.5], ['sq', 3, 0.7500000000000001, -1.0, 0.5]], [['cz', 2, 4], ['cz', 5, 3]], [['sq', 3, 0.5, 0.5000000000000001, 0.5], ['cz', 5, 4]], [['cz', 3, 1], ['sq', 5, 0.5, 4536.370029688775, 0.5], ['sq', 4, 0.5, 4536.870029688775, 0.5]], [['sq', 1, 0.5, 0.5, 0.5], ['cz', 3, 2]], [['cz', 1, 0], ['sq', 3, -0.49999999999999994, -4537.629970311225, 0.5], ['sq', 2, 3813.5, 2459.870029688775, 0.5]], [['sq', 1, 0.49999999999999994, 4537.370029688775, 0.5], ['sq', 0, 0.5, 4536.870029688775, 0.5]]], 'cz': 12, 'residual': 1.3081409952687938e-12, 'source': 'completion_search\\native_candidates\\candidate_6_3_12cz_sq.cirq.json'}]
_cf_COLUMNFORGE_TEMPLATES[8, 4] = _cf_COLUMNFORGE_TEMPLATES.get((8, 4), []) + [{'n': 8, 'k': 4, 'qubits': [[4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5], [6, 3], [6, 4]], 'moments': [[['sq', 0, 0.21859111351161822, -0.5000000000000016, 0.5], ['sq', 1, 0.5, -5991.363961377644, 0.5], ['sq', 2, 0.5, -1.0, 0.5], ['sq', 3, 0.6459721624073824, -2.0, 0.5], ['sq', 4, 0.5, -2516.846488170638, 0.5], ['sq', 5, 0.5, -0.5, 0.5], ['sq', 6, 0.6959132760153036, -4.3401850930671934e-17, 0.5], ['sq', 7, 0.5, -0.5, 0.5]], [['cz', 0, 1]], [['sq', 0, -1.0038758162973733, -2.0000000000000013, 0.5], ['cz', 1, 4]], [['cz', 0, 3], ['sq', 1, 0.5228141721046287, -649.6702398776288, 0.5], ['sq', 4, 1614.5060204698057, 6550.344745910827, 0.5]], [['sq', 0, 0.34546860330888607, -1.0, 0.5], ['sq', 3, -1.153912167686454, 6822.407375740574, 0.5], ['cz', 1, 2]], [['cz', 6, 3], ['sq', 1, 0.7730855641139139, -0.12687635107644896, 0.5], ['sq', 2, 0.5, -649.9042862624343, 0.5]], [['sq', 6, 0.19591327601530362, 0.5000000000000001, 0.5], ['sq', 3, -1.0000000000000002, 6822.407375740574, 0.5], ['cz', 0, 1]], [['cz', 3, 4], ['sq', 0, 0.5, 0.5000000000000001, 0.5], ['sq', 1, 0.2858525318292365, -59.09692417965118, 0.5]], [['sq', 3, 0.5, 24901.491702879117, 0.5], ['sq', 4, 0.24063253699046155, 3670.7500000000005, 0.5], ['cz', 1, 0]], [['cz', 3, 4], ['sq', 0, 0.3213834548125703, 0.75, 0.5], ['cz', 1, 2]], [['sq', 3, -0.9999999999999996, 24901.741702879117, 0.5], ['sq', 1, 0.75, -1.0000000000000002, 0.5], ['sq', 2, -1.0, -1299.0932286993045, 0.5]], [['cz', 3, 6], ['cz', 2, 5]], [['sq', 3, 0.5000000000000001, -0.5, 0.5], ['sq', 6, 0.24999999999999992, -1.0000000000000002, 0.5], ['sq', 2, 0.5, 18021.71484241765, 0.5], ['cz', 4, 5]], [['cz', 1, 2], ['sq', 4, 0.5, -1.0, 0.5], ['sq', 5, 0.5, -0.5, 0.5]], [['sq', 1, 0.5, -0.5, 0.5], ['sq', 2, -1.0, 18020.71484241765, 0.5], ['cz', 3, 4]], [['cz', 1, 0], ['sq', 3, 0.5, -1343.4996989341191, 0.5], ['sq', 4, 0.5, -0.5000000000000001, 0.5]], [['sq', 1, 0.5, -0.5, 0.5], ['sq', 0, 0.5, -1.7476160733215138e-17, 0.5], ['cz', 6, 3], ['cz', 7, 4]], [['sq', 6, 0.5, -0.5, 0.5], ['sq', 3, -1.0, -1342.4996989341191, 0.5], ['sq', 4, 0.5, -13358.71404856638, 0.5]], [['cz', 7, 6], ['cz', 4, 1]], [['sq', 7, 0.5, 0.6936596704594525, 0.5], ['sq', 6, 0.5, 0.19365967045945254, 0.5], ['sq', 4, -1.0, -13359.21404856638, 0.5], ['sq', 1, 0.5, -0.5, 0.5]], [['cz', 4, 3], ['cz', 1, 0]], [['sq', 4, 0.5, -0.30634032954054746, 0.5], ['sq', 3, 0.5, 0.19365967045945254, 0.5], ['sq', 1, 0.5, 0.6936596704594525, 0.5], ['sq', 0, 0.5, 0.19365967045945254, 0.5]], [['cz', 4, 5], ['cz', 1, 2]], [['sq', 5, 0.5, 0.19365967045945254, 0.5], ['sq', 2, 0.5, 0.19365967045945254, 0.5]]], 'cz': 24, 'residual': 3.87457269419548e-12, 'source': 'completion_search\\native_candidates\\candidate_8_4_24cz_sq.cirq.json'}]

"""DuQuantum 2026 - Google Quantum AI Dicke-state challenge submission.
Team: Kai Sandberg, Will, Ben (+ Claude / GPT-Astra).
Exact Dicke-state circuits (CZ + PhasedXZ only) with calibration-aware placement, ALAP scheduling and
dynamical decoupling. Entry points: create_dicke_state(...), create_dicke_state_adaptive(...).
"""
"Core machinery for the DuQuantum Dicke-state submission (cirq / cirq_google / numpy / stdlib only).\n\nA *template* is an abstract circuit on nodes 0..k-1:\n    ops: list of ('u', node, 2x2 unitary) | ('cz', i, j)\n    state_nodes: nodes that carry the Dicke state (others are ancillas)\nPipeline: template -> noise-aware embedding onto the hardware graph -> layered schedule\n(1q / CZ moments alternate, ALAP so qubits stay in |0> as long as possible) -> dynamical decoupling\non idle superposed qubits -> PhasedXZ conversion.\n"
import math as _peer_math
import itertools as _peer_itertools
import numpy as _peer_np
import cirq as _peer_cirq
_peer__I2 = _peer_np.eye(2, dtype=complex)
_peer__X = _peer_np.array([[0, 1], [1, 0]], dtype=complex)
_peer__Y = _peer_np.array([[0, -1j], [1j, 0]], dtype=complex)

def _peer_ry(t):
    c, s = (_peer_math.cos(t / 2), _peer_math.sin(t / 2))
    return _peer_np.array([[c, -s], [s, c]], dtype=complex)

def _peer_rz(t):
    return _peer_np.array([[_peer_np.exp(-0.5j * t), 0], [0, _peer_np.exp(0.5j * t)]], dtype=complex)

def _peer_is_identity_up_to_phase(U, atol=1e-09):
    ph = U[0, 0] if abs(U[0, 0]) > 0.5 else 1.0
    return abs(abs(U[0, 0]) - 1) < atol and _peer_np.allclose(U / ph * abs(ph), _peer__I2, atol=atol) and _peer_np.allclose(U, U[0, 0] * _peer__I2, atol=atol)

def _peer_hw_graph(cz_err, sq_err):
    """adjacency dict: q -> {nbr: cz_error}; only qubits with single-qubit gates allowed."""
    adj = {q: {} for q in sq_err}
    for pair, val in cz_err.items():
        a, b = tuple(pair)
        p = float(val[0])
        if a in adj and b in adj:
            adj[a][b] = p
            adj[b][a] = p
    return adj

def _peer_readout_cost(q, n_frac, zero_err, one_err, x_weight=0.5):
    e0 = float(zero_err.get(q, (0.05, 0))[0])
    e1 = float(one_err.get(q, (0.05, 0))[0])
    z = (1 - n_frac) * e0 + n_frac * e1
    x = 0.5 * (e0 + e1)
    return z + x_weight * x

def _peer_template_edges(t):
    cnt = {}
    for op in t['ops']:
        if op[0] == 'cz':
            k = (min(op[1], op[2]), max(op[1], op[2]))
            cnt[k] = cnt.get(k, 0) + 1
    return cnt

def _peer_node_weights(t):
    """number of 1q gates per node (for sq error weighting) and lifetime proxy."""
    k = t['k']
    n1 = [0] * k
    for op in t['ops']:
        if op[0] == 'u':
            n1[op[1]] += 1
    return n1

def _peer_embed(t, adj, node_cost, edge_weight=1.0, beam=3000, exhaustive_limit=200000, return_all=False):
    """Find an injective map template-node -> hardware qubit s.t. every template edge is a hardware edge,
    minimising sum(edge_weight * count * p_cz) + sum(node_cost[node](q)).
    node_cost: function(node, q) -> float.
    Beam search over nodes in BFS order of the template graph."""
    k = t['k']
    edges = _peer_template_edges(t)
    nbrs = {i: set() for i in range(k)}
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    order = [max(range(k), key=lambda i: len(nbrs[i]))]
    while len(order) < k:
        cand = [i for i in range(k) if i not in order]
        best = max(cand, key=lambda i: (len(nbrs[i] & set(order)), len(nbrs[i])))
        order.append(best)
    qubits = sorted(adj)
    states = []
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
                    c += edge_weight * edges[min(node, m), max(node, m)] * adj[q][asg[m]]
                d = dict(asg)
                d[node] = q
                new.append((c, d))
        if not new:
            return None
        new.sort(key=lambda x: x[0])
        states = new[:beam]
    return states[0] if not return_all else states

def _peer_build_circuit(t, mapping, dd=True, align='alap', eject_z=True):
    """Turn a template + mapping into a layered cirq.Circuit of PhasedXZ and CZ moments."""
    k = t['k']
    q = [mapping[i] for i in range(k)]
    ops = t['ops']
    pend = [_peer__I2.copy() for _ in range(k)]
    has_pend = [False] * k
    node_layer = [0] * k
    cz_layers = []
    oneq = {}
    for op in ops:
        if op[0] == 'u':
            pend[op[1]] = op[2] @ pend[op[1]]
            has_pend[op[1]] = True
        else:
            i, j = (op[1], op[2])
            L = max(node_layer[i], node_layer[j])
            for a in (i, j):
                if has_pend[a]:
                    oneq[L, a] = pend[a] @ oneq.get((L, a), _peer__I2)
                    pend[a] = _peer__I2.copy()
                    has_pend[a] = False
            while len(cz_layers) <= L:
                cz_layers.append([])
            cz_layers[L].append((i, j))
            node_layer[i] = node_layer[j] = L + 1
    nL = len(cz_layers)
    for a in range(k):
        if has_pend[a]:
            oneq[nL, a] = pend[a] @ oneq.get((nL, a), _peer__I2)
    oneq = {key: U for key, U in oneq.items() if not _peer_is_identity_up_to_phase(U)}
    if align == 'alap':
        cz_layers, oneq = _peer__alap(k, cz_layers, oneq)
    if eject_z:
        oneq = _peer__eject_z(k, nL, oneq)
    if dd:
        oneq = _peer__add_dd(k, cz_layers, oneq)
    moments = []
    for L in range(nL + 1):
        m1 = [_peer_cirq.PhasedXZGate.from_matrix(oneq[L, a]).on(q[a]) for a in range(k) if (L, a) in oneq]
        if m1:
            moments.append(_peer_cirq.Moment(m1))
        if L < nL and cz_layers[L]:
            moments.append(_peer_cirq.Moment([_peer_cirq.CZ(q[i], q[j]) for i, j in cz_layers[L]]))
    return _peer_cirq.Circuit(moments)

def _peer__split_z(U):
    """U = Rz-type diag D @ W with W = Z^a X^x Z^-a (no trailing Z). Returns (W, D)."""
    g = _peer_cirq.PhasedXZGate.from_matrix(U)
    W = _peer_cirq.unitary(_peer_cirq.PhasedXZGate(x_exponent=g.x_exponent, z_exponent=0, axis_phase_exponent=g.axis_phase_exponent))
    D = _peer_cirq.unitary(_peer_cirq.ZPowGate(exponent=g.z_exponent))
    return (W, D)

def _peer__eject_z(k, nL, oneq):
    """Push the Z part of every 1q gate forward through CZs (they commute) into the node's next gate.
    The leftover Z after the node's last gate stays in the final 1q layer (it matters for X-basis readout)."""
    out = {}
    for a in range(k):
        slots = sorted((L for L, b in oneq if b == a))
        carry = _peer__I2
        for L in slots:
            U = oneq[L, a] @ carry
            W, D = _peer__split_z(U)
            carry = D
            if not _peer_is_identity_up_to_phase(W):
                out[L, a] = W
        if not _peer_is_identity_up_to_phase(carry):
            if (nL, a) in out:
                out[nL, a] = carry @ out[nL, a]
            else:
                out[nL, a] = carry
    return out

def _peer__alap(k, cz_layers, oneq):
    """Push CZ layers as late as possible (reverse ASAP), keeping per-node order."""
    nL = len(cz_layers)
    next_free = [nL] * k
    new_layers = [[] for _ in range(nL)]
    remap = {}
    for L in range(nL - 1, -1, -1):
        for i, j in cz_layers[L]:
            nl = min(next_free[i], next_free[j]) - 1
            new_layers[nl].append((i, j))
            remap[L, i, j] = nl
            next_free[i] = next_free[j] = nl
    node_cz = {a: [] for a in range(k)}
    for L in range(nL):
        for i, j in cz_layers[L]:
            node_cz[i].append((L, remap[L, i, j]))
            node_cz[j].append((L, remap[L, i, j]))
    newq = {}
    for (L, a), U in oneq.items():
        nxt = [nl for ol, nl in node_cz[a] if ol >= L]
        nl = nxt[0] if nxt else nL
        newq[nl, a] = U @ newq.get((nl, a), _peer__I2) if (nl, a) in newq else U
    return (new_layers, newq)

def _peer__add_dd(k, cz_layers, oneq):
    """Insert X-X echo pairs on idle windows of qubits that are already 'live' (had a gate).
    Time slots: 1q-layer L sits before CZ layer L. A node idle over CZ layers [L1, L2) with L2-L1 >= 2
    gets X at 1q-layer L1+? ... we use: X in the first free 1q slot after going idle and X in the last
    free 1q slot before the next gate (merged into that gate if one exists)."""
    nL = len(cz_layers)
    busy = [set() for _ in range(k)]
    for L in range(nL):
        for i, j in cz_layers[L]:
            busy[i].add(L)
            busy[j].add(L)
    out = dict(oneq)
    for a in range(k):
        events = sorted(set([L for L in busy[a]] + [L for L, b in oneq if b == a]))
        if not events:
            continue
        start = min(events)
        L = start
        while L < nL:
            if L in busy[a]:
                L += 1
                continue
            s = L
            while L < nL and L not in busy[a]:
                L += 1
            e = L
            d = e - s
            if d >= 2:
                aa = int(round(d / 4))
                bb = aa + int(round(d / 2))
                x1, x2 = (s + aa, s + bb)
                out[x1, a] = _peer__X @ out[x1, a] if (x1, a) in out else _peer__X.copy()
                out[x2, a] = out[x2, a] @ _peer__X if (x2, a) in out else _peer__X.copy()
    return {key: U for key, U in out.items() if not _peer_is_identity_up_to_phase(U)}

def _peer_check_allowed(circuit, cz_err, sq_err):
    allowed = set((frozenset(p) for p in cz_err))
    for op in circuit.all_operations():
        if isinstance(op.gate, _peer_cirq.CZPowGate):
            assert op.gate.exponent == 1 and frozenset(op.qubits) in allowed, op
        elif isinstance(op.gate, _peer_cirq.PhasedXZGate):
            assert op.qubits[0] in sq_err, op
        elif isinstance(op.gate, _peer_cirq.ResetChannel):
            pass
        else:
            raise AssertionError(f'bad op {op}')
_peer__H = _peer_np.array([[1, 1], [1, -1]], dtype=complex) / _peer_math.sqrt(2)

def _peer__cnot(ops, a, b):
    ops += [('u', b, _peer__H), ('cz', a, b), ('u', b, _peer__H)]

def _peer__gadget2(ops, a, b, th):
    """(a,b): 00->00, 01->cos 01 + sin 10, 11->11 (input 10 unconstrained)."""
    _peer__cnot(ops, a, b)
    ops += [('u', a, _peer_ry(-th)), ('cz', b, a), ('u', a, _peer_ry(th))]
    _peer__cnot(ops, a, b)

def _peer__gadget3(ops, c, a, b, th):
    """path c-a-b. c=0: (a,b) 00->00, 01->cos 01+sin 10, 11->11;  c=1: 01->10, 11->11."""
    bp = (th + _peer_math.pi / 2) / 2
    ap = (th - _peer_math.pi / 2) / 2
    _peer__cnot(ops, a, b)
    ops += [('u', a, _peer_ry(-bp)), ('cz', c, a), ('u', a, _peer_ry(-ap)), ('cz', b, a), ('u', a, _peer_ry(ap)), ('cz', c, a), ('u', a, _peer_ry(bp))]
    _peer__cnot(ops, a, b)

def _peer__gadget3b(ops, c, a, b, th, skip_first_cnot=False):
    """path c-a-b, valid when input 111 is absent: c=0: 00->00, 01->cos 01+sin 10, 11->11; c=1: 01->10.
    4 CZ (3 if a is always 0 on the support, i.e. 011 also absent)."""
    if not skip_first_cnot:
        _peer__cnot(ops, a, b)
    g1 = th
    g2 = th + _peer_math.pi / 2
    ops += [('u', a, _peer_ry(-g1)), ('cz', b, a), ('u', a, _peer_ry(g1)), ('u', a, _peer_ry(-g2)), ('cz', c, a), ('u', a, _peer_ry(g2))]
    _peer__cnot(ops, a, b)

def _peer_be_path_template(N, n):
    """D(N,n) on path nodes 0..N-1 (node p-1 = B&E position p). Exact, ~5 CZ per gadget."""
    ops = []
    for p in range(N - n, N):
        ops.append(('u', p, _peer__X.copy()))
    for m in range(N, 1, -1):
        k = min(n, m - 1)
        w = [m - k - 1 + i for i in range(k + 1)]
        for i in range(k):
            th = _peer_math.acos(_peer_math.sqrt((k - i) / m))
            if i == 0:
                _peer__gadget2(ops, w[0], w[1], th)
            else:
                _peer__gadget3(ops, w[i - 1], w[i], w[i + 1], th)
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=n)

def _peer_template_state(t):
    """noiseless statevector of a template (node 0 = MSB)."""
    k = t['k']
    psi = _peer_np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1
    for op in t['ops']:
        if op[0] == 'u':
            psi = _peer_np.moveaxis(_peer_np.tensordot(op[2], psi, axes=([1], [op[1]])), 0, op[1])
        else:
            sl = [slice(None)] * k
            sl[op[1]] = 1
            sl[op[2]] = 1
            psi[tuple(sl)] *= -1
    return psi.reshape(-1)

def _peer__givens_ops(a, b, th):
    """exact 2-CZ real Givens on (a,b): 01 -> cos 01 + sin 10, 10 -> -sin 01 + cos 10 (node a = first bit)."""
    c, s = (_peer_math.cos(th), _peer_math.sin(th))
    G = _peer_np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=complex)
    qa, qb = (_peer_cirq.LineQubit(0), _peer_cirq.LineQubit(1))
    cops = _peer_cirq.two_qubit_matrix_to_cz_operations(qa, qb, G, allow_partial_czs=False)
    ops = []
    for op in cops:
        idx = [0 if q == qa else 1 for q in op.qubits]
        if len(idx) == 2:
            ops.append(('cz', a, b))
        else:
            ops.append(('u', a if idx[0] == 0 else b, _peer_cirq.unitary(op)))
    return ops

def _peer__classical_synth(support, edges3):
    """support: dict input_bits(tuple)->output_bits(tuple) on 3 local bits (c,a,b). Find min-CNOT circuit
    (CNOTs along edges3 in either direction, X free). Returns list of ('x',q) / ('cnot',ctl,tgt)."""
    from collections import deque as _peer_deque
    ins = list(support)
    target = tuple((support[i] for i in ins))
    start = tuple(ins)
    nb = len(ins[0])
    moves = []
    for u, v in edges3:
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
        diff = None
        for s, t in zip(state, target):
            d = tuple((x ^ y for x, y in zip(s, t)))
            if diff is None:
                diff = d
            elif d != diff:
                return None
        return diff
    seen = {start: []}
    dq = _peer_deque([start])
    while dq:
        st = dq.popleft()
        path = seen[st]
        d = xfix(st)
        if d is not None:
            return path + [('x', q) for q in range(nb) if d[q]]
        if len(path) >= 6:
            continue
        for xs in _peer_itertools.product([0, 1], repeat=nb):
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

def _peer_be_path_template_opt(N, n):
    """Support-aware B&E nearest-neighbour construction on a path 0..N-1."""
    ops = []
    for p in range(N - n, N):
        ops.append(('u', p, _peer__X.copy()))
    for m in range(N, 1, -1):
        k = min(n, m - 1)
        w = [m - k - 1 + i for i in range(k + 1)]
        J = [j for j in range(max(0, n - (N - m)), min(n, m) + 1)]
        for i in range(k):
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
            th = _peer_math.acos(_peer_math.sqrt((k - i) / m))
            if i == 0:
                if (0, 1) not in S:
                    continue
                if S == {(0, 1)}:
                    ops += [('u', w[0], _peer_ry(2 * th))]
                    _peer__cnot(ops, w[0], w[1])
                else:
                    ops += _peer__givens_ops(w[0], w[1], th)
                continue
            P, Q = ((0, 0, 1) in S, (1, 0, 1) in S)
            c, a, b = (w[i - 1], w[i], w[i + 1])
            if P and Q:
                if (1, 1, 1) not in S:
                    _peer__gadget3b(ops, c, a, b, th, skip_first_cnot=(0, 1, 1) not in S)
                else:
                    _peer__gadget3(ops, c, a, b, th)
            elif P:
                if S <= {(0, 0, 1)}:
                    ops += [('u', a, _peer_ry(2 * th))]
                    _peer__cnot(ops, a, b)
                else:
                    ops += _peer__givens_ops(a, b, th)
            elif Q:
                sup = {}
                for s in S:
                    sup[s] = (1, 1, 0) if s == (1, 0, 1) else s
                circ = _peer__classical_synth(sup, [(0, 1), (1, 2)])
                loc = [c, a, b]
                for g in circ:
                    if g[0] == 'x':
                        ops.append(('u', loc[g[1]], _peer__X.copy()))
                    else:
                        _peer__cnot(ops, loc[g[1]], loc[g[2]])
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=n)

def _peer_grow_patch(adj, N, root, qcost, lam=0.002, forbid=()):
    """Greedy best-first patch of N qubits from root. qcost(q)->float. Compactness via BFS distance."""
    from collections import deque as _peer_deque
    dist = {root: 0}
    dq = _peer_deque([root])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    patch = [root]
    inP = {root}
    while len(patch) < N:
        best, bc = (None, 1000000000.0)
        for u in patch:
            for v, p in adj[u].items():
                if v in inP or v in forbid:
                    continue
                c = qcost(v) + p + lam * dist.get(v, 99) ** 2
                if c < bc:
                    best, bc = (v, c)
        if best is None:
            return None
        patch.append(best)
        inP.add(best)
    return patch

def _peer_broadcast_tree(adj, patch, root, prefer_good=True):
    """Rounds of parent->child splits inside patch (each node in <=1 split per round).
    Returns list of rounds, each a list of (parent, child)."""
    P = set(patch)
    informed = {root}
    rounds = []
    while len(informed) < len(P):
        busy = set()
        rnd = []
        cands = []
        for u in informed:
            for v, p in adj[u].items():
                if v in P and v not in informed:
                    fan = sum((1 for w in adj[v] if w in P and w not in informed))
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

def _peer_w_tree_template(patch, rounds):
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
            size[q] = 1 + sum((sz(c) for c in children[q]))
        return size[q]
    root = rounds[0][0][0]
    sz(root)
    remaining = {q: size[q] for q in patch}
    ops = []
    first = True
    for rnd in rounds:
        for u, v in rnd:
            R = remaining[u]
            s = size[v]
            th = _peer_math.asin(_peer_math.sqrt(s / R))
            a, b = (idx[u], idx[v])
            if first:
                ops.append(('u', a, _peer_ry(2 * th)))
                _peer__cnot(ops, a, b)
                ops.append(('u', a, _peer__X.copy()))
                first = False
            else:
                eta = (_peer_math.pi - 2 * th) / 2
                ops += [('u', b, _peer_ry(eta))]
                _peer__cnot(ops, a, b)
                ops += [('u', b, _peer_ry(-eta))]
                _peer__cnot(ops, b, a)
            remaining[u] -= s
    if N == 1:
        ops.append(('u', 0, _peer__X.copy()))
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=1)

def _peer_be_ops(nodes, k, jmin, jmax):
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
            th = _peer_math.acos(_peer_math.sqrt((kk - i) / m))
            if i == 0:
                if (0, 1) not in S:
                    continue
                if S == {(0, 1)}:
                    ops += [('u', w[0], _peer_ry(2 * th))]
                    _peer__cnot(ops, w[0], w[1])
                else:
                    ops += _peer__givens_ops(w[0], w[1], th)
                continue
            P, Q = ((0, 0, 1) in S, (1, 0, 1) in S)
            c, a, b = (w[i - 1], w[i], w[i + 1])
            if P and Q:
                if (1, 1, 1) not in S:
                    _peer__gadget3b(ops, c, a, b, th, skip_first_cnot=(0, 1, 1) not in S)
                else:
                    _peer__gadget3(ops, c, a, b, th)
            elif P:
                if S <= {(0, 0, 1)}:
                    ops += [('u', a, _peer_ry(2 * th))]
                    _peer__cnot(ops, a, b)
                else:
                    ops += _peer__givens_ops(a, b, th)
            elif Q:
                sup = {s: (1, 1, 0) if s == (1, 0, 1) else s for s in S}
                for g in _peer__classical_synth(sup, [(0, 1), (1, 2)]):
                    loc = [c, a, b]
                    if g[0] == 'x':
                        ops.append(('u', loc[g[1]], _peer__X.copy()))
                    else:
                        _peer__cnot(ops, loc[g[1]], loc[g[2]])
    return ops

def _peer_dc2_template(N, n, a=None):
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
    p = [_peer_math.comb(a, j) * _peer_math.comb(b, n - j) / _peer_math.comb(N, n) for j in range(n + 1)]
    jmin = min((j for j in range(n + 1) if p[j] > 0))
    jmax = max((j for j in range(n + 1) if p[j] > 0))
    T = [sum(p[r:]) for r in range(n + 2)]
    ops = []
    for r in range(1, jmax + 1):
        q = A[a - r]
        if r <= jmin:
            ops.append(('u', q, _peer__X.copy()))
            continue
        cond = T[r] / T[r - 1]
        phi = 2 * _peer_math.asin(_peer_math.sqrt(cond))
        if r == 1 or (r - 1 <= jmin and False):
            ops.append(('u', q, _peer_ry(phi)))
        elif r - 1 <= jmin:
            ops.append(('u', q, _peer_ry(phi)))
        else:
            eta = (_peer_math.pi - phi) / 2
            ops.append(('u', q, _peer_ry(eta)))
            _peer__cnot(ops, A[a - r + 1], q)
            ops.append(('u', q, _peer_ry(-eta)))
    for r in range(1, n + 1):
        bq = B[b - n + r - 1]
        if r <= jmin:
            continue
        ops.append(('u', bq, _peer__X.copy()))
        if r <= jmax:
            _peer__cnot(ops, A[a - r], bq)
    ops += _peer_be_ops(A, n, jmin, jmax)
    ops += _peer_be_ops(B, n, n - jmax, n - jmin)
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=n)
import time as _peer__time
_peer_T1_EST = 6.8e-05
_peer_TPHI_EST = 0.00012
_peer_LIBRARY = {(2, 1): [{'edges': [[0, 1]], 'seq': [0], 'params': [[[-1.5707963267948966], [-1.5707963267948966]], [[3.5839946445546094], [1.1283943358300803]]], 'mode': 'real', 'shape': 'path', 'L': 1}], (3, 1): [{'edges': [[0, 1], [1, 2]], 'seq': [0, 1, 0], 'params': [[[0.9647388984647173], [2.1655128646107205], [0.6095637510202409]], [[-0.32047720390272355], [-0.41556492137335116], [0.8704998254568896]], [[-0.10683041024328749], [1.7537808879518964], [0.6631725360545844]], [[0.18612742405578603], [-1.5707963267948968], [0.632334933556112]]], 'mode': 'real', 'shape': 'path', 'L': 3}], (4, 1): [{'edges': [[0, 1], [1, 2], [2, 3]], 'seq': [1, 0, 2, 0, 2], 'params': [[[2.7199545744311804], [-1.5707963267942358], [1.5707963267781015], [-1.1907928993461503]], [[1.8576165046566926], [-0.5210452755849042], [-3.0207211112265564], [-0.5041498718556253]], [[-1.0250439686294914], [2.2988535061538964], [-0.02975173938509952], [2.5391537710208882]], [[3.3812384587798032], [0.04814126609378741], [1.4246812278790024], [1.6629335683711233]], [[-3.2999156009234194], [2.3467062203906828], [-0.1839258346054574], [-0.877535404965622]], [[-1.6037213453905859], [2.3656827610826054], [1.5707963270395944], [-0.34965170202795304]]], 'mode': 'real', 'shape': 'path', 'L': 5}, {'edges': [[0, 1], [1, 2], [2, 3], [3, 0]], 'seq': [1, 0, 3, 2, 3], 'params': [[[-1.770600728962932], [2.37002849093598], [-0.7996258678769054], [-2.4673380108430387]], [[3.341397055822803], [-1.5905004115214552], [-1.248466153350983], [2.0607372825589705]], [[-3.14159265245787], [0.6610473992455239], [3.014545343977525], [1.977397055837326]], [[-2.115003230647901], [-1.9510269274471046], [-0.21526532556206346], [-1.5707963267670781]], [[-0.24119125952958684], [-0.5875999290241708], [2.7561724060545654], [-0.7853981634200258]], [[1.5707963294391918], [1.8775794696583936], [0.3854202503241914], [-1.570796325860376]]], 'mode': 'real', 'shape': 'sq', 'L': 5}], (4, 2): [{'edges': [[0, 1], [1, 2], [2, 3]], 'seq': [0, 1, 0, 2, 1, 0], 'params': [[[-2.3416837065745333], [-3.8612230600097015], [2.472269481272324], [0.7296811018585674]], [[1.5071211760433751], [0.8067586968681182], [-0.31829913599443815], [-0.14852627625340528]], [[-2.5223860049334053], [1.2739881882763675], [1.185837568978237], [-0.7471048632643251]], [[-3.159126768330054], [1.2380569855554084], [-3.0352294251605394], [1.7367463644540593]], [[0.36630442809658187], [2.3819489239444622], [-1.1583458400282287], [2.195559439898938]], [[2.633541133431555], [-1.5707963267948966], [0.8867452919364118], [0.4102757769411197]], [[3.453045107678416], [1.8822487808835195], [0.5244551278575348], [-1.6070852838127778]]], 'mode': 'real', 'shape': 'path', 'L': 6}, {'edges': [[1, 2], [2, 3], [0, 1]], 'seq': [1, 0, 2, 0, 0, 2], 'params': [[[-0.6039121422327106, 0.19970756966794445, -1.0233578197645476], [-1.256474366336012, 1.7582868853858806, 0.6889972074290514], [-3.490453674375172, 1.5707963269656442, 4.245637872466751], [1.078265462273148, 4.712388981336991, -2.158463830263483]], [[0.49432835102588724, 0.8960052231295885, 0.018877200901551212], [0.6208478809723648, -0.634055898005935, 0.7736593637878127], [4.309880069577817, 1.7766042768441401, 0.4445720950206556], [4.8172574183436065, 1.279716178250803, 1.1124023071937637]], [[1.243674941761388, 3.1347799740183726, -2.0593875103775825], [-1.3563411002607788, 1.3197393605552585, 1.6592757499519093], [3.9860093565332386, 1.1404603900872041, 2.994909830447573], [-0.7568716826100611, 3.105571455616389, -2.998212599691704]], [[-1.9966759049045804, 2.391842298091698, 2.4161568742472213], [1.120440418611106, 2.266047566874372, 0.14124630335870256], [5.280799816269282, 0.7703639399399251, 0.7283963122731854], [-0.043215791865314226, 0.17836438417617234, 1.1077302889869438]], [[1.411658218018404, 0.3327764563482991, -0.4223245391296947], [-3.422425495286281, 1.5707962502437534, 3.1604699230918376], [-0.7000155309592011, -1.5707963475057787, 0.6187684140343777], [0.8125718896547025, 1.2685919822856584, 0.18328183441941376]], [[1.0204490529787955, 0.9567560216845588, 0.6567130874272844], [1.5519190477555782, 0.5925287929291619, 1.4220482002146846], [3.0232391670244896, 1.6513982764464916, 0.5201202275604562], [-1.1014953437103794, -0.7269891648297138, -0.29675222511075816]], [[-2.5223646924077228, 1.570796274364101, 4.419311109424183], [-2.9928444644685213, 0.6443731957610619, 2.8485147334237366], [1.3119179741886373, 1.09985971652163, 1.0843499768651597], [0.2647567827704094, -0.11423046223982877, 0.9519340156775488]]], 'mode': 'complex', 'shape': 'path', 'L': 6}, {'edges': [[0, 1], [1, 2], [2, 3], [3, 0]], 'seq': [2, 3, 1, 0, 1], 'params': [[[-1.0715377340354926], [-1.2828464643572608], [1.6773254855236024], [1.466666445639058]], [[2.6423340608303882], [-1.8661451259843416], [3.148376786053271], [2.3648261207170034]], [[2.9773950008000805], [2.2540855807311213], [0.764425707266486], [0.7204934375943249]], [[-2.449935350851258], [0.8949060096104815], [-0.4435217910991425], [3.637990115492336]], [[0.2608088753003052], [-1.5707963267948966], [-1.1272745356957536], [3.6602944522199765]], [[-1.0221680499372667], [3.5862811780452515], [2.015484851250354], [-0.3986958960202929]]], 'mode': 'real', 'shape': 'sq', 'L': 5}, {'edges': [[0, 1], [1, 2], [2, 3], [3, 0]], 'seq': [2, 0, 1, 3, 2], 'params': [[[-1.8392262037038425], [2.0253180676598808], [0.8952846349999232], [-1.2455708246958712]], [[0.2684298769089459], [-0.4545217408649841], [-0.09810913742239129], [-0.265649669725784]], [[-2.1600918780266603], [-2.165424532631926], [0.3309796436649954], [-0.631282196660673]], [[0.44215597120112066], [-0.4845840896534056], [2.3372547163996993], [0.07330767049878766]], [[1.5583243722304456], [2.0991436200847584], [2.3751342639849904], [1.5707963267948966]], [[0.19147347966103503], [-0.9880693795283817], [2.794376985851346], [-5.059604648123138]]], 'mode': 'real', 'shape': 'sq', 'L': 5}], (5, 1): [{'edges': [[0, 1], [0, 2], [0, 3], [0, 4]], 'seq': [3, 1, 2, 1, 0, 2, 0], 'params': [[[0.9273109600913177], [1.3328201343530859], [1.5555274533938928], [-1.1451171341909177], [-1.5756556121771155]], [[1.824151273555135], [2.203275125754279], [-2.7259184372009457], [-1.4738778971438047], [-1.1598421345504548]], [[-0.5739404349404124], [1.2717781612578916], [-2.9211313990312333], [-2.0933939490499633], [-0.08917309011651643]], [[3.1415926535897936], [-0.5615361332163207], [-2.838455132550054], [-2.939814740280138], [2.367746983028962]], [[0.6154797086703868], [-2.6755409613540406], [0.1465204703469559], [0.3932059614212497], [-2.309222529008847]], [[9.042635941160197e-17], [-2.3925379880530326], [0.6388915135275444], [-3.121096819650311], [-2.042499698689464]], [[-2.3561944901923444], [-1.5344528289342092], [-1.6587855091961992], [-0.4442655364115883], [-0.06688936862399468]], [[-1.5707963267948963], [4.712388980384689], [-0.23055372848571773], [2.0150618632064887], [-1.4044102767903701]]], 'mode': 'real', 'shape': 'plus', 'L': 7}], (5, 2): [{'edges': [[0, 1], [0, 3], [1, 2], [1, 4], [3, 4]], 'seq': [1, 4, 0, 3, 2, 4, 1, 3, 4, 1], 'params': [[[3.8623485327866955], [2.631635228557763], [1.2951151030569221], [-1.7140013539373784], [0.9489070091398105]], [[1.248871998787486], [-0.6717638728129834], [-0.9464069105244818], [-3.0577664280901113], [-3.1898793442016875]], [[3.8111780670204793], [4.134769548679072], [3.3021127831390125], [0.32131750623567124], [-0.036041692728804677]], [[-4.256058203640799], [1.7870120735379385], [2.352500546917346], [-3.224605172576948], [-0.5663210194281699]], [[-2.241452963765802], [0.06983883732576081], [-1.2909325422040905], [-2.7025165828265956], [0.7011585928371439]], [[-0.22007984893675764], [-0.17212901586690355], [1.5882038712721065], [-1.9305151234432067], [-0.14533179054634188]], [[-0.6555742323404297], [4.3825378372431665], [-1.851741567560954], [-0.6154797086703865], [1.4548406002117762]], [[-4.462946120205325], [0.5019801590084255], [2.289032381413742], [-0.9261719554944098], [-4.132785644800764]], [[1.249293044591401], [3.4424742810220286], [0.0486353486798898], [-0.23701006121047596], [-1.5707963267948966]], [[-2.284134068168216], [0.6789560905898807], [-0.8150825536037867], [1.5707963267948963], [0.9632092654431255]], [[-4.0255396620159005e-16], [-2.5506340448170164], [3.4533415001836865], [1.570796326794896], [0.6075870613517693]]], 'mode': 'real', 'shape': 'g2x3', 'L': 10}, {'edges': [[0, 1], [1, 2], [2, 3], [3, 4]], 'seq': [2, 1, 2, 3, 2, 3, 0, 1, 2, 3], 'params': [[[2.624791018388875], [2.3422677928158224], [1.826369904940376], [0.6678275883300873], [0.2693967957420651]], [[3.3540835096953687], [-3.2905679169064017], [2.4341146637108686], [1.5520025505258572], [-3.3478712591326953]], [[4.160604311214936], [3.7740073463582324], [-3.9334205552738073], [3.1325424016621244], [-0.5004185549639748]], [[-0.15770421966221002], [1.3831856638400888], [-0.6529468725215047], [3.672552996690928], [-0.17817934896004245]], [[-1.7958361900215172], [2.2013597145720785], [0.12934808228622138], [2.1862760738459843], [3.7751541962227755]], [[-3.0446573739493186], [4.164422838776636], [6.50331258526692], [6.898665007443773], [-1.5888781585746214]], [[2.712700574777355], [-1.2469746603064618], [1.0972852831236284], [-3.2358159928612147], [0.7443502054342696]], [[0.98686110820448], [1.5707963255770836], [1.8241801081615323], [3.257583403322666], [0.977539588505094]], [[-4.72371797901878], [3.2079757349273774], [1.570796325927833], [-2.377961931734971], [3.2436499867568194]], [[-0.541154086910362], [2.905934326491898], [0.7142291673316673], [1.5707963290060203], [0.5322473354906517]], [[-0.4343780242445449], [1.740071571550981], [-2.2850254947263777], [-1.5707963308597501], [6.710246426848965e-09]]], 'mode': 'real', 'shape': 'path', 'L': 10}, {'edges': [[0, 1], [0, 2], [0, 3], [0, 4]], 'seq': [3, 0, 2, 0, 3, 0, 1, 2, 0, 3], 'params': [[[-1.7348846171868988], [1.8566941603013687], [-0.7192512651277753], [-0.9478850225221391], [1.2154165503195553]], [[3.575256652904579], [-0.07442805951050258], [2.7403556116320273], [-2.8905446572187254], [-0.3414614122972694]], [[-1.8831842640023553], [-2.9050562733487744], [-0.6314759975951348], [-0.192910011253669], [1.9695208842991967]], [[0.5286143822890906], [2.6442734829865553], [2.2432342292523777], [-0.8980450547647505], [1.68629326722746]], [[-1.3248403830705084], [-1.0399271400964583], [-1.093246666582409], [1.1746919878023308], [-2.0676119303553526]], [[2.6637092315526743], [-0.9764586065571927], [-2.4163987003104093], [-2.0710746544624565], [1.5124189899577178]], [[-1.3262642068983306], [-2.4525795210398837], [-1.6940135380635737], [2.754571586042372], [1.8473470660552997]], [[-1.570796326794897], [-3.0222825634016215], [-3.3120863663630447], [0.19398285680635008], [0.8342016405860311]], [[-3.141592653589794], [-2.81149682216691], [-1.8991466862965312], [-2.9056699358450633], [-0.40471176978083734]], [[-3.141592653589793], [2.1614514493860484], [-0.9804202223642033], [-0.8432368879302086], [-3.694336875843787]], [[1.5707963267948974], [0.9801412042037451], [1.47926429463909], [0.6073141701854794], [-2.0172661996909533e-16]]], 'mode': 'real', 'shape': 'plus', 'L': 10}, {'edges': [[0, 1], [1, 2], [2, 3], [3, 0], [3, 4]], 'seq': [2, 1, 4, 0, 3, 4, 1, 3, 4], 'params': [[[2.7228654999727686], [-1.4117794905168422], [1.4055832995926547], [-2.3628507692880176], [-3.532771005226294]], [[0.12143934882192703], [0.3645819393202445], [1.928705596308489], [-0.044973466594716544], [0.7172249063076739]], [[-0.3327660448714142], [2.374696192990228], [-0.8688193139924637], [-1.967742616707106], [0.2894331539992146]], [[-0.9407424771283848], [2.8612915629927613], [-2.4562683389356583], [-3.7348524149180276], [-2.7671162909950437]], [[1.5707963267948966], [1.2946610663637403], [-2.2787103912827296], [-1.59301627413705], [2.770511867332006]], [[-1.5762044919294893], [2.7905431550135544], [2.017562261577865], [-0.6154797086703868], [2.182880459128321]], [[0.24797902948454814], [3.7687774125971885], [2.015439455838089], [-0.831246630871125], [-0.7975473425102977]], [[-0.24257086434995484], [1.6034181492400663], [-0.3179198391221718], [3.18744112106347], [-1.8854045648083435]], [[-1.2033243663896092], [-1.6513774989676775], [3.504174230913894], [-1.570796326794897], [-1.2440389096686015]], [[-0.3674719604052878], [-1.5228369770672865], [-3.186254391791722], [-1.570796326794897], [3.1415926535897936]]], 'mode': 'real', 'shape': 'sq', 'L': 9}], (6, 1): [{'edges': [[0, 1], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [4, 5]], 'seq': [1, 3, 6, 0, 5, 4, 0, 3, 1, 2, 4], 'params': [[[1.1204871953046043], [2.7300047488037724], [-2.718237665143859], [0.833789559095197], [2.719247490006256], [-0.3554121462933529]], [[-2.605865519105164], [-1.159208422008876], [3.1535846691293075], [-1.9556653400351958], [-1.1484511632113599], [-3.093966183989911]], [[-2.67846861244667], [0.08787755905775241], [-1.9100719331086229], [0.8459167937845965], [-0.39960863092028376], [1.8785820034883671]], [[0.8606031815152847], [-1.2555566825915316], [2.119961698952862], [1.2989748644567103], [-0.8327994002346935], [-1.4506234094087938]], [[1.4542720907859865], [0.12631552123731443], [-0.8035763786863864], [-1.7435263149881428], [1.3563981758329924], [-2.4153602533060234]], [[0.3706229422442685], [-3.7033984869813046], [-1.4124567182125864], [-0.24454593774513292], [-1.1456667987238551], [1.6797076272495335]], [[1.3166976205595375], [-2.7061023414355962], [-0.9647464509867855], [1.283524226198704], [-0.7026688780193053], [-2.2751598680173872]], [[-3.1427685769772356], [0.003508572340834102], [0.14514807397073098], [1.5714557189146496], [0.8011381255465629], [1.6997786312159517]], [[0.799740343223147], [-3.133379087570038], [1.135140291075734], [-1.0396376805733243], [3.7894496912318476], [0.01837449572697834]], [[-1.8585883232899705], [-0.9635301841442648], [-1.8863382408545761], [-1.3960214775376047], [-1.920007179347761], [-2.544007657671594]], [[-1.0775657862688044], [-4.070840480888348], [-2.35619448951069], [-2.4777327108063494], [2.125681861197963], [-2.3967727450164333]], [[-0.20543854403101838], [-3.783141153086136], [-1.5707963267948963], [-0.8386347920407352], [3.8588572608924343], [-1.5707963267948966]]], 'mode': 'real', 'shape': 'g2x3', 'L': 11}], (6, 2): [{'edges': [[1, 2], [2, 4], [1, 5], [0, 1], [3, 4], [4, 5]], 'seq': [0, 1, 2, 3, 0, 3, 0, 3, 3, 5, 4, 5, 4, 5, 4], 'params': [[[0.0005451334927713248, -0.12462507729533245, -0.5156839524527694], [3.142035164259196, 0.5156372125745011, -0.019555014994773072], [0.0005253671210067595, 2.1953707144113275, 0.5893535084722639], [0.0006160514465605877, 0.003580968710618697, -0.4470683754588211], [0.0001965923143311429, -0.02914467890619503, -0.434216515085961], [-0.0009455004355393361, -0.02601010149755247, 0.7533862435174229]], [[-0.5156839523412462, -0.1572330987801794, -0.4026445106444522], [3.1220376386210593, 0.44383963713984553, -0.18410946871310857], [0.5893535087582676, 0.060552666405879516, 0.40527619957939887], [-0.44706837548567047, 0.10911848816071233, -0.423976785014694], [-3.575809168084634, 1.5519643061055568, 3.16222565593044], [0.7533862427802998, -0.13590492705526225, 0.334540747881726]], [[-0.40264451085752234, -0.22274620076334845, -0.28736258033665696], [-0.18410946856059487, 0.1000520353717732, -0.13342819053626567], [0.4052761993887638, 0.5414292790632769, 0.10854271491264889], [-0.42397678495605434, 0.016145210238691813, -0.6733294895088533], [3.1622256559314725, 1.5930260236889187, -0.18047661608837673], [-2.8070519053178744, 1.4787204928420592, 3.3608603171812064]], [[1.2834337374725382, 0.9662271883494425, -1.6914508953201013], [-1.7042245859218177, 3.141592650811128, 0.9372895521972113], [0.10854269427744626, 0.5187697331888612, 0.00859011891578323], [-0.6733294145046228, -0.13727625475559257, -0.36295372633455025], [-0.1804765872875444, 0.06573317824617442, -0.1065312167553447], [3.360860312099602, 1.3656479726357011, -0.02443204285927634]], [[1.450141758193758, 0.4968922958223141, -1.733658413170196], [0.9372895521972855, 1.5707960383522919, -1.6098826402605417], [0.008590090369215478, 0.5220480189410125, -0.02635478930010379], [-0.3629537959281678, -0.20941591588970207, -0.27249456065710037], [-0.10653119475933649, 0.04905210129356834, -0.050395211421453395], [-0.024432078662543107, -0.2553006957963141, 0.002285974873133606]], [[-0.11177986157086418, -0.25940261888739713, -0.23015581331104334], [-2.395872289446389, 0.8280070437779489, 3.476593681350572], [3.1463112432737597, 1.7173784549894195, -0.09300094968735846], [-0.30735057373383035, -0.031938421116544324, -0.31089781484157614], [-0.03861271643131182, 0.08324611322980831, -0.010325828208736029], [0.0006750810687083952, -0.3018317974936027, 0.025850587775132057]], [[-0.23015581317418013, 1.3828819934219907, 0.4780940916634994], [0.3350010279083963, 1.5707963277787593, -0.7853981636841352], [-0.09300094970763433, 0.47855439287447926, 0.057327020711657385], [-0.310897814980112, 0.10300078452642862, -0.32641240355782325], [-0.0103258279099308, 0.027755654589610688, 0.02879861359549316], [0.025850587666846742, -0.32202964156243685, 0.05572146982444047]], [[0.4780940919389024, 1.2788328197978034, 0.9867798492939369], [-3.9269908173255357, -1.5707972119552494, 2.356194489784648], [0.05732702093631705, 0.08016382053917699, 0.0621230687110068], [-0.32641240367897423, 0.28309485236746296, -0.35881283292477134], [0.028798613487608392, -0.08341280146105821, 0.10303559237213702], [0.055721470048153345, -0.2913543996109804, 0.08695700988415907]], [[0.9867798491166779, -0.7853981658590107, 0.45803641226444214], [-0.7853981638645994, 0.9450137067155043, -0.2618349776391529], [3.2037157223167805, 1.456111051884416, 0.017127703555698922], [-0.3588128331840169, 0.42499212237827066, -0.3892639693412714], [0.10303559240192126, -0.12098093912749447, 0.1921865438355235], [0.08695700988540267, -0.20909278343888626, 0.11310969594011563]], [[-2.6835562412369334, 1.2386319581937864, 3.079717243610375], [-0.2618349779196463, -0.3310157428961077, 0.47387243193701073], [0.017127703422616877, -0.07705746146948146, 0.005830665977153897], [-0.38926396915799033, 0.5597974969519385, -0.2267687398467541], [0.1921865435830169, -0.03403980769074405, 0.22430648183079338], [0.11310969596209992, -0.08326853175159393, 0.1162033969914322]], [[0.16435832581847926, -0.7704086244568162, 0.14774639000655324], [0.31106947834034027, -0.6791204859585686, 0.26850347373900096], [0.011849871386920887, -0.03284857157145175, 0.001948997075943028], [0.07451689379815198, 0.15410682570159268, -0.004830420142682285], [-3.31743109615013, 1.5707963279188488, 3.9003830526816428], [3.591443241918777, 1.8560917148748657, -0.05311042292924008]], [[0.14774638987161937, -0.7390433308399705, 0.20988670677724325], [0.2685034736700703, -0.6161803478752393, 0.23318871909001077], [0.0019489969805016327, 0.037777299477638006, -0.009328016653811062], [-0.004830419964266176, -0.7455389574654342, -0.06082005572071817], [0.7587903990794047, 0.6154797073282942, 0.47765830818397415], [-0.053110422992386136, 0.43002921816962797, -0.05162789167990372]], [[0.20988670703015355, -0.5645542561671452, 0.2585465526315805], [0.23318871910690964, -0.6644226336053064, 0.18212369926428598], [-0.009328016540280005, 0.04962409846971941, -0.018408810650278804], [-0.060820055694522285, -0.8725441855945641, -0.1442315515797183], [-2.6639343452357727, 1.5707958505124842, 2.3561944055084822], [-0.051627891733973, 0.30371219849146625, 0.10210730077063468]], [[0.258546552587312, -0.3806291309078782, 0.2613706553718031], [0.1821236993690975, -0.7308591220080732, 0.09287864048834607], [-0.018408810532600954, 0.06237984607193152, -0.02575698198473215], [-0.14423155156617906, 0.03538507538720753, -0.19196224490687377], [-0.7853982483163661, 1.5707963326548156, -0.7853981648779989], [3.2436999544077216, 1.2517831571745377, -0.002192975822295138]], [[0.2613706556188825, -0.0030012315808845983, 0.3080617964797402], [0.09287864049539267, -0.7066337682053232, -0.15495766246282627], [-0.025756982195897178, 0.0757266098711351, -0.03184692529629183], [-0.19196224493021002, 1.537985300608433, -0.7624178145499598], [-0.7853981647051502, 3.926990820450957, 0.20334540297948828], [3.1393996781708844, 3.3359507605424707, 0.06308732622607116]], [[0.3323740254646504, 0.2717007448606675, 0.3733250669003039], [-0.1062439855789354, -0.6157978317624687, -0.14709414156236225], [-0.022106923000601894, 0.07928928567584183, -0.02871453450457683], [-3.936719631876191, 1.5707953140803703, 2.7916357759775314], [-3.344938056990631, 1.5707963275471217, 2.791635773220857], [-0.14332667559846984, 1.377048000531219, -0.3344757159057178]]], 'mode': 'complex', 'shape': 'dc2', 'L': 15}, {'edges': [[0, 1], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [4, 5]], 'seq': [3, 1, 0, 1, 2, 6, 4, 3, 2, 0, 2, 4, 2, 5, 0, 0], 'params': [[[-1.7893580898396637], [-1.661070730515664], [0.7083430585296906], [-0.4050769286775253], [-1.064384559693149], [1.5303985516252423]], [[2.7179480132113856], [1.1127765766227802], [0.8353658610714314], [-1.511530686868125], [-2.878268103598425], [-3.4311965291459883]], [[-0.2814664651408732], [-1.5567113122325207], [3.6576658911485125], [-3.2356834404612496], [-2.8449386443419202], [0.26140192239676685]], [[1.4761640131656069], [1.9014839071991947], [3.7559028719912435], [-1.4218637923494546], [3.057172483093357], [-0.3265215190749185]], [[-2.1258367312709727], [0.08984655342426663], [4.293812215712716], [0.14349886045141805], [1.3634359598443857], [-1.3840111867380689]], [[0.5658615800947471], [0.6412364415930326], [-2.109859771391749], [-0.7195653679235644], [-0.2932408171873301], [-1.362460226719923]], [[-1.3224646358873413], [-0.9138538684278732], [-1.0317329072718608], [2.075056836809701], [2.981953805701934], [1.5707960877185616]], [[-2.6354944975063552], [-2.3453762747657336], [2.5619513205727813], [2.5505461589436855], [0.1596388597611379], [2.069554528932338]], [[3.5423322917425026], [-3.6651914331350186], [4.364735070394841], [3.799552329448912], [-2.704920598424602], [0.6087533153657795]], [[3.1588016109258277], [-3.1415926190230095], [1.8717408330302336], [0.011511387367138139], [2.3332549134312752], [-0.2971667156918079]], [[1.4155206725186562], [-0.8410685852457723], [-0.3009446096964161], [3.1627683273961], [1.4608764863997559], [-2.4320847600901416]], [[0.633694357353627], [-0.9386595844300332], [-1.8925469031570232], [1.187999369608989], [-1.7638439259154202], [-1.519852968015908]], [[1.2609249589882374], [-2.818412687875604], [1.5707962048087192], [3.5632599484215146], [-1.2885997870806973], [1.325987066293659]], [[-2.474153029401424], [-0.2478025044449754], [2.9294771381882114], [0.553584900501794], [-2.7491561366759956], [0.6808405275825749]], [[1.5202074518511357], [3.389395076054071], [2.625765399966214], [0.26143077439393225], [-1.1210001196304125], [2.536152630062609]], [[3.9269908566894567], [1.570796397342729], [1.859577131536977], [-2.9027608853395708], [-0.7264677994890536], [0.09997984879659966]], [[3.141592541958313], [1.5707962436692893], [-2.7024307876886695], [-0.5002624593861901], [3.4182642725305827], [0.06942893357760634]]], 'mode': 'real', 'shape': 'g2x3', 'L': 16}, {'edges': [[3, 4], [4, 5], [2, 3], [1, 2], [0, 1]], 'seq': [1, 0, 2, 2, 0, 3, 2, 3, 2, 4, 3, 4, 3, 4], 'params': [[[2.436509721304283, -0.18398196133696273, -0.4724559048865304], [0.876455088368926, -0.10352423612156612, 0.4024877281823959], [-1.063940778352714, 1.327766840911121, -1.579022775674419], [0.10615134111449721, 0.408672999757586, -0.6340408781744906], [-4.024077761571401, 1.6109250495366398, 3.696124531320609], [1.1043150785735898, 1.9083469846929682, -3.457695731567039]], [[-1.4328788322425408, 0.03021231885269648, 1.1701941007454961], [-0.540777112392086, -0.2257673534759112, -0.8382227445294269], [0.06114839295898588, 2.925986679666249, -0.820843688023214], [-0.16051781740496143, -1.234434657120727, 0.11930051447823663], [2.58706077922046, 1.6914426164632073, -0.5428813843856394], [3.0794321637058992, 2.1220464701676973, -0.0794525073846699]], [[-1.5074106997946, -0.07699922879725156, -0.6629320667911855], [-0.7717011583407308, 0.13148983799082226, -1.80125081982257], [-0.35026372016173946, 1.136580489625729, 0.19275759477987317], [-0.8321616071747017, 0.9124305431843097, 3.779673540179708], [3.4849816083076752, 0.9100499544234312, 0.20058975906010423], [0.3365981002371761, -1.2017263441500425, 0.3968342046319521]], [[1.1406543205242987, -0.34318385179326183, 0.10028811827093448], [0.05099286469041895, 0.47554338135235424, -0.005130863171262917], [-0.3084484954069188, -0.5408684996678091, 0.9580553004346727], [0.09648941399582135, 2.2797249392304932, 0.8850861775879039], [0.4076479750023255, -0.36569992820055663, 0.600674040263563], [0.24136311144498462, 0.8523336768737003, 0.6507270438619253]], [[0.9630602289843142, -0.18517295361706884, 0.5411431174613663], [-1.4189357745401254, 0.029287281435186413, -0.26930615409147113], [-0.4571706190893122, 0.8114385446706907, -1.7155677059422736], [-0.21129496560639804, 2.0097068037631907, -0.5738052514354826], [4.797547726759012, 0.9354954246792386, 0.46882156179055884], [0.27887505842861265, -0.4965657540021503, 0.7991330477633939]], [[-0.37098172557507525, 1.255318844938623, -0.4795281372438425], [0.1174093731976719, 0.49552181892217273, 1.8022683909965425], [1.1154264768869835, 3.1969777229350345, -0.40325571632330487], [-0.09674189644640288, 2.704985358977246e-09, -0.9005561265283686], [4.963027892666545, 2.3504090965658424, 0.05158835280439592], [0.17785459342268864, -0.9570964924102823, -0.4213785785079439]], [[-0.6819613095737728, -0.000701392811853746, -0.07883589065645245], [-3.1266904691361743, 1.0560377960414387, 2.192476738255283], [-2.4550536464386297, 1.0463407990909188, 0.9147485276509595], [1.1647821771713534, 3.1415926473546274, 0.2552648443924465], [0.11633183644735151, 1.2365412387070525, 1.5442505246010787], [0.7937685901839411, -0.8279386419108669, 0.8146745430743583]], [[-1.4417254551959753, -1.241777562834079, -0.04926429439304695], [0.5396004599429068, -1.189812688281604, 0.2557131239448777], [-7.0895374703268645, -0.5270106864527898, 3.396437001670845], [1.7840287907173076, -1.1179167001283403, -0.7808113835051305], [0.6332140153886954, 0.5064625448043223, 1.3682233892144395], [-1.554039650559028, 0.034652854909606845, 0.4750328090309278]], [[-0.41347900538768456, -0.1223383334048282, 1.0483324021481375], [-0.9652887861932654, 2.7653207088596345, -2.925090415463278], [-0.380091911660285, -4.162187641367355, -0.43588078298848787], [-0.600986074879981, -1.2023388613036854, 2.04315507342446], [-0.9313151197918438, -0.055539605095774415, 0.9821527148287652], [-0.3164039640594598, -1.2363624150985206, 1.2731817648677912]], [[0.8643333972277966, 3.1715086685033405, -2.187216692674866], [-2.981199294367488, -1.912781625708591, 0.7520317921265579], [-2.0829075326410957, -1.829478533460289, -0.18912132375727442], [3.1427243933940683, 2.487689070861481, -0.5712608574809529], [0.33046901279465946, -0.7654720489360406, -0.62428351256036], [0.04544806098648339, -0.22170672552707066, 0.044066397845372564]], [[1.595990664490684, 0.06666209443673321, -3.251995692409556], [2.4459597463960527, 2.521249813064806, -3.3351320149123143], [0.18912132072622864, -1.3121141205370956, -0.00536977875113656], [-0.7279203827106617, -0.07847855549229384, -1.0686269977610554], [-0.11509119447822601, -1.1626073802204855, 0.9987359645332383], [0.4140449098918826, 0.9440683501142845, -0.2616812495140742]], [[-1.107564644226782, 0.6544699497594729, 0.7545825279845393], [-5.205274502319869, 1.145627735332616, 3.0508674752175464], [3.835369268711106, 2.0885542457854736, -0.845098586982752], [-0.6921921710052509, -0.4910386332934258, -0.17207810203158683], [-0.3424970211307682, 0.7837700296036649, -0.8000416100428756], [-0.1314363718324124, -0.5431925772287923, -1.4995824584919974]], [[-0.960537696158997, 1.0024471090278768, -1.2147519652169283], [-1.1541639991162571, 4.8292358476839405, -0.12487645445058569], [0.5699039727560073, -0.5344118261430967, 0.735107169272211], [-1.289637502385441, 1.3470659153080409, -0.5194672161108205], [-0.6271218561276607, 1.3576158446870754, 0.66697342159105], [1.2280210097494675, 0.45288181026482494, -1.0181915370508552]], [[-0.9545718957435388, 0.895456636787502, -0.0428742404625836], [-2.2901443839280096, -1.5707963175430515, 4.016041193449703], [0.9487139897678502, 1.2699215574500613, 1.0359451916202524], [-1.1291073155413356, 0.9070265424033022, -1.3572909323976825], [-0.5502358380327791, -2.2245090837912653, 0.29964473731975433], [-1.3680300837055084, 0.14117171123075614, 0.8460333890986182]], [[-1.5539186943827208, 6.8353625093923e-09, 0.21909999676728253], [2.2671441218574304, 4.712388978912557, 0.3682795609173042], [3.7137687651074094, 1.4511103600983848, -2.232199771978807], [0.37570765580653737, -0.28275678091761686, 0.4572224807658288], [0.24289839144203926, -0.42143256484217706, -0.13991222970520914], [0.279323095556005, 1.7614115936876686, -0.3101729797307747]]], 'mode': 'complex', 'shape': 'path', 'L': 14}, {'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], 'seq': [1, 2, 2, 0, 3, 1, 2, 3, 0, 4, 3, 2, 1, 0], 'params': [[[1.893402453994438], [-1.358752630894669], [-5.324689090014993], [3.2753562805276197], [3.916781002118719], [2.7896427812180984]], [[2.2643873873174476], [2.891139804771109], [0.7929217843982659], [-0.6666372704197181], [3.1597905753751], [1.2417007950225747]], [[0.10325320556176924], [-2.4959160311764075], [1.3929692058770387], [2.0186284563993095], [3.410804358674936], [-0.4727318655369935]], [[-1.7349302905580017], [-0.2300119283947745], [-2.1170264161016017], [-3.6160275951199456], [-1.1466852994778987], [0.7444409253665526]], [[0.44294350330046994], [-2.179939161194875], [-2.6620068053091144], [2.672781784315362], [-3.6729851020394695], [-1.484054856099981]], [[1.665413833073859], [1.2246222567334384], [-2.092154461419412], [1.081482267919913], [3.1795369651185763], [-2.087721075974799]], [[-2.325374496319009], [-2.795846893365648], [3.141591969779358], [4.246386456723975], [0.880116053885941], [-2.7392822821836242]], [[1.1177850535800011], [-0.5466940773419275], [-0.4485586017425943], [2.526112911317482], [-1.873376980713394], [-3.362004492418501]], [[0.05454850501277645], [0.8164280403056939], [3.297424726619327], [-0.02876134910247], [3.7020507131142657], [-0.47759428123645464]], [[1.9811781893296643], [-3.057107502444674], [-2.8632673561378716], [0.437057302017024], [-4.487448936656213], [1.1352153466811334]], [[-1.0773785974814531], [-1.3256105165198433], [1.3031828404313772], [0.3771022811461984], [-1.5707962724955902], [-0.6973517522474663]], [[-0.9404122243751932], [1.7984224087675842], [1.8528112528822929], [1.5707963562129643], [-3.154376431742846], [-1.8228290677329044]], [[0.269029624979035], [1.798897044339205], [1.570796329711792], [-1.707168254591967], [1.4562048988759366], [4.345273637144328]], [[2.12377735914213], [1.5707963365588735], [3.8556694660324093], [1.5255613836302606], [4.603886372540069], [2.1227197950356844]], [[2.4471577787005244e-08], [-1.5707963003411747], [-2.2848731168513177], [-1.389189449882668], [-1.3349185193106352], [-2.377016371876192]]], 'mode': 'real', 'shape': 'path', 'L': 14}], (6, 3): [{'edges': [[1, 2], [0, 1], [2, 3], [1, 4], [0, 5], [3, 4], [4, 5]], 'seq': [0, 2, 3, 4, 1, 0, 1, 0, 1, 5, 6, 5, 6, 5], 'params': [[[-0.4697479101421825, 0.14354036589843083, 0.216971448937442], [3.5342193827324, 1.570796339944114, 0.6812775103480257], [-0.09182838696813195, 4.712388989433568, 0.41472687539139474], [0.3884418704858401, 0.2981614439536927, 0.9206523734745234], [-0.3743826302324516, 0.46510442417215064, -0.034364151423258216], [1.048167274275293, 0.011125975298355809, 0.664276930056111]], [[3.8397483337747427, 0.25523713291127703, 1.2570213026698478], [3.806962255562815, 0.29362557894655084, 0.15101740074706027], [-0.21597743183125, -1.6628406955112611, 1.2989971478230182], [1.451518635460448, -0.2011959176235893, 1.0148191295978792], [-0.6033305150819187, 0.31174033953658087, 0.5451111103630175], [1.4374574165694702, -0.27095597286915435, 0.07854421215245565]], [[-0.11513348620701185, -0.22095459237725634, 0.06240885895035636], [0.3727996752196948, -0.09192202451922124, -0.49131389732048025], [0.6527356269150453, 1.577294514954798, 1.1400765803191701], [3.1956759933273746, 0.4413769296076698, -0.7106960361937337], [-3.403750844981456, 1.1900432865540664, 3.228729719279937], [0.4224261083177043, -0.08271037771609357, 0.02765149695118153]], [[-0.433241651866293, 1.5771425341878251, 1.2475895847415637], [-0.8077697265636465, -1.3577046780376638, -0.9467043913079903], [0.24511308028646198, 0.797281519249375, -0.7408589982728321], [-0.24049018431166713, -0.44846015610607565, -0.02678036600406528], [4.469017159806411, 2.7345662425306796, 0.032580847822934605], [-2.2973630000119027, 1.4502313065981451, 2.991958751603726]], [[1.0653523965378575, -0.39008092302742114, -1.850360020199692], [-1.938703683114272, 4.000051496770453, 1.132492559232741], [-1.4413316858017793, -0.7201339115158631, -1.0887064378140232], [-0.9794419690955897, 0.05799065311713135, 0.5589669360390761], [-0.2968198695921201, 0.9089304465279313, -0.20419473073666086], [2.948905796955601, 2.443226790052348, 0.310356358991686]], [[0.07904622604538394, 1.1215979013819493, -0.5275629593184741], [3.102673775247766, 1.5707963276341408, -2.0899090959187743], [2.73505968185626, 2.772727551249343, 0.04879293319228327], [-1.2822765425807137, 0.47557120728015234, -0.10626051936603376], [-0.6895033203334753, 0.43591391347591657, -0.10316668932756197], [0.6159104766275649, -0.3002828603903151, 1.165475458916266]], [[-1.348028210586758, -1.0139805171464333, -0.26614661612937135], [-4.193276212412864, 1.570796330000439, 3.068305729100778], [3.150944762792256, 0.7024615822244114, 0.19929099663464], [-0.1481085945354065, -0.5035481825816117, -0.0824063430419744], [-1.1054370619322895, -0.3499698800788111, 0.8340244289869396], [0.5669117225033605, 0.36381492462098597, 0.04112114696069479]], [[-0.782871558582991, 0.5631589314078885, -0.48312997614564585], [0.07328690158836208, 1.5707963351722176, -0.06817553615543319], [0.012651488137764457, 0.8794727153976564, 0.4335011615415783], [-0.23065411198420974, -0.5229519528992616, -0.31661162614344773], [-0.45948024696626816, 0.21902330089938563, -0.12618828324560358], [0.19895485432260177, -1.12437280857267, -0.4143905887633831]], [[0.5010635849995209, 1.007709988785061, -1.0585910030218826], [-3.073417118469305, 1.5707963279143664, 2.759627956769439], [0.8424950612334743, 1.041501018019739, -0.08636385786799484], [0.5378932432695936, -1.2221165067101445, 0.7753131208719423], [0.05247038540230047, -0.04553799127009248, 0.24290939242997808], [0.6228944431227869, -0.16912772322693612, 0.6996954257075494]], [[-0.16632312785770775, -0.3411507656059588, -1.3373252243880447], [-0.11160859946158631, 1.5863562290544082, 0.5860852422482622], [3.629370677687296, 1.6943392290803603, -0.835807418146529], [0.1883468121078831, -0.2871194983033381, -0.4814102846520265], [-0.4388510052669365, -0.01646834681882102, -0.7711897082722302], [-0.8724554730959349, -0.9586730985522209, -0.05404491698309201]], [[0.19847407221373675, -0.3711139750372618, 0.3938812947701678], [-0.788363805472659, 0.29675446168657504, -0.1733748583768095], [0.09600508910879216, -0.7753699594274615, 0.8761889036759968], [1.5421665731806253, -0.07297348785916358, -2.262730807000475], [1.729894106021427, 0.6154796985371431, -1.0692799516674394], [-0.6783577278475932, -0.21734078007676544, 0.6435699009844111]], [[-0.07991339063330909, -1.7919142057201367, 0.02751224134476518], [0.8688921373276319, -0.3574063341917124, -0.9192151588677052], [0.41538381161787846, -0.11532787928649982, 0.9727461858317555], [-0.6836399798035789, -1.6423884180429433, 1.8993941545077004], [-4.789857516734782, 1.570796340285785, 2.2052295432088584], [4.2366905623532665, 1.7909334470439344, 1.2629039771527593]], [[0.47845754556653364, -0.2845148731730981, 1.3130730467663336], [0.6619796975599961, 0.022944901597353483, 0.05164943004721586], [0.06207222181350653, 0.4684537853186901, 1.3091454292351161], [0.7497033213686964, 1.609868397798912, 0.28748739771507703], [0.9363631075204141, 1.5707963151271884, -0.17138152712093166], [1.073078572192245, 0.31246182960732516, 0.03400253036641449]], [[0.9714194829101646, -0.34126637147447414, 1.082658074433393], [1.138432896749022, -0.880258433018271, -0.22747538606547937], [0.8290692408783975, -0.5951384068183778, -1.267380020347334], [0.27517251505753204, -0.04618391849937205, -0.8673410015567765], [-2.5461632730970893, 1.5707963229658395, 4.842548827355712], [0.03859362763231639, 0.598690379904494, -1.0013052377947564]], [[0.5717458554507768, -1.365844658314836, 1.5984604289043463], [-1.1113096219350675, 0.13238199958692692, -0.8362229731365327], [0.702125088990648, 0.27157426180862493, 0.6382458931561625], [-2.8364302843888205, 1.5707962936451842, 2.414529934021484], [-0.4818947831305035, 3.141592624705666, 0.4919986730997689], [0.046222901206536555, 1.1954660730052373, 1.3218070747731556]]], 'mode': 'complex', 'shape': 'dc2', 'L': 14}, {'edges': [[0, 1], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [4, 5]], 'seq': [3, 2, 4, 3, 6, 0, 4, 1, 3, 6, 5, 5, 0, 2, 4], 'params': [[[2.4636320289916265], [2.133490754134326], [-0.6571009783349802], [0.3146590749055514], [-4.5611763834626124], [2.0222997511537653]], [[-1.2085317602635253], [0.7415575833102286], [3.116477268560992], [1.9929057026959274], [3.7279419641481484], [0.20708735000623493]], [[-0.9433845467022034], [-2.939471889404241], [-0.2446528490600339], [-1.0237736130861346], [-0.359683814193137], [-1.2740704764148463]], [[-3.7594401304339353], [-0.919024941546302], [1.342316366393771], [-2.5514529786214504], [4.32343719333378], [2.2443634903392584]], [[-3.130671680491299], [-0.9850304097742275], [-0.9362419908714198], [-2.1176676300176234], [-3.3025144117736063], [-0.058087459389658036]], [[1.8660071085146357], [1.7704285765638363], [0.345248648053129], [0.5114031973425801], [-3.538798170363096], [-0.6590838746104934]], [[-0.4490446138566135], [0.6621972514017227], [1.604871466004706], [2.5387457731334213], [-0.6846350151301367], [-5.624101428252175]], [[2.0198409406515068], [-1.7969948196293084], [1.8593089101332925], [-1.23561588950062], [-5.675689975211796], [0.9704782157843209]], [[-1.1078839035696348], [-0.4359987585673095], [1.6836905414805652], [0.7432312218373499], [2.0451415246023963], [0.3370392964112028]], [[1.4539020384038912], [2.067466039897508], [-1.117568589374367], [1.6929924920716077], [1.5707963267948968], [-1.3075175071374598]], [[2.5526417407105835], [2.472102178945462], [1.002180071093567], [0.8228034416264709], [-3.1415926535897927], [-3.7859559176111257]], [[1.5190340160693998], [2.7155250715315904], [-0.23497999625102936], [-1.5707963267948968], [3.141592653589793], [-1.9159675284225788]], [[-2.8468975648193426], [2.1696846703948185], [1.1529200035289289], [-0.6474896543929946], [2.227654054804875], [-1.3365690199917084]], [[-1.7508742233688714], [-4.712388989245195], [-3.5601527769496353], [-1.4705789760919428], [1.65257659235498], [-2.8244300544682193]], [[0.07021959757324019], [2.686159329013792], [1.5707963267948968], [0.30051556802999535], [0.48025646899234775], [-1.1326517670706449]], [[1.6806546118366226], [0.45543331777962176], [-3.1415926592250227], [0.3641912837503846], [0.351901868684398], [-1.5707963360454544]]], 'mode': 'real', 'shape': 'g3x3', 'L': 15}], (7, 2): [{'edges': [[1, 2], [2, 5], [1, 6], [0, 1], [4, 5], [5, 6], [3, 4]], 'seq': [0, 1, 2, 3, 0, 3, 0, 3, 3, 4, 5, 4, 5, 6, 4, 6, 4, 6], 'params': [[[3.514334393200789e-05, -0.14230148625046746, -0.33761513113744257], [3.141608849707121, 0.45175754265398305, 0.30893082219801804], [-0.0001250368895881783, 1.9590182667166123, -0.452281971099805], [-0.00014322897816364843, 0.09929771536644925, -0.04556057900869329], [-3.544540105523399e-05, 0.002999570497782332, -0.02962308275101221], [-0.00015724634556808895, -0.0008608246252839226, 0.023524355676889073], [-0.0005697101012550258, 0.0469520037574578, -1.2266223705131478]], [[-0.33761513126833065, -0.19538820654579586, -0.4091817927093358], [3.4505234759615138, 0.5277536883355236, 0.6087921803701046], [-0.45228197099833195, 0.10113899537562183, -0.23449374337788617], [-0.04556057890918054, 0.07853272448700692, 0.05634088655839601], [-0.029623082642244943, -0.022025547574877985, -0.5230439190741256], [-3.1180682977668837, 1.5699364584354314, 2.8524137773487563], [-1.2266223704654329, -0.08611816397448187, -0.5657417614362447]], [[-0.4091817926097234, -0.05507888647743497, -0.1339799031682119], [0.6087921803930283, -0.24437949036970658, 0.17913519283029805], [-0.23449374351069366, 0.10148015154454408, -0.30189593963595507], [0.05634088662852123, 0.013793543948253035, 0.6206262684926408], [-0.5230439188866762, 0.022075810734350037, -0.6062168254418362], [2.8524137774559977, 2.0495114352921266, 0.3582269633780081], [-3.707334414775364, 1.49075233677708, 3.164440443581182]], [[1.4368166318594198, 0.9356723264374378, -1.6762174908316703], [-1.3916611211854346, -2.427416158879793e-09, 1.8945707989856688], [-0.3018960401077659, 0.05382122919747977, -0.2732280579524396], [0.6206263180918266, -0.006668049458193751, 0.17630975813810928], [-0.606216775959603, 0.011386294201886783, -0.2705644453270107], [0.3582270843236659, 0.18580988746232618, 0.32699056343611194], [3.164440427073029, 1.7060464883385584, -0.6608873724568831]], [[-1.8089456587962078, 0.6384134240966632, 1.7160898304365202], [2.3408101089367346, 1.5707963315506883, -1.5710787178689236], [2.870761560716293, 1.6588140523659818, 0.08575023467405612], [0.2706382569660595, 0.039701904507770974, 0.1625017603398678], [-0.29683583984555745, -0.14757253396906897, -0.23976483913923285], [0.31526478557598403, 0.045541640201825107, 0.27478483493763617], [-0.6247850390697274, 0.19536608606146091, -0.6312894185653496]], [[0.14529364271235903, 0.3142034958988559, 0.06671068772839446], [-3.141874865156187, 1.5707963291771652, 2.6792570180504964], [3.227342888231807, 1.0315045558136278, -0.12813014958438065], [0.1625017603307269, 0.0783475087945932, -0.28212377552747303], [-0.23976483910614316, -0.3652464833885134, -0.11838402902814171], [0.27478483484519817, -0.09762530395272777, 0.23404758170471818], [-0.6312894185073316, 0.1800593855659064, -0.5143716142270983]], [[0.3717993321581336, -0.1817916921703556, 0.5724765853448129], [-0.49298082132092486, 1.5714877964433158, 1.318617384281792e-08], [3.0132621450671175, 1.954576835923572, -0.17992604182967353], [-0.43113848230352136, 0.11656036579793323, -0.5746551486698469], [-0.02155372240369196, -0.7168178043963873, 0.053944224557270284], [0.19247393032378998, -0.13087340061846112, 0.09584618973571618], [-0.42292048081855205, 0.3501904337771821, -0.4219326362869777]], [[0.5724765854833296, 1.6465853956666816, -0.30999974237737266], [1.3215290524152304e-08, 1.6421918206894661, -3.4190701487505502e-09], [2.9616666117213133, 1.5379170975271887, 0.1211444188734155], [-0.5746551484655683, 0.15032856352634533, -0.42986904707710943], [0.053944224574835706, -0.9156238647400392, -0.009188334113790166], [0.09584618985976995, -0.19403910690374454, -0.030248416303996464], [-0.42193263642426604, 0.3215564020720291, -0.2786431749815401]], [[-3.4515923959393606, 1.5707963346398746, 3.541901337635944], [-3.3281884065350922e-09, 0.7853981371582315, -0.10293854958120247], [0.12114441889119189, -0.017848449266141528, 0.13405148237146436], [-0.42986904722018277, -0.038935416669436666, -0.25588688472488674], [-0.009188334171686333, -0.9569746587259331, -0.14500770836545004], [-0.03024841633484461, -0.22618999409008994, -0.03141063941734234], [-0.2786431749231663, 0.4134449559766276, -0.14837057299849132]], [[-2.741283969566938, 1.4124696389541878, 3.112776054789329], [-3.244531203140922, 1.662567923923991, 3.0024005484298932], [0.13405148230992742, -0.003079399865136034, 0.1463827351891149], [-0.2558868846300857, -0.09935117967661362, 0.056813234556943225], [-0.14500770838963442, 0.7199413039400707, -0.3731424343678933], [-0.031410639323455586, 1.3449793332489477, 0.4208446272466601], [-0.14837057302765957, 0.4822780977706939, -0.015710540358224164]], [[0.05581519121949817, 0.06646513936012766, 0.16482224870563789], [-0.12970717490748537, 0.054011959770461415, -0.04135190812821674], [0.1637297625591777, 0.013492342652147974, 0.15784730869352578], [0.2901146517206549, 0.14660124194888394, 0.3897718791779084], [0.01426030673839385, -0.6333031272475159, 0.6106184559091417], [0.034504761005054195, 1.5708750489722103, 0.7853981705317902], [0.01080568849903446, 0.8177089552941014, 0.10558727626124854]], [[0.16482224873707596, -0.06090321797192035, 0.2531258432393738], [-0.041351908120652786, 0.1541639350410671, 0.1023526750647593], [0.1578473087706168, 0.04515125049040437, 0.14660554118534122], [0.38977187919061157, 0.1656402296232003, 0.1690608106207114], [0.6106184560103906, -0.3070689205657296, 1.233100859488472], [-2.356194483100858, 1.5706104792801407, 2.617993887216993], [0.10558727630353966, 0.38485898051597695, 0.1257047537540039]], [[0.25312584321729736, -0.23481785211026043, 0.2581270705591224], [0.10235267501889758, 0.04463762419093071, 0.09792071096660983], [0.1466055411701923, 0.04552372058799957, 0.11951013641141586], [0.1690608104825486, -0.10755171328917457, -0.1642980378462206], [1.2331008593627355, 1.1288994534941514, 0.716991376174257], [-0.5235987663777472, 1.5707963130543554, 0.07256345098490519], [3.2672974073439067, 1.418532737739155, 0.05621678676577428]], [[0.2581270705712419, -0.25881722071917607, 0.20211610109454037], [0.09792071097751934, -0.00017204477548375837, 0.11715773844917658], [0.11951013639880212, 0.08732043929355914, 0.08560103860208988], [1.4064982489353872, 1.323685592754783, -1.7729398647702876], [-0.8538048202914956, 0.4455368722197793, 1.5000108074589857], [0.07256348473167772, 0.4198977761957099, 0.03869260160790565], [3.1978094011455918, 1.5923412053641357, 0.028959729515413975]], [[0.060025001261069195, -0.18699027456277728, -0.031085782686096713], [0.14971162687662526, -0.02329825437824258, 0.1517609301454588], [0.11674188988313121, -0.05018270623838658, 0.11195174066132671], [0.30222826821687704, 0.8234845229182023, 0.08458752195635584], [0.2569928386729657, 0.6154796546508969, 1.093138052843461], [-0.0017393538883053482, -0.3356562012176489, -0.06601839867137446], [0.013986374427451837, -0.05031057232411135, -0.007290242122103742]], [[-0.031085782680036082, -0.12712598993559476, -0.04933846111286305], [0.15176093015397266, -0.06464051369850435, 0.08449849888123855], [0.11195174066369676, -0.05139323364994986, 0.11190959528724333], [0.08458752196309881, 0.8059859170944191, 0.08824998447650352], [-2.048454600753806, 1.5707963197001313, 2.356194488332286], [-0.06601839865660218, 0.5952875541624876, -0.01938349422195512], [-0.0072902421452765934, -0.037073676412924625, -0.0180956280354785]], [[-0.04933846115695575, -0.07700721613484582, -0.058505283828077737], [0.08449849891247961, -0.027528069259763262, 0.057490176238450055], [0.11190959521645837, -0.022556047915461865, 0.058679769601922685], [0.0882499844640472, 0.5590893076973317, -0.5589565364510246], [-0.785398165236327, 1.570796167173897, -0.7853981750442783], [3.1222091593367978, 1.5500882089360255, -0.00990716841989444], [-0.01809562797566631, 0.0447969980380188, -0.01751154611645986]], [[-0.05850528380700699, -0.03137499274645992, -0.0650837797656712], [0.0574901762166589, -0.04417682953568323, 0.029280493629061474], [0.05867976960972186, -0.0125572638402896, 0.0292795222130799], [-0.5589565364644861, 1.3035710157996514, 0.05189290260830724], [-0.7853981751079788, 3.9269908756958816, 0.00836196768519713], [3.1316854851969693, 1.6678987898547382, 0.07058115487891728], [-0.01751154609886034, 0.01354591372309916, -0.002335672041183919]], [[-0.14295501430198207, 0.11078684498141975, -0.20174753824092315], [-0.026098051423848505, -0.042383537266058605, 0.05289654273244213], [0.007585655989143303, 0.049414314889450944, -0.01974087901860147], [-4.267108631990265, 1.5707963105800502, 3.1675331008785896], [-3.149954632131081, 1.5707963121386463, 3.167533151540456], [-0.0004916878729186883, -0.09733995944578791, -0.04381819600490318], [0.013026552596016405, 0.007452703289302557, -0.027307373687155564]]], 'mode': 'complex', 'shape': 'dc2', 'L': 18}], (7, 3): [{'edges': [[1, 2], [0, 1], [2, 4], [1, 5], [0, 6], [3, 4], [4, 5], [5, 6]], 'seq': [0, 1, 2, 4, 1, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 5, 5, 6, 5, 6, 5], 'params': [[[-0.26348550118064473, -1.0206988418061158, 0.17002245917294215], [3.0852261393154894, 1.8473375452072458, 0.5374575871967219], [-0.006102940253998708, 1.1907402847861976, 1.1208194510346894], [0.03036694774339881, 0.020168308030659117, -0.21859936370923239], [-0.13553173363115567, -0.499174645978553, 0.3218338712397363], [0.5077188737084458, -0.03096077090790489, 0.27226672846968775], [-0.4509399465227487, -0.15662886716919058, -0.7440609497226147]], [[3.4470261075045694, 1.282190902943396, 0.00951176280418267], [3.695618796791894, 1.740575987610019, 0.0373030969776446], [0.8972540805897533, 0.4152825739759224, -0.2875104197594087], [-0.3834353463652349, -0.30709888434237764, -0.6446583932758173], [0.07043056585256206, -0.9053688583732189, 0.6212109835046178], [-1.2824436065644478, 0.1897056312973, 0.13688776721753582], [0.37602718018436926, 0.24765389163976426, -1.0113844472238323]], [[2.9192230194088937, -1.7177214347965846, -0.8717308914716253], [-0.6874739971603697, -1.551235424602963, -0.5328473766264741], [0.13327251179029523, -1.4073068151894303, -0.6264041581557835], [0.7051001397323053, -0.07750438849985833, 0.38411722293947326], [-2.4498741822483923, 1.230005945382917, 3.886208069204923], [0.05430498489690192, 0.45792028556672804, -0.8144806502538489], [-1.5040929796229023, -0.19121211245980735, 0.07875372215060676]], [[-0.6837437687713114, -0.3788180402949414, -1.6339650556038705], [0.18593677027804073, -0.7768017587413087, 0.701275013637262], [0.6310564387981086, -1.567946765738975, -1.078659809453305], [0.005787832547577228, 0.4008399005078065, 0.9406081416599956], [4.261716408817709, 1.7172754146120104, 0.7128500402842609], [-2.7515414228799893, 1.769494594730126, 2.6715079975762], [-0.45448278734550457, -1.3090574066686758, 0.2208062887103735]], [[1.692234087252278, -6.974625960418582e-08, -0.8646643520726237], [-0.3537946864779641, 0.7044299192686054, 1.4155373446220023], [0.39382526299511866, 0.12958926168231635, -0.08114532236852445], [0.1274277898563157, -0.2686769825018557, 0.9515889815461003], [1.0351958658666893, 1.4492163687113093, 0.37099706279590244], [0.057242049898989185, 1.6339036466949384, -0.6740694757916311], [3.9263297742194667, 2.7926529017735393, -0.23478790901481802]], [[1.7988247688657273, 2.3358676007311505, -2.004243666883213], [2.0524904907079518, -0.4765682869196775, -3.158122741859486], [0.1887650277178235, 0.8820638220224671, 0.6990789802336549], [0.8596369576852981, -0.10685587311708683, -0.20118598013604258], [-0.5387775675400884, 0.0948394432459385, -0.07509412001190553], [1.0852165303248824, 1.103137394121791, 0.4154690679774751], [-0.159451695223056, -0.4036238845118594, -0.39808922591164303]], [[-1.0267823680376993, 0.7681315963807629, -0.2374564300455789], [-3.398988625632168, 1.5707963485924255, 2.416321848168761], [3.4793319170955734, 1.5972922367369622, -0.6906195580813448], [0.45293103159982895, -0.10505862288969259, -0.3635841080180736], [-0.41357255263688475, -0.20856384671535302, -0.0027943069866463916], [0.12171110026575821, 0.28965727241161215, 1.9940481082681145], [-0.4789339058568859, 0.4187619224768192, -0.07569106483778525]], [[-0.725600168433081, -0.26402604145767106, 1.2486105937870962], [-0.8455255195942052, 1.5707963836988952, 0.44568891142534817], [3.6762882125610123, 2.370667327078278, 0.5396339940326697], [0.010538772323466533, 0.014178637394680702, 0.19915168396286223], [1.592744632123392, 0.8935327304797527, 1.1050112852887872], [0.8578560707925451, -0.4551527184459854, 0.6890534869313583], [-0.18854169063609177, -0.19758903139970996, -0.0077950933449052635]], [[1.1882128249620774, 4.5092958575642745, -0.9233777053454837], [1.125107438720388, 1.5707963247033816, -0.6086853497515219], [2.9456663246409636, 0.9682459563962468, 1.5237599816934655], [0.18392500910569554, 0.5835042688002354, -0.1845448456858499], [1.4271958616308695, 0.6951897059394333, 1.385314315865462], [-0.1262876198801402, -1.3111633732290469, 0.21445121390518604], [-0.9097543907357827, 0.7551000877434154, -0.24103243471945485]], [[-4.245057975771746, 1.3418764338582645, 2.3587602409863537], [-2.973358977905779, 1.997741903671785, 2.2049563132554337], [-0.7460538890514838, 0.18879939479259902, 1.1417740701421666], [0.36014888317456023, 2.522922305121133, 0.15793526943688613], [-2.7093828157584827, 1.684416270266911, 4.527148366862952], [0.9299983969602711, 0.29999800273025373, 0.6769180534427885], [0.06011540801992104, -0.9355183359465433, -0.13107696649108586]], [[1.1595774010813287, -1.4117456199744414, -0.02522079272041375], [-1.019826316234667, -0.5401429034133896, -0.3659531904960064], [0.412298116538112, 1.2245916769808682, 0.3354141443125156], [0.13729705577529372, 0.2330362666496508, 0.5345600899375715], [-3.2288787010878246, -0.44076139839632367, 4.051520310009035], [2.0151806709238946, -2.4205553079076263, 2.814492761954932], [0.5630678328003632, 1.3379605179073093, 0.028047981953711658]], [[-0.0022185226456561094, 0.3834819009409995, 0.5406048736559452], [-1.1071809539298485, 0.5041131264178942, 0.3903872988961908], [0.4184511418063779, -0.02727152873110674, 1.0748500718962728], [-0.08547441130235103, 0.4067584982212544, 0.47873230344055284], [1.309750222266483, -1.9586707945680542, 1.32557476392811], [1.1179606799044348, 1.370229255420093, 2.728009288902887], [2.059081604653072, 2.0599244343829204, -0.02194052357447921]], [[0.4844112702444055, 0.7877800054444111, -0.6807501481791213], [-0.36557243777853077, -0.23359558031778185, -0.5194833398031721], [-0.021732318269065673, 0.673661159380786, -0.09878197124561706], [0.2044571931223851, 0.11993466144273648, 0.9337965722744884], [-0.09560728108961682, 2.912481453539775, 1.0586082851748186], [-1.7192795942536045, -1.6649130466347049, 2.2275367124191288], [0.0219405265273162, 1.0816682457841171, -0.8981455488943052]], [[-0.10643572597608893, 0.12854786203859028, -0.7428094260014535], [0.9416309815049622, -0.2327709269917364, -0.11986303657979816], [-0.5591890360058629, 0.357985266618456, -0.4441228037459207], [-0.05492899295169904, -0.31711008880038644, -0.1397365188642181], [0.061754029440862314, 1.2590154802419804, -1.6563987065479167], [-1.6934576355849988, 1.2934761801559505, 2.3225107626915746], [0.2962475482132192, -0.29671719901331306, -0.5893788962438223]], [[-0.7374340931137106, -0.21777820545459226, 0.30674647819889994], [-0.3443753912293068, 0.7345349312872571, -0.12195487742185748], [0.059893129679426484, 0.5232940617696746, 0.6087689149656126], [0.06092042929558642, -0.7586427498823662, 0.5724576636575239], [-0.6870409326480157, 4.744375273522538, 0.6492749721299133], [-1.0152983601259609, 2.171700240677934, -0.2095496768491725], [2.5325863449317625, 1.4600035318495417, -0.14915354156451968]], [[-0.5341947984712329, -0.6699760612363124, -0.9061418011403044], [1.3498327358827655, 0.7310118099417731, 0.13140388411102052], [0.21498357313062094, 0.24985289649726689, 1.6573742694736253], [1.3364307710989465, 2.2328167225258304, -3.2268759542019883], [-1.612976197918249, 4.803359757266591, 1.2697492343120849], [-2.2120664467252937, -0.39709537535888684, -0.3738859745486365], [3.188371526274578, 1.8720835482965588, -0.7032895499622404]], [[0.6350704500907706, -0.07734643379033698, 0.2502791733967134], [-0.056405490632667435, 0.9265813769576525, 0.07471872768660687], [-0.8342706230238782, 0.6885867585718788, -1.304189357803026], [0.9943633684512004, -2.1838758107017586, -2.4242850859318934], [2.1237980017392992, 2.5250578898601193, -2.001609507094898], [0.21480253143331238, -0.49778489633832224, 0.15965886731989143], [-0.9387738930453569, 0.5793896799005187, -0.036673811933289]], [[0.6401079705196643, 0.2569512793809154, -0.6157936394876807], [-1.2051572941281, 0.20791476095051414, 0.1383892739838709], [-0.07257423378317661, -0.9717774930099571, 0.655611777101835], [-2.091259249101964, -0.48287913898405377, 2.1884864697521285], [1.997118240838962, 0.7854082435141824, -1.0502448333679204], [3.0607359605133118, 2.2497048355411096, -1.7723308862039386], [-0.6834868312920829, 0.25256451851461464, -0.45114384945102853]], [[-0.6133489885664967, -0.37508606685282686, 0.4392557582557361], [0.19361712269987733, -0.2476966995373518, -0.20000914865000735], [0.5000222227508261, -0.551790313878346, 0.2747480402138197], [-0.34916442081467564, -0.7089791039352653, -0.019900614315109623], [-2.39161767420513, -1.5707963332900805, 3.3394768929991736], [0.8208838545962066, 1.8801473948455314, 0.17130766024425176], [-0.33371829791185464, -1.1433234546178586, -0.8869414549864097]], [[0.7246805907003203, 0.37776580842988805, 0.5066772617764177], [0.15405240702704756, -0.5454224452530352, 0.5514839552772163], [0.1250341894932304, 0.6127966453022318, -0.5829864219935007], [-0.5334656673617482, -0.5966286129280696, -0.6081402822062533], [-0.1978842203682666, -1.570796337283134, -1.7693709619468334], [0.6401601256943016, -0.43464848474040374, -0.7111381450890302], [-1.4853949612411084, -1.1769835235158377, -0.08452584001140433]], [[0.2814444436427497, -0.3364513864548408, 0.621933706824765], [1.1407180000703545, 0.1595805309587136, 1.0398200950722956], [0.6051952284814879, -0.3347310297205185, -0.44464877885466797], [0.03872217935877827, -1.051195934521329, -0.4709949368046586], [-3.249639547948472, 1.570796331505269, 4.589610820890546], [-0.5492290713784035, -0.23949276037098408, 0.30473127412805], [0.6782127937457264, -1.5600544507895968, 0.6731881689583994]], [[0.025556443044610692, 0.04062020511965397, -0.4645248676995128], [-0.6286904755356497, 0.568371498714543, 0.9048772902705304], [-0.6030124319593163, 0.045349582383908495, 0.31990847480809853], [-0.03403138692644361, 5.393318289599633e-08, 0.3662523878097907], [1.6935744515728748, 1.5707963547036572, -0.4465297009781195], [3.3467792727267422, 1.3608218312665141, 1.0082228816371797], [1.219809902410463, -0.8559430178068528, 1.3222630538765108]]], 'mode': 'complex', 'shape': 'dc2', 'L': 21}], (8, 4): [{'edges': [[2, 3], [1, 2], [0, 1], [3, 4], [2, 5], [1, 6], [0, 7], [4, 5], [5, 6], [6, 7]], 'seq': [0, 1, 2, 3, 4, 5, 6, 2, 2, 1, 2, 1, 2, 1, 0, 1, 0, 1, 0, 2, 2, 1, 2, 1, 2, 1, 2, 2, 7, 7, 8, 7, 8, 7, 8, 9, 8, 9, 8, 9, 7, 7, 8, 7, 8, 7, 8, 7, 7], 'params': [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.141592653589793, 1.0683814272372003, 0.0], [0.0, 2.901974084960706, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [3.141592653589793, 0.6020729202613331, 0.0], [3.141592653589793, 1.0683814272372003, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[3.141592653589793, 0.24497866312686428, 0.0], [3.141592653589793, 0.6020729202613331, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[3.141592653589793, 0.2449786631268643, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 1.5707963267948966, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [-3.141592653589793, 1.5707963267948966, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [-3.141592653589793, 1.5707963267948966, 3.141592653589793], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [-3.141592653589793, 1.5707963267948966, 3.141592653589793]], [[0.0, 1.5707963267948966, 0.0], [-3.141592653589793, 1.5707963267948966, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.141592653589793, 1.5707963267948966, 0.0]], [[0.0, 0.5235987755982989, 0.0], [0.0, 0.5235987755982989, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[-3.141592653589793, 1.5707963267948966, 3.141592653589793], [0.0, 1.5707963267948966, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-3.141592653589793, 1.1780972450961724, 3.141592653589793], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.39269908169872414, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-3.141592653589793, 0.39269908169872414, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.1780972450961724, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 1.3089969389957472, 3.141592653589793], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.26179938779914935, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 0.26179938779914935, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.3089969389957472, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[1.5707963267948966, 1.5707963267948966, -1.5707963267948966], [-1.5707963267948957, 1.5707963267948966, 1.5707963267948957], [0.0, 0.0, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[1.5707963267948966, 0.6154797086703876, -1.5707963267948966], [1.570796326794898, 0.6154797086703868, -1.570796326794898], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[-1.5707963267948966, 1.5707963267948966, 1.5707963267948966], [1.5707963267948981, 1.5707963267948966, -1.5707963267948968], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-3.141592653589793, 1.2630564724597029, 3.141592653589793], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.30773985433519363, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-3.141592653589793, 0.30773985433519363, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.2630564724597029, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 1.5707963267948966, 0.0], [0.0, 1.5707963267948966, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[-3.141592653589793, 0.7853981633974483, 3.141592653589793], [1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[-3.141592653589793, 1.5707963267948966, 3.141592653589793], [-3.141592653589793, 1.5707963267948966, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.5707963267948966, 0.0], [-3.141592653589793, 1.5707963267948966, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5235987755982989, 0.0], [0.0, 0.5235987755982989, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 1.5707963267948966, 3.141592653589793], [0.0, 1.5707963267948966, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 1.1780972450961724, 3.141592653589793], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.39269908169872414, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 0.39269908169872414, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.1780972450961724, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [3.141592653589793, 1.5707963267948966, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 1.3089969389957472, 3.141592653589793], [3.141592653589793, 1.5707963267948966, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.26179938779914935, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 0.26179938779914935, 3.141592653589793], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.3089969389957472, 0.0], [3.141592653589793, 1.5707963267948966, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5707963267948966, 1.5707963267948966, -1.5707963267948966], [-1.5707963267948957, 1.5707963267948966, 1.5707963267948957], [0.0, 0.0, 0.0], [3.141592653589793, 1.5707963267948966, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5707963267948966, 0.6154797086703876, -1.5707963267948966], [1.570796326794898, 0.6154797086703868, -1.570796326794898], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.5707963267948966, 1.5707963267948966, 1.5707963267948966], [1.5707963267948981, 1.5707963267948966, -1.5707963267948968], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 1.2630564724597029, 3.141592653589793], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.30773985433519363, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 0.30773985433519363, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.2630564724597029, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.5707963267948966, 0.0], [0.0, 1.5707963267948966, 0.0], [3.141592653589793, 1.5707963267948966, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 0.7853981633974483, 3.141592653589793], [1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.141592653589793, 1.5707963267948966, 3.141592653589793], [-3.141592653589793, 1.5707963267948966, 3.141592653589793], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], 'mode': 'complex', 'shape': 'dc2', 'L': 49}]}
_peer_PEEP = {(5, 2): [(5, [('u', 2, (0.06417416539079573, 0.2002061048513533, -0.15863876446279643)), ('u', 3, (-0.25230469651845844, 0.11499575798417633, -0.11415213349578218)), ('u', 4, (-0.04641210951175617, 0.047097538848295754, -0.345014388292597)), ('u', 0, (3.4101130654655565, 2.3752905945655853, -0.8270713443163429)), ('u', 1, (0.06653450325499813, 1.6142750764453435, -0.020401255826077887)), ('cz', 0, 1), ('u', 2, (3.0824116174688734, 0.2665572236654895, -2.6718660872937625)), ('u', 3, (2.812749741470863, 1.1504773191668187, -3.291761196786068)), ('u', 4, (-0.3386503216440297, 0.2568185357469526, -0.958397282468898)), ('u', 0, (-1.3503117116976115, 0.059308187734340906, 2.4465534531296043)), ('u', 1, (-3.438206272788822, 1.0030328897431138, 2.834065801596405)), ('cz', 3, 1), ('u', 2, (2.8427869724867643, 0.0107155370572205, -3.3089695261608014)), ('u', 3, (3.662533412840636, 2.0711396814058896, 0.5934087022435817)), ('u', 4, (2.6013466232852736, 1.561574539622621, -3.312151135708441)), ('u', 0, (2.3692144472676833, 1.3200739426623036, -3.0607921888285197)), ('u', 1, (2.959402844346953, 0.3258250188609984, -2.5210423351250766)), ('cz', 4, 0), ('u', 2, (0.060212989652261226, 0.08199952126939657, 0.17983937989010085)), ('u', 3, (-2.0546677704294, 0.41359455675247786, 3.4966300158551316)), ('u', 4, (2.927982661767651, 2.3890840222883973, 0.008562608568400965)), ('u', 0, (0.0851468004887593, 1.5707962365246233, 0.45774770083199323)), ('u', 1, (0.48667563634765765, 1.7207376779086916, 0.06339294494997275)), ('cz', 0, 1), ('u', 2, (1.7682397169090667, 0.5733692770586745, -0.9852974041236715)), ('u', 3, (-3.97598845872157, 3.0958077893047333, -1.2519709441529943)), ('u', 4, (-1.411415665160749, 0.18948186757998975, 3.842454179297896)), ('u', 0, (2.974852292104601, 0.2951590468832118, -2.848831168965828)), ('u', 1, (-2.5984107882735894, 1.583536392500949, 2.9333194961275066)), ('cz', 2, 3), ('u', 2, (1.4379397283244173, 0.5785609573479032, -1.2449756204344027)), ('u', 3, (5.386783270571561, 2.1862762409213947, 0.06473778843364775)), ('u', 4, (0.42563975135858423, 0.890449079408611, -0.3326320131496935)), ('u', 0, (-0.03863288970513888, 0.02728274949310999, 0.06517792955787628)), ('u', 1, (0.17723956207029218, 0.30310176174565606, 0.01594192134885969)), ('cz', 3, 4), ('u', 2, (0.3197897423922124, 0.4995194824989192, 0.4758324674415158)), ('u', 3, (-3.205585019209745, 2.5261128353466136, 1.521770157525988)), ('u', 4, (-3.35713121360027, 1.3501617031512694, 2.3897042561083683)), ('u', 0, (-2.643147556157265, 0.4110150774511253, 3.0323963518084405)), ('u', 1, (-0.27413403963193306, 0.3086820392296995, 0.11691029273789783)), ('cz', 2, 3), ('u', 2, (3.022587830332299, 0.02825914011524722, -3.111678660603214)), ('u', 3, (-1.5208572416193316, 2.356194503198517, -1.630758653677268)), ('u', 4, (2.1805623755906116, 2.134080734985709, 0.38855058949595767)), ('u', 0, (0.0012482574713538336, 0.464629504416325, -0.09528659826040922)), ('u', 1, (0.1930212265602076, 0.2140220406991674, -0.04131447494920497)), ('cz', 3, 4), ('u', 2, (0.18359247568409248, 2.3838301091939544, 0.04108671285385225)), ('u', 3, (-1.5106188129922442, 1.570796368572668, -0.529214806467051)), ('u', 4, (3.939929685673091, 1.1987744112516259, 0.3078061173187987)), ('u', 0, (-0.19845511870835986, 0.3374254387257764, -0.07409905552826768)), ('u', 1, (0.04373643240782611, 0.1505983632221452, -0.15910901580761577)), ('cz', 2, 3), ('u', 2, (3.5794731185322934, 3.141592585539292, -2.59671999775915)), ('u', 3, (-2.612377760845247, 1.570796273340851, 3.2159650695398354)), ('u', 4, (0.017736528180962233, 0.39065518985106523, -0.2283821708560611)), ('u', 0, (2.5103147656811147, 1.6338905350785855, -3.240530077745801)), ('u', 1, (0.19260207987938857, 0.6221358807800801, -0.3705347463748))]), (5, [('u', 1, (0.0, 1.9823131728623846, 0.0)), ('u', 0, (0.0, 1.1831996401397158, 0.0)), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 0), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.1831996401397158, 3.141592653589793)), ('u', 3, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 0, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('cz', 0, 1), ('u', 1, (1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17)), ('u', 0, (-3.141592653589793, 0.7853981633974483, 3.141592653589793)), ('cz', 0, 1), ('u', 2, (1.5708446024674354, 0.615479708415582, -1.423395234744073)), ('u', 3, (-4.076420369540545, 3.141592626948408, -1.5120906372603562)), ('u', 4, (1.048234610153547, 0.35262501995127393, 0.5341844404437943)), ('u', 1, (3.13971064238359, 1.5549126675981364, -3.1198619513397956)), ('cz', 2, 3), ('u', 2, (1.7181966918454517, 0.4175592040251194, -1.129679736742519)), ('u', 3, (1.629501736812657, 1.5707963754323724, -1.9634953272520899)), ('u', 4, (0.5341847786752187, 1.395402396652681, -0.08565703082680046)), ('u', 1, (0.02173079729491832, 0.012249787592922284, 0.02192159707382869)), ('cz', 3, 4), ('u', 2, (0.36217946582783367, 0.24400971593032947, 0.21043539161929958)), ('u', 3, (-2.7488937843489896, 1.5707962363946044, 2.6639343464546186)), ('u', 4, (-0.035937284598891436, 0.22412384746199288, -0.08211648071987442)), ('u', 1, (0.02360274576274645, 0.010333005300155922, 0.018529438308069072)), ('cz', 2, 3), ('u', 2, (0.21043497530010674, 0.32477380231996045, 0.05986836762128027)), ('u', 3, (-0.47765830713519264, 1.5707963912970386, -0.7853981304146608)), ('u', 4, (3.0594760908471903, 1.1760553487634595, 1.0711803299696445)), ('u', 1, (0.018529854621483908, 0.003665687734053444, 0.00789648023365892)), ('cz', 3, 4), ('u', 2, (0.05986851767437061, 2.0343149862431438, -0.02696525028711255)), ('u', 3, (-0.7853981304146526, 1.570796374683024, -0.408473893033093)), ('u', 4, (4.212772983559444, 0.866754934231214, 0.34817009268817856)), ('u', 1, (-3.1336961733561233, 0.004371363852426555, 3.1328635209222155)), ('cz', 2, 3), ('u', 2, (3.1146274033026846, 0.7853981561867128, -2.936824902590321)), ('u', 3, (-0.40847389303309467, 1.5707963783916108, 2.981049246342238e-08)), ('u', 4, (0.3481700926881752, 0.34811323690933227, 0.2816367716829552)), ('u', 1, (-3.150321161808917, 0.01372663933785196, 3.1099203377723534)), ('cz', 2, 3), ('u', 2, (0.20476775099948474, 3.14159255591908, -5.638358777533247)), ('u', 3, (3.141592683400285, 1.5707963335529351, -2.906301630238328)), ('u', 4, (0.2816367716829386, 0.8086217290267099, 0.08239309705148327)), ('u', 1, (-3.1732650734801875, 0.024147437792669837, 3.080591167834259)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793))])], (6, 2): [(6, [('u', 5, (-2.453459795629976, 0.6154108398293135, 3.792701447413792)), ('u', 1, (3.0019054622941077, 0.5152502283591138, -0.033126937133005185)), ('u', 4, (0.2292433038815036, 0.01481289503683034, -1.0177080395201215)), ('u', 2, (0.013018042588220924, 2.1944221470493073, 0.32818112010404515)), ('u', 0, (-2.9787958806291632, 0.0886689973418528, 2.4614610365237706)), ('cz', 1, 2), ('u', 5, (3.3349764885375532, 0.0022967240122862384, -2.8964023772851433)), ('u', 1, (2.9589589148629383, 0.6176586630691273, -0.7274142970197928)), ('u', 4, (2.539613002099871, 1.570072339813219, -2.953840329876965)), ('u', 2, (-0.42430642488693754, 0.02300284793942631, -0.22225922963172645)), ('u', 0, (-0.5928730258605697, 0.13985223781419126, -0.06993066539964493)), ('cz', 4, 2), ('u', 5, (-3.139204532245132, 1.2536185284290928, 2.96666802332593)), ('u', 1, (-3.3075227590289553, 0.14398860885600887, 2.95586682816478)), ('u', 4, (2.780613807479522, 1.4554101551444294, 0.1051697514051142)), ('u', 2, (-2.7876697038606877, 0.1002101766399416, 3.1872120774420005)), ('u', 0, (-3.141880020564839, 0.20856433151600512, 2.704007495148254)), ('cz', 5, 1), ('u', 5, (3.6115223449033067, 1.5090964279556942, -0.6997219565492345)), ('u', 1, (0.0, 3.934340781158519e-13, -1.6892619956600294)), ('u', 4, (3.0937138685259247, 0.25794218728504037, -2.8904607651497876)), ('u', 2, (0.1469691841132235, 0.6052525332913032, -0.3091487519746785)), ('u', 0, (1.083382316755207, 0.7326507291357435, -1.340342122680559)), ('cz', 1, 0), ('u', 5, (-0.24588352327733107, 0.20669328721014976, -0.13976273443047652)), ('u', 1, (1.2596702877841461, 1.5707963267949496, -0.9671876089234042)), ('u', 4, (0.5205856127125582, 0.16359306049893949, -0.45270694994421545)), ('u', 2, (2.7668963848978025, 1.9655136715248875, 0.4670899607319061)), ('u', 0, (-1.49213980074785, 0.766068440906933, 0.8410618941063295)), ('cz', 1, 2), ('u', 5, (-0.03790008073856499, 0.24190180430634445, -0.45231215425760385)), ('u', 1, (-2.1744050446660097, 0.615479708670376, 3.586940755305892)), ('u', 4, (-0.33181423317831465, 0.24053639415852562, 0.1806657260712599)), ('u', 2, (3.394353501268674, 2.2503318610333842, 0.19837895625303315)), ('u', 0, (3.1183009959852743, 0.25342840054172383, -2.9027348844705507)), ('cz', 1, 0), ('u', 5, (0.1530580101984686, 0.07271622055816514, -0.5189376642463033)), ('u', 1, (-0.44534810171638234, 1.5707963267948928, 0.01191361420548298)), ('u', 4, (2.6531340607693896, 0.010161404945631879, -3.3598053328628263)), ('u', 2, (3.7296091757018006, 1.6160485227010506, -0.5714337244232652)), ('u', 0, (0.0, 1.0674903348085818e-13, 0.6006852000182925)), ('cz', 1, 2), ('u', 5, (-0.13358344378767256, 0.8542086362571331, -0.1881003560509148)), ('u', 1, (-3.1535062677953256, 1.570796326794941, 2.764678964026394)), ('u', 4, (1.4963787604077234, 0.0528843622355743, -0.9244454803613732)), ('u', 2, (3.761595471028183, 2.0372422896199693, -0.346298124793583)), ('u', 0, (0.0, 1.198704791825506e-13, 0.8572929198784051)), ('u', 3, (1.8883805791828714, 0.615479708670409, -0.6467094478986308)), ('cz', 3, 4), ('u', 3, (-1.9293815007118043, 0.6039066904739256, 1.4683043598780685)), ('u', 4, (1.8262065010289699, 1.57079632655862, -1.3475066836698089)), ('u', 5, (2.8630429597927805, 1.9992251818045819, 0.10434066042637791)), ('u', 2, (-2.579314349438724, 0.7452325091927046, 2.4866137061923834)), ('cz', 4, 5), ('u', 3, (3.0450181662748643, 0.32108853740524523, -2.9988091323485935)), ('u', 4, (2.918303010546812, 1.5707963267026572, -3.2308762283953705)), ('u', 5, (2.966749132139194, 0.9925184113293671, -0.1258080208303063)), ('u', 2, (-2.5999248213601502, 0.34978155544370026, 2.5401908507714905)), ('cz', 3, 4), ('u', 3, (-0.0530316095262567, 0.23572757035344816, 0.21812206059616673)), ('u', 4, (-0.8660330433189241, 1.5707963264492601, -0.6042175386464284)), ('u', 5, (3.1979447430427776, 1.9455319197599266, 0.1770682281920064)), ('u', 2, (0.16813662917413538, 0.4690709141383835, 0.4382762374570831)), ('cz', 4, 5), ('u', 3, (-0.5633511319579381, 1.3485100752261905, -0.4529163910462173)), ('u', 4, (0.604217538579338, 2.345214810300157, 0.2970227717171343)), ('u', 5, (2.8279125566361265, 1.7856160657121039, 0.23433833243729563)), ('u', 2, (3.0204092877430533, 0.8396472289164091, -2.7838510527484246)), ('cz', 3, 4), ('u', 3, (2.8882297471232, 1.570796326763543, -2.148206467032401)), ('u', 4, (-0.29702277190250914, 0.7853981633974549, 0.2693558231996239)), ('u', 5, (2.576228776662998, 0.18867689737221688, -3.0086311766791254)), ('u', 2, (-0.5736763156694709, 0.2827881667245027, 0.5029686019782795)), ('cz', 3, 4), ('u', 3, (-2.5641825131640075, 1.570796326931336, 2.832019659879205)), ('u', 4, (-3.410948476535545, 1.570796326772328, 2.8320196598791765)), ('u', 5, (3.212369746983524, 0.03635589774106469, -2.535249559059127)), ('u', 2, (-0.8989665873249468, 0.7843530048892474, -0.013499265403156602)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('cz', 0, 1), ('u', 1, (1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17)), ('u', 0, (-3.141592653589793, 0.7853981633974483, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793))]), (6, [('u', 2, (0.0, 2.214297435588181, 0.0)), ('u', 1, (0.0, 1.0471975511965976, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.0471975511965976, 3.141592653589793)), ('u', 4, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 1, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 0, 1), ('u', 1, (1.570796326794898, 0.6154797086703868, -1.570796326794898)), ('u', 0, (1.5707963267948966, 0.6154797086703876, -1.5707963267948966)), ('cz', 0, 1), ('u', 1, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 0, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 1, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 0.9553166181245093, 3.141592653589793)), ('cz', 2, 1), ('u', 1, (0.0, 0.9553166181245093, 0.0)), ('u', 1, (-3.141592653589793, 2.5261129449194057, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 2.5261129449194057, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 3, (1.8883805791828714, 0.615479708670409, -0.6467094478986308)), ('u', 4, (1.7118826188473402, 4.726163444567278e-10, -0.883301315865618)), ('u', 5, (-0.17120323373106677, 0.452290667529392, -0.0696741876668316)), ('u', 2, (4.017129896144521, 1.756817436765106, -0.07916448927145558)), ('cz', 3, 4), ('u', 3, (-1.9293815007118043, 0.6039066904739256, 1.4683043598780685)), ('u', 4, (1.8262065010289699, 1.57079632655862, -1.3475066836698089)), ('u', 5, (2.8630429597927805, 1.9992251818045819, 0.10434066042637791)), ('u', 2, (-2.579314349438724, 0.7452325091927046, 2.4866137061923834)), ('cz', 4, 5), ('u', 3, (3.0450181662748643, 0.32108853740524523, -2.9988091323485935)), ('u', 4, (2.918303010546812, 1.5707963267026572, -3.2308762283953705)), ('u', 5, (2.966749132139194, 0.9925184113293671, -0.1258080208303063)), ('u', 2, (-2.5999248213601502, 0.34978155544370026, 2.5401908507714905)), ('cz', 3, 4), ('u', 3, (-0.0530316095262567, 0.23572757035344816, 0.21812206059616673)), ('u', 4, (-0.8660330433189241, 1.5707963264492601, -0.6042175386464284)), ('u', 5, (3.1979447430427776, 1.9455319197599266, 0.1770682281920064)), ('u', 2, (0.16813662917413538, 0.4690709141383835, 0.4382762374570831)), ('cz', 4, 5), ('u', 3, (-0.5633511319579381, 1.3485100752261905, -0.4529163910462173)), ('u', 4, (0.604217538579338, 2.345214810300157, 0.2970227717171343)), ('u', 5, (2.8279125566361265, 1.7856160657121039, 0.23433833243729563)), ('u', 2, (3.0204092877430533, 0.8396472289164091, -2.7838510527484246)), ('cz', 3, 4), ('u', 3, (2.8882297471232, 1.570796326763543, -2.148206467032401)), ('u', 4, (-0.29702277190250914, 0.7853981633974549, 0.2693558231996239)), ('u', 5, (2.576228776662998, 0.18867689737221688, -3.0086311766791254)), ('u', 2, (-0.5736763156694709, 0.2827881667245027, 0.5029686019782795)), ('cz', 3, 4), ('u', 3, (-2.5641825131640075, 1.570796326931336, 2.832019659879205)), ('u', 4, (-3.410948476535545, 1.570796326772328, 2.8320196598791765)), ('u', 5, (3.212369746983524, 0.03635589774106469, -2.535249559059127)), ('u', 2, (-0.8989665873249468, 0.7843530048892474, -0.013499265403156602)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('cz', 0, 1), ('u', 1, (1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17)), ('u', 0, (-3.141592653589793, 0.7853981633974483, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793))])], (6, 3): [(6, [('u', 2, (0.0, 2.690565841793531, 0.0)), ('u', 1, (0.0, 0.7590702092666635, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 0.7590702092666635, 3.141592653589793)), ('u', 0, (0.0, 1.2490457723982544, 0.0)), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 0), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.2490457723982544, 3.141592653589793)), ('u', 3, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 4), ('u', 2, (2.572576065009661, 0.0736751545868733, -3.6612768177774417)), ('u', 1, (-3.0957378634480905, 0.017542113885202545, 3.1693247411303176)), ('u', 0, (0.2924246944778046, 2.0144049022332478e-09, 0.22401953854093656)), ('u', 5, (3.141621456214928, 1.570796330465726, -3.09805213094587)), ('cz', 0, 5), ('u', 2, (2.621908489402255, 0.03193775843880457, -3.509769218800873)), ('u', 1, (-1.5430642242387382, 1.57176882783286, 1.5620387653890344)), ('u', 0, (1.79481557405754, 1.5707963063843675, -1.5707962946296408)), ('u', 5, (3.1851331934529603, 1.7388192561033189, 0.1445328327997646)), ('cz', 1, 0), ('u', 2, (2.773416114790225, 0.011712512915930865, -3.487634647530157)), ('u', 1, (-1.5795538882007654, 0.3508795374768818, 2.356194469389992)), ('u', 0, (1.570796358960163, 0.9388818690097547, -2.0807717245217927)), ('u', 5, (0.14453277891283747, 0.17781580492317572, 0.10044140220025762)), ('cz', 1, 0), ('u', 2, (2.795550619992101, 1.6149998495382067, 0.038072395040432916)), ('u', 1, (2.35619446939, 1.5707963346497302, -1.570796372880089)), ('u', 0, (-2.080771724521777, 1.6422561796537511, 1.5129388782153543)), ('u', 5, (0.10044150750248625, 0.1809763328933769, 0.06649110164234236)), ('cz', 2, 1), ('u', 2, (3.1796650317276396, 1.65869340820513, 0.034550296368021094)), ('u', 1, (-3.141592654236178, 0.6621452985233158, 3.141592652492418)), ('u', 0, (3.083735263681261, 0.5549864772909748, -3.1041332876626053)), ('u', 5, (0.0664911016423407, 0.18100721434933958, 0.03954869590947857)), ('cz', 1, 0), ('u', 2, (0.034550296368022856, 0.12533726647302446, -0.014375091750484691)), ('u', 1, (-1.0973828975781453e-09, 1.5707963231684132, 3.53287090559963e-08)), ('u', 0, (3.1790520195169822, 0.17270729538097135, -3.082543531819984)), ('u', 5, (0.03954869590948736, 0.17952451149012152, 0.017037161255502448)), ('cz', 2, 1), ('u', 2, (-0.014375091750492976, 0.009768469000364495, -0.020839153120665917)), ('u', 1, (-3.1415926182610865, 0.631914291686104, 3.141592623127217)), ('u', 0, (3.200641775359596, 0.1787908520679271, -3.0619521477417697)), ('u', 5, (0.017037161255500654, 0.17703034470092435, -0.002895412580624177)), ('cz', 1, 0), ('u', 2, (3.120753500469131, 1.5805564419183182, -0.0003483929136263164)), ('u', 1, (-3.0462575746354565e-08, 1.5707962186264584, -1.5802167536612645e-08)), ('u', 0, (-3.06195214774176, 0.49600372218524175, 3.07762352867769)), ('u', 5, (-0.0028954125806292964, 0.17349234963220875, -0.02153413067381888)), ('cz', 2, 1), ('u', 2, (3.1412442606761646, 1.5776831967573992, -0.021025722803725166)), ('u', 1, (-1.5802168466866894e-08, 2.094395037081264, -3.3851818561640647e-08)), ('u', 0, (-0.06396912491211032, 1.012964249330451, -0.06087199156588615)), ('u', 5, (-0.021534130673809944, 0.1686473636293627, -0.039769651430084436)), ('cz', 1, 0), ('u', 2, (-3.162618376393516, 0.006892905953000969, 3.0823156099915896)), ('u', 1, (-3.385181899379086e-08, 1.570796175154427, -0.1013274575995535)), ('u', 0, (-3.202464645155681, 1.5707962322375566, 3.040265178083402)), ('u', 5, (-0.039769651430092846, 0.16213869678448167, -0.05820906109483628)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 4, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 3, 4), ('u', 4, (1.570796326794898, 0.6154797086703868, -1.570796326794898)), ('u', 3, (1.5707963267948966, 0.6154797086703876, -1.5707963267948966)), ('cz', 3, 4), ('u', 4, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 3, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 4, (0.0, 0.0, 1.3951473992034527e-15)), ('cz', 4, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (-3.141592653589793, 1.2630564724597029, 3.141592653589793)), ('cz', 3, 4), ('u', 4, (0.0, 0.30773985433519363, 0.0)), ('cz', 5, 4), ('u', 4, (-3.141592653589793, 0.30773985433519363, 3.141592653589793)), ('cz', 3, 4), ('u', 4, (0.0, 1.2630564724597029, 0.0)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 4, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (0.0, 1.5707963267948966, 0.0)), ('u', 4, (0.0, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17)), ('u', 3, (-3.141592653589793, 0.7853981633974483, 3.141592653589793)), ('cz', 3, 4), ('u', 4, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 3, (-3.141592653589793, 1.5707963267948966, 3.141592653589793))])], (7, 2): [(7, [('u', 2, (0.0, 2.0137073708685356, 0.0)), ('u', 1, (0.0, 1.1071487177940904, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.1071487177940904, 3.141592653589793)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 6), ('u', 3, (0.11601578574459524, 0.11445306532186036, -0.05873314765785108)), ('u', 4, (-2.5229814239155903, 0.7853981633929247, 2.672654067515391)), ('u', 5, (0.0, 3.141592653589684, -0.3445112607582286)), ('u', 6, (3.3781982997579023, 1.9938023267735092, -0.07359164259788864)), ('cz', 4, 5), ('u', 3, (-2.9515962975765566, 0.5056755477723608, 2.2200443890006474)), ('u', 4, (3.2149773257678795, 1.1992828892140965, -2.9928042977979965)), ('u', 5, (3.644958195962718, 1.5707963267948784, -2.5163472193680594)), ('u', 6, (3.0487820351105013, 1.9514455601263996, -0.8156097462450673)), ('cz', 5, 6), ('u', 3, (-0.34283879299373743, 0.044686926957994734, 0.07892007325733039)), ('u', 4, (-0.7221058691384219, 0.5262637442364827, -0.5551588226138113)), ('u', 5, (-2.196041761016409, 1.5707963267946825, 2.623777901877183)), ('u', 6, (2.92253223232039, 2.4036930191468513, -0.20277826325117143)), ('cz', 4, 5), ('u', 3, (-0.6305368457106076, 0.7980329266973532, -0.7336226221434942)), ('u', 4, (-0.5167182473551505, 0.18315113902045307, -0.23668899703635093)), ('u', 5, (-0.5293827994841317, 1.5707963267949543, -0.5798174889070856)), ('u', 6, (2.725010306136793, 1.507659020777457, 0.5282031131704192)), ('cz', 5, 6), ('u', 3, (0.6727279757044259, 0.9274976539154787, -1.6936955613998808)), ('u', 4, (-2.1475182100310173, 1.4789871430681827, 1.6303559354483252)), ('u', 5, (-0.2648048008870131, 0.5314794575963966, -0.27660674173470357)), ('u', 6, (3.038008047238889, 1.8625898384981554, -0.5902471651918946)), ('cz', 3, 4), ('u', 3, (1.3756922625147034, 0.6154797086710831, -1.2971086488369805)), ('u', 4, (1.6399186564016064, 0.6499151492340337, -1.919730869296622)), ('u', 5, (-0.1671310696782647, 0.45729015600664225, -0.7313084035710773)), ('u', 6, (2.3024305502496483, 0.05323677740319882, -3.7734166048553166)), ('cz', 3, 4), ('u', 3, (-1.3014447755735246, 1.678142726791292, 2.8032854998997374)), ('u', 4, (1.8150173940969985, 0.6193795682353069, -2.3039289911090957)), ('u', 5, (3.386502475533139, 2.3788032598392967, 0.6418405556062463)), ('u', 6, (0.2903706668900086, 0.2084988360844264, 0.1824124091269326)), ('cz', 4, 5), ('u', 3, (0.21109755371946107, 0.2156690401268879, 0.18913462830811542)), ('u', 4, (2.432294277063316, 0.9553166181254876, -3.76355827912454)), ('u', 5, (3.7507986984912347, 1.570796326794571, -0.3571488549968511)), ('u', 6, (2.9777689850569438, 0.20221729840515162, -2.589186588281091)), ('cz', 4, 5), ('u', 1, (1.5324194940067908, 5.3021376566375703e-08, -1.56266077741119)), ('u', 2, (0.6018106158184183, 0.15980498277650393, -0.6772602058207539)), ('u', 5, (3.017679557250371, 1.3299074850227444, 1.2846107161235554)), ('u', 0, (1.6467528317913072, 0.6154796932346827, -0.6398573753162243)), ('cz', 1, 0), ('u', 1, (3.2155180795844163, 1.5707963354517513, -2.1087019206789432)), ('u', 2, (3.254949555276662, 1.7061877643292427, 0.03301095216222327)), ('u', 5, (0.10046169634961145, 0.6469997841069166, 0.12375403821524343)), ('u', 0, (-1.3225583272613946, 1.174054518294708, 1.1485285047746567)), ('cz', 1, 2), ('u', 1, (3.6794982122158824, 1.5707963588827134, -2.2819746843423734)), ('u', 2, (4.0086681688709564, 0.8720561710448409, -0.4466142483244484)), ('u', 5, (-2.8822710281041393, 0.255541143352538, 3.134271415481891)), ('u', 0, (2.6671680637740542, 1.0254009757252611, -2.1493071334045513)), ('cz', 1, 0), ('u', 1, (-1.8149345872017317, 1.5707963580972752, -1.9944583751605043)), ('u', 2, (2.8394887155613904, 1.5449097868074715, 0.12236943438758852)), ('u', 5, (2.985036412643358, 0.32468604606484874, -3.0385823672053704)), ('u', 0, (-2.4222679250537213, 0.6530276918702874, 3.6461799725817956)), ('cz', 1, 2), ('u', 1, (0.42366197447181075, 1.570796335553448, 0.34961258600754963)), ('u', 2, (2.2835768926992426, 1.8663010908405264, -0.31629292507938644)), ('u', 5, (-0.37412930654170795, 0.4719137211966899, -0.07265702258492518)), ('u', 0, (0.0883085696115993, 1.285424226825953, 0.4271782882932581)), ('cz', 1, 0), ('u', 1, (0.8988429114876053, 1.5707963445850344, -0.3475797548284105)), ('u', 2, (-0.14751240629459822, 0.22395525895596402, 0.5253665284084357)), ('u', 5, (2.763039648099014, 0.12247638867714401, -3.039418776964069)), ('u', 0, (-3.0679695671358878, 0.7853981652638563, 2.8806735599399267)), ('cz', 1, 0), ('u', 1, (-2.794012910794895, 1.57079632381052, 1.849453006574815)), ('u', 2, (2.8025555652395773, 0.516012291944641, -2.9035700835849583)), ('u', 5, (0.48051934449338873, 0.06465831827926592, 0.05791233663936998)), ('u', 0, (-1.099696410105327, 7.545066320794333e-09, 0.06847583918922551)), ('u', 3, (1.366996796440895, 2.101372532601, 0.1847370593476193)), ('u', 4, (-1.0222896891335609, 1.5707963267948746, 1.3277776916500403)), ('u', 6, (-2.995036775581398, 0.17876252601705145, 2.971728660123455)), ('cz', 3, 4), ('u', 3, (-3.0813612244726043, 1.5707963267946186, 2.1469433645606504)), ('u', 4, (0.2430186351454288, 0.7853981633960306, -0.2004559077141073)), ('u', 6, (0.24137550723464346, 0.20561324850047275, 0.5592502294868626)), ('cz', 3, 4), ('u', 3, (2.5654456158246015, 1.5707963267949054, -3.1866241037341707)), ('u', 4, (-2.941136745875393, 1.5707963267949157, 3.0965612032655714)), ('u', 6, (-2.760376582028562, 0.30509340548022884, 3.3325566018123283))]), (7, [('u', 2, (0.0, 2.0137073708685356, 0.0)), ('u', 1, (0.0, 1.1071487177940904, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.1071487177940904, 3.141592653589793)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 6), ('u', 3, (0.11601578574459524, 0.11445306532186036, -0.05873314765785108)), ('u', 4, (-2.5229814239155903, 0.7853981633929247, 2.672654067515391)), ('u', 5, (0.0, 3.141592653589684, -0.3445112607582286)), ('u', 6, (3.3781982997579023, 1.9938023267735092, -0.07359164259788864)), ('cz', 4, 5), ('u', 3, (-2.9515962975765566, 0.5056755477723608, 2.2200443890006474)), ('u', 4, (3.2149773257678795, 1.1992828892140965, -2.9928042977979965)), ('u', 5, (3.644958195962718, 1.5707963267948784, -2.5163472193680594)), ('u', 6, (3.0487820351105013, 1.9514455601263996, -0.8156097462450673)), ('cz', 5, 6), ('u', 3, (-0.34283879299373743, 0.044686926957994734, 0.07892007325733039)), ('u', 4, (-0.7221058691384219, 0.5262637442364827, -0.5551588226138113)), ('u', 5, (-2.196041761016409, 1.5707963267946825, 2.623777901877183)), ('u', 6, (2.92253223232039, 2.4036930191468513, -0.20277826325117143)), ('cz', 4, 5), ('u', 3, (-0.6305368457106076, 0.7980329266973532, -0.7336226221434942)), ('u', 4, (-0.5167182473551505, 0.18315113902045307, -0.23668899703635093)), ('u', 5, (-0.5293827994841317, 1.5707963267949543, -0.5798174889070856)), ('u', 6, (2.725010306136793, 1.507659020777457, 0.5282031131704192)), ('cz', 5, 6), ('u', 3, (0.6727279757044259, 0.9274976539154787, -1.6936955613998808)), ('u', 4, (-2.1475182100310173, 1.4789871430681827, 1.6303559354483252)), ('u', 5, (-0.2648048008870131, 0.5314794575963966, -0.27660674173470357)), ('u', 6, (3.038008047238889, 1.8625898384981554, -0.5902471651918946)), ('cz', 3, 4), ('u', 3, (1.3756922625147034, 0.6154797086710831, -1.2971086488369805)), ('u', 4, (1.6399186564016064, 0.6499151492340337, -1.919730869296622)), ('u', 5, (-0.1671310696782647, 0.45729015600664225, -0.7313084035710773)), ('u', 6, (2.3024305502496483, 0.05323677740319882, -3.7734166048553166)), ('cz', 3, 4), ('u', 3, (-1.3014447755735246, 1.678142726791292, 2.8032854998997374)), ('u', 4, (1.8150173940969985, 0.6193795682353069, -2.3039289911090957)), ('u', 5, (3.386502475533139, 2.3788032598392967, 0.6418405556062463)), ('u', 6, (0.2903706668900086, 0.2084988360844264, 0.1824124091269326)), ('cz', 4, 5), ('u', 3, (0.21109755371946107, 0.2156690401268879, 0.18913462830811542)), ('u', 4, (2.432294277063316, 0.9553166181254876, -3.76355827912454)), ('u', 5, (3.7507986984912347, 1.570796326794571, -0.3571488549968511)), ('u', 6, (2.9777689850569438, 0.20221729840515162, -2.589186588281091)), ('cz', 4, 5), ('u', 3, (1.366996796440895, 2.101372532601, 0.1847370593476193)), ('u', 4, (-1.0222896891335609, 1.5707963267948746, 1.3277776916500403)), ('u', 5, (3.405589725198712, 1.5476978984160668, 0.26813170011961907)), ('u', 6, (-2.995036775581398, 0.17876252601705145, 2.971728660123455)), ('cz', 3, 4), ('u', 3, (-3.0813612244726043, 1.5707963267946186, 2.1469433645606504)), ('u', 4, (0.2430186351454288, 0.7853981633960306, -0.2004559077141073)), ('u', 5, (-2.6874617869140525, 0.004774720761996742, 3.000213729348172)), ('u', 6, (0.24137550723464346, 0.20561324850047275, 0.5592502294868626)), ('cz', 3, 4), ('u', 3, (2.5654456158246015, 1.5707963267949054, -3.1866241037341707)), ('u', 4, (-2.941136745875393, 1.5707963267949157, 3.0965612032655714)), ('u', 5, (0.7204058193865766, 0.09992579049839889, 0.1522237206849802)), ('u', 6, (-2.760376582028562, 0.30509340548022884, 3.3325566018123283)), ('u', 0, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 1, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 0, 1), ('u', 1, (1.570796326794898, 0.6154797086703868, -1.570796326794898)), ('u', 0, (1.5707963267948966, 0.6154797086703876, -1.5707963267948966)), ('cz', 0, 1), ('u', 1, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 0, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 1, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 0.9553166181245093, 3.141592653589793)), ('cz', 2, 1), ('u', 1, (0.0, 0.9553166181245093, 0.0)), ('u', 1, (-3.141592653589793, 2.5261129449194057, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 2.5261129449194057, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('cz', 0, 1), ('u', 1, (1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17)), ('u', 0, (-3.141592653589793, 0.7853981633974483, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793))])], (7, 3): [(7, [('u', 2, (0.0, 2.451879499619181, 0.0)), ('u', 1, (0.0, 0.866397140683011, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 0.866397140683011, 3.141592653589793)), ('u', 0, (0.0, 1.2897614252920828, 0.0)), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 0), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.2897614252920828, 3.141592653589793)), ('u', 4, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 0, 6), ('u', 2, (2.927990914951556, 0.037608684936839026, -2.9067897925621486)), ('u', 1, (0.18682830662460814, 1.570796328805722, 1.5459630175254293)), ('u', 0, (1.82281298683628, 1.570796352722352, -1.4226728678028702)), ('cz', 1, 0), ('u', 2, (0.4418700531653876, 0.22288047860758717, 0.0668089225595277)), ('u', 1, (1.5956295571676744, 0.35087940812151497, -0.5144426080850821)), ('u', 0, (1.4226730203959437, 2.202710638202488, -0.8958445494596178)), ('cz', 1, 0), ('u', 2, (2.8604488232479293, 1.7550979759013636, 0.06030027122532955)), ('u', 1, (2.0852391456306374, 1.5707963250304045, -1.5792191742993054)), ('u', 0, (-1.1924972438880368, 1.427759669397298, 1.2476289875914057)), ('cz', 2, 1), ('u', 2, (3.005610498223304, 1.5164310130476222, 0.09862978373833009)), ('u', 1, (3.150015490793008, 2.4794474210737114, -2.77230333335792)), ('u', 0, (0.07782077051936362, 0.5352106269078157, 0.5416462673949637)), ('cz', 1, 0), ('u', 2, (-0.05524356225504093, 0.2675497095603755, 0.20359478394100908)), ('u', 1, (-0.36928932655915003, 1.5707963636983906, -0.25635923965612184)), ('u', 0, (-1.9930155820684545, 0.2366494479958406, 3.5462227821027197)), ('cz', 2, 1), ('u', 2, (0.1197300595996944, 0.04626078208212085, 0.38579804536576356)), ('u', 1, (-2.885233482113633, 0.6319143150580288, 2.5803068225615586)), ('u', 0, (0.711762447438835, 0.3870087056939558, -0.7547733850526132)), ('cz', 1, 0), ('u', 2, (2.387786870978032, 1.6139637451898206, -0.08929192193052904)), ('u', 1, (0.5612858904313562, 1.570796346954555, -0.19916906104258825)), ('u', 0, (2.488438267223762, 0.6013013961678155, -2.773850216397627)), ('cz', 2, 1), ('u', 2, (3.023045408509128, 1.9646733061472654, -0.05500547276974377)), ('u', 1, (0.19916905562690662, 2.0943951032485066, 0.2126552942333672)), ('u', 0, (0.614458438682712, 0.5568168189701632, -1.1003495014264453)), ('cz', 1, 0), ('u', 2, (2.7294533248225488, 0.4356400138815296, -2.964725961166434)), ('u', 1, (-3.354247960909918, 1.5707962849175021, 2.8895760244168196)), ('u', 0, (2.3260298735930833, 1.5707964715903298, -3.393609304055196)), ('u', 6, (1.7096019472539603, 0.8268800259858878, 0.6226969142284112)), ('u', 5, (0.6775044359672656, 1.5707963248723165, 0.2814876547916655)), ('u', 4, (2.313489117220361, 1.5707963275148005, -2.472571387807654)), ('u', 3, (0.07712786118430925, 3.134153302552414, 1.7620614073562337)), ('cz', 5, 4), ('u', 6, (2.4164750875067043, 1.649752765635693, 0.43070363091364383)), ('u', 5, (2.8601049935471403, 0.7853981633417665, 1.2894281993906118)), ('u', 4, (-2.372974730053756, 0.8644842060716954, 2.125849068145324)), ('u', 3, (-2.0270749703059034, 0.8982487151767372, -0.3821936977835688)), ('cz', 6, 5), ('u', 6, (3.901448919233749, 2.288021044208705, 0.2504639141045335)), ('u', 5, (4.635944698043758, 2.0767669458294863e-10, -0.9985038795557417)), ('u', 4, (-2.9998415836464183, 0.3979549002656172, 1.8593632581251363)), ('u', 3, (-2.6039287213865485, 0.9180809613579872, 2.60890251925452)), ('cz', 5, 4), ('u', 6, (2.593988511809022, 0.23513988322678683, -2.5365709795549267)), ('u', 5, (1.3563162800358772, 1.5707963265867255, -0.1824074250789215)), ('u', 4, (1.2917180113702056, 1.5707963263431968, -1.7516479641056701)), ('u', 3, (-0.6476283685074675, 0.7529085504892408, 0.718844508690945)), ('cz', 5, 4), ('u', 6, (2.5871461198581907, 1.0323099097198227, 0.02409387642085714)), ('u', 5, (-0.7724120834509618, 1.0459822339013305, 0.2834198793904126)), ('u', 4, (-3.086202990435042, 0.784303810403754, 2.877674174726899)), ('u', 3, (-1.0601937628728253, 0.7801694791108086, -0.8228006946101142)), ('cz', 6, 5), ('u', 6, (3.0559736873781795, 1.3347778860466655, -0.31702683437110935)), ('u', 5, (0.2615161709271716, 0.48384369269099137, 1.0627992426627673)), ('u', 4, (-0.046050816455809995, 2.3306961543571387, 1.181660419129296)), ('u', 3, (1.5700812638871322, 1.7664317780705903, -1.0534565184505182)), ('cz', 4, 3), ('u', 6, (2.466756499952858, 0.8138789408014773, -3.391253695161958)), ('u', 5, (2.8241488022799595, 1.9393207387956741, -0.2589588097378437)), ('u', 4, (1.7427773661823036, 1.5707963268846783, -0.7773601156092436)), ('u', 3, (-0.98774642550761, 1.5679244775504066, 3.197838646334241)), ('cz', 5, 4), ('u', 6, (2.3718838348624693, 0.1610554902863146, -3.1451609260925926)), ('u', 5, (3.1812438859274077, 1.2956388035229423, -0.9363594158902351)), ('u', 4, (-2.3642325351621944, 1.5707963268307001, 3.708296298777864)), ('u', 3, (-1.6583297901709708, 0.09154773214686243, 3.9445860178747347)), ('cz', 4, 3), ('u', 6, (2.0833912437352335, 0.4052932148735028, -4.128846981673444)), ('u', 5, (-0.5306514304602161, 1.2190942552290325, -0.5988795404597256)), ('u', 4, (-0.5667036454713823, 0.9553166181638688, 0.9868405769569011)), ('u', 3, (1.3477092368498658, 0.9888417142673291, 0.1453391073528154)), ('cz', 5, 4), ('u', 6, (-1.9734238059199634, 0.9210446919760125, 3.2753461886166804)), ('u', 5, (-0.7250272566644168, 0.13951695534942818, -0.966388415505158)), ('u', 4, (-3.1448769066377587, 4.806953846162392e-11, 3.1151898621906)), ('u', 3, (0.26855113730749347, 0.6231306260987792, -1.6117960646084089)), ('cz', 4, 3), ('u', 6, (3.3847826758333106, 0.00596977314801572, -2.8428600224957288)), ('u', 5, (2.1293157236538565, 0.7214702929695543, -0.14006408794000524)), ('u', 4, (0.6136427935159456, 1.570796326733598, -0.8101638400353446)), ('u', 3, (-2.5464923216353004, 0.8906782349719334, 3.3661107025646615)), ('cz', 5, 4), ('u', 6, (-2.818715732777364, 0.952878782096676, 3.0043417437514988)), ('u', 5, (0.7409371417897517, 2.9556950151821755, -2.2286588731330164)), ('u', 4, (-2.3314288130784124, 1.5707963269612122, -0.7960277889820586)), ('u', 3, (0.6793869157364618, 2.2236498925058483, 0.7190561537490348)), ('cz', 4, 3), ('u', 6, (3.031965526893565, 1.3094348861348473, -1.7242464394092538)), ('u', 5, (-0.12184148553446827, 1.439333407376042, -0.7002989429508656)), ('u', 4, (-0.7747685409843722, 1.5707963280360524, -2.1392781173878195)), ('u', 3, (-0.7249110800378891, 3.141592651784004, -4.3731490908327375))]), (7, [('u', 2, (0.0, 2.451879499619181, 0.0)), ('u', 1, (0.0, 0.866397140683011, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 0.866397140683011, 3.141592653589793)), ('u', 0, (0.0, 1.2897614252920828, 0.0)), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 0), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.2897614252920828, 3.141592653589793)), ('u', 4, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 0, 6), ('u', 6, (1.7096019472539603, 0.8268800259858878, 0.6226969142284112)), ('u', 5, (0.6775044359672656, 1.5707963248723165, 0.2814876547916655)), ('u', 4, (2.313489117220361, 1.5707963275148005, -2.472571387807654)), ('u', 1, (0.08830390508230578, 0.002530548629232911, 0.08153420123903503)), ('u', 3, (0.07712786118430925, 3.134153302552414, 1.7620614073562337)), ('cz', 5, 4), ('u', 6, (2.4164750875067043, 1.649752765635693, 0.43070363091364383)), ('u', 5, (2.8601049935471403, 0.7853981633417665, 1.2894281993906118)), ('u', 4, (-2.372974730053756, 0.8644842060716954, 2.125849068145324)), ('u', 1, (-3.0600584523507566, 0.004890692384524863, 3.216147527607153)), ('u', 3, (-2.0270749703059034, 0.8982487151767372, -0.3821936977835688)), ('cz', 6, 5), ('u', 6, (3.901448919233749, 2.288021044208705, 0.2504639141045335)), ('u', 5, (4.635944698043758, 2.0767669458294863e-10, -0.9985038795557417)), ('u', 4, (-2.9998415836464183, 0.3979549002656172, 1.8593632581251363)), ('u', 1, (-3.067037779572436, 0.010194790493636361, 3.2094698535851096)), ('u', 3, (-2.6039287213865485, 0.9180809613579872, 2.60890251925452)), ('cz', 5, 4), ('u', 6, (2.593988511809022, 0.23513988322678683, -2.5365709795549267)), ('u', 5, (1.3563162800358772, 1.5707963265867255, -0.1824074250789215)), ('u', 4, (1.2917180113702056, 1.5707963263431968, -1.7516479641056701)), ('u', 1, (-3.0737154535944806, 0.013494611910459637, 3.2034970868043264)), ('u', 3, (-0.6476283685074675, 0.7529085504892408, 0.718844508690945)), ('cz', 5, 4), ('u', 6, (2.5871461198581907, 1.0323099097198227, 0.02409387642085714)), ('u', 5, (-0.7724120834509618, 1.0459822339013305, 0.2834198793904126)), ('u', 4, (-3.086202990435042, 0.784303810403754, 2.877674174726899)), ('u', 1, (-3.079688220375256, 0.014919767336819259, 3.198529688640101)), ('u', 3, (-1.0601937628728253, 0.7801694791108086, -0.8228006946101142)), ('cz', 6, 5), ('u', 6, (3.0559736873781795, 1.3347778860466655, -0.31702683437110935)), ('u', 5, (0.2615161709271716, 0.48384369269099137, 1.0627992426627673)), ('u', 4, (-0.046050816455809995, 2.3306961543571387, 1.181660419129296)), ('u', 1, (-3.0846556185394896, 0.014604408480013018, 3.1947711245854737)), ('u', 3, (1.5700812638871322, 1.7664317780705903, -1.0534565184505182)), ('cz', 4, 3), ('u', 6, (2.466756499952858, 0.8138789408014773, -3.391253695161958)), ('u', 5, (2.8241488022799595, 1.9393207387956741, -0.2589588097378437)), ('u', 4, (1.7427773661823036, 1.5707963268846783, -0.7773601156092436)), ('u', 1, (-3.0884141825941067, 0.012684299018115628, 3.1923346473525807)), ('u', 3, (-0.98774642550761, 1.5679244775504066, 3.197838646334241)), ('cz', 5, 4), ('u', 6, (2.3718838348624693, 0.1610554902863146, -3.1451609260925926)), ('u', 5, (3.1812438859274077, 1.2956388035229423, -0.9363594158902351)), ('u', 4, (-2.3642325351621944, 1.5707963268307001, 3.708296298777864)), ('u', 1, (-3.0908506598270087, 0.009299651604358205, 3.1912508017852623)), ('u', 3, (-1.6583297901709708, 0.09154773214686243, 3.9445860178747347)), ('cz', 4, 3), ('u', 6, (2.0833912437352335, 0.4052932148735028, -4.128846981673444)), ('u', 5, (-0.5306514304602161, 1.2190942552290325, -0.5988795404597256)), ('u', 4, (-0.5667036454713823, 0.9553166181638688, 0.9868405769569011)), ('u', 1, (3.1912508017852623, 0.0046014604565363295, -3.0917101177485806)), ('u', 3, (1.3477092368498658, 0.9888417142673291, 0.1453391073528154)), ('cz', 5, 4), ('u', 6, (-1.9734238059199634, 0.9210446919760125, 3.2753461886166804)), ('u', 5, (-0.7250272566644168, 0.13951695534942818, -0.966388415505158)), ('u', 4, (-3.1448769066377587, 4.806953846162392e-11, 3.1151898621906)), ('u', 1, (0.04988253584121335, 0.0012411777267747558, 0.05130355500828455)), ('u', 3, (0.26855113730749347, 0.6231306260987792, -1.6117960646084089)), ('cz', 4, 3), ('u', 6, (3.3847826758333106, 0.00596977314801572, -2.8428600224957288)), ('u', 5, (2.1293157236538565, 0.7214702929695543, -0.14006408794000524)), ('u', 4, (0.6136427935159456, 1.570796326733598, -0.8101638400353446)), ('u', 1, (0.0513035550082845, 0.008035083641094912, 0.053750229853381834)), ('u', 3, (-2.5464923216353004, 0.8906782349719334, 3.3661107025646615)), ('cz', 5, 4), ('u', 6, (-2.818715732777364, 0.952878782096676, 3.0043417437514988)), ('u', 5, (0.7409371417897517, 2.9556950151821755, -2.2286588731330164)), ('u', 4, (-2.3314288130784124, 1.5707963269612122, -0.7960277889820586)), ('u', 1, (0.05375022985338495, 0.01555988486698217, 0.05700069010366602)), ('u', 3, (0.6793869157364618, 2.2236498925058483, 0.7190561537490348)), ('cz', 4, 3), ('u', 6, (3.031965526893565, 1.3094348861348473, -1.7242464394092538)), ('u', 5, (-0.12184148553446827, 1.439333407376042, -0.7002989429508656)), ('u', 4, (-0.7747685409843722, 1.5707963280360524, -2.1392781173878195)), ('u', 1, (-1.5137956366912224, 1.6241176501118906, 1.53859742869943)), ('u', 3, (-0.7249110800378891, 3.141592651784004, -4.3731490908327375)), ('u', 0, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('cz', 0, 1), ('u', 1, (1.570796326794898, 0.6154797086703868, -1.570796326794898)), ('u', 0, (1.5707963267948966, 0.6154797086703876, -1.5707963267948966)), ('cz', 0, 1), ('u', 1, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 0, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 1, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.2630564724597029, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.30773985433519363, 0.0)), ('cz', 2, 1), ('u', 1, (-3.141592653589793, 0.30773985433519363, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 1.2630564724597029, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('cz', 0, 1), ('u', 1, (1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17)), ('u', 0, (-3.141592653589793, 0.7853981633974483, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793))])], (8, 2): [(8, [('u', 3, (0.0, 2.1790419057051063, 0.0)), ('u', 2, (0.0, 1.0213290820372691, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.0213290820372691, 3.141592653589793)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 6), ('u', 4, (0.37799737698828695, 0.13502176705432672, 0.3214040538289391)), ('u', 5, (0.3985397211331689, 0.00929248085827109, 0.6187820595919282)), ('u', 6, (3.2233516201965586, 1.9067757501890086, -0.05923425999101717)), ('u', 7, (2.62546993154895, 1.5707963142598302, -2.972963826565313)), ('u', 2, (-0.303217546481335, 1.6184756894581043e-08, 0.15983311941825973)), ('cz', 7, 2), ('u', 4, (-2.779877173779888, 0.42155213867689517, 3.331313669718659)), ('u', 5, (0.5589160952636802, 0.7818017314577775, 0.15057489574352603)), ('u', 6, (-2.8387553185480625, 0.3454173956786111, 1.395808581518468)), ('u', 7, (2.0668343681755905, 0.1279373175679867, 0.03892061028899807)), ('u', 2, (-2.812748656454745, 0.19672262791544828, 3.31337968219386)), ('cz', 5, 6), ('u', 4, (-0.1771306243590159, 0.13389517030463718, 0.006932825931703093)), ('u', 5, (3.562087886730953, 0.643980325868593, -2.5821124658608414)), ('u', 6, (1.3916958469779557, 1.5707963158511256, -4.256035537516305)), ('u', 7, (-1.1156796118587728, 0.19288764151068444, 0.732595011970502)), ('u', 2, (3.2541291973852826, 0.030967642953222154, -2.842303518727326)), ('cz', 6, 7), ('u', 4, (3.0498097753948747, 0.2672829582477914, -2.7924000323318396)), ('u', 5, (1.1326269267880138, 0.4031206522779272, 0.28389941516135236)), ('u', 6, (-3.597946013149868, 1.5707964348703332, 0.4449556999661015)), ('u', 7, (-3.3908973224820644, 0.12096164373375765, 2.6080349100269538)), ('u', 2, (-0.09237201813450077, 0.16111360667461563, 0.09151238895854427)), ('cz', 5, 6), ('u', 4, (0.19735485201244668, 0.25948365810477114, 0.010813617398872308)), ('u', 5, (-3.2311828692849645, 1.2051876216157658, 2.416521252848505)), ('u', 6, (-1.4921532509903792, 1.5707964153532405, -0.3490796341036889)), ('u', 7, (1.9753936781527108, 1.058644921011962, 0.16157583510960716)), ('u', 2, (-3.162834510023021, 0.5747347109672598, 2.988971814363478)), ('cz', 6, 7), ('u', 4, (1.0697635739624871, 1.52658235064331, -1.5672771782289299)), ('u', 5, (-4.2127273977581225, 2.0992493467840383, 0.33134519297886533)), ('u', 6, (0.6984144283130203, 1.5252938260335696, -0.21440583620126957)), ('u', 7, (-0.19358815636896123, 0.2559393139967764, 0.47420913599389414)), ('u', 2, (-0.24775721799526773, 0.8788483436124567, 0.6306936253356296)), ('cz', 4, 5), ('u', 4, (-1.582827360163844, 0.21776170604375605, 0.9591539302732023)), ('u', 5, (2.3503014223355794, 0.6154796976043173, -1.7855613183375023)), ('u', 6, (3.3231024420029933, 1.6156397416055137, 0.03364890849704727)), ('u', 7, (2.709376855117922, 0.1955528890284926, -3.2338974877284086)), ('u', 2, (0.8680317109333897, 0.4997448463602015, -0.056827154909853406)), ('cz', 5, 6), ('u', 4, (3.377610500647098, 1.6793898967932746, -2.231024413023696)), ('u', 5, (2.7408778918623415, 1.5707963059636127, -2.1365798542758636)), ('u', 6, (0.58536098355167, 0.6094105028753255, 0.969585707515199)), ('u', 7, (-3.1168755558467276, 0.33665600594107586, 2.811651952041584)), ('u', 2, (0.005221149173498196, 0.19690787579305224, -0.3665845427224871)), ('cz', 4, 5), ('u', 4, (0.29504406680573697, 0.44846474121374397, -0.07571209595121042)), ('u', 5, (0.5657835138432934, 1.570796357313495, 0.8143058310464644)), ('u', 6, (4.55504008542289, 1.7809235481243086, 0.427220452988589)), ('u', 7, (0.3766145100605592, 0.06878691216042872, -0.009096775098752952)), ('u', 2, (3.2696925203060507, 0.7014902173162261, -2.9962649279847104)), ('cz', 5, 6), ('u', 4, (-0.1860683590876881, 1.1357275504580875, 0.8290934330764712)), ('u', 5, (0.7564903845717192, 2.3561944594688504, -0.008435285167640416)), ('u', 6, (3.492481894270857, 1.4998778378919626, 1.1331841437533055)), ('u', 7, (-0.17712062085308095, 0.10840439565034213, 0.5326929757954737)), ('u', 2, (0.02449124975477296, 0.21458275829027432, -0.8363589811847667)), ('cz', 4, 5), ('u', 4, (0.8541514401733135, 1.5707963383368226, 0.028213333138799424)), ('u', 5, (3.1500279094418913, 1.570796348241581, -3.113379339992024)), ('u', 6, (3.8253251949344014, 0.2836996500115563, -1.7985178959753672)), ('u', 7, (2.8333938913609673, 1.8008917306853038, 0.2999084398575791)), ('u', 2, (-3.015514266938172, 0.16650447060293516, 2.5369635946250644)), ('u', 3, (2.0233960134963267, 1.551684614316178, -3.7235890685381516)), ('u', 1, (0.839139008888907, 2.3561944756657436, 0.2504183418438984)), ('u', 0, (-0.4857212906342475, 0.36627051757076884, 0.7052590757840566)), ('cz', 2, 1), ('u', 3, (-0.9925699444130445, 1.3760589052021495, 0.030269165866424308)), ('u', 2, (0.10192704701523331, 1.5707963660171047, 0.9817870956243782)), ('u', 7, (-3.05783079748452, 0.44702950751519543, 3.0591122753422684)), ('u', 1, (-2.401632134400665, 2.011001083856294, 2.7586445799647334)), ('u', 0, (-2.8349759805099564, 0.11167287121304673, 2.4995603415012106)), ('cz', 3, 2), ('u', 3, (2.6528659076669325, 1.4955652126472454, -0.581422947523186)), ('u', 2, (0.5890092258714162, 1.5707964336192224, 0.9543365451447087)), ('u', 7, (-2.4564367081972427, 0.001145535296781138, 2.9803634148299323)), ('u', 1, (2.629943460823213, 0.8604362142849246, -1.384439748694525)), ('u', 0, (-2.7696027145682147, 0.5293282705771406, 2.868909903382035)), ('cz', 2, 1), ('u', 3, (2.7543724602965853, 1.9386575875706562, -0.09457060820058993)), ('u', 2, (1.1400585522478939, 1.5707963842675132, -0.24709348408042153)), ('u', 7, (-0.03540724420330254, 0.36887871489577734, -0.02926433557526944)), ('u', 1, (-2.7464252866341696, 0.1488361173078092, 2.928654164947201)), ('u', 0, (0.09806607900511372, 0.6087747244522421, 1.0134836678818133)), ('cz', 3, 2), ('u', 3, (2.5411515937880274, 2.070356994783111, 0.004172307630795125)), ('u', 2, (-0.43614010449775026, 1.397947603639385, -0.3403222040372)), ('u', 7, (-2.9865958698364845, 0.022585922142579373, 2.6479809788314825)), ('u', 1, (-1.7708976029458179, 1.5767573848468537, 1.0113815559405186)), ('u', 0, (2.081989898519472, 1.78915173207941, -1.7763334126680412)), ('cz', 1, 0), ('u', 3, (0.200114952140458, 0.23801058572618608, 0.38802668476110397)), ('u', 2, (-3.403426308712772, 0.5983450085064305, 2.7416840598397942)), ('u', 7, (0.370147200546696, 0.2863852341889236, 0.8553200698689292)), ('u', 1, (1.99367864708359, 0.6185023678637499, -1.5377647673237942)), ('u', 0, (2.0945541847977083, 0.6154797183900051, -1.6181223319858802)), ('cz', 1, 0), ('u', 3, (3.0225188973485375, 0.33980460339632457, -3.1942827347279366)), ('u', 2, (2.908460845636855, 1.788529110679806, 1.0458870914226317)), ('u', 7, (-0.5566969562654057, 0.12065418147763163, 0.5761517344939163)), ('u', 1, (2.272652938329278, 2.2497998576543905, -0.17164199950911319)), ('u', 0, (-0.8132156765144153, 1.286884569753828, 1.954955932162643)), ('cz', 2, 1), ('u', 3, (3.4756460073539834, 0.7671276329455101, -2.7413935139021897)), ('u', 2, (3.8615107972244402, 1.5707963156913627, -1.1462191953360066)), ('u', 7, (2.5838542391729247, 0.09476825405531618, -3.344643902779925)), ('u', 1, (1.1351466721560337, 0.9553166000479915, -1.3468939314589248)), ('u', 0, (0.12449216004804736, 0.06041407986919991, 0.5463272926004652)), ('cz', 2, 1), ('u', 3, (2.9487294693336192, 0.20656136724811683, -3.1522284666004605)), ('u', 2, (2.7238893579338144, 1.2645000680574323, -0.0010032797489376577)), ('u', 7, (2.840595257658502, 0.42664993018723324, -2.869969576330104)), ('u', 1, (-1.6423778391830317, 1.5707962923311725, -0.7817027492867199)), ('u', 0, (-0.04584132279425973, 0.9891115437703512, -0.43347269804319366)), ('cz', 1, 0), ('u', 3, (-0.33353164980451383, 0.3778770017806244, -0.32176578169817877)), ('u', 2, (-1.8943642451851161, 1.0336000024491467, 3.770440671799113)), ('u', 7, (0.06280046937907335, 0.19132150331072284, -0.5823857950392706)), ('u', 1, (-0.7890937151215436, 0.7853981666124227, -0.26129160392022577)), ('u', 0, (2.2487105993192125, 1.570796405161783, -2.6885381644534028)), ('cz', 1, 0), ('u', 3, (2.8047752638780143, 0.0038477722018873685, -3.247686687003211)), ('u', 2, (3.0859650167633217, 0.6241101156589471, -3.085616678830329)), ('u', 7, (-2.8848664924229377, 0.17511456838923553, 2.654796525999915)), ('u', 1, (-2.880301048039854, 1.5707963283207336, 2.2654738534785963)), ('u', 0, (-2.0238507562988173, 1.5707963070424775, 2.265473890438774))]), (8, [('u', 3, (0.0, 2.1790419057051063, 0.0)), ('u', 2, (0.0, 1.0213290820372691, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.0213290820372691, 3.141592653589793)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 7), ('u', 7, (1.9364614740197674, 0.3995472174128471, -0.408778391043077)), ('u', 6, (0.4970631468127439, 2.8946905580841084e-12, -1.5005067128754823)), ('u', 5, (0.0002449918799918571, 0.7853981633974442, 0.14927729107622487)), ('cz', 6, 5), ('u', 7, (-0.40877839104307756, 0.29611018684241097, 0.5906329498660204)), ('u', 6, (1.641085940714299, 1.5707963267928753, -3.534996487819914)), ('u', 5, (3.290869944666034, 0.9263657552645578, -2.6747629075099306)), ('cz', 7, 6), ('u', 7, (-0.45678950456718925, 0.3862279409890685, -0.5209146414558576)), ('u', 6, (-2.748188819359843, 2.6179938779913856, 1.5707963267949048)), ('u', 5, (0.4503821596144114, 0.2650723925833735, 0.34202380955999967)), ('cz', 6, 5), ('u', 7, (2.620678012133932, 1.2160822408932725, -0.19248102329695715)), ('u', 6, (-1.570796326794875, 1.5707963267950042, 1.0391687510491465e-13)), ('u', 5, (3.483616463149799, 0.13205510060954728, -2.790922164413332)), ('cz', 7, 6), ('u', 4, (1.49256647844945, 0.6154796918608355, -1.8844929670572883)), ('u', 5, (-3.4922631182103436, 3.00953755305233, 0.04060685949754905)), ('u', 6, (0.8929093440695672, 1.6746171378896129, -0.8480474977555372)), ('cz', 4, 5), ('u', 4, (-1.8661064027322336, 0.5855771439992585, 0.9447317300849666)), ('u', 5, (2.8383102399215123, 1.5707963228286488, -1.208114548790991)), ('u', 6, (3.191576520277917, 0.8831819477647298, 0.332112845138266)), ('cz', 5, 6), ('u', 4, (2.5690938753165358, 1.1387393304331621, -2.319003702245468)), ('u', 5, (2.778910873138539, 1.5707963379414593, -2.4279982424653257)), ('u', 6, (0.3141930947397381, 0.3488528338498121, 0.7468902940538816)), ('cz', 4, 5), ('u', 4, (-2.380244555886975, 0.2490873873079936, 3.580992716413699)), ('u', 5, (0.24172220724633975, 1.5707963295175347, 0.9321949381804698)), ('u', 6, (3.9966154250803028, 0.8979612823653983, 0.4082511888815892)), ('cz', 5, 6), ('u', 4, (0.17555151328320606, 0.9799303489539507, 0.014303856419479288)), ('u', 5, (0.6386013883004471, 1.5707963200948742, 0.10528611287434297)), ('u', 6, (3.298804248697997, 2.043171274968978, 0.5241489165152349)), ('cz', 4, 5), ('u', 4, (3.329802219944857, 0.7853981620109924, -2.867304092250632)), ('u', 5, (0.09570264480656124, 1.5707963320520875, 0.02166701053358127)), ('u', 6, (-3.3915051572212875, 0.516402974394895, 2.773526478019853)), ('cz', 4, 5), ('u', 4, (2.9009472042826023, 1.2392423742256807e-09, -2.9332453251897133)), ('u', 5, (3.1199256405272573, 1.5707963166739496, -2.899602255211545)), ('u', 6, (0.1400655483674396, 0.02759067538688454, 0.4273349636324735)), ('u', 3, (2.0233960134963267, 1.551684614316178, -3.7235890685381516)), ('u', 2, (0.12588726673913567, 2.7749175075435377e-08, 0.030011978062980678)), ('u', 7, (2.6373121963296677, 1.7319304813933931, -0.47863191543048367)), ('u', 1, (0.839139008888907, 2.3561944756657436, 0.2504183418438984)), ('u', 0, (-0.4857212906342475, 0.36627051757076884, 0.7052590757840566)), ('cz', 2, 1), ('u', 3, (-0.9925699444130445, 1.3760589052021495, 0.030269165866424308)), ('u', 2, (0.10192704701523331, 1.5707963660171047, 0.9817870956243782)), ('u', 7, (-3.05783079748452, 0.44702950751519543, 3.0591122753422684)), ('u', 1, (-2.401632134400665, 2.011001083856294, 2.7586445799647334)), ('u', 0, (-2.8349759805099564, 0.11167287121304673, 2.4995603415012106)), ('cz', 3, 2), ('u', 3, (2.6528659076669325, 1.4955652126472454, -0.581422947523186)), ('u', 2, (0.5890092258714162, 1.5707964336192224, 0.9543365451447087)), ('u', 7, (-2.4564367081972427, 0.001145535296781138, 2.9803634148299323)), ('u', 1, (2.629943460823213, 0.8604362142849246, -1.384439748694525)), ('u', 0, (-2.7696027145682147, 0.5293282705771406, 2.868909903382035)), ('cz', 2, 1), ('u', 3, (2.7543724602965853, 1.9386575875706562, -0.09457060820058993)), ('u', 2, (1.1400585522478939, 1.5707963842675132, -0.24709348408042153)), ('u', 7, (-0.03540724420330254, 0.36887871489577734, -0.02926433557526944)), ('u', 1, (-2.7464252866341696, 0.1488361173078092, 2.928654164947201)), ('u', 0, (0.09806607900511372, 0.6087747244522421, 1.0134836678818133)), ('cz', 3, 2), ('u', 3, (2.5411515937880274, 2.070356994783111, 0.004172307630795125)), ('u', 2, (-0.43614010449775026, 1.397947603639385, -0.3403222040372)), ('u', 7, (-2.9865958698364845, 0.022585922142579373, 2.6479809788314825)), ('u', 1, (-1.7708976029458179, 1.5767573848468537, 1.0113815559405186)), ('u', 0, (2.081989898519472, 1.78915173207941, -1.7763334126680412)), ('cz', 1, 0), ('u', 3, (0.200114952140458, 0.23801058572618608, 0.38802668476110397)), ('u', 2, (-3.403426308712772, 0.5983450085064305, 2.7416840598397942)), ('u', 7, (0.370147200546696, 0.2863852341889236, 0.8553200698689292)), ('u', 1, (1.99367864708359, 0.6185023678637499, -1.5377647673237942)), ('u', 0, (2.0945541847977083, 0.6154797183900051, -1.6181223319858802)), ('cz', 1, 0), ('u', 3, (3.0225188973485375, 0.33980460339632457, -3.1942827347279366)), ('u', 2, (2.908460845636855, 1.788529110679806, 1.0458870914226317)), ('u', 7, (-0.5566969562654057, 0.12065418147763163, 0.5761517344939163)), ('u', 1, (2.272652938329278, 2.2497998576543905, -0.17164199950911319)), ('u', 0, (-0.8132156765144153, 1.286884569753828, 1.954955932162643)), ('cz', 2, 1), ('u', 3, (3.4756460073539834, 0.7671276329455101, -2.7413935139021897)), ('u', 2, (3.8615107972244402, 1.5707963156913627, -1.1462191953360066)), ('u', 7, (2.5838542391729247, 0.09476825405531618, -3.344643902779925)), ('u', 1, (1.1351466721560337, 0.9553166000479915, -1.3468939314589248)), ('u', 0, (0.12449216004804736, 0.06041407986919991, 0.5463272926004652)), ('cz', 2, 1), ('u', 3, (2.9487294693336192, 0.20656136724811683, -3.1522284666004605)), ('u', 2, (2.7238893579338144, 1.2645000680574323, -0.0010032797489376577)), ('u', 7, (2.840595257658502, 0.42664993018723324, -2.869969576330104)), ('u', 1, (-1.6423778391830317, 1.5707962923311725, -0.7817027492867199)), ('u', 0, (-0.04584132279425973, 0.9891115437703512, -0.43347269804319366)), ('cz', 1, 0), ('u', 3, (-0.33353164980451383, 0.3778770017806244, -0.32176578169817877)), ('u', 2, (-1.8943642451851161, 1.0336000024491467, 3.770440671799113)), ('u', 7, (0.06280046937907335, 0.19132150331072284, -0.5823857950392706)), ('u', 1, (-0.7890937151215436, 0.7853981666124227, -0.26129160392022577)), ('u', 0, (2.2487105993192125, 1.570796405161783, -2.6885381644534028)), ('cz', 1, 0), ('u', 3, (2.8047752638780143, 0.0038477722018873685, -3.247686687003211)), ('u', 2, (3.0859650167633217, 0.6241101156589471, -3.085616678830329)), ('u', 7, (-2.8848664924229377, 0.17511456838923553, 2.654796525999915)), ('u', 1, (-2.880301048039854, 1.5707963283207336, 2.2654738534785963)), ('u', 0, (-2.0238507562988173, 1.5707963070424775, 2.265473890438774))])], (8, 3): [(8, [('u', 3, (0.0, 2.6004931276326473, 0.0)), ('u', 2, (0.0, 0.7468985930690364, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 0.7468985930690364, 3.141592653589793)), ('u', 1, (0.0, 1.183199640139716, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.183199640139716, 3.141592653589793)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 7), ('u', 4, (3.3728209205811615, 0.39106167710400747, -2.8687717055019393)), ('u', 5, (-0.267976535731745, 1.3346243562138432e-07, 0.8176961999754309)), ('u', 6, (0.4187445910782187, 1.5707965062407465, -0.08319310590049234)), ('u', 7, (3.222557639729355, 1.5974818362769578, 0.1385678699936217)), ('cz', 5, 6), ('u', 4, (-0.16412447805231375, 0.09470900943880617, 0.5194297750567337)), ('u', 5, (-2.899229607793419, 2.3561944831090047, 2.8887968053752866)), ('u', 6, (3.224785594544601, 1.5707963204610151, 0.6050629969762096)), ('u', 7, (0.3138636053409246, 0.11902835982758336, -0.40969146823104996)), ('cz', 5, 6), ('u', 4, (-3.391915998449925, 0.1158294466557484, 2.5286218601502957)), ('u', 5, (-0.027236169388895837, 1.079027130972998, -0.24536614290042535)), ('u', 6, (-0.4183939664384144, 1.5707961904868029, -3.6094495596543448)), ('u', 7, (3.705920082451782, 1.7564319375656394, -0.17092894557287108)), ('cz', 6, 7), ('u', 4, (2.577118808011268, 0.30839765583209766, -3.2767384268107502)), ('u', 5, (0.4494697052401245, 1.3024393647865542, 0.6675843672387018)), ('u', 6, (2.0386529421286115, 1.570796209493473, -2.200135053185118)), ('u', 7, (0.29050724618378365, 0.25784242113455164, -0.02500257415237106)), ('cz', 5, 6), ('u', 4, (2.7251109785560708, 0.15695392791937654, -3.4302069182176824)), ('u', 5, (-0.061131030905422476, 0.35545862486106106, -0.28617913977166287)), ('u', 6, (1.1529375052313986, 1.5707961932582082, -0.2660537828520958)), ('u', 7, (3.029563323818107, 1.3023766007358066, -0.03784718681305499)), ('cz', 6, 7), ('u', 4, (2.023165182622548, 1.7155532643163143, -1.5341110023713176)), ('u', 5, (-1.6048632094672302, 1.2603414399435735, 1.0688840184512896)), ('u', 6, (-0.5928528095969762, 1.0498745444589463, -0.5844455246568968)), ('u', 7, (3.388646972626647, 1.8617076689373708, 0.4354472908986371)), ('cz', 4, 5), ('u', 4, (1.1816979073348377, 1.5707963789697903, -1.6682862462773675)), ('u', 5, (1.458072891279728, 1.570796386836844, -0.12583102055826956)), ('u', 6, (-0.7194021009941884, 1.2450507217702333, -0.013951281373488833)), ('u', 7, (-2.325339774940423, 0.11129343719290953, 2.7753564553207064)), ('cz', 4, 5), ('u', 4, (1.64271682518899, 3.073442043114022, 4.538690545281303)), ('u', 5, (3.1941662040641745, 1.724911806660808, -1.9410001509872072)), ('u', 6, (2.698541789712482, 0.4479972486841587, -0.6627965288374593)), ('u', 7, (-2.993942680108106, 0.23521611280486165, 3.097752498315438)), ('cz', 5, 6), ('u', 4, (0.11728795446952442, 0.6778994045979737, -0.2951661800459031)), ('u', 5, (-3.217350156074537, 0.6226665641915875, 2.584004647876074)), ('u', 6, (2.5593474395550335, 3.019683074405999, 0.3386062766998035)), ('u', 7, (-0.4266726648541127, 0.09281550326566909, 0.034186827855751084)), ('cz', 4, 5), ('u', 4, (2.5219147368006984, 1.2004686024392561, -3.1671833488467156)), ('u', 5, (-2.584004676052905, 1.570796371905883, 2.6318988603455256)), ('u', 6, (0.37457047092306783, 1.663190656482429, 0.5335023393030396)), ('u', 7, (0.07100987203302439, 0.041022120614617595, -0.3616200218093768)), ('cz', 5, 6), ('u', 4, (-2.7001331184400894, 0.24822547857918786, 3.182386433519112)), ('u', 5, (-0.2808515276680261, 1.5707963220356849, 0.3676041932232932)), ('u', 6, (-2.910272582160462, 0.1449916061870487, 3.3087391888891764)), ('u', 7, (3.0001224361099816, 0.367051690636943, -3.1198527489791386)), ('cz', 4, 5), ('u', 4, (-0.5966271584471116, 0.507432086302431, -0.11837694143314106)), ('u', 5, (1.2031920948608357, 1.5707962973025824, 0.14638436427251433)), ('u', 6, (3.865227906681971, 1.4792373391097866, 0.14001516136296965)), ('u', 7, (2.8617752403302896, 0.3403339271803459, -3.247823365181908)), ('cz', 5, 6), ('u', 2, (0.17019482054623225, 1.5707963164237424, 0.22167593381239736)), ('u', 1, (0.17019485468649442, 1.5707963021933358, 4.616040168497951)), ('u', 0, (0.0022605489924552025, 0.04447500135352692, -1.3393501587254977)), ('u', 3, (-0.8298645381396916, 0.11094846618151442, -0.5972517157377744)), ('u', 6, (2.2120238062657993, 1.9637734623613083, -0.249278751266526)), ('cz', 2, 1), ('u', 2, (-0.22167592189386123, 0.7853981645675656, 0.21898891430049736)), ('u', 1, (3.4761012864725442, 1.339117622575057, -2.7786848898724745)), ('u', 0, (-3.5817562877890836, 0.0531638441594182, 2.5006606863981418)), ('u', 3, (-3.6712572805973256, 0.6313327419861082, 2.299248835676188)), ('u', 6, (-0.48352378709204075, 0.14751049185047072, -0.5142732815555632)), ('cz', 2, 1), ('u', 2, (-0.831132653929176, 3.1415926469123683, -1.8307981270898233)), ('u', 1, (-3.0548966138432574, 1.703265840356079, 3.1790026129297555)), ('u', 0, (-1.1953518813256916, 0.8135945097755325, -2.2144782903357014)), ('u', 3, (2.846945313107482, 1.201742192282055, 0.24909406789456812)), ('u', 6, (-3.0178589908010416, 0.41418460790010425, 2.7033554245375715)), ('cz', 2, 3), ('u', 2, (-1.9229382570990945, 1.5707963290416478, 3.9876628047495686)), ('u', 1, (-1.2146135345598759, 0.3344005973948035, -0.2075894528765152)), ('u', 0, (2.8311582662723764, 0.6607943247008267, -2.9227888201568075)), ('u', 3, (3.3682524746099234, 1.0570166230980986, -0.06205178028258995)), ('u', 6, (-0.4079146951896172, 0.11307885034027033, 0.36102093499718313)), ('cz', 2, 1), ('u', 2, (1.2727790177339775, 1.8558570651771735, -0.4158153352822194)), ('u', 1, (0.949875237522008, 0.6871912870686274, 1.05491839076666)), ('u', 0, (3.7796322374143703, 0.2348284227396344, -2.4546769113537206)), ('u', 3, (-2.8265724101130063, 0.01014374652125673, 2.869560565339029)), ('u', 6, (3.4099064525789466, 0.15348046900051315, -2.5429653211922267)), ('cz', 2, 3), ('u', 2, (-3.723377953773289, 0.5488934566153154, 2.1188625818399984)), ('u', 1, (-2.1981442303549708, 0.3359809558076345, 3.7118480301551884)), ('u', 0, (0.2601192024789832, 0.7567804762796897, -0.653628005679572)), ('u', 3, (-3.051596086358009, 0.646999330693771, 2.751925526860514)), ('u', 6, (-2.850494657567627, 0.31387073000199656, 3.295942863131725)), ('cz', 1, 0), ('u', 2, (-0.27539592757718384, 1.3990719693955174, -0.9872357964566447)), ('u', 1, (1.6516313176946151, 2.3868530285797958, -1.1318144059868365)), ('u', 0, (1.5799187073848207, 1.5707963204526236, -0.9098736884358005)), ('u', 3, (-0.017585970514645693, 0.05100395494315851, 0.4278381579929925)), ('u', 6, (2.677794619512114, 0.23454084676659628, -2.6973212261886648)), ('cz', 1, 0), ('u', 2, (1.83508579342028, 1.0489196739086213, -0.46860092122164476)), ('u', 1, (1.617433129108679, 1.5707963231186162, -0.9841912789155298)), ('u', 0, (-0.6883037262441306, 1.8730241766776343, 1.114845018021535)), ('u', 3, (2.9080536897331903, 0.7424522884930189, -2.740764599761599)), ('u', 6, (3.464127252804771, 0.1676096031474168, -2.6333572906808755)), ('cz', 2, 1), ('u', 2, (4.093737716561863, 1.9146357703355779, 0.4447267859329944)), ('u', 1, (-2.157401373465824, 2.427378425544589, 2.992414488474941)), ('u', 0, (-0.1502087048392144, 0.857195019281663, 0.36540691589954716)), ('u', 3, (0.11480280359866452, 0.05493838074684792, 0.041683955656111576)), ('u', 6, (0.37091179090473403, 0.04141032392649731, 0.25964304423216455)), ('cz', 1, 0), ('u', 2, (1.2908280750009269, 1.1412600605504781, -0.6447271282888243)), ('u', 1, (0.1491781757101821, 1.5707963131232534, 0.7542296017172242)), ('u', 0, (0.8467628977387938, 0.6157647454113345, -0.29417591386695435)), ('u', 3, (2.851518341415716, 0.3231521525718433, -3.2220833280099668)), ('u', 6, (3.0486628701566914, 0.11703664480778077, -3.1755722016658248)), ('cz', 2, 1), ('u', 2, (3.0242614699725854, 0.06841861725852429, -2.7333397051135018)), ('u', 1, (-3.026197618562068, 1.570796315558012, 2.3464466578212173)), ('u', 0, (-0.4142166030401052, 0.5429224848251636, 0.5009461149031439)), ('u', 3, (-3.1952693412540296, 0.16387128079898583, 2.8757264082391285)), ('u', 6, (0.12236014889985145, 0.07368483489706198, -0.47480261029214943)), ('cz', 1, 0), ('u', 2, (3.116353052237453, 1.5073213088663675, -0.10665115514559842)), ('u', 1, (-0.775650340619295, 1.570796314882542, -0.5489102592853109)), ('u', 0, (-3.1315365664496158, 1.6080089822094911e-09, 2.2169944770832615)), ('u', 3, (0.07988145091868482, 0.12412243747944651, 0.05485126932871501)), ('u', 6, (-0.2668207252199987, 0.21757742495350252, -0.37446882952067745)), ('cz', 2, 1), ('u', 2, (3.222691765488767, 1.570796326840022, 0.7040609460816769)), ('u', 1, (0.5489102714949137, 1.5707963415457094, 0.13850967320827018)), ('u', 0, (-1.033353067503298, 5.344050763992638e-09, 0.45510447918870073)), ('u', 3, (-0.07919187134436817, 0.0737716818862282, 0.04330619866210563)), ('u', 6, (-0.004550713039591592, 0.23691724900672498, -0.300794685100584)), ('u', 7, (0.0032436760622015504, 0.6834998443485418, -0.37543550660464076)), ('u', 7, (0.17790794256295062, 0.2778823884636485, 0.06771807429415676)), ('u', 7, (0.03844868731066941, 0.08811718482662821, -0.05371352853264836)), ('u', 7, (3.087879124972279, 0.16187704781250048, -3.1863285152784178)), ('u', 7, (-3.1863285153525838, 0.44829498318534017, 3.047077383201628)), ('u', 4, (0.11837684645650276, 2.634160693045651, 1.050387936672795)), ('u', 5, (-0.14638444338338535, 1.5707962775860798, -0.35482118268032997)), ('u', 7, (-0.5167990477960749, 0.15328110898445052, 0.03839322151376151)), ('u', 7, (-0.2924525804492283, 0.01042098486691196, 0.23685163099643003)), ('u', 7, (-2.6898985381570064, 0.5497250222820544, 3.199934063053756)), ('u', 7, (0.007825319632610211, 0.25680036324009115, -0.2604650333637528)), ('u', 7, (-0.09248169206998845, 0.12472079356709315, 0.10177411824068093)), ('u', 7, (0.1525375994889486, 0.3687417995323103, 0.010712472004470425)), ('u', 7, (2.773136649248701, 0.08129114514751729, -2.9820161047644085)), ('u', 7, (2.65924046167778, 0.07459776920251235, -2.830238065771656)), ('u', 7, (-2.8575393144935695, 0.21072989668142014, 3.326969708625929))]), (8, [('u', 3, (0.0, 2.6004931276326473, 0.0)), ('u', 2, (0.0, 0.7468985930690364, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 0.7468985930690364, 3.141592653589793)), ('u', 1, (0.0, 1.183199640139716, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.183199640139716, 3.141592653589793)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 7), ('u', 4, (3.3728209205811615, 0.39106167710400747, -2.8687717055019393)), ('u', 5, (-0.267976535731745, 1.3346243562138432e-07, 0.8176961999754309)), ('u', 6, (0.4187445910782187, 1.5707965062407465, -0.08319310590049234)), ('u', 7, (3.222557639729355, 1.5974818362769578, 0.1385678699936217)), ('cz', 5, 6), ('u', 4, (-0.16412447805231375, 0.09470900943880617, 0.5194297750567337)), ('u', 5, (-2.899229607793419, 2.3561944831090047, 2.8887968053752866)), ('u', 6, (3.224785594544601, 1.5707963204610151, 0.6050629969762096)), ('u', 7, (0.3138636053409246, 0.11902835982758336, -0.40969146823104996)), ('cz', 5, 6), ('u', 4, (-3.391915998449925, 0.1158294466557484, 2.5286218601502957)), ('u', 5, (-0.027236169388895837, 1.079027130972998, -0.24536614290042535)), ('u', 6, (-0.4183939664384144, 1.5707961904868029, -3.6094495596543448)), ('u', 7, (3.705920082451782, 1.7564319375656394, -0.17092894557287108)), ('cz', 6, 7), ('u', 4, (2.577118808011268, 0.30839765583209766, -3.2767384268107502)), ('u', 5, (0.4494697052401245, 1.3024393647865542, 0.6675843672387018)), ('u', 6, (2.0386529421286115, 1.570796209493473, -2.200135053185118)), ('u', 7, (0.29050724618378365, 0.25784242113455164, -0.02500257415237106)), ('cz', 5, 6), ('u', 4, (2.7251109785560708, 0.15695392791937654, -3.4302069182176824)), ('u', 5, (-0.061131030905422476, 0.35545862486106106, -0.28617913977166287)), ('u', 6, (1.1529375052313986, 1.5707961932582082, -0.2660537828520958)), ('u', 7, (3.029563323818107, 1.3023766007358066, -0.03784718681305499)), ('cz', 6, 7), ('u', 4, (2.023165182622548, 1.7155532643163143, -1.5341110023713176)), ('u', 5, (-1.6048632094672302, 1.2603414399435735, 1.0688840184512896)), ('u', 6, (-0.5928528095969762, 1.0498745444589463, -0.5844455246568968)), ('u', 7, (3.388646972626647, 1.8617076689373708, 0.4354472908986371)), ('cz', 4, 5), ('u', 4, (1.1816979073348377, 1.5707963789697903, -1.6682862462773675)), ('u', 5, (1.458072891279728, 1.570796386836844, -0.12583102055826956)), ('u', 6, (-0.7194021009941884, 1.2450507217702333, -0.013951281373488833)), ('u', 7, (-2.325339774940423, 0.11129343719290953, 2.7753564553207064)), ('cz', 4, 5), ('u', 4, (1.64271682518899, 3.073442043114022, 4.538690545281303)), ('u', 5, (3.1941662040641745, 1.724911806660808, -1.9410001509872072)), ('u', 6, (2.698541789712482, 0.4479972486841587, -0.6627965288374593)), ('u', 7, (-2.993942680108106, 0.23521611280486165, 3.097752498315438)), ('cz', 5, 6), ('u', 4, (0.11728795446952442, 0.6778994045979737, -0.2951661800459031)), ('u', 5, (-3.217350156074537, 0.6226665641915875, 2.584004647876074)), ('u', 6, (2.5593474395550335, 3.019683074405999, 0.3386062766998035)), ('u', 7, (-0.4266726648541127, 0.09281550326566909, 0.034186827855751084)), ('cz', 4, 5), ('u', 4, (2.5219147368006984, 1.2004686024392561, -3.1671833488467156)), ('u', 5, (-2.584004676052905, 1.570796371905883, 2.6318988603455256)), ('u', 6, (0.37457047092306783, 1.663190656482429, 0.5335023393030396)), ('u', 7, (0.07100987203302439, 0.041022120614617595, -0.3616200218093768)), ('cz', 5, 6), ('u', 4, (-2.7001331184400894, 0.24822547857918786, 3.182386433519112)), ('u', 5, (-0.2808515276680261, 1.5707963220356849, 0.3676041932232932)), ('u', 6, (-2.910272582160462, 0.1449916061870487, 3.3087391888891764)), ('u', 7, (3.0001224361099816, 0.367051690636943, -3.1198527489791386)), ('cz', 4, 5), ('u', 4, (-0.5966271584471116, 0.507432086302431, -0.11837694143314106)), ('u', 5, (1.2031920948608357, 1.5707962973025824, 0.14638436427251433)), ('u', 6, (3.865227906681971, 1.4792373391097866, 0.14001516136296965)), ('u', 7, (2.8617752403302896, 0.3403339271803459, -3.247823365181908)), ('cz', 5, 6), ('u', 7, (0.0032436760622015504, 0.6834998443485418, -0.37543550660464076)), ('u', 1, (0.0, 3.1415926535890537, 3.88744303690369)), ('u', 6, (2.751834112798575, 2.1012039094847754, -0.1517714776595427)), ('u', 2, (-0.29549666251093637, 0.7050390407397258, -0.0504640633361539)), ('u', 0, (-8.823988715167541e-05, 0.5235987755986036, -0.2713082920613179)), ('cz', 1, 0), ('u', 7, (0.17790794256295062, 0.2778823884636485, 0.06771807429415676)), ('u', 1, (-0.3733786938654473, 1.5707963267944591, -0.3926991012583947)), ('u', 6, (0.05980748153639656, 0.3963037782683509, -0.006982201558379128)), ('u', 2, (-0.05046390721346742, 0.8682743217517934, 0.0020060945442915423)), ('u', 0, (-0.27130828125150197, 0.022777442641511807, 0.5193627464818114)), ('cz', 1, 2), ('u', 7, (0.03844868731066941, 0.08811718482662821, -0.05371352853264836)), ('u', 1, (-2.748893552328643, 0.7853981633375778, 3.1415926763397994)), ('u', 6, (-3.1646736998580014, 0.008360673113919317, 3.1112138433362153)), ('u', 2, (-0.010116346277316357, 0.08552576244548307, 0.04031331707155163)), ('u', 0, (0.044455988032640906, 0.5042175350645443, -0.19952287723631784)), ('cz', 1, 0), ('u', 7, (3.087879124972279, 0.16187704781250048, -3.1863285152784178)), ('u', 1, (-2.2751715630221348e-08, 1.57079632673552, 3.817196586803185e-08)), ('u', 6, (3.111213843333554, 0.3129850431921355, -3.156571838238244)), ('u', 2, (3.1819062965079126, 2.441467427658711, 0.004864965594580273)), ('u', 0, (0.0, 2.6513916189822675e-13, -0.9494262989530413)), ('cz', 1, 2), ('u', 7, (-3.1863285153525838, 0.44829498318534017, 3.047077383201628)), ('u', 1, (3.1415926154163, 1.5707963267948102, -2.769120964141354)), ('u', 6, (-3.1565718382321757, 0.6175309700502564, 3.0883100452959544)), ('u', 2, (3.1464573749932385, 1.570796326794898, 0.3724716894484392)), ('u', 0, (0.0, 9.751213919706676e-14, -1.6445174275305499)), ('u', 4, (0.11837684645650276, 2.634160693045651, 1.050387936672795)), ('u', 5, (-0.14638444338338535, 1.5707962775860798, -0.35482118268032997)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.3089969389957472, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 0.26179938779914935, 0.0)), ('cz', 3, 2), ('u', 2, (-3.141592653589793, 0.26179938779914935, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 1.3089969389957472, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 1, (-1.6785921893219244, 1.570796326932515, 1.6831464583629798)), ('u', 0, (1.4630004612841487, 1.5707963257673239, -1.6507071894784535)), ('u', 7, (-0.5167990477960749, 0.15328110898445052, 0.03839322151376151)), ('u', 2, (-3.1224273432308918, 0.41587665949883285, 2.939095964828157)), ('cz', 1, 0), ('u', 1, (1.4584461959307882, 0.35087940973568627, -0.7353608165593173)), ('u', 0, (1.6507071889773584, 2.2027106391988873, -1.0129019142750488)), ('u', 7, (-0.2924525804492283, 0.01042098486691196, 0.23685163099643003)), ('u', 2, (-2.8993921940537586, 0.11401683106714594, 3.2404143397766507)), ('cz', 1, 0), ('u', 1, (2.306157142595466, 1.5707963256255622, -1.6180373583861483)), ('u', 0, (-0.8987723413484447, 1.6185336315711922, 1.2736150288731265)), ('u', 7, (-2.6898985381570064, 0.5497250222820544, 3.199934063053756)), ('u', 2, (2.8538763397046765, 1.0463658462712422, -0.22826259701931817)), ('cz', 1, 2), ('u', 1, (-3.0943516163981646, 2.479447418865614, 3.0364459711810023)), ('u', 0, (0.43090293811141517, 0.34407505220565543, -0.09433502182989134)), ('u', 7, (0.007825319632610211, 0.25680036324009115, -0.2604650333637528)), ('u', 2, (3.386076292433321, 1.4467275743010095, -0.2690597066109728)), ('cz', 1, 0), ('u', 1, (0.10514668426339173, 1.5707963256835809, 0.13713489223040934)), ('u', 0, (-2.7763584089935467, 0.2296184329001961, 2.4227388217069032)), ('u', 7, (-0.09248169206998845, 0.12472079356709315, 0.10177411824068093)), ('u', 2, (0.45307542307963095, 0.342763381940028, -0.3672684148838083)), ('cz', 1, 2), ('u', 1, (3.0044577596882385, 0.6319143128224841, -3.1236522678810394)), ('u', 0, (2.882192858058798, 0.16702240300304355, -2.6640807431236553)), ('u', 7, (0.1525375994889486, 0.3687417995323103, 0.010712472004470425)), ('u', 2, (3.288950206193982, 0.21976111948157684, -2.8800839393473523)), ('cz', 1, 0), ('u', 1, (-0.01794038353676515, 1.5707963253622785, -0.2665683448864032)), ('u', 0, (3.022013975396951, 0.9319708637721523, -3.180375462917911)), ('u', 7, (2.773136649248701, 0.08129114514751729, -2.9820161047644085)), ('u', 2, (2.9139555600883, 1.3511572433220294, -0.1296179801412065)), ('cz', 1, 2), ('u', 1, (0.26656833931441565, 2.0943951027946675, -0.1617002129103472)), ('u', 0, (-0.2286978311965311, 0.43284174135734543, 0.06931291087109169)), ('u', 7, (2.65924046167778, 0.07459776920251235, -2.830238065771656)), ('u', 2, (3.2400301606384145, 1.4131105162038662, 0.06800845002444533)), ('cz', 1, 0), ('u', 1, (-2.9798924454688773, 1.5707963272351246, 3.2493885166068024)), ('u', 0, (-2.772648015074517, 1.57079632611796, 3.249388518729874)), ('u', 7, (-2.8575393144935695, 0.21072989668142014, 3.326969708625929)), ('u', 2, (0.08242219689306605, 0.15945674688551645, -0.040754500573628616)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0))])], (8, 4): [(8, [('u', 3, (0.0, 2.901974084960706, 0.0)), ('u', 2, (0.0, 0.5024148995576962, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 0.5024148995576962, 3.141592653589793)), ('u', 1, (0.0, 0.9687234065335635, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 0.9687234065335635, 3.141592653589793)), ('u', 0, (0.0, 1.3258176636680323, 0.0)), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 0), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.3258176636680323, 3.141592653589793)), ('u', 4, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 0, 7), ('u', 0, (0.24740256010545428, 1.5707963077782177, 2.347196534357998e-08)), ('u', 1, (3.1242004886093913, 1.5707963371671434, -3.141592665871107)), ('u', 2, (-3.177895552668161, 0.0166128724599478, 3.104620604231258)), ('u', 6, (0.017392164990191028, 0.025281051720125753, 0.01653321709368015)), ('u', 3, (2.9721414483883977, 0.4120439856151028, -2.9410094461947893)), ('u', 4, (0.16945120521196874, 0.0698923834482002, 0.13384585665249996)), ('cz', 0, 1), ('u', 0, (2.347977496886802e-08, 0.5235987762469274, 0.0006192026588912668)), ('u', 1, (-1.2307312022493454e-08, 0.5235987750710543, 4.0957679120355254e-09)), ('u', 2, (-0.03697204936691645, 0.00048686681607281563, -0.03589569340259112)), ('u', 6, (0.01653321712495465, 0.018805607579500085, 0.01511671621412466)), ('u', 3, (3.3421758611358854, 0.29943193629874465, -2.801280139749614)), ('u', 4, (0.13384585669619314, 0.06445488851009958, 0.09366300171440965)), ('cz', 0, 1), ('u', 0, (3.1422118562561323, 1.5765068190908538, -3.03480848239328)), ('u', 1, (4.0618448639284654e-09, 1.570796349677758, -0.038233242362414155)), ('u', 2, (3.1056969602348747, 1.554845145696804, 0.0012595738688832014)), ('u', 6, (0.015116716172206337, 0.013269990713690632, 0.013296369549273422)), ('u', 3, (-2.801280139660277, 0.1072229900875749, 3.4654009918427096)), ('u', 4, (0.09366300138962583, 0.004386940699479065, 0.06148599970898438)), ('cz', 1, 2), ('u', 0, (0.10678417118613182, 0.005843266668250025, 0.10474113038325222)), ('u', 1, (3.1033594113958802, 1.9659883474831268, -3.014698581466032)), ('u', 2, (3.1428522273951844, 1.5677942961108704, -0.023574434816298062)), ('u', 6, (0.013296369425864006, 0.008680015539861844, 0.011198541354675486)), ('u', 3, (0.3238083383413242, 0.08011083568791795, 0.27039913352409645)), ('u', 4, (0.06148600024379006, 0.0037736151387163202, 0.032378317282689306)), ('cz', 0, 1), ('u', 0, (3.2463337840273927, 7.150014946169529e-09, -3.008359840538833)), ('u', 1, (-3.0146985813499216, 0.40689197767059143, 3.141592654109527)), ('u', 2, (-0.023574434888719034, 0.00300537583725997, 0.029476304993231756)), ('u', 6, (0.011198541314167774, 0.0049966365584722125, 0.008926991461794022)), ('u', 3, (0.2703991333973423, 0.2276636401301653, 0.13420829392291284)), ('u', 4, (0.03237831709066832, 0.00028431420947649667, 0.005699356664471493)), ('cz', 1, 2), ('u', 0, (0.13323281300465886, 3.84756981491908e-09, 0.13825310112295322)), ('u', 1, (4.0711090054657006e-10, 0.4068919782469871, -0.12190029762589617)), ('u', 2, (0.029476305361641867, 0.7835378123971821, 3.452390887059842e-09)), ('u', 6, (0.008926991925730862, 0.002147874588835068, 0.0065658296753449965)), ('u', 3, (0.13420829339400075, 0.3277509783372616, 0.13594225576027558)), ('u', 4, (0.00569935647647184, 0.05443235537030334, -0.02529474470543954)), ('cz', 0, 1), ('u', 0, (3.2798457547072815, 0.04042449991685485, -2.9936036274925577)), ('u', 1, (-0.12190029748491468, 1.5594833340583052, -0.011653884343328196)), ('u', 2, (3.1415926568818042, 0.7835378104262972, 0.03488296439235583)), ('u', 6, (0.006565829258824254, 3.877275334941494e-05, 0.0041817744694165205)), ('u', 3, (0.13594225605744475, 0.40343337433406096, 0.10254334770957577)), ('u', 4, (-0.025294744418642594, 0.0061686684257190125, -0.05784649635865514)), ('cz', 2, 3), ('u', 0, (3.2895816799099253, 0.018841485951206053, -2.983977420578603)), ('u', 1, (-0.011653884296412806, 0.4066214082631142, 0.028392030143334628)), ('u', 2, (-3.1067096893045836, 2.8797148839802347, 3.1295315135530695)), ('u', 6, (-3.1374108789156523, 0.001440328390608002, 3.143418905676553)), ('u', 3, (3.24413600129048, 1.561398986569674, -0.009654370797877121)), ('u', 4, (-0.057846496375142484, 0.0026019334314028924, -0.08978871887270674)), ('cz', 1, 2), ('u', 0, (3.299207886364588, 0.005510219285898618, -2.976744077689273)), ('u', 1, (-3.1132006235541954, 1.6154230131070403e-07, 3.1306276257288856)), ('u', 2, (-0.012061140075379756, 1.8326013431221995, -0.04647442187655725)), ('u', 6, (-3.1397664014354603, 0.002406973196539197, 3.1411301860299656)), ('u', 3, (-0.009654370648902611, 0.0093990921605391, 0.0004294948300030358)), ('u', 4, (-3.231381372704636, 0.0008496024159753081, 3.0226345604417624)), ('cz', 2, 3), ('u', 0, (0.1648485760667228, 0.0021809407580145697, 0.16828362485439727)), ('u', 1, (-0.010965027630111281, 1.637056331664681e-07, -0.01269871139026909)), ('u', 2, (3.0951182316139176, 0.26288855875943395, -3.096706221181777)), ('u', 6, (-3.1420551213177568, 0.0029810570217928167, 3.138935623980257)), ('u', 3, (0.00042949486698879753, 0.012491695224934785, -0.0007960693692592453)), ('u', 4, (-3.260550746593288, 0.004700915511354513, 3.0002669274861367)), ('cz', 1, 2), ('u', 0, (0.16828362481483822, 0.00572269746304183, 0.16728867344561255)), ('u', 1, (-3.154291365115813, 0.018826034010998034, 3.128855482149092)), ('u', 2, (0.044886432450014314, 1.30891855727653, -0.045829524379756785)), ('u', 6, (-3.144249683147679, 0.0032804266365954802, 3.136853971042753)), ('u', 3, (3.1407965841908574, 1.5832880257026303, -0.0017197159331123757)), ('u', 4, (-3.2829183797662975, 0.008354769336868528, 2.9894702050321116)), ('cz', 2, 3), ('u', 0, (1.7380847604433431, 1.5107689911037403, -1.571324882275377)), ('u', 1, (-1.5835338264006893, 1.5703167289765865, 1.5613863603126945)), ('u', 2, (3.0957630142659, 0.584451077565623, -3.145699371051266)), ('u', 6, (-3.146331352457625, 0.0034174669627471403, 3.1348976928597034)), ('u', 3, (3.1398728835465652, 1.5376419799231713, 0.009492169126966488)), ('u', 4, (2.989470308545099, 0.01059604756820988, -3.2894516273286976)), ('cz', 0, 1), ('u', 0, (1.5702677713006499, 0.6154797072258071, -1.562243490887305)), ('u', 1, (1.561386360289, 0.6154797110922112, -1.093384343890502)), ('u', 2, (3.1374860576555013, 0.5234099894947305, -3.1390603303721556)), ('u', 6, (-3.1482874525086646, 0.003496569783050107, 3.1330754973688877)), ('u', 3, (-3.132100495325359, 0.027922222706663613, 3.148858360521322)), ('u', 4, (2.9937335297800525, 0.01041795155640386, -3.2694726600341175)), ('cz', 0, 1), ('u', 0, (-1.5622434908862481, 1.5685524404469555, 1.6056213996369189)), ('u', 1, (2.048208309723095, 1.5334813495880195, -1.5575957781587224)), ('u', 2, (3.1441250348259335, 2.0337419782229196, -0.020710776611550408)), ('u', 6, (-3.15010981015512, 0.003612193970001695, 3.131393821723035)), ('u', 3, (-3.1343268765560137, 0.021481875053809187, 3.1468180449301104)), ('u', 4, (3.0137126291018803, 0.0076370837612826535, -3.235962942998284)), ('cz', 1, 2), ('u', 0, (-3.106767784095532, 0.0019089473699859106, 3.172179534140847)), ('u', 1, (3.1547930906241923, 0.9553166197494333, -3.061391390515589)), ('u', 2, (3.120881877022657, 1.5707963169910486, -0.004226921051762478)), ('u', 6, (-3.1517914854691287, 0.003847534019552029, 3.1298599946401118)), ('u', 3, (-3.1363672621309844, 0.01411764137434865, 3.1449914125570997)), ('u', 4, (3.047222364307676, 0.0032825647198644393, -3.192834676468687)), ('cz', 1, 2), ('u', 0, (3.179329951343629, 0.015347223655525247, -3.0987623973207636)), ('u', 1, (-3.086559748241109, 0.7853981587838752, 3.1415926541008123)), ('u', 2, (-0.005908311286144508, 0.006152121796748064, 0.005067712979498832)), ('u', 6, (-3.153444735118384, 0.007543148283208885, 3.128302296756359)), ('u', 3, (0.003157281073491996, 0.004004813866927766, 0.0017976638359951114)), ('u', 4, (3.119194523485116, 0.0018677697291707613, -3.11628581011227)), ('cz', 0, 1), ('u', 0, (-3.0987623970550167, 0.3191079304856438, 3.1810842006744395)), ('u', 1, (5.962494737083578e-10, 0.7853981697864992, 1.5213844843458757e-09)), ('u', 2, (3.146660366552159, 1.576948127007934, -0.02770539087803514)), ('u', 6, (-3.154883010482205, 0.00851161057330634, 3.127035260463243)), ('u', 3, (0.0017976638470731906, 0.012089232256596767, 0.000718599521156016)), ('u', 4, (3.1668994970224933, 0.005817503998311398, -3.072119650548287)), ('cz', 1, 2), ('u', 0, (0.039491546971138984, 1.2732788594102697, 0.015160252314586625)), ('u', 1, (1.4302492379899154e-09, 0.7853981592693322, 7.428270823840504e-10)), ('u', 2, (3.1138872627906964, 2.080222468738006, -0.010864583275173256)), ('u', 6, (-3.156150046624366, 0.009746482137883213, 3.125964659276624)), ('u', 3, (0.00071859940018132, 0.01990596591083898, -6.178743292940315e-05)), ('u', 4, (3.211065656781986, 0.01417286467163173, -3.0358731670780625)), ('cz', 0, 1), ('u', 0, (-3.1264324012417974, 0.7853981574777166, 3.1333786159945234)), ('u', 1, (9.41110679187132e-10, 1.047197554106395, -0.0021205647076725847)), ('u', 2, (-0.010864583312786566, 0.525026676441514, 0.018992696265370222)), ('u', 6, (-3.1572206479508256, 0.011263392747836276, 3.125124278204042)), ('u', 3, (-6.178743661237085e-05, 0.027173244443369486, -0.0005393802257674917)), ('u', 4, (3.247312140084665, 0.026374950718660223, -3.0104594681177015)), ('cz', 0, 1), ('u', 0, (3.107819054907817, 0.6540555043263074, -3.0993775140200084)), ('u', 1, (3.139472091708805, 1.249202282975233e-08, -3.137351524031184)), ('u', 2, (0.014504636230908797, 0.5041748607112927, 0.05400415949506655)), ('u', 6, (3.115152592641828, 0.011939375919032371, -3.136810400210318)), ('u', 3, (-0.0005393626875548407, 0.03361669691817586, -0.0007268792286183329)), ('u', 4, (0.29901481253552936, 1.5134045451940534, 0.1712135923327947)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (3.2967664143484856, 0.37746088244651743, -2.90770693507335)), ('u', 5, (-3.065351981199511, 1.570796325998965, 3.141592810233495)), ('cz', 5, 4), ('u', 0, (-3.1336817276541074, 0.03988000500708097, 3.1477971068631443)), ('u', 7, (3.3754782677989725, 0.2957003307021647, -2.8697445297881767)), ('u', 6, (3.1332705618836747, 0.000670964769903343, -3.14776404325472)), ('u', 5, (-1.2304880342357098e-07, 0.5235987812541554, -2.9738587354984345e-08)), ('u', 2, (3.1384898734794255, 0.03459206355040011, -3.14347395431741)), ('u', 4, (-3.5769262187636676e-08, 0.523598771695477, -0.00014103518936014022)), ('cz', 5, 4), ('u', 0, (-3.1353881481357257, 0.04020044576188537, 3.1458639800638526)), ('u', 7, (3.4134410998710325, 0.18048471998342222, -2.8492079037556315)), ('u', 6, (3.1354213213872644, 1.5700021680961975, -4.475632626022019e-05)), ('u', 5, (-7.312312765017648e-08, 1.5707963100608628, -0.08150001422869849)), ('u', 2, (3.1397114051832484, 0.035398571332497825, -3.1424570006512282)), ('u', 4, (3.141451865879188, 1.5700162024807447, -2.9682222802448033)), ('cz', 6, 5), ('u', 0, (-3.1373213273631406, 0.03904869406383659, 3.1437810459612163)), ('u', 7, (0.29238516947199533, 0.018297901744561286, 0.2766538192545155)), ('u', 6, (3.141547879107957, 1.5729530881038725, -0.023488310377399735)), ('u', 5, (3.0600922833400723, 1.9638558876725143, -3.092184870708789)), ('u', 2, (3.1407285146219714, 0.034780753328794205, -3.1416728739950286)), ('u', 4, (3.3149630865408977, 0.0008295588521163107, -2.966619969343351)), ('cz', 5, 4), ('u', 0, (-3.1394044172879125, 0.036450418837086045, 3.141636995139753)), ('u', 7, (0.27665329513227915, 0.20639358701078347, 0.2227248088732203)), ('u', 6, (3.1181046088741766, 0.0021592102046690117, -3.1285213458022136)), ('u', 5, (-3.0921852441269735, 0.3947927655786377, 3.1415942468560623)), ('u', 2, (3.1415122251017022, 0.03275584928687877, -3.1411353417589982)), ('u', 4, (-2.966619878409117, 3.4215350387617885e-08, 3.2842899804013013)), ('cz', 6, 5), ('u', 0, (-3.141548051954922, 0.03249920709244061, 3.1395296271679576)), ('u', 7, (0.22272481234282604, 0.3511173537588248, 0.15800993946820546)), ('u', 6, (0.013071320724860377, 0.7902924514564892, -2.1749857445456378e-08)), ('u', 5, (1.7116094205398003e-06, 0.3947928587638951, -0.029447614128055424)), ('u', 2, (3.1420500173930836, 0.02940971648979802, -3.1408404266649796)), ('u', 4, (0.14269720984164003, 2.879100755880473e-08, 0.14726041862431766)), ('cz', 5, 4), ('u', 0, (-3.143655783856359, 0.027345703916849064, 3.1375606772273215)), ('u', 7, (0.15800983204431243, 0.43823920309252273, 0.10504175242947242)), ('u', 6, (3.1415925128035482, 0.7902924864061348, 0.021359803985439463)), ('u', 5, (-0.029447717332347988, 1.5600925697278034, -0.04678493351120711)), ('u', 2, (3.1423449846850295, 0.024885924431284862, -3.140768586890696)), ('u', 4, (0.1472604315872025, 0.010083607399992922, 0.15109805594159728)), ('cz', 7, 6), ('u', 0, (-3.1456246299521164, 0.021185628864440732, 3.135835306625308)), ('u', 7, (3.2466346084025846, 1.5796661546578783, 0.011580378728456964)), ('u', 6, (-3.120232697463237, 2.879768739966685, 3.1348440222472)), ('u', 5, (-0.046785070205743386, 0.4056326085404054, 0.002375184506990438)), ('u', 2, (-3.140768795152512, 0.019373565259231323, 3.142300169777424)), ('u', 4, (0.1510978477008815, 0.015762532093672142, 0.15269082699043238)), ('cz', 6, 5), ('u', 0, (-3.1473500005943658, 0.014247799177776082, 3.1344570279934265)), ('u', 7, (-3.130012079446099, 0.008872207960859949, 3.148903141930863)), ('u', 6, (-0.006748918374636249, 1.8325974783381196, -0.02605154015202726)), ('u', 5, (-3.139217322386877, 0.00042255887548605883, 3.1407520214683635)), ('u', 2, (-3.140885241671172, 0.013095025040535484, 3.1420438569013545)), ('u', 4, (0.1526909573242352, 0.004927917258527168, 0.15043588650271156)), ('cz', 7, 6), ('u', 0, (-3.148728435722247, 0.0067827495190096185, 3.1335248220764322)), ('u', 7, (-3.1342821914188344, 0.028129641733904823, 3.139195407234359)), ('u', 6, (3.115541081546385, 0.262140876213786, -3.116429733833928)), ('u', 5, (-0.000840907442600364, 0.0004225930836379312, 0.007189232126876074)), ('u', 2, (-3.141141450070426, 0.006294436631301056, 3.1417061604825034)), ('u', 4, (-2.9911567934896848, 0.0009174423925525192, 3.2859525219007946)), ('cz', 6, 5), ('u', 0, (-0.008067571093408008, 0.000947631652682133, -0.00846216706528116)), ('u', 7, (3.1391956704722497, 1.5426670246695326, 0.0016073258126736434)), ('u', 6, (0.02516320223213957, 1.3089724159489882, -0.00656582071820867)), ('u', 5, (0.007189282899427954, 0.007842517840848119, 0.0028880269470614836)), ('u', 2, (0.00011340286348652562, 0.0007728097840692104, -0.000240879009985278)), ('u', 4, (-2.997232798093956, 0.0029487648706772516, 3.2770051945109633)), ('cz', 7, 6), ('u', 0, (-0.008462270998998336, 0.008679554484677866, -0.008239315058771884)), ('u', 7, (3.1431997551015023, 1.613540236976378, 0.008052968942624084)), ('u', 6, (-3.1481582863499415, 0.5685299684110979, 3.1155206918765828)), ('u', 5, (-1.5679083628424837, 1.5707509915693976, 1.5747175472181811)), ('u', 2, (-0.0002406189394174081, 0.007849405234238405, -0.0005440818251163691)), ('u', 4, (1.7062087148483527, 1.597774835124818, -1.5742161163239792)), ('cz', 5, 4), ('u', 0, (-0.008239206153869909, 0.016155984559545464, -0.007332514493955876)), ('u', 7, (0.008052838513088807, 0.03301089031760844, 0.006835686452027138)), ('u', 6, (3.1155205945275726, 0.5259323273395712, -3.125798377686331)), ('u', 5, (1.5747175471772599, 0.615479715722885, -1.0933194658614573)), ('u', 2, (-0.0005441989674194679, 0.01468457018831093, -0.0007309857091677702)), ('u', 4, (1.5673765373273134, 0.6154797008465961, -1.5400280847058372)), ('cz', 5, 4), ('u', 0, (-0.0073324152878501215, 0.02313481789955032, -0.005691990145514318)), ('u', 7, (0.006835375912833112, 0.02319276961434259, 0.005462251066907252)), ('u', 6, (3.1573866993336663, 2.0475425271728978, -0.0019187775552635955)), ('u', 5, (2.048273187532868, 1.5387685129310567, -1.5594679923212609)), ('u', 2, (-0.0007309271365018807, 0.021041951779091485, -0.0007404893827961025)), ('u', 4, (-1.5400280845256442, 1.5662716165591075, 1.591065342381638)), ('cz', 6, 5), ('u', 0, (-0.005692146265052731, 0.029396965666583372, -0.003285938952843994)), ('u', 7, (0.0054625371585474815, 0.013256196626928542, 0.003982673656713087)), ('u', 6, (3.13967410900957, 1.5707963701698164, -0.014377665440666654)), ('u', 5, (3.1529211212809507, 0.9553166204523242, -3.088479950265933)), ('u', 2, (-0.0007404307098160015, 0.026707380495143162, -0.0005207704482750253)), ('u', 4, (-3.1213234383275608, 0.01206942282101771, 3.153586974922128)), ('cz', 6, 5), ('u', 0, (-0.004193396363146381, 0.03904086961061197, -0.00049550232005148)), ('u', 7, (-3.1392117600713445, 0.0005483262234133876, 3.142488866062188)), ('u', 6, (3.1371272120918303, 0.0037461657141409886, -3.132170972597222)), ('u', 5, (-3.093726690620392, 0.7853981458661417, 3.141592612743864)), ('u', 2, (-0.000993259034395914, 0.035147769751032384, -0.0003747208955972169)), ('u', 4, (3.1703255251510423, 0.0496409716618151, -3.1060708565458612)), ('cz', 5, 4), ('u', 0, (-0.0004954498865382018, 0.042501802368178576, 0.0039263487131322205)), ('u', 7, (-3.140696102630624, 0.011105050147155305, 3.1410754044502793)), ('u', 6, (3.151014190354829, 1.567051569205639, -0.03353177992885881)), ('u', 5, (-6.818412173190677e-08, 0.7853981829662565, -6.261316257286463e-08)), ('u', 2, (-0.00037503346505089865, 0.03793018123688908, 0.0005501239732354355)), ('u', 4, (3.177114138123289, 0.3171364432240032, -3.103697454714576)), ('cz', 6, 5), ('u', 0, (0.003926452593371207, 0.04475061279264097, 0.009034314717243109)), ('u', 7, (-3.1421096424833355, 0.022026519589627508, 3.1398122594858933)), ('u', 6, (3.1080606857057025, 2.1002030280554758, -0.00961579530284773)), ('u', 5, (5.311327925947482e-08, 0.7853981195657265, 3.9747502027850575e-08)), ('u', 2, (0.000550124029179258, 0.03953586933873909, 0.001788159299299279)), ('u', 4, (0.03789506456208645, 1.2714054890423696, 0.014461751057602206)), ('cz', 5, 4), ('u', 0, (0.009034314759717738, 0.04580901009949549, 0.014778175510685176)), ('u', 7, (-3.143373386030864, 0.03338036152990201, 3.138785488245085)), ('u', 6, (-0.009615836838893426, 0.5215262138225559, 0.016704541600747583)), ('u', 5, (-5.127014592034207e-08, 1.0471969548697333, -6.761251225238653e-08)), ('u', 2, (0.0017884197671162879, 0.03998215933670839, 0.0033330064818323925)), ('u', 4, (-3.1271311847527232, 0.7853981645517627, 3.1415925588548226)), ('cz', 5, 4), ('u', 0, (0.014778331645939283, 0.04577123238559829, 0.02110404060006271)), ('u', 7, (-3.1444002354656693, 0.04522320030185381, 3.1380885312288216)), ('u', 6, (0.01670479529149806, 0.5200234341008141, 0.0025332547510830035)), ('u', 5, (-3.1415926097045292, 1.5707962792032464, 3.1294229390228527)), ('u', 2, (0.0033327462044416226, 0.03935447684242358, 0.005167416291726443)), ('u', 4, (-3.1415927722030283, 0.6154806146991095, 3.129423088604382)), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793))]), (8, [('u', 3, (0.0, 2.901974084960706, 0.0)), ('u', 2, (0.0, 0.5024148995576962, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 0.5024148995576962, 3.141592653589793)), ('u', 1, (0.0, 0.9687234065335635, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 0.9687234065335635, 3.141592653589793)), ('u', 0, (0.0, 1.3258176636680323, 0.0)), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 0), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.3258176636680323, 3.141592653589793)), ('u', 4, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 0, 7), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.5235987755982989, 0.0)), ('u', 0, (0.0, 0.5235987755982989, 0.0)), ('cz', 0, 1), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.1780972450961724, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.39269908169872414, 0.0)), ('cz', 2, 1), ('u', 1, (-3.141592653589793, 0.39269908169872414, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 1.1780972450961724, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.3089969389957472, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 0.26179938779914935, 0.0)), ('cz', 3, 2), ('u', 2, (-3.141592653589793, 0.26179938779914935, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 1.3089969389957472, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 1, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 0, 1), ('u', 1, (1.570796326794898, 0.6154797086703868, -1.570796326794898)), ('u', 0, (1.5707963267948966, 0.6154797086703876, -1.5707963267948966)), ('cz', 0, 1), ('u', 1, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 0, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 1, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.2630564724597029, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.30773985433519363, 0.0)), ('cz', 2, 1), ('u', 1, (-3.141592653589793, 0.30773985433519363, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 1.2630564724597029, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('cz', 0, 1), ('u', 1, (1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17)), ('u', 0, (-3.141592653589793, 0.7853981633974483, 3.141592653589793)), ('cz', 0, 1), ('u', 0, (3.116033094504285, 1.6089339729985166, -3.133681864957694)), ('u', 7, (3.2967664143484856, 0.37746088244651743, -2.90770693507335)), ('u', 6, (3.131620967463964, 0.00012329162616528996, -3.149914485095441)), ('u', 5, (-3.065351981199511, 1.570796325998965, 3.141592810233495)), ('u', 2, (3.1371045869684653, 0.032420700396367345, -3.1446953815086034)), ('u', 4, (0.1678815913663732, 1.5707964120927065, -3.6217239721136885e-08)), ('cz', 5, 4), ('u', 0, (-3.1336817276541074, 0.03988000500708097, 3.1477971068631443)), ('u', 7, (3.3754782677989725, 0.2957003307021647, -2.8697445297881767)), ('u', 6, (3.1332705618836747, 0.000670964769903343, -3.14776404325472)), ('u', 5, (-1.2304880342357098e-07, 0.5235987812541554, -2.9738587354984345e-08)), ('u', 2, (3.1384898734794255, 0.03459206355040011, -3.14347395431741)), ('u', 4, (-3.5769262187636676e-08, 0.523598771695477, -0.00014103518936014022)), ('cz', 5, 4), ('u', 0, (-3.1353881481357257, 0.04020044576188537, 3.1458639800638526)), ('u', 7, (3.4134410998710325, 0.18048471998342222, -2.8492079037556315)), ('u', 6, (3.1354213213872644, 1.5700021680961975, -4.475632626022019e-05)), ('u', 5, (-7.312312765017648e-08, 1.5707963100608628, -0.08150001422869849)), ('u', 2, (3.1397114051832484, 0.035398571332497825, -3.1424570006512282)), ('u', 4, (3.141451865879188, 1.5700162024807447, -2.9682222802448033)), ('cz', 6, 5), ('u', 0, (-3.1373213273631406, 0.03904869406383659, 3.1437810459612163)), ('u', 7, (0.29238516947199533, 0.018297901744561286, 0.2766538192545155)), ('u', 6, (3.141547879107957, 1.5729530881038725, -0.023488310377399735)), ('u', 5, (3.0600922833400723, 1.9638558876725143, -3.092184870708789)), ('u', 2, (3.1407285146219714, 0.034780753328794205, -3.1416728739950286)), ('u', 4, (3.3149630865408977, 0.0008295588521163107, -2.966619969343351)), ('cz', 5, 4), ('u', 0, (-3.1394044172879125, 0.036450418837086045, 3.141636995139753)), ('u', 7, (0.27665329513227915, 0.20639358701078347, 0.2227248088732203)), ('u', 6, (3.1181046088741766, 0.0021592102046690117, -3.1285213458022136)), ('u', 5, (-3.0921852441269735, 0.3947927655786377, 3.1415942468560623)), ('u', 2, (3.1415122251017022, 0.03275584928687877, -3.1411353417589982)), ('u', 4, (-2.966619878409117, 3.4215350387617885e-08, 3.2842899804013013)), ('cz', 6, 5), ('u', 0, (-3.141548051954922, 0.03249920709244061, 3.1395296271679576)), ('u', 7, (0.22272481234282604, 0.3511173537588248, 0.15800993946820546)), ('u', 6, (0.013071320724860377, 0.7902924514564892, -2.1749857445456378e-08)), ('u', 5, (1.7116094205398003e-06, 0.3947928587638951, -0.029447614128055424)), ('u', 2, (3.1420500173930836, 0.02940971648979802, -3.1408404266649796)), ('u', 4, (0.14269720984164003, 2.879100755880473e-08, 0.14726041862431766)), ('cz', 5, 4), ('u', 0, (-3.143655783856359, 0.027345703916849064, 3.1375606772273215)), ('u', 7, (0.15800983204431243, 0.43823920309252273, 0.10504175242947242)), ('u', 6, (3.1415925128035482, 0.7902924864061348, 0.021359803985439463)), ('u', 5, (-0.029447717332347988, 1.5600925697278034, -0.04678493351120711)), ('u', 2, (3.1423449846850295, 0.024885924431284862, -3.140768586890696)), ('u', 4, (0.1472604315872025, 0.010083607399992922, 0.15109805594159728)), ('cz', 7, 6), ('u', 0, (-3.1456246299521164, 0.021185628864440732, 3.135835306625308)), ('u', 7, (3.2466346084025846, 1.5796661546578783, 0.011580378728456964)), ('u', 6, (-3.120232697463237, 2.879768739966685, 3.1348440222472)), ('u', 5, (-0.046785070205743386, 0.4056326085404054, 0.002375184506990438)), ('u', 2, (-3.140768795152512, 0.019373565259231323, 3.142300169777424)), ('u', 4, (0.1510978477008815, 0.015762532093672142, 0.15269082699043238)), ('cz', 6, 5), ('u', 0, (-3.1473500005943658, 0.014247799177776082, 3.1344570279934265)), ('u', 7, (-3.130012079446099, 0.008872207960859949, 3.148903141930863)), ('u', 6, (-0.006748918374636249, 1.8325974783381196, -0.02605154015202726)), ('u', 5, (-3.139217322386877, 0.00042255887548605883, 3.1407520214683635)), ('u', 2, (-3.140885241671172, 0.013095025040535484, 3.1420438569013545)), ('u', 4, (0.1526909573242352, 0.004927917258527168, 0.15043588650271156)), ('cz', 7, 6), ('u', 0, (-3.148728435722247, 0.0067827495190096185, 3.1335248220764322)), ('u', 7, (-3.1342821914188344, 0.028129641733904823, 3.139195407234359)), ('u', 6, (3.115541081546385, 0.262140876213786, -3.116429733833928)), ('u', 5, (-0.000840907442600364, 0.0004225930836379312, 0.007189232126876074)), ('u', 2, (-3.141141450070426, 0.006294436631301056, 3.1417061604825034)), ('u', 4, (-2.9911567934896848, 0.0009174423925525192, 3.2859525219007946)), ('cz', 6, 5), ('u', 0, (-0.008067571093408008, 0.000947631652682133, -0.00846216706528116)), ('u', 7, (3.1391956704722497, 1.5426670246695326, 0.0016073258126736434)), ('u', 6, (0.02516320223213957, 1.3089724159489882, -0.00656582071820867)), ('u', 5, (0.007189282899427954, 0.007842517840848119, 0.0028880269470614836)), ('u', 2, (0.00011340286348652562, 0.0007728097840692104, -0.000240879009985278)), ('u', 4, (-2.997232798093956, 0.0029487648706772516, 3.2770051945109633)), ('cz', 7, 6), ('u', 0, (-0.008462270998998336, 0.008679554484677866, -0.008239315058771884)), ('u', 7, (3.1431997551015023, 1.613540236976378, 0.008052968942624084)), ('u', 6, (-3.1481582863499415, 0.5685299684110979, 3.1155206918765828)), ('u', 5, (-1.5679083628424837, 1.5707509915693976, 1.5747175472181811)), ('u', 2, (-0.0002406189394174081, 0.007849405234238405, -0.0005440818251163691)), ('u', 4, (1.7062087148483527, 1.597774835124818, -1.5742161163239792)), ('cz', 5, 4), ('u', 0, (-0.008239206153869909, 0.016155984559545464, -0.007332514493955876)), ('u', 7, (0.008052838513088807, 0.03301089031760844, 0.006835686452027138)), ('u', 6, (3.1155205945275726, 0.5259323273395712, -3.125798377686331)), ('u', 5, (1.5747175471772599, 0.615479715722885, -1.0933194658614573)), ('u', 2, (-0.0005441989674194679, 0.01468457018831093, -0.0007309857091677702)), ('u', 4, (1.5673765373273134, 0.6154797008465961, -1.5400280847058372)), ('cz', 5, 4), ('u', 0, (-0.0073324152878501215, 0.02313481789955032, -0.005691990145514318)), ('u', 7, (0.006835375912833112, 0.02319276961434259, 0.005462251066907252)), ('u', 6, (3.1573866993336663, 2.0475425271728978, -0.0019187775552635955)), ('u', 5, (2.048273187532868, 1.5387685129310567, -1.5594679923212609)), ('u', 2, (-0.0007309271365018807, 0.021041951779091485, -0.0007404893827961025)), ('u', 4, (-1.5400280845256442, 1.5662716165591075, 1.591065342381638)), ('cz', 6, 5), ('u', 0, (-0.005692146265052731, 0.029396965666583372, -0.003285938952843994)), ('u', 7, (0.0054625371585474815, 0.013256196626928542, 0.003982673656713087)), ('u', 6, (3.13967410900957, 1.5707963701698164, -0.014377665440666654)), ('u', 5, (3.1529211212809507, 0.9553166204523242, -3.088479950265933)), ('u', 2, (-0.0007404307098160015, 0.026707380495143162, -0.0005207704482750253)), ('u', 4, (-3.1213234383275608, 0.01206942282101771, 3.153586974922128)), ('cz', 6, 5), ('u', 0, (-0.004193396363146381, 0.03904086961061197, -0.00049550232005148)), ('u', 7, (-3.1392117600713445, 0.0005483262234133876, 3.142488866062188)), ('u', 6, (3.1371272120918303, 0.0037461657141409886, -3.132170972597222)), ('u', 5, (-3.093726690620392, 0.7853981458661417, 3.141592612743864)), ('u', 2, (-0.000993259034395914, 0.035147769751032384, -0.0003747208955972169)), ('u', 4, (3.1703255251510423, 0.0496409716618151, -3.1060708565458612)), ('cz', 5, 4), ('u', 0, (-0.0004954498865382018, 0.042501802368178576, 0.0039263487131322205)), ('u', 7, (-3.140696102630624, 0.011105050147155305, 3.1410754044502793)), ('u', 6, (3.151014190354829, 1.567051569205639, -0.03353177992885881)), ('u', 5, (-6.818412173190677e-08, 0.7853981829662565, -6.261316257286463e-08)), ('u', 2, (-0.00037503346505089865, 0.03793018123688908, 0.0005501239732354355)), ('u', 4, (3.177114138123289, 0.3171364432240032, -3.103697454714576)), ('cz', 6, 5), ('u', 0, (0.003926452593371207, 0.04475061279264097, 0.009034314717243109)), ('u', 7, (-3.1421096424833355, 0.022026519589627508, 3.1398122594858933)), ('u', 6, (3.1080606857057025, 2.1002030280554758, -0.00961579530284773)), ('u', 5, (5.311327925947482e-08, 0.7853981195657265, 3.9747502027850575e-08)), ('u', 2, (0.000550124029179258, 0.03953586933873909, 0.001788159299299279)), ('u', 4, (0.03789506456208645, 1.2714054890423696, 0.014461751057602206)), ('cz', 5, 4), ('u', 0, (0.009034314759717738, 0.04580901009949549, 0.014778175510685176)), ('u', 7, (-3.143373386030864, 0.03338036152990201, 3.138785488245085)), ('u', 6, (-0.009615836838893426, 0.5215262138225559, 0.016704541600747583)), ('u', 5, (-5.127014592034207e-08, 1.0471969548697333, -6.761251225238653e-08)), ('u', 2, (0.0017884197671162879, 0.03998215933670839, 0.0033330064818323925)), ('u', 4, (-3.1271311847527232, 0.7853981645517627, 3.1415925588548226)), ('cz', 5, 4), ('u', 0, (0.014778331645939283, 0.04577123238559829, 0.02110404060006271)), ('u', 7, (-3.1444002354656693, 0.04522320030185381, 3.1380885312288216)), ('u', 6, (0.01670479529149806, 0.5200234341008141, 0.0025332547510830035)), ('u', 5, (-3.1415926097045292, 1.5707962792032464, 3.1294229390228527)), ('u', 2, (0.0033327462044416226, 0.03935447684242358, 0.005167416291726443)), ('u', 4, (-3.1415927722030283, 0.6154806146991095, 3.129423088604382)), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793))])], (9, 2): [(9, [('u', 3, (0.0, 2.031350318476219, 0.0)), ('u', 2, (0.0, 1.0697033135295393, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.0697033135295393, 3.141592653589793)), ('u', 7, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 7), ('u', 3, (-1.1751575440460897, 0.26067405538424787, -0.6871036784333111)), ('u', 2, (-2.904130104115435, 1.2877711084423432e-09, 2.934059522144068)), ('u', 1, (-2.9086275139335025, 1.0453477160710642, 3.153321898055499)), ('u', 0, (0.26021582247647235, 0.23328645359957653, 0.5640517095927046)), ('u', 8, (-3.372168548020394, 1.5707965990295842, 2.6266535483509807)), ('cz', 2, 8), ('u', 3, (-4.067784972759647, 0.18118027078371085, 1.9687154621961178)), ('u', 2, (0.6657778834497072, 3.1415924348112156, -0.3848408382298367)), ('u', 1, (0.4128655706052585, 0.29169992513927007, -0.048757596765475836)), ('u', 0, (-0.01538434204453374, 0.2152073564927922, -0.10649516288155722)), ('u', 8, (3.600940580816046, 1.9302884105856912, 0.062230185539184024)), ('cz', 2, 1), ('u', 3, (2.2110636077782657, 1.4366927582046864, 0.05927407796264106)), ('u', 2, (2.8501197239585907, 1.5707961081228536, -2.218612322475263)), ('u', 1, (3.2730549986716997, 0.6886101178919402, -2.575993276237892)), ('u', 0, (0.33564578704341586, 0.34380107009482985, 0.2686060068208995)), ('u', 8, (3.197192101971976, 0.46678376334754723, -2.8464804259837937)), ('cz', 3, 2), ('u', 3, (3.1810500031135835, 2.6883966429925574, -0.38800367167046135)), ('u', 2, (-2.493776771582644, 1.5707962109638427, 2.51030076068191)), ('u', 1, (0.9999694117196286, 0.4176562437414257, 0.20160956604139202)), ('u', 0, (3.6808361043184217, 0.1668222398191245, -2.4812094831857348)), ('u', 8, (0.24504038097329958, 0.047348615295652756, 0.3719166684254209)), ('cz', 2, 1), ('u', 3, (3.230091617416889, 0.6067888110944931, 0.4002632658023271)), ('u', 2, (-1.4631032074158488, 1.57079664710389, -0.14921893023630595)), ('u', 1, (-2.6839994517778503, 1.0495876640777395, 3.460245914604988)), ('u', 0, (0.7187858549182815, 0.037212705319108294, 0.12527987749719643)), ('u', 8, (3.2479004661525797, 0.1907831125143074, -3.0242116567792574)), ('cz', 3, 2), ('u', 3, (3.1985339495323744, 1.2421178686929215, 0.7685533630154877)), ('u', 2, (-2.9302892955456565, 0.40750919585425915, 2.9472548692323572)), ('u', 1, (-1.6084257572840468, 2.0207749134588644, 1.994304461676151)), ('u', 0, (1.8809384577251094, 1.6092895321161782, -1.5211683708909391)), ('u', 8, (0.09200442112359732, 0.16982459029422026, -0.09778032844482153)), ('cz', 1, 0), ('u', 3, (-2.2568501372182834, 0.5240394855151633, 3.002106847942613)), ('u', 2, (-3.072655095440928, 0.3455044194612774, 3.0252096244217404)), ('u', 1, (1.7738837963033383, 0.6154796975274284, -1.3202384912232124)), ('u', 0, (1.2641821205066717, 1.5955335708882186, -1.6258849719860065)), ('u', 8, (0.21689653646156265, 0.20357238920732112, 0.36478729690880035)), ('cz', 2, 1), ('u', 3, (0.1562798261086924, 0.16400185626187547, 0.013634776231117654)), ('u', 2, (3.1243115228579392, 0.13718296059318005, -3.027750293623534)), ('u', 1, (3.5065144559822388, 1.5707965873327723, -2.111596372205712)), ('u', 0, (0.24866742049844764, 0.04413150060343756, -0.35796327032862846)), ('u', 8, (-0.060059179136186525, 0.12055765815950181, 0.5683452214035626)), ('cz', 1, 0), ('u', 3, (-0.28236679701589895, 0.19307427245131378, 0.11749337831775906)), ('u', 2, (3.2175722375293705, 2.051762696906425, 0.20621420836491122)), ('u', 1, (0.5408000122591998, 1.5707964395303073, -0.799879876487902)), ('u', 0, (3.405057351876788, 0.34159515819165936, -2.624131222806333)), ('u', 8, (3.045772691560537, 0.06817211668520294, -2.845739293405762)), ('cz', 2, 1), ('u', 8, (-1.1360710551574318, 1.505239968654346, -4.9016265742176195)), ('u', 7, (-3.1415926600046964, 1.570796324839406, -0.41699610807162846)), ('u', 2, (3.1147814607134094, 1.721597437073412, -0.6333155888908188)), ('u', 6, (-0.32050837989764996, 2.2555155331604326, -0.3438787166564868)), ('u', 5, (0.17174521324562275, 0.031146318806421287, 1.4509660709986316)), ('cz', 7, 6), ('u', 8, (0.4518976427418188, 0.5380965465711559, -0.7076166725272257)), ('u', 7, (2.377563839793169, 1.5707963276492578, -3.5920506340583915)), ('u', 2, (-2.68824546699976, 0.24094033773879892, 3.237162941690255)), ('u', 6, (-2.4810190398761565, 2.000382542730125, 3.1271194163022438)), ('u', 5, (1.5432381968531852, 0.3953680477116938, 0.49343335857450055)), ('cz', 8, 7), ('u', 8, (-3.4986968589449665, 0.42101655926489817, 2.697973728216149)), ('u', 7, (2.613559354625364, 0.7303878192993729, -2.7116651640459857)), ('u', 2, (0.08318659923227986, 0.0589607804817785, -0.1756097494762521)), ('u', 6, (2.4239619553838874, 0.36922021676920536, -2.812224339696096)), ('u', 5, (-3.1398940978223853, 0.33785932587456824, 3.1167451417159167)), ('cz', 7, 6), ('u', 8, (2.9667798658034155, 1.6993058586079814, -0.27691705791021914)), ('u', 7, (0.4062211191842141, 1.5707963256045296, -0.10975701098500565)), ('u', 2, (-3.2830295374537153, 0.560642007824093, 2.996823018344159)), ('u', 6, (-0.7133892649258873, 2.5020118574577993e-08, -0.1758884761136349)), ('u', 5, (-3.0653246916751273, 0.3576152282275567, 3.085728183937486)), ('cz', 8, 7), ('u', 8, (2.933480189689565, 0.27753841802099527, -3.3484508810627425)), ('u', 7, (3.593755397978441, 1.0317532409179107e-08, -2.3057648178684724)), ('u', 2, (-0.5662649656646794, 0.40579970829580525, 0.08065303252572728)), ('u', 6, (2.9916587456425994, 1.4721995737462234e-08, -3.1972097818947)), ('u', 5, (-1.157118088638148, 0.6510021827888648, -0.45425267384030044)), ('cz', 6, 5), ('u', 8, (3.375107406833436, 0.728719169615773, -2.6564227900838295)), ('u', 7, (4.717882284688699, 2.9014631877375603e-08, 0.03499669314220544)), ('u', 2, (3.2566859996186186, 0.13634784137903486, -2.731561020643601)), ('u', 6, (-1.4741812562965175, 1.5707963299940688, 3.8850569601487615)), ('u', 5, (2.944475558010268, 0.7947711174071349, -3.3245539777200115)), ('cz', 7, 6), ('u', 8, (0.3510188461157579, 0.022763173674389754, -0.18008573438180525)), ('u', 7, (-3.0310357172669966, 0.503129628002097, 2.7062472045923314)), ('u', 2, (-0.3376220458028091, 0.09688717601731367, 0.09132756359967983)), ('u', 6, (-2.3142606340546887, 2.1573716490417874, 2.198652747023293)), ('u', 5, (-1.0670745154961903, 0.03110142674163011, -0.5236146847714218)), ('cz', 6, 5), ('u', 8, (0.4671096828828962, 0.008090946775635946, 0.1684230948562603)), ('u', 7, (3.785692462019081, 1.6033891770995943, -0.4677534521033184)), ('u', 2, (-0.011921597402141097, 0.6455752830311845, 0.07236214366382598)), ('u', 6, (-0.10425764881894511, 1.570796340093249, 0.732992731562423)), ('u', 5, (-1.4589561764307943, 0.15632034326211913, -0.4613416507330019)), ('cz', 7, 6), ('u', 8, (0.3054932649597624, 0.7349006554385038, -0.44790492963445244)), ('u', 7, (2.9439525993830005, 1.8998281046082481, -0.9263836084918353)), ('u', 2, (2.7064458565936973, 0.304221399010612, -2.9313223612746593)), ('u', 6, (0.05687524151417256, 0.41419353368705036, -0.6369174662855737)), ('u', 5, (-3.0402980869294605, 1.7172409046188886, 1.5578030228290438)), ('u', 3, (-2.7877311679092527, 0.5131360297009971, 2.4612435316960513)), ('u', 1, (-0.7709163438843996, 0.7853981682079562, -1.3106286904277253)), ('u', 0, (0.2231414472615762, 1.8274158817631607, -0.7523404924869557)), ('cz', 1, 0), ('u', 3, (0.07311385607457008, 0.3503418608868059, 0.31347456825108533)), ('u', 1, (-1.8309640698265242, 1.5707964356112187, 2.4580318925890987)), ('u', 0, (-3.7320491347325393, 1.5707968395486716, 2.4580318218348425)), ('u', 4, (2.6611157981787685, 0.3348491692101108, -3.478146822551501)), ('u', 4, (0.025099278002302938, 0.020289187327979002, 1.0994964759651373)), ('u', 4, (1.1856425386986325, 0.10635862335364886, -0.2028633457911238)), ('u', 4, (-0.502874265909908, 0.20858030829480428, -0.3670399133532716)), ('u', 4, (2.2511667393630503, 1.2655340329567368, -1.360636025018449)), ('cz', 4, 5), ('u', 4, (1.3977632976181833, 0.6525439868142484, -1.121158885295027)), ('u', 5, (1.296487072623602, 0.6154797086700415, -1.2957682456694826)), ('u', 6, (-0.071151212705162, 0.7854015269467203, -0.6100752785286383)), ('u', 7, (2.8446334516331584, 0.7676417279941344, -2.574966225015541)), ('cz', 4, 5), ('u', 4, (-1.861323235743385, 2.3325757957288817, 1.8904113388852222)), ('u', 5, (2.1957840870535623, 1.9503858232632065, -1.7356342087318746)), ('u', 6, (2.363249945498356, 1.3411433722001864, 0.6795535715525908)), ('u', 7, (-3.2164691598207225, 0.08261866080509377, 2.1635538126647655)), ('cz', 5, 6), ('u', 4, (-2.4502383748008967, 0.5066738472749666, 2.658848459855456)), ('u', 5, (-0.1211349652721937, 0.9553166181244168, 0.9193737914269516)), ('u', 6, (4.017787111955051, 1.5707963267950278, -0.14625006961981346)), ('u', 7, (2.5641160141141413, 0.04650871485508044, -3.241997417780418)), ('cz', 5, 6), ('u', 4, (-2.777227076724675, 0.8219273046226232, 2.9033750331038646)), ('u', 5, (3.6497578309532717, 1.5707963267956213, -2.3607616933088007)), ('u', 6, (-0.43303647828505254, 0.23307421193024305, 0.34113733371999544)), ('u', 7, (-3.462869472210422, 0.04369737122806576, 2.659609116246114)), ('cz', 4, 5), ('u', 4, (3.5342993639707627, 1.570796326793336, -1.5745998474870015)), ('u', 5, (-2.3516272870779646, 2.3561944901923084, -2.942884153984175)), ('u', 6, (2.821397764753086, 1.4231330404147982, 0.26950514138026715)), ('u', 7, (2.592944263744413, 0.12303314857564963, -3.235575396438218)), ('cz', 4, 5), ('u', 4, (3.145396174280426, 1.5707963267946714, -2.914376451603585)), ('u', 5, (2.9428841539840684, 1.5707963267948422, -2.9143764516023642)), ('u', 6, (0.768553377240472, 0.663097505060459, 0.7319728725594956)), ('u', 7, (1.6634127072052856, 0.24170758579466498, -3.039721204637301)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0))]), (9, [('u', 3, (0.0, 2.031350318476219, 0.0)), ('u', 2, (0.0, 1.0697033135295393, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.0697033135295393, 3.141592653589793)), ('u', 7, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 7), ('u', 3, (-1.1751575440460897, 0.26067405538424787, -0.6871036784333111)), ('u', 2, (-2.904130104115435, 1.2877711084423432e-09, 2.934059522144068)), ('u', 1, (-2.9086275139335025, 1.0453477160710642, 3.153321898055499)), ('u', 0, (0.26021582247647235, 0.23328645359957653, 0.5640517095927046)), ('u', 8, (-3.372168548020394, 1.5707965990295842, 2.6266535483509807)), ('cz', 2, 8), ('u', 3, (-4.067784972759647, 0.18118027078371085, 1.9687154621961178)), ('u', 2, (0.6657778834497072, 3.1415924348112156, -0.3848408382298367)), ('u', 1, (0.4128655706052585, 0.29169992513927007, -0.048757596765475836)), ('u', 0, (-0.01538434204453374, 0.2152073564927922, -0.10649516288155722)), ('u', 8, (3.600940580816046, 1.9302884105856912, 0.062230185539184024)), ('cz', 2, 1), ('u', 3, (2.2110636077782657, 1.4366927582046864, 0.05927407796264106)), ('u', 2, (2.8501197239585907, 1.5707961081228536, -2.218612322475263)), ('u', 1, (3.2730549986716997, 0.6886101178919402, -2.575993276237892)), ('u', 0, (0.33564578704341586, 0.34380107009482985, 0.2686060068208995)), ('u', 8, (3.197192101971976, 0.46678376334754723, -2.8464804259837937)), ('cz', 3, 2), ('u', 3, (3.1810500031135835, 2.6883966429925574, -0.38800367167046135)), ('u', 2, (-2.493776771582644, 1.5707962109638427, 2.51030076068191)), ('u', 1, (0.9999694117196286, 0.4176562437414257, 0.20160956604139202)), ('u', 0, (3.6808361043184217, 0.1668222398191245, -2.4812094831857348)), ('u', 8, (0.24504038097329958, 0.047348615295652756, 0.3719166684254209)), ('cz', 2, 1), ('u', 3, (3.230091617416889, 0.6067888110944931, 0.4002632658023271)), ('u', 2, (-1.4631032074158488, 1.57079664710389, -0.14921893023630595)), ('u', 1, (-2.6839994517778503, 1.0495876640777395, 3.460245914604988)), ('u', 0, (0.7187858549182815, 0.037212705319108294, 0.12527987749719643)), ('u', 8, (3.2479004661525797, 0.1907831125143074, -3.0242116567792574)), ('cz', 3, 2), ('u', 3, (3.1985339495323744, 1.2421178686929215, 0.7685533630154877)), ('u', 2, (-2.9302892955456565, 0.40750919585425915, 2.9472548692323572)), ('u', 1, (-1.6084257572840468, 2.0207749134588644, 1.994304461676151)), ('u', 0, (1.8809384577251094, 1.6092895321161782, -1.5211683708909391)), ('u', 8, (0.09200442112359732, 0.16982459029422026, -0.09778032844482153)), ('cz', 1, 0), ('u', 3, (-2.2568501372182834, 0.5240394855151633, 3.002106847942613)), ('u', 2, (-3.072655095440928, 0.3455044194612774, 3.0252096244217404)), ('u', 1, (1.7738837963033383, 0.6154796975274284, -1.3202384912232124)), ('u', 0, (1.2641821205066717, 1.5955335708882186, -1.6258849719860065)), ('u', 8, (0.21689653646156265, 0.20357238920732112, 0.36478729690880035)), ('cz', 2, 1), ('u', 3, (0.1562798261086924, 0.16400185626187547, 0.013634776231117654)), ('u', 2, (3.1243115228579392, 0.13718296059318005, -3.027750293623534)), ('u', 1, (3.5065144559822388, 1.5707965873327723, -2.111596372205712)), ('u', 0, (0.24866742049844764, 0.04413150060343756, -0.35796327032862846)), ('u', 8, (-0.060059179136186525, 0.12055765815950181, 0.5683452214035626)), ('cz', 1, 0), ('u', 3, (-0.28236679701589895, 0.19307427245131378, 0.11749337831775906)), ('u', 2, (3.2175722375293705, 2.051762696906425, 0.20621420836491122)), ('u', 1, (0.5408000122591998, 1.5707964395303073, -0.799879876487902)), ('u', 0, (3.405057351876788, 0.34159515819165936, -2.624131222806333)), ('u', 8, (3.045772691560537, 0.06817211668520294, -2.845739293405762)), ('cz', 2, 1), ('u', 3, (-2.7877311679092527, 0.5131360297009971, 2.4612435316960513)), ('u', 2, (2.7309235597960666, 1.9679245926489324, 0.005706386482318138)), ('u', 1, (-0.7709163438843996, 0.7853981682079562, -1.3106286904277253)), ('u', 0, (0.2231414472615762, 1.8274158817631607, -0.7523404924869557)), ('u', 8, (3.4861799926859622, 0.01496747952486012, -2.645700594264318)), ('cz', 1, 0), ('u', 3, (0.07311385607457008, 0.3503418608868059, 0.31347456825108533)), ('u', 2, (2.711866938377728, 0.4312735279598157, -3.435996990272945)), ('u', 1, (-1.8309640698265242, 1.5707964356112187, 2.4580318925890987)), ('u', 0, (-3.7320491347325393, 1.5707968395486716, 2.4580318218348425)), ('u', 8, (2.880664166591831, 1.6294531686526075, 0.17219468854533693)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 1.5707963267948966, 0.0)), ('u', 7, (0.0, 1.5707963267948966, 0.0)), ('cz', 6, 7), ('u', 7, (2.983475401827938e-18, 0.886077123792614, -1.140057778643436e-16)), ('u', 6, (-3.141592653589793, 0.886077123792614, 3.141592653589793)), ('cz', 6, 7), ('u', 7, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 6, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('cz', 7, 8), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (-3.141592653589793, 1.1071487177940904, 3.141592653589793)), ('cz', 8, 7), ('u', 7, (0.0, 1.1071487177940904, 0.0)), ('u', 7, (-3.141592653589793, 2.677945044588987, 3.141592653589793)), ('cz', 6, 7), ('u', 7, (0.0, 2.677945044588987, 0.0)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 7, 8), ('u', 4, (2.6611157981787685, 0.3348491692101108, -3.478146822551501)), ('u', 5, (-0.6924935761410032, 0.785398163397586, -0.053508259482247134)), ('u', 6, (0.0, 3.2169798970235493e-13, -0.0718275583082407)), ('u', 7, (0.21662944741760443, 0.009419871126993384, 0.2605984439252987)), ('cz', 5, 6), ('u', 4, (0.025099278002302938, 0.020289187327979002, 1.0994964759651373)), ('u', 5, (2.5315883843221796, 1.126852592477393, -3.148371794074219)), ('u', 6, (-1.4998309724843017, 1.5707963267947973, 3.6496660852705918)), ('u', 7, (4.234308233013136, 1.572828927799284, 0.2331611443336068)), ('cz', 6, 7), ('u', 4, (1.1856425386986325, 0.10635862335364886, -0.2028633457911238)), ('u', 5, (-0.8219614493886984, 0.6321024525621387, -0.5878027940943908)), ('u', 6, (-2.0788697584754736, 1.570796326795042, 2.347021439424257)), ('u', 7, (-3.0080223501098153, 0.5096791005494581, 2.4845323181052326)), ('cz', 5, 6), ('u', 4, (-0.502874265909908, 0.20858030829480428, -0.3670399133532716)), ('u', 5, (-0.9707185567135301, 0.13951846091143721, -0.4780125605259198)), ('u', 6, (-0.2526263370310758, 1.5707963267949179, 0.3004834247059334)), ('u', 7, (3.4904687242914436, 1.6114464165634266, -0.4680790966247721)), ('cz', 6, 7), ('u', 4, (2.2511667393630503, 1.2655340329567368, -1.360636025018449)), ('u', 5, (-2.7157115308499935, 1.7101276920868471, 1.6054345492242823)), ('u', 6, (0.13739192132802322, 0.7397973392342709, -0.22411577638408456)), ('u', 7, (3.388780904747323, 2.07951450118411, -0.5454979476806319)), ('cz', 4, 5), ('u', 4, (1.3977632976181833, 0.6525439868142484, -1.121158885295027)), ('u', 5, (1.296487072623602, 0.6154797086700415, -1.2957682456694826)), ('u', 6, (-0.071151212705162, 0.7854015269467203, -0.6100752785286383)), ('u', 7, (2.8446334516331584, 0.7676417279941344, -2.574966225015541)), ('cz', 4, 5), ('u', 4, (-1.861323235743385, 2.3325757957288817, 1.8904113388852222)), ('u', 5, (2.1957840870535623, 1.9503858232632065, -1.7356342087318746)), ('u', 6, (2.363249945498356, 1.3411433722001864, 0.6795535715525908)), ('u', 7, (-3.2164691598207225, 0.08261866080509377, 2.1635538126647655)), ('cz', 5, 6), ('u', 4, (-2.4502383748008967, 0.5066738472749666, 2.658848459855456)), ('u', 5, (-0.1211349652721937, 0.9553166181244168, 0.9193737914269516)), ('u', 6, (4.017787111955051, 1.5707963267950278, -0.14625006961981346)), ('u', 7, (2.5641160141141413, 0.04650871485508044, -3.241997417780418)), ('cz', 5, 6), ('u', 4, (-2.777227076724675, 0.8219273046226232, 2.9033750331038646)), ('u', 5, (3.6497578309532717, 1.5707963267956213, -2.3607616933088007)), ('u', 6, (-0.43303647828505254, 0.23307421193024305, 0.34113733371999544)), ('u', 7, (-3.462869472210422, 0.04369737122806576, 2.659609116246114)), ('cz', 4, 5), ('u', 4, (3.5342993639707627, 1.570796326793336, -1.5745998474870015)), ('u', 5, (-2.3516272870779646, 2.3561944901923084, -2.942884153984175)), ('u', 6, (2.821397764753086, 1.4231330404147982, 0.26950514138026715)), ('u', 7, (2.592944263744413, 0.12303314857564963, -3.235575396438218)), ('cz', 4, 5), ('u', 4, (3.145396174280426, 1.5707963267946714, -2.914376451603585)), ('u', 5, (2.9428841539840684, 1.5707963267948422, -2.9143764516023642)), ('u', 6, (0.768553377240472, 0.663097505060459, 0.7319728725594956)), ('u', 7, (1.6634127072052856, 0.24170758579466498, -3.039721204637301)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0))])], (9, 4): [(9, [('u', 3, (0.0, 2.740499937812782, 0.0)), ('u', 2, (0.0, 0.6125547383393389, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 0.6125547383393389, 3.141592653589793)), ('u', 1, (0.0, 1.0365702822269822, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.0365702822269822, 3.141592653589793)), ('u', 0, (0.0, 1.3508083493994372, 0.0)), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 0), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.3508083493994372, 3.141592653589793)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 7), ('u', 4, (1.6783995219433123, 2.677945059334841, -2.347528943738099)), ('u', 5, (-1.5846090155505541, 3.141592648803949, 0.9739539182276005)), ('u', 6, (-0.043128525936613135, 1.1363142332841882, 0.048276423210969005)), ('u', 7, (3.6131240518873415, 1.1107010877583263, -0.9418354587012361)), ('cz', 4, 5), ('u', 4, (-1.6523744495521213, 0.16800233720120406, 0.9527359405859664)), ('u', 5, (1.0712416136914256, 1.5707963153748048, -2.1972073505033416)), ('u', 6, (-0.768693759082117, 0.5531753246206115, 0.5396942354555023)), ('u', 7, (-2.9148007132765867, 0.2008037332194747, 3.017645632458623)), ('cz', 5, 6), ('u', 4, (0.22916288305642551, 0.37442170013424514, 0.49572334230405124)), ('u', 5, (2.197207529734123, 0.8860771116860425, -3.0442805131145416)), ('u', 6, (2.843357985790216, 0.3495555085185406, 0.4905298831427509)), ('u', 7, (0.2979688948426138, 0.24839385813200238, -0.43803588336134747)), ('cz', 4, 5), ('u', 4, (0.41570487214026375, 0.2787985107045163, 0.2414760855845465)), ('u', 5, (-0.09731230047945474, 1.5707963219500847, 0.18693697618542235)), ('u', 6, (3.369013850140026, 2.7578072190774874, -0.10129045285916272)), ('u', 7, (-0.15950293778007263, 0.6278305605369155, 0.29241996204870857)), ('cz', 5, 6), ('u', 4, (0.19929743529559463, 0.88167062540414, 0.15890588000448685)), ('u', 5, (1.142456664092194, 0.2517907327213283, -0.38435357509787066)), ('u', 6, (3.6071107386496584, 1.5707962929360881, 1.0300524832632352)), ('u', 7, (3.0661863018689504, 1.5717431102987751, -0.1241345695646574)), ('cz', 6, 7), ('u', 4, (-0.3120984854440979, 0.3324426283546392, -0.8392019353490036)), ('u', 5, (-0.9523727258677195, 1.5112007017232427, -1.782143104443146)), ('u', 6, (-2.8107096237423645, 1.2281134369081435, 3.454138659309974)), ('u', 7, (3.3599544039824045, 1.5363198332342687, 0.5738148273621144)), ('cz', 5, 6), ('u', 4, (-0.17407864683084506, 1.1981581532047667, 0.720754058352084)), ('u', 5, (-1.0485857025157856, 0.25106472349746944, -0.026165490427976112)), ('u', 6, (-0.1946523375171709, 0.34489327860458485, -0.577293313523412)), ('u', 7, (0.21409937561655093, 0.048860780375053335, 0.5401001834509469)), ('cz', 6, 7), ('u', 4, (0.08604625009239802, 0.25603321491570213, -0.6234998208627462)), ('u', 5, (-3.115427233337476, 0.25106470278189263, 3.0001440070116105)), ('u', 6, (-2.564299345768496, 0.34489330545421637, 3.14097197578436)), ('u', 7, (-0.7944434936537823, 0.15214440370723623, -0.38700938932873424)), ('cz', 5, 6), ('u', 4, (0.8539870509991602, 0.048184497575099156, 0.3465322584014089)), ('u', 5, (0.2753512792741515, 1.4910808473812084e-07, 0.5142367925003585)), ('u', 6, (-0.11727296544005196, 1.2281134650062855, 0.20497368134664418)), ('u', 7, (3.849810984935723, 1.7152699274463379, 0.6275369163340978)), ('cz', 6, 7), ('u', 4, (-0.03876308831078168, 0.8460430127672541, 0.8580307302996752)), ('u', 5, (2.840620213688528, 2.032147342582146e-07, -3.066629645265795)), ('u', 6, (2.3444126865023827, 1.656357328196517e-09, -2.662531125816978)), ('u', 7, (2.5619230084226126, 1.5707963266568443, -0.061782989421305334)), ('u', 8, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 0, 8), ('u', 2, (0.7253513784390777, 0.6044711094741366, 0.20469392790644375)), ('u', 3, (3.311225398804729, 0.45344431468038326, -2.811379919385577)), ('u', 1, (-2.967218355247499, 1.570796434542743, 2.782623640362077)), ('u', 0, (0.17437422015585258, 1.5707965042184875, 0.0936275536825773)), ('u', 5, (-2.981966217109752, 0.15731121643895493, 2.938590834577346)), ('cz', 1, 0), ('u', 2, (0.35596645780294867, 0.18572047518147186, 0.1993350213934021)), ('u', 3, (0.27426031921868355, 0.23461703245156662, 0.8733942285264623)), ('u', 1, (0.3589689683381108, 0.5235987755137496, 0.808167059420607)), ('u', 0, (-0.0936272595693031, 0.5235987756882645, 0.24164140029433875)), ('u', 5, (-0.3732167230268989, 0.1692158665066253, -0.5138610131235901)), ('cz', 1, 0), ('u', 2, (3.3065392993238065, 2.155371992839475, 0.5333711558032743)), ('u', 3, (0.17598851577546493, 0.15670123378522505, 0.3917956014566585)), ('u', 1, (0.762629607489512, 2.888760746641521, 1.273509717372987)), ('u', 0, (-2.8326884570670465, 1.2221833849116261, 2.9006620162668915)), ('u', 5, (-0.29514789830304344, 0.08656595906264068, -0.2957995850638988)), ('cz', 2, 1), ('u', 2, (3.137649879014554, 1.9634954073118636, -0.6075427985534696)), ('u', 3, (2.641964011958269, 0.22113794763596387, -2.499181790905139)), ('u', 1, (-1.2735092335544311, 1.570796389596646, 2.6537438706729946)), ('u', 0, (2.7190493967038063, 0.4335249098769316, -3.5464227887719253)), ('u', 5, (-0.6391916737665571, 0.1657229044953988, -0.024541329372776033)), ('cz', 2, 1), ('u', 2, (0.17775264740865682, 0.3058307733501329, -0.4417742235515568)), ('u', 3, (-0.24049984202753, 0.0778915304240662, -0.01766321037003693)), ('u', 1, (-3.046442949408996, 1.570796448575322, 2.117105098339699)), ('u', 0, (2.589347008940728, 0.28886655857903776, -3.6589389197843123)), ('u', 5, (0.0508440614483701, 0.07203248788002012, 0.029932533667720446)), ('cz', 1, 0), ('u', 2, (2.424643232393627, 1.6965841706658102, 0.29519056307839975)), ('u', 3, (-2.9031023199264445, 0.06550196451547159, 3.0490689833889255)), ('u', 1, (-0.5463092573212669, 2.748893521815632, -0.6787734947855513)), ('u', 0, (-0.8667873688727257, 0.19392965245174282, 0.26794479272692207)), ('u', 5, (3.093960255811586, 0.3684130596408908, -2.752177636430363)), ('cz', 2, 1), ('u', 2, (0.48622642934218985, 4.965240116914444e-08, 3.1314145386877534)), ('u', 3, (3.4555174636846955, 1.825677016721441, -0.10375945734790903)), ('u', 1, (3.1274454219227774, 0.3987010753799351, -2.7245703058001975)), ('u', 0, (-0.7836119278695097, 0.44642916421938117, -0.010954000022640242)), ('u', 5, (-0.35407245584296543, 0.1009018219745298, 0.12449653021465411)), ('cz', 2, 3), ('u', 2, (-2.95124738442682, 2.872225615416436, 3.168474353976826)), ('u', 3, (2.98864467816622, 1.8770178510291984, 0.03374189545665063)), ('u', 1, (-0.04688921667005741, 0.742010041871667, -0.42431748027242494)), ('u', 0, (2.517451732666387, 0.19291873478940372, -2.9271814483694167)), ('u', 5, (-0.6494532035971243, 0.6081156702407481, 0.4080258856521049)), ('cz', 2, 1), ('u', 2, (0.20730116474291263, 1.8331478973531157, 0.46325541824844596)), ('u', 3, (2.692541925224733, 0.332696125034078, -2.6878777969285363)), ('u', 1, (2.490863927966439, 0.283599555584151, -3.18962913786471)), ('u', 0, (0.17017190346421693, 0.20177925921584974, -0.08704916348372774)), ('u', 5, (2.7417475746980577, 0.5655452491435636, -3.0339198029229153)), ('cz', 2, 3), ('u', 2, (-2.8611849940478558, 0.34985907247470327, 2.4858926099614247)), ('u', 3, (2.87078395521952, 0.16840848054093654, -3.2915600146968)), ('u', 1, (0.048036261016780685, 0.28359965336885995, -0.2578628245055713)), ('u', 0, (-0.04579375400203432, 0.37476347219220135, -0.0019053945699001838)), ('u', 5, (-2.9901835580399307, 0.29429615142429255, 2.5298877934002943)), ('cz', 2, 1), ('u', 2, (-0.056927005043825446, 1.3014292976901551, -0.5532071820595532)), ('u', 3, (3.4785109576898656, 1.40526859119825, -0.254041621662799)), ('u', 1, (2.6370874705266063, 0.27233958144772935, -3.545853570337978)), ('u', 0, (3.105802818900175, 0.21107958032122495, -2.861471392990672)), ('u', 5, (3.0189880563377156, 0.19670772384256013, -3.0445309207076874)), ('cz', 2, 3), ('u', 2, (0.01339029507980527, 0.0416754087342334, -1.347371724509553)), ('u', 3, (3.499319469952347, 1.355519791945654, -0.5991567935202315)), ('u', 1, (-2.1086415347436596, 1.3486442481739653, 1.9995738248248909)), ('u', 0, (1.6776715535216509, 1.3775652646994023, -2.0667194984618944)), ('u', 5, (-0.9230392567029317, 0.3689502539357166, -0.027637551969518004)), ('cz', 1, 0), ('u', 2, (-0.6422086949145018, 0.6564240419907547, 0.3050682210391836)), ('u', 3, (-0.3570995437419664, 0.3230608004096355, -0.4818739743945121)), ('u', 1, (0.9831653765370997, 0.6154796885409349, -2.044379448865771)), ('u', 0, (1.2690644052694249, 0.6154797227608092, -1.450257109340972)), ('u', 5, (0.32441597127291805, 0.35884350817350413, -0.5202647635263868)), ('cz', 1, 0), ('u', 2, (3.0849500536513323, 0.5822022691130421, -0.24393556074215605)), ('u', 3, (-0.07734822951641647, 0.4435772553137907, -0.06596287366372468)), ('u', 1, (1.1222045403932766, 1.2721557575140932, -1.643452931303447)), ('u', 0, (-1.518587141618921, 1.6648415595850694, 1.767035244953014)), ('u', 5, (-2.8821138871253633, 0.6916920960996875, 2.47733528484548)), ('cz', 2, 1), ('u', 2, (2.2401631579399766, 1.57079612890024, -0.6250479515489906)), ('u', 3, (0.019069698632826204, 0.004875598650025819, 0.48157043024513996)), ('u', 1, (2.9948115830243927, 0.9587582103095811, -3.1531878704143894)), ('u', 0, (2.558845213379647, 0.0026409094544454616, -3.144567419915024)), ('u', 5, (-0.24806262907427706, 0.5508837005772087, -0.9339898849771324)), ('cz', 2, 1), ('u', 2, (-2.9673015967720766, 0.08948535665168787, 3.1599312277533493)), ('u', 3, (2.3790348862594284, 0.2712248006470355, -3.2040591439620414)), ('u', 1, (-3.129997892854534, 0.7853981335305925, 3.0269609858112005)), ('u', 0, (3.0359102852588746, 0.1938390933531295, -2.778825432657605)), ('u', 5, (2.705457550215509, 0.48649144979514203, -2.964461900581061)), ('cz', 1, 0), ('u', 2, (3.572868238545496, 1.5598263045480647, -0.19509687224531835)), ('u', 3, (2.998883524589488, 0.15059171477843603, -3.0599404762732667)), ('u', 1, (0.11463202022691882, 0.7853981649586114, 0.17677739302616385)), ('u', 0, (0.25127933585639883, 0.48994533516927113, -0.3547313940345701)), ('u', 5, (-3.157884001950231, 0.5858331264680662, 2.9538041371221344)), ('cz', 2, 1), ('u', 2, (3.3460062868220524, 0.5910844552042714, -0.052680694256529215)), ('u', 3, (-3.2368160309582397, 0.17716874842237512, 2.754312610522317)), ('u', 1, (-0.1767774518436646, 2.356194531997464, 0.14646174769130244)), ('u', 0, (-0.02761925796749795, 1.7818724057866364, 0.1473634861913784)), ('u', 5, (-0.516455160657608, 0.21800904325982232, -0.05002503924043772)), ('cz', 1, 0), ('u', 2, (3.3160876593419366, 0.6613514697864085, -2.9498361401440376)), ('u', 3, (-3.107577927801868, 0.24180153100131335, 2.725398761803262)), ('u', 1, (-0.1464617655355463, 1.1059759781713967, 0.12979889868104255)), ('u', 0, (-3.065137472118039, 0.7853982422009756, 3.1407867514401095)), ('u', 5, (-0.19268113051809058, 0.36691758663500634, -0.12489290891100432)), ('cz', 1, 0), ('u', 2, (0.5529122404706256, 0.10008139403023099, -0.20710168438656099)), ('u', 3, (0.009481441806211063, 0.4136301104545076, 0.1852669385705289)), ('u', 1, (3.0117937696416934, 1.5707962567027152, -2.456961495765487)), ('u', 0, (3.142398533867907, 2.616299151423349, -2.4569616077509973)), ('u', 5, (-2.7450832431691454, 1.5044247017835743, 3.1714501363368495)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 7, 8), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (-3.141592653589793, 1.3389725222944935, 3.141592653589793)), ('cz', 6, 7), ('u', 7, (0.0, 0.23182380450040305, 0.0)), ('cz', 8, 7), ('u', 7, (-3.141592653589793, 0.23182380450040305, 3.141592653589793)), ('cz', 6, 7), ('u', 7, (0.0, 1.3389725222944935, 0.0)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 7, 8), ('u', 3, (-0.011998651932780754, 0.035068744618698446, -0.009927966776869981)), ('u', 5, (-3.118408077145403, 8.341254967389955e-09, 3.1300003637502196)), ('u', 6, (3.0799739856600827, 0.04101951625127175, -3.175201664685296)), ('u', 4, (0.15270318509924602, 1.5707963180645836, -3.6543846698711135e-09)), ('u', 7, (-3.6749238119769676, 0.14412996902599418, 2.584848365873525)), ('cz', 5, 4), ('u', 3, (-0.009927966742459823, 0.033442075322526583, -0.008093153569313113)), ('u', 5, (-0.011592289857122701, 0.5235987754951184, -6.505802567675911e-09)), ('u', 6, (3.1079836425593723, 0.00022290334956562985, -3.152959083725753)), ('u', 4, (-3.529107380115216e-09, 0.523598775865786, 6.919391735800735e-05)), ('u', 7, (-0.5567442878288905, 0.09333323753923148, -0.45001944877466343)), ('cz', 5, 4), ('u', 3, (-0.008093153636521204, 0.03100252754737548, -0.006484194519471317)), ('u', 5, (-6.5420543392258335e-09, 1.5707963322971892, -0.03003248731706627)), ('u', 6, (3.1302262234116123, 1.529719592297703, 0.0015703886429401237)), ('u', 4, (3.1416618474185785, 1.5725046920439383, -3.101181087375395)), ('u', 7, (-0.4500194486291905, 0.2781111344697831, -0.2419468717914133)), ('cz', 5, 6), ('u', 3, (-0.006484194433627047, 0.02788255396182956, -0.005084810560235049)), ('u', 5, (3.1115601662193346, 1.9644974755453697, -3.0598092275346036)), ('u', 6, (3.1431630422315986, 1.5756628864443611, -0.0560709050552215)), ('u', 4, (0.040411566141154244, 0.0017139732709513032, 0.02499535504874261)), ('u', 7, (-0.2419468718423906, 0.32065403979562107, -0.1075942805260432)), ('cz', 5, 4), ('u', 3, (-0.005084810613438314, 0.024207170535394285, -0.0038744853192448432)), ('u', 5, (-3.059809227525655, 0.39848399027480996, 3.141592677496497)), ('u', 6, (3.085521748543134, 0.004897338332805505, -3.0994895928797357)), ('u', 4, (0.02499535504225195, 1.3993813386454823e-08, 0.13996816694155428)), ('u', 7, (-0.10759428056327414, 0.30114307748768226, -0.05120056142974744)), ('cz', 5, 6), ('u', 3, (-0.003874485332422318, 0.02009428710042876, -0.002829891589337413)), ('u', 5, (1.2870066012915515e-08, 0.3984839951383076, -0.08631632879508466)), ('u', 6, (0.04210306071591691, 0.80323408105442, -1.854085829658403e-08)), ('u', 4, (3.281560820607734, 8.719845644681125e-09, -2.90502094908558)), ('u', 7, (-0.05120056134528083, 0.29207557255071703, 0.049664416154679626)), ('cz', 5, 4), ('u', 3, (-0.002829891589588122, 0.015654826620762414, -0.0019258915303032504)), ('u', 5, (-0.08631632876698098, 1.5702104351477197, 0.010895163553423598)), ('u', 6, (3.1415926350686307, 0.8032340684189889, 0.0386198062833798)), ('u', 4, (-2.9050209490607397, 0.024789647960802845, 3.3613476276605736)), ('u', 7, (0.04966441621010443, 0.3181230954544873, 0.1275500313587809)), ('cz', 6, 7), ('u', 3, (-0.0019258915260883522, 0.010992808236069998, -0.0011362416795111148)), ('u', 5, (0.01089516367403582, 0.39438583187416293, 0.10283365860826715)), ('u', 6, (-3.102972847332388, 2.8796760336871428, 3.1268429577084658)), ('u', 4, (-2.9218376795000625, 0.019162103192931518, 3.3464864753681027)), ('u', 7, (3.2691426849289273, 1.5993447648568127, -0.010245007543836504)), ('cz', 5, 6), ('u', 3, (-0.0011362416747914867, 0.006205499937120086, -0.00043410405116450464)), ('u', 5, (0.10283365853007384, 0.005250612216072889, 1.229654780485201e-06)), ('u', 6, (-0.014749695915711764, 1.8326041378362559, -0.05675808125253269)), ('u', 4, (-2.936698831881056, 0.013365379715964415, 3.333327885337581)), ('u', 7, (3.1313476460548166, 0.0285544281829088, -3.150346969873966)), ('cz', 6, 7), ('u', 3, (-0.0004341040523492897, 0.0013836895457048843, 0.00020756594846420192)), ('u', 5, (-3.14159142395901, 0.005250595670552266, 3.064537988460174)), ('u', 6, (3.0848345584228554, 0.2634265832319953, -3.0867766015809632)), ('u', 4, (-2.949857421832063, 0.0065639168782822815, 3.3203616507843052)), ('u', 7, (3.1328383373143294, 0.007214997267805406, -3.1278215477172604)), ('cz', 5, 6), ('u', 3, (3.1418002195198937, 0.0033879211136994573, -3.1407769464413704)), ('u', 5, (-0.07705466513291019, 0.012064066316140654, -0.07249966970117298)), ('u', 6, (0.05481605201591043, 1.3088797028288943, -0.05243650089403663)), ('u', 4, (0.17876899717765427, 0.0023099115013583828, 0.16470245329335875)), ('u', 7, (3.155363759457692, 1.5635840616073071, 0.0010188476113417444)), ('cz', 6, 7), ('u', 3, (3.1424083607471442, 0.008030220964960526, -3.140175767869725)), ('u', 5, (-1.643295996502317, 1.5725395677602834, 1.5767650599885832)), ('u', 6, (-3.194029154466275, 0.5109292593725938, 3.069038656875536)), ('u', 4, (1.7354987800932298, 1.5132754165511737, -1.5695222465353704)), ('u', 7, (3.142611501175785, 1.60023235754839, -0.0018191973740395717)), ('cz', 5, 4), ('u', 3, (3.143009539303383, 0.012469131249627013, -3.1395555980269063)), ('u', 5, (1.5767650601072347, 0.6154797050427105, -1.0935137006446563)), ('u', 6, (3.0690386568654087, 0.5191225209576913, -3.1002343444807328)), ('u', 4, (1.5720704071213085, 0.6154797097037269, -1.5675748019018252)), ('u', 7, (-0.0018191974056423884, 0.024172558097420544, -0.0019053133573421854)), ('cz', 5, 4), ('u', 3, (3.1436297091392724, 0.01663502733022202, -3.1388913174911597)), ('u', 5, (2.048078952851612, 1.6168694345030314, -1.587100047237085)), ('u', 6, (3.1829509627314927, 2.1145173881119947, -0.04868326884435881)), ('u', 4, (-1.5675748019755396, 1.5696413567756455, 1.642788204071491)), ('u', 7, (-0.0019053133080495692, 0.018165194301069252, -0.001537521951944066)), ('cz', 5, 6), ('u', 3, (3.1442939896801985, 0.020462181721422237, -3.1381588532690596)), ('u', 5, (3.1252889331316185, 0.9553166185256176, -3.098979656315909)), ('u', 6, (3.0929093847390288, 1.570796343534117, 0.0005513990363770205)), ('u', 4, (-3.0696007762834774, 0.0038697493500351554, 3.1847176674592004)), ('u', 7, (-0.001537521962138161, 0.011414675394315833, -0.0006543839782734821)), ('cz', 5, 6), ('u', 3, (3.145999426008122, 0.024306044078736606, -3.136168317795093)), ('u', 5, (-3.1260808992185045, 0.7853981574050549, 3.141592662162083)), ('u', 6, (-3.13583945440354, 0.014439433724023936, 3.1384400222854185)), ('u', 4, (-3.1031216024398542, 0.0026812704966859917, 3.132629288626685)), ('u', 7, (0.00016945599671113875, 2.7901969406265238e-05, 0.0021370127558946084)), ('cz', 5, 4), ('u', 3, (3.1470169893657323, 0.027215365804092718, -3.1350187258135094)), ('u', 5, (8.588510223293177e-09, 0.7853981661272267, -5.7097120770185845e-09)), ('u', 6, (3.1384400222535827, 1.5563572172741813, -0.08097022914912522)), ('u', 4, (-3.150556018558537, 0.3173188418447972, 3.078454534587191)), ('u', 7, (3.143729666358712, 0.008041009902001532, -3.1368442389081657)), ('cz', 5, 6), ('u', 3, (3.1481665813959263, 0.02958570059649117, -3.133720308378973)), ('u', 5, (-7.224919709996777e-09, 0.7853981552422288, 4.939926870876897e-09)), ('u', 6, (3.060622424441615, 2.092354849933306, 0.003145599496788165)), ('u', 4, (-0.06313811903278108, 1.2697266125414508, -0.024073164687880056)), ('u', 7, (3.1463410682592032, 0.016472200295720375, -3.133585214724122)), ('cz', 5, 4), ('u', 3, (3.149464998810803, 0.03136023613261947, -3.132258540085515)), ('u', 5, (4.926433343212702e-09, 1.0471975265305735, -6.057048844095855e-09)), ('u', 6, (0.003145599502141147, 0.5253013898862602, -0.00545145978210719)), ('u', 4, (3.117519488887346, 0.7853981708664443, -3.141592673128462)), ('u', 7, (3.1496000924388747, 0.025081184697559405, -3.1297020648728147)), ('cz', 5, 4), ('u', 3, (3.15092676707859, 0.03248320803675271, -3.130621577715438)), ('u', 5, (3.141592647523828, 1.5707963173438424, -3.0165452970539737)), ('u', 6, (-0.005451459797216783, 0.523953607673068, 0.02744646447431627)), ('u', 4, (3.141592634091762, 0.6154797504055258, -3.0165452647593365)), ('u', 7, (3.15348324229967, 0.033655719295381435, -3.1252478838979543)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0))]), (9, [('u', 3, (0.0, 2.740499937812782, 0.0)), ('u', 2, (0.0, 0.6125547383393389, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 0.6125547383393389, 3.141592653589793)), ('u', 1, (0.0, 1.0365702822269822, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.0365702822269822, 3.141592653589793)), ('u', 0, (0.0, 1.3508083493994372, 0.0)), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 0), ('u', 0, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.3508083493994372, 3.141592653589793)), ('u', 5, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 7), ('u', 4, (1.6783995219433123, 2.677945059334841, -2.347528943738099)), ('u', 5, (-1.5846090155505541, 3.141592648803949, 0.9739539182276005)), ('u', 6, (-0.043128525936613135, 1.1363142332841882, 0.048276423210969005)), ('u', 7, (3.6131240518873415, 1.1107010877583263, -0.9418354587012361)), ('cz', 4, 5), ('u', 4, (-1.6523744495521213, 0.16800233720120406, 0.9527359405859664)), ('u', 5, (1.0712416136914256, 1.5707963153748048, -2.1972073505033416)), ('u', 6, (-0.768693759082117, 0.5531753246206115, 0.5396942354555023)), ('u', 7, (-2.9148007132765867, 0.2008037332194747, 3.017645632458623)), ('cz', 5, 6), ('u', 4, (0.22916288305642551, 0.37442170013424514, 0.49572334230405124)), ('u', 5, (2.197207529734123, 0.8860771116860425, -3.0442805131145416)), ('u', 6, (2.843357985790216, 0.3495555085185406, 0.4905298831427509)), ('u', 7, (0.2979688948426138, 0.24839385813200238, -0.43803588336134747)), ('cz', 4, 5), ('u', 4, (0.41570487214026375, 0.2787985107045163, 0.2414760855845465)), ('u', 5, (-0.09731230047945474, 1.5707963219500847, 0.18693697618542235)), ('u', 6, (3.369013850140026, 2.7578072190774874, -0.10129045285916272)), ('u', 7, (-0.15950293778007263, 0.6278305605369155, 0.29241996204870857)), ('cz', 5, 6), ('u', 4, (0.19929743529559463, 0.88167062540414, 0.15890588000448685)), ('u', 5, (1.142456664092194, 0.2517907327213283, -0.38435357509787066)), ('u', 6, (3.6071107386496584, 1.5707962929360881, 1.0300524832632352)), ('u', 7, (3.0661863018689504, 1.5717431102987751, -0.1241345695646574)), ('cz', 6, 7), ('u', 4, (-0.3120984854440979, 0.3324426283546392, -0.8392019353490036)), ('u', 5, (-0.9523727258677195, 1.5112007017232427, -1.782143104443146)), ('u', 6, (-2.8107096237423645, 1.2281134369081435, 3.454138659309974)), ('u', 7, (3.3599544039824045, 1.5363198332342687, 0.5738148273621144)), ('cz', 5, 6), ('u', 4, (-0.17407864683084506, 1.1981581532047667, 0.720754058352084)), ('u', 5, (-1.0485857025157856, 0.25106472349746944, -0.026165490427976112)), ('u', 6, (-0.1946523375171709, 0.34489327860458485, -0.577293313523412)), ('u', 7, (0.21409937561655093, 0.048860780375053335, 0.5401001834509469)), ('cz', 6, 7), ('u', 4, (0.08604625009239802, 0.25603321491570213, -0.6234998208627462)), ('u', 5, (-3.115427233337476, 0.25106470278189263, 3.0001440070116105)), ('u', 6, (-2.564299345768496, 0.34489330545421637, 3.14097197578436)), ('u', 7, (-0.7944434936537823, 0.15214440370723623, -0.38700938932873424)), ('cz', 5, 6), ('u', 4, (0.8539870509991602, 0.048184497575099156, 0.3465322584014089)), ('u', 5, (0.2753512792741515, 1.4910808473812084e-07, 0.5142367925003585)), ('u', 6, (-0.11727296544005196, 1.2281134650062855, 0.20497368134664418)), ('u', 7, (3.849810984935723, 1.7152699274463379, 0.6275369163340978)), ('cz', 6, 7), ('u', 4, (-0.03876308831078168, 0.8460430127672541, 0.8580307302996752)), ('u', 5, (2.840620213688528, 2.032147342582146e-07, -3.066629645265795)), ('u', 6, (2.3444126865023827, 1.656357328196517e-09, -2.662531125816978)), ('u', 7, (2.5619230084226126, 1.5707963266568443, -0.061782989421305334)), ('u', 8, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 0, 8), ('u', 2, (0.7253513784390777, 0.6044711094741366, 0.20469392790644375)), ('u', 3, (3.311225398804729, 0.45344431468038326, -2.811379919385577)), ('u', 1, (-2.967218355247499, 1.570796434542743, 2.782623640362077)), ('u', 0, (0.17437422015585258, 1.5707965042184875, 0.0936275536825773)), ('u', 5, (-2.981966217109752, 0.15731121643895493, 2.938590834577346)), ('cz', 1, 0), ('u', 2, (0.35596645780294867, 0.18572047518147186, 0.1993350213934021)), ('u', 3, (0.27426031921868355, 0.23461703245156662, 0.8733942285264623)), ('u', 1, (0.3589689683381108, 0.5235987755137496, 0.808167059420607)), ('u', 0, (-0.0936272595693031, 0.5235987756882645, 0.24164140029433875)), ('u', 5, (-0.3732167230268989, 0.1692158665066253, -0.5138610131235901)), ('cz', 1, 0), ('u', 2, (3.3065392993238065, 2.155371992839475, 0.5333711558032743)), ('u', 3, (0.17598851577546493, 0.15670123378522505, 0.3917956014566585)), ('u', 1, (0.762629607489512, 2.888760746641521, 1.273509717372987)), ('u', 0, (-2.8326884570670465, 1.2221833849116261, 2.9006620162668915)), ('u', 5, (-0.29514789830304344, 0.08656595906264068, -0.2957995850638988)), ('cz', 2, 1), ('u', 2, (3.137649879014554, 1.9634954073118636, -0.6075427985534696)), ('u', 3, (2.641964011958269, 0.22113794763596387, -2.499181790905139)), ('u', 1, (-1.2735092335544311, 1.570796389596646, 2.6537438706729946)), ('u', 0, (2.7190493967038063, 0.4335249098769316, -3.5464227887719253)), ('u', 5, (-0.6391916737665571, 0.1657229044953988, -0.024541329372776033)), ('cz', 2, 1), ('u', 2, (0.17775264740865682, 0.3058307733501329, -0.4417742235515568)), ('u', 3, (-0.24049984202753, 0.0778915304240662, -0.01766321037003693)), ('u', 1, (-3.046442949408996, 1.570796448575322, 2.117105098339699)), ('u', 0, (2.589347008940728, 0.28886655857903776, -3.6589389197843123)), ('u', 5, (0.0508440614483701, 0.07203248788002012, 0.029932533667720446)), ('cz', 1, 0), ('u', 2, (2.424643232393627, 1.6965841706658102, 0.29519056307839975)), ('u', 3, (-2.9031023199264445, 0.06550196451547159, 3.0490689833889255)), ('u', 1, (-0.5463092573212669, 2.748893521815632, -0.6787734947855513)), ('u', 0, (-0.8667873688727257, 0.19392965245174282, 0.26794479272692207)), ('u', 5, (3.093960255811586, 0.3684130596408908, -2.752177636430363)), ('cz', 2, 1), ('u', 2, (0.48622642934218985, 4.965240116914444e-08, 3.1314145386877534)), ('u', 3, (3.4555174636846955, 1.825677016721441, -0.10375945734790903)), ('u', 1, (3.1274454219227774, 0.3987010753799351, -2.7245703058001975)), ('u', 0, (-0.7836119278695097, 0.44642916421938117, -0.010954000022640242)), ('u', 5, (-0.35407245584296543, 0.1009018219745298, 0.12449653021465411)), ('cz', 2, 3), ('u', 2, (-2.95124738442682, 2.872225615416436, 3.168474353976826)), ('u', 3, (2.98864467816622, 1.8770178510291984, 0.03374189545665063)), ('u', 1, (-0.04688921667005741, 0.742010041871667, -0.42431748027242494)), ('u', 0, (2.517451732666387, 0.19291873478940372, -2.9271814483694167)), ('u', 5, (-0.6494532035971243, 0.6081156702407481, 0.4080258856521049)), ('cz', 2, 1), ('u', 2, (0.20730116474291263, 1.8331478973531157, 0.46325541824844596)), ('u', 3, (2.692541925224733, 0.332696125034078, -2.6878777969285363)), ('u', 1, (2.490863927966439, 0.283599555584151, -3.18962913786471)), ('u', 0, (0.17017190346421693, 0.20177925921584974, -0.08704916348372774)), ('u', 5, (2.7417475746980577, 0.5655452491435636, -3.0339198029229153)), ('cz', 2, 3), ('u', 2, (-2.8611849940478558, 0.34985907247470327, 2.4858926099614247)), ('u', 3, (2.87078395521952, 0.16840848054093654, -3.2915600146968)), ('u', 1, (0.048036261016780685, 0.28359965336885995, -0.2578628245055713)), ('u', 0, (-0.04579375400203432, 0.37476347219220135, -0.0019053945699001838)), ('u', 5, (-2.9901835580399307, 0.29429615142429255, 2.5298877934002943)), ('cz', 2, 1), ('u', 2, (-0.056927005043825446, 1.3014292976901551, -0.5532071820595532)), ('u', 3, (3.4785109576898656, 1.40526859119825, -0.254041621662799)), ('u', 1, (2.6370874705266063, 0.27233958144772935, -3.545853570337978)), ('u', 0, (3.105802818900175, 0.21107958032122495, -2.861471392990672)), ('u', 5, (3.0189880563377156, 0.19670772384256013, -3.0445309207076874)), ('cz', 2, 3), ('u', 2, (0.01339029507980527, 0.0416754087342334, -1.347371724509553)), ('u', 3, (3.499319469952347, 1.355519791945654, -0.5991567935202315)), ('u', 1, (-2.1086415347436596, 1.3486442481739653, 1.9995738248248909)), ('u', 0, (1.6776715535216509, 1.3775652646994023, -2.0667194984618944)), ('u', 5, (-0.9230392567029317, 0.3689502539357166, -0.027637551969518004)), ('cz', 1, 0), ('u', 2, (-0.6422086949145018, 0.6564240419907547, 0.3050682210391836)), ('u', 3, (-0.3570995437419664, 0.3230608004096355, -0.4818739743945121)), ('u', 1, (0.9831653765370997, 0.6154796885409349, -2.044379448865771)), ('u', 0, (1.2690644052694249, 0.6154797227608092, -1.450257109340972)), ('u', 5, (0.32441597127291805, 0.35884350817350413, -0.5202647635263868)), ('cz', 1, 0), ('u', 2, (3.0849500536513323, 0.5822022691130421, -0.24393556074215605)), ('u', 3, (-0.07734822951641647, 0.4435772553137907, -0.06596287366372468)), ('u', 1, (1.1222045403932766, 1.2721557575140932, -1.643452931303447)), ('u', 0, (-1.518587141618921, 1.6648415595850694, 1.767035244953014)), ('u', 5, (-2.8821138871253633, 0.6916920960996875, 2.47733528484548)), ('cz', 2, 1), ('u', 2, (2.2401631579399766, 1.57079612890024, -0.6250479515489906)), ('u', 3, (0.019069698632826204, 0.004875598650025819, 0.48157043024513996)), ('u', 1, (2.9948115830243927, 0.9587582103095811, -3.1531878704143894)), ('u', 0, (2.558845213379647, 0.0026409094544454616, -3.144567419915024)), ('u', 5, (-0.24806262907427706, 0.5508837005772087, -0.9339898849771324)), ('cz', 2, 1), ('u', 2, (-2.9673015967720766, 0.08948535665168787, 3.1599312277533493)), ('u', 3, (2.3790348862594284, 0.2712248006470355, -3.2040591439620414)), ('u', 1, (-3.129997892854534, 0.7853981335305925, 3.0269609858112005)), ('u', 0, (3.0359102852588746, 0.1938390933531295, -2.778825432657605)), ('u', 5, (2.705457550215509, 0.48649144979514203, -2.964461900581061)), ('cz', 1, 0), ('u', 2, (3.572868238545496, 1.5598263045480647, -0.19509687224531835)), ('u', 3, (2.998883524589488, 0.15059171477843603, -3.0599404762732667)), ('u', 1, (0.11463202022691882, 0.7853981649586114, 0.17677739302616385)), ('u', 0, (0.25127933585639883, 0.48994533516927113, -0.3547313940345701)), ('u', 5, (-3.157884001950231, 0.5858331264680662, 2.9538041371221344)), ('cz', 2, 1), ('u', 2, (3.3460062868220524, 0.5910844552042714, -0.052680694256529215)), ('u', 3, (-3.2368160309582397, 0.17716874842237512, 2.754312610522317)), ('u', 1, (-0.1767774518436646, 2.356194531997464, 0.14646174769130244)), ('u', 0, (-0.02761925796749795, 1.7818724057866364, 0.1473634861913784)), ('u', 5, (-0.516455160657608, 0.21800904325982232, -0.05002503924043772)), ('cz', 1, 0), ('u', 2, (3.3160876593419366, 0.6613514697864085, -2.9498361401440376)), ('u', 3, (-3.107577927801868, 0.24180153100131335, 2.725398761803262)), ('u', 1, (-0.1464617655355463, 1.1059759781713967, 0.12979889868104255)), ('u', 0, (-3.065137472118039, 0.7853982422009756, 3.1407867514401095)), ('u', 5, (-0.19268113051809058, 0.36691758663500634, -0.12489290891100432)), ('cz', 1, 0), ('u', 2, (0.5529122404706256, 0.10008139403023099, -0.20710168438656099)), ('u', 3, (0.009481441806211063, 0.4136301104545076, 0.1852669385705289)), ('u', 1, (3.0117937696416934, 1.5707962567027152, -2.456961495765487)), ('u', 0, (3.142398533867907, 2.616299151423349, -2.4569616077509973)), ('u', 5, (-2.7450832431691454, 1.5044247017835743, 3.1714501363368495)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 7, 8), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (-3.141592653589793, 1.3389725222944935, 3.141592653589793)), ('cz', 6, 7), ('u', 7, (0.0, 0.23182380450040305, 0.0)), ('cz', 8, 7), ('u', 7, (-3.141592653589793, 0.23182380450040305, 3.141592653589793)), ('cz', 6, 7), ('u', 7, (0.0, 1.3389725222944935, 0.0)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 7, 8), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (0.0, 1.5707963267948966, 0.0)), ('cz', 4, 5), ('u', 5, (0.0, 0.5235987755982989, 0.0)), ('u', 4, (0.0, 0.5235987755982989, 0.0)), ('cz', 4, 5), ('u', 5, (0.0, 1.5707963267948966, 0.0)), ('u', 4, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 5, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 5, (-3.141592653589793, 1.1780972450961724, 3.141592653589793)), ('cz', 4, 5), ('u', 5, (0.0, 0.39269908169872414, 0.0)), ('cz', 6, 5), ('u', 5, (-3.141592653589793, 0.39269908169872414, 3.141592653589793)), ('cz', 4, 5), ('u', 5, (0.0, 1.1780972450961724, 0.0)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 5, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 6, 7), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (-3.141592653589793, 1.3089969389957472, 3.141592653589793)), ('cz', 5, 6), ('u', 6, (0.0, 0.26179938779914935, 0.0)), ('cz', 7, 6), ('u', 6, (-3.141592653589793, 0.26179938779914935, 3.141592653589793)), ('cz', 5, 6), ('u', 6, (0.0, 1.3089969389957472, 0.0)), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 6, 7), ('u', 7, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 5, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 4, 5), ('u', 5, (1.570796326794898, 0.6154797086703868, -1.570796326794898)), ('u', 4, (1.5707963267948966, 0.6154797086703876, -1.5707963267948966)), ('cz', 4, 5), ('u', 5, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 4, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 5, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 5, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 5, (-3.141592653589793, 1.2630564724597029, 3.141592653589793)), ('cz', 4, 5), ('u', 5, (0.0, 0.30773985433519363, 0.0)), ('cz', 6, 5), ('u', 5, (-3.141592653589793, 0.30773985433519363, 3.141592653589793)), ('cz', 4, 5), ('u', 5, (0.0, 1.2630564724597029, 0.0)), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 5, 6), ('u', 6, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (0.0, 1.5707963267948966, 0.0)), ('u', 5, (0.0, 1.5707963267948966, 0.0)), ('cz', 4, 5), ('u', 5, (1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17)), ('u', 4, (-3.141592653589793, 0.7853981633974483, 3.141592653589793)), ('cz', 4, 5), ('u', 5, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 4, (-3.141592653589793, 1.5707963267948966, 3.141592653589793))])], (13, 5): [(13, [('u', 5, (0.0, 2.8854165799347697, 0.0)), ('u', 4, (0.0, 0.4194735802213585, 0.0)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 5, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (-3.141592653589793, 0.4194735802213585, 3.141592653589793)), ('u', 3, (0.0, 0.7825572390207782, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 4, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (-3.141592653589793, 0.7825572390207782, 3.141592653589793)), ('u', 2, (0.0, 1.0959425247457042, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.0959425247457042, 3.141592653589793)), ('u', 1, (0.0, 1.3361538466137453, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.3361538466137453, 3.141592653589793)), ('u', 8, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 5, 8), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 9, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 4, 9), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 10, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 11, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 11), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 12, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 12), ('u', 0, (0.2895396804257915, 0.42053432963085197, 0.2186344537665792)), ('u', 1, (-2.172596698081646, 2.7771176198164215e-09, 2.5876963239152198)), ('u', 2, (-3.228353500519151, 0.14952614669551412, 3.045051894219899)), ('cz', 0, 1), ('u', 0, (2.9792540206708886, 0.8208288587674605, -3.2394362427156973)), ('u', 1, (-0.5189128758343072, 1.5707963278808252, 0.28787753873452127)), ('u', 2, (3.2208886381194444, 1.4212920830783036, 0.02240296332127567)), ('cz', 1, 2), ('u', 0, (0.15665480505572835, 0.40150716667742453, -0.24764615828620704)), ('u', 1, (2.85371509660119, 0.955316603180468, -3.1457179085982)), ('u', 2, (2.87772559676124, 1.3299879385880082, -0.18603273356959749)), ('cz', 0, 1), ('u', 0, (0.5338595447223321, 0.019280386943774562, -0.026246824335227292)), ('u', 1, (0.00412526566754981, 1.5707963120452608, 0.197486676731702)), ('u', 2, (3.119123150084535, 0.7395037326590929, 0.17542817108495523)), ('cz', 1, 2), ('u', 0, (-3.1153459688867526, 0.019280386430582032, 2.8536491653853044)), ('u', 1, (2.944105989056777, 1.5707963266274152, -3.03777941513755)), ('u', 2, (3.3217938815324324, 1.5707963268105951, 0.10381323861495151)), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.1780972450961724, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 0.39269908169872414, 0.0)), ('cz', 3, 2), ('u', 2, (-3.141592653589793, 0.39269908169872414, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 1.1780972450961724, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (-3.141592653589793, 1.2630564724597029, 3.141592653589793)), ('cz', 2, 3), ('u', 3, (0.0, 0.30773985433519363, 0.0)), ('cz', 4, 3), ('u', 3, (-3.141592653589793, 0.30773985433519363, 3.141592653589793)), ('cz', 2, 3), ('u', 3, (0.0, 1.2630564724597029, 0.0)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 4, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (-3.141592653589793, 1.3605291591529143, 3.141592653589793)), ('cz', 3, 4), ('u', 4, (0.0, 0.2102671676419825, 0.0)), ('cz', 5, 4), ('u', 4, (-3.141592653589793, 0.2102671676419825, 3.141592653589793)), ('cz', 3, 4), ('u', 4, (0.0, 1.3605291591529143, 0.0)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 4, 5), ('u', 6, (-2.7956955245943567, 0.006638396074130232, 1.9839531812880271)), ('u', 7, (3.3060693547306053, 2.57765001221808, 2.7878048943093074)), ('u', 8, (-0.9802959536225644, 3.1415926212758114, -5.141306813857595)), ('u', 5, (3.1836423183974443, 1.368608423331714, -0.35706132391778844)), ('u', 9, (3.282137172694394, 0.22185546077212015, -2.37617977051081)), ('cz', 7, 8), ('u', 6, (-0.863687536887122, 0.2036612508775408, -0.49096140311386094)), ('u', 7, (2.8866631605901194, 2.0572489031345156, -2.832204402499025)), ('u', 8, (0.8352656017295159, 1.5707963294313407, 0.3463688831494805)), ('u', 5, (-2.899488852373683, 0.023818332706368017, 2.986727851963161)), ('u', 9, (2.4208820653955025, 1.3491553116568324, 0.8806290474140755)), ('cz', 8, 9), ('u', 6, (2.832707855812823, 0.15840937628477464, -3.115598061390125)), ('u', 7, (-2.937596754155037, 0.6717279144270676, 3.0266444728300304)), ('u', 8, (2.7952252909671804, 0.8570719478274053, -3.0533719187797725)), ('u', 5, (0.12071465626978571, 0.013907386342634304, 0.332968572639534)), ('u', 9, (0.9569479039919327, 1.094241866608657, 0.41378535469459476)), ('cz', 7, 8), ('u', 6, (-0.1186231744977771, 0.12081883678314975, 0.8833194869947525)), ('u', 7, (2.8448936708527257, 2.0093937785142094e-08, -3.234615357195483)), ('u', 8, (-0.08822174150423814, 1.5707963267868776, 0.47942481146490246)), ('u', 5, (-0.4232632365402874, 0.07836685227309081, 0.06556582823325244)), ('u', 9, (3.4238520709664293, 1.3817784233230703, 0.7518585830252498)), ('cz', 8, 9), ('u', 6, (3.227019319836882, 0.17251920947071164, -3.001264902666509)), ('u', 7, (-0.4519555042207237, 2.180643064547514e-08, -0.17549316113966626)), ('u', 8, (-0.4794260024013855, 1.5707963376176648, -0.18415260485211554)), ('u', 5, (-0.004473691871499111, 0.13802087244552758, 0.17356360142769778)), ('u', 9, (3.4455507137660275, 1.5707963267535443, -0.1841526048505968)), ('u', 0, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 1, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 0, 1), ('u', 1, (1.570796326794898, 0.4636476090008061, -1.570796326794898)), ('u', 0, (1.5707963267948966, 0.4636476090008061, -1.5707963267948966)), ('cz', 0, 1), ('u', 1, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 0, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 1, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.1277577648985897, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.44303856189630686, 0.0)), ('cz', 2, 1), ('u', 1, (-3.141592653589793, 0.44303856189630686, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 1.1277577648985897, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.2284367252937551, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 0.3423596015011414, 0.0)), ('cz', 3, 2), ('u', 2, (-3.141592653589793, 0.3423596015011414, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 1.2284367252937551, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (-3.141592653589793, 1.3389725222944935, 3.141592653589793)), ('cz', 2, 3), ('u', 3, (0.0, 0.23182380450040305, 0.0)), ('cz', 4, 3), ('u', 3, (-3.141592653589793, 0.23182380450040305, 3.141592653589793)), ('cz', 2, 3), ('u', 3, (0.0, 1.3389725222944935, 0.0)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.5235987755982989, 0.0)), ('u', 0, (0.0, 0.5235987755982989, 0.0)), ('cz', 0, 1), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.1780972450961724, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.39269908169872414, 0.0)), ('cz', 2, 1), ('u', 1, (-3.141592653589793, 0.39269908169872414, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 1.1780972450961724, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.3089969389957472, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 0.26179938779914935, 0.0)), ('cz', 3, 2), ('u', 2, (-3.141592653589793, 0.26179938779914935, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 1.3089969389957472, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 1, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 0, 1), ('u', 1, (1.570796326794898, 0.6154797086703868, -1.570796326794898)), ('u', 0, (1.5707963267948966, 0.6154797086703876, -1.5707963267948966)), ('cz', 0, 1), ('u', 1, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 0, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 1, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.2630564724597029, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.30773985433519363, 0.0)), ('cz', 2, 1), ('u', 1, (-3.141592653589793, 0.30773985433519363, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 1.2630564724597029, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('cz', 0, 1), ('u', 1, (1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17)), ('u', 0, (-3.141592653589793, 0.7853981633974483, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 9, (-3.141592653589793, 1.2139341373225139, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 0.35686218947238274, 0.0)), ('cz', 10, 9), ('u', 9, (-3.141592653589793, 0.35686218947238274, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 1.2139341373225139, 0.0)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 10, 11), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 10, (-3.141592653589793, 1.2888250061145825, 3.141592653589793)), ('cz', 9, 10), ('u', 10, (0.0, 0.2819713206803144, 0.0)), ('cz', 11, 10), ('u', 10, (-3.141592653589793, 0.2819713206803144, 3.141592653589793)), ('cz', 9, 10), ('u', 7, (-0.0010222009133052934, 1.3041008747344632e-09, 0.2003370551542191)), ('u', 8, (0.023930023575935412, 0.2953173773967398, -0.12864126913934845)), ('u', 9, (-0.030708083689677756, 0.12471016231533803, -0.5936932650946682)), ('u', 6, (0.07121035154428908, 0.4205343267071202, -0.2832141972956279)), ('u', 4, (0.07530003087270047, 0.16373930134914844, -0.26222930989690885)), ('cz', 7, 6), ('u', 7, (-0.12687198917430412, 1.5707963360271298, 0.2038898277475962)), ('u', 8, (3.2812013072679873, 1.866096954858985, 0.30184449475142205)), ('u', 9, (-0.2750715762601466, 0.1444293365187298, -0.5619473575075802)), ('u', 6, (-3.0846318122760255, 0.7274540347580297, 2.685237625937257)), ('u', 4, (-2.6959001663774766, 0.023390897362289824, 3.3774526330675045)), ('cz', 7, 8), ('u', 7, (2.937702797807351, 0.9553165545708295, -3.0503475368792787)), ('u', 8, (3.3207345893489655, 1.8839026748016603, -0.007233571707222719)), ('u', 9, (0.7067021023968258, 0.1861341116737401, 0.6824799267543968)), ('u', 6, (0.16900917346851702, 0.3290496173131335, 0.14955942962615904)), ('u', 4, (0.13632596908237446, 0.5047675908256389, 0.19005678333059065)), ('cz', 7, 6), ('u', 7, (-0.0912450930625181, 1.5707962664208395, 0.03309400916940823)), ('u', 8, (3.425558647277073, 1.3322673600128185, -0.027285494896139628)), ('u', 9, (2.9343637010964203, 0.3849224661080576, -3.181083530011608)), ('u', 6, (3.0294580093363934, 0.13808492223951388, -3.0008784410797094)), ('u', 4, (2.808193939953363, 0.34629612135864385, -3.327594214191119)), ('cz', 7, 8), ('u', 7, (-3.0107057097410745, 1.034782101861369, 3.2649574529238565)), ('u', 8, (2.701874333399131, 1.5707963333587613, 0.01699495886877811)), ('u', 9, (3.0218224318300324, 1.613517291586055, 0.20240124091333955)), ('u', 6, (-0.19260732367119549, 0.0506388373239212, 0.49460918963778)), ('u', 4, (3.2519730805535927, 0.002414860976536661, -2.897569812225165)), ('cz', 8, 9), ('u', 7, (2.704908443716258, 0.5581851719223989, -3.2239761040741026)), ('u', 8, (-2.7715068751111733, 1.1780352387189146, 2.8119235453642313)), ('u', 9, (3.2650506749779575, 1.7643530058585124, 0.13265112018338687)), ('u', 6, (0.524846134475885, 0.13013042238544234, 0.6969355712505834)), ('u', 4, (0.31766617169078154, 0.29627032380597074, -0.25692668698661764)), ('cz', 7, 8), ('u', 7, (2.9242210810494496, 0.07895765790608032, -3.0403033743301244)), ('u', 8, (0.28854329030454917, 0.39306029653695296, -0.018521965452702627)), ('u', 9, (-2.4666500872262773, 0.27631055933330684, 3.652088429466448)), ('u', 6, (-2.8645903242081086, 0.00811891627662618, 2.960010214724399)), ('u', 4, (-3.0255760348888474, 0.3172366772250132, 2.742544186481971)), ('cz', 8, 9), ('u', 7, (-0.10128924503841866, 0.07895765916161333, -0.54812026936145)), ('u', 8, (-3.1230706552997467, 0.39306028706200213, 2.9768939074869922)), ('u', 9, (0.22834942953139625, 0.16951111093992396, -0.46976825774972053)), ('u', 6, (-0.05157190861832458, 0.2831035587339365, 0.040556641699272526)), ('u', 4, (-2.9585527433127945, 0.18789853193368702, 3.150001059291019)), ('cz', 7, 8), ('u', 7, (0.30452840482299254, 1.497244522071447e-07, -0.21779698434427083)), ('u', 8, (0.20582453884073654, 1.1780352293228151, 0.36255569092721357)), ('u', 9, (2.5401593790964863, 1.6526078360053855, 0.15843675001271862)), ('u', 6, (-3.2179817957037806, 0.2740507804104168, 2.703112265700859)), ('u', 4, (-3.275513452054124, 0.15846558147607348, 2.9939277750514792)), ('cz', 8, 9), ('u', 7, (3.459547899859897, 1.5115771072655002e-07, -2.5026524551797933)), ('u', 8, (2.92342169170971, 2.5221885901529127e-09, -3.245476641550241)), ('u', 9, (2.834527942346747, 1.5707963245157934, 0.01822865285915709)), ('u', 6, (-2.6012871529086037, 0.10687131265079321, 1.7398627461175855)), ('u', 4, (0.05015276160771126, 0.10355943414250254, -0.4269578465944131)), ('u', 10, (0.0, 1.2888250061145825, 0.0)), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 10, 11), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 11, 12), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 11, (-3.141592653589793, 1.3769979834673063, 3.141592653589793)), ('cz', 10, 11), ('u', 11, (0.0, 0.1937983433275903, 0.0)), ('cz', 12, 11), ('u', 11, (-3.141592653589793, 0.1937983433275903, 3.141592653589793)), ('cz', 10, 11), ('u', 11, (0.0, 1.3769979834673063, 0.0)), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 11, 12), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 9, (-3.141592653589793, 1.2630564724597029, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 0.30773985433519363, 0.0)), ('cz', 10, 9), ('u', 9, (-3.141592653589793, 0.30773985433519363, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 1.2630564724597029, 0.0)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 10, 11), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 10, (-3.141592653589793, 1.3605291591529143, 3.141592653589793)), ('cz', 9, 10), ('u', 10, (0.0, 0.2102671676419825, 0.0)), ('cz', 11, 10), ('u', 10, (-3.141592653589793, 0.2102671676419825, 3.141592653589793)), ('cz', 9, 10), ('u', 10, (0.0, 1.3605291591529143, 0.0)), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 10, 11), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 7, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 6, 7), ('u', 7, (1.570796326794898, 0.4636476090008061, -1.570796326794898)), ('u', 6, (1.5707963267948966, 0.4636476090008061, -1.5707963267948966)), ('cz', 6, 7), ('u', 7, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 6, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 7, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 7, 8), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (-3.141592653589793, 1.1277577648985897, 3.141592653589793)), ('cz', 6, 7), ('u', 7, (0.0, 0.44303856189630686, 0.0)), ('cz', 8, 7), ('u', 7, (-3.141592653589793, 0.44303856189630686, 3.141592653589793)), ('cz', 6, 7), ('u', 7, (0.0, 1.1277577648985897, 0.0)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 7, 8), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 8, 9), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 8, (-3.141592653589793, 1.2284367252937551, 3.141592653589793)), ('cz', 7, 8), ('u', 8, (0.0, 0.3423596015011414, 0.0)), ('cz', 9, 8), ('u', 8, (-3.141592653589793, 0.3423596015011414, 3.141592653589793)), ('cz', 7, 8), ('u', 8, (0.0, 1.2284367252937551, 0.0)), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 8, 9), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 9, (-3.141592653589793, 1.3389725222944935, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 0.23182380450040305, 0.0)), ('cz', 10, 9), ('u', 9, (-3.141592653589793, 0.23182380450040305, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 1.3389725222944935, 0.0)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 6, (-0.10337358284045067, 1.5707963975858181, 1.015835984878588e-06)), ('u', 7, (3.038218917454895, 1.5707961616258277, -3.1415926581399134)), ('u', 8, (0.019828976040683857, 0.013724809664014377, 0.01783890191797663)), ('u', 9, (0.0035274908852903856, 0.07434633330967455, 0.0010581438542694391)), ('cz', 6, 7), ('u', 6, (5.959103240659553e-07, 0.5235987669981246, 0.0008503728370569114)), ('u', 7, (-7.864374259467977e-07, 0.5235987850137847, 2.1514876648522244e-08)), ('u', 8, (0.017838485853208828, 3.491322085471117e-05, 0.015992763508435617)), ('u', 9, (0.0010573111958433138, 0.055537721042183685, -0.00015720448849646705)), ('cz', 6, 7), ('u', 6, (-3.1407408931064547, 1.561653060584754, 3.108194496565501)), ('u', 7, (1.2678335777061012e-07, 1.5707963436427994, 0.1330778157276038)), ('u', 8, (3.1575855108403723, 1.5845243547676091, 0.00046389001286240017)), ('u', 9, (-0.00015720450130723025, 0.03568220976022034, 0.0007906627104473196)), ('cz', 7, 8), ('u', 6, (-0.03339896159371908, 0.0009456606723681459, -0.053307384901516736)), ('u', 7, (-3.008514747676956, 1.1780973866952744, 3.1415924849124677)), ('u', 8, (3.1420579084520557, 1.5707964334536435, 0.013241139958231063)), ('u', 9, (0.0007889975112366238, 0.012161875644356494, 0.005072193771559766)), ('cz', 7, 8), ('u', 6, (-3.1948983668400106, 0.01024058544541974, 3.0631718212986123)), ('u', 7, (3.141592626777089, 1.9634956698150956, -3.141592497081211)), ('u', 8, (0.013241406399454505, 0.811795812113978, -5.1901048898732316e-08)), ('u', 9, (3.14666567994145, 0.0183872739501306, -3.1274161263819162)), ('cz', 6, 7), ('u', 6, (-0.07842081277498814, 0.0636763764592665, -0.11265482417680511)), ('u', 7, (-1.5174894674825287e-07, 2.748894824926096, 0.004717116185436816)), ('u', 8, (3.1415931115099625, 0.8117960196269799, 0.0024698603439405886)), ('u', 9, (3.155767515687325, 0.057705323320175354, -3.1121926183807496)), ('cz', 7, 8), ('u', 6, (-0.11265400467920675, 0.036843437586738984, -0.14917897433769264)), ('u', 7, (0.004716613199078894, 0.21246252000648394, -0.010068637593983236)), ('u', 8, (0.0024699010658877363, 3.723199423905932e-07, 3.156587806026825)), ('u', 9, (3.1709922127383923, 1.6717308694221615, 0.006357761268223072)), ('cz', 8, 9), ('u', 6, (-0.149179819856092, 0.0073166448209931205, -0.18873465895337813)), ('u', 7, (-0.010068429561566238, 0.180256711353873, -0.025384936550044934)), ('u', 8, (-3.1265982994149426, 2.879792808347136, 3.14262416800779)), ('u', 9, (3.147950783718318, 1.5810194225609866, 0.0051794225141676975)), ('cz', 7, 8), ('u', 6, (-3.3303273125335644, 0.012528338239947763, 2.9115537090683197)), ('u', 7, (-0.02538514458468043, 8.543096018313841e-07, -0.020319696249047118)), ('u', 8, (0.0010318785259384909, 1.8325959332673774, 0.00398736224364416)), ('u', 9, (-3.1364136051101927, 0.010223660278832961, 3.135896011527404)), ('cz', 8, 9), ('u', 6, (-3.371630765613361, 0.01443821671985917, 2.871845946781179)), ('u', 7, (-0.02032094450144711, 1.8712083349506447e-07, -0.01330079721428616)), ('u', 8, (-3.137607417068711, 0.26180723510489173, 3.137741517763965)), ('u', 9, (-0.005695809571485449, 0.00403625252390588, 0.011325767365440383)), ('cz', 7, 8), ('u', 6, (-0.2697483848758344, 0.0015061147350338621, -0.341535392261783)), ('u', 7, (-3.1548932427496057, 0.006187151643163442, 3.1248315042568446)), ('u', 8, (-0.003849809993246951, 1.308996287788808, 0.01065630030694045)), ('u', 9, (3.1529182975762176, 1.5748316101961628, 0.0014652491448243765)), ('cz', 8, 9), ('u', 6, (1.2292585095449935, 1.5295686302113742, -1.5278527879141417)), ('u', 7, (-1.5875594496229066, 1.5705884974312958, 1.5677044976095087)), ('u', 8, (-3.1309342313072106, 0.49019375451821573, 3.0891773474553403)), ('u', 9, (3.1430586698649674, 1.543729458425524, 0.05079693796372453)), ('cz', 6, 7), ('u', 6, (1.613738941174663, 0.6154797193336008, -1.5693249595122236)), ('u', 7, (1.5677041944582901, 0.6154796712782338, -1.0941551216460745)), ('u', 8, (3.089176372066575, 0.5197091220145889, -3.1124591545135134)), ('u', 9, (-3.0907953812292854, 0.02205876862949908, 3.1797309234268294)), ('cz', 6, 7), ('u', 6, (-1.5693234376800447, 1.5710268263607212, 1.5320474481219526)), ('u', 7, (2.0474392804567048, 1.6465095989641034, -1.5976301429120348)), ('u', 8, (3.170724942066729, 2.133202830195918, 0.04435021715472276)), ('u', 9, (-3.103454591900695, 0.015664100610430897, 3.16660790256985)), ('cz', 7, 8), ('u', 6, (3.1028434940412866, 0.0029342574022471267, -3.170238681167906)), ('u', 7, (3.1147605899518007, 0.9553166913707994, -3.159526019221204)), ('u', 8, (3.1859433528692485, 1.5707966240942846, -0.00899140872696691)), ('u', 9, (-3.116576572034678, 0.00825150306986584, 3.153170208668008)), ('cz', 7, 8), ('u', 6, (3.1063913060080957, 1.881494231841266e-05, -3.1706637323774425)), ('u', 7, (3.124081332256533, 0.7853981594575237, -3.1415930460131394)), ('u', 8, (3.084263402421209, 0.022685295311754557, -3.10842371493674)), ('u', 9, (-0.0019941521657814727, 0.0006813777759123335, -0.01574928663732262)), ('cz', 6, 7), ('u', 6, (3.1125213667304164, 0.3092041646086194, -3.12625345487787)), ('u', 7, (-4.080796685731872e-08, 0.7853981558924876, 7.772173463874602e-07)), ('u', 8, (3.1747613891209787, 1.5481628904529872, -0.009805768630532175)), ('u', 9, (-0.015748454043375226, 0.008431297425970982, -0.0295910929262283)), ('cz', 7, 8), ('u', 6, (0.01533891114478319, 1.2643549581204578, 0.0057153799668915855)), ('u', 7, (-5.018874719283185e-07, 0.7853980462868956, 6.318653877145332e-07)), ('u', 8, (3.131787219964067, 2.0895288501316824, 0.0025644477643140817)), ('u', 9, (-0.029591092924892516, 0.015601263069330702, -0.04344267924801568)), ('cz', 6, 7), ('u', 6, (-3.1358763000417835, 0.7853983234787361, 3.141594348345165)), ('u', 7, (-7.034785056979964e-07, 1.0471960345973994, 3.6104177568044136e-07)), ('u', 8, (0.0025637937964505214, 0.5245194617938411, -0.00442326163518103)), ('u', 9, (-0.043441430274428344, 0.02175985552611398, -0.05718132719321954)), ('cz', 6, 7), ('u', 6, (3.14159139058198, 0.6154825200535677, -3.1045753624373456)), ('u', 7, (3.141592370965353, 1.570796391465584, -3.1045751262997925)), ('u', 8, (-0.004423053508179188, 0.5275553257727209, -0.011433642825968456)), ('u', 9, (-0.057182576164625784, 0.026436670338707852, -0.07062164023534258)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0))]), (13, [('u', 5, (0.0, 2.8854165799347697, 0.0)), ('u', 4, (0.0, 0.4194735802213585, 0.0)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 5, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (-3.141592653589793, 0.4194735802213585, 3.141592653589793)), ('u', 3, (0.0, 0.7825572390207782, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 4, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (-3.141592653589793, 0.7825572390207782, 3.141592653589793)), ('u', 2, (0.0, 1.0959425247457042, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.0959425247457042, 3.141592653589793)), ('u', 1, (0.0, 1.3361538466137453, 0.0)), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 1), ('u', 1, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.3361538466137453, 3.141592653589793)), ('u', 8, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 5, 8), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 9, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 4, 9), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 10, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 11, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 11), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 12, (0.0, 3.141592653589793, -3.141592653589793)), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 12), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (-6.026816351730587e-17, 0.4205343352839649, 4.75701228604804e-18)), ('u', 0, (-6.026816351730587e-17, 0.4205343352839649, 4.75701228604804e-18)), ('cz', 0, 1), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 0.6154797086703874, 3.141592653589793)), ('cz', 2, 1), ('u', 1, (0.0, 0.6154797086703874, 0.0)), ('u', 1, (-3.141592653589793, 2.186276035465284, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 2.186276035465284, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.1780972450961724, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 0.39269908169872414, 0.0)), ('cz', 3, 2), ('u', 2, (-3.141592653589793, 0.39269908169872414, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 1.1780972450961724, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (-3.141592653589793, 1.2630564724597029, 3.141592653589793)), ('cz', 2, 3), ('u', 3, (0.0, 0.30773985433519363, 0.0)), ('cz', 4, 3), ('u', 3, (-3.141592653589793, 0.30773985433519363, 3.141592653589793)), ('cz', 2, 3), ('u', 3, (0.0, 1.2630564724597029, 0.0)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 4, 5), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (-3.141592653589793, 1.3605291591529143, 3.141592653589793)), ('cz', 3, 4), ('u', 4, (0.0, 0.2102671676419825, 0.0)), ('cz', 5, 4), ('u', 4, (-3.141592653589793, 0.2102671676419825, 3.141592653589793)), ('cz', 3, 4), ('u', 4, (0.0, 1.3605291591529143, 0.0)), ('u', 5, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 4, 5), ('u', 6, (-2.7956955245943567, 0.006638396074130232, 1.9839531812880271)), ('u', 7, (3.3060693547306053, 2.57765001221808, 2.7878048943093074)), ('u', 8, (-0.9802959536225644, 3.1415926212758114, -5.141306813857595)), ('u', 5, (3.1836423183974443, 1.368608423331714, -0.35706132391778844)), ('u', 9, (3.282137172694394, 0.22185546077212015, -2.37617977051081)), ('cz', 7, 8), ('u', 6, (-0.863687536887122, 0.2036612508775408, -0.49096140311386094)), ('u', 7, (2.8866631605901194, 2.0572489031345156, -2.832204402499025)), ('u', 8, (0.8352656017295159, 1.5707963294313407, 0.3463688831494805)), ('u', 5, (-2.899488852373683, 0.023818332706368017, 2.986727851963161)), ('u', 9, (2.4208820653955025, 1.3491553116568324, 0.8806290474140755)), ('cz', 8, 9), ('u', 6, (2.832707855812823, 0.15840937628477464, -3.115598061390125)), ('u', 7, (-2.937596754155037, 0.6717279144270676, 3.0266444728300304)), ('u', 8, (2.7952252909671804, 0.8570719478274053, -3.0533719187797725)), ('u', 5, (0.12071465626978571, 0.013907386342634304, 0.332968572639534)), ('u', 9, (0.9569479039919327, 1.094241866608657, 0.41378535469459476)), ('cz', 7, 8), ('u', 6, (-0.1186231744977771, 0.12081883678314975, 0.8833194869947525)), ('u', 7, (2.8448936708527257, 2.0093937785142094e-08, -3.234615357195483)), ('u', 8, (-0.08822174150423814, 1.5707963267868776, 0.47942481146490246)), ('u', 5, (-0.4232632365402874, 0.07836685227309081, 0.06556582823325244)), ('u', 9, (3.4238520709664293, 1.3817784233230703, 0.7518585830252498)), ('cz', 8, 9), ('u', 6, (3.227019319836882, 0.17251920947071164, -3.001264902666509)), ('u', 7, (-0.4519555042207237, 2.180643064547514e-08, -0.17549316113966626)), ('u', 8, (-0.4794260024013855, 1.5707963376176648, -0.18415260485211554)), ('u', 5, (-0.004473691871499111, 0.13802087244552758, 0.17356360142769778)), ('u', 9, (3.4455507137660275, 1.5707963267535443, -0.1841526048505968)), ('u', 0, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 1, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 0, 1), ('u', 1, (1.570796326794898, 0.4636476090008061, -1.570796326794898)), ('u', 0, (1.5707963267948966, 0.4636476090008061, -1.5707963267948966)), ('cz', 0, 1), ('u', 1, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 0, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 1, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.1277577648985897, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.44303856189630686, 0.0)), ('cz', 2, 1), ('u', 1, (-3.141592653589793, 0.44303856189630686, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 1.1277577648985897, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.2284367252937551, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 0.3423596015011414, 0.0)), ('cz', 3, 2), ('u', 2, (-3.141592653589793, 0.3423596015011414, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 1.2284367252937551, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (-3.141592653589793, 1.3389725222944935, 3.141592653589793)), ('cz', 2, 3), ('u', 3, (0.0, 0.23182380450040305, 0.0)), ('cz', 4, 3), ('u', 3, (-3.141592653589793, 0.23182380450040305, 3.141592653589793)), ('cz', 2, 3), ('u', 3, (0.0, 1.3389725222944935, 0.0)), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 3, 4), ('u', 4, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.5235987755982989, 0.0)), ('u', 0, (0.0, 0.5235987755982989, 0.0)), ('cz', 0, 1), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.1780972450961724, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.39269908169872414, 0.0)), ('cz', 2, 1), ('u', 1, (-3.141592653589793, 0.39269908169872414, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 1.1780972450961724, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 2, (-3.141592653589793, 1.3089969389957472, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 0.26179938779914935, 0.0)), ('cz', 3, 2), ('u', 2, (-3.141592653589793, 0.26179938779914935, 3.141592653589793)), ('cz', 1, 2), ('u', 2, (0.0, 1.3089969389957472, 0.0)), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 2, 3), ('u', 3, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 1, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 0, 1), ('u', 1, (1.570796326794898, 0.6154797086703868, -1.570796326794898)), ('u', 0, (1.5707963267948966, 0.6154797086703876, -1.5707963267948966)), ('cz', 0, 1), ('u', 1, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 0, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 1, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 1, (-3.141592653589793, 1.2630564724597029, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 0.30773985433519363, 0.0)), ('cz', 2, 1), ('u', 1, (-3.141592653589793, 0.30773985433519363, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (0.0, 1.2630564724597029, 0.0)), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 1, 2), ('u', 2, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 0, (0.0, 1.5707963267948966, 0.0)), ('u', 1, (0.0, 1.5707963267948966, 0.0)), ('cz', 0, 1), ('u', 1, (1.2251903831398114e-16, 0.7853981633974483, -1.1496735851465472e-17)), ('u', 0, (-3.141592653589793, 0.7853981633974483, 3.141592653589793)), ('cz', 0, 1), ('u', 1, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 0, (-3.141592653589793, 1.5707963267948966, 3.141592653589793)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 9, (-3.141592653589793, 1.2139341373225139, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 0.35686218947238274, 0.0)), ('cz', 10, 9), ('u', 9, (-3.141592653589793, 0.35686218947238274, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 1.2139341373225139, 0.0)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 10, 11), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 10, (-3.141592653589793, 1.2888250061145825, 3.141592653589793)), ('cz', 9, 10), ('u', 10, (0.0, 0.2819713206803144, 0.0)), ('cz', 11, 10), ('u', 10, (-3.141592653589793, 0.2819713206803144, 3.141592653589793)), ('cz', 9, 10), ('u', 7, (-0.0010222009133052934, 1.3041008747344632e-09, 0.2003370551542191)), ('u', 8, (0.023930023575935412, 0.2953173773967398, -0.12864126913934845)), ('u', 9, (-0.030708083689677756, 0.12471016231533803, -0.5936932650946682)), ('u', 6, (0.07121035154428908, 0.4205343267071202, -0.2832141972956279)), ('u', 4, (0.07530003087270047, 0.16373930134914844, -0.26222930989690885)), ('cz', 7, 6), ('u', 7, (-0.12687198917430412, 1.5707963360271298, 0.2038898277475962)), ('u', 8, (3.2812013072679873, 1.866096954858985, 0.30184449475142205)), ('u', 9, (-0.2750715762601466, 0.1444293365187298, -0.5619473575075802)), ('u', 6, (-3.0846318122760255, 0.7274540347580297, 2.685237625937257)), ('u', 4, (-2.6959001663774766, 0.023390897362289824, 3.3774526330675045)), ('cz', 7, 8), ('u', 7, (2.937702797807351, 0.9553165545708295, -3.0503475368792787)), ('u', 8, (3.3207345893489655, 1.8839026748016603, -0.007233571707222719)), ('u', 9, (0.7067021023968258, 0.1861341116737401, 0.6824799267543968)), ('u', 6, (0.16900917346851702, 0.3290496173131335, 0.14955942962615904)), ('u', 4, (0.13632596908237446, 0.5047675908256389, 0.19005678333059065)), ('cz', 7, 6), ('u', 7, (-0.0912450930625181, 1.5707962664208395, 0.03309400916940823)), ('u', 8, (3.425558647277073, 1.3322673600128185, -0.027285494896139628)), ('u', 9, (2.9343637010964203, 0.3849224661080576, -3.181083530011608)), ('u', 6, (3.0294580093363934, 0.13808492223951388, -3.0008784410797094)), ('u', 4, (2.808193939953363, 0.34629612135864385, -3.327594214191119)), ('cz', 7, 8), ('u', 7, (-3.0107057097410745, 1.034782101861369, 3.2649574529238565)), ('u', 8, (2.701874333399131, 1.5707963333587613, 0.01699495886877811)), ('u', 9, (3.0218224318300324, 1.613517291586055, 0.20240124091333955)), ('u', 6, (-0.19260732367119549, 0.0506388373239212, 0.49460918963778)), ('u', 4, (3.2519730805535927, 0.002414860976536661, -2.897569812225165)), ('cz', 8, 9), ('u', 7, (2.704908443716258, 0.5581851719223989, -3.2239761040741026)), ('u', 8, (-2.7715068751111733, 1.1780352387189146, 2.8119235453642313)), ('u', 9, (3.2650506749779575, 1.7643530058585124, 0.13265112018338687)), ('u', 6, (0.524846134475885, 0.13013042238544234, 0.6969355712505834)), ('u', 4, (0.31766617169078154, 0.29627032380597074, -0.25692668698661764)), ('cz', 7, 8), ('u', 7, (2.9242210810494496, 0.07895765790608032, -3.0403033743301244)), ('u', 8, (0.28854329030454917, 0.39306029653695296, -0.018521965452702627)), ('u', 9, (-2.4666500872262773, 0.27631055933330684, 3.652088429466448)), ('u', 6, (-2.8645903242081086, 0.00811891627662618, 2.960010214724399)), ('u', 4, (-3.0255760348888474, 0.3172366772250132, 2.742544186481971)), ('cz', 8, 9), ('u', 7, (-0.10128924503841866, 0.07895765916161333, -0.54812026936145)), ('u', 8, (-3.1230706552997467, 0.39306028706200213, 2.9768939074869922)), ('u', 9, (0.22834942953139625, 0.16951111093992396, -0.46976825774972053)), ('u', 6, (-0.05157190861832458, 0.2831035587339365, 0.040556641699272526)), ('u', 4, (-2.9585527433127945, 0.18789853193368702, 3.150001059291019)), ('cz', 7, 8), ('u', 7, (0.30452840482299254, 1.497244522071447e-07, -0.21779698434427083)), ('u', 8, (0.20582453884073654, 1.1780352293228151, 0.36255569092721357)), ('u', 9, (2.5401593790964863, 1.6526078360053855, 0.15843675001271862)), ('u', 6, (-3.2179817957037806, 0.2740507804104168, 2.703112265700859)), ('u', 4, (-3.275513452054124, 0.15846558147607348, 2.9939277750514792)), ('cz', 8, 9), ('u', 7, (3.459547899859897, 1.5115771072655002e-07, -2.5026524551797933)), ('u', 8, (2.92342169170971, 2.5221885901529127e-09, -3.245476641550241)), ('u', 9, (2.834527942346747, 1.5707963245157934, 0.01822865285915709)), ('u', 6, (-2.6012871529086037, 0.10687131265079321, 1.7398627461175855)), ('u', 4, (0.05015276160771126, 0.10355943414250254, -0.4269578465944131)), ('u', 10, (0.0, 1.2888250061145825, 0.0)), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 10, 11), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 11, 12), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 11, (-3.141592653589793, 1.3769979834673063, 3.141592653589793)), ('cz', 10, 11), ('u', 11, (0.0, 0.1937983433275903, 0.0)), ('cz', 12, 11), ('u', 11, (-3.141592653589793, 0.1937983433275903, 3.141592653589793)), ('cz', 10, 11), ('u', 11, (0.0, 1.3769979834673063, 0.0)), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 11, 12), ('u', 12, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 9, (-3.141592653589793, 1.2630564724597029, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 0.30773985433519363, 0.0)), ('cz', 10, 9), ('u', 9, (-3.141592653589793, 0.30773985433519363, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 1.2630564724597029, 0.0)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 10, 11), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 10, (-3.141592653589793, 1.3605291591529143, 3.141592653589793)), ('cz', 9, 10), ('u', 10, (0.0, 0.2102671676419825, 0.0)), ('cz', 11, 10), ('u', 10, (-3.141592653589793, 0.2102671676419825, 3.141592653589793)), ('cz', 9, 10), ('u', 10, (0.0, 1.3605291591529143, 0.0)), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 10, 11), ('u', 11, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 6, (1.5707963267948966, 1.5707963267948966, -1.5707963267948966)), ('u', 7, (-1.5707963267948957, 1.5707963267948966, 1.5707963267948957)), ('cz', 6, 7), ('u', 7, (1.570796326794898, 0.4636476090008061, -1.570796326794898)), ('u', 6, (1.5707963267948966, 0.4636476090008061, -1.5707963267948966)), ('cz', 6, 7), ('u', 7, (1.5707963267948981, 1.5707963267948966, -1.5707963267948981)), ('u', 6, (-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)), ('u', 7, (0.0, 0.0, 1.3951473992034527e-15)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 7, 8), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 7, (-3.141592653589793, 1.1277577648985897, 3.141592653589793)), ('cz', 6, 7), ('u', 7, (0.0, 0.44303856189630686, 0.0)), ('cz', 8, 7), ('u', 7, (-3.141592653589793, 0.44303856189630686, 3.141592653589793)), ('cz', 6, 7), ('u', 7, (0.0, 1.1277577648985897, 0.0)), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 7, 8), ('u', 8, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 8, 9), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 8, (-3.141592653589793, 1.2284367252937551, 3.141592653589793)), ('cz', 7, 8), ('u', 8, (0.0, 0.3423596015011414, 0.0)), ('cz', 9, 8), ('u', 8, (-3.141592653589793, 0.3423596015011414, 3.141592653589793)), ('cz', 7, 8), ('u', 8, (0.0, 1.2284367252937551, 0.0)), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 8, 9), ('u', 9, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('u', 9, (-3.141592653589793, 1.3389725222944935, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 0.23182380450040305, 0.0)), ('cz', 10, 9), ('u', 9, (-3.141592653589793, 0.23182380450040305, 3.141592653589793)), ('cz', 8, 9), ('u', 9, (0.0, 1.3389725222944935, 0.0)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0)), ('cz', 9, 10), ('u', 6, (-0.10337358284045067, 1.5707963975858181, 1.015835984878588e-06)), ('u', 7, (3.038218917454895, 1.5707961616258277, -3.1415926581399134)), ('u', 8, (0.019828976040683857, 0.013724809664014377, 0.01783890191797663)), ('u', 9, (0.0035274908852903856, 0.07434633330967455, 0.0010581438542694391)), ('cz', 6, 7), ('u', 6, (5.959103240659553e-07, 0.5235987669981246, 0.0008503728370569114)), ('u', 7, (-7.864374259467977e-07, 0.5235987850137847, 2.1514876648522244e-08)), ('u', 8, (0.017838485853208828, 3.491322085471117e-05, 0.015992763508435617)), ('u', 9, (0.0010573111958433138, 0.055537721042183685, -0.00015720448849646705)), ('cz', 6, 7), ('u', 6, (-3.1407408931064547, 1.561653060584754, 3.108194496565501)), ('u', 7, (1.2678335777061012e-07, 1.5707963436427994, 0.1330778157276038)), ('u', 8, (3.1575855108403723, 1.5845243547676091, 0.00046389001286240017)), ('u', 9, (-0.00015720450130723025, 0.03568220976022034, 0.0007906627104473196)), ('cz', 7, 8), ('u', 6, (-0.03339896159371908, 0.0009456606723681459, -0.053307384901516736)), ('u', 7, (-3.008514747676956, 1.1780973866952744, 3.1415924849124677)), ('u', 8, (3.1420579084520557, 1.5707964334536435, 0.013241139958231063)), ('u', 9, (0.0007889975112366238, 0.012161875644356494, 0.005072193771559766)), ('cz', 7, 8), ('u', 6, (-3.1948983668400106, 0.01024058544541974, 3.0631718212986123)), ('u', 7, (3.141592626777089, 1.9634956698150956, -3.141592497081211)), ('u', 8, (0.013241406399454505, 0.811795812113978, -5.1901048898732316e-08)), ('u', 9, (3.14666567994145, 0.0183872739501306, -3.1274161263819162)), ('cz', 6, 7), ('u', 6, (-0.07842081277498814, 0.0636763764592665, -0.11265482417680511)), ('u', 7, (-1.5174894674825287e-07, 2.748894824926096, 0.004717116185436816)), ('u', 8, (3.1415931115099625, 0.8117960196269799, 0.0024698603439405886)), ('u', 9, (3.155767515687325, 0.057705323320175354, -3.1121926183807496)), ('cz', 7, 8), ('u', 6, (-0.11265400467920675, 0.036843437586738984, -0.14917897433769264)), ('u', 7, (0.004716613199078894, 0.21246252000648394, -0.010068637593983236)), ('u', 8, (0.0024699010658877363, 3.723199423905932e-07, 3.156587806026825)), ('u', 9, (3.1709922127383923, 1.6717308694221615, 0.006357761268223072)), ('cz', 8, 9), ('u', 6, (-0.149179819856092, 0.0073166448209931205, -0.18873465895337813)), ('u', 7, (-0.010068429561566238, 0.180256711353873, -0.025384936550044934)), ('u', 8, (-3.1265982994149426, 2.879792808347136, 3.14262416800779)), ('u', 9, (3.147950783718318, 1.5810194225609866, 0.0051794225141676975)), ('cz', 7, 8), ('u', 6, (-3.3303273125335644, 0.012528338239947763, 2.9115537090683197)), ('u', 7, (-0.02538514458468043, 8.543096018313841e-07, -0.020319696249047118)), ('u', 8, (0.0010318785259384909, 1.8325959332673774, 0.00398736224364416)), ('u', 9, (-3.1364136051101927, 0.010223660278832961, 3.135896011527404)), ('cz', 8, 9), ('u', 6, (-3.371630765613361, 0.01443821671985917, 2.871845946781179)), ('u', 7, (-0.02032094450144711, 1.8712083349506447e-07, -0.01330079721428616)), ('u', 8, (-3.137607417068711, 0.26180723510489173, 3.137741517763965)), ('u', 9, (-0.005695809571485449, 0.00403625252390588, 0.011325767365440383)), ('cz', 7, 8), ('u', 6, (-0.2697483848758344, 0.0015061147350338621, -0.341535392261783)), ('u', 7, (-3.1548932427496057, 0.006187151643163442, 3.1248315042568446)), ('u', 8, (-0.003849809993246951, 1.308996287788808, 0.01065630030694045)), ('u', 9, (3.1529182975762176, 1.5748316101961628, 0.0014652491448243765)), ('cz', 8, 9), ('u', 6, (1.2292585095449935, 1.5295686302113742, -1.5278527879141417)), ('u', 7, (-1.5875594496229066, 1.5705884974312958, 1.5677044976095087)), ('u', 8, (-3.1309342313072106, 0.49019375451821573, 3.0891773474553403)), ('u', 9, (3.1430586698649674, 1.543729458425524, 0.05079693796372453)), ('cz', 6, 7), ('u', 6, (1.613738941174663, 0.6154797193336008, -1.5693249595122236)), ('u', 7, (1.5677041944582901, 0.6154796712782338, -1.0941551216460745)), ('u', 8, (3.089176372066575, 0.5197091220145889, -3.1124591545135134)), ('u', 9, (-3.0907953812292854, 0.02205876862949908, 3.1797309234268294)), ('cz', 6, 7), ('u', 6, (-1.5693234376800447, 1.5710268263607212, 1.5320474481219526)), ('u', 7, (2.0474392804567048, 1.6465095989641034, -1.5976301429120348)), ('u', 8, (3.170724942066729, 2.133202830195918, 0.04435021715472276)), ('u', 9, (-3.103454591900695, 0.015664100610430897, 3.16660790256985)), ('cz', 7, 8), ('u', 6, (3.1028434940412866, 0.0029342574022471267, -3.170238681167906)), ('u', 7, (3.1147605899518007, 0.9553166913707994, -3.159526019221204)), ('u', 8, (3.1859433528692485, 1.5707966240942846, -0.00899140872696691)), ('u', 9, (-3.116576572034678, 0.00825150306986584, 3.153170208668008)), ('cz', 7, 8), ('u', 6, (3.1063913060080957, 1.881494231841266e-05, -3.1706637323774425)), ('u', 7, (3.124081332256533, 0.7853981594575237, -3.1415930460131394)), ('u', 8, (3.084263402421209, 0.022685295311754557, -3.10842371493674)), ('u', 9, (-0.0019941521657814727, 0.0006813777759123335, -0.01574928663732262)), ('cz', 6, 7), ('u', 6, (3.1125213667304164, 0.3092041646086194, -3.12625345487787)), ('u', 7, (-4.080796685731872e-08, 0.7853981558924876, 7.772173463874602e-07)), ('u', 8, (3.1747613891209787, 1.5481628904529872, -0.009805768630532175)), ('u', 9, (-0.015748454043375226, 0.008431297425970982, -0.0295910929262283)), ('cz', 7, 8), ('u', 6, (0.01533891114478319, 1.2643549581204578, 0.0057153799668915855)), ('u', 7, (-5.018874719283185e-07, 0.7853980462868956, 6.318653877145332e-07)), ('u', 8, (3.131787219964067, 2.0895288501316824, 0.0025644477643140817)), ('u', 9, (-0.029591092924892516, 0.015601263069330702, -0.04344267924801568)), ('cz', 6, 7), ('u', 6, (-3.1358763000417835, 0.7853983234787361, 3.141594348345165)), ('u', 7, (-7.034785056979964e-07, 1.0471960345973994, 3.6104177568044136e-07)), ('u', 8, (0.0025637937964505214, 0.5245194617938411, -0.00442326163518103)), ('u', 9, (-0.043441430274428344, 0.02175985552611398, -0.05718132719321954)), ('cz', 6, 7), ('u', 6, (3.14159139058198, 0.6154825200535677, -3.1045753624373456)), ('u', 7, (3.141592370965353, 1.570796391465584, -3.1045751262997925)), ('u', 8, (-0.004423053508179188, 0.5275553257727209, -0.011433642825968456)), ('u', 9, (-0.057182576164625784, 0.026436670338707852, -0.07062164023534258)), ('u', 10, (3.141592653589793, 1.5707963267948966, 0.0))])]}

def _peer__lib_template(d, N, n):
    edges, seq, P, mode = (d['edges'], d['seq'], d['params'], d['mode'])
    ops = []

    def layer(Pl):
        for q in range(N):
            if mode == 'real':
                ops.append(('u', q, _peer_ry(Pl[q][0])))
            else:
                a, t, b = Pl[q]
                ops.append(('u', q, _peer_rz(b) @ _peer_ry(t) @ _peer_rz(a)))
    layer(P[0])
    for t, e in enumerate(seq):
        a, b = edges[e]
        ops.append(('cz', a, b))
        layer(P[t + 1])
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=n)

def _peer_estimate_error(circuit, state_qubits, n, cz_err, sq_err, zero_err, one_err, dd=True):
    """Crude log-infidelity proxy for choosing between exact candidates."""
    czp = {}
    for pair, val in cz_err.items():
        czp[frozenset(pair)] = float(val[0])
    N = len(state_qubits)
    e = 0.0
    t = 0.0
    first = {}
    for mi, m in enumerate(circuit):
        has2 = any((len(op.qubits) == 2 for op in m.operations))
        dt = 4.2e-08 if has2 else 2.5e-08
        for op in m.operations:
            for q in op.qubits:
                first.setdefault(q, t)
            if len(op.qubits) == 2:
                e += 0.8 * czp[frozenset(op.qubits)]
            else:
                e += 0.67 * float(sq_err[op.qubits[0]][0])
        t += dt
    T = t
    expo = _peer__exposure(circuit) if len(circuit.all_qubits()) <= 14 else None
    if expo is None:
        expo = n * (T - min(first.values())) if first else 0.0
    e += expo / _peer_T1_EST
    e += (0.1 if dd else 0.2) * sum((T - first[q] for q in first)) / _peer_TPHI_EST
    for q in state_qubits:
        e += _peer_readout_cost(q, n / N, zero_err, one_err, x_weight=0.5)
    M = _peer_math.comb(N, n)
    if 1 < M <= 200:
        d = [float(one_err.get(q, (0, 0))[0]) - float(zero_err.get(q, (0, 0))[0]) for q in state_qubits]
        var = float(_peer_np.var(d))
        delta2 = n * (N - n) / (N - 1) * var
        K = 10000.0 * _peer_math.exp(-e)
        e += _peer_math.log(1 + K * delta2 / (M - 1))
    return e

def _peer__exposure(circuit):
    """sum over moments of (sum_q P(q=1)) * duration, from the exact noiseless state."""
    order = sorted(circuit.all_qubits())
    idx = {q: i for i, q in enumerate(order)}
    k = len(order)
    psi = _peer_np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1
    tot = 0.0
    for mo in circuit:
        two = False
        for op in mo.operations:
            if len(op.qubits) == 1:
                a = idx[op.qubits[0]]
                psi = _peer_np.moveaxis(_peer_np.tensordot(_peer_cirq.unitary(op), psi, axes=([1], [a])), 0, a)
            else:
                two = True
                a, b = (idx[op.qubits[0]], idx[op.qubits[1]])
                sl = [slice(None)] * k
                sl[a] = 1
                sl[b] = 1
                psi[tuple(sl)] *= -1
        p = _peer_np.abs(psi) ** 2
        tot += sum((float(p.take(1, axis=i).sum()) for i in range(k))) * (4.2e-08 if two else 2.5e-08)
    return tot

def _peer__verify(circuit, state_qubits, n, atol=1e-06):
    sq = sorted(state_qubits)
    anc = sorted(set(circuit.all_qubits()) - set(sq))
    N = len(sq)
    if len(sq) + len(anc) > 18:
        return True
    psi = _peer_cirq.final_state_vector(circuit, qubit_order=sq + anc, dtype=_peer_np.complex128).reshape(2 ** N, -1)
    tgt = _peer_np.zeros(2 ** N)
    for c in _peer_itertools.combinations(range(N), n):
        tgt[sum((1 << N - 1 - i for i in c))] = 1
    tgt /= _peer_np.linalg.norm(tgt)
    F = float(_peer_np.sum(_peer_np.abs(tgt @ psi) ** 2))
    return abs(F - 1) < atol

def _peer__diverse(states, k):
    """top-k embeddings with distinct qubit sets"""
    out, seen = ([], set())
    for st in states:
        key = frozenset(st[1].values())
        if key in seen:
            continue
        seen.add(key)
        out.append(st)
        if len(out) >= k:
            break
    return out

def _peer__candidates(N, n, adj, cz_err, sq_err, zero_err, one_err, deadline):
    qcost = lambda q: _peer_readout_cost(q, n / N, zero_err, one_err) + 2 * float(sq_err[q][0])
    out = []
    good = sorted(adj, key=lambda q: qcost(q) + min(adj[q].values(), default=1))
    if n == 1 and N >= 2:
        for root in good[:10]:
            for lam in (0.0005, 0.003, 0.02):
                if _peer__time.time() > deadline:
                    break
                patch = _peer_grow_patch(adj, N, root, qcost, lam=lam)
                if patch is None:
                    continue
                rounds = _peer_broadcast_tree(adj, patch, root)
                t = _peer_w_tree_template(patch, rounds)
                out.append(('wtree', t, {i: q for i, q in enumerate(patch)}))
    for d in _peer_LIBRARY.get((N, n), []):
        t = _peer__lib_template(d, N, n)
        nw = _peer_node_weights(t)
        res = _peer_embed(t, adj, lambda node, q: qcost(q) + nw[node] * 0.0, beam=2000, return_all=True)
        if res:
            for r in _peer__diverse(res, 6 if _peer_math.comb(N, n) <= 200 else 2):
                out.append(('lib', t, r[1]))
    for k, pops in _peer_PEEP.get((N, n), []):
        ops = [('u', o[1], _peer_rz(o[2][2]) @ _peer_ry(o[2][1]) @ _peer_rz(o[2][0])) if o[0] == 'u' else o for o in pops]
        t = dict(k=k, ops=ops, state_nodes=list(range(N)), N=N, n=n)
        res = _peer_embed(t, adj, lambda node, q: qcost(q), beam=1500, return_all=True)
        if res:
            for r in _peer__diverse(res, 4 if _peer_math.comb(N, n) <= 200 else 2):
                out.append(('peep', t, r[1]))
    if n >= 2 and N >= 2 * n:
        splits = sorted({N // 2, (N + 1) // 2}, key=lambda a: abs(a - N / 2))
        for a in splits:
            if a < n or N - a < n or _peer__time.time() > deadline:
                continue
            t = _peer_dc2_template(N, n, a)
            res = _peer_embed(t, adj, lambda node, q: qcost(q), beam=600 if N > 12 else 1500)
            if res is not None:
                out.append(('dc2', t, res[1]))
    if N >= 2:
        t = _peer_be_path_template_opt(N, n)
        nw = _peer_node_weights(t)
        res = _peer_embed(t, adj, lambda node, q: qcost(q), beam=600 if N > 12 else 2000)
        if res is not None:
            out.append(('bepath', t, res[1]))
    return out

def _peer_create_dicke_state(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    t0 = _peer__time.time()
    deadline = t0 + 8.0
    N, n = (int(num_qubits), int(num_excitations))
    if N < 1 or n < 0 or n > N:
        raise NotImplementedError
    flip = False
    if n > N - n:
        n, flip = (N - n, True)
    adj = _peer_hw_graph(cz_pauli_error_rate, sq_pauli_error_rate)
    if n == 0:
        qs = sorted(adj, key=lambda q: _peer_readout_cost(q, 1.0 if flip else 0.0, zero_errors, one_errors))[:N]
        if flip:
            c = _peer_cirq.Circuit([_peer_cirq.Moment([_peer_cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)(q) for q in qs])])
        else:
            c = _peer_cirq.Circuit([_peer_cirq.Moment([_peer_cirq.PhasedXZGate(x_exponent=0, z_exponent=0, axis_phase_exponent=0)(q) for q in qs])])
        return (c, set(qs))
    cands = _peer__candidates(N, n, adj, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors, deadline)
    scored = []
    for name, t, mapping in cands:
        for dd in (True, False):
            c = _peer_build_circuit(t, mapping, dd=dd)
            if flip:
                c = _peer__append_flip(c, [mapping[i] for i in t['state_nodes']])
            qs = {mapping[i] for i in t['state_nodes']}
            e = _peer_estimate_error(c, qs, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors, dd=dd)
            scored.append((e, name, c, qs))
    scored.sort(key=lambda x: x[0])
    for e, name, c, qs in scored:
        if len(c) > 1750:
            continue
        if _peer__verify(c, qs, num_excitations):
            return (c, qs)
    raise NotImplementedError

def _peer__append_flip(c, qubits):
    c = c.copy()
    last = c[-1] if len(c) else None
    if last is not None and all((len(op.qubits) == 1 for op in last.operations)):
        ops = {op.qubits[0]: op for op in last.operations}
        new = []
        for q in qubits:
            U = _peer_cirq.unitary(ops[q]) if q in ops else _peer__I2
            new.append(_peer_cirq.PhasedXZGate.from_matrix(_peer__X @ U).on(q))
        for q, op in ops.items():
            if q not in set(qubits):
                new.append(op)
        c[-1] = _peer_cirq.Moment(new)
    else:
        c.append(_peer_cirq.Moment([_peer_cirq.PhasedXZGate.from_matrix(_peer__X).on(q) for q in qubits]))
    return c


# Final public entry points. Earlier constructors are private portfolio members.
# Runtime decisions use only supplied calibration dictionaries. The peer's
# relaxation/dephasing constants are heuristic assumptions, not hardware inputs.
def _boundary_product_state(n, k, available=None):
    """Product states need no synthesis: depth zero or one, with no CZs."""
    if type(n) is not int or type(k) is not int or n < 1 or not 0 <= k <= n:
        raise ValueError('Require integer N>=1 and 0<=n<=N')
    device_qubits = set(cirq_google.Willow105.metadata.qubit_set)
    if available is not None:
        device_qubits &= set(available)
    outputs = set(sorted(device_qubits)[:n])
    if len(outputs) != n:
        raise NotImplementedError('Not enough available Willow qubits')
    circuit = cirq.Circuit()
    if k:
        gate = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)
        circuit.append(cirq.Moment(gate.on(q) for q in sorted(outputs)))
    return circuit, outputs


def _team_checked_small(candidate, n, k, cz, sq):
    circuit,outputs=_checked_hardware(*candidate,n,cz,sq)
    cirq_google.Willow105.validate_circuit(circuit)
    if circuit.all_qubits()!=outputs:
        raise NotImplementedError('Unexpected ancillary qubits')
    state=cirq.Simulator(dtype=np.complex128).simulate(circuit,qubit_order=sorted(outputs)).final_state_vector
    target=np.array([int(i).bit_count()==k for i in range(1<<n)],dtype=np.complex128)
    target/=np.linalg.norm(target)
    amplitude=np.vdot(target,state)
    if np.linalg.norm(state-amplitude*target)>1e-6:
        raise NotImplementedError('Candidate failed noiseless state verification')
    return circuit,outputs


def create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit,cirq.GridQubit],tuple[float,float]],
    sq_pauli_error_rate: dict[cirq.GridQubit,tuple[float,float]],
    zero_errors: dict[cirq.GridQubit,tuple[float,float]],
    one_errors: dict[cirq.GridQubit,tuple[float,float]],
) -> tuple[cirq.Circuit,set[cirq.GridQubit]]:
    """Prepare an exact Dicke state using the notebook's calibration inputs.

    Only CZ and PhasedXZ gates; no measurements or external file dependencies.
    Small circuits are compared under independent Pauli/readout noise. These
    scores are selection heuristics, not a complete model of Willow hardware.
    """
    n,k=num_qubits,num_excitations
    if not isinstance(n,int) or not isinstance(k,int) or n<1 or not 0<=k<=n:
        raise ValueError('Require integer N>=1 and 0<=n<=N')
    args=cz_pauli_error_rate,sq_pauli_error_rate,zero_errors,one_errors
    if k in (0,n):
        available = set(sq_pauli_error_rate) & set(zero_errors) & set(one_errors)
        return _boundary_product_state(n, k, available)
    if n>8:
        return _FINAL_ASTRA_MAIN(n,k,*args)
    started=time.monotonic()
    choices=[]
    def consider(builder):
        try:
            circuit,outputs=_team_checked_small(builder(n,k,*args),n,k,*args[:2])
            score=small_proxy(circuit,sorted(outputs),k,*args)
            if np.isfinite(score):choices.append((score,circuit,outputs))
        except Exception:
            # An unavailable optional family must not discard another valid one.
            return
    consider(_FINAL_ASTRA_MAIN)
    # ColumnForge has its own 45-second search guard. Start it early enough
    # to preserve the notebook's 60-second construction budget.
    if time.monotonic()-started<9:
        consider(_cf_create_dicke_state)
    if time.monotonic()-started<35:
        consider(_peer_create_dicke_state)
    if not choices and time.monotonic()-started<40:
        consider(_idea_create_dicke_state)
    if not choices:raise NotImplementedError('No valid exact circuit on the supplied calibration')
    _,circuit,outputs=min(choices,key=lambda row:row[0])
    return circuit,set(outputs)
