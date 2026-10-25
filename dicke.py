"""Dicke-state preparation on Willow: nearest-neighbour Bärtschi–Eidenbenz on a qubit path.

Logical picture (positions 0..N-1 along a path). Start from |0^{N-k} 1^k>. For m = N..2 the
split-and-cyclic-shift step SCS_m decides position m-1: with l ones among the first m positions,
sorted to the right, it keeps it 1 w.p. l/m, otherwise moves the "hole" just left of the ones
block to position m-1. On a line this is a left-to-right sweep over pairs (j, j+1):
  - first pair of the window: Givens rotation 01 -> sqrt(l/m) 01 + sqrt(1-l/m) 10 (2 CZ),
  - later pairs, with c = j-1 as control: Givens if c=0 (boundary), SWAP if c=1 (moving hole).
    Implemented as G(phi2) CZ(c,a) G(phi1) Z_c with phi1+phi2 = gamma, phi1-phi2 = pi/2 (5 CZ);
    exact on every input that can occur (000, 001, 011, 101, 111 on (c, a, b)).
"""
import functools
import math

import cirq
import networkx as nx
import numpy as np


def givens_matrix(phi: float) -> np.ndarray:
    """On (a, b): |01> -> cos(phi)|01> + sin(phi)|10>, |00>, |11> fixed."""
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=complex)


@functools.lru_cache(maxsize=None)
def _givens_template(phi: float) -> tuple:
    """2-CZ decomposition of givens_matrix(phi) on LineQubits(0, 1), cached per angle."""
    a, b = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_cz_operations(a, b, givens_matrix(phi), allow_partial_czs=False)
    return tuple(ops)


def givens_ops(a: cirq.Qid, b: cirq.Qid, phi: float) -> list[cirq.Operation]:
    mapping = dict(zip(cirq.LineQubit.range(2), (a, b)))
    return [op.transform_qubits(mapping) for op in _givens_template(phi)]


def line_dicke_ops(path: list[cirq.Qid], k: int) -> list[cirq.Operation]:
    """Logical-to-CZ ops preparing D(len(path), k) on a path whose consecutive qubits are coupled."""
    N = len(path)
    ops = [cirq.X(q) for q in path[N - k:]]
    for m in range(N, 1, -1):
        kk = min(k, m - 1)
        for j in range(m - 1 - kk, m - 1):
            l = m - 1 - j
            gamma = math.acos(math.sqrt(l / m))
            a, b = path[j], path[j + 1]
            if j == m - 1 - kk:  # left neighbour is |0> (or absent): plain Givens
                ops += givens_ops(a, b, gamma)
            else:
                c = path[j - 1]
                ops += givens_ops(a, b, (gamma + math.pi / 2) / 2)
                ops.append(cirq.CZ(c, a))
                ops += givens_ops(a, b, (gamma - math.pi / 2) / 2)
                ops.append(cirq.Z(c))
    return ops


def cz_counts_per_edge(N: int, k: int) -> dict[tuple[int, int], int]:
    """How many CZs line_dicke_ops puts on each path edge (j, j+1); used to weight placement."""
    counts = {(j, j + 1): 0 for j in range(N - 1)}
    for m in range(N, 1, -1):
        kk = min(k, m - 1)
        for j in range(m - 1 - kk, m - 1):
            counts[(j, j + 1)] += 2
            if j != m - 1 - kk:
                counts[(j, j + 1)] += 2
                counts[(j - 1, j)] += 1
    return counts


def compile_to_hardware(ops: list[cirq.Operation], flip_all: list[cirq.Qid] = ()) -> cirq.Circuit:
    """CZ + PhasedXZ only, single-qubit gates merged, earliest scheduling.

    The ops are already CZ + 1q; cirq.optimize_for_target_gateset found no CZ savings here and
    took 10x longer (23 s at N=30), so only single-qubit runs are merged.
    """
    circuit = cirq.Circuit(ops)
    if flip_all:
        circuit.append(cirq.Moment(cirq.X(q) for q in flip_all))
    circuit = cirq.merge_single_qubit_gates_to_phxz(circuit)
    circuit = cirq.drop_negligible_operations(circuit)
    circuit = cirq.drop_empty_moments(circuit)
    return cirq.stratified_circuit(
        circuit, categories=[lambda op: len(op.qubits) == 1, lambda op: len(op.qubits) == 2]
    )


def hardware_graph(cz_pauli_error_rate, sq_pauli_error_rate) -> nx.Graph:
    g = nx.Graph()
    for q, (e, _) in sq_pauli_error_rate.items():
        g.add_node(q, sq_error=e)
    for (a, b), (e, _) in cz_pauli_error_rate.items():
        if a in g and b in g:
            g.add_edge(a, b, cz_error=e)
    return g


def best_path(g: nx.Graph, N: int, k: int, zero_errors, one_errors, beam: int = 400) -> list:
    """Beam search for a simple path minimising expected CZ error + readout error.

    Both orientations matter (edge CZ usage is asymmetric), so paths are scored as built
    left-to-right, i.e. path[0] is logical position 0.
    """
    use = cz_counts_per_edge(N, k)

    def readout(q):
        e0 = zero_errors.get(q, (0.05, 0))[0]
        e1 = one_errors.get(q, (0.05, 0))[0]
        return (N - k) / N * e0 + k / N * e1

    beams = [((readout(q), (q,))) for q in g.nodes]
    for j in range(1, N):
        cand = {}
        for cost, path in beams:
            for r in g[path[-1]]:
                if r in path:
                    continue
                new = cost + use[(j - 1, j)] * g.edges[path[-1], r]["cz_error"] + readout(r)
                key = (frozenset(path) | {r}, r)
                if key not in cand or cand[key][0] > new:
                    cand[key] = (new, path + (r,))
        beams = sorted(cand.values(), key=lambda t: t[0])[:beam]
        if not beams:
            raise NotImplementedError(f"no path of length {N} on the available qubits")
    return list(beams[0][1])


def create_dicke_state(
    num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors
):
    N, n = num_qubits, num_excitations
    if not (1 <= N and 0 <= n <= N):
        raise NotImplementedError
    flip = n > N / 2  # build D(N, N-n) with fewer excitations, flip at the end
    k = N - n if flip else n
    g = hardware_graph(cz_pauli_error_rate, sq_pauli_error_rate)
    if N == 1:
        q = min(g.nodes, key=lambda q: (zero_errors if n == 0 else one_errors)[q][0])
        path = [q]
    else:
        path = best_path(g, N, k, zero_errors, one_errors)
    circuit = compile_to_hardware(line_dicke_ops(path, k), flip_all=path if flip else ())
    return circuit, set(path)
