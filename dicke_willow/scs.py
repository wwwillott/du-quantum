"""Linear-nearest-neighbor Split & Cyclic Shift Dicke unitary.

Implements U_{n,k} from Bärtschi & Eidenbenz, arXiv:1904.07358, Definition 2
and Lemma 2. Every two-qubit gate in the returned operation list touches a
consecutive pair of the supplied path, which is the LNN embedding described
in Section 3 of that paper (the bottom wire of each SCS stair is sifted up
to the control pair and sifted back).

Gate angles follow Section 2.2:

    gate (i)    Ry(2 arccos sqrt(1 / n))
    gate (ii)_l Ry(2 arccos sqrt(l / n))

conjugated by the CNOTs given in the paper and in the standard SCS circuit.
The three-qubit block is expanded into adjacent CNOTs with a single SWAP
across the middle qubit, then that SWAP is undone, so the path order is
unchanged at the end of every SCS.
"""

from __future__ import annotations

import math

import cirq
import numpy as np


def _ry_angle(numerator: int, denominator: int) -> float:
    """Ry(2 arccos sqrt(numerator / denominator)), in radians."""
    return 2.0 * math.acos(math.sqrt(numerator / denominator))


def scs_unitary_ops(qubits: list[cirq.Qid], k: int) -> list[cirq.Operation]:
    """Operations for U_{n,k} on a path. Does not prepare the unary input.

    ``qubits`` is ordered along the path, index 0 on the left. On every input
    ``|0^{n-ell} 1^{ell}>`` with ``ell <= k`` the circuit prepares ``|D_ell^n>``.
    Qubit order is restored at the end (sift-down), so the caller does not
    have to track a permutation.
    """
    n = len(qubits)
    if k < 0 or k > n:
        raise ValueError(f"k={k} is outside 0..n={n}")
    if k <= 1 and n <= 1:
        return []
    if k == 0:
        return []

    ops: list[cirq.Operation] = []

    def swap(i: int) -> None:
        ops.append(cirq.SWAP(qubits[i], qubits[i + 1]))

    def cnot(i: int, j: int) -> None:
        ops.append(cirq.CNOT(qubits[i], qubits[j]))

    def cry(control: int, target: int, theta: float) -> None:
        # Adjacent controlled-Ry(theta). cry in Cirq is Ry(theta) when control is |1>.
        if abs(control - target) != 1:
            raise RuntimeError("CRY must be nearest-neighbor")
        ops.append(cirq.ry(theta / 2).on(qubits[target]))
        cnot(control, target)
        ops.append(cirq.ry(-theta / 2).on(qubits[target]))
        cnot(control, target)

    def cnot_distance_two(i: int, j: int) -> None:
        """CNOT between path indices distance 2, restoring the middle qubit."""
        if abs(i - j) != 2:
            raise RuntimeError(f"expected distance 2, got {i}, {j}")
        middle = (i + j) // 2
        left = min(middle, j)
        swap(left)
        cnot(i, middle)
        swap(left)

    def gate_i(end: int, size: int) -> None:
        # Section 2.2 gate (i) on the last two wires of this SCS.
        theta = _ry_angle(1, size)
        cnot(end - 1, end)
        cry(end, end - 1, theta)
        cnot(end - 1, end)

    def gate_ii(p0: int, p1: int, p2: int, ell: int, size: int) -> None:
        # Section 2.2 gate (ii)_ell on three adjacent wires (p0, p1, p2).
        # Logical circuit, with q0=p0, q1=p1, q2=p2:
        #   CNOT(q0, q2); CCRy(theta) controls (q2, q1) target q0; CNOT(q0, q2)
        # theta = 2 arccos sqrt(ell / size).
        theta = _ry_angle(ell, size)
        cnot_distance_two(p0, p2)
        half = theta / 2
        ops.append(cirq.ry(half / 2).on(qubits[p0]))
        cnot(p1, p0)
        ops.append(cirq.ry(-half / 2).on(qubits[p0]))
        cnot_distance_two(p2, p0)
        ops.append(cirq.ry(half / 2).on(qubits[p0]))
        cnot(p1, p0)
        ops.append(cirq.ry(-half / 2).on(qubits[p0]))
        cnot_distance_two(p2, p0)
        cnot_distance_two(p0, p2)

    def scs_prefix(size: int, weight: int) -> None:
        """SCS_{size, weight} on qubits[:size], bottom wire sifted and restored."""
        gate_i(size - 1, size)
        for ell in range(2, weight + 1):
            destination = size - ell + 1
            position = size - 1
            while position > destination:
                swap(position - 1)
                position -= 1
            gate_ii(size - ell - 1, size - ell, position, ell, size)
            while position < size - 1:
                swap(position)
                position += 1

    # Lemma 2: SCS_{n,k}, SCS_{n-1,k}, ..., SCS_{k+1,k}, then SCS_{k,k-1}, ..., SCS_{2,1}.
    for size in range(n, k, -1):
        scs_prefix(size, k)
    for size in range(k, 1, -1):
        scs_prefix(size, size - 1)
    return ops


def prepare_dicke_ops(qubits: list[cirq.Qid], k: int) -> list[cirq.Operation]:
    """Prepare |D_k^n> from |0...0> along ``qubits``.

    Uses the smaller of k and n-k (X on every qubit turns |D_{n-k}> into |D_k>),
    which is the size bound in Theorem 1 of arXiv:1904.07358.
    """
    n = len(qubits)
    if k < 0 or k > n:
        raise ValueError(f"k={k} is outside 0..n={n}")
    if k == 0:
        return []
    if k == n:
        return [cirq.X(q) for q in qubits]

    excite = min(k, n - k)
    ops: list[cirq.Operation] = [cirq.X(qubits[i]) for i in range(n - excite, n)]
    ops.extend(scs_unitary_ops(qubits, excite))
    if excite != k:
        ops.extend(cirq.X(q) for q in qubits)
    return ops


def prepare_dicke_circuit(qubits: list[cirq.Qid], k: int) -> cirq.Circuit:
    """Pack the path preparation into moments, leaving long-range gates out."""
    return cirq.Circuit(prepare_dicke_ops(qubits, k))


def _gate_i_ops(a: cirq.Qid, b: cirq.Qid, size: int) -> list[cirq.Operation]:
    """Section 2.2 gate (i). ``a`` is wire n-2 and ``b`` is wire n-1."""
    theta = _ry_angle(1, size)
    cry = cirq.ControlledGate(cirq.ry(theta))
    return [cirq.CNOT(a, b), cry.on(b, a), cirq.CNOT(a, b)]


def _gate_ii_ops(q0: cirq.Qid, q1: cirq.Qid, q2: cirq.Qid, ell: int, size: int) -> list[cirq.Operation]:
    """Section 2.2 gate (ii)_ell on wires (n-ell, n-ell+1, n)."""
    theta = _ry_angle(ell, size)
    ccry = cirq.ControlledGate(cirq.ry(theta), num_controls=2)
    return [cirq.CNOT(q0, q2), ccry.on(q2, q1, q0), cirq.CNOT(q0, q2)]


def logical_dicke_circuit(qubits: list[cirq.Qid], k: int) -> cirq.Circuit:
    """Dicke preparation before nearest-neighbor routing.

    This is the same U_{n,k} as ``prepare_dicke_circuit``, written with the
    long-range controls of the 2019 stair. A later router has to move those
    controls onto edges that exist. Keeping the stair intact, instead of
    sifting inside every three-qubit block, is what leaves room for a router
    to parallelize the SWAPs.
    """
    n = len(qubits)
    if k < 0 or k > n:
        raise ValueError(f"k={k} is outside 0..n={n}")
    if k == 0:
        return cirq.Circuit()
    if k == n:
        return cirq.Circuit(cirq.X.on_each(*qubits))
    excite = min(k, n - k)
    ops: list[cirq.Operation] = [cirq.X(qubits[i]) for i in range(n - excite, n)]

    def scs_on_prefix(size: int, weight: int) -> None:
        window = qubits[:size]
        # SCS_{size, weight} acts on the last weight+1 qubits of this prefix.
        block = window[size - weight - 1 : size]
        ops.extend(_gate_i_ops(block[weight - 1], block[weight], size))
        for ell in range(2, weight + 1):
            ops.extend(
                _gate_ii_ops(block[weight - ell], block[weight - ell + 1], block[weight], ell, size)
            )

    for size in range(n, excite, -1):
        scs_on_prefix(size, excite)
    for size in range(excite, 1, -1):
        scs_on_prefix(size, size - 1)
    if excite != k:
        ops.extend(cirq.X(q) for q in qubits)
    return cirq.Circuit(ops)


def interaction_span_weights(n: int, k: int) -> list[int]:
    """How many logical multi-qubit gates cover each path edge.

    A gate on logical qubits i..j has to be routed across every edge between
    them, so those edges pick up the gate's traffic.
    """
    if n <= 1:
        return []
    qubits = list(cirq.LineQubit.range(n))
    circuit = logical_dicke_circuit(qubits, k)
    index = {q: i for i, q in enumerate(qubits)}
    weights = [0] * (n - 1)
    for op in circuit.all_operations():
        if len(op.qubits) < 2:
            continue
        positions = sorted(index[q] for q in op.qubits)
        for edge in range(positions[0], positions[-1]):
            weights[edge] += 1
    return weights


def adjacent_cz_counts(n: int, k: int) -> list[int]:
    """How many CZ gates the compiled path circuit places on each path edge.

    Index i counts the coupler between path positions i and i+1. Used to
    weight noise-aware placement toward quiet couplers that the stair actually
    uses.
    """
    if n <= 1:
        return []
    qubits = cirq.LineQubit.range(n)
    logical = prepare_dicke_circuit(list(qubits), k)
    native = cirq.optimize_for_target_gateset(logical, gateset=cirq.CZTargetGateset())
    counts = [0] * (n - 1)
    index = {q: i for i, q in enumerate(qubits)}
    for op in native.all_operations():
        if not isinstance(op.gate, cirq.CZPowGate):
            continue
        a, b = op.qubits
        ia, ib = index[a], index[b]
        if abs(ia - ib) != 1:
            raise RuntimeError("compiled SCS used a non-adjacent CZ")
        counts[min(ia, ib)] += 1
    return counts
