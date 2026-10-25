"""Lower logical gates to one-qubit unitaries and full CZ, without rerouting.

Multi-qubit helper gates are expanded with Cirq's structured decompositions
(CNOT, controlled-Ry, Fredkin, Toffoli), never left as a 3-qubit matrix.
A two-qubit gate that is not yet on a path edge is walked together along
that path with explicit SWAPs, then walked back, so the qubit order is
unchanged and every final entangler is a path edge.
"""

from __future__ import annotations

from collections.abc import Sequence

import cirq


def is_full_cz(op: cirq.Operation) -> bool:
    """True when ``op`` is a full CZ, including the inverse (CZ squared is identity)."""
    gate = op.gate
    if not isinstance(gate, cirq.CZPowGate):
        return False
    nearest = round(gate.exponent)
    if nearest % 2 == 0:
        return False
    return abs(gate.exponent - nearest) < 1e-8


def _expand(op: cirq.Operation) -> list[cirq.Operation]:
    """Structured expansion down to gates on at most two qubits."""
    if cirq.num_qubits(op) == 0 or isinstance(op.gate, cirq.GlobalPhaseGate):
        return []
    if cirq.num_qubits(op) <= 2:
        return [op]
    parts = list(cirq.decompose(op))
    if not parts or parts == [op]:
        raise ValueError(f"no structured decomposition for {op.gate} on {len(op.qubits)} qubits")
    expanded: list[cirq.Operation] = []
    for part in parts:
        if cirq.num_qubits(part) > 2:
            expanded.extend(_expand(part))
        elif cirq.num_qubits(part) == 0 or isinstance(part.gate, cirq.GlobalPhaseGate):
            continue
        else:
            expanded.append(part)
    return expanded


def expand_circuit(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """Replace every 3-or-more-qubit gate by 1- and 2-qubit gates, moment by moment."""
    expanded = cirq.Circuit()
    for moment in circuit:
        sequences = [_expand(op) for op in moment.operations]
        if not sequences:
            continue
        width = max(len(sequence) for sequence in sequences)
        for index in range(width):
            operations = [sequence[index] for sequence in sequences if index < len(sequence)]
            if operations:
                expanded.append(cirq.Moment(operations))
    return expanded


def bridge_along_path(circuit: cirq.AbstractCircuit, path: Sequence[cirq.Qid]) -> cirq.Circuit:
    """Move non-adjacent path gates onto a path edge, apply them, and move back.

    Gates in one moment are bridged one after another. Two bridges on the
    same path both need the path qubits, so they cannot share a moment.
    """
    index = {qubit: position for position, qubit in enumerate(path)}
    bridged = cirq.Circuit()
    for moment in circuit:
        for op in moment.operations:
            bridged += cirq.Circuit(_bridge_op(op, path, index))
    return bridged


def _bridge_op(
    op: cirq.Operation,
    path: Sequence[cirq.Qid],
    index: dict[cirq.Qid, int],
) -> list[cirq.Operation]:
    if cirq.num_qubits(op) == 1:
        return [op]
    if cirq.num_qubits(op) != 2:
        raise ValueError(f"bridge expects 1- or 2-qubit gates, got {op}")
    left, right = op.qubits
    if left not in index or right not in index:
        raise ValueError(f"{op} is not on the staging path")
    lo = min(index[left], index[right])
    hi = max(index[left], index[right])
    if hi - lo == 1:
        return [op]
    forth: list[cirq.Operation] = []
    token = list(range(len(path)))
    for position in range(hi - 1, lo, -1):
        forth.append(cirq.SWAP(path[position], path[position + 1]))
        token[position], token[position + 1] = token[position + 1], token[position]
    location = {original: position for position, original in enumerate(token)}
    moved = op.with_qubits(
        path[location[index[left]]],
        path[location[index[right]]],
    )
    return [*forth, moved, *reversed(forth)]


def bridge_in_region(
    circuit: cirq.AbstractCircuit,
    willow,
    region: Sequence[cirq.Qid],
) -> cirq.Circuit:
    """Walk each non-adjacent two-qubit gate along a region path, then walk back.

    The logical qubit order is unchanged. Used for a leaf that has no Hamiltonian
    path, so the line circuit cannot be laid directly on the device.
    """
    allowed = set(region)
    bridged = cirq.Circuit()
    for moment in circuit:
        for op in moment.operations:
            bridged += cirq.Circuit(_bridge_region_op(op, willow, allowed))
    return bridged


def _bridge_region_op(op: cirq.Operation, willow, allowed: set[cirq.Qid]) -> list[cirq.Operation]:
    if cirq.num_qubits(op) <= 1:
        return [op]
    if cirq.num_qubits(op) != 2:
        raise ValueError(f"region bridge expects 1- or 2-qubit gates, got {op}")
    left, right = op.qubits
    if left not in allowed or right not in allowed:
        raise ValueError(f"{op} is outside the leaf region")
    if willow.has_edge(left, right):
        return [op]
    path = _region_path(willow, left, right, allowed)
    if path is None:
        raise RuntimeError(f"no path inside the leaf between {left} and {right}")
    return _bridge_op(op, path, {qubit: i for i, qubit in enumerate(path)})


def _region_path(willow, start: cirq.Qid, goal: cirq.Qid, allowed: set[cirq.Qid]):
    from collections import deque

    from willow_dicke.graph import neighbors

    previous: dict[cirq.Qid, cirq.Qid | None] = {start: None}
    queue: deque[cirq.Qid] = deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(previous[path[-1]])
            path.reverse()
            return tuple(path)
        for other in neighbors(willow, node, allowed):
            if other in previous:
                continue
            previous[other] = node
            queue.append(other)
    return None


def lower_to_willow_cz(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """Gate decomposition only. Does not remap qubits onto a different coupling graph."""
    gateset = cirq.CZTargetGateset(allow_partial_czs=False, preserve_moment_structure=True)
    lowered = cirq.optimize_for_target_gateset(
        circuit,
        gateset=gateset,
        context=cirq.TransformerContext(deep=True),
    )
    return cirq.Circuit(lowered)


def assert_cz_and_single_qubit(circuit: cirq.AbstractCircuit) -> None:
    """Raise, after printing the offender, if any gate is not 1-qubit or full CZ."""
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.MeasurementGate):
            continue
        if cirq.num_qubits(op) == 1 and cirq.has_unitary(op):
            continue
        if is_full_cz(op):
            continue
        print(f"offending operation: {op!r}")
        raise AssertionError(f"gate {op.gate} is not a one-qubit unitary or a full CZ")


def assert_cz_on_graph(circuit: cirq.AbstractCircuit, willow) -> None:
    """Every CZ must be an edge of the device graph that was actually routed."""
    for op in circuit.all_operations():
        if not is_full_cz(op):
            continue
        left, right = op.qubits
        if not willow.has_edge(left, right):
            print(f"offending CZ: {op!r}")
            raise AssertionError(f"CZ {left} -- {right} is not a device edge")
