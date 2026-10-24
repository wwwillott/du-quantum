"""Coupling graphs taken from a Cirq device or from the challenge error maps.

Vertices are physical qubits. An edge exists only when the device (or the
supplied CZ error dictionary) lists that pair. Missing couplers are absent
from the graph; nothing in this layer invents them.
"""

from __future__ import annotations

from collections import defaultdict

import cirq
import cirq_google


def _endpoints(pair) -> tuple[cirq.Qid, cirq.Qid]:
    if isinstance(pair, tuple) and len(pair) == 2:
        return pair[0], pair[1]
    qubits = list(pair)
    if len(qubits) != 2:
        raise ValueError(f"expected a qubit pair, got {pair!r}")
    return qubits[0], qubits[1]


def graph_from_pairs(
    pairs, qubits: set[cirq.Qid] | None = None
) -> dict[cirq.Qid, list[cirq.Qid]]:
    """Undirected adjacency list. ``qubits`` restricts the vertex set."""
    adjacency: dict[cirq.Qid, list[cirq.Qid]] = defaultdict(list)
    for pair in pairs:
        a, b = _endpoints(pair)
        if qubits is not None and (a not in qubits or b not in qubits):
            continue
        if b not in adjacency[a]:
            adjacency[a].append(b)
        if a not in adjacency[b]:
            adjacency[b].append(a)
    if qubits is not None:
        for qubit in qubits:
            adjacency.setdefault(qubit, [])
    return dict(adjacency)


def graph_from_device(device: cirq.Device) -> dict[cirq.Qid, list[cirq.Qid]]:
    """Coupling graph of a Cirq device, usually ``cirq_google.Willow105``."""
    metadata = device.metadata
    if metadata is None or metadata.qubit_set is None:
        raise ValueError("device does not expose a qubit set")
    pairs = metadata.qubit_pairs or ()
    return graph_from_pairs(pairs, set(metadata.qubit_set))


def willow105_graph() -> dict[cirq.Qid, list[cirq.Qid]]:
    return graph_from_device(cirq_google.Willow105)


def graph_from_error_rates(
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict | None = None,
    one_errors: dict | None = None,
) -> dict[cirq.Qid, list[cirq.Qid]]:
    """Challenge graph: a coupler is usable only if both qubits have data.

    Qubits missing from the single-qubit, zero, or one error maps are treated
    as disabled, matching a live processor with dropped qubits.
    """
    allowed = set(sq_pauli_error_rate)
    if zero_errors is not None:
        allowed &= set(zero_errors)
    if one_errors is not None:
        allowed &= set(one_errors)
    return graph_from_pairs(cz_pauli_error_rate.keys(), allowed)


def error_value(entry) -> float:
    """First element of an (error, uncertainty) pair, or the bare float."""
    if isinstance(entry, tuple):
        return float(entry[0])
    return float(entry)


def edge_error(cz_pauli_error_rate: dict, a: cirq.Qid, b: cirq.Qid) -> float:
    if (a, b) in cz_pauli_error_rate:
        return error_value(cz_pauli_error_rate[(a, b)])
    if (b, a) in cz_pauli_error_rate:
        return error_value(cz_pauli_error_rate[(b, a)])
    return 1.0


def connected_components(adjacency: dict[cirq.Qid, list[cirq.Qid]]) -> list[set[cirq.Qid]]:
    seen: set[cirq.Qid] = set()
    components: list[set[cirq.Qid]] = []
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        component: set[cirq.Qid] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node])
        seen |= component
        components.append(component)
    return components
