"""Noise-aware placement of a Dicke path onto a coupling graph.

The 2019 LNN unitary needs a simple path. On Willow that path is chosen from
the measured couplers, not from a drawing of the ideal grid. The objective is
the expected CZ error along the path, weighted by how many compiled CZ gates
the stair puts on each logical edge, plus single-qubit and readout error.
"""

from __future__ import annotations

import cirq

from dicke_willow.scs import interaction_span_weights
from dicke_willow.topology import edge_error, error_value


def _node_cost(qubit, sq_errors, zero_errors, one_errors, excitation_fraction: float) -> float:
    sq = error_value(sq_errors[qubit]) if qubit in sq_errors else 1e-3
    zero = error_value(zero_errors[qubit]) if qubit in zero_errors else 1e-2
    one = error_value(one_errors[qubit]) if qubit in one_errors else 1e-2
    readout = (1.0 - excitation_fraction) * zero + excitation_fraction * one
    return sq + readout


def place_path(
    adjacency: dict[cirq.Qid, list[cirq.Qid]],
    n: int,
    k: int,
    *,
    cz_errors: dict | None = None,
    sq_errors: dict | None = None,
    zero_errors: dict | None = None,
    one_errors: dict | None = None,
    beam_width: int = 48,
    n_starts: int = 24,
) -> list[cirq.Qid] | None:
    """Return a simple path of ``n`` qubits, or None if the graph has none.

    With no error maps the search still prefers high-degree compact growth,
    which is what the adaptive Willow question needs.
    """
    if n <= 0:
        return []
    usable = [q for q, neighbors in adjacency.items() if neighbors or n == 1]
    if n == 1:
        if not adjacency:
            return None
        return [next(iter(adjacency))]
    if len(usable) < n:
        return None

    weights = interaction_span_weights(n, k)
    fraction = 0.0 if n == 0 else k / n
    sq_errors = sq_errors or {}
    zero_errors = zero_errors or {}
    one_errors = one_errors or {}
    cz_errors = cz_errors or {}

    def node(qubit: cirq.Qid) -> float:
        return _node_cost(qubit, sq_errors, zero_errors, one_errors, fraction)

    def coupler(a: cirq.Qid, b: cirq.Qid, index: int) -> float:
        return weights[index] * edge_error(cz_errors, a, b)

    starts = sorted(usable, key=node)[:n_starts]
    beam: list[tuple[float, tuple[cirq.Qid, ...]]] = [(node(start), (start,)) for start in starts]
    for step in range(1, n):
        candidates: list[tuple[float, tuple[cirq.Qid, ...]]] = []
        seen: set[tuple[cirq.Qid, ...]] = set()
        for cost, path in beam:
            occupied = set(path)
            for neighbor in adjacency[path[-1]]:
                if neighbor in occupied:
                    continue
                extended = path + (neighbor,)
                if extended in seen:
                    continue
                seen.add(extended)
                candidates.append((cost + node(neighbor) + coupler(path[-1], neighbor, step - 1), extended))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        beam = candidates[:beam_width]
    best_cost, best_path = min(beam, key=lambda item: item[0])
    del best_cost
    return list(best_path)


def snake_path(adjacency: dict[cirq.Qid, list[cirq.Qid]], n: int) -> list[cirq.Qid] | None:
    """Deterministic path: walk rows of a grid, then fill by DFS if needed.

    Used for the adaptive question, where there is no calibration and any
    Willow105 path of length n is legal. GridQubits sort by (row, col).
    """
    if n <= 0:
        return []
    qubits = sorted(adjacency)
    if len(qubits) < n:
        return None
    # Prefer a row-wise snake that only steps on real edges.
    by_row: dict[int, list[cirq.Qid]] = {}
    for qubit in qubits:
        row = qubit.row if isinstance(qubit, cirq.GridQubit) else 0
        by_row.setdefault(row, []).append(qubit)
    snake: list[cirq.Qid] = []
    for row_index, row in enumerate(sorted(by_row)):
        ordered = sorted(by_row[row], key=lambda q: q.col if isinstance(q, cirq.GridQubit) else 0)
        if row_index % 2 == 1:
            ordered.reverse()
        snake.extend(ordered)
    path: list[cirq.Qid] = []
    for qubit in snake:
        if not path:
            path.append(qubit)
        elif qubit in adjacency[path[-1]] and qubit not in path:
            path.append(qubit)
        if len(path) == n:
            return path
    # The pure snake can break at disabled qubits. Fall back to an unweighted search.
    return place_path(adjacency, n, k=1)
