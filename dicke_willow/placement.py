"""Noise-aware placement of a Dicke path onto a coupling graph.

The 2019 LNN unitary needs a simple path. On Willow that path is chosen from
the measured couplers, not from a drawing of the ideal grid. Edge cost is the
simulator's Pauli error, turned into a travel time

    t(e) = -log(1 - error(e))

so Dijkstra's shortest path is the fastest corridor of a given length. A
coupler the calibration marks as noisier is a slower edge. When several
paths run at once and the chip has qubits left over, each path is planned
in the residual graph: the first gets the fastest route, and the next gets
the fastest route that does not touch it.
"""

from __future__ import annotations

import math

import cirq

from dicke_willow.scs import interaction_span_weights
from dicke_willow.topology import edge_error, error_value


def _node_cost(qubit, sq_errors, zero_errors, one_errors, excitation_fraction: float) -> float:
    sq = error_value(sq_errors[qubit]) if qubit in sq_errors else 1e-3
    zero = error_value(zero_errors[qubit]) if qubit in zero_errors else 1e-2
    one = error_value(one_errors[qubit]) if qubit in one_errors else 1e-2
    readout = (1.0 - excitation_fraction) * zero + excitation_fraction * one
    return _travel_time(sq) + _travel_time(readout)


def _travel_time(error: float) -> float:
    """Positive duration of one use of a coupler or qubit.

    ``-log(1 - error)`` is additive across a path and grows sharply as a
    coupler gets worse, so the shortest path is the one that fails least often.
    """
    clipped = min(max(error, 0.0), 1.0 - 1e-9)
    return -math.log(1.0 - clipped)


def _edge_time(cz_errors: dict, a: cirq.Qid, b: cirq.Qid) -> float:
    return _travel_time(edge_error(cz_errors, a, b))


def dijkstra_path(
    adjacency: dict[cirq.Qid, list[cirq.Qid]],
    n: int,
    k: int,
    *,
    cz_errors: dict | None = None,
    sq_errors: dict | None = None,
    zero_errors: dict | None = None,
    one_errors: dict | None = None,
    forbidden: set[cirq.Qid] | None = None,
    width: int = 64,
    n_starts: int = 16,
) -> list[cirq.Qid] | None:
    """Fastest simple path of ``n`` qubits under the simulator error dictionary.

    This is Dijkstra's algorithm on paths of exact length ``n``: partial paths
    are expanded in increasing travel time, and only the fastest ``width``
    survivors are kept at each length. ``forbidden`` qubits are already taken
    by a path that runs in parallel. The search may use any other coupler, so
    a path can step off a noisy row when a quieter one is free.
    """
    if n <= 0:
        return []
    blocked = set(forbidden or ())
    usable = [q for q in adjacency if q not in blocked and (adjacency[q] or n == 1)]
    if n == 1:
        if not usable:
            return None
        fraction = 0.0 if n == 0 else k / n
        usable.sort(key=lambda q: _node_cost(q, sq_errors or {}, zero_errors or {}, one_errors or {}, fraction))
        return [usable[0]]
    if len(usable) < n:
        return None

    weights = interaction_span_weights(n, k)
    fraction = k / n
    sq_errors = sq_errors or {}
    zero_errors = zero_errors or {}
    one_errors = one_errors or {}
    cz_errors = cz_errors or {}

    def node(qubit: cirq.Qid) -> float:
        return _node_cost(qubit, sq_errors, zero_errors, one_errors, fraction)

    starts = sorted(usable, key=node)[:n_starts]
    frontier: list[tuple[float, tuple[cirq.Qid, ...]]] = [(node(start), (start,)) for start in starts]
    for step in range(n - 1):
        expanded: list[tuple[float, tuple[cirq.Qid, ...]]] = []
        seen: set[tuple[cirq.Qid, ...]] = set()
        for cost, path in frontier:
            occupied = set(path)
            for neighbor in adjacency[path[-1]]:
                if neighbor in occupied or neighbor in blocked:
                    continue
                extended = path + (neighbor,)
                if extended in seen:
                    continue
                seen.add(extended)
                step_cost = node(neighbor) + weights[step] * _edge_time(cz_errors, path[-1], neighbor)
                expanded.append((cost + step_cost, extended))
        if not expanded:
            return None
        expanded.sort(key=lambda item: item[0])
        frontier = expanded[:width]
    return list(frontier[0][1])


def dijkstra_parallel_paths(
    adjacency: dict[cirq.Qid, list[cirq.Qid]],
    lengths: list[int],
    k: int,
    *,
    cz_errors: dict | None = None,
    sq_errors: dict | None = None,
    zero_errors: dict | None = None,
    one_errors: dict | None = None,
) -> list[list[cirq.Qid]] | None:
    """Disjoint fastest paths for regions that execute in the same moment.

    Longer paths are placed first, while the chip still has the most room.
    Each later path runs Dijkstra again on the qubits that are left, so it
    takes the fastest corridor that does not collide with a sibling.
    """
    claimed: set[cirq.Qid] = set()
    placed: list[list[cirq.Qid] | None] = [None] * len(lengths)
    order = sorted(range(len(lengths)), key=lambda index: lengths[index], reverse=True)
    for index in order:
        path = dijkstra_path(
            adjacency,
            lengths[index],
            k,
            cz_errors=cz_errors,
            sq_errors=sq_errors,
            zero_errors=zero_errors,
            one_errors=one_errors,
            forbidden=claimed,
        )
        if path is None:
            return None
        claimed.update(path)
        placed[index] = path
    return placed  # type: ignore[return-value]


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

    fastest = dijkstra_path(
        adjacency,
        n,
        k,
        cz_errors=cz_errors,
        sq_errors=sq_errors,
        zero_errors=zero_errors,
        one_errors=one_errors,
        width=beam_width,
        n_starts=n_starts,
    )
    if fastest is not None and len(fastest) == n:
        return fastest

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
