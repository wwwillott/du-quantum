"""Boundary staging paths for the Divide register.

A legal path has length ``2k``, lies on device edges, puts its first ``k``
qubits in the left child, and crosses into the right child exactly once.
"""

from __future__ import annotations

import cirq

from willow_dicke.graph import WillowGraph, neighbors
from willow_dicke.partition.guillotine import GuillotineCut


def _turns(path: tuple[cirq.GridQubit, ...]) -> int:
    turns = 0
    previous = None
    for left, right in zip(path, path[1:]):
        step = (right.row - left.row, right.col - left.col)
        if previous is not None and step != previous:
            turns += 1
        previous = step
    return turns


def _span(path: tuple[cirq.GridQubit, ...]) -> int:
    rows = [qubit.row for qubit in path]
    cols = [qubit.col for qubit in path]
    return (max(rows) - min(rows)) + (max(cols) - min(cols))


def _path_ending_at(
    willow: WillowGraph,
    region: frozenset[cirq.GridQubit],
    end: cirq.GridQubit,
    length: int,
) -> tuple[cirq.GridQubit, ...] | None:
    """Simple path of ``length`` qubits inside ``region`` whose last vertex is ``end``."""
    if length <= 0 or end not in region or len(region) < length:
        return None
    if length == 1:
        return (end,)
    grown = _grow(willow, region, end, length)
    if grown is None:
        return None
    return tuple(reversed(grown))


def _path_starting_at(
    willow: WillowGraph,
    region: frozenset[cirq.GridQubit],
    start: cirq.GridQubit,
    length: int,
) -> tuple[cirq.GridQubit, ...] | None:
    if length <= 0 or start not in region or len(region) < length:
        return None
    if length == 1:
        return (start,)
    grown = _grow(willow, region, start, length)
    if grown is None:
        return None
    return grown


def _grow(
    willow: WillowGraph,
    region: frozenset[cirq.GridQubit],
    origin: cirq.GridQubit,
    length: int,
) -> tuple[cirq.GridQubit, ...] | None:
    """Grow a simple path outward from ``origin``. Exact when the region is tight."""
    if len(region) <= length + 4:
        return _exact(willow, region, origin, length)
    beam = _beam(willow, region, origin, length, width=48)
    if beam is not None:
        return beam
    return _exact(willow, region, origin, length, budget=250_000)


def _onward(
    willow: WillowGraph,
    region: frozenset[cirq.GridQubit],
    qubit: cirq.GridQubit,
    used: set[cirq.GridQubit],
) -> int:
    return sum(1 for other in neighbors(willow, qubit, region) if other not in used)


def _exact(
    willow: WillowGraph,
    region: frozenset[cirq.GridQubit],
    origin: cirq.GridQubit,
    length: int,
    budget: int = 200_000,
) -> tuple[cirq.GridQubit, ...] | None:
    found: list[tuple[cirq.GridQubit, ...]] = []
    visits = 0

    def rec(path: tuple[cirq.GridQubit, ...], used: set[cirq.GridQubit]) -> None:
        nonlocal visits
        if found or visits > budget:
            return
        visits += 1
        if len(path) == length:
            found.append(path)
            return
        options = neighbors(willow, path[-1], region)
        options = [qubit for qubit in options if qubit not in used]
        options.sort(key=lambda qubit: (_onward(willow, region, qubit, used | {qubit}), qubit.row, qubit.col))
        for qubit in options:
            used.add(qubit)
            rec(path + (qubit,), used)
            used.remove(qubit)

    rec((origin,), {origin})
    return found[0] if found else None


def _beam(
    willow: WillowGraph,
    region: frozenset[cirq.GridQubit],
    origin: cirq.GridQubit,
    length: int,
    width: int,
) -> tuple[cirq.GridQubit, ...] | None:
    frontier: list[tuple[int, tuple[cirq.GridQubit, ...]]] = [(0, (origin,))]
    for _step in range(length - 1):
        expanded: list[tuple[int, tuple[cirq.GridQubit, ...]]] = []
        seen: set[tuple[cirq.GridQubit, ...]] = set()
        for turns, path in frontier:
            previous = None
            if len(path) >= 2:
                previous = (path[-1].row - path[-2].row, path[-1].col - path[-2].col)
            for qubit in neighbors(willow, path[-1], region):
                if qubit in path:
                    continue
                extended = path + (qubit,)
                if extended in seen:
                    continue
                seen.add(extended)
                step = (qubit.row - path[-1].row, qubit.col - path[-1].col)
                expanded.append((turns + (previous is not None and step != previous), extended))
        if not expanded:
            return None
        expanded.sort(key=lambda item: (item[0], _span(item[1]), [(q.row, q.col) for q in item[1]]))
        frontier = expanded[:width]
    return frontier[0][1]


def simple_path(
    willow: WillowGraph,
    region: frozenset[cirq.GridQubit] | set[cirq.GridQubit],
    length: int,
) -> tuple[cirq.GridQubit, ...] | None:
    """A constructed simple path of ``length`` inside ``region``, or None."""
    qubits = frozenset(region)
    starts = sorted(qubits, key=lambda qubit: (_degree(willow, qubits, qubit), qubit.row, qubit.col))
    for origin in starts:
        path = _path_starting_at(willow, qubits, origin, length)
        if path is not None:
            return path
    return None


def _degree(willow: WillowGraph, region: frozenset[cirq.GridQubit], qubit: cirq.GridQubit) -> int:
    return len(neighbors(willow, qubit, region))


def force_staging_prefix(
    willow: WillowGraph,
    cut: GuillotineCut,
    prefix: tuple[cirq.GridQubit, ...],
) -> tuple[cirq.GridQubit, ...] | None:
    """Use ``prefix`` as the left half when it already ends on the cut."""
    k = len(prefix)
    if k < 1 or any(qubit not in cut.left for qubit in prefix):
        return None
    for left, right in zip(prefix, prefix[1:]):
        if not willow.has_edge(left, right):
            return None
    for neighbor in neighbors(willow, prefix[-1], set(cut.right)):
        right_path = _path_starting_at(willow, cut.right, neighbor, k)
        if right_path is not None:
            return prefix + right_path
    return None


def path_quality(path: tuple[cirq.GridQubit, ...]) -> tuple[int, int]:
    """Lower is better: fewer turns, then a tighter bounding box."""
    return (_turns(path), _span(path))


def find_boundary_staging_path(
    willow: WillowGraph,
    cut: GuillotineCut,
    k: int,
) -> tuple[cirq.GridQubit, ...] | None:
    """Best 2k-path that crosses ``cut`` once, or None if none is constructed."""
    if k < 1 or len(cut.left) < k or len(cut.right) < k:
        return None
    best: tuple[cirq.GridQubit, ...] | None = None
    best_key: tuple[int, int, tuple] | None = None
    edges = []
    for qubit in sorted(cut.left, key=lambda q: (q.row, q.col)):
        for other in neighbors(willow, qubit, set(cut.right)):
            edges.append((qubit, other))
    for left_end, right_start in edges:
        left_path = _path_ending_at(willow, cut.left, left_end, k)
        if left_path is None:
            continue
        right_path = _path_starting_at(willow, cut.right, right_start, k)
        if right_path is None:
            continue
        path = left_path + right_path
        if len(path) != 2 * k or len(set(path)) != 2 * k:
            continue
        if not willow.has_edge(left_path[-1], right_path[0]):
            continue
        key = (*path_quality(path), (left_end.row, left_end.col, right_start.row, right_start.col))
        if best_key is None or key < best_key:
            best = path
            best_key = key
            if key[0] == 0:
                break
    return best


def validate_staging_path(
    willow: WillowGraph,
    path: tuple[cirq.GridQubit, ...],
    left: frozenset[cirq.GridQubit],
    right: frozenset[cirq.GridQubit],
    k: int,
) -> list[str]:
    """Human-readable failures. An empty list is the certificate for this path."""
    problems = []
    if len(path) != 2 * k:
        problems.append(f"staging path has length {len(path)}, expected {2 * k}")
    if len(set(path)) != len(path):
        problems.append("staging path repeats a qubit")
    for qubit in path[:k]:
        if qubit not in left:
            problems.append(f"{qubit} is on the left half but not in the left child")
    for qubit in path[k:]:
        if qubit not in right:
            problems.append(f"{qubit} is on the right half but not in the right child")
    for left_q, right_q in zip(path, path[1:]):
        if not willow.has_edge(left_q, right_q):
            problems.append(f"missing device edge {left_q} -- {right_q}")
    crossings = [
        (path[i], path[i + 1])
        for i in range(len(path) - 1)
        if (path[i] in left and path[i + 1] in right) or (path[i] in right and path[i + 1] in left)
    ]
    if len(crossings) != 1:
        problems.append(f"staging path crosses the cut {len(crossings)} times, expected 1")
    return problems
