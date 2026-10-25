"""A corridor is a device path that contains both the source and target register."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cirq

from willow_dicke.graph import WillowGraph, neighbors


@dataclass(frozen=True)
class RoutingCorridor:
    path: tuple[cirq.Qid, ...]
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]


def _bfs(
    willow: WillowGraph,
    start: cirq.GridQubit,
    goal: cirq.GridQubit,
    allowed: set[cirq.GridQubit],
) -> tuple[cirq.GridQubit, ...] | None:
    if start == goal:
        return (start,)
    previous: dict[cirq.GridQubit, cirq.GridQubit | None] = {start: None}
    queue: deque[cirq.GridQubit] = deque([start])
    while queue:
        node = queue.popleft()
        for other in neighbors(willow, node, allowed):
            if other in previous:
                continue
            previous[other] = node
            if other == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                path.reverse()
                return tuple(path)
            queue.append(other)
    return None


def _assemble(
    source: tuple[cirq.GridQubit, ...],
    target: tuple[cirq.GridQubit, ...],
    src_end: int,
    tgt_end: int,
    connector: tuple[cirq.GridQubit, ...],
) -> RoutingCorridor | None:
    src_walk = source if src_end == len(source) - 1 else tuple(reversed(source))
    tgt_walk = target if tgt_end == 0 else tuple(reversed(target))
    if connector[0] != src_walk[-1] or connector[-1] != tgt_walk[0]:
        return None
    path = src_walk + connector[1:-1] + tgt_walk
    if len(path) != len(set(path)):
        return None
    index = {qubit: i for i, qubit in enumerate(path)}
    return RoutingCorridor(
        path=path,
        source_indices=tuple(index[qubit] for qubit in source),
        target_indices=tuple(index[qubit] for qubit in target),
    )


def _endpoint_corridor(
    willow: WillowGraph,
    region: set[cirq.GridQubit],
    source: tuple[cirq.GridQubit, ...],
    target: tuple[cirq.GridQubit, ...],
) -> RoutingCorridor | None:
    """Connect one end of the source path to one end of the target path."""
    if set(source) & set(target):
        return None
    k_source = len(source)
    k_target = len(target)
    for src_end in (k_source - 1, 0):
        for tgt_end in (0, k_target - 1):
            frozen = set(source) | set(target)
            frozen.discard(source[src_end])
            frozen.discard(target[tgt_end])
            allowed = set(region) - frozen
            allowed.add(source[src_end])
            allowed.add(target[tgt_end])
            connector = _bfs(willow, source[src_end], target[tgt_end], allowed)
            if connector is None:
                continue
            corridor = _assemble(source, target, src_end, tgt_end, connector)
            if corridor is not None:
                return corridor
    return None


def _covering_path(
    willow: WillowGraph,
    region: set[cirq.GridQubit],
    must: set[cirq.GridQubit],
    budget: int = 100_000,
) -> tuple[cirq.GridQubit, ...] | None:
    """Simple path inside ``region`` that visits every qubit in ``must``."""
    if not must:
        return ()
    found: list[tuple[cirq.GridQubit, ...]] = []
    visits = 0

    def rec(path: tuple[cirq.GridQubit, ...], unseen: set[cirq.GridQubit]) -> None:
        nonlocal visits
        if found or visits > budget:
            return
        visits += 1
        if not unseen:
            found.append(path)
            return
        options = neighbors(willow, path[-1], region)
        options = [qubit for qubit in options if qubit not in path]
        options.sort(
            key=lambda qubit: (
                qubit not in unseen,
                sum(1 for other in neighbors(willow, qubit, region) if other not in path),
                qubit.row,
                qubit.col,
            )
        )
        for qubit in options:
            rec(path + (qubit,), unseen - {qubit})

    for start in sorted(must, key=lambda qubit: (qubit.row, qubit.col)):
        if found:
            break
        rec((start,), set(must) - {start})
    if not found:
        return None
    path = found[0]
    while path and path[0] not in must:
        path = path[1:]
    while path and path[-1] not in must:
        path = path[:-1]
    return path


def find_corridor(
    willow: WillowGraph,
    region: set[cirq.GridQubit],
    source: tuple[cirq.GridQubit, ...],
    target: tuple[cirq.GridQubit, ...],
) -> RoutingCorridor | None:
    """Path through ``region`` on which the source register can slide onto the target."""
    if len(source) != len(target) or not source:
        return None
    if any(qubit not in region for qubit in source + target):
        return None
    if source == target:
        return RoutingCorridor(
            path=source,
            source_indices=tuple(range(len(source))),
            target_indices=tuple(range(len(target))),
        )
    if set(source) == set(target):
        index = {qubit: i for i, qubit in enumerate(source)}
        return RoutingCorridor(
            path=source,
            source_indices=tuple(range(len(source))),
            target_indices=tuple(index[qubit] for qubit in target),
        )
    direct = _endpoint_corridor(willow, region, source, target)
    if direct is not None:
        return direct
    covered = _covering_path(willow, region, set(source) | set(target))
    if covered is None:
        return None
    index = {qubit: i for i, qubit in enumerate(covered)}
    if any(qubit not in index for qubit in source + target):
        return None
    return RoutingCorridor(
        path=covered,
        source_indices=tuple(index[qubit] for qubit in source),
        target_indices=tuple(index[qubit] for qubit in target),
    )
