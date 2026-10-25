"""Deterministic guillotine tree. The same graph and k always rebuild it."""

from __future__ import annotations

from dataclasses import dataclass

import cirq

from willow_dicke.graph import WillowGraph
from willow_dicke.partition.guillotine import GuillotineCut, enumerate_cuts
from willow_dicke.partition.region import Region
from willow_dicke.placement.anchors import split_anchors
from willow_dicke.placement.staging import (
    find_boundary_staging_path,
    force_staging_prefix,
    path_quality,
    simple_path,
)
from willow_dicke.routing.corridor import find_corridor

LEAF_FACTOR = 2


@dataclass
class PartitionNode:
    region: Region
    left: "PartitionNode | None"
    right: "PartitionNode | None"
    split_axis: str | None
    split_coordinate: int | None
    left_size: int
    right_size: int
    staging_path: tuple[cirq.GridQubit, ...] | None
    left_anchor: tuple[cirq.GridQubit, ...] | None
    right_anchor: tuple[cirq.GridQubit, ...] | None
    cut_edges: tuple[tuple[cirq.GridQubit, cirq.GridQubit], ...]
    leaf_anchor: tuple[cirq.GridQubit, ...] | None = None
    failure: str | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def walk(self) -> list["PartitionNode"]:
        nodes = [self]
        if self.left is not None:
            nodes.extend(self.left.walk())
        if self.right is not None:
            nodes.extend(self.right.walk())
        return nodes

    def to_dict(self) -> dict:
        def coords(path):
            if path is None:
                return None
            return [(qubit.row, qubit.col) for qubit in path]

        return {
            "region": self.region.to_dict(),
            "split_axis": self.split_axis,
            "split_coordinate": self.split_coordinate,
            "left_size": self.left_size,
            "right_size": self.right_size,
            "failure": self.failure,
            "staging_path": coords(self.staging_path),
            "left_anchor": coords(self.left_anchor),
            "right_anchor": coords(self.right_anchor),
            "leaf_anchor": coords(self.leaf_anchor),
            "cut_edges": [
                [(a.row, a.col), (b.row, b.col)] for a, b in self.cut_edges
            ],
            "left": None if self.left is None else self.left.to_dict(),
            "right": None if self.right is None else self.right.to_dict(),
        }


class _Ids:
    def __init__(self) -> None:
        self.count = 0

    def next(self, depth: int) -> str:
        region_id = f"d{depth}_{self.count}"
        self.count += 1
        return region_id


def build_partition_tree(
    willow: WillowGraph,
    k: int,
    *,
    leaf_factor: int = LEAF_FACTOR,
) -> PartitionNode:
    """Partition every device qubit. Leaves start at size ``leaf_factor * k``."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if leaf_factor < 1:
        raise ValueError("leaf_factor must be at least 1")
    root_region = Region.from_qubits(
        willow,
        willow.qubits,
        region_id="d0_0",
        parent_id=None,
        depth=0,
    )
    return _split(willow, root_region, k, leaf_factor, _Ids(), None)


def _split(
    willow: WillowGraph,
    region: Region,
    k: int,
    leaf_factor: int,
    ids: _Ids,
    incoming: tuple[cirq.GridQubit, ...] | None,
) -> PartitionNode:
    leaf_limit = leaf_factor * k
    if region.size <= leaf_limit:
        node = _leaf(region)
        length = min(k, region.size)
        if incoming is not None and len(incoming) == length and set(incoming) <= region.qubits:
            node.leaf_anchor = incoming
        else:
            path = simple_path(willow, region.qubits, length)
            if path is None:
                node.failure = (
                    f"no simple path of length {length} was constructed in leaf {region.id}"
                )
            else:
                node.leaf_anchor = path
        return node
    ranked = _ranked_cuts(willow, region, k, incoming)
    child_failures: list[str] = []
    for cut, path, edges in ranked:
        left_anchor, right_anchor = split_anchors(path, k)
        left_region = Region.from_qubits(
            willow,
            cut.left,
            region_id=ids.next(region.depth + 1),
            parent_id=region.id,
            depth=region.depth + 1,
        )
        right_region = Region.from_qubits(
            willow,
            cut.right,
            region_id=ids.next(region.depth + 1),
            parent_id=region.id,
            depth=region.depth + 1,
        )
        left_node = _split(willow, left_region, k, leaf_factor, ids, left_anchor)
        if left_node.failure:
            child_failures.append(
                f"{region.id} via {cut.axis} {cut.coordinate}: {left_node.failure}"
            )
            continue
        right_node = _split(willow, right_region, k, leaf_factor, ids, right_anchor)
        if right_node.failure:
            child_failures.append(
                f"{region.id} via {cut.axis} {cut.coordinate}: {right_node.failure}"
            )
            continue
        return PartitionNode(
            region=region,
            left=left_node,
            right=right_node,
            split_axis=cut.axis,
            split_coordinate=cut.coordinate,
            left_size=len(cut.left),
            right_size=len(cut.right),
            staging_path=path,
            left_anchor=left_anchor,
            right_anchor=right_anchor,
            cut_edges=tuple(edges),
            failure=None,
        )
    node = _leaf(region)
    node.failure = child_failures[0] if child_failures else (
        f"no connected guillotine cut of {region.id} "
        f"({region.size} qubits) has a constructed {2 * k}-qubit staging path"
    )
    return node


def _leaf(region: Region) -> PartitionNode:
    return PartitionNode(
        region=region,
        left=None,
        right=None,
        split_axis=None,
        split_coordinate=None,
        left_size=0,
        right_size=0,
        staging_path=None,
        left_anchor=None,
        right_anchor=None,
        cut_edges=(),
        failure=None,
    )


def _ranked_cuts(
    willow: WillowGraph,
    region: Region,
    k: int,
    incoming: tuple[cirq.GridQubit, ...] | None,
) -> list[tuple[GuillotineCut, tuple[cirq.GridQubit, ...], list]]:
    """Legal cuts, best score first.

    A cut is dropped when the incoming register has no constructed corridor
    onto that cut's staging anchor. The Divide path and the route have to
    exist together; otherwise the split is not a proof.
    """
    ranked = []
    for cut in enumerate_cuts(willow, region):
        if len(cut.left) < k or len(cut.right) < k:
            continue
        for oriented in _orientations(cut, incoming):
            for path, _matched in _staging_candidates(willow, oriented, k, incoming):
                left_anchor = path[:k]
                if (
                    incoming is not None
                    and incoming != left_anchor
                    and find_corridor(willow, set(region.qubits), incoming, left_anchor) is None
                ):
                    continue
                edges = _cut_edges(willow, oriented)
                left_span = _bbox_span(oriented.left)
                right_span = _bbox_span(oriented.right)
                turns, path_span = path_quality(path)
                key = (
                    max(len(oriented.left), len(oriented.right)),
                    max(left_span, right_span),
                    -len(edges),
                    0 if incoming is not None and incoming == left_anchor else 1,
                    turns,
                    path_span,
                    0 if oriented.axis == "vertical" else 1,
                    oriented.coordinate,
                    0 if set(oriented.left) == set(cut.left) else 1,
                )
                ranked.append((key, oriented, path, edges))
    ranked.sort(key=lambda item: item[0])
    # One entry per child pair. A flipped orientation is a different pair.
    unique = []
    seen = set()
    for _key, cut, path, edges in ranked:
        marker = (frozenset(cut.left), path)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append((cut, path, edges))
    return unique


def _orientations(
    cut: GuillotineCut, incoming: tuple[cirq.GridQubit, ...] | None
) -> list[GuillotineCut]:
    options = [cut]
    flipped = GuillotineCut(cut.axis, cut.coordinate, cut.right, cut.left)
    if incoming is not None and set(incoming) <= cut.right:
        return [flipped, cut]
    options.append(flipped)
    return options


def _staging_candidates(
    willow: WillowGraph,
    cut: GuillotineCut,
    k: int,
    incoming: tuple[cirq.GridQubit, ...] | None,
) -> list[tuple[tuple[cirq.GridQubit, ...], bool]]:
    found = []
    if incoming is not None and len(incoming) == k:
        for prefix in (incoming, tuple(reversed(incoming))):
            forced = force_staging_prefix(willow, cut, prefix)
            if forced is not None:
                found.append((forced, prefix == incoming))
    natural = find_boundary_staging_path(willow, cut, k)
    if natural is not None:
        found.append((natural, False))
    return found


def _cut_edges(willow: WillowGraph, cut: GuillotineCut) -> list:
    edges = []
    for qubit in sorted(cut.left, key=lambda q: (q.row, q.col)):
        for other in sorted(willow.graph.neighbors(qubit), key=lambda q: (q.row, q.col)):
            if other in cut.right:
                edges.append((qubit, other))
    return edges


def _bbox_span(qubits: frozenset[cirq.GridQubit]) -> int:
    rows = [qubit.row for qubit in qubits]
    cols = [qubit.col for qubit in qubits]
    return (max(cols) - min(cols)) + (max(rows) - min(rows))
