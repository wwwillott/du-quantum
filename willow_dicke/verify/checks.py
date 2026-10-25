"""Assertions that turn the partition and the routes into a certificate."""

from __future__ import annotations

import cirq

from willow_dicke.graph import WillowGraph
from willow_dicke.partition.tree import PartitionNode
from willow_dicke.placement.staging import validate_staging_path
from willow_dicke.routing.certificate import RouteCertificate


def region_problems(willow: WillowGraph, root: PartitionNode) -> list[str]:
    """Connectivity, disjointness, and exact cover of the device."""
    problems = []
    seen: set[cirq.GridQubit] = set()
    for node in root.walk():
        if not node.region.is_connected(willow):
            problems.append(f"{node.region.id} is disconnected")
        if node.is_leaf:
            overlap = seen & node.region.qubits
            if overlap:
                problems.append(f"{node.region.id} overlaps another leaf")
            seen |= node.region.qubits
        else:
            if node.left is None or node.right is None or node.staging_path is None:
                problems.append(f"{node.region.id} is internal but missing a child or a staging path")
                continue
            if node.failure:
                problems.append(f"{node.region.id}: {node.failure}")
            left = node.left.region.qubits
            right = node.right.region.qubits
            if left & right:
                problems.append(f"{node.region.id} children overlap")
            if left | right != node.region.qubits:
                problems.append(f"{node.region.id} children do not cover the parent")
            problems.extend(
                f"{node.region.id}: {problem}"
                for problem in validate_staging_path(
                    willow, node.staging_path, left, right, len(node.staging_path) // 2
                )
            )
    if seen != willow.qubits:
        problems.append(
            f"leaves cover {len(seen)} qubits, device has {len(willow.qubits)}"
        )
    return problems


def route_problems(routes: list[RouteCertificate]) -> list[str]:
    problems = []
    for route in routes:
        if not route.verified:
            problems.append(f"{route.region_id} depth {route.depth}: {'; '.join(route.problems)}")
    return problems


def matching_problems(willow: WillowGraph, routes: list[RouteCertificate]) -> list[str]:
    problems = []
    for route in routes:
        for index, layer in enumerate(route.swap_layers):
            used: set[cirq.Qid] = set()
            for left, right in layer.swaps:
                if not willow.has_edge(left, right):
                    problems.append(f"{route.region_id} layer {index} uses a non-device edge")
                if left in used or right in used:
                    problems.append(f"{route.region_id} layer {index} is not a matching")
                used.add(left)
                used.add(right)
    return problems
