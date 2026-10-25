"""Depth metrics for the topology stage. No gate counts yet: rotations are not inserted."""

from __future__ import annotations

import math

from willow_dicke.partition.tree import PartitionNode
from willow_dicke.routing.certificate import RouteCertificate
from willow_dicke.routing.scheduler import merge_disjoint_route_schedules


def routing_metrics(
    root: PartitionNode,
    routes: list[RouteCertificate],
    *,
    n: int,
    k: int,
) -> dict:
    nodes = root.walk()
    internals = [node for node in nodes if not node.is_leaf]
    leaves = [node for node in nodes if node.is_leaf]
    child_sizes = [size for node in internals for size in (node.left_size, node.right_size)]
    staging_lengths = [len(node.staging_path) for node in internals if node.staging_path is not None]
    corridors = [len(route.corridor) for route in routes]
    by_depth: dict[int, list] = {}
    for route in routes:
        by_depth.setdefault(route.depth, []).append(route.swap_layers)
    parallel_depth = 0
    for schedules in by_depth.values():
        merged = merge_disjoint_route_schedules(schedules)
        parallel_depth += len(merged)
    swaps = sum(route.swap_count for route in routes)
    benchmark = k * math.log2(n / k) + 15
    return {
        "n": n,
        "k": k,
        "partition_depth": max(node.region.depth for node in nodes),
        "internal_nodes": len(internals),
        "leaves": len(leaves),
        "min_child_size": min(child_sizes) if child_sizes else None,
        "max_child_size": max(child_sizes) if child_sizes else None,
        "max_staging_path_length": max(staging_lengths) if staging_lengths else 0,
        "max_corridor_length": max(corridors) if corridors else 0,
        "mean_corridor_length": (sum(corridors) / len(corridors)) if corridors else 0.0,
        "parallel_routing_depth": parallel_depth,
        "swap_count": swaps,
        "routes": len(routes),
        "failed_routes": sum(not route.verified for route in routes),
        "structural_benchmark": benchmark,
        "routing_depth_over_benchmark": parallel_depth / benchmark if benchmark else None,
    }
