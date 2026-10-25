"""Topology-only plan: partition, staging paths, and SWAP certificates.

Divide rotations are intentionally absent. A passing plan is the permission
to insert them later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cirq

from willow_dicke.benchmark.metrics import routing_metrics
from willow_dicke.graph import WillowGraph
from willow_dicke.partition.tree import LEAF_FACTOR, PartitionNode, build_partition_tree
from willow_dicke.placement.staging import simple_path
from willow_dicke.routing.certificate import RouteCertificate, certify_route
from willow_dicke.verify.checks import matching_problems, region_problems, route_problems


@dataclass
class TopologyPlan:
    willow: WillowGraph
    k: int
    leaf_factor: int
    root: PartitionNode
    routes: list[RouteCertificate] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures


def build_topology_plan(
    willow: WillowGraph,
    k: int,
    *,
    leaf_factor: int = LEAF_FACTOR,
) -> TopologyPlan:
    """Construct and check the routing plan on ``willow`` for weight ``k``."""
    root = build_partition_tree(willow, k, leaf_factor=leaf_factor)
    _assign_leaf_anchors(willow, root, None, k)
    routes = _collect_routes(willow, root, None)
    failures = region_problems(willow, root)
    failures.extend(node.failure for node in root.walk() if node.failure)
    failures.extend(route_problems(routes))
    failures.extend(matching_problems(willow, routes))
    # region_problems already includes node.failure; drop exact duplicates.
    deduped = list(dict.fromkeys(failures))
    plan = TopologyPlan(
        willow=willow,
        k=k,
        leaf_factor=leaf_factor,
        root=root,
        routes=routes,
        failures=deduped,
    )
    plan.metrics = routing_metrics(root, routes, n=len(willow.qubits), k=k)
    return plan


def _assign_leaf_anchors(
    willow: WillowGraph,
    node: PartitionNode,
    incoming: tuple[cirq.GridQubit, ...] | None,
    k: int,
) -> None:
    if node.is_leaf:
        length = min(k, node.region.size)
        if (
            incoming is not None
            and len(incoming) == length
            and set(incoming) <= node.region.qubits
        ):
            node.leaf_anchor = incoming
            return
        path = simple_path(willow, node.region.qubits, length)
        if path is None:
            node.failure = (
                f"no simple path of length {length} was constructed in leaf {node.region.id}"
            )
            return
        node.leaf_anchor = path
        return
    _assign_leaf_anchors(willow, node.left, node.left_anchor, k)
    _assign_leaf_anchors(willow, node.right, node.right_anchor, k)


def _collect_routes(
    willow: WillowGraph,
    node: PartitionNode,
    incoming: tuple[cirq.GridQubit, ...] | None,
) -> list[RouteCertificate]:
    routes: list[RouteCertificate] = []
    if node.is_leaf:
        target = node.leaf_anchor
        if incoming is not None and target is not None and incoming != target:
            routes.append(
                certify_route(
                    willow,
                    region_id=node.region.id,
                    depth=node.region.depth,
                    region=set(node.region.qubits),
                    source=incoming,
                    target=target,
                )
            )
        return routes
    # The root register is prepared on the staging input. Every deeper
    # register is routed from the anchor its parent already holds.
    if incoming is not None and node.left_anchor is not None and incoming != node.left_anchor:
        routes.append(
            certify_route(
                willow,
                region_id=node.region.id,
                depth=node.region.depth,
                region=set(node.region.qubits),
                source=incoming,
                target=node.left_anchor,
            )
        )
    routes.extend(_collect_routes(willow, node.left, node.left_anchor))
    routes.extend(_collect_routes(willow, node.right, node.right_anchor))
    return routes
