"""Partition, staging, and SWAP-certificate tests. No quantum rotations."""

from __future__ import annotations

import cirq
import networkx as nx
import pytest

from willow_dicke.device import load_static_willow
from willow_dicke.graph import WillowGraph, topology_report
from willow_dicke.math.coefficients import amplitude_norm, vandermonde_holds
from willow_dicke.plan import build_topology_plan
from willow_dicke.routing.certificate import certify_route


@pytest.mark.parametrize(
    "n,m,k",
    [(4, 2, 2), (6, 3, 3), (7, 3, 3), (10, 4, 3)],
)
def test_split_coefficients_and_vandermonde(n, m, k):
    for ell in range(0, k + 1):
        assert vandermonde_holds(n, m, ell)
        assert amplitude_norm(n, m, ell) == pytest.approx(1.0)


def test_static_willow_is_the_published_chip():
    willow = load_static_willow()
    report = topology_report(willow)
    assert report["n_qubits"] == 105
    assert report["n_edges"] == 182
    assert report["connected"] is True
    assert report["diameter"] == 14
    assert report["row_occupancy"]["6"] == list(range(0, 15))


def test_line_corridor_moves_a_register():
    qubits = [cirq.GridQubit(0, index) for index in range(4)]
    graph = nx.Graph()
    graph.add_nodes_from(qubits)
    graph.add_edges_from(zip(qubits, qubits[1:]))
    willow = WillowGraph("line", graph, frozenset(qubits), static_reference=False)
    route = certify_route(
        willow,
        region_id="line",
        depth=1,
        region=set(qubits),
        source=(qubits[0], qubits[1]),
        target=(qubits[2], qubits[3]),
    )
    assert route.verified
    assert route.swap_count > 0
    assert all(willow.has_edge(a, b) for layer in route.swap_layers for a, b in layer.swaps)


def test_willow_k1_partition_certificate():
    willow = load_static_willow()
    plan = build_topology_plan(willow, k=1, leaf_factor=2)
    assert plan.passed, plan.failures
    assert plan.metrics["internal_nodes"] >= 1
    assert plan.metrics["max_staging_path_length"] == 2
    assert plan.root.left_size + plan.root.right_size == 105
    assert {plan.root.left_size, plan.root.right_size} == {46, 59}
