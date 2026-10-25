"""Load Willow from Cirq. The device graph is never hand-copied."""

from __future__ import annotations

import cirq
import cirq_google
import networkx as nx

from willow_dicke.graph import WillowGraph


def _as_grid_graph(graph: nx.Graph, name: str, *, static_reference: bool) -> WillowGraph:
    qubits = frozenset(graph.nodes)
    if not qubits:
        raise ValueError(f"{name} has an empty qubit set")
    if not all(isinstance(qubit, cirq.GridQubit) for qubit in qubits):
        raise TypeError(f"{name} nodes are not cirq.GridQubit")
    copied = graph.copy()
    return WillowGraph(name=name, graph=copied, qubits=qubits, static_reference=static_reference)


def load_static_willow() -> WillowGraph:
    """Static 105-qubit Willow105 topology. This is the theorem target."""
    device = cirq_google.Willow105
    if len(device.metadata.qubit_set) != 105:
        raise AssertionError(
            f"Willow105 qubit_set has {len(device.metadata.qubit_set)} qubits, expected 105"
        )
    willow = _as_grid_graph(device.metadata.nx_graph, "Willow105", static_reference=True)
    if len(willow.qubits) != 105:
        raise AssertionError(f"Willow105 graph has {len(willow.qubits)} nodes, expected 105")
    return willow


def load_live_device(processor_id: str) -> WillowGraph:
    """Coupling graph of a virtual or cloud processor, as that processor reports it.

    Loading this graph does not submit a job.
    """
    engine = cirq_google.engine.create_default_noisy_quantum_virtual_machine(processor_id)
    device = engine.get_processor(processor_id).get_device()
    return _as_grid_graph(
        device.metadata.nx_graph,
        f"live:{processor_id}",
        static_reference=False,
    )


def diff_against_static(live: WillowGraph, static: WillowGraph | None = None) -> dict:
    """Qubits and couplers present in one graph and not the other."""
    static = static or load_static_willow()
    static_edges = _edge_set(static)
    live_edges = _edge_set(live)
    return {
        "live_name": live.name,
        "static_name": static.name,
        "missing_qubits": sorted(_coord(q) for q in static.qubits - live.qubits),
        "extra_qubits": sorted(_coord(q) for q in live.qubits - static.qubits),
        "missing_edges": sorted(static_edges - live_edges),
        "extra_edges": sorted(live_edges - static_edges),
        "same_topology": static.qubits == live.qubits and static_edges == live_edges,
    }


def _coord(qubit: cirq.GridQubit) -> tuple[int, int]:
    return (qubit.row, qubit.col)


def _edge_set(willow: WillowGraph) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    edges = set()
    for a, b in willow.graph.edges:
        left, right = sorted((_coord(a), _coord(b)))
        edges.add((left, right))
    return edges
