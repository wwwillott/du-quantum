"""Authoritative Willow coupling graph and the statistics the proof records."""

from __future__ import annotations

from dataclasses import dataclass

import cirq
import networkx as nx


@dataclass(frozen=True)
class WillowGraph:
    """A copied device coupling graph. ``graph`` is authoritative for edges."""

    name: str
    graph: nx.Graph
    qubits: frozenset[cirq.GridQubit]
    static_reference: bool

    def adjacency(self) -> dict[cirq.GridQubit, tuple[cirq.GridQubit, ...]]:
        return {
            qubit: tuple(sorted(self.graph.neighbors(qubit), key=_coord))
            for qubit in self.qubits
        }

    def has_edge(self, a: cirq.Qid, b: cirq.Qid) -> bool:
        return self.graph.has_edge(a, b)


def _coord(qubit: cirq.GridQubit) -> tuple[int, int]:
    return (qubit.row, qubit.col)


def neighbors(
    willow: WillowGraph,
    qubit: cirq.GridQubit,
    allowed: set[cirq.GridQubit],
) -> list[cirq.GridQubit]:
    """Device neighbors inside ``allowed``, in deterministic coordinate order."""
    return [other for other in sorted(willow.graph.neighbors(qubit), key=_coord) if other in allowed]


def topology_report(willow: WillowGraph) -> dict:
    """Vertex, edge, occupancy, and diameter facts used by the certificate."""
    graph = willow.graph
    qubits = sorted(willow.qubits, key=_coord)
    rows: dict[int, list[int]] = {}
    cols: dict[int, list[int]] = {}
    for qubit in qubits:
        rows.setdefault(qubit.row, []).append(qubit.col)
        cols.setdefault(qubit.col, []).append(qubit.row)
    components = [sorted(component, key=_coord) for component in nx.connected_components(graph)]
    diameter = nx.diameter(graph) if nx.is_connected(graph) else None
    return {
        "name": willow.name,
        "static_reference": willow.static_reference,
        "n_qubits": len(qubits),
        "n_edges": graph.number_of_edges(),
        "connected": nx.is_connected(graph),
        "n_components": len(components),
        "diameter": diameter,
        "row_min": qubits[0].row,
        "row_max": qubits[-1].row,
        "col_min": min(qubit.col for qubit in qubits),
        "col_max": max(qubit.col for qubit in qubits),
        "row_occupancy": {str(row): sorted(cols_) for row, cols_ in sorted(rows.items())},
        "column_occupancy": {str(col): sorted(rows_) for col, rows_ in sorted(cols.items())},
        "edge_list": sorted(
            (min((a.row, a.col), (b.row, b.col)), max((a.row, a.col), (b.row, b.col)))
            for a, b in graph.edges
        ),
    }
