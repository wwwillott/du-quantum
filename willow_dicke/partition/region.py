"""An induced, connected piece of the Willow graph."""

from __future__ import annotations

from dataclasses import dataclass

import cirq
import networkx as nx

from willow_dicke.graph import WillowGraph


def _coord(qubit: cirq.GridQubit) -> tuple[int, int]:
    return (qubit.row, qubit.col)


@dataclass
class Region:
    """Qubits of one recursive region. Connectivity is checked, not assumed."""

    id: str
    qubits: frozenset[cirq.GridQubit]
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    parent_id: str | None
    depth: int

    @classmethod
    def from_qubits(
        cls,
        willow: WillowGraph,
        qubits: frozenset[cirq.GridQubit] | set[cirq.GridQubit],
        *,
        region_id: str,
        parent_id: str | None,
        depth: int,
    ) -> "Region":
        frozen = frozenset(qubits)
        if not frozen:
            raise ValueError("a region cannot be empty")
        unknown = frozen - willow.qubits
        if unknown:
            raise ValueError(f"region {region_id} contains qubits outside the device")
        rows = [qubit.row for qubit in frozen]
        cols = [qubit.col for qubit in frozen]
        region = cls(
            id=region_id,
            qubits=frozen,
            row_min=min(rows),
            row_max=max(rows),
            col_min=min(cols),
            col_max=max(cols),
            parent_id=parent_id,
            depth=depth,
        )
        if not region.is_connected(willow):
            raise ValueError(f"region {region_id} is disconnected ({region.size} qubits)")
        return region

    @property
    def size(self) -> int:
        return len(self.qubits)

    @property
    def width(self) -> int:
        return self.col_max - self.col_min + 1

    @property
    def height(self) -> int:
        return self.row_max - self.row_min + 1

    def bbox_span(self) -> int:
        """(width - 1) + (height - 1), the bounding-box path length."""
        return (self.width - 1) + (self.height - 1)

    def induced_graph(self, willow: WillowGraph) -> nx.Graph:
        return willow.graph.subgraph(self.qubits).copy()

    def is_connected(self, willow: WillowGraph) -> bool:
        if self.size <= 1:
            return True
        return nx.is_connected(willow.graph.subgraph(self.qubits))

    def boundary_edges(
        self, willow: WillowGraph, other: "Region"
    ) -> list[tuple[cirq.GridQubit, cirq.GridQubit]]:
        """Device edges with one end in this region and one end in ``other``."""
        edges = []
        for qubit in sorted(self.qubits, key=_coord):
            for neighbor in sorted(willow.graph.neighbors(qubit), key=_coord):
                if neighbor in other.qubits:
                    edges.append((qubit, neighbor))
        return edges

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "size": self.size,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "row_min": self.row_min,
            "row_max": self.row_max,
            "col_min": self.col_min,
            "col_max": self.col_max,
            "bbox_span": self.bbox_span(),
            "qubits": [_coord(qubit) for qubit in sorted(self.qubits, key=_coord)],
        }
