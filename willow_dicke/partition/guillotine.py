"""Axis-aligned guillotine cuts. A cut is legal only when both sides are connected."""

from __future__ import annotations

from dataclasses import dataclass

import cirq
import networkx as nx

from willow_dicke.graph import WillowGraph
from willow_dicke.partition.region import Region


@dataclass(frozen=True)
class GuillotineCut:
    axis: str
    coordinate: int
    left: frozenset[cirq.GridQubit]
    right: frozenset[cirq.GridQubit]

    @property
    def imbalance(self) -> int:
        return max(len(self.left), len(self.right))


def _connected(willow: WillowGraph, qubits: frozenset[cirq.GridQubit]) -> bool:
    if len(qubits) <= 1:
        return bool(qubits)
    return nx.is_connected(willow.graph.subgraph(qubits))


def enumerate_cuts(willow: WillowGraph, region: Region) -> list[GuillotineCut]:
    """Every vertical and horizontal cut with two nonempty sides.

    Disconnected sides are omitted. Emptiness is omitted. The caller decides
    the weight constraint ``|A| >= k`` and ``|B| >= k``.
    """
    qubits = region.qubits
    rows = sorted({qubit.row for qubit in qubits})
    cols = sorted({qubit.col for qubit in qubits})
    cuts: list[GuillotineCut] = []
    for coordinate in cols[:-1]:
        left = frozenset(qubit for qubit in qubits if qubit.col <= coordinate)
        right = frozenset(qubit for qubit in qubits if qubit.col > coordinate)
        if left and right and _connected(willow, left) and _connected(willow, right):
            cuts.append(GuillotineCut("vertical", coordinate, left, right))
    for coordinate in rows[:-1]:
        left = frozenset(qubit for qubit in qubits if qubit.row <= coordinate)
        right = frozenset(qubit for qubit in qubits if qubit.row > coordinate)
        if left and right and _connected(willow, left) and _connected(willow, right):
            cuts.append(GuillotineCut("horizontal", coordinate, left, right))
    return cuts
