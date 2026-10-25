"""Hamiltonian paths are searched only inside small leaves."""

from __future__ import annotations

import cirq

from willow_dicke.graph import WillowGraph
from willow_dicke.placement.staging import simple_path


def find_hamiltonian_path(
    willow: WillowGraph, qubits: set[cirq.GridQubit]
) -> tuple[cirq.GridQubit, ...] | None:
    """Exact simple path through every qubit of a leaf. Not for the full chip."""
    if len(qubits) > 16:
        raise ValueError(f"refusing Hamiltonian search on {len(qubits)} qubits")
    return simple_path(willow, qubits, len(qubits))
