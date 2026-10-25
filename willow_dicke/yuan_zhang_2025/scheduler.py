"""Moment packing that keeps disjoint branches concurrent."""

from __future__ import annotations

from collections.abc import Sequence

import cirq

from willow_dicke.yuan_zhang_2025.divide_lnn import pack_earliest_respecting_order


def merge_parallel_circuits(left: cirq.Circuit, right: cirq.Circuit) -> cirq.Circuit:
    """Zip two circuits whose qubits are disjoint. Depth is the longer one."""
    left_qubits = set(left.all_qubits())
    right_qubits = set(right.all_qubits())
    shared = left_qubits & right_qubits
    if shared:
        raise ValueError(f"parallel circuits share {len(shared)} qubits")
    if not left:
        return cirq.Circuit(right)
    if not right:
        return cirq.Circuit(left)
    merged = cirq.Circuit()
    width = max(len(left), len(right))
    for index in range(width):
        operations = []
        if index < len(left):
            operations.extend(left[index].operations)
        if index < len(right):
            operations.extend(right[index].operations)
        if operations:
            merged.append(cirq.Moment(operations))
    return merged


def concatenate(circuits: Sequence[cirq.AbstractCircuit]) -> cirq.Circuit:
    combined = cirq.Circuit()
    for circuit in circuits:
        combined += circuit
    return combined


def pack(ops: Sequence[cirq.Operation]) -> cirq.Circuit:
    return pack_earliest_respecting_order(ops)
