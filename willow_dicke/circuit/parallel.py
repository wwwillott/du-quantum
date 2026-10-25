"""Schedule disjoint regions in the same moments."""

from __future__ import annotations

import cirq


def parallel_compose(*circuits: cirq.AbstractCircuit) -> cirq.Circuit:
    """Zip circuits qubit-wise. Depth is the longest circuit, not the sum."""
    present = [cirq.Circuit(circuit) for circuit in circuits if len(circuit)]
    if not present:
        return cirq.Circuit()
    if len(present) == 1:
        return present[0]
    merged = cirq.Circuit()
    width = max(len(circuit) for circuit in present)
    for index in range(width):
        operations = []
        for circuit in present:
            if index < len(circuit):
                operations.extend(circuit[index].operations)
        merged.append(cirq.Moment(operations))
    return merged


def concatenate(*circuits: cirq.AbstractCircuit) -> cirq.Circuit:
    """Append circuits in order. Used where the spec sequences route, then Divide."""
    combined = cirq.Circuit()
    for circuit in circuits:
        combined += circuit
    return combined
