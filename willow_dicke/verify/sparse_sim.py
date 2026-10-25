"""Sparse statevector for the full Willow circuit.

Intermediate Ry gates leave the fixed-weight subspace, so the state is a
dictionary of basis bitmasks, pruned when an amplitude underflows. Bit 0 of
the mask is ``qubits[0]``.
"""

from __future__ import annotations

import cirq
import numpy as np

from willow_dicke.circuit.lower import is_full_cz


class SparseState:
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.amplitudes: dict[int, complex] = {0: 1.0 + 0.0j}

    def apply(self, op: cirq.Operation, index_of: dict[cirq.Qid, int]) -> None:
        if cirq.num_qubits(op) == 1:
            self._apply_single(op, index_of[op.qubits[0]])
            return
        if is_full_cz(op):
            bit_a = index_of[op.qubits[0]]
            bit_b = index_of[op.qubits[1]]
            mask = (1 << bit_a) | (1 << bit_b)
            for basis, amplitude in self.amplitudes.items():
                if basis & mask == mask:
                    self.amplitudes[basis] = -amplitude
            return
        raise TypeError(f"sparse simulator expected 1-qubit or CZ, got {op.gate}")

    def _apply_single(self, op: cirq.Operation, bit: int) -> None:
        matrix = np.asarray(cirq.unitary(op), dtype=np.complex128)
        updated: dict[int, complex] = {}
        flag = 1 << bit
        for basis, amplitude in self.amplitudes.items():
            value = 1 if basis & flag else 0
            for output in (0, 1):
                coefficient = matrix[output, value] * amplitude
                if abs(coefficient) < 1e-12:
                    continue
                next_basis = basis & ~flag
                if output:
                    next_basis |= flag
                updated[next_basis] = updated.get(next_basis, 0.0j) + coefficient
        self.amplitudes = updated

    def run(self, circuit: cirq.AbstractCircuit, qubits: list[cirq.Qid]) -> None:
        index_of = {qubit: index for index, qubit in enumerate(qubits)}
        for op in circuit.all_operations():
            self.apply(op, index_of)


def weight_amplitudes(state: SparseState, qubits: list[cirq.Qid], weight: int) -> list[tuple[int, complex]]:
    """Amplitudes of the Hamming-weight ``weight`` basis states, as bitmasks."""
    del qubits
    found = []
    for basis, amplitude in state.amplitudes.items():
        if basis.bit_count() == weight and abs(amplitude) > 1e-10:
            found.append((basis, amplitude))
    return found
