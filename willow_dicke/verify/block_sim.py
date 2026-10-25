"""Replay Divide, leaf, and SWAP blocks on a sparse computational-basis state.

Gate-by-gate replay of the compiled 105-qubit circuit leaves the fixed-weight
subspace at every Ry. Each block, applied as one unitary, maps the unary
subspace of that block back into a Dicke state, so a weight-k input stays
weight k. SWAP layers are bit permutations.
"""

from __future__ import annotations

import math

import cirq
import numpy as np


def replay(trace: list, qubits: list[cirq.Qid]) -> dict[int, complex]:
    """Return basis-mask amplitudes. Bit ``i`` is ``qubits[i]``."""
    index = {qubit: bit for bit, qubit in enumerate(qubits)}
    amplitudes: dict[int, complex] = {0: 1.0 + 0.0j}
    for kind, *payload in trace:
        if kind == "pauli_x":
            for qubit in payload[0]:
                amplitudes = _flip(amplitudes, index[qubit])
        elif kind == "swaps":
            for left, right in payload[0]:
                amplitudes = _exchange(amplitudes, index[left], index[right])
        elif kind == "unitary":
            circuit, block = payload
            if len(circuit) == 0 or len(block) == 0:
                continue
            matrix = np.asarray(
                cirq.Circuit(circuit).unitary(qubit_order=list(block)),
                dtype=np.complex128,
            )
            amplitudes = _apply(amplitudes, [index[qubit] for qubit in block], matrix)
        else:
            raise ValueError(f"unknown trace event {kind}")
    return amplitudes


def fixed_weight_error(amplitudes: dict[int, complex], n_qubits: int, weight: int) -> dict:
    target = 1.0 / math.sqrt(math.comb(n_qubits, weight))
    seen = 0
    worst = 0.0
    leaked = 0.0
    for basis, amplitude in amplitudes.items():
        if basis.bit_count() == weight:
            worst = max(worst, abs(abs(amplitude) - target))
            seen += 1
        else:
            leaked += abs(amplitude) ** 2
    expected = math.comb(n_qubits, weight)
    return {
        "weight": weight,
        "basis_states_found": seen,
        "basis_states_expected": expected,
        "max_abs_amplitude_error": float(worst),
        "leaked_probability": float(leaked),
        "passed": seen == expected and worst < 1e-8 and leaked < 1e-8,
    }


def _flip(amplitudes: dict[int, complex], bit: int) -> dict[int, complex]:
    updated: dict[int, complex] = {}
    flag = 1 << bit
    for basis, amplitude in amplitudes.items():
        updated[basis ^ flag] = updated.get(basis ^ flag, 0.0j) + amplitude
    return updated


def _exchange(amplitudes: dict[int, complex], left: int, right: int) -> dict[int, complex]:
    if left == right:
        return amplitudes
    updated: dict[int, complex] = {}
    for basis, amplitude in amplitudes.items():
        if ((basis >> left) & 1) != ((basis >> right) & 1):
            basis ^= (1 << left) | (1 << right)
        updated[basis] = updated.get(basis, 0.0j) + amplitude
    return updated


def _apply(amplitudes: dict[int, complex], bits: list[int], unitary: np.ndarray) -> dict[int, complex]:
    """Apply a block unitary. ``bits[0]`` is the most significant bit, matching Cirq."""
    width = len(bits)
    groups: dict[int, np.ndarray] = {}
    clear = 0
    for bit in bits:
        clear |= 1 << bit
    for basis, amplitude in amplitudes.items():
        outside = basis & ~clear
        local = 0
        for offset, bit in enumerate(bits):
            if basis & (1 << bit):
                local |= 1 << (width - 1 - offset)
        bucket = groups.get(outside)
        if bucket is None:
            bucket = np.zeros(1 << width, dtype=np.complex128)
            groups[outside] = bucket
        bucket[local] += amplitude
    updated: dict[int, complex] = {}
    for outside, vector in groups.items():
        output = unitary @ vector
        for local, amplitude in enumerate(output):
            if abs(amplitude) < 1e-12:
                continue
            basis = outside
            for offset, bit in enumerate(bits):
                if local & (1 << (width - 1 - offset)):
                    basis |= 1 << bit
            updated[basis] = updated.get(basis, 0.0j) + amplitude
    return updated
