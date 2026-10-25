"""Statevector checks and Willow gate checks for the 2025 compiler."""

from __future__ import annotations

import math

import cirq
import numpy as np

from willow_dicke.yuan_zhang_2025.config import AMPLITUDE_ATOL, MAX_STATEVECTOR_QUBITS


def statevector_bytes(n_qubits: int) -> int:
    """Complex128 statevector footprint. Callers refuse this past the limit."""
    return (1 << n_qubits) * 16


def assert_simulable(n_qubits: int) -> None:
    if n_qubits > MAX_STATEVECTOR_QUBITS:
        raise MemoryError(
            f"{n_qubits} qubits need {statevector_bytes(n_qubits)} bytes; "
            f"the limit is {MAX_STATEVECTOR_QUBITS}"
        )


def simulate(circuit: cirq.AbstractCircuit, qubits: list[cirq.Qid]) -> np.ndarray:
    assert_simulable(len(qubits))
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubits)
    return np.asarray(result.final_state_vector, dtype=np.complex128)


def dicke_fidelity(vector: np.ndarray, ell: int) -> float:
    """Squared overlap with ``|D_ell^n>``. Global phase is ignored."""
    n = int(math.log2(vector.shape[0]))
    if 1 << n != vector.shape[0]:
        raise ValueError("statevector length is not a power of two")
    ideal = np.zeros(vector.shape[0], dtype=np.complex128)
    if ell < 0 or ell > n:
        raise ValueError(f"ell={ell} is outside 0..{n}")
    if math.comb(n, ell) == 0:
        return 0.0
    amplitude = 1.0 / math.sqrt(math.comb(n, ell))
    for mask in range(vector.shape[0]):
        if mask.bit_count() == ell:
            ideal[mask] = amplitude
    overlap = np.vdot(ideal, vector)
    leaked = 0.0
    for mask, amp in enumerate(vector):
        if mask.bit_count() != ell:
            leaked += float(abs(amp) ** 2)
    fidelity = float(abs(overlap) ** 2)
    if leaked > 1e-8 and fidelity < 1.0 - AMPLITUDE_ATOL:
        return fidelity
    return fidelity


def relative_phase_spread(vector: np.ndarray, ell: int) -> float:
    """Max distance between normalized weight-``ell`` amplitudes, after removing one phase."""
    reference = None
    spread = 0.0
    for mask, amplitude in enumerate(vector):
        if mask.bit_count() != ell or abs(amplitude) <= AMPLITUDE_ATOL:
            continue
        if reference is None:
            reference = amplitude
            continue
        aligned = amplitude * abs(reference) / (reference if reference != 0 else 1)
        # Compare amplitudes up to the global phase of ``reference``.
        ratio = amplitude / reference
        spread = max(spread, abs(ratio - abs(ratio)))
    return spread
