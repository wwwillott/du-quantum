"""Statevector, subspace, and sampling checks for Dicke states."""

from __future__ import annotations

import itertools
import math

import cirq
import numpy as np


def ideal_dicke_vector(num_qubits: int, num_excitations: int) -> np.ndarray:
    """Dicke state in Cirq's big-endian statevector convention.

    Qubit 0 is the most significant bit, matching
    ``cirq.Simulator().simulate(circuit).final_state_vector`` when the
    circuit was built on ``LineQubit.range`` in that order. The notebook's
    little-endian helper is the bit-reversal of this vector; Dicke states are
    invariant under that reversal, so the fidelity agrees either way.
    """
    psi = np.zeros(2**num_qubits, dtype=complex)
    if num_excitations < 0 or num_excitations > num_qubits:
        raise ValueError("excitation count out of range")
    scale = 1.0 / math.sqrt(math.comb(num_qubits, num_excitations))
    for excited in itertools.combinations(range(num_qubits), num_excitations):
        index = sum(2 ** (num_qubits - 1 - qubit) for qubit in excited)
        psi[index] = scale
    return psi


def notebook_dicke_vector(num_qubits: int, num_excitations: int) -> np.ndarray:
    """The little-endian vector written in ``DuQuantum_Google_Standard.ipynb``."""
    psi = np.zeros(2**num_qubits, dtype=complex)
    scale = 1.0 / math.sqrt(math.comb(num_qubits, num_excitations))
    for excited in itertools.combinations(range(num_qubits), num_excitations):
        index = sum(2**qubit for qubit in excited)
        psi[index] = scale
    return psi


def fidelity(circuit: cirq.AbstractCircuit, k: int, qubit_order: list[cirq.Qid] | None = None) -> float:
    """|<&D_k|psi>|^2. Uses every qubit of the circuit, in ``qubit_order``."""
    qubits = list(qubit_order) if qubit_order is not None else sorted(circuit.all_qubits())
    if len(qubits) > 16:
        raise ValueError("exact fidelity is only evaluated for n <= 16")
    vector = cirq.Simulator().simulate(circuit, qubit_order=qubits).final_state_vector
    # Compare against both endian conventions and keep the better overlap.
    # A correctly prepared Dicke state scores 1 on both, because it is symmetric.
    n = len(qubits)
    big = ideal_dicke_vector(n, k)
    little = notebook_dicke_vector(n, k)
    return float(max(abs(np.vdot(big, vector)) ** 2, abs(np.vdot(little, vector)) ** 2))


def hamming_fraction(bitstrings: np.ndarray, k: int) -> float:
    if len(bitstrings) == 0:
        return 0.0
    weights = np.sum(bitstrings, axis=1)
    return float(np.mean(weights == k))


def total_variation_weight_k(bitstrings: np.ndarray, k: int) -> float:
    """Total variation between the observed weight-k strings and the uniform Dicke distribution.

    Strings of the wrong weight are excluded, so this is a shape diagnostic
    inside the subspace, not a fidelity.
    """
    n = bitstrings.shape[1]
    mask = np.sum(bitstrings, axis=1) == k
    kept = bitstrings[mask]
    if len(kept) == 0:
        return 1.0
    counts: dict[bytes, int] = {}
    for row in kept:
        counts[row.tobytes()] = counts.get(row.tobytes(), 0) + 1
    ideal = 1.0 / math.comb(n, k)
    variation = 0.0
    seen = set()
    for excited in itertools.combinations(range(n), k):
        state = np.zeros(n, dtype=int)
        state[list(excited)] = 1
        key = state.tobytes()
        seen.add(key)
        empirical = counts.get(key, 0) / len(kept)
        variation += abs(empirical - ideal)
    for key, count in counts.items():
        if key not in seen:
            variation += count / len(kept)
    return 0.5 * variation


def marginal_excitation(bitstrings: np.ndarray) -> np.ndarray:
    if len(bitstrings) == 0:
        return np.zeros(bitstrings.shape[1])
    return np.mean(bitstrings, axis=0)


def pair_excitation(bitstrings: np.ndarray, i: int, j: int) -> float:
    if len(bitstrings) == 0:
        return 0.0
    return float(np.mean((bitstrings[:, i] == 1) & (bitstrings[:, j] == 1)))
