"""Lowering, device checks, and Divide equivalence."""

from __future__ import annotations

import cirq
import cirq_google
import numpy as np

from willow_dicke.circuit.lower import (
    assert_cz_and_single_qubit,
    is_full_cz,
    lower_to_willow_cz,
)
from willow_dicke.divide.logical_divide import build_logical_divide
from willow_dicke.rectangle import adjacent
from willow_dicke.yuan_zhang_2025.config import AMPLITUDE_ATOL
from willow_dicke.yuan_zhang_2025.divide_lnn import compile_divide_lnn
from willow_dicke.yuan_zhang_2025.simulator import simulate


def lower_willow(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """PhasedXZ and full CZ. Does not add SWAPs of its own."""
    return lower_to_willow_cz(circuit)


def assert_patch_cz(circuit: cirq.AbstractCircuit, patch_qubits: set, edge_ok) -> None:
    assert_cz_and_single_qubit(circuit)
    for op in circuit.all_operations():
        if not is_full_cz(op):
            continue
        left, right = op.qubits
        if left not in patch_qubits or right not in patch_qubits or not edge_ok(left, right):
            raise AssertionError(f"CZ {left} -- {right} is outside the patch")


def assert_grid_cz(circuit: cirq.AbstractCircuit) -> None:
    assert_cz_and_single_qubit(circuit)
    for op in circuit.all_operations():
        if is_full_cz(op) and not adjacent(op.qubits[0], op.qubits[1]):
            raise AssertionError(f"CZ {op.qubits[0]} -- {op.qubits[1]} is not a grid edge")


def validate_willow_device(circuit: cirq.AbstractCircuit) -> None:
    cirq_google.Willow105.validate_circuit(circuit)


def verify_divide_equivalence(n: int, m: int, k: int, *, check_superposition: bool = True) -> dict:
    """Compare the LNN Divide to the verified logical block on every weight."""
    qubits = list(cirq.LineQubit.range(2 * k))
    reference = build_logical_divide(n, m, k, qubits)
    optimized = compile_divide_lnn(n, m, k, qubits)
    fidelities = []
    for ell in range(k + 1):
        prep = cirq.Circuit(cirq.X(qubits[2 * k - offset]) for offset in range(1, ell + 1))
        left = simulate(prep + reference, qubits)
        right = simulate(prep + optimized, qubits)
        fidelities.append(float(abs(np.vdot(left, right)) ** 2))
    superposition = None
    if check_superposition and k >= 2:
        prep = cirq.Circuit([cirq.X(qubits[-1]), cirq.H(qubits[-2])])
        left = simulate(prep + reference, qubits)
        right = simulate(prep + optimized, qubits)
        superposition = float(abs(np.vdot(left, right)) ** 2)
    passed = all(value >= 1.0 - AMPLITUDE_ATOL for value in fidelities) and (
        superposition is None or superposition >= 1.0 - AMPLITUDE_ATOL
    )
    return {
        "n": n,
        "m": m,
        "k": k,
        "fidelities": fidelities,
        "superposition_fidelity": superposition,
        "passed": passed,
    }
