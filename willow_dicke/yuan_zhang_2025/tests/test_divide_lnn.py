"""Equivalence of the path Divide against the verified logical block."""

from __future__ import annotations

import cirq
import numpy as np
import pytest

from willow_dicke.divide.logical_divide import build_logical_divide
from willow_dicke.yuan_zhang_2025.divide_lnn import compile_divide_lnn


def _prepare(qubits: list[cirq.Qid], ell: int) -> cirq.Circuit:
    k = len(qubits) // 2
    circuit = cirq.Circuit()
    if ell:
        circuit.append(cirq.X(qubits[2 * k - offset]) for offset in range(1, ell + 1))
    return circuit


def _vector(circuit: cirq.Circuit, qubits: list[cirq.Qid]) -> np.ndarray:
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubits)
    return np.asarray(result.final_state_vector, dtype=np.complex128)


def _fidelity(left: np.ndarray, right: np.ndarray) -> float:
    return float(abs(np.vdot(left, right)) ** 2)


def test_adjacent_controlled_ry_matches_cirq():
    qubits = cirq.LineQubit.range(2)
    theta = 0.7
    manual = cirq.Circuit(
        cirq.ry(theta / 2).on(qubits[1]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.ry(-theta / 2).on(qubits[1]),
        cirq.CNOT(qubits[0], qubits[1]),
    )
    native = cirq.Circuit(cirq.ry(theta).controlled().on(qubits[0], qubits[1]))
    assert _fidelity(_vector(manual, qubits), _vector(native, qubits)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("n", "m", "k"),
    [
        (4, 2, 1),
        (8, 4, 2),
        (8, 3, 2),
        (12, 6, 3),
        (12, 5, 3),
    ],
)
def test_lnn_divide_matches_reference_on_every_weight(n, m, k):
    qubits = list(cirq.LineQubit.range(2 * k))
    reference = build_logical_divide(n, m, k, qubits)
    optimized = compile_divide_lnn(n, m, k, qubits)
    for ell in range(k + 1):
        left = _vector(_prepare(qubits, ell) + reference, qubits)
        right = _vector(_prepare(qubits, ell) + optimized, qubits)
        fidelity = _fidelity(left, right)
        assert fidelity == pytest.approx(1.0, abs=1e-10), ell


@pytest.mark.parametrize(("n", "m", "k"), [(8, 4, 2), (12, 6, 3)])
def test_lnn_divide_preserves_relative_phases(n, m, k):
    qubits = list(cirq.LineQubit.range(2 * k))
    reference = build_logical_divide(n, m, k, qubits)
    optimized = compile_divide_lnn(n, m, k, qubits)
    prep = cirq.Circuit()
    # (|ell=1> + |ell=2>) / sqrt(2), both unary and ones on the right.
    prep.append([cirq.X(qubits[-1]), cirq.H(qubits[-2])])
    left = _vector(prep + reference, qubits)
    right = _vector(prep + optimized, qubits)
    assert _fidelity(left, right) == pytest.approx(1.0, abs=1e-10)


def test_output_supports_are_unary():
    """The nonzero outputs of a k=2 divide are the unary pairs from Eq. (7)."""
    n, m, k, ell = 8, 3, 2, 2
    qubits = list(cirq.LineQubit.range(2 * k))
    vector = _vector(_prepare(qubits, ell) + compile_divide_lnn(n, m, k, qubits), qubits)
    seen = []
    for basis, amplitude in enumerate(vector):
        if abs(amplitude) > 1e-8:
            seen.append(basis)
    expected = []
    for i in range(0, ell + 1):
        if i > m or ell - i > n - m:
            continue
        # Cirq's statevector index is big-endian: qubit 0 is the high bit.
        bits = [0] * (2 * k)
        for offset in range(i):
            bits[k - 1 - offset] = 1
        for offset in range(ell - i):
            bits[2 * k - 1 - offset] = 1
        mask = 0
        for bit in bits:
            mask = (mask << 1) | bit
        expected.append(mask)
    assert sorted(seen) == sorted(expected)
