"""Theorem 14 on a rectangle, checked with Cirq's simulator."""

from __future__ import annotations

import math

import cirq
import numpy as np
import pytest

from willow_dicke.circuit.lower import is_full_cz
from willow_dicke.device import load_static_willow
from willow_dicke.rectangle import N1, N2, ROW0, COL0, abstract_rectangle, adjacent, willow_7x8_rectangle
from willow_dicke.yuan_zhang import (
    coherent_preparation,
    logical_circuit,
    lowered_circuit,
    verify_willow,
    with_excitation,
)


def test_willow_patch_is_the_solid_7_by_8():
    willow = load_static_willow()
    rect = willow_7x8_rectangle(willow)
    assert (rect.height, rect.width, rect.n) == (N1, N2, N1 * N2)
    coords = {(qubit.row, qubit.col) for qubit in rect.qubits()}
    expected = {(row, col) for row in range(ROW0, ROW0 + N2) for col in range(COL0, COL0 + N1)}
    assert coords == expected
    assert len(rect.edges()) == N1 * (N2 - 1) + N2 * (N1 - 1)


def _fidelity(circuit: cirq.Circuit, qubits: list[cirq.Qid], ell: int) -> float:
    vector = np.asarray(
        cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubits).final_state_vector,
        dtype=np.complex128,
    )
    ideal = np.zeros(2 ** len(qubits), dtype=np.complex128)
    if 0 <= ell <= len(qubits):
        amplitude = 1.0 / math.sqrt(math.comb(len(qubits), ell))
        for mask in range(2 ** len(qubits)):
            if mask.bit_count() == ell:
                ideal[mask] = amplitude
    overlap = np.vdot(ideal, vector)
    return float(abs(overlap) ** 2)


@pytest.mark.parametrize("n1,n2,k", [(2, 4, 1), (2, 4, 2), (4, 4, 2)])
def test_dense_simulator_matches_dicke_on_small_grids(n1, n2, k):
    rect = abstract_rectangle(n1, n2)
    body, port = coherent_preparation(rect, k)
    qubits = list(rect.qubits())
    assert len(port) == k
    for ell in range(k + 1):
        trace = with_excitation(body, port, ell)
        logical = logical_circuit(trace)
        assert _fidelity(logical, qubits, ell) == pytest.approx(1.0, abs=1e-8)
        lowered = lowered_circuit(trace)
        for op in lowered.all_operations():
            if is_full_cz(op):
                assert adjacent(op.qubits[0], op.qubits[1])
        assert _fidelity(lowered, qubits, ell) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("k", [1, 2])
def test_block_simulator_on_the_willow_patch(k):
    report = verify_willow(k)
    assert report["cz_on_patch"] is True
    assert report["passed"], report
