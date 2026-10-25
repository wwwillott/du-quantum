"""Quantum half: Divide on the certified paths, leaves, and CZ lowering."""

from __future__ import annotations

import math

import cirq
import networkx as nx
import numpy as np
import pytest

from willow_dicke.circuit.lower import (
    assert_cz_and_single_qubit,
    bridge_along_path,
    expand_circuit,
    is_full_cz,
    lower_to_willow_cz,
)
from willow_dicke.circuit.recursive import build_coherent_unitary, build_root_dicke_circuit
from willow_dicke.divide.logical_divide import build_logical_divide
from willow_dicke.graph import WillowGraph
from willow_dicke.partition.tree import build_partition_tree
from dicke_willow.weight_distribution import ideal_divided_state


def _line(n: int) -> tuple[WillowGraph, list[cirq.GridQubit]]:
    qubits = [cirq.GridQubit(0, index) for index in range(n)]
    graph = nx.Graph()
    graph.add_nodes_from(qubits)
    graph.add_edges_from(zip(qubits, qubits[1:]))
    return WillowGraph("line", graph, frozenset(qubits), static_reference=False), qubits


def _ideal_dicke(n: int, ell: int) -> np.ndarray:
    vector = np.zeros(2**n, dtype=np.complex128)
    if ell < 0 or ell > n:
        return vector
    amplitude = 1.0 / math.sqrt(math.comb(n, ell))
    for mask in range(2**n):
        if mask.bit_count() == ell:
            vector[mask] = amplitude
    return vector


def _fidelity(circuit: cirq.Circuit, qubits: list[cirq.Qid], ell: int) -> float:
    simulator = cirq.Simulator(dtype=np.complex128)
    state = simulator.simulate(circuit, qubit_order=qubits).final_state_vector
    # Cirq's statevector uses the last qubit as the least significant bit,
    # which matches _ideal_dicke.
    overlap = np.vdot(_ideal_dicke(len(qubits), ell), np.asarray(state, dtype=np.complex128))
    return float(abs(overlap) ** 2)


@pytest.mark.parametrize(
    "n,m,k",
    [(4, 2, 1), (4, 2, 2), (5, 2, 2), (6, 3, 2), (6, 3, 3), (7, 3, 2)],
)
def test_logical_divide_every_weight(n, m, k):
    qubits = list(cirq.LineQubit.range(2 * k))
    divide = build_logical_divide(n, m, k, qubits)
    right = qubits[k:]
    for ell in range(k + 1):
        excited = right[-ell:] if ell else ()
        circuit = cirq.Circuit(cirq.X(qubit) for qubit in excited) + divide
        vector = np.asarray(
            cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubits).final_state_vector,
            dtype=np.complex128,
        )
        expected = ideal_divided_state(n, m, ell)
        occupied = 0.0
        for weight_left, weight_right in expected:
            bits = [0] * (2 * k)
            for offset in range(weight_left):
                bits[k - 1 - offset] = 1
            for offset in range(weight_right):
                bits[2 * k - 1 - offset] = 1
            index = 0
            for bit in bits:
                index = (index << 1) | bit
            assert vector[index] == pytest.approx(expected[(weight_left, weight_right)], abs=1e-8)
            occupied += abs(vector[index]) ** 2
        assert occupied == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize(
    "n,m,k",
    [(4, 2, 1), (4, 2, 2), (6, 3, 2), (7, 3, 2)],
)
def test_cz_lowered_divide_matches_logical(n, m, k):
    qubits = list(cirq.LineQubit.range(2 * k))
    logical = build_logical_divide(n, m, k, qubits)
    lowered = lower_to_willow_cz(bridge_along_path(expand_circuit(logical), qubits))
    assert_cz_and_single_qubit(lowered)
    for ell in range(k + 1):
        excited = qubits[k:][-ell:] if ell else ()
        prefix = cirq.Circuit(cirq.X(qubit) for qubit in excited)
        logical_state = cirq.Simulator(dtype=np.complex128).simulate(prefix + logical, qubit_order=qubits).final_state_vector
        lowered_state = cirq.Simulator(dtype=np.complex128).simulate(prefix + lowered, qubit_order=qubits).final_state_vector
        overlap = abs(np.vdot(logical_state, lowered_state)) ** 2
        assert overlap == pytest.approx(1.0, abs=1e-8)


def test_swap_lowers_to_three_cz_layers():
    qubits = list(cirq.LineQubit.range(2))
    logical = cirq.Circuit(cirq.SWAP(*qubits))
    lowered = lower_to_willow_cz(logical)
    assert_cz_and_single_qubit(lowered)
    cz_moments = [moment for moment in lowered if any(is_full_cz(op) for op in moment)]
    assert len(cz_moments) == 3
    assert all(sum(is_full_cz(op) for op in moment) == 1 for moment in cz_moments)
    logical_state = cirq.Simulator(dtype=np.complex128).simulate(
        cirq.Circuit(cirq.X(qubits[0]), cirq.SWAP(*qubits)), qubit_order=qubits
    ).final_state_vector
    lowered_state = cirq.Simulator(dtype=np.complex128).simulate(
        cirq.Circuit(cirq.X(qubits[0])) + lowered, qubit_order=qubits
    ).final_state_vector
    assert abs(np.vdot(logical_state, lowered_state)) ** 2 == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n,k", [(8, 2), (10, 2), (12, 3)])
def test_recursive_line_prepares_every_weight(n, k):
    willow, qubits = _line(n)
    root = build_partition_tree(willow, k, leaf_factor=2)
    assert root.failure is None
    assert not root.is_leaf
    built = build_coherent_unitary(willow, root, k)
    incoming = _right_end_register(built, root, k, willow)
    for ell in range(k + 1):
        excited = incoming[-ell:] if ell else ()
        circuit = cirq.Circuit(cirq.X(qubit) for qubit in excited) + built.circuit
        fidelity = _fidelity(circuit, qubits, ell)
        assert fidelity == pytest.approx(1.0, abs=1e-10), f"n={n} k={k} ell={ell} fidelity={fidelity}"


def test_block_replay_matches_dicke_on_a_line():
    willow, qubits = _line(8)
    root = build_partition_tree(willow, 2, leaf_factor=2)
    built = build_root_dicke_circuit(willow, root, 2)
    from willow_dicke.verify.block_sim import fixed_weight_error, replay

    report = fixed_weight_error(replay(built.trace, qubits), 8, 2)
    assert report["passed"], report


def _right_end_register(built, root, k, willow):
    del built, willow
    if root.is_leaf:
        raise AssertionError("the miniature tree did not recurse")
    return tuple(root.staging_path[k:])
