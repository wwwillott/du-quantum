"""Tests for the Dicke mathematics, the divide unitary, routing, and Willow legality."""

from __future__ import annotations

import math

import cirq
import cirq_google
import numpy as np
import pytest

from dicke_willow.challenge import create_dicke_state, create_dicke_state_adaptive
from dicke_willow.dicke_math import amplitude_norm, binomial, split_amplitudes, split_weights
from dicke_willow.divide import prepare_balanced
from dicke_willow.routing import apply_swap_layers, route_permutation
from dicke_willow.scs import prepare_dicke_circuit
from dicke_willow.topology import graph_from_pairs, willow105_graph
from dicke_willow.verify import fidelity
from dicke_willow.weight_distribution import ideal_divided_state, weight_distribution_ops


def test_vandermonde_and_normalization():
    cases = [(4, 2, 2), (6, 3, 3), (8, 3, 2), (9, 4, 3), (12, 5, 4)]
    for n, m, ell in cases:
        weights = split_weights(n, m, ell)
        assert sum(weight for _, weight in weights) == binomial(n, ell)
        assert amplitude_norm(n, m, ell) == pytest.approx(1.0)
        amplitudes = split_amplitudes(n, m, ell)
        assert all(amp >= -1e-15 for _, amp in amplitudes)


@pytest.mark.parametrize("n,k", [(4, 1), (4, 2), (6, 2), (6, 3), (8, 2), (5, 1), (7, 3)])
def test_scs_path_fidelity(n, k):
    qubits = list(cirq.LineQubit.range(n))
    circuit = prepare_dicke_circuit(qubits, k)
    assert fidelity(circuit, k, qubits) == pytest.approx(1.0, abs=1e-5)


def _unary_index(n: int, ell: int) -> int:
    """Big-endian index of |0^{n-ell} 1^{ell}>."""
    return (1 << ell) - 1


def test_weight_distribution_small_splits():
    specs = [(4, 2, 2), (6, 3, 3), (6, 3, 2), (5, 2, 2)]
    for n, m, k in specs:
        for ell in range(0, k + 1):
            b = min(m, k)
            a = min(n - m, k)
            left = list(cirq.LineQubit.range(m))
            right = list(cirq.LineQubit.range(m, n))
            reg_b = left[m - b :]
            reg_a = right[(n - m) - a :]
            ops = weight_distribution_ops(reg_b, reg_a, n=n, m=m, k=k)
            circuit = cirq.Circuit(ops)
            qubits = left + right
            # Unary input sits on the right end of the full register.
            vector = cirq.Simulator().simulate(
                circuit, qubit_order=qubits, initial_state=_unary_index(n, ell)
            ).final_state_vector
            expected = ideal_divided_state(n, m, ell)
            for i, j in expected:
                # left child |0^{m-i} 1^i>, right child |0^{n-m-j} 1^j>
                bits = [0] * n
                for t in range(i):
                    bits[m - 1 - t] = 1
                for t in range(j):
                    bits[n - 1 - t] = 1
                index = 0
                for bit in bits:
                    index = (index << 1) | bit
                assert vector[index] == pytest.approx(expected[(i, j)], abs=1e-6)
            occupied = 0.0
            for i, j in expected:
                bits = [0] * n
                for t in range(i):
                    bits[m - 1 - t] = 1
                for t in range(j):
                    bits[n - 1 - t] = 1
                index = 0
                for bit in bits:
                    index = (index << 1) | bit
                occupied += abs(vector[index]) ** 2
            assert occupied == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("n,k", [(4, 1), (4, 2), (6, 2), (6, 3)])
def test_balanced_preparation(n, k):
    qubits = list(cirq.LineQubit.range(n))
    circuit, region = prepare_balanced(qubits, k)
    assert fidelity(circuit, k, qubits) == pytest.approx(1.0, abs=1e-6)
    assert region.n == n


def test_route_permutation_on_a_path():
    qubits = list(cirq.LineQubit.range(6))
    adjacency = graph_from_pairs((qubits[i], qubits[i + 1]) for i in range(5))
    labels = list("abcdef")
    initial = dict(zip(labels, qubits))
    # Reverse the path.
    target = dict(zip(labels, reversed(qubits)))
    layers = route_permutation(adjacency, initial, target, set(qubits))
    logical = dict(initial)
    physical = {vertex: token for token, vertex in logical.items()}
    for layer in layers:
        used = set()
        for a, b in layer:
            assert a not in used and b not in used
            used.add(a)
            used.add(b)
            assert b in adjacency[a]
    apply_swap_layers(layers, logical, physical)
    assert logical == target


def test_route_matching_on_a_grid():
    grid = [cirq.GridQubit(r, c) for r in range(3) for c in range(3)]
    pairs = []
    for q in grid:
        for delta in ((0, 1), (1, 0)):
            neighbor = cirq.GridQubit(q.row + delta[0], q.col + delta[1])
            if neighbor in grid:
                pairs.append((q, neighbor))
    adjacency = graph_from_pairs(pairs, set(grid))
    labels = list(range(9))
    initial = dict(zip(labels, grid))
    # Swap two adjacent corners' tokens and leave the rest.
    target_qubits = list(grid)
    target_qubits[0], target_qubits[1] = target_qubits[1], target_qubits[0]
    target = dict(zip(labels, target_qubits))
    layers = route_permutation(adjacency, initial, target, set(grid))
    logical = dict(initial)
    physical = {vertex: token for token, vertex in logical.items()}
    apply_swap_layers(layers, logical, physical)
    assert logical == target
    for layer in layers:
        seen = set()
        for a, b in layer:
            assert a not in seen and b not in seen
            seen.add(a)
            seen.add(b)


def _fake_errors(dropout_rate=0.0, seed=0):
    rng = np.random.default_rng(seed)
    device = cirq_google.Willow105
    qubits = list(device.metadata.qubit_set)
    dropouts = {q for q in qubits if rng.random() < dropout_rate}
    alive = [q for q in qubits if q not in dropouts]

    def sq(median):
        rates = rng.lognormal(np.log(median), 0.5, len(alive))
        return {q: (float(rate), 1e-4) for q, rate in zip(alive, rates)}

    pairs = []
    for pair in device.metadata.qubit_pairs:
        a, b = list(pair)
        if a in dropouts or b in dropouts:
            continue
        if rng.random() < dropout_rate:
            continue
        pairs.append((a, b))
    cz_rates = rng.lognormal(np.log(2.5e-3), 0.5, len(pairs))
    cz = {(a, b): (float(rate), 5e-4) for (a, b), rate in zip(pairs, cz_rates)}
    return cz, sq(4e-4), sq(5e-3), sq(1e-2)


@pytest.mark.parametrize("n,k", [(4, 1), (4, 2), (6, 2), (8, 3)])
def test_challenge_circuit_matches_notebook_rules(n, k):
    cz, sq, zero, one = _fake_errors(dropout_rate=0.02, seed=n * 10 + k)
    circuit, qubits = create_dicke_state(n, k, cz, sq, zero, one)
    assert len(qubits) == n
    assert len(circuit) <= 1750
    ordered = sorted(qubits)
    # The notebook compares against its little-endian Dicke vector after sorting
    # the GridQubits. The prepared state is Dicke in path order, which is not
    # the sorted-coordinate order. Reorder the check to the path order used
    # by the circuit: every computational-basis sample must have weight k and
    # the statevector fidelity in the circuit's own qubit order is 1.
    # Dicke states are symmetric, so the notebook's sorted-qubit order matches.
    assert fidelity(circuit, k, ordered) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("n,k", [(4, 2), (6, 3), (8, 2)])
def test_adaptive_validates_on_willow105(n, k):
    circuit, qubits = create_dicke_state_adaptive(n, k)
    cirq_google.Willow105.validate_circuit(circuit)
    assert qubits <= cirq_google.Willow105.metadata.qubit_set
    assert fidelity(circuit, k, sorted(qubits)) == pytest.approx(1.0, abs=1e-5)


def test_willow_graph_has_no_invented_edges():
    adjacency = willow105_graph()
    device_pairs = {frozenset(pair) for pair in cirq_google.Willow105.metadata.qubit_pairs}
    for qubit, neighbors in adjacency.items():
        assert qubit in cirq_google.Willow105.metadata.qubit_set
        for neighbor in neighbors:
            assert frozenset((qubit, neighbor)) in device_pairs
