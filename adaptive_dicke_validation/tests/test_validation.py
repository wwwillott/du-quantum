import math

import cirq
import numpy as np
import pytest

from adaptive_dicke_validation.audit import verify_all_branches, structure
from adaptive_dicke_validation.paper_prototype import weight_counter, markov


def test_branch_auditor_detects_missing_feedforward_correction():
    a, helper, b = cirq.LineQubit.range(3)
    base = cirq.Circuit(cirq.H(helper), cirq.CNOT(helper, a), cirq.CNOT(helper, b),
                        cirq.H(helper), cirq.measure(helper, key="m"), cirq.X(b))
    invalid = verify_all_branches(base, {a, b}, 1)
    assert invalid["branches"] == 2
    assert invalid["status"] == "invalid_state"
    base.append(cirq.Z(a).with_classical_controls("m"))
    correct = verify_all_branches(base, {a, b}, 1)
    assert correct["status"] == "verified"
    assert correct["minimum_branch_fidelity"] == pytest.approx(1)


@pytest.mark.parametrize("n", [2, 4])
def test_weight_counter_measures_only_total_weight_without_relative_phases(n):
    m = math.ceil(math.log2(n + 1))
    data = list(cirq.LineQubit.range(n))
    ancillas = list(cirq.LineQubit.range(n, n + m))
    circuit = weight_counter(data, ancillas)
    order = data + ancillas[::-1]
    simulator = cirq.Simulator(dtype=np.complex128)
    # Every basis amplitude gets the correct register value and common phase.
    for index in range(2**n):
        state = simulator.simulate(circuit, qubit_order=order, initial_state=index * 2**m).final_state_vector
        expected = np.zeros(2 ** (n + m), complex)
        expected[index * 2**m + index.bit_count()] = 1
        assert np.allclose(state, expected, atol=1e-12)


def test_markov_first_round_agrees_with_binomial_projection():
    result = markov(4, 2)
    assert result["success_by_round"][0] == pytest.approx(6/16)
    assert np.allclose(np.sum(result["transition_matrix"], axis=0), 1)
    assert all(b >= a for a, b in zip(result["success_by_round"], result["success_by_round"][1:]))
    assert result["success_by_round"][-1] < 1  # no finite exactness by truncation


def test_validator_checks_full_device_diagonal_not_only_output_lines():
    import cirq_google
    helper = cirq.GridQubit(4, 4)
    outputs = {cirq.GridQubit(4, 3), cirq.GridQubit(4, 5)}
    circuit = cirq.Circuit(cirq.Moment(cirq.PhasedXZGate(x_exponent=0, z_exponent=0, axis_phase_exponent=0).on(q) for q in outputs),
                           cirq.Moment(cirq.measure(helper, key="m")))
    assert any("partial readout" in error for error in structure(circuit, outputs, 2))
    line = [q for q in cirq_google.Willow105.metadata.qubit_set if q.row + q.col == 8]
    circuit[-1] = cirq.Moment(cirq.measure(q, key=str(q)) for q in line)
    assert structure(circuit, outputs, 2) == []
