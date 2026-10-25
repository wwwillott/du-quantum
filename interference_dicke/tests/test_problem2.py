"""Cross-check the new scorer verification and exact-state admission rules."""
import cirq
import numpy as np
import pytest
from interference_dicke.problem2 import create_dicke_state_adaptive
from interference_dicke.problem2.verify import apply, unitary_fidelity, adaptive_branches, validate_structure
from interference_dicke.problem2.paper import collective_rotation, feedback_statistics


def test_independent_native_simulator_matches_cirq():
    q = list(cirq.LineQubit.range(3))
    c = cirq.Circuit(cirq.PhasedXZGate(x_exponent=.31, z_exponent=.7, axis_phase_exponent=.2)(q[1]),
                     cirq.H(q[2]), cirq.CZ(q[2], q[0]), cirq.X(q[0]),
                     cirq.CZ(q[1], q[2]), cirq.ry(.8)(q[2]))
    state = np.zeros(8, complex); state[0] = 1
    for op in c.all_operations():
        apply(state, op, q)
    expected = cirq.final_state_vector(c, qubit_order=q, dtype=np.complex128)
    assert np.allclose(state, expected, atol=1e-13)


@pytest.mark.parametrize('n,k', [(4, 2), (6, 3), (16, 8)])
def test_scored_circuits_exact_and_legal(n, k):
    c, q = create_dicke_state_adaptive(n, k)
    validate_structure(c, q, n)
    assert unitary_fidelity(c, q, k)['fidelity'] > 1-1e-9


def test_every_complement_branch_passes():
    c, q = create_dicke_state_adaptive(7, 3)
    assert c.has_measurements()
    result = adaptive_branches(c, q, 3)
    assert result['branches'] == 2
    assert result['probability_sum'] == pytest.approx(1)
    assert result['fidelity'] > 1-1e-9


def test_partial_diagonal_readout_is_rejected():
    q = cirq.GridQubit(0, 6)
    with pytest.raises(AssertionError):
        validate_structure(cirq.Circuit(cirq.measure(q)), set(), 0)


def test_collective_rotation_and_finite_failure():
    n, angle = 4, .723
    amplitudes = collective_rotation(n, angle)[:, 0]
    q = cirq.LineQubit.range(n)
    full = cirq.final_state_vector(cirq.Circuit(cirq.ry(angle).on_each(*q)), dtype=np.complex128)
    from interference_dicke.problem2.verify import target
    assert np.allclose([np.vdot(target(n, k), full) for k in range(n+1)], amplitudes)
    result = feedback_statistics(8, 4)
    assert 0 < result['failure_after_rounds'][-1] < result['failure_after_rounds'][0]
    assert not result['eligible_for_score']
