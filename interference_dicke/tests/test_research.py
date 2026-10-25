import itertools

import cirq
import numpy as np
import pytest
from scipy.linalg import expm

from interference_dicke.core import Sector, schedule_for
from interference_dicke.hardware import circuit_for, compile_validate, willow_path, ideal_fidelity


@pytest.mark.parametrize("n,l", [(4, 1), (4, 2), (6, 3), (8, 4), (12, 6)])
def test_basis(n, l):
    sector = Sector(n, l, [(i, i + 1) for i in range(n - 1)])
    from math import comb
    assert len(sector.masks) == comb(n, l)
    assert all(int(x).bit_count() == l for x in sector.masks)
    assert np.all(np.diff(sector.masks) > 0)


def test_hamiltonian_against_independent_full_space_and_conservation():
    n, l = 4, 2
    sector = Sector(n, l, [(0, 1), (1, 2), (2, 3)])
    exchange = {(0, 1): .7, (1, 2): -.4, (2, 3): .3}
    fields, interactions = [.1, -.2, .3, -.4], {(1, 2): .8}
    full = np.zeros((2**n, 2**n), complex)
    number = np.diag([x.bit_count() for x in range(2**n)])
    # Independent tensor construction in little-endian site order.
    ident, z = np.eye(2), np.diag([1, -1])
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    occupation = (ident - z) / 2
    def tensor(operators):
        out = np.ones((1, 1))
        for i in reversed(range(n)):
            out = np.kron(out, operators.get(i, ident))
        return out
    for (i, j), v in exchange.items():
        full += v * (tensor({i: x, j: x}) + tensor({i: y, j: y})) / 2
    for i, h in enumerate(fields):
        full += h * tensor({i: z})
    full += .8 * tensor({1: occupation, 2: occupation})
    assert np.max(abs(full @ number - number @ full)) < 1e-14
    restricted = sector.hamiltonian(exchange, fields, interactions).toarray()
    assert np.allclose(restricted, full[np.ix_(sector.masks, sector.masks)])
    state = np.zeros(2**n, complex); state[(1 << l) - 1] = 1
    assert np.allclose((expm(-.3j * full) @ state)[sector.masks],
                       expm(-.3j * restricted) @ sector.initial)


@pytest.mark.parametrize("n,l", [(4, 1), (4, 2), (6, 3)])
def test_gates_and_adjoint_gradient_against_cirq_and_finite_difference(n, l):
    sector = Sector(n, l, [(i, i + 1) for i in range(n - 1)])
    schedule = schedule_for(sector, 4, 1)
    angles = np.random.default_rng(9).normal(size=len(schedule))
    state = sector.evolve(schedule, angles)
    qs = cirq.LineQubit.range(n)
    circuit = circuit_for(sector, schedule, angles, qs)
    # Reversed qubit order makes Cirq index i agree with mask bit i.
    full = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qs[::-1]).final_state_vector
    assert abs(np.vdot(full[sector.masks], state)) ** 2 == pytest.approx(1, abs=1e-12)
    assert np.vdot(state, state).real == pytest.approx(1, abs=1e-12)
    loss, gradient = sector.objective(schedule, angles)
    assert loss == pytest.approx(1 - abs(np.vdot(sector.target, state)) ** 2)
    for i in range(len(angles)):
        offset = np.zeros(len(angles)); offset[i] = 1e-6
        difference = (sector.objective(schedule, angles + offset)[0] -
                      sector.objective(schedule, angles - offset)[0]) / 2e-6
        assert gradient[i] == pytest.approx(difference, abs=2e-8)


def test_slater_bound_two_independent_constructions():
    # Antisymmetric coefficient matrix, normalized two-form coefficients 1/sqrt(6).
    matrix = np.triu(np.ones((4, 4)), 1) / np.sqrt(6)
    matrix -= matrix.T
    bound = (3 + 2 * np.sqrt(2)) / 6
    assert np.linalg.svd(matrix, compute_uv=False)[0] ** 2 == pytest.approx(bound)
    # Real skew eigenplane constructs an orthonormal Slater maximizer.
    values, vectors = np.linalg.eigh(1j * matrix)
    v = vectors[:, np.argmax(values)]
    u, w = np.sqrt(2) * v.real, np.sqrt(2) * v.imag
    assert np.allclose(np.array([u, w]) @ np.array([u, w]).T, np.eye(2))
    minors = np.array([u[i] * w[j] - u[j] * w[i] for i, j in itertools.combinations(range(4), 2)])
    assert abs(minors.sum() / np.sqrt(6)) ** 2 == pytest.approx(bound)
    assert minors[0] * minors[5] - minors[1] * minors[4] + minors[2] * minors[3] == pytest.approx(0)


def test_chain_xy_plucker():
    sector = Sector(4, 2, [(0, 1), (1, 2), (2, 3)])
    schedule = schedule_for(sector, 8)
    state = sector.evolve(schedule, np.random.default_rng(51).normal(size=len(schedule)))
    p = lambda i, j: state[sector.index[(1 << i) | (1 << j)]]
    assert abs(p(0, 1) * p(2, 3) - p(0, 2) * p(1, 3) + p(0, 3) * p(1, 2)) < 1e-13


def test_native_compilation_preserves_fidelity_and_topology():
    sector = Sector(4, 2, [(0, 1), (1, 2), (2, 3)])
    schedule = schedule_for(sector, 4, 1)
    angles = np.random.default_rng(4).normal(size=len(schedule))
    qubits = willow_path(4)
    logical = circuit_for(sector, schedule, angles, qubits)
    native = compile_validate(logical, qubits)
    assert ideal_fidelity(native, qubits, 2) == pytest.approx(ideal_fidelity(logical, qubits, 2), abs=1e-10)


def test_noiseless_density_and_damping_acceptance():
    from interference_dicke.hardware import IllustrativeNoise, noisy_metrics
    qs = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(qs[0]))
    zero = IllustrativeNoise(p1=0, p2=0, t1_ns=float("inf"), t2_ns=float("inf"), readout_flip=0)
    result = noisy_metrics(circuit, qs, 1, zero)
    assert result["fidelity"] == pytest.approx(.5)
    assert result["p_valid"] == pytest.approx(1)
    damping = IllustrativeNoise(p1=0, p2=0, t1_ns=100, t2_ns=200, single_ns=25, readout_flip=0)
    result = noisy_metrics(circuit, qs, 1, damping)
    assert result["p_valid"] == pytest.approx(np.exp(-.25))
    assert result["fidelity"] == pytest.approx(.5 * np.exp(-.25))
    assert result["f_post"] == pytest.approx(.5)


def test_state_aware_pruning_preserves_whole_prepared_state():
    from interference_dicke.prune import prune
    qs = list(cirq.LineQubit.range(4))
    sector = Sector(4, 2, [(0, 1), (1, 2), (2, 3)])
    schedule = schedule_for(sector, 6, 1)
    angles = np.random.default_rng(42).normal(size=len(schedule))
    circuit = circuit_for(sector, schedule, angles, qs)
    reduced = prune(circuit, qs)
    simulator = cirq.Simulator(dtype=np.complex128)
    a = simulator.simulate(circuit, qubit_order=qs).final_state_vector
    b = simulator.simulate(reduced, qubit_order=qs).final_state_vector
    assert abs(np.vdot(a, b)) ** 2 == pytest.approx(1, abs=1e-12)
    assert len(list(reduced.all_operations())) < len(list(circuit.all_operations()))


def test_multisector_phase_gradient():
    from interference_dicke.multisector import objective
    edges = [(0, 1), (0, 2), (0, 3)]
    sectors = [Sector(4, l, edges) for l in range(5)]
    schedule = schedule_for(sectors[2], 6, 1)
    angles = np.random.default_rng(8).normal(size=len(schedule))
    loss, gradient = objective(sectors, schedule, angles)
    for i in range(len(angles)):
        step = np.zeros(len(angles)); step[i] = 1e-6
        numerical = (objective(sectors, schedule, angles + step)[0] - objective(sectors, schedule, angles - step)[0]) / 2e-6
        assert gradient[i] == pytest.approx(numerical, abs=2e-8)
