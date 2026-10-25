"""Independent full-vector and adaptive-branch validation, no noisy score proxy."""
from itertools import combinations
import math

import cirq
import cirq_google
import numpy as np


def apply(state, op, order):
    """Native gates on a full vector, independently of Cirq's simulator."""
    indices = [order.index(q) for q in op.qubits]
    n = len(order)
    if len(indices) == 1:
        axis = indices[0]
        view = state.reshape(1 << axis, 2, 1 << (n-axis-1))
        a, b = view[:, 0, :].copy(), view[:, 1, :].copy()
        u = cirq.unitary(op)
        view[:, 0, :] = u[0, 0]*a + u[0, 1]*b
        view[:, 1, :] = u[1, 0]*a + u[1, 1]*b
    elif op.gate == cirq.CZ:
        i, j = sorted(indices)
        view = state.reshape(1 << i, 2, 1 << (j-i-1), 2, 1 << (n-j-1))
        view[:, 1, :, 1, :] *= -1
    else:
        raise ValueError(f'Unexpected verification gate {op}')


def target(n, k):
    v = np.zeros(1 << n, complex)
    for sites in combinations(range(n), k):
        v[sum(1 << (n-1-i) for i in sites)] = 1/math.sqrt(math.comb(n, k))
    return v


def unitary_fidelity(circuit, outputs, k):
    order = sorted(outputs) + sorted(circuit.all_qubits() - set(outputs))
    if len(order) > 22:
        raise ValueError('Full-vector verifier limited to 22 active qubits')
    state = np.zeros(1 << len(order), complex)
    state[0] = 1
    for op in circuit.all_operations():
        apply(state, op, order)
    t = target(len(outputs), k)
    tensor = state.reshape(len(t), -1)
    projections = t.conj() @ tensor
    fidelity = float(np.sum(abs(projections)**2))
    residual = float(np.linalg.norm(tensor - t[:, None]*projections[None, :]))
    return dict(fidelity=fidelity, amplitude_residual=residual,
                norm=float(np.linalg.norm(state)), active_qubits=len(order),
                ancilla_zero_probability=float(np.sum(abs(tensor[:, 0])**2)))


def validate_structure(circuit, outputs, n):
    device = cirq_google.Willow105
    physical = set(device.metadata.qubit_set)
    assert len(outputs) == n and set(outputs) <= physical
    available_keys = set()
    for moment in circuit:
        measured = {q for op in moment if cirq.is_measurement(op) for q in op.qubits}
        for line in {q.row+q.col for q in measured}:
            assert {q for q in physical if q.row+q.col == line} <= measured
        assert not measured.intersection(outputs), 'Measured an output data wire'
        for op in moment:
            assert set(cirq.control_keys(op)) <= available_keys, 'Noncausal feedforward'
            base = op.without_classical_controls()
            if not cirq.is_measurement(base):
                assert base.gate == cirq.CZ or isinstance(base.gate, cirq.PhasedXZGate) or base.gate in (cirq.X, cirq.Z)
            device.validate_operation(base)
        available_keys.update(key for op in moment for key in cirq.measurement_key_objs(op))
    if not circuit.has_measurements():
        device.validate_circuit(circuit)
    return dict(readout_lines_valid=True, causal_controls=True, device_valid=True)


def adaptive_branches(circuit, outputs, k):
    """Enumerate every nonzero branch; idle diagonal filler wires stay zero.

    Used for measurement/complement circuits. Large Clifford cat protocols
    have a separate channel certificate instead of exponential simulation.
    """
    active = set(outputs)
    active.update(q for op in circuit.all_operations() if not cirq.is_measurement(op) for q in op.qubits)
    order = sorted(outputs) + sorted(active - set(outputs))
    if len(order) > 22:
        raise ValueError('Use Clifford channel certificate for large adaptive circuits')
    initial = np.zeros(1 << len(order), complex); initial[0] = 1
    branches = [(initial, {})]
    for op in circuit.all_operations():
        if cirq.is_measurement(op):
            assert len(op.qubits) == 1
            q = op.qubits[0]; key = cirq.measurement_key_obj(op)
            invert = bool(op.gate.invert_mask and op.gate.invert_mask[0])
            next_branches = []
            for state, record in branches:
                if q not in active:
                    next_branches.append((state, dict(record, **{str(key): int(invert)})))
                    continue
                axis = order.index(q)
                for bit in (0, 1):
                    projected = state.copy().reshape(1 << axis, 2, -1)
                    projected[:, 1-bit, :] = 0
                    projected = projected.reshape(-1)
                    if np.vdot(projected, projected).real > 1e-24:
                        next_branches.append((projected, dict(record, **{str(key): bit ^ invert})))
            branches = next_branches
        else:
            for state, record in branches:
                data = cirq.ClassicalDataDictionaryStore()
                for key, value in record.items():
                    data.record_measurement(cirq.MeasurementKey(key), [value], [cirq.LineQubit(0)])
                if all(condition.resolve(data) for condition in op.classical_controls):
                    apply(state, op.without_classical_controls(), order)
    t = target(len(outputs), k)
    probabilities, fidelities = [], []
    for state, _ in branches:
        prob = float(np.vdot(state, state).real)
        overlap = t.conj() @ state.reshape(len(t), -1)
        fidelities.append(float(np.sum(abs(overlap)**2) / prob))
        probabilities.append(prob)
    return dict(fidelity=min(fidelities), branches=len(branches),
                probabilities=probabilities, probability_sum=sum(probabilities), active_qubits=len(order))
