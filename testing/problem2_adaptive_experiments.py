"""Bounded measurement-assisted Dicke-state experiments for Problem 2.

The magic-state cultivation paper motivates measure--uncompute and
feed-forward, but its postselected QEC protocol is not a depth optimization
for this challenge.  This module tests the smallest deterministic analogue:
prepare a Bell/GHZ resource through an ancilla, measure the ancilla in X, and
correct every branch into D(2, 1).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cirq
import cirq_google
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dicke_submission


def _native_h(qubit: cirq.Qid) -> cirq.Operation:
    return cirq.PhasedXZGate(
        x_exponent=0.5,
        z_exponent=1.0,
        axis_phase_exponent=1.5,
    ).on(qubit)


def _native_x(qubit: cirq.Qid) -> cirq.Operation:
    return cirq.PhasedXZGate(
        x_exponent=1.0,
        z_exponent=0.0,
        axis_phase_exponent=0.0,
    ).on(qubit)


def _native_z(qubit: cirq.Qid) -> cirq.Operation:
    return cirq.PhasedXZGate(
        x_exponent=0.0,
        z_exponent=1.0,
        axis_phase_exponent=0.0,
    ).on(qubit)


def _native_cnot(
    control: cirq.Qid,
    target: cirq.Qid,
) -> list[cirq.Operation]:
    return [_native_h(target), cirq.CZ(control, target), _native_h(target)]


def build_measurement_assisted_dicke_2_1() -> tuple[
    cirq.Circuit,
    set[cirq.GridQubit],
]:
    """Prepare D(2,1) on every measurement branch using one helper qubit."""
    left = cirq.GridQubit(4, 3)
    helper = cirq.GridQubit(4, 4)
    right = cirq.GridQubit(4, 5)

    circuit = cirq.Circuit()
    circuit.append(_native_h(helper))
    circuit.append(_native_cnot(helper, left))
    circuit.append(_native_cnot(helper, right))
    circuit.append(_native_h(helper))
    circuit.append(cirq.measure(helper, key="helper"))
    circuit.append(
        _native_z(left).with_classical_controls("helper")
    )
    circuit.append(_native_x(right))
    return circuit, {left, right}


def _target_density() -> np.ndarray:
    state = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.complex128)
    state /= np.linalg.norm(state)
    return np.outer(state, state.conj())


def _reduced_output_density(
    final_density: np.ndarray,
) -> np.ndarray:
    tensor = final_density.reshape((2, 2, 2, 2, 2, 2))
    # Qubit order is [left, right, helper]. Trace out helper.
    return np.trace(tensor, axis1=2, axis2=5).reshape((4, 4))


@dataclass(frozen=True)
class AdaptiveExperimentResult:
    """Branch validation and depth comparison for one prototype."""

    observed_branches: tuple[int, ...]
    minimum_branch_fidelity: float
    adaptive_depth: int
    baseline_depth: int
    readout_line_valid: bool

    @property
    def is_candidate(self) -> bool:
        return (
            self.observed_branches == (0, 1)
            and self.minimum_branch_fidelity >= 1.0 - 1e-8
            and self.adaptive_depth < self.baseline_depth
            and self.readout_line_valid
        )


def evaluate_experiment(max_trials: int = 128) -> AdaptiveExperimentResult:
    """Observe both branches, verify outputs, and compare total moments."""
    circuit, outputs = build_measurement_assisted_dicke_2_1()
    device_qubits = cirq_google.Willow105.metadata.qubit_set
    device_pairs = {
        frozenset(pair)
        for pair in cirq_google.Willow105.metadata.qubit_pairs
    }
    if not circuit.all_qubits().issubset(device_qubits):
        raise ValueError("prototype leaves the Willow device")
    for operation in circuit.all_operations():
        if operation.gate == cirq.CZ:
            if frozenset(operation.qubits) not in device_pairs:
                raise ValueError(f"invalid Willow CZ edge: {operation.qubits}")
    ordered_outputs = sorted(outputs)
    helper = next(iter(circuit.all_qubits().difference(outputs)))
    qubit_order = ordered_outputs + [helper]
    target = _target_density()
    branch_fidelities: dict[int, float] = {}

    for seed in range(max_trials):
        result = cirq.DensityMatrixSimulator(seed=seed).simulate(
            circuit,
            qubit_order=qubit_order,
        )
        branch = int(result.measurements["helper"][0])
        reduced = _reduced_output_density(result.final_density_matrix)
        branch_fidelities[branch] = float(
            np.real(np.trace(target @ reduced))
        )
        if len(branch_fidelities) == 2:
            break

    baseline, _ = dicke_submission.create_dicke_state_adaptive(2, 1)
    measured_line = helper.row + helper.col
    output_lines = {q.row + q.col for q in outputs}
    return AdaptiveExperimentResult(
        observed_branches=tuple(sorted(branch_fidelities)),
        minimum_branch_fidelity=min(branch_fidelities.values(), default=0.0),
        adaptive_depth=len(circuit),
        baseline_depth=len(baseline),
        readout_line_valid=measured_line not in output_lines,
    )


def main() -> None:
    result = evaluate_experiment()
    print(result)
    print(
        "PROMOTE" if result.is_candidate else
        "REJECT: exact feed-forward does not reduce total depth"
    )


if __name__ == "__main__":
    main()
