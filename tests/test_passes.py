"""Equivalence tests for reusable optimization passes."""

import cirq
import numpy as np

from dicke_compiler.passes import (
    cancel_adjacent_czs,
    circuit_metrics,
    optimize_unitary,
)


def test_cancel_adjacent_cz_across_disjoint_work() -> None:
    first, second, spectator = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.CZ(first, second),
        cirq.X(spectator),
        cirq.CZ(first, second),
    )

    optimized = cancel_adjacent_czs(circuit)

    assert sum(
        operation.gate == cirq.CZ
        for operation in optimized.all_operations()
    ) == 0
    assert any(
        operation.qubits == (spectator,)
        for operation in optimized.all_operations()
    )


def test_default_pipeline_preserves_unitary() -> None:
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.Z(qubits[2]) ** 0.25,
        cirq.CZ(qubits[0], qubits[1]),
        cirq.X(qubits[2]) ** 0.3,
        cirq.CZ(qubits[0], qubits[1]),
        cirq.Z(qubits[2]) ** -0.25,
    )

    optimized, _ = optimize_unitary(circuit)

    cirq.testing.assert_allclose_up_to_global_phase(
        circuit.unitary(qubit_order=qubits),
        optimized.unitary(qubit_order=qubits),
        atol=1e-8,
    )


def test_metrics_report_entangling_depth() -> None:
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.Moment(
            [cirq.CZ(qubits[0], qubits[1]), cirq.CZ(qubits[2], qubits[3])]
        )
    )

    metrics = circuit_metrics(circuit)

    assert metrics.cz_count == 2
    assert metrics.cz_depth == 1
    assert np.isfinite(metrics.operations)
