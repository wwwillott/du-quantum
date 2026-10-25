"""Public API and end-to-end compiler tests."""

import cirq
import pytest

from dicke_compiler import (
    CalibrationData,
    DickeTarget,
    HardwareTarget,
    Objective,
    VerificationStatus,
    compile_dicke,
    optimize_dicke,
    willow_target,
)


@pytest.mark.parametrize(
    ("num_qubits", "num_excitations"),
    [
        (1, 0),
        (2, 1),
        (4, 2),
        (6, 2),
        (6, 3),
        (8, 4),
    ],
)
def test_compile_dicke_exact(
    num_qubits: int,
    num_excitations: int,
) -> None:
    result = compile_dicke(num_qubits, num_excitations)
    assert result.verification.status is VerificationStatus.EXACT
    assert result.verification.fidelity == pytest.approx(1.0, abs=1e-8)
    assert len(result.output_qubits) == num_qubits
    assert result.after_metrics.moments <= result.before_metrics.moments


def test_optimize_existing_dicke_preparation() -> None:
    initial = compile_dicke(4, 2)
    qubits = sorted(initial.output_qubits)
    padded = initial.circuit.copy()
    padded.append(cirq.CZ(qubits[0], qubits[1]))
    padded.append(cirq.CZ(qubits[0], qubits[1]))

    result = optimize_dicke(
        padded,
        DickeTarget(4, 2),
        output_qubits=qubits,
    )

    assert result.verification.passed
    assert result.after_metrics.cz_count < sum(
        operation.gate == cirq.CZ
        for operation in padded.all_operations()
    )


def test_projected_survival_selects_current_calibration() -> None:
    qubits = tuple(cirq.LineQubit.range(4))
    couplers = ((qubits[0], qubits[1]), (qubits[2], qubits[3]))
    calibration = CalibrationData(
        single_qubit={qubit: 0.001 for qubit in qubits},
        two_qubit={
            couplers[0]: 0.2,
            couplers[1]: 0.001,
        },
        readout_zero={qubit: 0.01 for qubit in qubits},
        readout_one={qubit: 0.01 for qubit in qubits},
    )
    hardware = HardwareTarget.from_edges(
        qubits,
        couplers,
        calibration=calibration,
    )

    result = compile_dicke(
        2,
        1,
        hardware=hardware,
        objective=Objective.PROJECTED_SURVIVAL,
    )

    assert result.output_qubits == frozenset((qubits[2], qubits[3]))
    assert result.projected_survival is not None


def test_square_hardware_uses_registered_shallow_template() -> None:
    qubits = tuple(cirq.GridQubit.rect(2, 2))
    hardware = HardwareTarget.from_edges(
        qubits,
        (
            (qubits[0], qubits[1]),
            (qubits[1], qubits[3]),
            (qubits[3], qubits[2]),
            (qubits[2], qubits[0]),
        ),
    )

    result = compile_dicke(4, 2, hardware=hardware)

    assert result.candidate_family == "analytic_square_4_2"
    assert result.after_metrics.moments == 7
    assert result.verification.passed


@pytest.mark.parametrize(
    ("num_excitations", "family"),
    [(1, "w_tree"), (4, "anti_w_tree")],
)
def test_w_tree_supports_non_path_hardware(
    num_excitations: int,
    family: str,
) -> None:
    qubits = tuple(cirq.LineQubit.range(5))
    hardware = HardwareTarget.from_edges(
        qubits,
        ((qubits[0], qubit) for qubit in qubits[1:]),
    )

    result = compile_dicke(5, num_excitations, hardware=hardware)

    assert result.candidate_family == family
    assert result.verification.passed
    assert result.output_qubits == frozenset(qubits)


def test_scs_is_available_as_independent_reference_family() -> None:
    result = compile_dicke(4, 2, methods=("scs_path",))

    assert result.candidate_family == "scs_path"
    assert result.verification.passed


@pytest.mark.parametrize(
    ("num_qubits", "num_excitations", "maximum_depth"),
    [(6, 3, 14), (8, 4, 23)],
)
def test_registered_balanced_templates_on_willow(
    num_qubits: int,
    num_excitations: int,
    maximum_depth: int,
) -> None:
    pytest.importorskip("cirq_google")

    result = compile_dicke(
        num_qubits,
        num_excitations,
        hardware=willow_target(),
        path_limit=64,
    )

    assert result.candidate_family in {
        f"balanced_template_{num_qubits}_{num_excitations}",
        f"best_known_{num_qubits}_{num_excitations}",
    }
    assert result.after_metrics.moments <= maximum_depth
    assert result.verification.passed


def test_invalid_target_is_rejected() -> None:
    with pytest.raises(ValueError):
        compile_dicke(4, 5)


def test_full_symmetric_domain_through_fifteen_qubits() -> None:
    for num_qubits in range(2, 16):
        for num_excitations in range(1, num_qubits // 2 + 1):
            result = compile_dicke(num_qubits, num_excitations)
            assert result.verification.status is VerificationStatus.EXACT
            assert result.verification.fidelity == pytest.approx(
                1.0,
                abs=1e-8,
            )
