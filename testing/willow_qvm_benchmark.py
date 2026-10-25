"""Benchmark Problem 1 candidates on Cirq's calibrated Willow QVM."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import cirq
import cirq_google

from ideation import native_h
from problem1_variants import (
    VARIANTS,
    circuit_calibration_cost,
    validate_problem1_candidate,
)
from vqm_simulator_harness import VQMSimulatorHarness


DEFAULT_CASES = [(4, 2), (5, 2), (6, 3)]


@dataclass(frozen=True)
class QVMRow:
    """One calibrated-QVM evaluation."""

    variant: str
    num_qubits: int
    num_excitations: int
    repeat: int
    moments: int
    cz_count: int
    sq_count: int
    construction_seconds: float
    calibration_cost: float
    subspace_retention: float
    score_z: float
    score_x: float
    judge_score: float


def _metric_rate(value: object) -> float:
    """Extract the point estimate stored in a Cirq calibration metric."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("calibration metric has no values")
        return float(value[0])
    return float(value)


def _single_qubit_map(calibration: object, metric_name: str) -> dict:
    return {
        key[0] if isinstance(key, tuple) else key: (_metric_rate(value), 0.0)
        for key, value in calibration[metric_name].items()
    }


def load_willow_qvm(
    processor_id: str = "willow_pink",
    simulator_backend: str = "density_matrix",
) -> tuple[cirq.Sampler, cirq.Device, dict, dict, dict, dict]:
    """Create the QVM and convert its calibration to the challenge's maps."""
    simulator_class = (
        cirq.DensityMatrixSimulator
        if simulator_backend == "density_matrix"
        else cirq.Simulator
    )
    engine = cirq_google.engine.create_default_noisy_quantum_virtual_machine(
        processor_id,
        simulator_class=simulator_class,
    )
    processor = engine.get_processor(processor_id)
    calibration = processor.get_current_calibration()

    cz_errors = {
        tuple(key): (_metric_rate(value), 0.0)
        for key, value in calibration[
            "two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle"
        ].items()
    }
    sq_errors = _single_qubit_map(
        calibration,
        "single_qubit_rb_pauli_error_per_gate",
    )
    zero_errors = _single_qubit_map(
        calibration,
        "single_qubit_p00_error",
    )
    one_errors = _single_qubit_map(
        calibration,
        "single_qubit_p11_error",
    )
    return (
        engine.get_sampler(processor_id),
        processor.get_device(),
        sq_errors,
        cz_errors,
        zero_errors,
        one_errors,
    )


def evaluate_variant(
    *,
    sampler: cirq.Sampler,
    device: cirq.Device,
    variant: str,
    num_qubits: int,
    num_excitations: int,
    repeat: int,
    shots: int,
    sq_errors: dict,
    cz_errors: dict,
    zero_errors: dict,
    one_errors: dict,
) -> QVMRow:
    """Build, validate, and score one candidate on the calibrated QVM."""
    started = time.monotonic()
    circuit, outputs = VARIANTS[variant](
        num_qubits,
        num_excitations,
        cz_errors,
        sq_errors,
        zero_errors,
        one_errors,
    )
    construction_seconds = time.monotonic() - started
    outputs = set(outputs)
    violations = validate_problem1_candidate(
        circuit,
        outputs,
        num_qubits,
        num_excitations,
        construction_seconds,
        cz_errors,
        sq_errors,
        verify_fidelity=True,
    )
    if violations:
        raise ValueError("; ".join(violations))
    device.validate_circuit(circuit)

    ordered_outputs = sorted(outputs)
    z_circuit = circuit.copy()
    z_circuit.append(cirq.measure(*ordered_outputs, key="z"))
    x_circuit = circuit.copy()
    x_circuit.append(native_h(qubit) for qubit in ordered_outputs)
    x_circuit.append(cirq.measure(*ordered_outputs, key="x"))
    device.validate_circuit(z_circuit)
    device.validate_circuit(x_circuit)

    z_samples = sampler.run(
        z_circuit,
        repetitions=shots,
    ).measurements["z"]
    x_samples = sampler.run(
        x_circuit,
        repetitions=shots,
    ).measurements["x"]
    score_z = VQMSimulatorHarness.score_z_distribution(
        z_samples,
        num_qubits,
        num_excitations,
    )
    score_x = VQMSimulatorHarness.score_x_distribution(
        x_samples,
        num_excitations,
        max_correlator_size=num_qubits,
    )
    penalty = score_z + score_x
    judge_score = 0.0 if penalty == float("inf") else 1.0 / penalty

    return QVMRow(
        variant=variant,
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        repeat=repeat,
        moments=len(circuit),
        cz_count=sum(op.gate == cirq.CZ for op in circuit.all_operations()),
        sq_count=sum(
            isinstance(op.gate, cirq.PhasedXZGate)
            for op in circuit.all_operations()
        ),
        construction_seconds=construction_seconds,
        calibration_cost=circuit_calibration_cost(
            circuit,
            outputs,
            num_excitations,
            cz_errors,
            sq_errors,
            zero_errors,
            one_errors,
        ),
        subspace_retention=float(
            (z_samples.sum(axis=1) == num_excitations).mean()
        ),
        score_z=score_z,
        score_x=score_x,
        judge_score=judge_score,
    )


def _parse_cases(value: str) -> list[tuple[int, int]]:
    if value == "default":
        return DEFAULT_CASES
    if value == "stratified":
        return [(2, 1), (4, 1), (4, 2), (6, 1), (6, 3), (8, 1), (8, 4)]
    return [
        tuple(int(part) for part in item.split(":"))
        for item in value.split(",")
        if item
    ]


def print_report(rows: list[QVMRow]) -> None:
    """Print run-level metrics and stable per-case aggregates."""
    print(
        f"{'case':>8} {'variant':<22} {'run':>3} {'depth':>5} "
        f"{'CZ':>5} {'1Q':>5} {'retain':>8} {'Z':>9} {'X':>9} "
        f"{'score':>10}"
    )
    print("-" * 96)
    for row in rows:
        print(
            f"{f'({row.num_qubits},{row.num_excitations})':>8} "
            f"{row.variant:<22} {row.repeat:3d} {row.moments:5d} "
            f"{row.cz_count:5d} {row.sq_count:5d} "
            f"{100 * row.subspace_retention:7.2f}% "
            f"{row.score_z:9.4f} {row.score_x:9.4f} "
            f"{row.judge_score:10.6f}"
        )

    print("\nMean calibrated-QVM score by case")
    print("-" * 66)
    cases = sorted({(row.num_qubits, row.num_excitations) for row in rows})
    variants = sorted({row.variant for row in rows})
    totals = {variant: 0.0 for variant in variants}
    for case in cases:
        for variant in variants:
            scores = [
                row.judge_score
                for row in rows
                if row.variant == variant
                and (row.num_qubits, row.num_excitations) == case
            ]
            if not scores:
                continue
            mean = statistics.mean(scores)
            median = statistics.median(scores)
            deviation = statistics.stdev(scores) if len(scores) > 1 else 0.0
            case_rows = [
                row
                for row in rows
                if row.variant == variant
                and (row.num_qubits, row.num_excitations) == case
            ]
            mean_penalty = statistics.mean(
                row.score_z + row.score_x for row in case_rows
            )
            mean_retention = statistics.mean(
                row.subspace_retention for row in case_rows
            )
            totals[variant] += mean
            print(
                f"{str(case):>8} {variant:<22} "
                f"mean={mean:10.6f} median={median:10.6f} "
                f"std={deviation:9.6f} penalty={mean_penalty:9.4f} "
                f"retain={100 * mean_retention:6.2f}%"
            )
    print("\nSummed means over the cases above")
    for variant, total in sorted(totals.items()):
        print(f"{variant:<22} {total:12.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor-id", default="willow_pink")
    parser.add_argument(
        "--backend",
        choices=("density_matrix", "trajectory"),
        default="density_matrix",
    )
    parser.add_argument("--cases", default="default")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--shots", type=int, default=2_000)
    parser.add_argument("--repeats", type=int, default=1)
    arguments = parser.parse_args()

    variants = [item for item in arguments.variants.split(",") if item]
    unknown = set(variants).difference(VARIANTS)
    if unknown:
        parser.error(f"unknown variants: {sorted(unknown)}")
    if arguments.shots <= 0 or arguments.repeats <= 0:
        parser.error("shots and repeats must be positive")

    (
        sampler,
        device,
        sq_errors,
        cz_errors,
        zero_errors,
        one_errors,
    ) = load_willow_qvm(arguments.processor_id, arguments.backend)
    print(
        f"Loaded {arguments.processor_id}: {len(sq_errors)} qubits, "
        f"{len(cz_errors)} calibrated CZ couplers, "
        f"backend={arguments.backend}"
    )

    rows = []
    failures = []
    for num_qubits, num_excitations in _parse_cases(arguments.cases):
        for variant in variants:
            for repeat in range(1, arguments.repeats + 1):
                print(
                    f"Running ({num_qubits},{num_excitations}) "
                    f"{variant} repeat {repeat}/{arguments.repeats}",
                    flush=True,
                )
                try:
                    row = evaluate_variant(
                        sampler=sampler,
                        device=device,
                        variant=variant,
                        num_qubits=num_qubits,
                        num_excitations=num_excitations,
                        repeat=repeat,
                        shots=arguments.shots,
                        sq_errors=sq_errors,
                        cz_errors=cz_errors,
                        zero_errors=zero_errors,
                        one_errors=one_errors,
                    )
                except Exception as error:
                    failures.append(
                        (
                            num_qubits,
                            num_excitations,
                            variant,
                            repeat,
                            type(error).__name__,
                            str(error),
                        )
                    )
                    print(
                        f"FAIL ({num_qubits},{num_excitations}) {variant} "
                        f"repeat {repeat}: {type(error).__name__}: {error}",
                        flush=True,
                    )
                    continue
                rows.append(row)
    print_report(rows)
    if failures:
        raise SystemExit(f"{len(failures)} calibrated-QVM runs failed")


if __name__ == "__main__":
    main()
