"""Judge-aligned benchmarks for all Problem 1 circuit variants."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass

import cirq

from problem1_variants import (
    VARIANTS,
    circuit_calibration_cost,
    validate_problem1_candidate,
)
from vqm_simulator_harness import (
    VQMSimulatorHarness,
    generate_seeded_error_maps,
)


STRATIFIED_CASES = [
    (2, 1),
    (4, 1),
    (4, 2),
    (5, 2),
    (6, 1),
    (6, 3),
    (8, 1),
    (8, 4),
]


@dataclass(frozen=True)
class BenchmarkRow:
    """Metrics from one candidate, calibration map, and sampling seed."""

    variant: str
    num_qubits: int
    num_excitations: int
    error_seed: int
    sample_seed: int
    moments: int
    cz_count: int
    sq_count: int
    construction_seconds: float
    calibration_cost: float
    subspace_retention: float | None
    score_z: float | None
    score_x: float | None
    judge_score: float | None


def _parse_integer_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_cases(value: str) -> list[tuple[int, int]]:
    if value == "all":
        return [
            (num_qubits, num_excitations)
            for num_qubits in range(2, 16)
            for num_excitations in range(1, num_qubits // 2 + 1)
        ]
    if value == "stratified":
        return STRATIFIED_CASES
    return [
        tuple(int(part) for part in item.split(":"))
        for item in value.split(",")
        if item
    ]


def benchmark_variant(
    variant: str,
    num_qubits: int,
    num_excitations: int,
    error_seed: int,
    sample_seed: int,
    shots: int,
    structural_only: bool,
    backend: str,
    idle_phase_error: float,
) -> BenchmarkRow:
    """Build, validate, and optionally score one variant."""
    sq_errors, cz_errors, zero_errors, one_errors = (
        generate_seeded_error_maps(error_seed)
    )
    builder = VARIANTS[variant]
    started = time.perf_counter()
    circuit, outputs = builder(
        num_qubits,
        num_excitations,
        cz_errors,
        sq_errors,
        zero_errors,
        one_errors,
    )
    construction_seconds = time.perf_counter() - started
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

    cz_count = sum(
        operation.gate == cirq.CZ
        for operation in circuit.all_operations()
    )
    sq_count = sum(
        isinstance(operation.gate, cirq.PhasedXZGate)
        for operation in circuit.all_operations()
    )
    calibration_cost = circuit_calibration_cost(
        circuit,
        outputs,
        num_excitations,
        cz_errors,
        sq_errors,
        zero_errors,
        one_errors,
    )

    if structural_only:
        return BenchmarkRow(
            variant,
            num_qubits,
            num_excitations,
            error_seed,
            sample_seed,
            len(circuit),
            cz_count,
            sq_count,
            construction_seconds,
            calibration_cost,
            None,
            None,
            None,
            None,
        )

    harness = VQMSimulatorHarness(
        sq_errors,
        cz_errors,
        zero_errors,
        one_errors,
        idle_phase_error=idle_phase_error,
        simulator_backend=backend,
    )

    def cached_builder(**_):
        return circuit, outputs

    result = harness.evaluate_solution(
        cached_builder,
        num_qubits,
        num_excitations,
        shots=shots,
        seed=sample_seed,
    )
    return BenchmarkRow(
        variant,
        num_qubits,
        num_excitations,
        error_seed,
        sample_seed,
        len(circuit),
        cz_count,
        sq_count,
        construction_seconds,
        calibration_cost,
        float(result["subspace_retention"]),
        float(result["score_z"]),
        float(result["score_x"]),
        float(result["overall_score"]),
    )


def print_rows(rows: list[BenchmarkRow]) -> None:
    """Print per-run metrics and aggregate judge-score estimates."""
    print(
        f"{'case':>8} {'variant':<22} {'depth':>6} {'CZ':>5} {'1Q':>5} "
        f"{'build':>8} {'cost':>10} {'retain':>9} {'Z':>9} {'X':>9} "
        f"{'score':>10}"
    )
    print("-" * 118)
    for row in rows:
        case = f"({row.num_qubits},{row.num_excitations})"
        if row.judge_score is None:
            noisy_metrics = f"{'-':>9} {'-':>9} {'-':>9} {'-':>10}"
        else:
            noisy_metrics = (
                f"{100 * row.subspace_retention:8.2f}% "
                f"{row.score_z:9.4f} {row.score_x:9.4f} "
                f"{row.judge_score:10.6f}"
            )
        print(
            f"{case:>8} {row.variant:<22} {row.moments:6d} "
            f"{row.cz_count:5d} {row.sq_count:5d} "
            f"{row.construction_seconds:8.3f} "
            f"{row.calibration_cost:10.6f} {noisy_metrics}"
        )

    print("\nConstruction-time summary")
    print("-" * 58)
    for variant in sorted({row.variant for row in rows}):
        times = sorted(
            row.construction_seconds
            for row in rows
            if row.variant == variant
        )
        percentile_index = max(0, math.ceil(0.95 * len(times)) - 1)
        print(
            f"{variant:<24} median={statistics.median(times):8.3f}s "
            f"p95={times[percentile_index]:8.3f}s "
            f"max={max(times):8.3f}s"
        )

    scored_rows = [row for row in rows if row.judge_score is not None]
    if not scored_rows:
        return

    variants = sorted({row.variant for row in scored_rows})
    cases = sorted(
        {
            (row.num_qubits, row.num_excitations)
            for row in scored_rows
        }
    )
    print("\nEstimated summed Problem 1 scores")
    print("-" * 58)
    for variant in variants:
        case_means = []
        for case in cases:
            scores = [
                row.judge_score
                for row in scored_rows
                if row.variant == variant
                and (row.num_qubits, row.num_excitations) == case
            ]
            if scores:
                case_means.append(statistics.mean(scores))
        total = sum(case_means)
        trial_scores = [
            row.judge_score
            for row in scored_rows
            if row.variant == variant
        ]
        deviation = (
            statistics.stdev(trial_scores)
            if len(trial_scores) > 1
            else 0.0
        )
        print(
            f"{variant:<24} total={total:12.6f} "
            f"run_score_std={deviation:10.6f}"
        )


def write_csv(rows: list[BenchmarkRow], output_path: str) -> None:
    """Write exact validation and clearly labeled projection metrics."""
    fieldnames = [
        "variant",
        "num_qubits",
        "num_excitations",
        "error_seed",
        "sample_seed",
        "contract_pass",
        "noiseless_fidelity",
        "moments",
        "cz_count",
        "single_qubit_gate_count",
        "construction_seconds",
        "projected_log_loss",
        "projected_survival",
        "projected_infidelity",
        "subspace_retention",
        "score_z",
        "score_x",
        "estimated_judge_score",
        "projection_note",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            projected_survival = math.exp(-row.calibration_cost)
            writer.writerow(
                {
                    "variant": row.variant,
                    "num_qubits": row.num_qubits,
                    "num_excitations": row.num_excitations,
                    "error_seed": row.error_seed,
                    "sample_seed": row.sample_seed,
                    "contract_pass": True,
                    "noiseless_fidelity": "1.0",
                    "moments": row.moments,
                    "cz_count": row.cz_count,
                    "single_qubit_gate_count": row.sq_count,
                    "construction_seconds": f"{row.construction_seconds:.9f}",
                    "projected_log_loss": f"{row.calibration_cost:.12f}",
                    "projected_survival": f"{projected_survival:.12f}",
                    "projected_infidelity": f"{1.0 - projected_survival:.12f}",
                    "subspace_retention": row.subspace_retention,
                    "score_z": row.score_z,
                    "score_x": row.score_x,
                    "estimated_judge_score": row.judge_score,
                    "projection_note": (
                        "Uncertainty-adjusted independent gate/readout "
                        "survival proxy; not an official hardware score"
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="stratified",
        help="'stratified', 'all', or comma-separated N:n pairs",
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
    )
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--error-seeds", default="2026")
    parser.add_argument("--sample-seeds", default="12345")
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("auto", "density_matrix", "trajectory"),
        default="auto",
    )
    parser.add_argument("--idle-phase-error", type=float, default=0.0)
    parser.add_argument(
        "--csv",
        help="Optional path for per-run validation and projection metrics",
    )
    arguments = parser.parse_args()

    cases = _parse_cases(arguments.cases)
    variants = [
        variant
        for variant in arguments.variants.split(",")
        if variant
    ]
    unknown = set(variants).difference(VARIANTS)
    if unknown:
        parser.error(f"unknown variants: {sorted(unknown)}")

    rows = []
    failures = []
    for error_seed in _parse_integer_list(arguments.error_seeds):
        for sample_seed in _parse_integer_list(arguments.sample_seeds):
            for num_qubits, num_excitations in cases:
                for variant in variants:
                    try:
                        row = benchmark_variant(
                            variant,
                            num_qubits,
                            num_excitations,
                            error_seed,
                            sample_seed,
                            arguments.shots,
                            arguments.structural_only,
                            arguments.backend,
                            arguments.idle_phase_error,
                        )
                        rows.append(row)
                    except Exception as error:
                        failures.append(
                            (
                                variant,
                                num_qubits,
                                num_excitations,
                                type(error).__name__,
                                str(error),
                            )
                        )
                        print(
                            f"FAIL ({num_qubits},{num_excitations}) "
                            f"{variant}: {type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )

    print_rows(rows)
    if arguments.csv:
        write_csv(rows, arguments.csv)
    if failures:
        raise SystemExit(f"{len(failures)} benchmark runs failed")


if __name__ == "__main__":
    main()
