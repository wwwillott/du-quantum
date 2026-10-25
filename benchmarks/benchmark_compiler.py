"""Reproducible benchmark CLI for the public Dicke compiler."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dicke_compiler import compile_dicke, willow_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-qubits", type=int, default=8)
    parser.add_argument(
        "--domain",
        choices=("all", "half"),
        default="all",
        help="'half' evaluates 1 <= k <= floor(N/2), matching Dicke symmetry",
    )
    parser.add_argument(
        "--hardware",
        choices=("line", "willow"),
        default="line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/compiler_results.csv"),
    )
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    hardware = willow_target() if arguments.hardware == "willow" else None

    rows = []
    minimum_qubits = 2 if arguments.domain == "half" else 1
    for num_qubits in range(minimum_qubits, arguments.max_qubits + 1):
        excitations = (
            range(1, num_qubits // 2 + 1)
            if arguments.domain == "half"
            else range(num_qubits + 1)
        )
        for num_excitations in excitations:
            result = compile_dicke(
                num_qubits,
                num_excitations,
                hardware=hardware,
                path_limit=64 if hardware is not None else 16,
            )
            rows.append(
                {
                    "hardware": arguments.hardware,
                    "num_qubits": num_qubits,
                    "num_excitations": num_excitations,
                    "candidate_family": result.candidate_family,
                    "moments": result.after_metrics.moments,
                    "cz_depth": result.after_metrics.cz_depth,
                    "cz_count": result.after_metrics.cz_count,
                    "single_qubit_gates": (
                        result.after_metrics.single_qubit_gates
                    ),
                    "runtime_seconds": result.runtime_seconds,
                    "verification_status": result.verification.status.value,
                    "fidelity": result.verification.fidelity,
                }
            )

    with arguments.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {arguments.output}")


if __name__ == "__main__":
    main()
