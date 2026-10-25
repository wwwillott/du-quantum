"""Compare the reusable compiler with the preserved competition incumbent."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from dicke_compiler import compile_dicke, willow_target
from dicke_compiler.passes import circuit_metrics
from dicke_compiler.verification import state_preparation_fidelity


def _load_incumbent():
    submission_path = Path(__file__).resolve().parents[1] / "dicke_submission.py"
    spec = importlib.util.spec_from_file_location(
        "dicke_submission_benchmark",
        submission_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load incumbent from {submission_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    dicke_submission = _load_incumbent()
    rows = []
    for num_qubits, num_excitations in ((4, 2), (6, 3), (8, 4)):
        compiled = compile_dicke(
            num_qubits,
            num_excitations,
            hardware=willow_target(),
            path_limit=64,
        )
        incumbent_circuit, incumbent_outputs = (
            dicke_submission.create_dicke_state_adaptive(
                num_qubits,
                num_excitations,
            )
        )
        incumbent_metrics = circuit_metrics(incumbent_circuit)
        incumbent_fidelity = state_preparation_fidelity(
            incumbent_circuit,
            sorted(incumbent_outputs),
            compiled.target,
        )
        for implementation, family, metrics, fidelity in (
            (
                "dicke_compiler",
                compiled.candidate_family,
                compiled.after_metrics,
                compiled.verification.fidelity,
            ),
            (
                "competition_incumbent",
                "adaptive_portfolio",
                incumbent_metrics,
                incumbent_fidelity,
            ),
        ):
            rows.append(
                {
                    "num_qubits": num_qubits,
                    "num_excitations": num_excitations,
                    "implementation": implementation,
                    "family": family,
                    "moments": metrics.moments,
                    "cz_depth": metrics.cz_depth,
                    "cz_count": metrics.cz_count,
                    "fidelity": fidelity,
                }
            )
    output_path = Path("benchmarks/incumbent_comparison.csv")
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
