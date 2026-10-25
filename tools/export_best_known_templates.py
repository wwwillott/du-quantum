"""Generate the versioned best-known Dicke template registry.

This development tool intentionally imports the preserved competition
portfolio. The generated package asset has no runtime dependency on it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cirq


def _load_submission(repository: Path):
    path = repository / "dicke_submission.py"
    spec = importlib.util.spec_from_file_location(
        "dicke_submission_template_source",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load template source from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode_circuit(
    circuit: cirq.AbstractCircuit,
    outputs: set,
) -> tuple[list, tuple[cirq.Qid, ...]]:
    ordered = tuple(sorted(outputs, key=str))
    positions = {qubit: index for index, qubit in enumerate(ordered)}
    if circuit.all_qubits() != set(ordered):
        raise ValueError("best-known templates must not contain ancillas")
    moments = []
    for moment in circuit:
        entries = []
        for operation in moment.operations:
            if operation.gate == cirq.CZ:
                entries.append(
                    ["cz", *(positions[qubit] for qubit in operation.qubits)]
                )
            elif isinstance(operation.gate, cirq.PhasedXZGate):
                gate = operation.gate
                entries.append(
                    [
                        "phxz",
                        positions[operation.qubits[0]],
                        float(gate.x_exponent),
                        float(gate.z_exponent),
                        float(gate.axis_phase_exponent),
                    ]
                )
            else:
                raise ValueError(f"unsupported template operation {operation!r}")
        moments.append(entries)
    return moments, ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/dicke_compiler/data/best_known_templates.json"
        ),
    )
    parser.add_argument("--source-commit", default="5f15cbe")
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    submission = _load_submission(repository)
    templates = []
    for num_qubits in range(2, 16):
        for num_excitations in range(1, num_qubits // 2 + 1):
            circuit, outputs = submission.create_dicke_state_adaptive(
                num_qubits,
                num_excitations,
            )
            moments, ordered = _encode_circuit(circuit, set(outputs))
            payload = {
                "name": f"best_known_{num_qubits}_{num_excitations}",
                "num_qubits": num_qubits,
                "num_excitations": num_excitations,
                "source": (
                    "DuQuantum competition portfolio "
                    f"at commit {arguments.source_commit}"
                ),
                "grid_qubits": [
                    [qubit.row, qubit.col] for qubit in ordered
                ],
                "moments": moments,
            }
            canonical = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            payload["sha256"] = hashlib.sha256(canonical).hexdigest()
            templates.append(payload)
    document = {
        "format_version": 1,
        "source_commit": arguments.source_commit,
        "template_count": len(templates),
        "templates": templates,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(templates)} templates to {arguments.output}")


if __name__ == "__main__":
    main()
