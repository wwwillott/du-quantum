"""Same state-aware gate elimination for candidates AND baselines, n<=6.

This is a state-preparation optimization, not a coherent multi-input unitary
replacement. A gate is removed only if its action on the current pure state
is a global phase within 1e-12. Full 2^n vectors here are justified because
the baseline temporarily leaves fixed weight; this pass is capped at n=6.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import cirq
import numpy as np
from dicke_willow.scs import prepare_dicke_circuit
from dicke_willow.divide import prepare_balanced
from willow_dicke.circuit.lower import bridge_along_path, expand_circuit
from interference_dicke.core import Sector
from interference_dicke.hardware import circuit_for, compile_validate, ideal_fidelity, metrics, noisy_metrics


def prune(circuit, qubits, atol=1e-12):
    n = len(qubits)
    if n > 6:
        raise ValueError("state-aware pruning capped at n <= 6")
    state = np.zeros(2**n, complex); state[0] = 1
    kept = []
    for op in circuit.all_operations():
        axes = [qubits.index(q) for q in op.qubits]
        tensor = state.reshape((2,) * n)
        moved = np.moveaxis(tensor, axes, range(len(axes)))
        out = (cirq.unitary(op) @ moved.reshape(2 ** len(axes), -1)).reshape(moved.shape)
        evolved = np.moveaxis(out, range(len(axes)), axes).reshape(-1)
        overlap = np.vdot(state, evolved)
        phase = overlap / abs(overlap) if abs(overlap) > 0 else 1
        if np.linalg.norm(evolved - phase * state) > atol:
            kept.append(op)
            state = evolved
    return cirq.Circuit(kept)


def main():
    root = Path(__file__).parent / "artifacts"
    runs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(root.glob("202*/results.json"))]
    small_path, small = next((p, d) for p, d in runs if any("noise" in r for r in d["experiments"]))
    extended_path, extended = next((p, d) for p, d in runs if any(r["n"] == 6 for r in d["experiments"]))
    geometry_path = sorted(root.glob("geometry-*/results.json"))[-1]
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    output = root / ("pruned-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    for filename in ["prune.py", "hardware.py"]:
        shutil.copyfile(Path(__file__).parent / filename, output / filename)
    data = {"run_id": output.name, "environment": extended["environment"], "git_commit": extended["git_commit"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "sources": [str(p.resolve()) for p in [small_path, extended_path, geometry_path]],
            "records": [], "scope": "same state-aware pass on all methods; input |0> incl initialization; atol=1e-12"}
    inputs = [r for r in small["experiments"] if r["l"] == 2]
    inputs += [r for r in extended["experiments"] if r["n"] == 6]
    inputs += [dict(r, id="star-depth6", cp_count=0, topology={"qubits": r["qubits"], "edges": r["edges"]})
               for r in geometry["records"] if r["geometry"] == "star" and r["matching_layers"] == 6]
    inputs += small["baselines"] + [r for r in extended["baselines"] if r["n"] == 6]
    for record in inputs:
        n, l = record["n"], record["l"]
        qubits = [cirq.GridQubit(*xy) for xy in record["topology"]["qubits"]]
        label = record.get("id", f"{record.get('method')}-{n}-{l}")
        if "schedule" in record:
            sector = Sector(n, l, record["topology"]["edges"])
            logical = circuit_for(sector, record["schedule"], record["optimization"]["parameters"], qubits)
        elif record["method"] == "SCS":
            logical = prepare_dicke_circuit(qubits, l)
        else:
            logical, _ = prepare_balanced(qubits, l)
            logical = bridge_along_path(expand_circuit(logical), qubits)
        reduced = prune(logical, qubits)
        native = compile_validate(reduced, qubits)
        before = ideal_fidelity(logical, qubits, l)
        after = ideal_fidelity(native, qubits, l)
        if abs(before - after) > 1e-9:
            raise AssertionError("state-aware pruning changed fidelity")
        result = {"id": label, "n": n, "l": l, "ideal_fidelity": after,
                  "unpruned_CZ": record["compiled_CZ"], "compiled_CZ": metrics(native),
                  "logical_removed_operations": sum(1 for _ in logical.all_operations()) - sum(1 for _ in reduced.all_operations()),
                  "topology": record["topology"], "device_validated": True,
                  "optimization_seconds": record.get("optimization", {}).get("seconds", 0),
                  "matching_layers": record.get("matching_layers"), "cp_count": record.get("cp_count")}
        if n == 4 or (record.get("matching_layers"), record.get("cp_count")) in [(6, 0), (6, 2), (10, 1)] or "method" in record:
            result["noise"] = noisy_metrics(native, qubits, l)
        data["records"].append(result)
        cirq.to_json(native, output / f"{label}.cirq.json")
        (output / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"{label}: F={after:.10f}, CZ={result['compiled_CZ']['two_qubit_gates']} (before {record['compiled_CZ']['two_qubit_gates']})", flush=True)


if __name__ == "__main__":
    main()
