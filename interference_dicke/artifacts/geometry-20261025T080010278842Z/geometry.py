"""Cheapest non-chain checks: actual Willow square and degree-three star.

Question: does geometry alone remove the open-chain ceiling? A cycle still
has a parity-conditioned quadratic Jordan-Wigner representation; the star
does not have the same argument. Failure of a shallow optimizer is not a
reachability certificate. Budget: six optimizations, <=12 s each, n=4 only.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from dicke_willow.topology import willow105_graph
from interference_dicke.core import Sector, schedule_for, optimize
from interference_dicke.hardware import circuit_for, compile_validate, ideal_fidelity, metrics, noisy_metrics


def main():
    graph = willow105_graph()
    center = next(q for q in sorted(graph) if len(graph[q]) >= 3)
    star = [center] + sorted(graph[center])[:3]
    square = None
    for a in sorted(graph):
        for b in sorted(graph[a]):
            for c in sorted(graph[a]):
                if b >= c:
                    continue
                common = sorted((set(graph[b]) & set(graph[c])) - {a})
                if common:
                    square = [a, b, common[0], c]
                    break
            if square:
                break
        if square:
            break
    if square is None:
        raise ValueError("no square exists in device topology")
    output = Path(__file__).parent / "artifacts" / ("geometry-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    for source in ["core.py", "hardware.py", "geometry.py"]:
        shutil.copyfile(Path(__file__).parent / source, output / source)
    data = {"run_id": output.name, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "records": [], "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "noise": "synthetic defaults from source snapshot; no hardware calibration",
            "conventions": "same as experiment.py; initial bits 0,1 occupied; uniform positive target"}
    for kind, qubits, edges, depths in [("square", square, [(0, 1), (1, 2), (2, 3), (0, 3)], [4, 6, 8]),
                                       ("star", star, [(0, 1), (0, 2), (0, 3)], [6, 9, 12])]:
        assert all(qubits[j] in graph[qubits[i]] for i, j in edges)
        sector = Sector(4, 2, edges)
        for depth in depths:
            schedule = schedule_for(sector, depth, 0)
            result = optimize(sector, schedule, starts=4, seconds=12)
            logical = circuit_for(sector, schedule, result["parameters"], qubits)
            native = compile_validate(logical, qubits)
            fidelity = ideal_fidelity(native, qubits, 2)
            if abs(result["fidelity"] - fidelity) > 1e-9:
                raise AssertionError("geometry simulation mismatch")
            record = {"geometry": kind, "n": 4, "l": 2, "edges": edges,
                      "qubits": [[q.row, q.col] for q in qubits], "matching_layers": depth,
                      "schedule": schedule, "optimization": result, "compiled_fidelity": fidelity,
                      "compiled_CZ": metrics(native), "device_validated": True,
                      "noise": noisy_metrics(native, qubits, 2)}
            data["records"].append(record)
            (output / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            import cirq
            cirq.to_json(native, output / f"{kind}-{depth}.cirq.json")
            print(f"{kind} layers={depth} F={fidelity:.10f} CZ={record['compiled_CZ']['two_qubit_gates']}", flush=True)
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
