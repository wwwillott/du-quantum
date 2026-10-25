"""Bounded n=6 density checks after ideal/native validation; no optimization.

Resolves whether the small-system compiled-cost benefit survives the same
synthetic noise. Four representative schedules and both verified baselines;
one sensitivity setting only. State dimension is 64, density is 64x64.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import cirq
from interference_dicke.hardware import noisy_metrics


def main():
    root = Path(__file__).parent / "artifacts"
    sources = []
    for path in root.glob("*/results.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if any(x["n"] == 6 for x in data.get("experiments", [])):
            sources.append((path, data))
    source, data = sorted(sources, key=lambda x: str(x[0]))[-1]
    output = root / ("noise-six-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    result = {"source": str(source.resolve()), "environment": data["environment"],
              "git_commit": data["git_commit"], "timestamp_utc": datetime.now(timezone.utc).isoformat(), "records": []}
    selection = [x for x in data["experiments"] if x["n"] == 6 and
                 (x["matching_layers"], x["cp_count"]) in [(6, 0), (6, 1), (6, 2), (10, 1)]]
    selection += [x for x in data["baselines"] if x["n"] == 6]
    for record in selection:
        label = record.get("id", record.get("method"))
        filename = record.get("native_circuit_path", f"{label}-6-3.cirq.json")
        circuit = cirq.read_json(source.parent / filename)
        qubits = [cirq.GridQubit(*xy) for xy in record["topology"]["qubits"]]
        start = time.perf_counter()
        noise = noisy_metrics(circuit, qubits, 3)
        result["records"].append({"id": label, "noise": noise, "simulation_seconds": time.perf_counter() - start})
        (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{label} noisy={noise['fidelity']:.8f} valid={noise['p_valid']:.8f} elapsed={time.perf_counter()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
