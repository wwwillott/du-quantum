"""Run: python -m interference_dicke.experiment [--extended]

Each optimization answers whether more matching layers or CP gates increase
fidelity enough to justify compiled CZ cost. n=4 resolves the analytic ceiling;
n=6 is optional and follows only after the small milestone is completed.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import os
import subprocess
from time import perf_counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cirq
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from dicke_willow.divide import prepare_balanced
from dicke_willow.scs import prepare_dicke_circuit
from willow_dicke.circuit.lower import bridge_along_path, expand_circuit
from interference_dicke.core import Sector, optimize, schedule_for
from interference_dicke.hardware import (IllustrativeNoise, circuit_for, compile_validate,
                                         ideal_fidelity, metrics, noisy_metrics, willow_path)


def kernel_benchmarks():
    rows = []
    for n, l in [(4, 2), (6, 3), (8, 4), (10, 5), (12, 6)]:
        sector = Sector(n, l, [(i, i + 1) for i in range(n - 1)])
        # Equal task: one isolated XY evolution, not an entire noncommuting layer.
        gate = ("xy", (0, 1))
        h = sector.hamiltonian({(0, 1): 1.}, [0.] * n, {})
        state = np.random.default_rng(15).normal(size=len(sector.masks)).astype(complex)
        state /= np.linalg.norm(state)
        expected = sector.apply(state, gate, .37)
        methods = {"direct": lambda: sector.apply(state, gate, .37),
                   "sparse_expm_multiply": lambda: expm_multiply(-.37j * h, state)}
        if len(state) <= 70:
            dense = h.toarray()
            methods["dense_expm"] = lambda: expm(-.37j * dense) @ state
        row = {"n": n, "l": l, "dimension": len(state),
               "state_bytes": state.nbytes, "sparse_operator_bytes": h.data.nbytes + h.indices.nbytes + h.indptr.nbytes}
        for name, function in methods.items():
            error = float(np.max(abs(function() - expected)))
            times = []
            for _ in range(7):
                start = perf_counter(); function(); times.append(perf_counter() - start)
            row[name] = {"median_seconds": float(np.median(times)), "max_error": error,
                         "repetitions": 7}
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--noise", action="store_true", help="run bounded compiled density simulations")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(__file__).parent / "artifacts" / stamp
    output.mkdir(parents=True, exist_ok=False)
    data = {"run_id": stamp, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "processor": platform.processor(), "threads": 1,
                            "packages": {p: importlib.metadata.version(p) for p in
                                         ["numpy", "scipy", "cirq-core", "cirq-google", "matplotlib"]}},
            "conventions": "bit i = qubit i; initial lowest l bits; uniform positive target; hbar=1; see core.py",
            "noise_model": "none for optimization; optional synthetic channels on compiled CZ circuits",
            "output_path": str(output.resolve()), "experiments": [], "baselines": []}
    # Snapshot exact sources because the run may originate in an uncommitted tree.
    import shutil
    for name in ["core.py", "hardware.py", "experiment.py"]:
        shutil.copyfile(Path(__file__).parent / name, output / name)

    def save():
        temporary = output / "results.tmp"
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(output / "results.json")

    with nullcontext():
        data["kernel_benchmarks"] = kernel_benchmarks()
        save()
        cases = [(4, 1), (4, 2)] + ([(6, 3)] if args.extended else [])
        for n, l in cases:
            qubits = willow_path(n)
            edges = [(i, i + 1) for i in range(n - 1)]
            sector = Sector(n, l, edges)
            topology = {"source": "dicke_willow.topology.willow105_graph / Cirq Willow105 static",
                        "qubits": [[q.row, q.col] for q in qubits], "edges": edges,
                        "kind": "physical path (other device edges unused)"}
            depths = [2, 4, 6, 8] if n == 4 and l == 2 else ([4] if l == 1 else [6, 10])
            cp_counts = [0, 1] if l != 1 else [0]
            if n == 6:
                cp_counts = [0, 1, 2]
            for depth in depths:
                for cp in cp_counts:
                    schedule = schedule_for(sector, depth, cp)
                    result = optimize(sector, schedule, starts=4, maxiter=350, seconds=12)
                    logical = circuit_for(sector, schedule, result["parameters"], qubits)
                    native = compile_validate(logical, qubits)
                    compiled_fidelity = ideal_fidelity(native, qubits, l)
                    if abs(result["fidelity"] - compiled_fidelity) > 1e-9:
                        raise AssertionError("fixed-sector and compiled Cirq simulation disagree")
                    record = {"id": f"chain-{n}-{l}-depth{depth}-cp{cp}", "n": n, "l": l,
                              "dimension": len(sector.masks), "topology": topology,
                              "family": "free_XY" if cp == 0 else "interacting_XY",
                              "matching_layers": depth, "cp_count": cp, "parameter_count": len(schedule),
                              "schedule": schedule, "optimization": result,
                              "logical": metrics(logical), "compiled_CZ": metrics(native),
                              "compiled_fidelity": compiled_fidelity, "device_validated": True,
                              "duration_calibrated_ns": None}
                    if args.noise and n == 4 and l == 2:
                        record["noise"] = [noisy_metrics(native, qubits, l, replace(IllustrativeNoise(), p2=p))
                                           for p in [0., .001, .005]]
                    data["experiments"].append(record)
                    cirq_path = output / (record["id"] + ".cirq.json")
                    cirq.to_json(native, cirq_path)
                    record["native_circuit_path"] = cirq_path.name
                    save()
                    print(f"{record['id']}: F={compiled_fidelity:.10f} CZ={record['compiled_CZ']['two_qubit_gates']} "
                          f"depth={len(native)} optimization={result['seconds']:.3f}s", flush=True)
            if l == 1:
                continue
            for method in ["SCS", "recursive"]:
                start = perf_counter()
                if method == "SCS":
                    baseline = prepare_dicke_circuit(qubits, l)
                else:
                    baseline, _ = prepare_balanced(qubits, l)
                    baseline = bridge_along_path(expand_circuit(baseline), qubits)
                native = compile_validate(baseline, qubits)
                fidelity = ideal_fidelity(native, qubits, l)
                if fidelity < 1 - 1e-9:
                    raise AssertionError(f"baseline {method} failed exact state verification: {fidelity}")
                record = {"method": method, "n": n, "l": l, "topology": topology,
                          "ideal_fidelity": fidelity, "logical_after_routing": metrics(baseline),
                          "compiled_CZ": metrics(native), "device_validated": True,
                          "construction_compilation_seconds": perf_counter() - start,
                          "optimization_seconds": 0., "duration_calibrated_ns": None}
                if args.noise and n == 4:
                    record["noise"] = [noisy_metrics(native, qubits, l, replace(IllustrativeNoise(), p2=p))
                                       for p in [0., .001, .005]]
                data["baselines"].append(record)
                cirq.to_json(native, output / f"{method}-{n}-{l}.cirq.json")
                save()
                print(f"baseline {method}-{n}-{l}: F={fidelity:.10f} CZ={record['compiled_CZ']['two_qubit_gates']}", flush=True)
    print(f"Results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
