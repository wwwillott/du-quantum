"""Bounded problem-1 contract smoke check of both latest public APIs."""
import argparse
import json
import sys
from time import perf_counter

import cirq
from adaptive_dicke_validation.audit import ROOT, load_source, structure, verify_all_branches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "snapshots/kai-48115b9"))
    calibrator = load_source(ROOT / "snapshots/kai-48115b9/test_dicke.py", "calibration_source")
    calibration = calibrator.fake_calibration(seed=101)
    module = load_source(args.source, args.label.replace("-", "_"))
    results = {"label": args.label, "calibration_seed": 101, "calibration_source": "Kai test_dicke.fake_calibration; synthetic", "records": []}
    output = ROOT / "artifacts" / f"main-{args.label}.json"
    for n, k in [(2, 1), (4, 2), (6, 3), (8, 4), (10, 3)]:
        start = perf_counter()
        try:
            circuit, outputs = module.create_dicke_state(n, k, *calibration)
            build_seconds = perf_counter() - start
            problems = structure(circuit, outputs, n)
            cz, sq, zero, one = calibration
            literal_key_warnings = []
            for op in circuit.all_operations():
                if op.classical_controls or cirq.is_measurement(op):
                    problems.append("forbidden feedback/measurement in problem 1")
                if op.gate == cirq.CZ and op.qubits not in cz:
                    if op.qubits[::-1] in cz:
                        literal_key_warnings.append([str(q) for q in op.qubits])
                    else:
                        problems.append("CZ edge absent in both orientations from supplied calibration")
                if not set(op.qubits) <= set(sq) & set(zero) & set(one):
                    problems.append("unavailable calibrated/readout qubit")
            if build_seconds >= 60 or len(circuit) > 1750:
                problems.append("runtime or moment limit exceeded")
            validation = verify_all_branches(circuit, outputs, k)
            record = {"n": n, "k": k, "build_seconds": build_seconds, "depth": len(circuit),
                      "CZ": sum(op.gate == cirq.CZ for op in circuit.all_operations()),
                      "violations": sorted(set(problems)), "verification": validation,
                      "literal_key_warnings": literal_key_warnings,
                      "accepted": not problems and validation["status"] == "verified"}
        except Exception as error:
            record = {"n": n, "k": k, "accepted": False, "error": f"{type(error).__name__}: {error}"}
        results["records"].append(record)
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(args.label, record, flush=True)


if __name__ == "__main__":
    main()
