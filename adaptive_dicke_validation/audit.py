"""Independent full-branch, pure-state verification of adaptive submissions.

Run --source PATH --label LABEL [--nmax 15]. Score is literally sum N*k/depth,
with unsupported cases contributing zero. Unverified cases never count as
validated. No sampling-based branch coverage or proxy noiseless validation.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import cirq
import cirq_google
import numpy as np

ROOT = Path(__file__).parent


def load_source(path, name="audited_submission"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def structure(circuit, outputs, n):
    violations = []
    device = cirq_google.Willow105
    qubits = set(device.metadata.qubit_set)
    if len(outputs) != n:
        violations.append(f"expected {n} outputs, got {len(outputs)}")
    if not set(outputs) <= circuit.all_qubits():
        violations.append("outputs missing from circuit")
    if not circuit.all_qubits() <= qubits:
        violations.append("qubits outside Willow105")
    for moment_index, moment in enumerate(circuit):
        measured = set()
        for op in moment:
            bare = op.without_classical_controls()
            if cirq.is_measurement(bare):
                measured.update(bare.qubits)
            elif not (bare.gate == cirq.CZ or isinstance(bare.gate, (cirq.PhasedXZGate, cirq.ResetChannel))):
                violations.append(f"unsupported gate at moment {moment_index}: {bare.gate}")
            try:
                device.validate_operation(bare)
            except (ValueError, TypeError) as error:
                violations.append(f"device gate/edge failure: {error}")
        for diagonal in {q.row + q.col for q in measured}:
            missing = {q for q in qubits if q.row + q.col == diagonal} - measured
            if missing:
                violations.append(f"partial readout diagonal {diagonal} at moment {moment_index}: {len(missing)} missing")
    return sorted(set(violations))


def apply_matrix(state, matrix, sites, n):
    axes = list(sites)
    tensor = np.moveaxis(state.reshape((2,) * n), axes, range(len(axes)))
    transformed = (matrix @ tensor.reshape(2 ** len(axes), -1)).reshape(tensor.shape)
    return np.moveaxis(transformed, range(len(axes)), axes).reshape(-1)


def target(n, k):
    state = np.array([int(i).bit_count() == k for i in range(2**n)], dtype=complex)
    return state / np.linalg.norm(state)


def output_fidelity(state, active, outputs, k):
    axes = [active.index(q) for q in sorted(outputs)]
    tensor = np.moveaxis(state.reshape((2,) * len(active)), axes, range(len(axes)))
    matrix = tensor.reshape(2 ** len(outputs), -1)
    projected = target(len(outputs), k).conj() @ matrix
    return float(np.vdot(projected, projected).real)


def verify_all_branches(circuit, outputs, k, max_qubits=20, max_branches=4096):
    # Untouched measured spectators are known |0>, retained in classical records
    # but factored out of the quantum state. No active ancilla is discarded.
    active = sorted(set(outputs) | {q for op in circuit.all_operations()
                                   if not cirq.is_measurement(op.without_classical_controls()) for q in op.qubits})
    n = len(active)
    if n > max_qubits:
        return {"status": "unverified_resource_limit", "active_qubits": n}
    if not circuit.has_measurements() and not any(isinstance(op.gate, cirq.ResetChannel) for op in circuit.all_operations()):
        vector = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=active).final_state_vector
        fidelity = output_fidelity(vector, active, outputs, k)
        return {"status": "verified" if fidelity > 1 - 1e-8 else "invalid_state", "minimum_branch_fidelity": fidelity,
                "branches": 1, "total_probability": 1., "active_qubits": n}
    state = np.zeros(2**n, complex); state[0] = 1
    branches = [(1., state, [])]
    dropped_mass = 0.
    for op in circuit.all_operations():
        bare = op.without_classical_controls()
        new_branches = []
        for weight, vector, records in branches:
            store = cirq.ClassicalDataDictionaryStore()
            for key, bits, qs in records:
                store.record_measurement(cirq.MeasurementKey.parse_serialized(key), bits, qs)
            if not all(condition.resolve(store) for condition in op.classical_controls):
                new_branches.append((weight, vector, records)); continue
            is_reset = isinstance(bare.gate, cirq.ResetChannel)
            if not cirq.is_measurement(bare) and not is_reset:
                vector = apply_matrix(vector, cirq.unitary(bare), [active.index(q) for q in bare.qubits], n)
                new_branches.append((weight, vector, records)); continue
            qs = bare.qubits
            sites = [active.index(q) for q in qs if q in active]
            # Enumerate projector outcomes in one pass with tensor reshaping.
            shaped = np.moveaxis(vector.reshape((2,) * n), sites, range(len(sites))).reshape(2 ** len(sites), -1)
            probabilities = np.sum(abs(shaped) ** 2, axis=1)
            for outcome, probability in enumerate(probabilities):
                if weight * probability < 1e-15:
                    dropped_mass += weight * float(probability); continue
                digits_active = [(outcome >> (len(sites) - 1 - i)) & 1 for i in range(len(sites))]
                digits = []; cursor = 0
                for q in qs:
                    if q in active:
                        digits.append(digits_active[cursor]); cursor += 1
                    else:
                        digits.append(0)
                selected = np.zeros_like(shaped)
                selected[outcome] = shaped[outcome] / np.sqrt(probability)
                moved_shape = (2,) * len(sites) + (2,) * (n - len(sites))
                projected = np.moveaxis(selected.reshape(moved_shape), range(len(sites)), sites).reshape(-1)
                updated = records
                if is_reset:
                    if digits[0]:
                        projected = apply_matrix(projected, cirq.unitary(cirq.X), sites, n)
                else:
                    invert = bare.gate.full_invert_mask()
                    digits = [int(bit) ^ int(flip) for bit, flip in zip(digits, invert)]
                    updated = records + [(cirq.measurement_key_name(bare), digits, qs)]
                new_branches.append((weight * float(probability), projected, updated))
        branches = new_branches
        if len(branches) > max_branches:
            return {"status": "unverified_branch_limit", "active_qubits": n, "branches": len(branches)}
    fidelities = [output_fidelity(vector, active, outputs, k) for _, vector, _ in branches]
    minimum = min(fidelities)
    total = sum(weight for weight, _, _ in branches)
    return {"status": "verified" if minimum > 1 - 1e-8 and abs(total - 1) < 1e-9 and dropped_mass < 1e-10 else "invalid_state",
            "minimum_branch_fidelity": minimum, "branches": len(branches), "active_qubits": n,
            "total_probability": total, "discarded_roundoff_probability": dropped_mass}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--nmax", type=int, default=20)
    args = parser.parse_args()
    module = load_source(Path(args.source).resolve())
    output = ROOT / "artifacts" / args.label
    output.mkdir(parents=True, exist_ok=True)
    data = {"label": args.label, "source": str(Path(args.source).resolve()), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "cirq": cirq.__version__, "numpy": np.__version__,
            "score_definition": "sum N*k / len(circuit); text domain N<=15, literal notebook N<=20",
            "records": []}
    for n in range(2, args.nmax + 1):
        for k in range(1, n // 2 + 1):
            record = {"n": n, "k": k}
            start = perf_counter()
            try:
                circuit, outputs = module.create_dicke_state_adaptive(n, k)
                construction = perf_counter() - start
                record.update(construction_seconds=construction, depth=len(circuit),
                              measurements=sum(cirq.is_measurement(op) for op in circuit.all_operations()),
                              total_qubits=len(circuit.all_qubits()))
                problems = structure(circuit, outputs, n)
                if construction >= 60:
                    problems.append("construction exceeds 60 seconds")
                record["violations"] = problems
                record["verification"] = verify_all_branches(circuit, outputs, k, max_qubits=16)
                record["accepted"] = not problems and record["verification"]["status"] == "verified"
                record["score_term"] = n * k / len(circuit) if record["accepted"] else 0.
                cirq.to_json(circuit, output / f"{n}-{k}.cirq.json")
            except NotImplementedError as error:
                record.update(accepted=False, unsupported=True, score_term=0., reason=str(error))
            except Exception as error:
                record.update(accepted=False, error=f"{type(error).__name__}: {error}", score_term=0.)
            record["audit_seconds"] = perf_counter() - start
            data["records"].append(record)
            data["validated_score_15"] = sum(r["score_term"] for r in data["records"] if r["n"] <= 15)
            data["validated_score_20"] = sum(r["score_term"] for r in data["records"] if r["n"] <= 20)
            (output / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"{args.label} D({n},{k}) depth={record.get('depth')} accepted={record['accepted']} "
                  f"{record.get('verification', record.get('error', record.get('reason','')))}", flush=True)
    print(f"SCORE {args.label}: N<=15 {data['validated_score_15']:.10f}; N<=20 {data['validated_score_20']:.10f}", flush=True)


if __name__ == "__main__":
    main()
