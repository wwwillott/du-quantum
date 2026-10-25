"""One isolated timing-only refinement; never rewrites the primary evaluation record."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import cirq
from evaluate_worker import atomic_json, exactness, validate
from official_calibration import calibration_fingerprint, load_calibration


def main():
    source = HERE / "sources/C_submission.py"
    record_path = HERE / "records/C_s202_N08_n03.json"
    original = json.loads(record_path.read_text(encoding="utf-8"))
    calibration = load_calibration(202)
    result = dict(
        candidate="C", seed=202, N=8, n=3,
        purpose="Single isolated timing refinement; primary record and eligibility unchanged",
        python=sys.executable, pid=os.getpid(), pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        cpu_thread_limit=1, authorized_total_worker_cap=5,
        source_path="team_eval/sources/C_submission.py",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_record="team_eval/records/C_s202_N08_n03.json",
        original_construction_seconds=original.get("construction_seconds"),
        original_circuit_sha256=original.get("circuit_sha256"),
        calibration_sha256=calibration_fingerprint(calibration),
        status="importing", started_unix=time.time(),
    )
    output = HERE / "timing_retry_results.json"
    atomic_json(output, result)
    try:
        spec = importlib.util.spec_from_file_location("timing_retry_candidate_C", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        start = time.perf_counter()
        spec.loader.exec_module(module)
        result["import_seconds"] = time.perf_counter() - start
        result["status"] = "building"
        atomic_json(output, result)
        start = time.perf_counter()
        circuit, qubits = module.create_dicke_state(
            8, 3, **{key: dict(value) for key, value in calibration.items()}
        )
        result["construction_seconds"] = time.perf_counter() - start
        result["over_60_seconds"] = result["construction_seconds"] > 60
        circuit_text = cirq.to_json(circuit)
        result["circuit_sha256"] = hashlib.sha256(circuit_text.encode()).hexdigest()
        result["same_circuit_sha256"] = result["circuit_sha256"] == original.get("circuit_sha256")
        result["qubits"] = [[int(q.row), int(q.col)] for q in sorted(qubits)]
        result["same_qubits"] = result["qubits"] == original.get("qubits")
        result["moments"] = len(circuit)
        result["validity"], result["cz_count"] = validate(circuit, qubits, 8, calibration)
        result["exactness"] = exactness(circuit, qubits, 8, 3)
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    result["finished_unix"] = time.time()
    atomic_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
