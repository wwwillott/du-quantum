"""Reproducible CPU-only scoring of saved DuQuantum circuits.

Example (run from repository root with the miniforge Python):
    python team_eval/score_circuit.py team_eval/circuits/A_s101_N06_n03.json \
        --qubits team_eval/records/A_s101_N06_n03.json --N 6 --n 3 --seed 101

Qubits may be a JSON file containing [[row, col], ...], an inline JSON array,
or a Cirq JSON list. A construction record with a "qubits" field is accepted.
If omitted, every circuit qubit is a state qubit.
The function accepts an already-loaded Circuit and set of GridQubits too.
All requested output paths must be within team_eval/.

By default, exactness (fidelity > 1 - 1e-8, ancillas traced) and calibrated
Willow gate/moment validity are checked before simulation. Failed circuits are
refused; --output preserves the failed preflight metadata for inspection.
Construction time cannot be established from an already-built circuit.

The simulator and scorer are tools/sim.py:simulate and score_stats. Default
``backend='numba'`` replaces only Batch's arithmetic kernels, retaining the
original simulator's operations, noise model, RNG draws, and shot scorer.
It uses CPU, never CUDA. Statistics are simulator proxies, not hardware data.
"""
from __future__ import annotations

import os

# Set before importing NumPy, and in every standalone worker process.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "team_eval"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cirq
import numpy as np
from tools import sim

SIM_SOURCE_SHA256 = hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest()


def _output_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"Output must be inside {OUTPUT_ROOT}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _rng_seed(seed: int, N: int, n: int, model: str, purpose: str) -> int:
    """Stable across Python processes; independent of candidate and execution order."""
    payload = f"duquantum-head-to-head-v1:{seed}:{N}:{n}:{model}:{purpose}"
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "little")


def _write_result(path: str | Path, result: dict) -> None:
    path = _output_path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(30):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.1)


def load_state_qubits(value: Any, circuit: cirq.Circuit) -> set[cirq.GridQubit]:
    if value is None:
        return set(circuit.all_qubits())
    if isinstance(value, (str, Path)):
        raw = str(value)
        if not raw.lstrip().startswith(("[", "{")):
            raw = Path(raw).read_text(encoding="utf-8-sig")
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                value = value.get("state_qubits", value.get("qubits", value))
            if isinstance(value, list) and all(isinstance(q, list) and len(q) == 2 for q in value):
                return {cirq.GridQubit(int(row), int(col)) for row, col in value}
        except (json.JSONDecodeError, TypeError):
            pass
        value = cirq.read_json(json_text=raw)
    if isinstance(value, dict):
        value = value.get("state_qubits", value.get("qubits", value))
    qubits = set()
    for qubit in value:
        if isinstance(qubit, cirq.GridQubit):
            qubits.add(qubit)
        elif isinstance(qubit, (tuple, list)) and len(qubit) == 2:
            qubits.add(cirq.GridQubit(int(qubit[0]), int(qubit[1])))
        elif isinstance(qubit, dict) and "row" in qubit and "col" in qubit:
            qubits.add(cirq.GridQubit(int(qubit["row"]), int(qubit["col"])))
        else:
            raise ValueError(f"Cannot parse state qubit: {qubit!r}")
    return qubits


_NUMBA_BATCH = None


def _make_numba_batch():
    """Compile serial, complex128, non-fastmath arithmetic kernels on first use."""
    global _NUMBA_BATCH
    if _NUMBA_BATCH is not None:
        return _NUMBA_BATCH
    from numba import njit

    @njit(cache=True)
    def apply1_kernel(psi, q, U, mask, nq):
        stride = 1 << (nq - q - 1)
        dim = psi.shape[1]
        for b in range(psi.shape[0]):
            if mask.size and not mask[b]:
                continue
            for base in range(0, dim, stride * 2):
                for offset in range(stride):
                    i = base + offset
                    a, z = psi[b, i], psi[b, i + stride]
                    psi[b, i] = U[0, 0] * a + U[0, 1] * z
                    psi[b, i + stride] = U[1, 0] * a + U[1, 1] * z

    @njit(cache=True)
    def per_traj_kernel(psi, q, Us, nq):
        stride = 1 << (nq - q - 1)
        dim = psi.shape[1]
        for b in range(psi.shape[0]):
            U = Us[b]
            if U[0, 1] == 0 and U[1, 0] == 0:
                if U[0, 0] == 1 and U[1, 1] == 1:
                    continue
                for base in range(0, dim, stride * 2):
                    for offset in range(stride):
                        i = base + offset
                        psi[b, i] *= U[0, 0]
                        psi[b, i + stride] *= U[1, 1]
            else:
                for base in range(0, dim, stride * 2):
                    for offset in range(stride):
                        i = base + offset
                        a, z = psi[b, i], psi[b, i + stride]
                        psi[b, i] = U[0, 0] * a + U[0, 1] * z
                        psi[b, i + stride] = U[1, 0] * a + U[1, 1] * z

    @njit(cache=True)
    def population_kernel(psi, q, nq):
        stride = 1 << (nq - q - 1)
        dim = psi.shape[1]
        out = np.zeros(psi.shape[0], dtype=np.float64)
        for b in range(psi.shape[0]):
            total = 0.0
            for base in range(0, dim, stride * 2):
                for offset in range(stride):
                    val = psi[b, base + stride + offset]
                    total += val.real * val.real + val.imag * val.imag
            out[b] = total
        return out

    @njit(cache=True)
    def normalize_kernel(psi):
        for b in range(psi.shape[0]):
            total = 0.0
            for i in range(psi.shape[1]):
                val = psi[b, i]
                total += val.real * val.real + val.imag * val.imag
            norm = np.sqrt(total)
            for i in range(psi.shape[1]):
                psi[b, i] /= norm

    class NumbaBatch(sim.Batch):
        def apply1(self, q, U, mask=None):
            apply1_kernel(self.psi.reshape(self.B, -1), q, np.asarray(U),
                          np.empty(0, dtype=np.bool_) if mask is None else mask, self.nq)

        def apply1_per_traj(self, q, Us):
            per_traj_kernel(self.psi.reshape(self.B, -1), q, Us, self.nq)

        def pop1(self, q):
            return population_kernel(self.psi.reshape(self.B, -1), q, self.nq)

        def normalize(self):
            normalize_kernel(self.psi.reshape(self.B, -1))

    _NUMBA_BATCH = NumbaBatch
    return _NUMBA_BATCH


@contextmanager
def simulation_backend(backend: str = "numba"):
    """Process-local kernel selection; do not run concurrent threads in one process."""
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be numpy or numba")
    old = sim.Batch
    if backend == "numba":
        sim.Batch = _make_numba_batch()
    try:
        yield
    finally:
        sim.Batch = old


def score_circuit(circuit_json: str | Path | cirq.Circuit, qubits: Any, N: int,
                  n: int, seed: int, *, calibration: dict | None = None,
                  pauli_trajectories: int = 2000, hw_trajectories: int = 1000,
                  resamplings: int = 400, shots: int = 10000,
                  models: tuple[str, ...] = ("pauli", "hw"),
                  backend: str = "numba", probabilities_path: str | Path | None = None,
                  output_path: str | Path | None = None, check_circuit: bool = True,
                  progress=None) -> dict:
    """Score a circuit under fixed common calibration and independent deterministic RNGs.

    ``progress(model, result)`` is invoked immediately after each model finishes,
    permitting an orchestrator to persist partial results. Trajectory minima and
    400 resamplings are enforced for reportable results. ``check_circuit=False``
    is for orchestrators that have already performed the same strict preflight;
    skipped verification is explicitly recorded in the returned metadata.
    """
    if pauli_trajectories < 2000 or hw_trajectories < 1000 or resamplings < 400:
        raise ValueError("Need pauli>=2000, hw>=1000 trajectories and >=400 resamplings")
    if shots != 10000:
        raise ValueError("Official scoring requires 10000 shots per basis")
    if not 1 <= n <= N // 2:
        raise ValueError("Expected 1 <= n <= N//2")
    if calibration is None:
        from team_eval.official_calibration import load_calibration
        calibration = load_calibration(seed)
    from team_eval.official_calibration import calibration_fingerprint
    circuit = (circuit_json if isinstance(circuit_json, cirq.Circuit)
               else cirq.read_json(str(circuit_json)))
    state_qubits = load_state_qubits(qubits, circuit)
    if not check_circuit and len(state_qubits) != N:
        raise ValueError(f"N={N}, but {len(state_qubits)} state qubits supplied")
    cz = calibration["cz_pauli_error_rate"]
    sq = calibration["sq_pauli_error_rate"]
    zero = calibration["zero_errors"]
    one = calibration["one_errors"]
    result = dict(N=N, n=n, calibration_seed=seed, K=N, shots_per_basis=shots,
                  resamplings=resamplings, backend=backend,
                  state_qubits=[[int(q.row), int(q.col)] if isinstance(q, cirq.GridQubit)
                                else str(q) for q in sorted(state_qubits)],
                  simulator="tools/sim.py:simulate", scorer="tools/sim.py:score_stats",
                  simulator_source_sha256=SIM_SOURCE_SHA256,
                  calibration_fingerprint=calibration_fingerprint(calibration),
                  simulator_proxy=True, models={})
    result["preflight"] = {"checked": bool(check_circuit),
                           "construction_time_checked": False}
    if check_circuit:
        # The construction worker owns the same acceptance checks used in the
        # head-to-head. Import here to avoid unnecessary/circular module imports.
        from team_eval.evaluate_worker import validate, exactness
        started = time.perf_counter()
        result["validity"], result["cz_count"] = validate(circuit, state_qubits, N, calibration)
        result["moments"] = len(circuit)
        if not result["validity"]["passed"]:
            result["exactness"] = {"passed": None, "skipped": True,
                                   "reason": "invalid_circuit"}
            result["preflight"].update(passed=False, seconds=time.perf_counter() - started)
            if output_path is not None:
                _write_result(output_path, result)
            raise ValueError("Circuit failed validity preflight: " +
                             "; ".join(result["validity"]["errors"]))
        try:
            result["exactness"] = exactness(circuit, state_qubits, N, n)
        except Exception as exc:
            result["exactness"] = {"passed": False, "exception": type(exc).__name__,
                                   "error": str(exc)}
            result["preflight"].update(passed=False, seconds=time.perf_counter() - started)
            if output_path is not None:
                _write_result(output_path, result)
            raise ValueError(f"Circuit exactness preflight failed: {exc}") from exc
        result["preflight"].update(passed=result["exactness"]["passed"],
                                   seconds=time.perf_counter() - started)
        if not result["exactness"]["passed"]:
            if output_path is not None:
                _write_result(output_path, result)
            raise ValueError("Circuit is not exact: fidelity " +
                             str(result["exactness"]["fidelity"]) +
                             " must be greater than " + str(result["exactness"]["threshold"]))
        if output_path is not None:
            _write_result(output_path, result)
    else:
        result["preflight"]["reason"] = "Caller disabled duplicate verification"
    probabilities = {}
    for model in models:
        if model not in ("pauli", "hw"):
            raise ValueError(f"Unknown noise model {model!r}")
        hw = sim.HW(T1=1e9, Tphi_white=0, T2star=0) if model == "pauli" else sim.HW()
        trajectory_seed = _rng_seed(seed, N, n, model, "trajectories")
        shot_seed = _rng_seed(seed, N, n, model, "resampling")
        trajectories = pauli_trajectories if model == "pauli" else hw_trajectories
        start = time.perf_counter()
        with simulation_backend(backend):
            pz, px, info = sim.simulate(circuit, state_qubits, cz, sq, zero, one,
                                       hw=hw, n_traj=trajectories,
                                       rng=np.random.default_rng(trajectory_seed))
        simulation_seconds = time.perf_counter() - start
        start = time.perf_counter()
        stats = sim.score_stats(pz, px, N, n, K=N, shots=shots, reps=resamplings,
                                rng=np.random.default_rng(shot_seed))
        model_result = dict(stats, n_traj=trajectories, simulation_seconds=simulation_seconds,
                            scoring_seconds=time.perf_counter() - start,
                            trajectory_seed=trajectory_seed, resampling_seed=shot_seed,
                            hw=asdict(hw), simulator_info=info,
                            pz_sum=float(pz.sum()), px_sum=float(px.sum()))
        result["models"][model] = model_result
        probabilities[model + "_pz"] = pz
        probabilities[model + "_px"] = px
        if probabilities_path is not None:
            path = _output_path(probabilities_path)
            with path.open("wb") as file:
                np.savez_compressed(file, **probabilities)
            result["probabilities_path"] = str(path)
        if output_path is not None:
            _write_result(output_path, result)
        if progress is not None:
            progress(model, model_result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit_json", type=Path, nargs="?")
    parser.add_argument("--circuit", type=Path, help="Alternative to positional circuit_json")
    parser.add_argument("--qubits", help="Qubit JSON path or inline [[row,col],...] (default: all circuit qubits)")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pauli-trajectories", type=int, default=2000)
    parser.add_argument("--hw-trajectories", type=int, default=1000)
    parser.add_argument("--resamplings", type=int, default=400)
    parser.add_argument("--models", nargs="+", choices=("pauli", "hw"), default=["pauli", "hw"])
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numba")
    parser.add_argument("--output", type=Path, help="JSON result path inside team_eval")
    parser.add_argument("--probabilities", type=Path, help="NPZ probability path inside team_eval")
    parser.add_argument("--skip-circuit-check", action="store_true",
                        help="Skip preflight only when the caller already verified this circuit")
    args = parser.parse_args()
    circuit_path = args.circuit or args.circuit_json
    if circuit_path is None:
        parser.error("Supply circuit_json or --circuit")
    result = score_circuit(circuit_path, args.qubits, args.N, args.n, args.seed,
                           pauli_trajectories=args.pauli_trajectories,
                           hw_trajectories=args.hw_trajectories,
                           resamplings=args.resamplings, models=tuple(args.models),
                           backend=args.backend, probabilities_path=args.probabilities,
                           output_path=args.output,
                           check_circuit=not args.skip_circuit_check,
                           progress=lambda model, row: print(
                               f"{model}: median={row['median']:.8g} mean={row['mean']:.8g} "
                               f"simulate={row['simulation_seconds']:.2f}s", file=sys.stderr, flush=True))
    payload = json.dumps(result, indent=2, allow_nan=False)
    print(payload)


if __name__ == "__main__":
    main()
