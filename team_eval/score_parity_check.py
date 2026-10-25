"""Compare tools.sim's count scorer directly with the notebook's unmodified scorer.

Run with C:/Users/kaisa/miniforge3/python.exe; artifacts stay under team_eval/.
"""
import argparse
import ast
import itertools
import json
import math
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from team_eval.official_calibration import (
    calibration_fingerprint, load_calibration, seeded_calibration, write_calibration,
)
from tools.sim import dicke_vec, score_from_counts


def notebook_functions():
    notebook = json.loads((ROOT / "DuQuantum_Google_Standard.ipynb").read_text(encoding="utf-8"))
    scorer_cell = next(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if "def overall_score(" in "".join(cell.get("source", []))
    )
    namespace = dict(np=np, math=math, itertools=itertools)
    exec(compile(scorer_cell, "DuQuantum_Google_Standard.ipynb#scorer", "exec"), namespace)
    return notebook, namespace


def check_generator_source(notebook):
    source = (ROOT / "team_eval/official_calibration.py").read_text(encoding="utf-8")
    notebook_source = next(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if "def pick_single_qubit_dropouts(" in "".join(cell.get("source", []))
    )
    # Direct substring test verifies the source, including comments and whitespace.
    return notebook_source in source


def fingerprints():
    result = {}
    for seed in (101, 202):
        path = write_calibration(seed)
        fresh = seeded_calibration(seed)
        saved = load_calibration(seed)
        result[str(seed)] = dict(
            path=str(path.relative_to(ROOT)),
            fresh_sha256=calibration_fingerprint(fresh),
            saved_sha256=calibration_fingerprint(saved),
            equal=calibration_fingerprint(fresh) == calibration_fingerprint(saved),
            n_qubits=len(saved["sq_pauli_error_rate"]),
            n_couplers=len(saved["cz_pauli_error_rate"]),
        )
    return result


def _hadamard_probabilities(N, n):
    psi = dicke_vec(N, n).reshape((2,) * N)
    h = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    for axis in range(N):
        psi = np.moveaxis(np.tensordot(h, psi, axes=([1], [axis])), 0, axis)
    return np.abs(psi.reshape(-1)) ** 2


def parity_cases(official, shots=10000):
    rng = np.random.default_rng(830172)
    rows = []
    for N, n in ((2, 1), (3, 1), (4, 2), (6, 2), (8, 4), (12, 6)):
        dimension = 2**N
        bit_table = ((np.arange(dimension)[:, None] >> (N - 1 - np.arange(N))) & 1).astype(np.int8)
        ideal_z = dicke_vec(N, n) ** 2
        ideal_x = _hadamard_probabilities(N, n)
        for distribution in ("ideal_dicke", "noisy_nonuniform"):
            if distribution == "ideal_dicke":
                pz, px = ideal_z.copy(), ideal_x.copy()
            else:
                # Include both shell nonuniformity and leakage, and distorted X probabilities.
                pz = 0.8 * ideal_z * rng.lognormal(0, 0.12, dimension) + 0.2 / dimension
                px = 0.85 * ideal_x * rng.lognormal(0, 0.12, dimension) + 0.15 / dimension
            pz /= pz.sum()
            px /= px.sum()
            cz = rng.multinomial(shots, pz)
            cx = rng.multinomial(shots, px)
            z_bits = np.repeat(bit_table, cz, axis=0)
            x_bits = np.repeat(bit_table, cx, axis=0)
            fast_score, fast_z, fast_x, fraction = score_from_counts(cz, cx, N, n, N)
            official_z = official["check_z_bitstring_distribution"](z_bits, n)
            official_x = official["check_x_bitstring_distribution"](x_bits, n, N)
            official_score = official["overall_score"](z_bits, x_bits, n, N)
            actual = np.array([fast_score, fast_z, fast_x])
            reference = np.array([official_score, official_z, official_x])
            rows.append(dict(
                N=N, n=n, K=N, shots=shots, distribution=distribution,
                fast_score=float(fast_score), official_score=float(official_score),
                fast_Z=float(fast_z), official_Z=float(official_z),
                fast_X=float(fast_x), official_X=float(official_x),
                shell_fraction=float(fraction),
                max_abs_difference=float(np.max(np.abs(actual - reference))),
                passed=bool(np.allclose(actual, reference, atol=1e-11, rtol=1e-10)),
            ))
    # The official formula is undefined at zero shell retention, while the fast
    # implementation returns the sensible zero-score limiting convention.
    N, n = 4, 2
    cz = np.zeros(2**N, dtype=np.int64)
    cx = np.zeros_like(cz)
    cz[0] = cx[0] = shots
    z_bits = x_bits = np.zeros((shots, N), dtype=np.int8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        official_zero = official["overall_score"](z_bits, x_bits, n, N)
    fast_zero = score_from_counts(cz, cx, N, n, N)[0]
    edge = dict(
        case="zero_correct_weight_z_shots", N=N, n=n, K=N,
        official_score="NaN" if np.isnan(official_zero) else float(official_zero),
        fast_score=float(fast_zero),
        note="The official notebook divides by zero; tools.sim returns zero by convention."
    )
    return rows, edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprints-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "team_eval/score_parity_results.json")
    args = parser.parse_args()
    result = dict(python=sys.executable, pythonhashseed=os.environ.get("PYTHONHASHSEED"), calibrations=fingerprints())
    if not args.fingerprints_only:
        notebook, official = notebook_functions()
        result["generators_verbatim"] = check_generator_source(notebook)
        result["cases"], result["edge_case"] = parity_cases(official)
        result["all_sampled_cases_pass"] = all(row["passed"] for row in result["cases"])
        result["maximum_absolute_difference"] = max(row["max_abs_difference"] for row in result["cases"])
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not all(row["equal"] for row in result["calibrations"].values()):
        raise SystemExit("Fresh and saved calibrations differ")
    if not args.fingerprints_only and not (result["all_sampled_cases_pass"] and result["generators_verbatim"]):
        raise SystemExit("Scorer parity or verbatim calibration check failed")


if __name__ == "__main__":
    main()
