"""Challenge noise scores for the 2025 balanced Dicke circuit.

``benchmarks/noisy_score.csv`` scores the 2019 stair on the Willow snake.
This script scores Yuan and Zhang's balanced recursion on that same snake,
with the same timing and the same error model:

- a moment costs 42 ns if it holds a CZ, otherwise 25 ns
- ``prepare_us`` is the sum of those moments
- ``idle_us`` is the average, over the path, of time a qubit is not in a gate
- ``gate_visibility = exp(-e1 * (SQ gates per qubit) - e2 * (CZ gates per qubit))``
- ``p_decay = 1 - exp(-idle_us / 68)`` with T1 = 68 us
- ``p_flip_z = (1 - gate_visibility) / 2``
- ``p_flip_x = (1 - gate_visibility * exp(-idle_us / 136)) / 2`` with T2 = 136 us

``e1`` and ``e2`` are the unique per-qubit Pauli rates that reproduce the
``n=2, k=1`` and ``n=3, k=1`` visibilities in the 2019 table. Z samples are
ideal Dicke strings, then amplitude damping and bit flips. X samples are
ideal X-basis Dicke strings, then bit flips at ``p_flip_x``. Both use 10000
shots, and the score is the notebook's ``overall_score``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import cirq
import cirq_google
import networkx as nx
import numpy as np

from dicke_willow.challenge import create_dicke_state_adaptive
from dicke_willow.compile import finalize, to_willow_gates
from dicke_willow.divide import prepare_balanced
from dicke_willow.placement import snake_path
from dicke_willow.topology import graph_from_device

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks" / "noisy_score.csv"
OUTPUT = ROOT / "benchmarks" / "noisy_score_2025.csv"

SQ_NS = 25
CZ_NS = 42
T1_US = 68.0
T2_US = 136.0
SHOTS = 10_000
COLUMNS = (
    "n",
    "k",
    "moments",
    "prepare_us",
    "idle_us",
    "gate_visibility",
    "p_decay",
    "p_flip_z",
    "p_flip_x",
    "z_subspace_fraction",
    "z_penalty",
    "x_penalty",
    "overall_score",
    "noiseless_overall_score",
    "shots",
    "max_correlator_size",
)


def snake(n: int) -> list[cirq.GridQubit]:
    path = snake_path(graph_from_device(cirq_google.Willow105), n)
    if path is None or len(path) != n:
        raise RuntimeError(f"Willow has no snake of length {n}")
    return list(path)


def compile_2025(path: list[cirq.Qid], k: int) -> cirq.Circuit:
    """Balanced 2025 unitary, routed along ``path`` like the 2019 stair."""
    logical, _region = prepare_balanced(list(path), k)
    native = to_willow_gates(logical)
    graph = nx.Graph()
    graph.add_nodes_from(path)
    graph.add_edges_from(zip(path, path[1:]))
    routed = cirq.RouteCQC(graph)(native)
    return finalize(routed, decouple=False)


def moment_ns(moment: cirq.Moment) -> int:
    if any(isinstance(op.gate, cirq.CZPowGate) for op in moment):
        return CZ_NS
    return SQ_NS


def schedule(circuit: cirq.AbstractCircuit) -> tuple[float, float, float, float]:
    """Return prepare_us, idle_us, mean SQ count, mean CZ count."""
    qubits = list(circuit.all_qubits())
    durations = [moment_ns(moment) for moment in circuit]
    total = sum(durations)
    sq = {qubit: 0 for qubit in qubits}
    cz = {qubit: 0 for qubit in qubits}
    busy = {qubit: 0 for qubit in qubits}
    for moment, duration in zip(circuit, durations):
        touched = set()
        for op in moment:
            touched.update(op.qubits)
            if isinstance(op.gate, cirq.CZPowGate):
                for qubit in op.qubits:
                    cz[qubit] += 1
            else:
                sq[op.qubits[0]] += 1
        for qubit in touched:
            busy[qubit] += duration
    idle = [total - busy[qubit] for qubit in qubits]
    return (
        total / 1000.0,
        (sum(idle) / len(idle)) / 1000.0 if idle else 0.0,
        sum(sq.values()) / len(qubits),
        sum(cz.values()) / len(qubits),
    )


def fit_pauli_rates() -> tuple[float, float]:
    """Rates that reproduce the two shortest rows of the 2019 score file."""
    reference = {
        (int(row["n"]), int(row["k"])): float(row["gate_visibility"])
        for row in csv.DictReader(REFERENCE.open(encoding="utf-8"))
    }
    matrix = []
    target = []
    for key in ((2, 1), (3, 1)):
        circuit, _qubits = create_dicke_state_adaptive(*key)
        _prepare, _idle, mean_sq, mean_cz = schedule(circuit)
        matrix.append([mean_sq, mean_cz])
        target.append(-math.log(reference[key]))
    e1, e2 = np.linalg.solve(np.array(matrix, dtype=float), np.array(target))
    return float(e1), float(e2)


def noise_parameters(circuit: cirq.AbstractCircuit, e1: float, e2: float) -> dict:
    prepare, idle, mean_sq, mean_cz = schedule(circuit)
    visibility = math.exp(-(e1 * mean_sq + e2 * mean_cz))
    return {
        "moments": len(circuit),
        "prepare_us": prepare,
        "idle_us": idle,
        "gate_visibility": visibility,
        "p_decay": 1.0 - math.exp(-idle / T1_US),
        "p_flip_z": (1.0 - visibility) / 2.0,
        "p_flip_x": (1.0 - visibility * math.exp(-idle / T2_US)) / 2.0,
    }


def _ideal_z(n: int, k: int, shots: int, rng: np.random.Generator) -> np.ndarray:
    bits = np.zeros((shots, n), dtype=np.int8)
    for shot in range(shots):
        bits[shot, rng.choice(n, size=k, replace=False)] = 1
    return bits


def _ideal_x(n: int, k: int, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Computational-basis samples of the all-Hadamard Dicke state."""
    state = np.zeros(1 << n, dtype=np.float64)
    amplitude = 1.0 / math.sqrt(math.comb(n, k))
    for chosen in combinations(range(n), k):
        mask = 0
        for qubit in chosen:
            mask |= 1 << qubit
        state[mask] = amplitude
    step = 1
    while step < (1 << n):
        for start in range(0, 1 << n, step * 2):
            left = state[start : start + step].copy()
            right = state[start + step : start + 2 * step].copy()
            state[start : start + step] = left + right
            state[start + step : start + 2 * step] = left - right
        step *= 2
    probabilities = state * state / (1 << n)
    probabilities /= probabilities.sum()
    masks = rng.choice(1 << n, size=shots, p=probabilities)
    bits = np.zeros((shots, n), dtype=np.int8)
    for qubit in range(n):
        bits[:, qubit] = (masks >> qubit) & 1
    return bits


def _dampen_and_flip(
    bits: np.ndarray, p_decay: float, p_flip: float, rng: np.random.Generator
) -> np.ndarray:
    noisy = bits.copy()
    if p_decay:
        decay = (rng.random(noisy.shape) < p_decay) & (noisy == 1)
        noisy[decay] = 0
    if p_flip:
        noisy ^= rng.random(noisy.shape) < p_flip
    return noisy


def _flip(bits: np.ndarray, p_flip: float, rng: np.random.Generator) -> np.ndarray:
    if not p_flip:
        return bits.copy()
    return bits ^ (rng.random(bits.shape) < p_flip)


def z_penalty(bitstrings: np.ndarray, k: int) -> tuple[float, float]:
    shots, n = bitstrings.shape
    weight = bitstrings.sum(axis=1)
    kept = weight == k
    fraction = float(kept.mean())
    if fraction == 0.0:
        return fraction, math.inf
    ideal = 1.0 / math.comb(n, k)
    expected = shots * fraction * ideal
    packed = np.zeros(shots, dtype=np.int64)
    for qubit in range(n):
        packed |= bitstrings[:, qubit].astype(np.int64) << qubit
    _values, counts = np.unique(packed[kept], return_counts=True)
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    chi2 += (math.comb(n, k) - len(counts)) * expected
    return fraction, ideal * chi2 / fraction


def x_penalty(bitstrings: np.ndarray, k: int, max_size: int) -> float:
    _shots, n = bitstrings.shape
    score = 0.0
    bits = bitstrings.astype(np.int8, copy=False)
    for size in range(1, max_size + 1):
        half = size // 2
        if size % 2 == 0 and 0 <= k - half <= n - size:
            expected = (
                math.comb(size, half)
                * math.comb(n - size, k - half)
                / math.comb(n, k)
            )
        else:
            expected = 0.0
        total = 0.0
        number = 0
        for indices in combinations(range(n), size):
            parity = np.bitwise_xor.reduce(bits[:, indices], axis=1)
            total += 1.0 - 2.0 * float(parity.mean())
            number += 1
        measured = total / number
        if expected != 0.0:
            score += (expected - measured) ** 2 / expected**2
        else:
            score += (expected - measured) ** 2
    return score


def score_samples(
    n: int,
    k: int,
    parameters: dict,
    rng: np.random.Generator,
) -> dict:
    max_size = min(4, n)
    z_ideal = _ideal_z(n, k, SHOTS, rng)
    x_ideal = _ideal_x(n, k, SHOTS, rng)
    z_bits = _dampen_and_flip(z_ideal, parameters["p_decay"], parameters["p_flip_z"], rng)
    x_bits = _flip(x_ideal, parameters["p_flip_x"], rng)
    fraction, z_pen = z_penalty(z_bits, k)
    x_pen = x_penalty(x_bits, k, max_size)
    _fraction, z_clean = z_penalty(z_ideal, k)
    x_clean = x_penalty(x_ideal, k, max_size)
    noisy = _invert(z_pen + x_pen)
    clean = _invert(z_clean + x_clean)
    return {
        "z_subspace_fraction": fraction,
        "z_penalty": z_pen,
        "x_penalty": x_pen,
        "overall_score": noisy,
        "noiseless_overall_score": clean,
        "shots": SHOTS,
        "max_correlator_size": max_size,
    }


def _invert(total: float) -> float:
    if not math.isfinite(total) or total == 0.0:
        return 0.0 if total != 0.0 else math.inf
    return 1.0 / total


def check_reference(e1: float, e2: float) -> None:
    """The fitted rates must rebuild the 2019 visibility and decay columns."""
    reference = {
        (int(row["n"]), int(row["k"])): row
        for row in csv.DictReader(REFERENCE.open(encoding="utf-8"))
    }
    for key in ((2, 1), (4, 2), (6, 1), (8, 1)):
        circuit, _qubits = create_dicke_state_adaptive(*key)
        got = noise_parameters(circuit, e1, e2)
        row = reference[key]
        for name in ("prepare_us", "idle_us", "gate_visibility", "p_decay", "p_flip_z", "p_flip_x"):
            published = float(row[name])
            if abs(got[name] - published) > 5e-7:
                raise AssertionError(f"2019 {key} {name}: {got[name]} != {published}")
        if got["moments"] != int(row["moments"]):
            raise AssertionError(f"2019 {key} moments {got['moments']} != {row['moments']}")


def write_row(handle, row: dict) -> None:
    handle.write(",".join(_format(row[name]) for name in COLUMNS) + "\n")
    handle.flush()


def _format(value) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def score_pair(n: int, k: int, e1: float, e2: float) -> dict:
    path = snake(n)
    circuit = compile_2025(path, k)
    parameters = noise_parameters(circuit, e1, e2)
    rng = np.random.default_rng(10_000 * n + k)
    scored = score_samples(n, k, parameters, rng)
    return {"n": n, "k": k, **parameters, **scored}


def _done_keys(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {(int(row["n"]), int(row["k"])) for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    e1, e2 = fit_pauli_rates()
    print(f"pauli rates e1={e1:.8e} e2={e2:.8e}", flush=True)
    check_reference(e1, e2)
    print("2019 timing and visibility columns match noisy_score.csv", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(args.output) if args.resume else set()
    if not args.resume or not args.output.exists():
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            handle.write(",".join(COLUMNS) + "\n")
        done = set()
    pairs = [
        (n, k)
        for n in range(2, args.max_n + 1)
        for k in range(1, n // 2 + 1)
        if (n, k) not in done
    ]
    print(f"scoring {len(pairs)} circuits with {args.workers} workers", flush=True)
    with args.output.open("a", encoding="utf-8", newline="") as handle:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(score_pair, n, k, e1, e2): (n, k) for n, k in pairs}
            for future in as_completed(futures):
                n, k = futures[future]
                row = future.result()
                write_row(handle, row)
                print(
                    f"n={n:2} k={k:2} moments={row['moments']:5} "
                    f"score={row['overall_score']:.6g} "
                    f"noiseless={row['noiseless_overall_score']:.6g}",
                    flush=True,
                )
    rows = list(csv.DictReader(args.output.open(encoding="utf-8")))
    rows.sort(key=lambda row: (int(row["n"]), int(row["k"])))
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(COLUMNS) + "\n")
        for row in rows:
            handle.write(",".join(row[name] for name in COLUMNS) + "\n")


if __name__ == "__main__":
    main()
