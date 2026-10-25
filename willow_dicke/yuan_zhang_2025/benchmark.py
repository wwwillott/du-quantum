"""Reproducible Divide and rectangle benchmarks.

Each call writes a new JSON file. Nothing in ``artifacts/`` is overwritten.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cirq
import numpy as np

from willow_dicke.circuit.lower import expand_circuit, bridge_along_path, is_full_cz, lower_to_willow_cz
from willow_dicke.device import load_static_willow
from willow_dicke.divide.logical_divide import build_logical_divide
from willow_dicke.yuan_zhang_2025 import METHODOLOGY_VERSION
from willow_dicke.yuan_zhang_2025.compiler import compile_patch, with_unary
from willow_dicke.yuan_zhang_2025.config import ALGORITHM, PAPER
from willow_dicke.yuan_zhang_2025.divide_lnn import compile_divide_lnn
from willow_dicke.yuan_zhang_2025.geometry import abstract_patch, primary_willow_rectangle
from willow_dicke.yuan_zhang_2025.metrics import metrics_for
from willow_dicke.yuan_zhang_2025.verification import assert_grid_cz, lower_willow, verify_divide_equivalence

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write(name: str, payload: dict) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{name}_{_stamp()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _cz(circuit) -> int:
    return sum(1 for op in circuit.all_operations() if is_full_cz(op))


def _swaps(circuit) -> int:
    return sum(1 for op in circuit.all_operations() if op.gate == cirq.SWAP)


def divide_benchmark(ks: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    """Isolate one Divide. Equivalence is required before the speedup is recorded."""
    rows = []
    for k in ks:
        n, m = max(4 * k, 2 * k), max(2 * k, k)
        # Both children have at least k qubits. n = 4k, m = 2k is the balanced split.
        n, m = 4 * k, 2 * k
        path = cirq.LineQubit.range(2 * k)
        equivalence = verify_divide_equivalence(n, m, k)
        logical = build_logical_divide(n, m, k, path)
        reference = lower_to_willow_cz(bridge_along_path(expand_circuit(logical), path))
        optimized = compile_divide_lnn(n, m, k, path)
        lowered = lower_to_willow_cz(optimized)
        rows.append(
            {
                "k": k,
                "n": n,
                "m": m,
                "equivalence_passed": equivalence["passed"],
                "reference": {
                    "moments": len(reference),
                    "cz_count": _cz(reference),
                    "swap_count": _swaps(bridge_along_path(expand_circuit(logical), path)),
                },
                "lnn_optimized": {
                    "logical_moments": len(optimized),
                    "moments": len(lowered),
                    "cz_count": _cz(lowered),
                    "swap_count": _swaps(optimized),
                },
            }
        )
    payload = _envelope("divide_isolation", {"comparisons": rows})
    payload["artifact"] = str(_write("divide_isolation", payload))
    return payload


def scs_baseline_moments(patch, k: int) -> dict:
    """2019 SCS on the rectangle's snake, lowered with the same CZ pass."""
    from dicke_willow.scs import prepare_dicke_ops
    from willow_dicke.yuan_zhang_2025.scheduler import pack

    path = list(patch.rectangle().snake())
    lowered = lower_to_willow_cz(pack(prepare_dicke_ops(path, k)))
    return {"moments": len(lowered), "cz_count": _cz(lowered), "backend": "scs_2019_snake"}


def rectangle_benchmark(height: int, width: int, k: int, *, willow: bool = False) -> dict:
    device = load_static_willow() if willow else None
    if willow:
        patch = primary_willow_rectangle(device)
        height, width = patch.height, patch.width
    else:
        patch = abstract_patch(height, width)
    started = time.perf_counter()
    compiled = compile_patch(patch, k, device_graph=device)
    lowered = lower_willow(with_unary(compiled.circuit, compiled.port, k))
    assert_grid_cz(lowered)
    elapsed = time.perf_counter() - started
    metrics = metrics_for(compiled, lowered)
    baseline_a = scs_baseline_moments(patch, k)
    payload = _envelope(
        "rectangle",
        {
            "geometry": {
                "name": patch.name,
                "height": patch.height,
                "width": patch.width,
                "qubits": len(patch.qubits),
                "physical_rows": list(patch.rows),
                "physical_cols": list(patch.cols),
                "graph_hash": patch.graph_hash(),
            },
            "target": {"n": patch.height * patch.width, "k": k},
            "implementation": {
                "divide_backend": "lnn_optimized",
                "routing_backend": "structured_permutation",
            },
            "metrics": metrics,
            "baselines": {"scs_2019_snake": baseline_a},
            "validation": {
                "cz_on_grid_edges": True,
                "permutation_certificates": len(compiled.certificates),
                "permutations_correct": all(item.correct_permutation for item in compiled.certificates),
                "compile_seconds": elapsed,
            },
            "port": [[qubit.row, qubit.col] for qubit in compiled.port],
            "tree_kind": compiled.tree.kind,
        },
    )
    payload["artifact"] = str(_write(f"rectangle_{patch.name}_k{k}", payload))
    return payload


def _envelope(kind: str, body: dict) -> dict:
    payload = {
        "algorithm": ALGORITHM,
        "paper": PAPER,
        "kind": kind,
        "methodology_version": METHODOLOGY_VERSION,
        "git_commit": _git_commit(),
        "packages": {
            "python": platform.python_version(),
            "cirq": cirq.__version__,
            "numpy": np.__version__,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(body)
    return payload
