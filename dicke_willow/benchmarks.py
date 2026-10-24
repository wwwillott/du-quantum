"""Depth and gate-count records for the two Dicke constructions."""

from __future__ import annotations

import json
from pathlib import Path

import cirq

from dicke_willow.compile import gate_counts, to_willow_gates
from dicke_willow.divide import prepare_balanced
from dicke_willow.scs import prepare_dicke_circuit


def logical_metrics(n: int, k: int) -> dict:
    """Compare the 2019 path circuit with the 2025 balanced circuit.

    Both numbers are taken after compilation to PhasedXZ+CZ and earliest-moment
    packing, on an abstract line (the balanced circuit may still contain
    long-range CZs). Routing depth is reported only for the path circuit,
    where every CZ is already nearest-neighbor, so routing depth is 0.
    """
    path = list(cirq.LineQubit.range(n))
    scs = to_willow_gates(prepare_dicke_circuit(path, k))
    balanced, region = prepare_balanced(path, k)
    balanced_native = to_willow_gates(balanced)
    scs_counts = gate_counts(scs)
    balanced_counts = gate_counts(balanced_native)
    return {
        "n": n,
        "k": k,
        "scs_moments": scs_counts["moments"],
        "scs_cz": scs_counts["cz"],
        "scs_routing_depth": 0,
        "balanced_moments": balanced_counts["moments"],
        "balanced_cz": balanced_counts["cz"],
        "balanced_region_depth": max(node.depth for node in region.walk()),
        "balanced_splits": sum(1 for node in region.walk() if node.m is not None),
    }


def write_benchmarks(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    header = list(rows[0])
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in header))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
