"""Command line for the 2025 rectangle compiler.

Simulation and device validation only. This module never submits a hardware job.
"""

from __future__ import annotations

import argparse
import json
import sys

from willow_dicke.device import load_static_willow
from willow_dicke.yuan_zhang_2025.benchmark import divide_benchmark, rectangle_benchmark
from willow_dicke.yuan_zhang_2025.compiler import compile_patch, with_unary
from willow_dicke.yuan_zhang_2025.geometry import abstract_patch, primary_willow_rectangle
from willow_dicke.yuan_zhang_2025.metrics import metrics_for
from willow_dicke.yuan_zhang_2025.simulator import dicke_fidelity, simulate
from willow_dicke.yuan_zhang_2025.verification import (
    assert_grid_cz,
    lower_willow,
    validate_willow_device,
)
from willow_dicke.yuan_zhang_2025.visualization import patch_svg, permutation_frames


def _patch(geometry: str, height: int, width: int):
    if geometry == "willow_rectangle":
        willow = load_static_willow()
        return primary_willow_rectangle(willow), willow
    if geometry == "rectangle":
        return abstract_patch(height, width), None
    raise SystemExit(f"unknown geometry {geometry}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2025 Yuan-Zhang Dicke compiler on a rectangle.")
    parser.add_argument("--n", type=int, default=56)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--geometry", default="willow_rectangle", choices=("willow_rectangle", "rectangle"))
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--divide", default="lnn_optimized", choices=("lnn_optimized",))
    parser.add_argument("--routing", default="structured", choices=("structured",))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="Statevector fidelity. Refused when n is large.")
    args = parser.parse_args(argv)

    height, width = args.height, args.width
    if args.geometry == "rectangle" and height * width != args.n:
        raise SystemExit("--height and --width must multiply to --n")
    patch, device = _patch(args.geometry, height, width)
    if patch.height * patch.width != args.n and args.geometry == "willow_rectangle" and args.n != 56:
        raise SystemExit(f"the Willow rectangle has {patch.height * patch.width} qubits, not {args.n}")

    compiled = compile_patch(patch, args.k, device_graph=device)
    prepared = with_unary(compiled.circuit, compiled.port, args.k)
    lowered = lower_willow(prepared)
    assert_grid_cz(lowered)
    if args.geometry == "willow_rectangle":
        validate_willow_device(lowered)
    report = {
        "n": patch.height * patch.width,
        "k": args.k,
        "geometry": patch.name,
        "moments": len(lowered),
        "port": [[qubit.row, qubit.col] for qubit in compiled.port],
        "kind": compiled.tree.kind,
        "dominant": metrics_for(compiled, lowered)["dominant_at_root"],
        "device_validated": args.geometry == "willow_rectangle",
    }
    if args.simulate:
        vector = simulate(lowered, list(patch.qubits))
        report["fidelity"] = dicke_fidelity(vector, args.k)
    if args.benchmark:
        if args.geometry == "willow_rectangle":
            saved = rectangle_benchmark(patch.height, patch.width, args.k, willow=True)
        else:
            saved = rectangle_benchmark(patch.height, patch.width, args.k, willow=False)
        report["artifact"] = saved["artifact"]
        report["metrics"] = saved["metrics"]
        report["baselines"] = saved["baselines"]
    svg_path = patch_svg(patch, compiled)
    frames = permutation_frames(compiled)
    from pathlib import Path

    out = Path(__file__).resolve().parent / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"patch_k{args.k}.svg").write_text(svg_path, encoding="utf-8")
    (out / f"frames_k{args.k}.html").write_text(frames, encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
