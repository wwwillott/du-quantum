"""Command line for the Willow topology certificate.

    python -m willow_dicke --n 105 --k 3 --device static-willow --router custom --verify --export artifacts/run_k3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from willow_dicke.device import diff_against_static, load_live_device, load_static_willow
from willow_dicke.export import write_artifacts
from willow_dicke.plan import build_topology_plan


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify a Willow Dicke routing plan.")
    parser.add_argument("--n", type=int, default=105)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--device", choices=("static-willow", "live"), default="static-willow")
    parser.add_argument("--processor-id", default="willow_pink")
    parser.add_argument("--router", choices=("custom", "cirq"), default="custom")
    parser.add_argument("--leaf-factor", type=int, default=2)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument(
        "--execute-hardware",
        action="store_true",
        help="Refused. Topology inspection never submits a job.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.execute_hardware:
        print("refusing to submit a hardware job; --execute-hardware is not implemented")
        return 2
    if args.router == "cirq":
        print("RouteCQC is a later baseline. This stage only builds the custom corridor router.")
        return 2
    if args.device == "static-willow":
        willow = load_static_willow()
    else:
        willow = load_live_device(args.processor_id)
        diff = diff_against_static(willow)
        print(json.dumps({key: diff[key] for key in diff if key != "same_topology"}))
        print(f"same topology as Willow105: {diff['same_topology']}")
        if not diff["same_topology"]:
            print("live graph differs from static Willow105; the static-105 theorem is not claimed")
    if args.n != len(willow.qubits):
        print(f"--n {args.n} does not match the device, which has {len(willow.qubits)} qubits")
        return 2
    plan = build_topology_plan(willow, args.k, leaf_factor=args.leaf_factor)
    print(json.dumps(plan.metrics, indent=2))
    if plan.failures:
        print("failures:")
        for failure in plan.failures:
            print(f"  {failure}")
    if args.export is not None:
        write_artifacts(plan, args.export)
        print(f"wrote {args.export}")
    if args.verify and not plan.passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
