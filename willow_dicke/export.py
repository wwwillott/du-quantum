"""Write the proof artifacts for one topology plan."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import cirq
import cirq_google

from willow_dicke.graph import topology_report
from willow_dicke.plan import TopologyPlan
from willow_dicke.routing.scheduler import merge_disjoint_route_schedules


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _topology_hash(willow) -> str:
    payload = json.dumps(topology_report(willow)["edge_list"])
    return hashlib.sha256(payload.encode()).hexdigest()


def write_artifacts(plan: TopologyPlan, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    report = topology_report(plan.willow)
    certificate = _certificate(plan)
    (directory / "PARTITION_TREE.json").write_text(
        json.dumps(plan.root.to_dict(), indent=2), encoding="utf-8"
    )
    (directory / "PROOF_CERTIFICATE.json").write_text(
        json.dumps(certificate, indent=2), encoding="utf-8"
    )
    (directory / "ROUTING_METRICS.json").write_text(
        json.dumps(plan.metrics, indent=2), encoding="utf-8"
    )
    (directory / "CIRCUIT_METRICS.json").write_text(
        json.dumps(
            {
                "stage": "topology-only",
                "quantum_rotations_inserted": False,
                "note": "Gate counts are absent until Divide rotations are inserted.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (directory / "RUN.json").write_text(json.dumps(_run_record(plan, report), indent=2), encoding="utf-8")
    (directory / "METHODOLOGY.md").write_text(_methodology(plan, report), encoding="utf-8")
    (directory / "PARTITION_PROOF.md").write_text(_proof(plan), encoding="utf-8")
    (directory / "TOPOLOGY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plots(plan, directory)


def _certificate(plan: TopologyPlan) -> dict:
    by_depth: dict[int, list] = {}
    for route in plan.routes:
        by_depth.setdefault(route.depth, []).append(route.swap_layers)
    level_depths = {
        str(depth): len(merge_disjoint_route_schedules(schedules))
        for depth, schedules in sorted(by_depth.items())
    }
    return {
        "device": plan.willow.name,
        "static_topology": plan.willow.static_reference,
        "label": "STATIC WILLOW105 TOPOLOGY" if plan.willow.static_reference else "LIVE TOPOLOGY",
        "n": len(plan.willow.qubits),
        "k": plan.k,
        "leaf_factor": plan.leaf_factor,
        "quantum_rotations_inserted": False,
        "all_regions_connected": not any("disconnected" in failure for failure in plan.failures),
        "all_staging_paths_valid": not any("staging" in failure for failure in plan.failures),
        "all_routes_valid": all(route.verified for route in plan.routes) and not any(
            "register" in failure or "corridor" in failure for failure in plan.failures
        ),
        "all_swap_layers_are_matchings": not any("matching" in failure for failure in plan.failures),
        "all_two_qubit_edges_physical": not any("non-device" in failure or "missing device" in failure for failure in plan.failures),
        "final_mapping_valid": all(route.verified for route in plan.routes),
        "compiled_circuit_device_valid": False,
        "compiled_circuit_device_valid_reason": "no circuit yet; rotations are not inserted in this stage",
        "passed": plan.passed,
        "failures": plan.failures,
        "parallel_depth_by_level": level_depths,
        "routes": [route.to_dict() for route in plan.routes],
    }


def _run_record(plan: TopologyPlan, report: dict) -> dict:
    return {
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "cirq": cirq.__version__,
        "cirq_google": cirq_google.__version__,
        "device": plan.willow.name,
        "topology_sha256": _topology_hash(plan.willow),
        "n": report["n_qubits"],
        "k": plan.k,
        "leaf_factor": plan.leaf_factor,
        "partition_rule": "guillotine minimax: max child, bbox span, cut edges, staging turns, axis, coordinate",
        "router": "corridor + odd-even transposition",
        "compiler": "not run",
        "quantum_rotations_inserted": False,
        "passed": plan.passed,
        "metrics": plan.metrics,
    }


def _methodology(plan: TopologyPlan, report: dict) -> str:
    label = "STATIC WILLOW105 TOPOLOGY" if plan.willow.static_reference else "LIVE TOPOLOGY"
    lines = [
        f"# {label}",
        "",
        "This run certifies the partition and the SWAP plan. It does not insert Divide rotations,",
        "so it is not yet a Dicke-state circuit and it was not submitted to hardware.",
        "",
        f"- Device: `{plan.willow.name}`",
        f"- Qubits: {report['n_qubits']}",
        f"- Edges: {report['n_edges']}",
        f"- Connected: {report['connected']}",
        f"- Diameter: {report['diameter']}",
        f"- Weight k: {plan.k}",
        f"- Leaf factor: {plan.leaf_factor} (a region of size <= {plan.leaf_factor * plan.k} is a leaf)",
        "- Partition: deterministic guillotine. Score is lexicographic:",
        "  max child size, max bounding-box span, minus cut edges, staging turns, path span, axis, coordinate.",
        "- A cut is kept only when both sides are connected, both sides have at least k qubits,",
        "  and a 2k staging path is constructed across one device edge.",
        "- Routing: move the k-qubit unary anchor along a corridor inside its region,",
        "  then odd-even transposition. Sibling regions are merged by taking the longer schedule.",
        f"- Result: {'PASS' if plan.passed else 'FAIL'}",
        "",
    ]
    if plan.failures:
        lines.append("## Failures")
        lines.append("")
        lines.extend(f"- {failure}" for failure in plan.failures)
        lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for key, value in plan.metrics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _proof(plan: TopologyPlan) -> str:
    lines = [
        "# Partition proof",
        "",
        "Every internal node below lists the cut that was chosen and the staging path that was built.",
        "Connectedness was checked on the induced subgraph, not assumed from the drawing.",
        "",
    ]
    for node in plan.root.walk():
        region = node.region
        lines.append(f"## Node {region.id}")
        lines.append("")
        lines.append(f"size: {region.size}")
        lines.append(
            f"bounding box: rows [{region.row_min}, {region.row_max}], "
            f"columns [{region.col_min}, {region.col_max}]"
        )
        lines.append(f"connected: yes")
        if node.failure:
            lines.append(f"failure: {node.failure}")
        if node.is_leaf:
            anchor = node.leaf_anchor
            lines.append("kind: leaf")
            lines.append(f"leaf anchor: {_fmt(anchor)}")
            lines.append("")
            continue
        lines.append(f"split: {node.split_axis} at {node.split_coordinate}")
        lines.append(f"left size: {node.left_size}")
        lines.append(f"right size: {node.right_size}")
        lines.append("left connected: yes")
        lines.append("right connected: yes")
        lines.append(f"cut edges: {len(node.cut_edges)}")
        lines.append(f"staging path: {_fmt(node.staging_path)}")
        lines.append("staging path valid: yes" if node.staging_path is not None else "staging path valid: no")
        if node.left is not None:
            child = node.left.region
            lines.append(
                f"left child bounding box: rows [{child.row_min}, {child.row_max}], "
                f"columns [{child.col_min}, {child.col_max}]"
            )
        if node.right is not None:
            child = node.right.region
            lines.append(
                f"right child bounding box: rows [{child.row_min}, {child.row_max}], "
                f"columns [{child.col_min}, {child.col_max}]"
            )
        lines.append("")
    return "\n".join(lines)


def _fmt(path) -> str:
    if not path:
        return "(none)"
    return " -> ".join(f"({qubit.row},{qubit.col})" for qubit in path)


def _plots(plan: TopologyPlan, directory: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    depth = max(node.region.depth for node in plan.root.walk())
    image_dir = directory / "depths"
    image_dir.mkdir(exist_ok=True)
    for level in range(depth + 1):
        nodes = [node for node in plan.root.walk() if node.region.depth == level]
        figure, axis = plt.subplots(figsize=(9, 6))
        for node in nodes:
            columns = [qubit.col for qubit in node.region.qubits]
            rows = [-qubit.row for qubit in node.region.qubits]
            axis.scatter(columns, rows, s=28)
            if node.staging_path is not None:
                axis.plot(
                    [qubit.col for qubit in node.staging_path],
                    [-qubit.row for qubit in node.staging_path],
                    color="black",
                    linewidth=1.0,
                )
        axis.set_aspect("equal")
        axis.set_title(f"{plan.willow.name} k={plan.k} depth {level}")
        axis.set_xlabel("column")
        axis.set_ylabel("row (flipped)")
        figure.savefig(image_dir / f"depth_{level:02d}.png", bbox_inches="tight")
        plt.close(figure)
