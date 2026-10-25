"""Three-level cost of a compiled region.

Logical moments are counted on the nearest-neighbor circuit before it is
rewritten to CZ. Hardware moments are counted after that rewrite. Parallel
children contribute the longer branch, not the sum.
"""

from __future__ import annotations

import cirq

from willow_dicke.circuit.lower import is_full_cz, lower_to_willow_cz
from willow_dicke.yuan_zhang_2025.compiler import CompiledRegion, Stage


def _cz_count(circuit: cirq.AbstractCircuit) -> int:
    return sum(1 for op in circuit.all_operations() if is_full_cz(op))


def _swap_count(circuit: cirq.AbstractCircuit) -> int:
    return sum(1 for op in circuit.all_operations() if op.gate == cirq.SWAP)


def _lower_len(circuit: cirq.AbstractCircuit) -> tuple[int, int]:
    if len(circuit) == 0:
        return 0, 0
    lowered = lower_to_willow_cz(circuit)
    return len(lowered), _cz_count(lowered)


def stage_critical_path(stage: Stage) -> dict:
    """Lower each sequential piece, then take the longer parallel child."""
    divide_moments, divide_cz = _lower_len(stage.divide)
    route_moments, route_cz = _lower_len(stage.route)
    leaf_moments, leaf_cz = _lower_len(stage.leaf)
    children = []
    child_moments = 0
    child_cz = 0
    if stage.parallel is not None:
        children = [stage_critical_path(child) for child in stage.parallel]
        child_moments = max(child["critical_moments"] for child in children)
        child_cz = max(child["critical_cz"] for child in children)
    parts = {
        "divide": divide_moments,
        "route": route_moments,
        "leaf": leaf_moments,
        "children": child_moments,
    }
    dominant = max(parts, key=parts.get)
    return {
        "name": stage.name,
        "divide_moments": divide_moments,
        "divide_cz": divide_cz,
        "route_moments": route_moments,
        "route_cz": route_cz,
        "leaf_moments": leaf_moments,
        "leaf_cz": leaf_cz,
        "child_moments": child_moments,
        "child_cz": child_cz,
        "critical_moments": divide_moments + route_moments + leaf_moments + child_moments,
        "critical_cz": divide_cz + route_cz + leaf_cz + child_cz,
        "dominant": dominant,
        "children": children,
        "logical_divide_moments": len(stage.divide),
        "logical_route_moments": len(stage.route),
        "logical_leaf_moments": len(stage.leaf),
        "swap_count": _swap_count(stage.route) + _swap_count(stage.divide) + _swap_count(stage.leaf),
    }


def count_divides(stage: Stage) -> int:
    here = 0 if len(stage.divide) == 0 else 1
    if stage.parallel is None:
        return here
    return here + sum(count_divides(child) for child in stage.parallel)


def metrics_for(compiled: CompiledRegion, lowered: cirq.Circuit) -> dict:
    path = stage_critical_path(compiled.stage)
    return {
        "logical": {
            "divide_blocks": count_divides(compiled.stage),
            "certificates": len(compiled.certificates),
            "swap_layers": sum(len(item.swap_layers) for item in compiled.certificates),
            "swap_count": sum(item.as_dict()["n_swaps"] for item in compiled.certificates),
        },
        "hardware": {
            "moments": len(lowered),
            "cz_count": _cz_count(lowered),
            "single_qubit_count": sum(
                1 for op in lowered.all_operations() if cirq.num_qubits(op) == 1
            ),
        },
        "critical_path": path,
        "dominant_at_root": path["dominant"],
    }
