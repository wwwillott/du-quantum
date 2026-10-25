"""Merge SWAP schedules that live on disjoint regions."""

from __future__ import annotations

from willow_dicke.routing.path_permutation import SwapLayer


def merge_disjoint_route_schedules(schedules: list[list[SwapLayer]]) -> list[SwapLayer]:
    """Zip schedules. Depth is the longest schedule, not the sum."""
    if not schedules:
        return []
    used_by_layer: list[set] = []
    merged: list[list[tuple]] = []
    width = max(len(schedule) for schedule in schedules)
    for layer_index in range(width):
        swaps = []
        used: set = set()
        for schedule in schedules:
            if layer_index >= len(schedule):
                continue
            for left, right in schedule[layer_index].swaps:
                if left in used or right in used:
                    raise ValueError(f"merged layer {layer_index} reuses a qubit")
                used.add(left)
                used.add(right)
                swaps.append((left, right))
        if swaps:
            merged.append(swaps)
            used_by_layer.append(used)
    return [SwapLayer(tuple(swaps)) for swaps in merged]
