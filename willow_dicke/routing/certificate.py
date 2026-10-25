"""One certified relocation of a unary register."""

from __future__ import annotations

from dataclasses import dataclass

import cirq

from willow_dicke.graph import WillowGraph
from willow_dicke.routing.corridor import RoutingCorridor, find_corridor
from willow_dicke.routing.path_permutation import SwapLayer, corridor_destinations, route_path_permutation
from willow_dicke.routing.tokens import LogicalToken, Placement


@dataclass
class RouteCertificate:
    region_id: str
    depth: int
    source: tuple[cirq.Qid, ...]
    target: tuple[cirq.Qid, ...]
    corridor: tuple[cirq.Qid, ...]
    swap_layers: tuple[SwapLayer, ...]
    initial_mapping: dict[str, tuple[int, int]]
    final_mapping: dict[str, tuple[int, int]]
    verified: bool
    problems: tuple[str, ...]

    @property
    def swap_count(self) -> int:
        return sum(len(layer.swaps) for layer in self.swap_layers)

    def to_dict(self) -> dict:
        def coords(qubit: cirq.Qid) -> tuple[int, int]:
            return (qubit.row, qubit.col)

        return {
            "region_id": self.region_id,
            "depth": self.depth,
            "source": [coords(qubit) for qubit in self.source],
            "target": [coords(qubit) for qubit in self.target],
            "corridor": [coords(qubit) for qubit in self.corridor],
            "n_layers": len(self.swap_layers),
            "n_swaps": self.swap_count,
            "swap_layers": [
                [[coords(a), coords(b)] for a, b in layer.swaps] for layer in self.swap_layers
            ],
            "initial_mapping": self.initial_mapping,
            "final_mapping": self.final_mapping,
            "verified": self.verified,
            "problems": list(self.problems),
        }


def certify_route(
    willow: WillowGraph,
    *,
    region_id: str,
    depth: int,
    region: set[cirq.GridQubit],
    source: tuple[cirq.GridQubit, ...],
    target: tuple[cirq.GridQubit, ...],
) -> RouteCertificate:
    """Build the corridor and SWAP layers, then check them by replaying the placement."""
    problems: list[str] = []
    if len(source) != len(target) or not source:
        problems.append("source and target registers differ in length")
    if any(qubit not in region for qubit in (*source, *target)):
        problems.append("a register qubit is outside the region")
    corridor = None if problems else find_corridor(willow, set(region), source, target)
    if corridor is None:
        problems.append("no corridor was constructed")
        return _failed(region_id, depth, source, target, problems)
    for left, right in zip(corridor.path, corridor.path[1:]):
        if left not in region or right not in region:
            problems.append(f"corridor leaves the region at {left} -- {right}")
        if not willow.has_edge(left, right):
            problems.append(f"corridor edge {left} -- {right} is not on the device")
    try:
        destinations = corridor_destinations(corridor)
        layers = route_path_permutation(corridor.path, destinations)
    except ValueError as exc:
        problems.append(str(exc))
        return _failed(region_id, depth, source, target, problems, corridor)
    placement, initial = _initial_placement(corridor, source)
    try:
        placement.check()
    except AssertionError as exc:
        problems.append(str(exc))
    for layer_index, layer in enumerate(layers):
        used: set[cirq.Qid] = set()
        for left, right in layer.swaps:
            if not willow.has_edge(left, right):
                problems.append(f"layer {layer_index} swaps a missing edge {left} -- {right}")
            if left in used or right in used:
                problems.append(f"layer {layer_index} is not a matching")
            if left not in placement.physical_to_logical or right not in placement.physical_to_logical:
                problems.append(f"layer {layer_index} swaps a qubit off the corridor")
                continue
            used.add(left)
            used.add(right)
            placement.swap(left, right)
    for index, qubit in enumerate(target):
        token = placement.physical_to_logical.get(qubit)
        if token != LogicalToken("register", index):
            problems.append(f"register {index} finished on {token}, not {qubit}")
    final = {str(token): (qubit.row, qubit.col) for token, qubit in placement.logical_to_physical.items()}
    return RouteCertificate(
        region_id=region_id,
        depth=depth,
        source=source,
        target=target,
        corridor=tuple(corridor.path),
        swap_layers=tuple(layers),
        initial_mapping=initial,
        final_mapping=final,
        verified=not problems,
        problems=tuple(problems),
    )


def _initial_placement(
    corridor: RoutingCorridor, source: tuple[cirq.GridQubit, ...]
) -> tuple[Placement, dict[str, tuple[int, int]]]:
    register = {source[index]: LogicalToken("register", index) for index in range(len(source))}
    placement = Placement()
    spectator = 0
    for qubit in corridor.path:
        if qubit in register:
            token = register[qubit]
        else:
            token = LogicalToken("spectator", spectator)
            spectator += 1
        placement.logical_to_physical[token] = qubit
        placement.physical_to_logical[qubit] = token
    mapping = {str(token): (qubit.row, qubit.col) for token, qubit in placement.logical_to_physical.items()}
    return placement, mapping


def _failed(region_id, depth, source, target, problems, corridor=None) -> RouteCertificate:
    return RouteCertificate(
        region_id=region_id,
        depth=depth,
        source=source,
        target=target,
        corridor=() if corridor is None else tuple(corridor.path),
        swap_layers=(),
        initial_mapping={},
        final_mapping={},
        verified=False,
        problems=tuple(problems),
    )
