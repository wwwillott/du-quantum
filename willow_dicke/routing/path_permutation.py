"""Odd-even transposition sort along one corridor. Each layer is a matching."""

from __future__ import annotations

from dataclasses import dataclass

import cirq

from willow_dicke.routing.corridor import RoutingCorridor


@dataclass(frozen=True)
class SwapLayer:
    swaps: tuple[tuple[cirq.Qid, cirq.Qid], ...]


def route_path_permutation(
    path: tuple[cirq.Qid, ...],
    destination_of_position: list[int],
) -> list[SwapLayer]:
    """SWAP layers that send the token at position i to ``destination_of_position[i]``.

    Destinations must be a permutation of ``0 .. len(path)-1``.
    """
    length = len(path)
    if sorted(destination_of_position) != list(range(length)):
        raise ValueError("destinations are not a permutation of the path positions")
    order = list(destination_of_position)
    layers: list[SwapLayer] = []
    for round_index in range(length):
        start = round_index % 2
        swaps: list[tuple[cirq.Qid, cirq.Qid]] = []
        index = start
        while index + 1 < length:
            if order[index] > order[index + 1]:
                swaps.append((path[index], path[index + 1]))
                order[index], order[index + 1] = order[index + 1], order[index]
            index += 2
        if swaps:
            layers.append(SwapLayer(tuple(swaps)))
        if all(order[i] <= order[i + 1] for i in range(length - 1)):
            break
    return layers


def corridor_destinations(corridor: RoutingCorridor) -> list[int]:
    """Where the token that starts at each corridor index must finish.

    Register token i moves from ``source_indices[i]`` to ``target_indices[i]``.
    Every other token keeps the leftover indices in increasing order.
    """
    length = len(corridor.path)
    destination = [-1] * length
    claimed = set(corridor.target_indices)
    for source_index, target_index in zip(corridor.source_indices, corridor.target_indices):
        destination[source_index] = target_index
    leftovers = [index for index in range(length) if index not in claimed]
    spectators = [index for index, goal in enumerate(destination) if goal < 0]
    if len(leftovers) != len(spectators):
        raise ValueError("register destinations do not leave a matching set of spectator slots")
    for position, goal in zip(spectators, leftovers):
        destination[position] = goal
    return destination
