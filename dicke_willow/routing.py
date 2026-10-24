"""Permutation routing by layers of vertex-disjoint SWAPs.

``route_permutation`` moves every token from its current vertex to a target
vertex, using only edges of ``device_graph``. Each returned layer is a
matching: no vertex is in two SWAPs of the same layer. Disjoint regions can
therefore be routed in parallel by passing each region as ``allowed_region``
and concatenating the layers.

The router is intentionally visible. It is not Cirq's generic router. On a
short path it bubbles tokens with odd-even transposition sort, which is the
row routing the grid Dicke construction uses. On a general graph it repeatedly
swaps a misplaced token along a shortest path, accepting a SWAP only when it
does not increase the sum of remaining distances, and it packs those SWAPs
into a matching.
"""

from __future__ import annotations

from collections import deque

import cirq


def _shortest_path(
    adjacency: dict, source, dest, allowed: set
) -> list | None:
    if source == dest:
        return [source]
    previous = {source: None}
    queue: deque = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in allowed or neighbor in previous:
                continue
            previous[neighbor] = node
            if neighbor == dest:
                path = [dest]
                while path[-1] != source:
                    path.append(previous[path[-1]])
                path.reverse()
                return path
            queue.append(neighbor)
    return None


def _is_path_graph(adjacency: dict, allowed: set) -> bool:
    degrees = [len([n for n in adjacency.get(node, ()) if n in allowed]) for node in allowed]
    return degrees.count(1) == 2 and all(degree <= 2 for degree in degrees) and sum(degrees) == 2 * (len(allowed) - 1)


def _path_order(adjacency: dict, allowed: set) -> list:
    start = next(
        node
        for node in allowed
        if len([n for n in adjacency.get(node, ()) if n in allowed]) == 1
    )
    order = [start]
    seen = {start}
    while len(order) < len(allowed):
        nxt = next(n for n in adjacency[order[-1]] if n in allowed and n not in seen)
        seen.add(nxt)
        order.append(nxt)
    return order


def route_permutation(
    device_graph: dict,
    initial_mapping: dict,
    target_mapping: dict,
    allowed_region: set | None = None,
) -> list[list[tuple]]:
    """SWAP layers taking ``initial_mapping[logical]`` to ``target_mapping[logical]``.

    Mappings send a logical label to the physical vertex that currently holds
    it. Every physical vertex in the region appears exactly once.
    """
    if set(initial_mapping) != set(target_mapping):
        raise ValueError("initial and target mappings label different tokens")
    location = dict(initial_mapping)
    allowed = set(allowed_region) if allowed_region is not None else set(location.values())
    if set(location.values()) != allowed:
        raise ValueError("mapping is not a bijection onto the allowed region")

    # Where each token must end up, and which token is wanted at each vertex.
    goal_of_token = dict(target_mapping)
    if _is_path_graph(device_graph, allowed):
        return _route_path(device_graph, location, goal_of_token, allowed)
    return _route_graph(device_graph, location, goal_of_token, allowed)


def _route_path(adjacency, location, goal, allowed) -> list[list[tuple]]:
    """Odd-even transposition sort along the unique path."""
    order = _path_order(adjacency, allowed)
    index = {vertex: i for i, vertex in enumerate(order)}
    # token sitting at each path index
    at = [None] * len(order)
    for token, vertex in location.items():
        at[index[vertex]] = token
    target_index = {token: index[vertex] for token, vertex in goal.items()}
    layers: list[list[tuple]] = []
    max_rounds = len(order) * len(order) + 1
    for round_index in range(max_rounds):
        if all(target_index[token] == i for i, token in enumerate(at)):
            return layers
        layer = []
        offset = round_index % 2
        for i in range(offset, len(order) - 1, 2):
            left, right = at[i], at[i + 1]
            if target_index[left] > target_index[right]:
                layer.append((order[i], order[i + 1]))
                at[i], at[i + 1] = right, left
        if layer:
            layers.append(layer)
    raise RuntimeError("path routing failed to sort the permutation")


def _route_graph(adjacency, location, goal, allowed) -> list[list[tuple]]:
    """Greedy matching of improving SWAPs on a general coupling graph."""
    token_at = {vertex: token for token, vertex in location.items()}
    layers: list[list[tuple]] = []
    limit = len(allowed) * len(allowed) + 1

    def remaining(token_at_now) -> int:
        total = 0
        for vertex, token in token_at_now.items():
            path = _shortest_path(adjacency, vertex, goal[token], allowed)
            if path is None:
                return 10**9
            total += len(path) - 1
        return total

    for _ in range(limit):
        if all(vertex == goal[token] for vertex, token in token_at.items()):
            return layers
        candidates = []
        seen_edges = set()
        for vertex, token in token_at.items():
            if vertex == goal[token]:
                continue
            path = _shortest_path(adjacency, vertex, goal[token], allowed)
            if path is None or len(path) < 2:
                continue
            nxt = path[1]
            edge = tuple(sorted((vertex, nxt), key=str))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            # SWAP helps when the neighbor is not already home, or always if
            # this token's next step is that neighbor. Accept if total distance
            # does not increase.
            swapped = dict(token_at)
            swapped[vertex], swapped[nxt] = swapped[nxt], swapped[vertex]
            if remaining(swapped) <= remaining(token_at):
                candidates.append((vertex, nxt))
        if not candidates:
            raise RuntimeError("no improving SWAP; permutation is not routable in this region")
        layer = []
        used = set()
        for a, b in candidates:
            if a in used or b in used:
                continue
            layer.append((a, b))
            used.add(a)
            used.add(b)
            token_at[a], token_at[b] = token_at[b], token_at[a]
        layers.append(layer)
    raise RuntimeError("graph routing exceeded its swap budget")


def apply_swap_layers(
    layers: list[list[tuple]],
    logical_to_physical: dict,
    physical_to_logical: dict,
) -> None:
    """Update both bijections in place. Raises if a layer is not a matching."""
    for layer in layers:
        seen = set()
        for a, b in layer:
            if a in seen or b in seen:
                raise RuntimeError(f"SWAP layer is not a matching: {layer}")
            seen.add(a)
            seen.add(b)
            token_a = physical_to_logical[a]
            token_b = physical_to_logical[b]
            physical_to_logical[a] = token_b
            physical_to_logical[b] = token_a
            logical_to_physical[token_a] = b
            logical_to_physical[token_b] = a
        if len(logical_to_physical) != len(set(logical_to_physical.values())):
            raise RuntimeError("mapping ceased to be a bijection")


def layers_to_circuit(layers: list[list[tuple]]) -> cirq.Circuit:
    """One moment of SWAPs per matching layer."""
    circuit = cirq.Circuit()
    for layer in layers:
        circuit.append(cirq.Moment(cirq.SWAP(a, b) for a, b in layer))
    return circuit


def describe_route(
    *,
    level: int,
    region_qubits: list,
    before: dict[str, list],
    target: dict[str, list],
    layers: list[list[tuple]],
) -> str:
    """Debug text for one region's SWAP schedule."""
    lines = [
        f"recursion level {level}",
        f"region: {len(region_qubits)} physical qubits",
        "",
        "before:",
    ]
    for name, qubits in before.items():
        lines.append(f"weight register {name} -> {list(qubits)}")
    lines.append("")
    lines.append("target:")
    for name, qubits in target.items():
        lines.append(f"{name} -> {list(qubits)}")
    lines.append("")
    for index, layer in enumerate(layers):
        lines.append(f"routing layer {index}:")
        for a, b in layer:
            lines.append(f"SWAP {a}<->{b}")
    return "\n".join(lines)
