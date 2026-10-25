"""Structured permutation of a rectangle.

``P(S, S')`` exchanges two ``k``-qubit blocks. States on the destination
block are the only ones displaced, and they move into the sites the register
vacated. Every other qubit stays put. Odd-even transposition sort along
the region's snake implements that permutation. The snake's edges are grid
edges, each layer is a matching, and the sort is a sorting network: Dijkstra
distances are not the definition of the route. The certificate checks the
final placement, not the fact that a shortest-path heuristic was used.

Depth is at most the number of snake vertices. When the two blocks lie on
one row the snake still works; a row-only permutation does not need the
other rows and the sort stops as soon as the keys are ordered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cirq

from willow_dicke.rectangle import Rectangle, adjacent


@dataclass
class PermutationCertificate:
    region_id: str
    initial_mapping: dict
    target_mapping: dict
    swap_layers: list
    final_mapping: dict
    valid_edges: bool
    correct_permutation: bool
    collision_free: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        def pairs(mapping: dict) -> list[list[str]]:
            return [[_label(src), _label(dst)] for src, dst in mapping.items()]

        return {
            "region_id": self.region_id,
            "initial_mapping": pairs(self.initial_mapping),
            "target_mapping": pairs(self.target_mapping),
            "swap_layers": [
                [[_label(a), _label(b)] for a, b in layer] for layer in self.swap_layers
            ],
            "final_mapping": pairs(self.final_mapping),
            "valid_edges": self.valid_edges,
            "correct_permutation": self.correct_permutation,
            "collision_free": self.collision_free,
            "n_swaps": sum(len(layer) for layer in self.swap_layers),
            "n_layers": len(self.swap_layers),
            "notes": self.notes,
        }


def _label(qubit: cirq.GridQubit) -> str:
    return f"{qubit.row},{qubit.col}"


def _coord(qubit: cirq.GridQubit) -> tuple[int, int]:
    return (qubit.row, qubit.col)


def target_mapping(
    region: Rectangle,
    source: tuple[cirq.GridQubit, ...],
    destination: tuple[cirq.GridQubit, ...],
) -> dict[cirq.GridQubit, cirq.GridQubit]:
    """Order-preserving block map, extended to a permutation of the region.

    State at ``source[i]`` goes to ``destination[i]``. States outside the
    source fill the sites outside the destination, in coordinate order.
    The extension is deterministic and does not depend on a measured weight.
    """
    if len(source) != len(destination):
        raise ValueError("block exchange needs equal block sizes")
    if len(set(source)) != len(source) or len(set(destination)) != len(destination):
        raise ValueError("block contains a repeated qubit")
    allowed = region.qubit_set()
    if not set(source) <= allowed or not set(destination) <= allowed:
        raise ValueError("block exchange left its rectangle")
    # Only the register and the blanks it displaces move. Pairing every
    # other site by coordinate order is also a permutation, but it drags
    # idle |0> qubits across the rectangle and the snake sort then pays
    # for that unrelated traffic.
    mapping = {qubit: qubit for qubit in region.qubits()}
    for src, dst in zip(source, destination):
        mapping[src] = dst
    source_set = set(source)
    destination_set = set(destination)
    vacated = sorted((qubit for qubit in source if qubit not in destination_set), key=_coord)
    displaced = sorted((qubit for qubit in destination if qubit not in source_set), key=_coord)
    if len(vacated) != len(displaced):
        raise RuntimeError("block map does not cover the rectangle")
    for blank, home in zip(displaced, vacated):
        mapping[blank] = home
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("block map is not a permutation")
    return mapping


def route_block_exchange(
    region: Rectangle,
    source_block: tuple[cirq.GridQubit, ...],
    destination_block: tuple[cirq.GridQubit, ...],
    device_graph,
    *,
    region_id: str,
) -> PermutationCertificate:
    """Sort the region along its snake until ``source_block`` sits on ``destination_block``."""
    mapping = target_mapping(region, source_block, destination_block)
    order = region.snake()
    index = {qubit: position for position, qubit in enumerate(order)}
    token_at = list(order)
    layers: list[tuple[tuple[cirq.GridQubit, cirq.GridQubit], ...]] = []
    valid_edges = True
    collision_free = True
    notes = [
        "odd-even transposition sort along the row-major snake",
        "each layer is a matching of snake edges",
    ]

    def key(token: cirq.GridQubit) -> int:
        return index[mapping[token]]

    idle = 0
    phase = 0
    while phase < len(order) and idle < 2:
        layer = []
        used: set[cirq.GridQubit] = set()
        swapped = False
        cursor = phase % 2
        while cursor < len(order) - 1:
            left, right = order[cursor], order[cursor + 1]
            if not adjacent(left, right):
                valid_edges = False
            if key(token_at[cursor]) > key(token_at[cursor + 1]):
                if left in used or right in used:
                    collision_free = False
                if device_graph is not None and not device_graph.has_edge(left, right):
                    valid_edges = False
                layer.append((left, right))
                used.add(left)
                used.add(right)
                token_at[cursor], token_at[cursor + 1] = token_at[cursor + 1], token_at[cursor]
                swapped = True
            cursor += 2
        if layer:
            layers.append(tuple(layer))
        idle = 0 if swapped else idle + 1
        phase += 1

    final = {token_at[position]: order[position] for position in range(len(order))}
    correct = final == mapping
    if not correct:
        notes.append("sort ended before every token reached its target")
    return PermutationCertificate(
        region_id=region_id,
        initial_mapping={qubit: qubit for qubit in order},
        target_mapping=mapping,
        swap_layers=layers,
        final_mapping=final,
        valid_edges=valid_edges,
        correct_permutation=correct,
        collision_free=collision_free,
        notes=notes,
    )


def swap_circuit(certificate: PermutationCertificate) -> cirq.Circuit:
    """One moment per SWAP layer. A layer is already a matching."""
    moments = []
    for layer in certificate.swap_layers:
        moments.append(cirq.Moment(cirq.SWAP(left, right) for left, right in layer))
    return cirq.Circuit(moments)
