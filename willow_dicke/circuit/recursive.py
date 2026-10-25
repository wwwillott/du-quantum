"""Recursive Dicke unitary on a certified partition tree.

The contract at every node S is

    |0>^{|S|-k} |0^{k-ell} 1^{ell}>  ->  |D_ell(S)>

for every ell = 0..k. The incoming ones occupy that node's input register,
right-aligned. Child circuits that share no qubits are zipped into the same
moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import cirq

from dicke_willow.scs import scs_unitary_ops
from willow_dicke.circuit.lower import (
    bridge_along_path,
    bridge_in_region,
    expand_circuit,
    is_full_cz,
    lower_to_willow_cz,
)
from willow_dicke.circuit.parallel import concatenate, parallel_compose
from willow_dicke.divide.logical_divide import build_logical_divide
from willow_dicke.graph import WillowGraph
from willow_dicke.partition.tree import PartitionNode
from willow_dicke.placement.staging import simple_path
from willow_dicke.routing.certificate import certify_route


@dataclass
class BuiltCircuit:
    circuit: cirq.Circuit
    divide_logical_depth: int
    routing_swap_depth: int
    leaf_logical_depth: int
    divide_cz_depth: int
    routing_cz_depth: int
    leaf_cz_depth: int
    trace: list


def build_root_dicke_circuit(
    willow: WillowGraph, root: PartitionNode, k: int
) -> BuiltCircuit:
    """Prepare ``|D_k>`` on ``root`` from ``|0...0>`` by X on the input register."""
    paths = _preparation_paths(willow, root, k)
    incoming = _input_register(root, k, paths)
    prepared = cirq.Circuit(cirq.X(qubit) for qubit in incoming)
    body = _build(willow, root, incoming, k, paths)
    return BuiltCircuit(
        circuit=concatenate(prepared, body.circuit),
        divide_logical_depth=body.divide_logical_depth,
        routing_swap_depth=body.routing_swap_depth,
        leaf_logical_depth=body.leaf_logical_depth,
        divide_cz_depth=body.divide_cz_depth,
        routing_cz_depth=body.routing_cz_depth,
        leaf_cz_depth=body.leaf_cz_depth,
        trace=[("pauli_x", tuple(incoming)), *body.trace],
    )


def build_coherent_unitary(
    willow: WillowGraph, root: PartitionNode, k: int
) -> BuiltCircuit:
    """The unitary alone, so a caller can feed every unary weight ell <= k."""
    paths = _preparation_paths(willow, root, k)
    incoming = _input_register(root, k, paths)
    return _build(willow, root, incoming, k, paths)


def _preparation_paths(
    willow: WillowGraph, node: PartitionNode, k: int
) -> dict[int, tuple[cirq.Qid, ...]]:
    """Hamiltonian path of each leaf. Internal nodes keep their staging path."""
    found: dict[int, tuple[cirq.Qid, ...]] = {}

    def walk(current: PartitionNode) -> None:
        if current.is_leaf:
            path = simple_path(willow, current.region.qubits, current.region.size)
            if path is None or len(path) != current.region.size:
                # No Hamiltonian path. Keep a real device path as the unary
                # suffix so the parent can route onto it, and bridge the
                # remaining line interactions inside the region.
                weight = min(k, current.region.size)
                register = simple_path(willow, current.region.qubits, weight)
                if register is None:
                    raise RuntimeError(
                        f"leaf {current.region.id} has no {weight}-qubit path for its input register"
                    )
                rest = tuple(
                    sorted(
                        set(current.region.qubits) - set(register),
                        key=lambda qubit: (qubit.row, qubit.col),
                    )
                )
                path = rest + tuple(register)
            else:
                for left, right in zip(path, path[1:]):
                    if not willow.has_edge(left, right):
                        raise RuntimeError(f"leaf path edge {left} -- {right} is not on the device")
                path = tuple(path)
            found[id(current)] = path
            return
        if current.staging_path is None:
            raise RuntimeError(f"internal node {current.region.id} has no staging path")
        walk(current.left)
        walk(current.right)

    walk(node)
    return found


def _input_register(
    node: PartitionNode, k: int, paths: dict[int, tuple[cirq.Qid, ...]]
) -> tuple[cirq.Qid, ...]:
    """Where this node's unary weight must sit before its own unitary.

    An internal node reads the weight on the right half of its staging path.
    A leaf reads it on the right end of its Hamiltonian path, which is the
    ``|0^{s-ell} 1^{ell}>`` input of the 2019 SCS unitary.
    """
    if node.is_leaf:
        path = paths[id(node)]
        length = min(k, len(path))
        return tuple(path[-length:])
    path = node.staging_path
    if path is None or len(path) != 2 * k:
        raise RuntimeError(f"{node.region.id} staging path is not length {2 * k}")
    return tuple(path[k:])


def _build(
    willow: WillowGraph,
    node: PartitionNode,
    incoming: tuple[cirq.Qid, ...],
    k: int,
    paths: dict[int, tuple[cirq.Qid, ...]],
) -> BuiltCircuit:
    if node.failure:
        raise RuntimeError(node.failure)
    if node.is_leaf:
        return _leaf(willow, node, incoming, k, paths)
    if node.left is None or node.right is None or node.staging_path is None:
        raise RuntimeError(f"{node.region.id} is missing a child")
    path = tuple(node.staging_path)
    for left, right in zip(path, path[1:]):
        if not willow.has_edge(left, right):
            raise RuntimeError(f"staging edge {left} -- {right} is not on the device")
    left_register = path[:k]
    right_register = path[k:]
    if len(node.left.region.qubits) < k or len(node.right.region.qubits) < k:
        raise RuntimeError(
            f"{node.region.id} cannot run Divide_{k}: "
            f"children have {len(node.left.region.qubits)} and {len(node.right.region.qubits)} qubits"
        )
    route_in, route_in_swaps = _route(willow, node.region.qubits, incoming, right_register, node.region.id)
    divide = bridge_along_path(
        expand_circuit(
            build_logical_divide(
                node.region.size,
                node.left.region.size,
                k,
                path,
            )
        ),
        path,
    )
    left_target = _input_register(node.left, k, paths)
    right_target = _input_register(node.right, k, paths)
    left_route, left_swaps = _route(willow, node.left.region.qubits, left_register, left_target, node.left.region.id)
    right_route, right_swaps = _route(
        willow, node.right.region.qubits, right_register, right_target, node.right.region.id
    )
    routed_children = parallel_compose(left_route, right_route)
    left_body = _build(willow, node.left, left_target, k, paths)
    right_body = _build(willow, node.right, right_target, k, paths)
    children = parallel_compose(left_body.circuit, right_body.circuit)
    return BuiltCircuit(
        circuit=concatenate(route_in, divide, routed_children, children),
        divide_logical_depth=len(divide) + max(left_body.divide_logical_depth, right_body.divide_logical_depth),
        routing_swap_depth=(
            len(route_in)
            + max(len(left_route), len(right_route))
            + max(left_body.routing_swap_depth, right_body.routing_swap_depth)
        ),
        leaf_logical_depth=max(left_body.leaf_logical_depth, right_body.leaf_logical_depth),
        divide_cz_depth=_compiled_cz_depth(divide)
        + max(left_body.divide_cz_depth, right_body.divide_cz_depth),
        routing_cz_depth=(
            _compiled_cz_depth(route_in)
            + _compiled_cz_depth(routed_children)
            + max(left_body.routing_cz_depth, right_body.routing_cz_depth)
        ),
        leaf_cz_depth=max(left_body.leaf_cz_depth, right_body.leaf_cz_depth),
        trace=[
            ("swaps", route_in_swaps),
            ("unitary", divide, path),
            ("swaps", left_swaps + right_swaps),
            *left_body.trace,
            *right_body.trace,
        ],
    )


def _leaf(
    willow: WillowGraph,
    node: PartitionNode,
    incoming: tuple[cirq.Qid, ...],
    k: int,
    paths: dict[int, tuple[cirq.Qid, ...]],
) -> BuiltCircuit:
    """2019 U_{s, min(k,s)} on a Hamiltonian path. Handles every ell, not one k."""
    path = paths[id(node)]
    weight = min(k, len(path))
    target = path[-weight:]
    if len(incoming) != weight:
        raise RuntimeError(
            f"leaf {node.region.id} expected a length-{weight} register, got {len(incoming)}"
        )
    route, swaps = _route(willow, node.region.qubits, incoming, target, node.region.id)
    unitary = cirq.Circuit(scs_unitary_ops(list(path), weight))
    if any(not willow.has_edge(left, right) for left, right in zip(path, path[1:])):
        unitary = bridge_in_region(unitary, willow, path)
    return BuiltCircuit(
        circuit=concatenate(route, unitary),
        divide_logical_depth=0,
        routing_swap_depth=len(route),
        leaf_logical_depth=len(unitary),
        divide_cz_depth=0,
        routing_cz_depth=_compiled_cz_depth(route),
        leaf_cz_depth=_compiled_cz_depth(unitary),
        trace=[("swaps", swaps), ("unitary", unitary, tuple(path))],
    )


def _compiled_cz_depth(circuit: cirq.Circuit) -> int:
    if len(circuit) == 0:
        return 0
    lowered = lower_to_willow_cz(circuit)
    return sum(1 for moment in lowered if any(is_full_cz(op) for op in moment))


def _route(
    willow: WillowGraph,
    region: frozenset[cirq.Qid] | set[cirq.Qid],
    source: tuple[cirq.Qid, ...],
    target: tuple[cirq.Qid, ...],
    region_id: str,
) -> tuple[cirq.Circuit, tuple[tuple[cirq.Qid, cirq.Qid], ...]]:
    """Certified SWAP layers. An empty circuit when the register is already there."""
    if source == target:
        return cirq.Circuit(), ()
    certificate = certify_route(
        willow,
        region_id=region_id,
        depth=0,
        region=set(region),
        source=tuple(source),
        target=tuple(target),
    )
    if not certificate.verified:
        raise RuntimeError(
            f"route in {region_id} failed: {certificate.problems}; "
            f"source={[(q.row, q.col) for q in source]} "
            f"target={[(q.row, q.col) for q in target]} "
            f"region={len(region)}"
        )
    circuit = cirq.Circuit()
    swaps: list[tuple[cirq.Qid, cirq.Qid]] = []
    for layer in certificate.swap_layers:
        circuit.append(cirq.Moment(cirq.SWAP(left, right) for left, right in layer.swaps))
        swaps.extend(layer.swaps)
    return circuit, tuple(swaps)
