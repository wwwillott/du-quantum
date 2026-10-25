"""Balanced Divide, block permutation, parallel children.

The circuit returned here is still the logical nearest-neighbor circuit:
Ry, CNOT, CCRy decompositions, Fredkin decompositions, and SWAPs on grid
edges. ``lower_to_willow_cz`` is a separate pass. Child circuits are zipped
by ``merge_parallel_circuits``; the left child is not appended after the right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cirq

from willow_dicke.rectangle import Rectangle
from willow_dicke.yuan_zhang_2025.divide_lnn import compile_divide_lnn
from willow_dicke.yuan_zhang_2025.geometry import GridPatch
from willow_dicke.yuan_zhang_2025.leaf import compile_leaf
from willow_dicke.yuan_zhang_2025.partition import RecursionNode, is_leaf, peel_applies
from willow_dicke.yuan_zhang_2025.permutation import (
    PermutationCertificate,
    route_block_exchange,
    swap_circuit,
)
from willow_dicke.yuan_zhang_2025.placement import divide_sites, entry_port
from willow_dicke.yuan_zhang_2025.scheduler import concatenate, merge_parallel_circuits


@dataclass
class Stage:
    """One region's contribution to the critical path, before CZ lowering."""

    name: str
    divide: cirq.Circuit = field(default_factory=cirq.Circuit)
    route: cirq.Circuit = field(default_factory=cirq.Circuit)
    leaf: cirq.Circuit = field(default_factory=cirq.Circuit)
    parallel: tuple["Stage", "Stage"] | None = None

    def circuit(self) -> cirq.Circuit:
        head = concatenate([self.divide, self.route, self.leaf])
        if self.parallel is None:
            return head
        left, right = self.parallel
        return concatenate([head, merge_parallel_circuits(left.circuit(), right.circuit())])


@dataclass
class CompiledRegion:
    circuit: cirq.Circuit
    port: tuple[cirq.GridQubit, ...]
    tree: RecursionNode
    certificates: list[PermutationCertificate]
    stage: Stage
    rectangle: Rectangle


def compile_patch(
    patch: GridPatch,
    k: int,
    device_graph=None,
) -> CompiledRegion:
    """Compile ``U_k^n`` on ``patch``. The unary input sits on ``port``."""
    if k < 0 or k > patch.height * patch.width:
        raise ValueError(f"k={k} is outside 0..{patch.height * patch.width}")
    rectangle = patch.rectangle()
    certificates: list[PermutationCertificate] = []
    tree, stage = _compile(rectangle, k, device_graph, "root", certificates)
    port = entry_port(rectangle, k)
    return CompiledRegion(
        circuit=stage.circuit(),
        port=port,
        tree=tree,
        certificates=certificates,
        stage=stage,
        rectangle=rectangle,
    )


def with_unary(body: cirq.Circuit, port: tuple[cirq.GridQubit, ...], ell: int) -> cirq.Circuit:
    """Prefix ``X`` on the rightmost ``ell`` qubits of the port."""
    if ell < 0 or ell > len(port):
        raise ValueError(f"ell={ell} is outside 0..{len(port)}")
    if ell == 0:
        return cirq.Circuit(body)
    prefix = cirq.Circuit(cirq.X(qubit) for qubit in port[-ell:])
    return concatenate([prefix, body])


def _compile(region, k, device_graph, region_id, certificates):
    qubits = region.qubits()
    if is_leaf(region, k):
        leaf = compile_leaf(region, k)
        node = RecursionNode(
            region_id=region_id,
            physical_qubits=qubits,
            dimensions=(region.height, region.width),
            n=region.n,
            max_weight=min(k, region.n),
            split_axis=None,
            children=None,
            divide_path=(),
            left_register=(),
            right_register=(),
            destination_registers=(entry_port(region, k),),
            kind="leaf",
        )
        _certify_leaf(region, k, region_id)
        return node, Stage(name=region_id, leaf=leaf)

    sites = divide_sites(region, k)
    if sites is None:
        raise RuntimeError(f"{region_id} was not a leaf but has no divide")
    left, right, s1, s2 = sites
    if len(s1) != k or len(s2) != k:
        raise RuntimeError("divide registers are not length k")
    if set(s1) & set(s2):
        raise RuntimeError("divide registers overlap")
    if not set(s1) <= left.qubit_set() or not set(s2) <= right.qubit_set():
        raise RuntimeError("divide register escaped its child")

    divide = compile_divide_lnn(region.n, left.n, k, list(s1) + list(s2))
    left_dest = entry_port(left, k)
    right_dest = entry_port(right, k)
    left_cert = route_block_exchange(
        left, s1, left_dest, device_graph, region_id=f"{region_id}/to-left"
    )
    right_cert = route_block_exchange(
        right, s2, right_dest, device_graph, region_id=f"{region_id}/to-right"
    )
    for certificate in (left_cert, right_cert):
        if not certificate.correct_permutation or not certificate.collision_free:
            raise RuntimeError(f"permutation failed in {certificate.region_id}")
        if device_graph is not None and not certificate.valid_edges:
            raise RuntimeError(f"permutation used a non-edge in {certificate.region_id}")
        certificates.append(certificate)
    route = merge_parallel_circuits(swap_circuit(left_cert), swap_circuit(right_cert))

    # Peel moves the incoming unary from the top-left block onto the right
    # register before Divide. Balanced nodes are entered already on ``s2``.
    prelude = cirq.Circuit()
    if peel_applies(region, k):
        incoming = tuple(region.rows[0][:k])
        if incoming != s2:
            pre = route_block_exchange(
                region, incoming, s2, device_graph, region_id=f"{region_id}/peel-shift"
            )
            if not pre.correct_permutation:
                raise RuntimeError(f"peel shift failed in {region_id}")
            certificates.append(pre)
            prelude = swap_circuit(pre)

    left_node, left_stage = _compile(left, k, device_graph, region_id + "L", certificates)
    right_node, right_stage = _compile(right, k, device_graph, region_id + "R", certificates)
    axis = "cols" if region.width >= region.height or peel_applies(region, k) else "rows"
    if peel_applies(region, k):
        axis = "cols"
    node = RecursionNode(
        region_id=region_id,
        physical_qubits=qubits,
        dimensions=(region.height, region.width),
        n=region.n,
        max_weight=k,
        split_axis=axis,
        children=(left_node, right_node),
        divide_path=s1 + s2,
        left_register=s1,
        right_register=s2,
        destination_registers=(left_dest, right_dest),
        kind="peel" if peel_applies(region, k) else "balanced",
    )
    stage = Stage(
        name=region_id,
        divide=concatenate([prelude, divide]),
        route=route,
        parallel=(left_stage, right_stage),
    )
    return node, stage


def _certify_leaf(region: Rectangle, k: int, region_id: str) -> None:
    port = entry_port(region, k)
    if len(port) != min(k, region.n):
        raise RuntimeError(f"{region_id} leaf port has the wrong length")
    if not set(port) <= region.qubit_set():
        raise RuntimeError(f"{region_id} leaf port left the region")
