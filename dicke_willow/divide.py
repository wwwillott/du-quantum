"""Balanced divide-and-conquer Dicke preparation.

Yuan & Zhang, arXiv:2505.15413, build |D_k^n> by splitting a region into
children as equal as the weight constraint allows, applying one
weight-distribution block, then preparing both children at once. A child is
a base case when its size is at most the maximum weight still in play: the
2019 SCS unitary then spreads that unary weight into a Dicke state.

The split keeps the right child at least ``max_weight`` qubits wide whenever
that is possible, so the unary ones that enter the region stay inside the
right child and the weight-distribution block's hot registers do not see a
spilled excitation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cirq

from dicke_willow.scs import scs_unitary_ops
from dicke_willow.weight_distribution import weight_distribution_ops


@dataclass
class DickeRegion:
    """One node of the balanced recursion tree."""

    qubits: tuple[cirq.Qid, ...]
    n: int
    max_weight: int
    depth: int
    m: int | None = None
    children: tuple[DickeRegion, DickeRegion] | None = None
    hot_left: tuple[cirq.Qid, ...] = ()
    hot_right: tuple[cirq.Qid, ...] = ()

    def walk(self):
        yield self
        if self.children is not None:
            yield from self.children[0].walk()
            yield from self.children[1].walk()


def choose_split(n: int, max_weight: int) -> int | None:
    """Size of the left child, or None when this region is a base case.

    The right child has size ``n - m``. It is at least ``max_weight`` so a
    unary input of weight <= max_weight sits entirely in the right child,
    and as close to ``n / 2`` as that constraint allows.
    """
    if n <= max_weight or n <= 1 or max_weight <= 0:
        return None
    right = max(max_weight, (n + 1) // 2)
    if right >= n:
        right = n - 1
    if right < max_weight:
        return None
    return n - right


def build_region(
    qubits: tuple[cirq.Qid, ...], max_weight: int, depth: int = 0
) -> DickeRegion:
    """Explicit recursion tree. Does not emit gates."""
    n = len(qubits)
    weight = min(max_weight, n)
    region = DickeRegion(qubits=qubits, n=n, max_weight=weight, depth=depth)
    m = choose_split(n, weight)
    if m is None:
        return region
    left = qubits[:m]
    right = qubits[m:]
    b = min(m, weight)
    a = min(n - m, weight)
    region.m = m
    region.hot_left = left[m - b :]
    region.hot_right = right[(n - m) - a :]
    region.children = (
        build_region(left, weight, depth + 1),
        build_region(right, weight, depth + 1),
    )
    return region


def _zip_circuits(left: cirq.Circuit, right: cirq.Circuit) -> cirq.Circuit:
    """Run two disjoint circuits in the same moments."""
    merged = cirq.Circuit()
    for moment_left, moment_right in _pad(left, right):
        ops = []
        if moment_left is not None:
            ops.extend(moment_left.operations)
        if moment_right is not None:
            ops.extend(moment_right.operations)
        if ops:
            merged.append(cirq.Moment(ops))
    return merged


def _pad(left: cirq.Circuit, right: cirq.Circuit):
    length = max(len(left), len(right))
    for index in range(length):
        yield (
            left[index] if index < len(left) else None,
            right[index] if index < len(right) else None,
        )


def unitary_circuit(qubits: list[cirq.Qid], max_weight: int) -> tuple[cirq.Circuit, DickeRegion]:
    """U that maps every unary ``|0^{n-ell} 1^{ell}>``, ell <= max_weight, to ``|D_ell>``.

    Child unitaries are scheduled together. This is the all-to-all logical
    circuit: the weight-distribution block may touch non-adjacent hot
    registers. ``route_region`` is what makes those gates nearest-neighbor.
    """
    region = build_region(tuple(qubits), max_weight)
    return _emit(region), region


def _emit(region: DickeRegion) -> cirq.Circuit:
    if region.children is None:
        if region.max_weight <= 0 or region.n <= 1:
            return cirq.Circuit()
        # Base case: 2019 SCS unitary on this compact region. U_{n,n} covers
        # every unary weight when the region is no larger than the max weight.
        return cirq.Circuit(scs_unitary_ops(list(region.qubits), region.max_weight))

    assert region.m is not None
    divide = cirq.Circuit(
        weight_distribution_ops(
            list(region.hot_left),
            list(region.hot_right),
            n=region.n,
            m=region.m,
            k=region.max_weight,
        )
    )
    left_circuit = _emit(region.children[0])
    right_circuit = _emit(region.children[1])
    children = _zip_circuits(left_circuit, right_circuit)
    return divide + children


def prepare_balanced(
    qubits: list[cirq.Qid], k: int
) -> tuple[cirq.Circuit, DickeRegion]:
    """Prepare |D_k^n> from |0...0>, using the smaller of k and n-k."""
    n = len(qubits)
    if k < 0 or k > n:
        raise ValueError(f"k={k} is outside 0..n={n}")
    excite = min(k, n - k) if n else 0
    prefix: list[cirq.Operation] = []
    if excite == 0:
        region = build_region(tuple(qubits), 0)
        return cirq.Circuit(), region
    if k == n:
        region = build_region(tuple(qubits), 0)
        return cirq.Circuit(cirq.X(q) for q in qubits), region
    prefix = [cirq.X(qubits[i]) for i in range(n - excite, n)]
    body, region = unitary_circuit(qubits, excite)
    circuit = cirq.Circuit(prefix) + body
    if excite != k:
        circuit = circuit + cirq.Circuit(cirq.X(q) for q in qubits)
    return circuit, region
