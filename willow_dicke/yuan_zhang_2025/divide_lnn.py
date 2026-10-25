"""Nearest-neighbor Divide on a path of 2k qubits.

Lemma 8 of Yuan and Zhang (arXiv:2505.15413) cites the weight-distribution
block of Bärtschi and Eidenbenz (arXiv:2207.09998) and states that, on a path
of 2k qubits, that block has depth O(k) and size O(k^2). The 2025 text does
not expand the path schedule gate by gate. The circuit below is that block,
with two finite-size choices that are proved by the equivalence tests:

1. Each one-hot stair is a chain of singly-controlled and doubly-controlled
   Ry rotations. On the unary/one-hot input subspace a later rotation fires
   only if every earlier target is |1>, so the extra controls in the naive
   k-controlled stair are redundant. Two controls are enough to keep every
   other weight branch idle.
2. After the two unary strings are written, the right-register realignment
   cycles once per unary 1 on the left. That is the same permutation as the
   one-hot controlled cycle of Figure 4 in the 2022 paper, with O(k^2)
   Fredkins instead of O(k^3).

Every two-qubit gate is applied to path neighbors. A control that starts away
from its stair is bubbled to the boundary once and bubbled back once. A
3-qubit gate whose qubits are not yet contiguous is gathered once, applied
with an LNN Toffoli or CCRy decomposition, and gathered back. Elementary
gates inside that decomposition are not routed individually.

The output qubit order matches ``build_logical_divide``: the size-m register
is the left half of the path, the unary input occupies the right half, and
ones sit on the right of each register.
"""

from __future__ import annotations

from collections.abc import Sequence

import cirq

from dicke_willow.dicke_math import binomial, rotation_angle


class _Path:
    """Append gates, refusing a two-qubit gate that skips a path vertex."""

    def __init__(self, qubits: Sequence[cirq.Qid]) -> None:
        self.qubits = list(qubits)
        self.ops: list[cirq.Operation] = []

    def add(self, op: cirq.Operation) -> None:
        if cirq.num_qubits(op) > 2:
            pieces = [piece for piece in cirq.decompose(op) if piece != op]
            if not pieces:
                raise RuntimeError(f"no decomposition for {op.gate}")
            for piece in pieces:
                self.add(piece)
            return
        if cirq.num_qubits(op) == 0 or isinstance(op.gate, cirq.GlobalPhaseGate):
            return
        if cirq.num_qubits(op) == 2:
            left, right = op.qubits
            gap = abs(self.qubits.index(left) - self.qubits.index(right))
            if gap == 2:
                self._across_one(op)
                return
            if gap != 1:
                raise RuntimeError(f"{op} is not a nearest-neighbor path gate")
        self.ops.append(op)

    def _across_one(self, op: cirq.Operation) -> None:
        """Apply a distance-2 gate by parking the far state on the middle qubit."""
        left, right = op.qubits
        ia, ib = self.qubits.index(left), self.qubits.index(right)
        lo, hi = min(ia, ib), max(ia, ib)
        self.swap(lo + 1)
        shifted = []
        for qubit in op.qubits:
            index = self.qubits.index(qubit)
            shifted.append(self.qubits[lo + 1] if index == hi else qubit)
        self.ops.append(op.with_qubits(*shifted))
        self.swap(lo + 1)

    def swap(self, index: int) -> None:
        self.ops.append(cirq.SWAP(self.qubits[index], self.qubits[index + 1]))

    def cnot(self, control: int, target: int) -> None:
        if abs(control - target) != 1:
            raise RuntimeError("CNOT indices are not adjacent")
        self.ops.append(cirq.CNOT(self.qubits[control], self.qubits[target]))

    def cry(self, control: int, target: int, theta: float) -> None:
        """Adjacent controlled-Ry matching ``cirq.ry(theta).controlled()``."""
        if abs(theta) < 1e-15:
            return
        if abs(control - target) != 1:
            raise RuntimeError("CRY indices are not adjacent")
        target_q = self.qubits[target]
        control_q = self.qubits[control]
        self.ops.append(cirq.ry(theta / 2).on(target_q))
        self.ops.append(cirq.CNOT(control_q, target_q))
        self.ops.append(cirq.ry(-theta / 2).on(target_q))
        self.ops.append(cirq.CNOT(control_q, target_q))

    def ccry(self, start: int, theta: float) -> None:
        """CCRy on ``(start, start+1, start+2)`` = (control, second control, target)."""
        if abs(theta) < 1e-15:
            return
        qubits = self.qubits[start : start + 3]
        op = cirq.ry(theta).controlled(2).on(*qubits)
        for piece in cirq.decompose(op):
            self.add(piece)

    def fredkin(self, start: int) -> None:
        """CSWAP with control at ``start`` and targets ``start+1``, ``start+2``."""
        control, left, right = self.qubits[start : start + 3]
        self.ops.append(cirq.CNOT(right, left))
        for piece in cirq.decompose(cirq.CCNOT(control, left, right)):
            self.add(piece)
        self.ops.append(cirq.CNOT(right, left))

    def circuit(self) -> cirq.Circuit:
        return pack_earliest_respecting_order(self.ops)


def pack_earliest_respecting_order(ops: Sequence[cirq.Operation]) -> cirq.Circuit:
    """Pack ops into moments. An op waits only for earlier ops on its qubits."""
    ready: dict[cirq.Qid, int] = {}
    buckets: list[list[cirq.Operation]] = []
    for op in ops:
        moment = 0
        for qubit in op.qubits:
            moment = max(moment, ready.get(qubit, 0))
        if moment == len(buckets):
            buckets.append([])
        buckets[moment].append(op)
        for qubit in op.qubits:
            ready[qubit] = moment + 1
    return cirq.Circuit(cirq.Moment(bucket) for bucket in buckets if bucket)


def _stair_angles(n: int, m: int, k: int, ell: int) -> list[float] | None:
    """Stop angles for one one-hot value, or None when that value has no weight."""
    weights: list[int] = []
    for i in range(k + 1):
        if i > ell or ell - i > (n - m):
            weights.append(0)
        else:
            weights.append(binomial(m, i) * binomial(n - m, ell - i))
    total = sum(weights)
    if total == 0:
        return None
    thetas: list[float] = []
    remaining = total
    for weight in weights[:k]:
        if remaining <= 0:
            thetas.append(0.0)
            continue
        probability = min(1.0, max(0.0, weight / remaining))
        thetas.append(rotation_angle(probability))
        remaining -= weight
    return thetas


def _gather(path: _Path, originals: list[int]) -> list[int]:
    """Bubble ``originals`` into a contiguous block in that order.

    Returns the adjacent swaps, as window offsets, so the caller can invert
    them. Swaps act on the whole path; offsets are relative to the leftmost
    index in the gathered window's original span.
    """
    lo = min(originals)
    hi = max(originals)
    origin = list(range(lo, hi + 1))
    recorded: list[int] = []

    def find(label: int) -> int:
        return origin.index(label)

    def swap_offset(offset: int) -> None:
        path.swap(lo + offset)
        origin[offset], origin[offset + 1] = origin[offset + 1], origin[offset]
        recorded.append(offset)

    for slot, label in enumerate(originals):
        while find(label) > slot:
            swap_offset(find(label) - 1)
    return recorded


def _ungather(path: _Path, lo: int, recorded: list[int]) -> None:
    for offset in reversed(recorded):
        path.swap(lo + offset)


def _cnot_restoring(path: _Path, control: int, target: int) -> None:
    if abs(control - target) == 1:
        path.cnot(control, target)
        return
    lo = min(control, target)
    recorded = _gather(path, [control, target])
    path.cnot(lo, lo + 1)
    _ungather(path, lo, recorded)


def _cswap_restoring(path: _Path, control: int, left: int, right: int) -> None:
    lo = min(control, left, right)
    recorded = _gather(path, [control, left, right])
    path.fredkin(lo)
    _ungather(path, lo, recorded)


def _cycle_right_once(path: _Path, control: int, start: int, end: int) -> None:
    """Cycle path positions ``[start, end)`` one step right, controlled on ``control``."""
    for index in range(end - 2, start - 1, -1):
        _cswap_restoring(path, control, index, index + 1)


def _distribute(path: _Path, k: int, thetas: list[float]) -> None:
    """Write the weighted unary state on the left register.

    The one-hot control state starts at path index ``k``. It walks left
    through the left register so each rotation is a 2- or 3-qubit gate on
    three consecutive vertices, then walks back to index ``k``.
    """
    for step, theta in enumerate(thetas):
        control = k - step
        target = control - 1
        if step == 0:
            path.cry(control, target, theta)
        else:
            # Layout is (target, control, previous target). Cirq's CCRy wants
            # (control, previous, target), three adjacent vertices.
            if abs(theta) >= 1e-15:
                path.swap(target)
                path.swap(target + 1)
                path.ccry(target, theta)
                path.swap(target + 1)
                path.swap(target)
        path.swap(target)
    for index in range(k):
        path.swap(index)


def compile_divide_lnn(
    n: int,
    m: int,
    k: int,
    path: tuple[cirq.Qid, ...] | list[cirq.Qid],
) -> cirq.Circuit:
    """``Divide_k^{n,m}`` using nearest-neighbor gates on ``path``.

    ``path`` has length ``2k``. ``path[:k]`` is the size-``m`` register and
    starts in ``|0^k>``. ``path[k:]`` holds ``|0^{k-ell} 1^{ell}>``.
    """
    if k < 0:
        raise ValueError(f"k={k} is negative")
    if len(path) != 2 * k:
        raise ValueError(f"Divide path has {len(path)} qubits, expected {2 * k}")
    if k == 0:
        return cirq.Circuit()
    if m < k or n - m < k:
        raise ValueError(
            f"Divide^{n},{m}_{k} needs both children to have at least {k} qubits"
        )
    if len(set(path)) != len(path):
        raise ValueError("Divide path repeats a qubit")

    line = _Path(path)
    # Unary to one-hot on the right register. Adjacent CNOTs, no routing.
    for index in range(2 * k - 1, k, -1):
        line.cnot(index - 1, index)

    for ell in range(1, k + 1):
        thetas = _stair_angles(n, m, k, ell)
        if thetas is None:
            continue
        home = 2 * k - ell
        for index in range(home - 1, k - 1, -1):
            line.swap(index)
        _distribute(line, k, thetas)
        for index in range(k, home):
            line.swap(index)

    for index in range(k + 1, 2 * k):
        line.cnot(index - 1, index)

    # Cancel right-aligned ones, then shift the survivors back to the right
    # once for every unary 1 on the left register.
    for offset in range(k):
        _cnot_restoring(line, k - 1 - offset, 2 * k - 1 - offset)
    for offset in range(k):
        _cycle_right_once(line, k - 1 - offset, k, 2 * k)

    return line.circuit()
