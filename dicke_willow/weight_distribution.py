"""Weight-distribution block, ancilla-free and deterministic.

Bärtschi & Eidenbenz, arXiv:2207.09998, Definition 2:

    |0^{n-ell} 1^{ell}>
        -> sum_i sqrt( C(m,i) C(n-m, ell-i) / C(n, ell) )
           |0^{m-i} 1^i> |0^{n-m-(ell-i)} 1^{ell-i}>

The circuit follows Section IV of that paper, restricted to the k qubits at
the hot end of each child (the block acts as the identity on the remaining
qubits):

1. Unary -> one-hot on the register that already holds the weight (A).
2. For each one-hot value ell, a stair of controlled Ry rotations writes the
   weighted unary state on register B. The angle of step i is
   ``rotation_angle(x_i / s_i)`` with s_i the suffix sum of binomial weights
   (Figure 4, left).
3. One-hot -> unary on A.
4. Unary subtraction of B's weight from A's weight: cancel the overlapping
   right-aligned ones, then cyclically shift A's remaining ones back to the
   right, controlled on B's one-hot weight (Figure 4, right, as controlled
   permutations).

No measurement, reset, or postselection. The dense 2^n matrix is never built;
every gate is a one- or two-qubit rotation, CNOT, or controlled-SWAP.
"""

from __future__ import annotations

import math

import cirq

from dicke_willow.dicke_math import binomial, rotation_angle


def _unary_to_one_hot(register: list[cirq.Qid]) -> list[cirq.Operation]:
    """|0^{a-ell} 1^{ell}> -> |0^{a-ell} 1 0^{ell-1}> for every ell."""
    ops: list[cirq.Operation] = []
    for j in range(len(register) - 1, 0, -1):
        ops.append(cirq.CNOT(register[j - 1], register[j]))
    return ops


def _one_hot_to_unary(register: list[cirq.Qid]) -> list[cirq.Operation]:
    """Inverse of ``_unary_to_one_hot`` (the CNOT stair run forwards)."""
    ops: list[cirq.Operation] = []
    for j in range(1, len(register)):
        ops.append(cirq.CNOT(register[j - 1], register[j]))
    return ops


def _controlled_ry(
    controls: list[cirq.Qid], target: cirq.Qid, theta: float
) -> cirq.Operation:
    """Ry(theta) on ``target`` when every control is |1>."""
    if abs(theta) < 1e-15:
        return cirq.I(target)
    gate = cirq.ry(theta)
    if not controls:
        return gate.on(target)
    controlled = cirq.ControlledGate(gate, num_controls=len(controls))
    return controlled.on(*controls, target)


def _controlled_cycle_right(control: cirq.Qid, register: list[cirq.Qid], steps: int):
    """Cycle ``register`` right by ``steps``, controlled on ``control``."""
    ops: list[cirq.Operation] = []
    width = len(register)
    if width <= 1 or steps % width == 0:
        return ops
    for _ in range(steps % width):
        for index in range(width - 2, -1, -1):
            ops.append(cirq.CSWAP(control, register[index], register[index + 1]))
    return ops


def weight_distribution_ops(
    register_b: list[cirq.Qid],
    register_a: list[cirq.Qid],
    *,
    n: int,
    m: int,
    k: int,
) -> list[cirq.Operation]:
    """Divide a unary weight on A into children of sizes m and n-m.

    ``register_b`` is the hot end of the size-m child (initially |0>).
    ``register_a`` is the hot end of the size-(n-m) child (holds the unary
    input). Both registers are ordered left to right, ones growing on the
    right. Their lengths must be ``min(m, k)`` and ``min(n-m, k)``.
    """
    if len(register_b) != min(m, k) or len(register_a) != min(n - m, k):
        raise ValueError("register lengths must be min(child, k)")
    if k == 0 or not register_a:
        return []

    ops: list[cirq.Operation] = []
    ops.extend(_unary_to_one_hot(register_a))

    # Step (2): controlled unary state preparation on B, one stair per ell.
    for ell in range(1, k + 1):
        if ell > len(register_a):
            break
        weights = []
        for i in range(len(register_b) + 1):
            if i > ell or ell - i > (n - m):
                weights.append(0)
            else:
                weights.append(binomial(m, i) * binomial(n - m, ell - i))
        total = sum(weights)
        if total == 0:
            continue
        control = register_a[len(register_a) - ell]
        controls = [control]
        remaining = total
        # The last bucket is "every hot qubit is |1|"; it needs no rotation.
        for i, weight in enumerate(weights[: len(register_b)]):
            if remaining <= 0:
                break
            probability = weight / remaining
            theta = rotation_angle(min(1.0, max(0.0, probability)))
            target = register_b[len(register_b) - 1 - i]
            op = _controlled_ry(controls, target, theta)
            if op.gate != cirq.I:
                ops.append(op)
            controls = controls + [target]
            remaining -= weight

    ops.extend(_one_hot_to_unary(register_a))

    # Step (4): cancel right-aligned ones, then shift the survivors right by i.
    overlap = min(len(register_a), len(register_b))
    for offset in range(overlap):
        ops.append(
            cirq.CNOT(
                register_b[len(register_b) - 1 - offset],
                register_a[len(register_a) - 1 - offset],
            )
        )
    if len(register_b) == 0:
        return ops
    ops.extend(_unary_to_one_hot(register_b))
    for i in range(1, len(register_b) + 1):
        ops.extend(
            _controlled_cycle_right(
                register_b[len(register_b) - i], register_a, i
            )
        )
    ops.extend(_one_hot_to_unary(register_b))
    return ops


def ideal_divided_state(
    n: int, m: int, ell: int
) -> dict[tuple[int, int], float]:
    """Map (weight_on_left, weight_on_right) -> amplitude, for tests."""
    total = binomial(n, ell)
    if total == 0:
        return {}
    scale = math.sqrt(total)
    amplitudes: dict[tuple[int, int], float] = {}
    lo = max(0, ell - (n - m))
    hi = min(ell, m)
    for i in range(lo, hi + 1):
        weight = binomial(m, i) * binomial(n - m, ell - i)
        if weight:
            amplitudes[(i, ell - i)] = math.sqrt(weight) / scale
    return amplitudes
