"""Path Divide on a certified 2k staging path.

The block is the ancilla-free weight distribution from arXiv:2207.09998,
Definition 2, restricted to the hot end of each child. It is not the
all-to-all Divide that borrows idle qubits.

``qubits`` is the staging path, ordered so ``qubits[:k]`` lies in the size-m
child and ``qubits[k:]`` lies in the other child. The unary input occupies
the right register. The left register enters as ``|0^k>``.
"""

from __future__ import annotations

from collections.abc import Sequence

import cirq

from dicke_willow.weight_distribution import weight_distribution_ops


def build_logical_divide(
    n: int,
    m: int,
    k: int,
    qubits: Sequence[cirq.Qid],
) -> cirq.Circuit:
    """``Divide^{n,m}_k`` on a path of exactly ``2k`` qubits.

    Requires ``m >= k`` and ``n - m >= k``, so each register has length ``k``.
    """
    if len(qubits) != 2 * k:
        raise ValueError(f"Divide path has {len(qubits)} qubits, expected {2 * k}")
    if m < k or n - m < k:
        raise ValueError(f"Divide^{n},{m}_{k} needs both children to have at least {k} qubits")
    if k == 0:
        return cirq.Circuit()
    left = list(qubits[:k])
    right = list(qubits[k:])
    # register_b is the size-m child and starts at |0>.
    # register_a holds the unary input. That is the user's RA, RB order.
    ops = weight_distribution_ops(left, right, n=n, m=m, k=k)
    return cirq.Circuit(op for op in ops if op.gate != cirq.I)
