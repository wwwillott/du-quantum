"""Balanced and peeled partitions, with the paper's rounding made explicit.

Theorem 14 assumes ``sqrt(n1 k / n2)`` and ``n/k`` are integers and says
ceiling values may be used. On a 7 by 8 rectangle those quantities are not
integers. The rules below are the finite-size adaptation:

- Split the longer side. The cut is the integer closest to half the qubits
  among the cuts that leave both children with at least ``k`` qubits.
- A Divide is emitted only when that cut exists. Otherwise the region is a
  Lemma 7 leaf, even if it has more than ``k`` qubits.
- When ``k < n2 / n1`` and ``k`` divides ``n2``, peel a strip of width ``k``
  (Figure 3b) instead of cutting in half. ``k = 1`` on 7 by 8 is this case.
  ``k >= 2`` is the balanced case (Figure 3a).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from willow_dicke.rectangle import Rectangle


@dataclass
class RecursionNode:
    """One region in the balanced (or peeled) recursion."""

    region_id: str
    physical_qubits: tuple
    dimensions: tuple[int, int]
    n: int
    max_weight: int
    split_axis: str | None
    children: tuple["RecursionNode", ...] | None
    divide_path: tuple
    left_register: tuple
    right_register: tuple
    destination_registers: tuple
    kind: str


def peel_applies(region: Rectangle, k: int) -> bool:
    """Figure 3(b): ``k < n2 / n1`` and the long side is a whole number of strips."""
    return k * region.height < region.width and region.width % k == 0


def split_longer(region: Rectangle, k: int) -> tuple[Rectangle, Rectangle, str] | None:
    """Cut the longer side so both children have at least ``k`` qubits.

    Returns ``(left, right, axis)`` or ``None``. ``left`` is the low-index side.
    Ties in distance-to-half prefer the smaller left piece.
    """
    height, width = region.height, region.width
    if width >= height:
        axis, length, other = "cols", width, height
    else:
        axis, length, other = "rows", height, width
    need = math.ceil(k / other)
    if length < 2 * need:
        return None
    target = region.n / 2

    def score(lines: int) -> tuple[float, int]:
        return (abs(lines * other - target), lines)

    lines = min(range(need, length - need + 1), key=score)
    if axis == "cols":
        return region.sub_cols(0, lines), region.sub_cols(lines, width), axis
    return region.sub_rows(0, lines), region.sub_rows(lines, height), axis


def is_leaf(region: Rectangle, k: int) -> bool:
    """A leaf is a region that cannot host ``Divide_k`` under the rules above."""
    if region.n <= k:
        return True
    if peel_applies(region, k):
        return region.width == k
    return split_longer(region, k) is None
