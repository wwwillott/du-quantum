"""Split amplitudes for Divide, Eq. (7) of arXiv:2505.15413.

The integer identity is Vandermonde. Floating amplitudes use the same
log-gamma path as the rest of this repository so small and large splits
share one convention.

Bit convention used by every circuit in this package:

- A register is ordered left to right along its path.
- The unary encoding of weight ell is ``|0^{q-ell} 1^{ell}>``, ones on the right.
- One-hot encoding of ell is ``|0^{q-ell} 1 0^{ell-1}>`` (the 2019/2022 convention).
- ``Divide_k^{n,m}`` writes weight ``i`` on the size-``m`` register and
  weight ``ell - i`` on the size-``(n-m)`` register.
"""

from __future__ import annotations

import math

from dicke_willow.dicke_math import rotation_angle
from willow_dicke.math.coefficients import (
    amplitude_norm,
    binomial,
    legal_split_range,
    split_amplitude,
    vandermonde_holds,
)

__all__ = [
    "amplitude_norm",
    "binomial",
    "legal_split_range",
    "rotation_angle",
    "split_amplitude",
    "vandermonde_holds",
    "weight_distribution_amplitudes",
]


def weight_distribution_amplitudes(n: int, m: int, ell: int) -> dict[int, float]:
    """Map the size-``m`` excitation count ``i`` to its nonnegative amplitude.

    The legal window is ``max(0, ell-(n-m)) <= i <= min(ell, m)``.
    Amplitudes outside that window are omitted. Squared values sum to 1
    whenever ``C(n, ell) > 0``.
    """
    if m < 0 or m > n:
        raise ValueError(f"child size m={m} is outside 0..n={n}")
    if ell < 0 or ell > n:
        raise ValueError(f"weight ell={ell} is outside 0..n={n}")
    return {i: split_amplitude(n, m, ell, i) for i in legal_split_range(n, m, ell)}


def unary_mask(length: int, weight: int, shift: int = 0) -> int:
    """Bitmask with ``weight`` ones in the highest ``weight`` bits of a block.

    Bit ``shift`` is the left end of the block. Ones sit at the right end,
    which is the unary convention above. Bit 0 of the mask is path index 0.
    """
    if weight < 0 or weight > length:
        raise ValueError(f"unary weight {weight} is outside 0..{length}")
    mask = 0
    for offset in range(weight):
        mask |= 1 << (shift + length - 1 - offset)
    return mask


def squared_norm(amplitudes: dict[int, float]) -> float:
    return math.fsum(value * value for value in amplitudes.values())
