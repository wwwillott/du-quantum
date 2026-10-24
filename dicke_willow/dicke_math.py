"""Binomial amplitudes for the Dicke divide identity.

The split used by Bärtschi & Eidenbenz (2022, Def. 2) and Yuan & Zhang
(2025) is Vandermonde's identity in amplitude form:

    |D_ell^n> = sum_i sqrt( C(m, i) C(n - m, ell - i) / C(n, ell) )
                |D_i^m> |D_{ell - i}^{n - m}>

with i ranging over max(0, ell - (n - m)) .. min(ell, m).

Binomial coefficients are exact Python integers (``math.comb``). Amplitudes
for large arguments use log-gamma so the square root never overflows.
"""

from __future__ import annotations

import math


def binomial(n: int, k: int) -> int:
    """Exact binomial coefficient. Zero when k is outside 0..n."""
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)


def legal_split_range(n: int, m: int, ell: int) -> range:
    """Legal excitation counts i on the size-m child."""
    if m < 0 or m > n:
        raise ValueError(f"split m={m} is outside 0..n={n}")
    if ell < 0 or ell > n:
        raise ValueError(f"weight ell={ell} is outside 0..n={n}")
    lo = max(0, ell - (n - m))
    hi = min(ell, m)
    return range(lo, hi + 1)


def split_weights(n: int, m: int, ell: int) -> list[tuple[int, int]]:
    """Unnormalized integer weights C(m, i) C(n - m, ell - i) for legal i."""
    return [
        (i, binomial(m, i) * binomial(n - m, ell - i))
        for i in legal_split_range(n, m, ell)
    ]


def split_amplitudes(n: int, m: int, ell: int) -> list[tuple[int, float]]:
    """Real nonnegative amplitudes for each legal child weight i.

    Uses log-gamma. The squared amplitudes sum to 1 whenever C(n, ell) > 0,
    because the integer weights sum to C(n, ell) by Vandermonde.
    """
    weights = split_weights(n, m, ell)
    total = sum(weight for _, weight in weights)
    if total == 0:
        return []
    # math.lgamma is accurate for the arguments we pass (nonnegative ints).
    log_total = math.lgamma(total + 1) - math.lgamma(total)  # log(total), stable
    # log(total) for a Python int can exceed float range; fall back to the ratio.
    amplitudes: list[tuple[int, float]] = []
    if total <= 10**12:
        scale = math.sqrt(total)
        for i, weight in weights:
            amplitudes.append((i, math.sqrt(weight) / scale))
        return amplitudes
    for i, weight in weights:
        if weight == 0:
            amplitudes.append((i, 0.0))
            continue
        log_amp = 0.5 * (
            math.lgamma(weight + 1) - math.lgamma(weight) - log_total
        )
        amplitudes.append((i, math.exp(log_amp)))
    return amplitudes


def amplitude_norm(n: int, m: int, ell: int) -> float:
    """Sum of squared split amplitudes. Equals 1 when the split is possible."""
    return sum(amp * amp for _, amp in split_amplitudes(n, m, ell))


def rotation_angle(keep_probability: float) -> float:
    """Ry angle that maps |0> to sqrt(p)|0> + sqrt(1-p)|1>.

    Cirq's ``ry(theta)`` is exp(-i theta Y / 2), so
    ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>.
    Setting theta/2 = arccos(sqrt(p)) gives theta = 2 arccos(sqrt(p)).

    This is the same convention as Bärtschi & Eidenbenz 2019, Sec. 2.2:
    Ry(2 arccos sqrt(ell / n)) sends |0> to sqrt(ell/n)|0> + sqrt(1-ell/n)|1>.
    """
    if keep_probability < 0.0 or keep_probability > 1.0:
        raise ValueError(f"probability {keep_probability} is outside [0, 1]")
    if keep_probability == 1.0:
        return 0.0
    if keep_probability == 0.0:
        return math.pi
    return 2.0 * math.acos(math.sqrt(keep_probability))


def suffix_stop_probabilities(weights: list[int]) -> list[float]:
    """Turn integer branch weights into the unary-preparation stop probabilities.

    The i-th value is the probability of writing a 0 (stopping) at unary
    position i, conditioned on having already written i ones. Yuan & Zhang
    and Bärtschi & Eidenbenz both build the weighted unary state from these
    suffix ratios: p_i = x_i / sum_{j >= i} x_j.
    """
    remaining = sum(weights)
    probabilities: list[float] = []
    for weight in weights:
        if remaining <= 0:
            probabilities.append(1.0)
            continue
        probabilities.append(weight / remaining)
        remaining -= weight
    return probabilities
