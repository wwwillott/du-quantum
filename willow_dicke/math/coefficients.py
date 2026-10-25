"""Stable split amplitudes.

The identity implemented here is geometry-independent:

    |D_ell(S)> = sum_i sqrt( C(|A|, i) C(|B|, ell - i) / C(|S|, ell) )
                |D_i(A)> |D_{ell - i}(B)>

with i in max(0, ell - |B|) .. min(ell, |A|). Integer binomials are exact.
Log-gamma is used only when a probability is turned into a floating amplitude.
"""

from __future__ import annotations

import math


def binomial(n: int, r: int) -> int:
    """Exact C(n, r). Zero outside 0..n."""
    if r < 0 or n < 0 or r > n:
        return 0
    return math.comb(n, r)


def log_binom(n: int, r: int) -> float:
    """Natural log of C(n, r). -inf when the coefficient is zero."""
    if r < 0 or n < 0 or r > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)


def legal_split_range(n: int, m: int, ell: int) -> range:
    """Legal excitation counts on the size-m side of an n-qubit region."""
    if m < 0 or m > n:
        raise ValueError(f"child size m={m} is outside 0..n={n}")
    if ell < 0 or ell > n:
        raise ValueError(f"weight ell={ell} is outside 0..n={n}")
    return range(max(0, ell - (n - m)), min(ell, m) + 1)


def split_log_probability(n: int, m: int, ell: int, i: int) -> float:
    """Log of C(m, i) C(n - m, ell - i) / C(n, ell)."""
    return log_binom(m, i) + log_binom(n - m, ell - i) - log_binom(n, ell)


def split_amplitude(n: int, m: int, ell: int, i: int) -> float:
    """Nonnegative amplitude sqrt of the split probability. Zero when illegal."""
    log_p = split_log_probability(n, m, ell, i)
    if log_p == float("-inf"):
        return 0.0
    return math.exp(0.5 * log_p)


def amplitude_norm(n: int, m: int, ell: int) -> float:
    """Sum of squared amplitudes over legal i. Equals 1 when C(n, ell) > 0."""
    return sum(split_amplitude(n, m, ell, i) ** 2 for i in legal_split_range(n, m, ell))


def vandermonde_holds(n: int, m: int, ell: int) -> bool:
    """Exact integer check of sum_i C(m, i) C(n - m, ell - i) = C(n, ell)."""
    total = sum(binomial(m, i) * binomial(n - m, ell - i) for i in legal_split_range(n, m, ell))
    return total == binomial(n, ell)
