"""Tolerances and methodology identity for the 2025 compiler."""

from __future__ import annotations

from willow_dicke.yuan_zhang_2025 import METHODOLOGY_VERSION

PAPER = "arxiv:2505.15413"
ALGORITHM = "yuan_zhang_2025_grid"

# Noiseless subspace comparisons. Relative phases use the same tolerance.
AMPLITUDE_ATOL = 1e-10
FIDELITY_ATOL = 1e-10

# Full statevector is refused at or above this width.
MAX_STATEVECTOR_QUBITS = 22

# Divide isolation benchmark. Both children must have at least k qubits.
DIVIDE_BENCHMARK_KS = (1, 2, 3, 4, 5)

# Willow rectangle benchmarks requested by the specification.
WILLOW_BENCHMARK_KS = (1, 2, 3, 4, 5, 8, 10)

METHODOLOGY = METHODOLOGY_VERSION
