"""2025 Yuan–Zhang balanced Dicke compiler on a Willow rectangle.

This package is independent of the 105-qubit guillotine compiler. It follows
Yuan and Zhang, arXiv:2505.15413, Section 3.2: balanced Divide, a structured
block permutation, then parallel child recursion. The Divide primitive is the
ancilla-free weight-distribution block of Bärtschi and Eidenbenz (2022),
scheduled on a path of 2k qubits (Lemma 8), not a per-gate SWAP bridge.
"""

METHODOLOGY_VERSION = "yz2025-2"

__all__ = ["METHODOLOGY_VERSION"]
