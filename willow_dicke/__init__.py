"""Willow-specific balanced Dicke preparation.

The topology stage builds a guillotine partition of the authoritative
device graph, a Divide staging path at every internal node, and a
nearest-neighbor SWAP certificate. Quantum rotations are a later stage:
they must not be inserted until that certificate passes.
"""

from willow_dicke.plan import build_topology_plan

__all__ = ["build_topology_plan"]
