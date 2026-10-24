"""Deterministic Dicke-state preparation for Google Willow.

The challenge entry points match ``DuQuantum_Google_Standard.ipynb``:

* ``create_dicke_state`` — noise-aware, ancilla-free, measurement-free
* ``create_dicke_state_adaptive`` — same unitary on the static Willow105 grid
"""

from dicke_willow.challenge import create_dicke_state, create_dicke_state_adaptive

__all__ = ["create_dicke_state", "create_dicke_state_adaptive"]
