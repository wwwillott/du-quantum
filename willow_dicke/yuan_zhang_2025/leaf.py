"""Lemma 7 leaf: the 2019 SCS unitary on a Hamiltonian path of the region.

The leaf must accept every unary weight its parent can send, so the stair is
``U_{s, q}`` with ``q`` the parent's capacity, not a circuit built for one
measured excitation count. A one-qubit region is already ``|D_ell^1>``.
"""

from __future__ import annotations

import cirq

from dicke_willow.scs import scs_unitary_ops
from willow_dicke.rectangle import Rectangle, assert_path
from willow_dicke.yuan_zhang_2025.partition import peel_applies
from willow_dicke.yuan_zhang_2025.placement import snake, strip_path
from willow_dicke.yuan_zhang_2025.scheduler import pack


def leaf_path(region: Rectangle, k: int) -> tuple[cirq.GridQubit, ...]:
    if peel_applies(region, k) and region.width == k:
        path = strip_path(region)
    else:
        path = snake(region)
    assert_path(path)
    if len(path) != region.n:
        raise RuntimeError("leaf path does not cover the region")
    return path


def compile_leaf(region: Rectangle, k: int) -> cirq.Circuit:
    path = leaf_path(region, k)
    if region.n == 1 or k == 0:
        return cirq.Circuit()
    ops = scs_unitary_ops(list(path), min(k, region.n))
    return pack(ops)
