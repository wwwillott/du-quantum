"""Small-grid fidelity and Willow patch invariants."""

from __future__ import annotations

import cirq
import pytest

from willow_dicke.device import load_static_willow
from willow_dicke.yuan_zhang_2025.compiler import compile_patch, with_unary
from willow_dicke.yuan_zhang_2025.geometry import (
    abstract_patch,
    primary_willow_rectangle,
    shifted_willow_rectangle,
)
from willow_dicke.yuan_zhang_2025.mathematical_primitives import (
    vandermonde_holds,
    weight_distribution_amplitudes,
)
from willow_dicke.yuan_zhang_2025.simulator import dicke_fidelity, simulate
from willow_dicke.yuan_zhang_2025.verification import (
    assert_grid_cz,
    lower_willow,
    validate_willow_device,
)


def test_split_amplitudes_are_normalized_and_vandermonde():
    for n, m, ell in ((8, 3, 2), (56, 28, 3), (9, 4, 4)):
        amplitudes = weight_distribution_amplitudes(n, m, ell)
        assert sum(value * value for value in amplitudes.values()) == pytest.approx(1.0)
        assert vandermonde_holds(n, m, ell)


def test_primary_and_shifted_patches_match_willow105():
    willow = load_static_willow()
    for patch in (primary_willow_rectangle(willow), shifted_willow_rectangle(willow)):
        assert len(patch.qubits) == 56
        counts = patch.physical_edge_counts()
        assert counts == {"horizontal": 48, "vertical": 49, "total": 97}
        assert patch.height == 7 and patch.width == 8


@pytest.mark.parametrize(
    ("height", "width", "k"),
    [
        (2, 2, 1),
        (2, 3, 1),
        (2, 4, 1),
        (2, 4, 2),
        (3, 3, 1),
        (3, 3, 2),
    ],
)
def test_small_rectangles_prepare_every_weight(height, width, k):
    patch = abstract_patch(height, width)
    compiled = compile_patch(patch, k)
    lowered = lower_willow(with_unary(compiled.circuit, compiled.port, 0))
    assert_grid_cz(lowered)
    qubits = list(patch.qubits)
    for ell in range(k + 1):
        logical = with_unary(compiled.circuit, compiled.port, ell)
        vector = simulate(logical, qubits)
        assert dicke_fidelity(vector, ell) == pytest.approx(1.0, abs=1e-8)
        lowered_ell = lower_willow(logical)
        assert_grid_cz(lowered_ell)
        assert dicke_fidelity(simulate(lowered_ell, qubits), ell) == pytest.approx(1.0, abs=1e-8)


def test_willow_k1_is_device_valid():
    willow = load_static_willow()
    patch = primary_willow_rectangle(willow)
    compiled = compile_patch(patch, 1, device_graph=willow)
    lowered = lower_willow(with_unary(compiled.circuit, compiled.port, 1))
    assert_grid_cz(lowered)
    validate_willow_device(lowered)
    assert all(certificate.correct_permutation for certificate in compiled.certificates)
