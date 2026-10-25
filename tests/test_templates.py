"""Best-known template registry tests."""

import cirq
import pytest

from dicke_compiler import (
    DickeTarget,
    EquivalenceMode,
    HardwareTarget,
    willow_target,
)
from dicke_compiler.templates import (
    DEFAULT_REGISTRY,
    CircuitTemplate,
    template_embeddings,
)
from dicke_compiler.verification import verify_dicke_preparation


def _generated_templates() -> list[CircuitTemplate]:
    return [
        template
        for num_qubits in range(2, 16)
        for num_excitations in range(1, num_qubits // 2 + 1)
        for template in DEFAULT_REGISTRY.find(
            DickeTarget(num_qubits, num_excitations)
        )
        if template.name.startswith("best_known_")
    ]


def test_registry_contains_all_google_challenge_templates() -> None:
    templates = _generated_templates()

    assert len(templates) == 56
    assert len({template.name for template in templates}) == 56
    assert all(template.preferred_qubits is not None for template in templates)


def test_all_best_known_templates_prepare_exact_dicke_states() -> None:
    for template in _generated_templates():
        report = verify_dicke_preparation(
            template.circuit,
            template.output_qubits,
            template.target,
            atol=1e-8,
            max_exact_qubits=15,
            analytic_family=template.name,
        )
        assert report.passed, template.name
        assert report.fidelity == pytest.approx(1.0, abs=1e-8)


def test_large_template_uses_preferred_willow_embedding() -> None:
    pytest.importorskip("cirq_google")
    template = next(
        template
        for template in DEFAULT_REGISTRY.find(DickeTarget(15, 7))
        if template.name == "best_known_15_7"
    )

    mappings = template_embeddings(template, willow_target(), limit=16)

    assert len(mappings) == 1
    assert tuple(
        mappings[0][logical] for logical in template.output_qubits
    ) == template.preferred_qubits


def test_generic_embedding_search_honors_budget() -> None:
    logical = tuple(cirq.LineQubit.range(3))
    template = CircuitTemplate(
        name="budget_test",
        target=DickeTarget(3, 1),
        circuit=cirq.FrozenCircuit(
            cirq.CZ(logical[0], logical[1]),
            cirq.CZ(logical[1], logical[2]),
        ),
        output_qubits=logical,
        equivalence=EquivalenceMode.STATE_PREPARATION,
        source="test",
    )
    hardware = HardwareTarget.from_edges(
        logical,
        zip(logical, logical[1:], strict=False),
    )

    assert template_embeddings(template, hardware, search_budget=1) == ()
