"""Boundary states bypass routing/search and work on disconnected qubits."""
import importlib

import cirq
import cirq_google
import numpy as np
import pytest


@pytest.mark.parametrize("module_name", [
    "dicke_willow.challenge", "dicke_submission", "problem2_submission",
])
@pytest.mark.parametrize("n", [1, 6, 105])
@pytest.mark.parametrize("full", [False, True])
def test_product_boundaries(module_name, n, full, monkeypatch):
    module = importlib.import_module(module_name)

    def unexpected(*args, **kwargs):
        pytest.fail("Product-state preparation entered routing or candidate search")

    for name in ("graph_from_error_rates", "graph_from_device", "route_on_path",
                 "_FINAL_ASTRA_BONUS", "_FINAL_ASTRA_MAIN", "_best"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, unexpected)
    k = n if full else 0
    candidates = [module.create_dicke_state_adaptive(n, k)]
    if hasattr(module, "create_dicke_state"):
        rates = {q: (0.001, 0) for q in cirq_google.Willow105.metadata.qubit_set}
        # No couplers: neither product state needs connectivity.
        candidates.append(module.create_dicke_state(n, k, {}, rates, rates, rates))
    for circuit, outputs in candidates:
        assert len(outputs) == n
        assert len(circuit) == int(full)
        assert len(list(circuit.all_operations())) == (n if full else 0)
        assert not circuit.has_measurements()
        cirq_google.Willow105.validate_circuit(circuit)
        if n <= 6:
            state = cirq.Simulator().simulate(circuit, qubit_order=sorted(outputs)).final_state_vector
            expected = np.zeros(2**n)
            expected[-1 if full else 0] = 1
            assert np.allclose(abs(state), expected)


@pytest.mark.parametrize("module_name", ["dicke_willow.challenge", "dicke_submission"])
def test_missing_calibration_qubits_are_not_used(module_name):
    module = importlib.import_module(module_name)
    q = min(cirq_google.Willow105.metadata.qubit_set)
    with pytest.raises(NotImplementedError):
        module.create_dicke_state(1, 0, {}, {q: (0, 0)}, {}, {})
