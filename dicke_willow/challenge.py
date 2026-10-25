"""Challenge entry points for ``DuQuantum_Google_Standard.ipynb``.

The hardware function returns ``PhasedXZGate`` and full CZ only.
Problem 2 additionally permits measurements and classically controlled
single-qubit corrections. Its independent depth portfolio is described in
``interference_dicke/problem2/README.md``.
The hardware unitary is the ancilla-free Split & Cyclic Shift preparation of
Bärtschi & Eidenbenz (arXiv:1904.07358). It is placed on a path of the
device graph and SWAP-routed along that path, so every CZ is a coupler the
caller actually listed.

The balanced 2025 divide (``divide.py``) is available for the research
comparison. It is not the default challenge circuit: moving its hot
registers adds two-qubit gates, and the notebook scores hardware noise.
"""

from __future__ import annotations

import cirq
import cirq_google
import networkx as nx

from dicke_willow.compile import (
    MOMENT_LIMIT,
    assert_challenge_gates,
    finalize,
    gate_counts,
    to_willow_gates,
)
from dicke_willow.placement import place_path, snake_path
from dicke_willow.scs import logical_dicke_circuit
from dicke_willow.topology import graph_from_device, graph_from_error_rates


def route_on_path(path: list[cirq.Qid], num_excitations: int, *, decouple: bool) -> cirq.Circuit:
    """Compile the logical stair, then SWAP-route it along ``path``.

    The path qubits are already in stair order: logical neighbor i touches
    logical neighbor i+1. ``RouteCQC`` only inserts a SWAP when a stair
    control is not on that edge, and every SWAP uses a path edge.
    """
    logical = logical_dicke_circuit(list(path), num_excitations)
    native = to_willow_gates(logical)
    graph = nx.Graph()
    graph.add_nodes_from(path)
    for left, right in zip(path, path[1:]):
        graph.add_edge(left, right)
    routed = cirq.RouteCQC(graph)(native)
    return finalize(routed, decouple=decouple)


def _cz_pair_set(cz_pauli_error_rate: dict) -> set[tuple[cirq.Qid, cirq.Qid]]:
    pairs = set()
    for key in cz_pauli_error_rate:
        a, b = (key[0], key[1]) if not isinstance(key, frozenset) else tuple(key)
        pairs.add((a, b))
    return pairs


def _product_state(qubits, num_qubits, num_excitations):
    """Zero depth for vacuum, one parallel native-X layer for full weight."""
    outputs = set(sorted(qubits)[:num_qubits])
    if len(outputs) != num_qubits:
        raise NotImplementedError("not enough available qubits")
    circuit = cirq.Circuit()
    if num_excitations:
        gate = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)
        circuit.append(cirq.Moment(gate.on(q) for q in sorted(outputs)))
    return circuit, outputs


def create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a Dicke state.

    Args:
        num_qubits: The number, N, of qubits in the Dicke state.
        num_excitations: The number, n, of 1s in the state.
        cz_pauli_error_rate: Measured CZ Pauli error rates and their uncertainties.
        sq_pauli_error_rate: Measured single-qubit Pauli error rates.
        zero_errors: Measured 0 -> 1 readout error rates.
        one_errors: Measured 1 -> 0 readout error rates.

    Returns:
        The circuit and the set of qubits that hold the Dicke state.

    Raises:
        NotImplementedError: The requested size does not fit on the live
            coupling graph, or the compiled circuit exceeds 1750 moments.
    """
    if num_qubits < 1 or num_excitations < 0 or num_excitations > num_qubits:
        raise NotImplementedError(
            f"unsupported Dicke parameters N={num_qubits}, n={num_excitations}"
        )
    if num_excitations in (0, num_qubits):
        available = set(sq_pauli_error_rate) & set(zero_errors) & set(one_errors)
        available &= set(cirq_google.Willow105.metadata.qubit_set)
        return _product_state(available, num_qubits, num_excitations)
    adjacency = graph_from_error_rates(
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )
    path = place_path(
        adjacency,
        num_qubits,
        num_excitations,
        cz_errors=cz_pauli_error_rate,
        sq_errors=sq_pauli_error_rate,
        zero_errors=zero_errors,
        one_errors=one_errors,
    )
    if path is None or len(path) != num_qubits:
        raise NotImplementedError(
            f"no connected path of {num_qubits} qubits in the supplied error map"
        )
    circuit = route_on_path(path, num_excitations, decouple=True)
    if len(circuit) > MOMENT_LIMIT:
        raise NotImplementedError(
            f"compiled circuit has {len(circuit)} moments, above the limit of {MOMENT_LIMIT}"
        )
    qubits = set(path)
    assert_challenge_gates(
        circuit,
        qubits=qubits,
        cz_pairs=_cz_pair_set(cz_pauli_error_rate),
        sq_qubits=set(sq_pauli_error_rate),
    )
    return circuit, qubits


def create_dicke_state_adaptive(
    num_qubits: int, num_excitations: int
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Depth-scored Problem 2 portfolio, including exact adaptive candidates.

    N=2..20 uses the independently checked integration in
    interference_dicke.problem2. Readout uses complete Willow diagonals.
    The original unitary path is retained outside the scored domain.
    """
    if num_qubits < 1 or num_excitations < 0 or num_excitations > num_qubits:
        raise NotImplementedError(
            f"unsupported Dicke parameters N={num_qubits}, n={num_excitations}"
        )
    if num_qubits > len(cirq_google.Willow105.metadata.qubit_set):
        raise NotImplementedError(f"Willow105 has fewer than {num_qubits} qubits")
    if num_excitations in (0, num_qubits):
        return _product_state(cirq_google.Willow105.metadata.qubit_set,
                              num_qubits, num_excitations)
    if 2 <= num_qubits <= 20 and 1 <= num_excitations <= num_qubits // 2:
        from interference_dicke.problem2 import create_dicke_state_adaptive as prepare
        return prepare(num_qubits, num_excitations)
    adjacency = graph_from_device(cirq_google.Willow105)
    path = snake_path(adjacency, num_qubits)
    if path is None:
        raise NotImplementedError(f"no Willow105 path of length {num_qubits}")
    # The bonus score divides by the number of moments, so decoupling is left
    # off. The routed stair uses only Willow105 edges.
    circuit = route_on_path(path, num_excitations, decouple=False)
    cirq_google.Willow105.validate_circuit(circuit)
    return circuit, set(path)


def circuit_report(circuit: cirq.Circuit, qubits: set, num_excitations: int) -> dict:
    """Numbers printed before any hardware submission."""
    counts = gate_counts(circuit)
    counts["num_qubits"] = len(qubits)
    counts["num_excitations"] = num_excitations
    counts["qubits"] = sorted(qubits, key=str)
    return counts
