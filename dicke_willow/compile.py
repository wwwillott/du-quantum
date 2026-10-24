"""Compile a logical circuit to the gates Willow actually runs.

The logical layers may contain Ry, CNOT, SWAP, and controlled rotations.
Those are decomposed with Cirq's CZ target gateset, which emits
``PhasedXZGate`` and ``CZPowGate`` and nothing else. Dynamical decoupling
fills idle slots with Pauli X; those Paulis are compiled again so the
circuit handed to the device is still only PhasedXZ and CZ.

Moment count is checked against the challenge limit of 1750. Decoupling is
kept only when it does not push the circuit over that limit.
"""

from __future__ import annotations

import cirq

MOMENT_LIMIT = 1750


def pack(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """Greedy earliest-moment scheduling. Disjoint gates share a moment."""
    packed = cirq.Circuit()
    packed.append(list(circuit.all_operations()), strategy=cirq.InsertStrategy.EARLIEST)
    return packed


def compress(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """Merge adjacent single-qubit gates, then return to PhasedXZ + CZ.

    Routing leaves runs of PhasedXZ on the same qubit in separate moments.
    Folding each run into one PhasedXZ removes those moments without changing
    the unitary. This is what keeps the N=20 stair under the 1750-moment cap.
    """
    merged = cirq.merge_k_qubit_unitaries(circuit, k=1)
    native = cirq.optimize_for_target_gateset(merged, gateset=cirq.CZTargetGateset())
    return pack(native)


def to_willow_gates(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    native = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    return pack(native)


def with_dynamical_decoupling(circuit: cirq.Circuit) -> cirq.Circuit:
    """Insert decoupling pulses, then return to PhasedXZ + CZ."""
    decoupled = cirq.add_dynamical_decoupling(circuit)
    return compress(to_willow_gates(decoupled))


def finalize(circuit: cirq.AbstractCircuit, *, decouple: bool = True) -> cirq.Circuit:
    """Native gates, packed, optionally decoupled, never over the moment limit.

    If decoupling would exceed 1750 moments the undecoupled circuit is
    returned. If that is also over the limit the caller decides whether to
    raise ``NotImplementedError``.
    """
    native = compress(to_willow_gates(circuit))
    if not decouple:
        return native
    if len(native) > MOMENT_LIMIT:
        return native
    decoupled = with_dynamical_decoupling(native)
    if len(decoupled) > MOMENT_LIMIT:
        return native
    return decoupled


def gate_counts(circuit: cirq.AbstractCircuit) -> dict[str, int]:
    cz = 0
    single = 0
    other = 0
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.CZPowGate):
            cz += 1
        elif isinstance(op.gate, cirq.PhasedXZGate):
            single += 1
        else:
            other += 1
    return {
        "moments": len(circuit),
        "cz": cz,
        "single_qubit": single,
        "other": other,
    }


def assert_challenge_gates(
    circuit: cirq.AbstractCircuit,
    *,
    qubits: set[cirq.Qid],
    cz_pairs: set[tuple[cirq.Qid, cirq.Qid]],
    sq_qubits: set[cirq.Qid],
) -> None:
    """Raise if the circuit uses a gate, qubit, or coupler the notebook forbids."""
    allowed_pairs = set()
    for a, b in cz_pairs:
        allowed_pairs.add(frozenset((a, b)))
    for op in circuit.all_operations():
        gate = op.gate
        if isinstance(gate, cirq.PhasedXZGate):
            qubit = op.qubits[0]
            if qubit not in sq_qubits:
                raise ValueError(f"single-qubit gate on disabled qubit {qubit}")
            continue
        if isinstance(gate, cirq.CZPowGate) and abs(gate.exponent - 1) < 1e-9:
            pair = frozenset(op.qubits)
            if pair not in allowed_pairs:
                raise ValueError(f"CZ on a coupler that is not in the error map: {op.qubits}")
            if not set(op.qubits) <= qubits:
                raise ValueError(f"CZ leaves the selected patch: {op.qubits}")
            continue
        raise ValueError(f"gate {gate} is not a Willow PhasedXZ or CZ")
    if not circuit.all_qubits() <= qubits:
        raise ValueError("circuit uses a qubit outside the selected set")
