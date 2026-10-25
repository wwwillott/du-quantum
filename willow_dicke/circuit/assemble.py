"""Build the 105-qubit circuit and record only checks that actually ran."""

from __future__ import annotations

import math
from pathlib import Path

import cirq
import cirq_google

from willow_dicke.circuit.lower import (
    assert_cz_and_single_qubit,
    assert_cz_on_graph,
    is_full_cz,
    lower_to_willow_cz,
)
from willow_dicke.circuit.recursive import BuiltCircuit, build_root_dicke_circuit
from willow_dicke.device import load_static_willow
from willow_dicke.plan import build_topology_plan
from willow_dicke.verify.sparse_sim import SparseState


def cz_depth(circuit: cirq.AbstractCircuit) -> int:
    return sum(1 for moment in circuit if any(is_full_cz(op) for op in moment))


def gate_tally(circuit: cirq.AbstractCircuit) -> dict:
    single = 0
    cz = 0
    per_qubit: dict[cirq.Qid, int] = {}
    for op in circuit.all_operations():
        if is_full_cz(op):
            cz += 1
            for qubit in op.qubits:
                per_qubit[qubit] = per_qubit.get(qubit, 0) + 1
        elif cirq.num_qubits(op) == 1:
            single += 1
    return {
        "moments": len(circuit),
        "cz": cz,
        "cz_depth": cz_depth(circuit),
        "single_qubit": single,
        "max_cz_on_one_qubit": max(per_qubit.values()) if per_qubit else 0,
    }


def assemble(k: int, *, leaf_factor: int = 2):
    """Partition, insert Divide and leaves, lower to CZ, validate on Willow105."""
    willow = load_static_willow()
    plan = build_topology_plan(willow, k, leaf_factor=leaf_factor)
    if not plan.passed:
        raise RuntimeError(plan.failures)
    built = build_root_dicke_circuit(willow, plan.root, k)
    logical_counts = gate_tally(built.circuit)
    final = lower_to_willow_cz(built.circuit)
    assert_cz_and_single_qubit(final)
    assert_cz_on_graph(final, willow)
    cirq_google.Willow105.validate_circuit(final)
    final_counts = gate_tally(final)
    report = {
        "device": "Willow105",
        "n": 105,
        "k": k,
        "leaf_factor": leaf_factor,
        "quantum_rotations_inserted": True,
        "compiled_circuit_device_valid": True,
        "divide_logical_depth": built.divide_logical_depth,
        "divide_cz_depth": built.divide_cz_depth,
        "routing_swap_depth": built.routing_swap_depth,
        "routing_cz_depth": built.routing_cz_depth,
        "leaf_logical_depth": built.leaf_logical_depth,
        "leaf_cz_depth": built.leaf_cz_depth,
        "logical": logical_counts,
        "compiled": final_counts,
        "partition": plan.metrics,
    }
    return willow, final, report, built


def verify_fixed_weight(circuit: cirq.Circuit, qubits: list[cirq.Qid], weight: int) -> dict:
    """Check every weight-k basis amplitude equals 1/sqrt(C(n, k))."""
    ordered = list(qubits)
    state = SparseState(len(ordered))
    # The circuit is applied to |0...0>. build_root already includes the X gates.
    state.run(circuit, ordered)
    target = 1.0 / math.sqrt(math.comb(len(ordered), weight))
    seen = 0
    worst = 0.0
    leaked = 0.0
    for basis, amplitude in state.amplitudes.items():
        if basis.bit_count() == weight:
            worst = max(worst, abs(abs(amplitude) - target))
            seen += 1
        else:
            leaked += abs(amplitude) ** 2
    expected = math.comb(len(ordered), weight)
    return {
        "weight": weight,
        "basis_states_found": seen,
        "basis_states_expected": expected,
        "max_abs_amplitude_error": worst,
        "leaked_probability": leaked,
        "passed": seen == expected and worst < 1e-6 and leaked < 1e-8,
    }


def write_report(report: dict, directory: Path) -> None:
    import json

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "PROOF_CERTIFICATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
