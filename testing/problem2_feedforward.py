"""Problem 2 feed-forward experiments inspired by Willow mid-circuit readout.

The Science magic-state paper is used only for hardware-shaped ideas:
parallel diagonal measurement, measure-instead-of-uncompute, and Pauli-frame
feed-forward.  Those tools are applied to exact Dicke identities, not color
codes.  A candidate is kept only when it is shallower than the current
unitary incumbent, exact on every logical branch, Willow-routed, and
diagonal-legal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cirq
import cirq_google
import numpy as np
import sympy

from dicke_compiler import compile_dicke, dicke_state_vector, willow_target


def native_x(qubit: cirq.Qid) -> cirq.Operation:
    return cirq.PhasedXZGate(
        x_exponent=1.0,
        z_exponent=0.0,
        axis_phase_exponent=0.0,
    ).on(qubit)


def native_h(qubit: cirq.Qid) -> cirq.Operation:
    return cirq.PhasedXZGate(
        x_exponent=0.5,
        z_exponent=1.0,
        axis_phase_exponent=1.5,
    ).on(qubit)


def native_z(qubit: cirq.Qid) -> cirq.Operation:
    return cirq.PhasedXZGate(
        x_exponent=0.0,
        z_exponent=1.0,
        axis_phase_exponent=0.0,
    ).on(qubit)


def _device_qubits() -> set[cirq.GridQubit]:
    return set(cirq_google.Willow105.metadata.qubit_set)


def readout_line(qubit: cirq.GridQubit) -> int:
    return qubit.row + qubit.col


def full_readout_line(
    qubit: cirq.GridQubit,
    device: set[cirq.GridQubit] | None = None,
) -> tuple[cirq.GridQubit, ...]:
    device = device or _device_qubits()
    line = readout_line(qubit)
    return tuple(sorted(item for item in device if readout_line(item) == line))


def isolated_outputs(
    outputs: set[cirq.GridQubit],
) -> tuple[cirq.GridQubit, ...]:
    return tuple(
        sorted(
            qubit
            for qubit in outputs
            if sum(readout_line(other) == readout_line(qubit) for other in outputs)
            == 1
        )
    )


def diagonal_measurements_are_legal(circuit: cirq.AbstractCircuit) -> bool:
    """Every measured qubit's full Willow readout line must be measured with it."""
    device = _device_qubits()
    for moment in circuit:
        measured = {
            operation.qubits[0]
            for operation in moment.operations
            if cirq.is_measurement(operation)
        }
        if not measured:
            continue
        lines = {readout_line(qubit) for qubit in measured}
        for line in lines:
            required = {
                qubit for qubit in device if readout_line(qubit) == line
            }
            if not required.issubset(measured):
                return False
    return True


def validate_willow_adaptive(circuit: cirq.AbstractCircuit) -> None:
    device = _device_qubits()
    pairs = {
        frozenset(pair) for pair in cirq_google.Willow105.metadata.qubit_pairs
    }
    if not circuit.all_qubits().issubset(device):
        raise ValueError("circuit leaves Willow105")
    for operation in circuit.all_operations():
        bare = operation.without_classical_controls()
        if cirq.is_measurement(bare):
            continue
        if len(bare.qubits) == 2:
            if frozenset(bare.qubits) not in pairs:
                raise ValueError(f"non-adjacent CZ {bare.qubits}")
            if bare.gate != cirq.CZ:
                raise ValueError(f"non-native two-qubit gate {bare}")
        elif len(bare.qubits) == 1:
            if not isinstance(bare.gate, cirq.PhasedXZGate):
                raise ValueError(f"non-native single-qubit gate {bare}")
        else:
            raise ValueError(f"unsupported operation {bare}")
    if not diagonal_measurements_are_legal(circuit):
        raise ValueError("partial readout-line measurement is illegal")


def _unitary_incumbent(num_qubits: int, num_excitations: int) -> tuple[
    cirq.Circuit,
    tuple[cirq.GridQubit, ...],
    int,
]:
    result = compile_dicke(
        num_qubits,
        num_excitations,
        hardware=willow_target(),
        path_limit=16,
    )
    outputs = tuple(
        sorted(result.output_qubits, key=lambda qubit: (qubit.row, qubit.col))
    )
    return cirq.Circuit(result.circuit), outputs, result.after_metrics.moments


def _statevector(
    circuit: cirq.AbstractCircuit,
    qubits: tuple[cirq.Qid, ...],
) -> np.ndarray:
    return cirq.Simulator(dtype=np.complex128).simulate(
        circuit,
        qubit_order=qubits,
    ).final_state_vector


def _dicke_fidelity(state: np.ndarray, num_qubits: int, num_excitations: int) -> float:
    from dicke_compiler.models import DickeTarget

    ideal = dicke_state_vector(DickeTarget(num_qubits, num_excitations))
    return float(abs(np.vdot(ideal, state)) ** 2)


def verify_complement_from_parent(
    parent_circuit: cirq.AbstractCircuit,
    parent_outputs: tuple[cirq.GridQubit, ...],
    measured: cirq.GridQubit,
    *,
    atol: float = 1e-8,
) -> float:
    """Project a balanced parent D(N+1,k+1) and complement the zero branch."""
    parent_n = len(parent_outputs)
    parent_k = parent_n // 2
    if parent_n != 2 * parent_k:
        raise ValueError("complement identity requires a balanced parent")
    child_n = parent_n - 1
    child_k = parent_k - 1
    remaining = tuple(qubit for qubit in parent_outputs if qubit != measured)
    index = parent_outputs.index(measured)
    state = _statevector(parent_circuit, parent_outputs).reshape((2,) * parent_n)
    # Cirq statevector index 0 is the first qubit in qubit_order.
    axis = index
    fidelities = []
    for bit, apply_all_x in ((0, True), (1, False)):
        projected = np.take(state, bit, axis=axis).copy()
        norm = np.linalg.norm(projected)
        if norm < atol:
            raise ValueError(f"vanishing projection on measure bit {bit}")
        projected /= norm
        if apply_all_x:
            for child_axis in range(child_n):
                projected = np.flip(projected, axis=child_axis)
        vector = projected.reshape(-1)
        fidelities.append(_dicke_fidelity(vector, child_n, child_k))
    worst = min(fidelities)
    if worst < 1.0 - atol:
        raise ValueError(f"complement branches are not exact: {fidelities}")
    return worst


def attach_diagonal_complement(
    parent_circuit: cirq.AbstractCircuit,
    parent_outputs: set[cirq.GridQubit],
    measured: cirq.GridQubit,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Measure a full readout line and X-correct the zero branch."""
    if measured not in parent_outputs:
        raise ValueError("measured qubit must be a parent output")
    device = _device_qubits()
    line = full_readout_line(measured, device)
    if set(parent_circuit.all_qubits()) & set(line) - {measured}:
        raise ValueError("readout line intersects other active parent qubits")
    remaining = set(parent_outputs) - {measured}
    key = f"complement_{measured.row}_{measured.col}"
    circuit = cirq.Circuit(parent_circuit)
    circuit.append(
        cirq.Moment(
            cirq.measure(qubit, key=f"line_{qubit.row}_{qubit.col}")
            if qubit != measured
            else cirq.measure(qubit, key=key)
            for qubit in line
        )
    )
    condition = cirq.SympyCondition(sympy.Eq(sympy.Symbol(key), 0))
    circuit.append(
        cirq.Moment(
            native_x(qubit).with_classical_controls(condition)
            for qubit in sorted(remaining)
        )
    )
    validate_willow_adaptive(circuit)
    return circuit, remaining


def measure_uncompute_parity_gadget(
    data: tuple[cirq.GridQubit, cirq.GridQubit],
    ancilla: cirq.GridQubit,
) -> cirq.Circuit:
    """Paper-style measure-instead-of-uncompute parity onto an isolated ancilla.

    Computes Z(data[0])Z(data[1]) onto the ancilla, measures the whole
    readout line, and applies a feed-forward Z.  This is a depth probe, not
    a Dicke compiler by itself.
    """
    left, right = data
    circuit = cirq.Circuit()
    circuit.append(native_h(ancilla))
    circuit.append([native_h(right), cirq.CZ(left, right), native_h(right)])
    circuit.append([native_h(ancilla), cirq.CZ(right, ancilla), native_h(ancilla)])
    circuit.append([native_h(right), cirq.CZ(left, right), native_h(right)])
    line = full_readout_line(ancilla)
    key = f"parity_{ancilla.row}_{ancilla.col}"
    circuit.append(
        cirq.Moment(
            cirq.measure(qubit, key=key if qubit == ancilla else f"idle_{qubit.row}_{qubit.col}")
            for qubit in line
        )
    )
    circuit.append(
        native_z(left).with_classical_controls(key)
    )
    validate_willow_adaptive(circuit)
    return circuit


@dataclass(frozen=True)
class AdaptiveCandidate:
    num_qubits: int
    num_excitations: int
    family: str
    moments: int
    unitary_moments: int
    fidelity: float
    wins: bool


def try_complement(num_qubits: int, num_excitations: int) -> AdaptiveCandidate | None:
    """D(2k+1, k) from D(2k+2, k+1) plus one diagonal measurement and X-frame."""
    if num_qubits != 2 * num_excitations + 1:
        return None
    parent_n = num_qubits + 1
    parent_k = num_excitations + 1
    if parent_n > 15:
        return None
    parent, parent_outputs, parent_moments = _unitary_incumbent(parent_n, parent_k)
    _, _, unitary_moments = _unitary_incumbent(num_qubits, num_excitations)
    isolated = isolated_outputs(set(parent_outputs))
    if not isolated:
        return None
    measured = isolated[0]
    verify_complement_from_parent(parent, parent_outputs, measured)
    adaptive, _outputs = attach_diagonal_complement(
        parent,
        set(parent_outputs),
        measured,
    )
    return AdaptiveCandidate(
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        family="measure_complement_from_balanced_parent",
        moments=len(adaptive),
        unitary_moments=unitary_moments,
        fidelity=1.0,
        wins=len(adaptive) < unitary_moments,
    )


def scan_complement_domain() -> list[AdaptiveCandidate]:
    results = []
    for num_qubits in range(3, 16, 2):
        num_excitations = num_qubits // 2
        candidate = try_complement(num_qubits, num_excitations)
        if candidate is not None:
            results.append(candidate)
    return results


def gadget_depth_probe() -> dict[str, int]:
    left = cirq.GridQubit(4, 4)
    right = cirq.GridQubit(4, 5)
    ancilla = cirq.GridQubit(5, 5)
    gadget = measure_uncompute_parity_gadget((left, right), ancilla)
    unitary = cirq.Circuit(
        native_h(right),
        cirq.CZ(left, right),
        native_h(right),
        native_h(right),
        cirq.CZ(left, right),
        native_h(right),
    )
    return {
        "measure_uncompute_moments": len(gadget),
        "direct_cnot_uncompute_moments": len(unitary),
    }


def main() -> None:
    print("=== Measure-uncompute gadget probe ===")
    probe = gadget_depth_probe()
    print(probe)
    print("\n=== Complement construction vs unitary incumbents ===")
    print(
        f"{'D(N,n)':<10} {'adaptive':>8} {'unitary':>8} {'delta':>8} {'keep':>6}"
    )
    wins = []
    for candidate in scan_complement_domain():
        delta = candidate.moments - candidate.unitary_moments
        keep = "YES" if candidate.wins else "no"
        print(
            f"D({candidate.num_qubits},{candidate.num_excitations})"
            f"{'':<4} {candidate.moments:>8} {candidate.unitary_moments:>8} "
            f"{delta:>+8} {keep:>6}"
        )
        if candidate.wins:
            wins.append(candidate)
    print(f"\nWinning adaptive cases: {len(wins)}")
    for candidate in wins:
        print(
            f"  D({candidate.num_qubits},{candidate.num_excitations}): "
            f"{candidate.unitary_moments} -> {candidate.moments}"
        )


if __name__ == "__main__":
    main()
