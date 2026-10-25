"""Modular circuit-optimization passes for Dicke-state preparation."""

from collections.abc import Callable

import cirq


def native_x(qubit: cirq.Qid) -> cirq.Operation:
    """Return a native PhasedXZ Pauli-X pulse."""
    return cirq.PhasedXZGate(
        x_exponent=1.0,
        z_exponent=0.0,
        axis_phase_exponent=0.0,
    ).on(qubit)


# ==============================================================================
# Pass 1: Adjacent CZ gate-pair cancellation
# ==============================================================================


def cancel_adjacent_czs(circuit: cirq.Circuit) -> cirq.Circuit:
    """Cancel unobstructed pairs of CZ gates on the same two qubits."""
    moments = [list(moment.operations) for moment in circuit.moments]
    optimized = True

    while optimized:
        optimized = False
        cz_positions = []
        for moment_index, moment_operations in enumerate(moments):
            for operation in moment_operations:
                if operation.gate == cirq.CZ and len(operation.qubits) == 2:
                    qubit_pair = tuple(sorted(operation.qubits))
                    cz_positions.append(
                        (moment_index, operation, qubit_pair)
                    )

        index = 0
        while index < len(cz_positions) - 1:
            first_moment, first_operation, first_pair = cz_positions[index]
            second_moment, second_operation, second_pair = cz_positions[
                index + 1
            ]
            if first_pair == second_pair:
                first_qubit, second_qubit = first_pair
                can_cancel = True
                for intermediate_moment in range(
                    first_moment + 1,
                    second_moment,
                ):
                    for operation in moments[intermediate_moment]:
                        if (
                            first_qubit in operation.qubits
                            or second_qubit in operation.qubits
                        ):
                            can_cancel = False
                            break
                    if not can_cancel:
                        break

                if can_cancel:
                    moments[first_moment].remove(first_operation)
                    moments[second_moment].remove(second_operation)
                    optimized = True
                    break
            index += 1

    return cirq.drop_empty_moments(
        cirq.Circuit(
            cirq.Moment(moment)
            for moment in moments
            if moment
        )
    )


# ==============================================================================
# Pass 2: Threshold-based dynamical decoupling
# ==============================================================================


def apply_threshold_dd(
    circuit: cirq.Circuit,
    path_qubits: set[cirq.Qid],
    min_gap_moments: int = 8,
) -> cirq.Circuit:
    """Insert CPMG X-X pulses into sufficiently long idle windows."""
    moments = [
        list(moment.operations)
        for moment in cirq.Circuit(circuit).moments
    ]
    num_moments = len(moments)
    qubit_idles: dict[cirq.Qid, list[tuple[int, int]]] = {
        qubit: [] for qubit in path_qubits
    }

    for qubit in path_qubits:
        idle_start = None
        for moment_index in range(num_moments):
            is_active = any(
                qubit in operation.qubits
                for operation in moments[moment_index]
            )
            if not is_active:
                if idle_start is None:
                    idle_start = moment_index
            else:
                if idle_start is not None:
                    gap_length = moment_index - idle_start
                    if gap_length >= min_gap_moments:
                        qubit_idles[qubit].append(
                            (idle_start, moment_index)
                        )
                idle_start = None

        if (
            idle_start is not None
            and num_moments - idle_start >= min_gap_moments
        ):
            qubit_idles[qubit].append((idle_start, num_moments))

    for qubit, gaps in qubit_idles.items():
        for start, end in gaps:
            length = end - start
            first_pulse = start + length // 4
            second_pulse = start + 3 * length // 4
            if (
                first_pulse < num_moments
                and second_pulse < num_moments
                and first_pulse != second_pulse
            ):
                moments[first_pulse].append(native_x(qubit))
                moments[second_pulse].append(native_x(qubit))

    return cirq.Circuit(
        cirq.Moment(moment)
        for moment in moments
        if moment
    )


# ==============================================================================
# Master compiler and optimizer selector
# ==============================================================================


def compile_dicke_state(
    num_qubits: int,
    num_excitations: int,
    path_qubits: list[cirq.Qid],
    circuit_builder_fn: Callable[
        [list[cirq.Qid], int, int],
        cirq.Circuit,
    ],
    mode: str = "baseline_adjacent_threshold",
    dd_threshold: int = 8,
) -> cirq.Circuit:
    """Build a Dicke-state circuit and apply the selected optimization passes."""
    circuit = circuit_builder_fn(
        path_qubits,
        num_qubits,
        num_excitations,
    )
    qubit_set = set(path_qubits)

    if mode == "raw_baseline":
        return circuit
    if mode == "baseline_adjacent":
        return cancel_adjacent_czs(circuit)
    if mode == "baseline_threshold":
        return apply_threshold_dd(
            circuit,
            qubit_set,
            min_gap_moments=dd_threshold,
        )
    if mode == "baseline_adjacent_threshold":
        cz_canceled = cancel_adjacent_czs(circuit)
        return apply_threshold_dd(
            cz_canceled,
            qubit_set,
            min_gap_moments=dd_threshold,
        )

    raise ValueError(f"Unknown optimization mode: {mode!r}")
