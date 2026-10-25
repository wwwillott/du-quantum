"""Independent Problem 1 circuit families and a bounded portfolio selector."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cirq
import cirq_google
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dicke_submission as runtime_submission
import ideation
import midtime_problem1
from modular_optimizations import (
    apply_threshold_dd,
    cancel_adjacent_czs,
    compress_moments,
    fuse_single_qubit_gates,
)


ErrorValue = float | tuple[float, float]
ErrorMap = Mapping[object, ErrorValue]
CreateFunction = Callable[..., tuple[cirq.Circuit, set[cirq.GridQubit]]]


@dataclass(frozen=True)
class Candidate:
    """One validated Problem 1 circuit candidate."""

    family: str
    circuit: cirq.Circuit
    outputs: set[cirq.GridQubit]
    calibration_cost: float
    construction_seconds: float


def _risk_rate(
    error_map: ErrorMap,
    key: object,
    default: float,
    uncertainty_weight: float,
) -> float:
    value = error_map.get(key, default)
    if isinstance(value, tuple):
        mean, uncertainty = value
    else:
        mean, uncertainty = value, 0.0
    return float(
        np.clip(mean + uncertainty_weight * uncertainty, 0.0, 1.0 - 1e-12)
    )


def _negative_log_survival(error: float) -> float:
    return -math.log(max(1.0 - error, 1e-12))


def circuit_calibration_cost(
    circuit: cirq.Circuit,
    outputs: set[cirq.GridQubit],
    num_excitations: int,
    cz_errors: ErrorMap,
    sq_errors: ErrorMap,
    zero_errors: ErrorMap,
    one_errors: ErrorMap,
    uncertainty_weight: float = 1.0,
) -> float:
    """Estimate aggregate circuit risk from actual gate traffic and readout."""
    cost = 0.0
    for operation in circuit.all_operations():
        if isinstance(operation.gate, cirq.PhasedXZGate):
            error = _risk_rate(
                sq_errors,
                operation.qubits[0],
                0.001,
                uncertainty_weight,
            )
        elif operation.gate == cirq.CZ:
            first, second = operation.qubits
            error = _risk_rate(
                cz_errors,
                (first, second),
                _risk_rate(
                    cz_errors,
                    (second, first),
                    0.01,
                    uncertainty_weight,
                ),
                uncertainty_weight,
            )
        else:
            continue
        cost += _negative_log_survival(error)

    excitation_probability = num_excitations / len(outputs)
    for qubit in outputs:
        zero_to_one = _risk_rate(
            zero_errors,
            qubit,
            0.01,
            uncertainty_weight,
        )
        one_to_zero = _risk_rate(
            one_errors,
            qubit,
            0.01,
            uncertainty_weight,
        )
        effective_readout_error = (
            (1.0 - excitation_probability) * zero_to_one
            + excitation_probability * one_to_zero
        )
        cost += _negative_log_survival(effective_readout_error)
    return cost


def validate_problem1_candidate(
    circuit: cirq.Circuit,
    outputs: set[cirq.GridQubit],
    num_qubits: int,
    num_excitations: int,
    construction_seconds: float,
    cz_errors: ErrorMap,
    sq_errors: ErrorMap,
    verify_fidelity: bool = False,
) -> list[str]:
    """Return deterministic violations of the Problem 1 contract."""
    violations: list[str] = []
    active_qubits = set(sq_errors)
    active_couplers = {frozenset(pair) for pair in cz_errors}

    if not isinstance(circuit, cirq.Circuit):
        return ["candidate is not a cirq.Circuit"]
    if not isinstance(outputs, set):
        violations.append("outputs are not a set")
    if len(outputs) != num_qubits:
        violations.append(
            f"returned {len(outputs)} outputs, expected {num_qubits}"
        )
    if not outputs.issubset(active_qubits):
        violations.append("outputs include unavailable qubits")
    if not outputs.issubset(circuit.all_qubits()):
        violations.append("outputs are not all present in the circuit")
    if len(circuit) > 1750:
        violations.append(f"depth {len(circuit)} exceeds 1750")
    if construction_seconds >= 60.0:
        violations.append(
            f"construction took {construction_seconds:.3f}s"
        )

    for operation in circuit.all_operations():
        if getattr(operation, "classical_controls", ()):
            violations.append("classically controlled gate is unsupported")
        if isinstance(operation.gate, cirq.PhasedXZGate):
            if operation.qubits[0] not in active_qubits:
                violations.append("single-qubit gate uses unavailable qubit")
        elif operation.gate == cirq.CZ:
            if frozenset(operation.qubits) not in active_couplers:
                violations.append("CZ uses unavailable coupler")
        elif isinstance(operation.gate, cirq.ResetChannel):
            if any(qubit not in active_qubits for qubit in operation.qubits):
                violations.append("reset uses unavailable qubit")
        else:
            violations.append(f"unsupported operation: {operation!r}")

    try:
        cirq_google.Willow105.validate_circuit(circuit)
    except ValueError as error:
        violations.append(f"Willow validation failed: {error}")

    if verify_fidelity and not violations:
        fidelity = ideation.calculate_state_fidelity(
            circuit,
            num_excitations,
            outputs,
            dtype=np.complex128,
        )
        if not np.isclose(fidelity, 1.0, rtol=0.0, atol=1e-8):
            violations.append(f"noiseless fidelity is {fidelity:.12f}")

    return list(dict.fromkeys(violations))


def _traffic_profile(
    num_qubits: int,
    num_excitations: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """Count actual single-qubit and CZ traffic at each path position."""
    logical_qubits = list(cirq.LineQubit.range(num_qubits))
    positions = {
        qubit: index for index, qubit in enumerate(logical_qubits)
    }
    circuit = ideation.build_full_u_nk(
        logical_qubits,
        num_qubits,
        num_excitations,
    )
    sq_counts = {index: 0 for index in range(num_qubits)}
    cz_counts = {index: 0 for index in range(num_qubits - 1)}
    for operation in circuit.all_operations():
        if isinstance(operation.gate, cirq.PhasedXZGate):
            sq_counts[positions[operation.qubits[0]]] += 1
        elif operation.gate == cirq.CZ:
            first, second = (
                positions[qubit] for qubit in operation.qubits
            )
            if abs(first - second) != 1:
                raise ValueError("logical path contains non-nearest CZ")
            cz_counts[min(first, second)] += 1
    return sq_counts, cz_counts


def _find_score_matched_path(
    num_qubits: int,
    num_excitations: int,
    cz_errors: ErrorMap,
    sq_errors: ErrorMap,
    zero_errors: ErrorMap,
    one_errors: ErrorMap,
    uncertainty_weight: float,
    time_budget_seconds: float,
) -> tuple[list[cirq.GridQubit] | None, float]:
    """Find a simple Willow path using circuit-specific risk weights."""
    started = time.monotonic()
    sq_counts, cz_counts = _traffic_profile(
        num_qubits,
        num_excitations,
    )
    qubits = list(sq_errors)
    adjacency = {qubit: [] for qubit in qubits}
    for first, second in cz_errors:
        if first in adjacency and second in adjacency:
            adjacency[first].append(second)
            adjacency[second].append(first)

    excitation_probability = num_excitations / num_qubits

    def qubit_cost(qubit: cirq.GridQubit, position: int) -> float:
        sq_error = _risk_rate(
            sq_errors,
            qubit,
            0.001,
            uncertainty_weight,
        )
        zero_to_one = _risk_rate(
            zero_errors,
            qubit,
            0.01,
            uncertainty_weight,
        )
        one_to_zero = _risk_rate(
            one_errors,
            qubit,
            0.01,
            uncertainty_weight,
        )
        readout_error = (
            (1.0 - excitation_probability) * zero_to_one
            + excitation_probability * one_to_zero
        )
        return (
            sq_counts[position] * _negative_log_survival(sq_error)
            + _negative_log_survival(readout_error)
        )

    def edge_cost(
        first: cirq.GridQubit,
        second: cirq.GridQubit,
        edge_position: int,
    ) -> float:
        error = _risk_rate(
            cz_errors,
            (first, second),
            _risk_rate(
                cz_errors,
                (second, first),
                0.01,
                uncertainty_weight,
            ),
            uncertainty_weight,
        )
        return cz_counts[edge_position] * _negative_log_survival(error)

    # Keep the strongest partial paths at every length. This makes runtime
    # predictable while preserving alternatives that avoid geometric dead ends.
    beam_width = 1024
    states = [
        (qubit_cost(start, 0), (start,), frozenset((start,)))
        for start in qubits
    ]
    states.sort(key=lambda state: state[0])
    states = states[:beam_width]

    for position in range(1, num_qubits):
        if time.monotonic() - started >= time_budget_seconds:
            break
        edge_position = position - 1
        next_states = []
        for cost, path, visited in states:
            current = path[-1]
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                next_cost = (
                    cost
                    + edge_cost(current, neighbor, edge_position)
                    + qubit_cost(neighbor, position)
                )
                next_states.append(
                    (
                        next_cost,
                        path + (neighbor,),
                        visited.union((neighbor,)),
                    )
                )
        if not next_states:
            return None, float("inf")
        next_states.sort(key=lambda state: state[0])
        states = next_states[:beam_width]

    if not states or len(states[0][1]) != num_qubits:
        return None, float("inf")
    best_cost, best_path, _ = states[0]
    return list(best_path), best_cost


def create_current_scs(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return the unchanged current SCS implementation."""
    return ideation.create_dicke_state(
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


def _optimize_current_scs(
    optimizer: Callable[[cirq.Circuit, set[cirq.GridQubit]], cirq.Circuit],
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    circuit, outputs = create_current_scs(
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )
    return optimizer(circuit, outputs), outputs


def create_current_compressed(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return current-path SCS with dependency-safe moment compression."""
    return _optimize_current_scs(
        lambda circuit, _: compress_moments(circuit),
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


def create_current_cz_cancelled(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return current-path SCS after the proposed adjacent-CZ pass."""
    return _optimize_current_scs(
        lambda circuit, _: cancel_adjacent_czs(circuit),
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


def create_current_dd8(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return current-path SCS with threshold-8 CPMG pulses."""
    return _optimize_current_scs(
        lambda circuit, outputs: apply_threshold_dd(
            circuit,
            outputs,
            min_gap_moments=8,
        ),
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


def create_current_cz_dd8(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return current-path SCS with CZ cancellation followed by DD."""

    def optimize(
        circuit: cirq.Circuit,
        outputs: set[cirq.GridQubit],
    ) -> cirq.Circuit:
        return apply_threshold_dd(
            cancel_adjacent_czs(circuit),
            outputs,
            min_gap_moments=8,
        )

    return _optimize_current_scs(
        optimize,
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


def create_current_fused(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return current-path SCS with native 1Q fusion and compression."""
    return _optimize_current_scs(
        lambda circuit, _: fuse_single_qubit_gates(circuit),
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


def create_current_fused_dd8(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return fused current-path SCS followed by threshold-8 DD."""

    def optimize(
        circuit: cirq.Circuit,
        outputs: set[cirq.GridQubit],
    ) -> cirq.Circuit:
        return apply_threshold_dd(
            fuse_single_qubit_gates(circuit),
            outputs,
            min_gap_moments=8,
        )

    return _optimize_current_scs(
        optimize,
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


def create_score_matched_scs(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
    *,
    uncertainty_weight: float = 1.0,
    time_budget_seconds: float = 45.0,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return SCS placed with circuit- and score-aware risk weights."""
    path, _ = _find_score_matched_path(
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
        uncertainty_weight,
        time_budget_seconds,
    )
    if path is None:
        raise NotImplementedError("score-matched path search found no path")
    circuit = ideation.build_full_u_nk(
        path,
        num_qubits,
        num_excitations,
    )
    circuit = cirq.drop_negligible_operations(circuit, atol=1e-8)
    return circuit, set(path)


def create_score_matched_fused(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return score-matched SCS with exact native 1Q fusion and compression."""
    circuit, outputs = create_score_matched_scs(
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )
    return fuse_single_qubit_gates(circuit), outputs


def create_score_matched_compressed(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Match the current submission: score-aware placement plus compression."""
    circuit, outputs = create_score_matched_scs(
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )
    return compress_moments(circuit), outputs


def create_midtime_portfolio(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return the preserved mid-time Problem 1 portfolio candidate."""
    return midtime_problem1.create_dicke_state(
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


def create_shallow_incumbent(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return the Astra shallow incumbent without later family replacement."""
    return runtime_submission._FINAL_ASTRA_MAIN(
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


def create_runtime_anytime(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Return the authoritative deadline-safe runtime optimizer candidate."""
    return runtime_submission.create_dicke_state(
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


def _build_candidate(
    family: str,
    builder: CreateFunction,
    num_qubits: int,
    num_excitations: int,
    cz_errors: dict,
    sq_errors: dict,
    zero_errors: dict,
    one_errors: dict,
    verify_fidelity: bool,
) -> Candidate:
    started = time.monotonic()
    circuit, outputs = builder(
        num_qubits,
        num_excitations,
        cz_errors,
        sq_errors,
        zero_errors,
        one_errors,
    )
    elapsed = time.monotonic() - started
    outputs = set(outputs)
    violations = validate_problem1_candidate(
        circuit,
        outputs,
        num_qubits,
        num_excitations,
        elapsed,
        cz_errors,
        sq_errors,
        verify_fidelity=verify_fidelity,
    )
    if violations:
        raise NotImplementedError(
            f"{family} failed contract: {'; '.join(violations)}"
        )
    return Candidate(
        family=family,
        circuit=circuit,
        outputs=outputs,
        calibration_cost=circuit_calibration_cost(
            circuit,
            outputs,
            num_excitations,
            cz_errors,
            sq_errors,
            zero_errors,
            one_errors,
        ),
        construction_seconds=elapsed,
    )


def create_portfolio_best(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    zero_errors: dict,
    one_errors: dict,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Select the lowest-risk valid candidate under a bounded runtime."""
    started = time.monotonic()
    candidates: list[Candidate] = []

    try:
        candidates.append(
            _build_candidate(
                "midtime",
                create_midtime_portfolio,
                num_qubits,
                num_excitations,
                cz_pauli_error_rate,
                sq_pauli_error_rate,
                zero_errors,
                one_errors,
                verify_fidelity=num_qubits <= 8,
            )
        )
    except (NotImplementedError, ValueError):
        pass

    # Compare all families where exact simulation and both searches are cheap.
    if num_qubits <= 8 and time.monotonic() - started < 30.0:
        for family, builder in (
            ("score_matched_scs", create_score_matched_scs),
            ("current_scs", create_current_scs),
        ):
            if time.monotonic() - started >= 45.0:
                break
            try:
                candidates.append(
                    _build_candidate(
                        family,
                        builder,
                        num_qubits,
                        num_excitations,
                        cz_pauli_error_rate,
                        sq_pauli_error_rate,
                        zero_errors,
                        one_errors,
                        verify_fidelity=True,
                    )
                )
            except (NotImplementedError, ValueError):
                pass

    if candidates:
        proxy_costs: dict[str, float] = {}
        if num_qubits <= 8 and time.monotonic() - started < 48.0:
            for candidate in candidates:
                if time.monotonic() - started >= 55.0:
                    break
                try:
                    proxy_costs[candidate.family] = midtime_problem1.small_proxy(
                        candidate.circuit,
                        sorted(candidate.outputs),
                        num_excitations,
                        cz_pauli_error_rate,
                        sq_pauli_error_rate,
                        zero_errors,
                        one_errors,
                    )
                except Exception:
                    continue

        winner = min(
            candidates,
            key=lambda candidate: (
                (
                    proxy_costs.get(candidate.family, float("inf"))
                    if proxy_costs
                    else candidate.calibration_cost
                ),
                candidate.calibration_cost,
                len(candidate.circuit),
            ),
        )
        return winner.circuit, winner.outputs

    # Universal exact fallback for unsupported calibrations or portfolio cases.
    return create_current_scs(
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


VARIANTS: dict[str, CreateFunction] = {
    "current_scs": create_current_scs,
    "current_compressed": create_current_compressed,
    "current_cz_cancelled": create_current_cz_cancelled,
    "current_dd8": create_current_dd8,
    "current_cz_dd8": create_current_cz_dd8,
    "current_fused": create_current_fused,
    "current_fused_dd8": create_current_fused_dd8,
    "score_matched_scs": create_score_matched_scs,
    "score_matched_compressed": create_score_matched_compressed,
    "score_matched_fused": create_score_matched_fused,
    "midtime_portfolio": create_midtime_portfolio,
    "shallow_incumbent": create_shallow_incumbent,
    "runtime_anytime": create_runtime_anytime,
    "portfolio_best": create_portfolio_best,
}
