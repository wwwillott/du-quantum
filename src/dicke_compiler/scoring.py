"""Transparent calibration projections for candidate ranking."""

from __future__ import annotations

import math
from collections.abc import Collection

import cirq

from .models import CalibrationData, DickeTarget


def projected_log_loss(
    circuit: cirq.AbstractCircuit,
    output_qubits: Collection[cirq.Qid],
    target: DickeTarget,
    calibration: CalibrationData,
) -> float:
    """Return uncertainty-adjusted independent gate/readout log loss.

    This is a ranking proxy, not a prediction of an official hardware score.
    It models multiplicative survival by adding ``-log(1-error)`` for each
    physical operation and the target's expected asymmetric readout mixture.
    """
    loss = 0.0
    for operation in circuit.all_operations():
        if len(operation.qubits) == 1:
            error = calibration.single_qubit_rate(operation.qubits[0])
        elif len(operation.qubits) == 2:
            error = calibration.two_qubit_rate(*operation.qubits)
        else:
            raise ValueError(
                "projected survival supports only one- and two-qubit gates"
            )
        loss -= math.log1p(-error)

    excitation_probability = (
        target.num_excitations / target.num_qubits
    )
    for qubit in output_qubits:
        zero_to_one, one_to_zero = calibration.readout_rates(qubit)
        effective = (
            (1.0 - excitation_probability) * zero_to_one
            + excitation_probability * one_to_zero
        )
        loss -= math.log1p(-effective)
    return loss


def projected_survival(
    circuit: cirq.AbstractCircuit,
    output_qubits: Collection[cirq.Qid],
    target: DickeTarget,
    calibration: CalibrationData,
) -> float:
    """Convert projected log loss to a multiplicative survival proxy."""
    return math.exp(
        -projected_log_loss(
            circuit,
            output_qubits,
            target,
            calibration,
        )
    )


def projected_judge_penalty(
    circuit: cirq.AbstractCircuit,
    output_qubits: Collection[cirq.Qid],
    target: DickeTarget,
    calibration: CalibrationData,
) -> float:
    """Return state- and N-aware expected judge-penalty surrogate (Score_Z + Score_X).

    Models:
    - Z-subspace retention floor and asymmetric readout (1 - p)*e0 + p*e1;
    - Active wire T1 relaxation loss scaled by excitation fraction p = k/N;
    - Readout dispersion scaled by multinomial variance factor (S/M) * k(N-k)/(N(N-1));
    - Symmetric X-readout e0 + e1 and Hadamard gate loss;
    - Idle dephasing scaled by superposition coherence 4*p*(1-p);
    - Even X-basis correlators sum over l = 2..2*floor(N/2).
    """
    n = target.num_qubits
    k = target.num_excitations
    if n == 0 or len(output_qubits) == 0:
        return 0.0
    p = k / n
    M = math.comb(n, k) if 0 <= k <= n else 1

    t_curr = 0.0
    wire_first: dict[cirq.Qid, float] = {}
    wire_cz: dict[cirq.Qid, int] = {q: 0 for q in output_qubits}
    wire_sq: dict[cirq.Qid, int] = {q: 0 for q in output_qubits}
    cz_count = 0
    sq_count = 0
    gate_lz = 0.0
    gate_lx = 0.0

    for moment in circuit:
        has2 = any(len(op.qubits) == 2 for op in moment.operations)
        dt = 4.2e-8 if has2 else 2.5e-8
        for op in moment.operations:
            if len(op.qubits) == 1:
                q = op.qubits[0]
                if q not in wire_first:
                    wire_first[q] = t_curr
                if q in wire_sq:
                    wire_sq[q] += 1
                sq_count += 1
                err = calibration.single_qubit_rate(q)
                gate_lz -= math.log1p(-err)
                gate_lx -= math.log1p(-err)
            elif len(op.qubits) == 2:
                for q in op.qubits:
                    if q not in wire_first:
                        wire_first[q] = t_curr
                    if q in wire_cz:
                        wire_cz[q] += 1
                cz_count += 1
                err = calibration.two_qubit_rate(*op.qubits)
                gate_lz -= math.log1p(-err)
                gate_lx -= math.log1p(-err)
        t_curr += dt

    t_total = t_curr
    t1_est = 7.5e-5
    tphi_est = 8.5e-5

    active_t1_loss = 0.0
    idle_tphi_loss = 0.0
    for q in output_qubits:
        if q in wire_first:
            act = max(0.0, t_total - wire_first[q])
            active_t1_loss += p * act / t1_est
            gate_busy = wire_cz.get(q, 0) * 4.2e-8 + wire_sq.get(q, 0) * 2.5e-8
            idle_time = max(0.0, act - gate_busy)
            idle_tphi_loss += 4.0 * p * (1.0 - p) * (idle_time / tphi_est)

    readout_lz = 0.0
    readout_deltas: list[float] = []
    x_readout_lx = 0.0
    x_deltas: list[float] = []
    for q in output_qubits:
        e0, e1 = calibration.readout_rates(q)
        readout_lz -= (1.0 - p) * math.log1p(-e0) + p * math.log1p(-e1)
        delta = math.log1p(-e1) - math.log1p(-e0)
        readout_deltas.append(delta)
        eff_x_ro = min(0.99, e0 + e1)
        esq = calibration.single_qubit_rate(q)
        x_readout_lx -= math.log1p(-eff_x_ro) + 4.0 / 3.0 * esq
        x_deltas.append(e1 - e0)

    total_lz = gate_lz + readout_lz + active_t1_loss
    shots = 10000.0
    f_z = math.exp(-min(total_lz, 20.0))
    inv_fz = 1.0 / max(f_z, 1e-6)
    score_z_sub = (1.0 - 1.0 / M) * inv_fz if M > 1 else 0.0

    mean_delta = sum(readout_deltas) / max(1, n)
    if M > 1 and n > 1:
        disp_scale = (shots * f_z / M) * (k * (n - k) / (n * (n - 1)))
        score_z_disp = disp_scale * sum((d - mean_delta) ** 2 for d in readout_deltas)
    else:
        score_z_disp = 0.0
    score_z = score_z_sub + score_z_disp

    total_lx = gate_lx + x_readout_lx + idle_tphi_loss
    mean_lx_per_qubit = total_lx / max(1, n)
    score_x = 0.0
    for m in range(1, n // 2 + 1):
        ell = 2 * m
        damp = math.exp(-min(ell * mean_lx_per_qubit, 20.0))
        score_x += (1.0 - damp) ** 2
    mean_x_bias = sum(x_deltas) / max(1, n)
    score_x += mean_x_bias ** 2

    return score_z + score_x + 1e-7 * cz_count + 1e-9 * len(circuit)

