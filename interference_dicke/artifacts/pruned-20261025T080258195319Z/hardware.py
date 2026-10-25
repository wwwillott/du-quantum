"""Reuse repository topology/compiler; no processor jobs or calibration claims."""
from dataclasses import asdict, dataclass

import cirq
import cirq_google
import numpy as np

from dicke_willow.compile import finalize, assert_challenge_gates
from dicke_willow.topology import willow105_graph
from dicke_willow.verify import ideal_dicke_vector


def willow_path(n):
    graph = willow105_graph()

    def search(path):
        if len(path) == n:
            return path
        for q in sorted(graph[path[-1]]):
            if q not in path:
                found = search(path + [q])
                if found:
                    return found
        return None

    for start in sorted(graph):
        found = search([start])
        if found:
            return found
    raise ValueError("no physical path found")


def circuit_for(sector, schedule, parameters, qubits):
    circuit = cirq.Circuit(cirq.X(q) for q in qubits[:sector.l])
    for (kind, sites), angle in zip(schedule, parameters):
        qs = [qubits[i] for i in sites]
        if kind == "xy":
            circuit.append(cirq.FSimGate(float(angle), 0).on(*qs))
        elif kind == "cp":
            circuit.append(cirq.CZPowGate(exponent=-float(angle) / np.pi).on(*qs))
        elif kind == "z":
            circuit.append(cirq.rz(2 * float(angle)).on(*qs))
        else:
            raise ValueError(kind)
    return circuit


def metrics(circuit):
    operations = list(circuit.all_operations())
    return {"two_qubit_gates": sum(len(op.qubits) == 2 for op in operations),
            "single_qubit_gates": sum(len(op.qubits) == 1 for op in operations),
            "depth": len(circuit),
            "two_qubit_depth": sum(any(len(op.qubits) == 2 for op in moment)
                                   for moment in circuit)}


def compile_validate(circuit, qubits):
    native = finalize(circuit, decouple=False)
    graph = willow105_graph()
    pairs = {(a, b) for a in qubits for b in graph[a] if b in qubits}
    assert_challenge_gates(native, qubits=set(qubits), sq_qubits=set(qubits), cz_pairs=pairs)
    cirq_google.Willow105.validate_circuit(native)
    return native


def ideal_fidelity(circuit, qubits, l):
    state = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubits).final_state_vector
    return float(abs(np.vdot(ideal_dicke_vector(len(qubits), l), state)) ** 2)


@dataclass(frozen=True)
class IllustrativeNoise:
    """Depolarizing channel probabilities, NOT measured gate infidelities.

    These values are synthetic sensitivity settings, not Willow calibration.
    Damping/dephasing act on ALL qubits after each moment, including idles.
    """
    p1: float = .0001
    p2: float = .001
    t1_ns: float = 100_000
    t2_ns: float = 80_000
    single_ns: float = 25
    two_ns: float = 40
    readout_flip: float = .01


def noisy_metrics(circuit, qubits, l, settings=IllustrativeNoise()):
    if len(qubits) > 6:
        raise ValueError("full density simulation is budgeted only for n <= 6")
    if not (0 <= settings.p1 <= 1 and 0 <= settings.p2 <= 1
            and 0 <= settings.readout_flip <= 1 and 0 < settings.t2_ns <= 2 * settings.t1_ns):
        raise ValueError("invalid noise parameters")
    noisy = cirq.Circuit()
    duration = 0.
    for moment in circuit:
        noisy.append(moment)
        channels = []
        dt = max((settings.two_ns if len(op.qubits) == 2 else settings.single_ns
                  for op in moment), default=0.)
        duration += dt
        for op in moment:
            p = settings.p2 if len(op.qubits) == 2 else settings.p1
            channels.append(cirq.depolarize(p, n_qubits=len(op.qubits)).on(*op.qubits))
        noisy.append(cirq.Moment(channels))
        gamma = 1 - np.exp(-dt / settings.t1_ns)
        inv_tphi = 1 / settings.t2_ns - 1 / (2 * settings.t1_ns)
        phase = 1 - np.exp(-2 * dt * inv_tphi)
        noisy.append(cirq.Moment(cirq.amplitude_damp(gamma).on(q) for q in qubits))
        noisy.append(cirq.Moment(cirq.phase_damp(phase).on(q) for q in qubits))
    rho = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(
        noisy, qubit_order=qubits).final_density_matrix
    target = ideal_dicke_vector(len(qubits), l)
    fidelity = float(np.vdot(target, rho @ target).real)
    masks = np.arange(2 ** len(qubits))
    valid = np.array([int(x).bit_count() == l for x in masks])
    probabilities = np.diag(rho).real
    acceptance = float(probabilities[valid].sum())
    # Classical readout acts on probabilities only; it is not state infidelity.
    observed = probabilities.copy()
    for i in range(len(qubits)):
        observed = (1 - settings.readout_flip) * observed + settings.readout_flip * observed[masks ^ (1 << i)]
    return {"fidelity": fidelity, "p_valid": acceptance,
            "f_post": fidelity / acceptance if acceptance > 0 else None,
            "observed_valid_after_readout": float(observed[valid].sum()),
            "duration_ns_illustrative": duration, "settings": asdict(settings),
            "model": "synthetic Markovian channels on compiled CZ circuit; no crosstalk/leakage"}
