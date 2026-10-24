"""Simulate the Dicke-state circuits on Cirq's Willow Quantum Virtual Machine.

The virtual processor is ``willow_pink``, the Willow device bundled with
``cirq_google``. Path selection uses that processor's median calibration.
Circuits are built by ``ideation.py``, routed so every CZ lies on a device
edge, then executed through Cirq's virtual engine.

``cirq_google.engine.create_noiseless_virtual_engine_from_latest_templates``
validates and samples with a noiseless simulator. ``create_default_noisy_quantum_virtual_machine``
samples with the processor's calibration noise model. Exact state fidelity is
computed with ``cirq.Simulator`` on the same device-validated circuit, because
the virtual-engine sampler returns shots rather than a state vector.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import cirq
import cirq_google
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ideation

PROCESSOR_ID = "willow_pink"
PATH_MODES = ("dijkstra", "branch_and_bound", "geometric_snake")
# (N, n) pairs exercised by the SCS builder, including the three-qubit gate.
CIRCUIT_CASES = ((2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2))


def load_willow_qvm(*, noisy: bool) -> cirq_google.engine.SimulatedLocalEngine:
    """Return a Cirq virtual engine for the bundled Willow processor."""
    if noisy:
        return cirq_google.engine.create_default_noisy_quantum_virtual_machine(
            processor_id=PROCESSOR_ID,
            simulator_class=cirq.Simulator,
        )
    return cirq_google.engine.create_noiseless_virtual_engine_from_latest_templates()


def calibration_maps(
    calibration: cirq_google.Calibration,
) -> tuple[dict, dict, dict]:
    """Convert Willow calibration metrics into pathfinder error maps."""
    sq_errors: dict[cirq.GridQubit, float] = {}
    for key, value in calibration["single_qubit_rb_pauli_error_per_gate"].items():
        sq_errors[calibration.key_to_qubit(key)] = calibration.value_to_float(value)

    cz_errors: dict[tuple[cirq.GridQubit, cirq.GridQubit], float] = {}
    cz_metric = calibration["two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle"]
    for key, value in cz_metric.items():
        q0, q1 = calibration.key_to_qubits(key)
        error = calibration.value_to_float(value)
        cz_errors[(q0, q1)] = error
        cz_errors[(q1, q0)] = error

    readout_errors: dict[cirq.GridQubit, float] = {}
    zero_errors = calibration["single_qubit_p00_error"]
    one_errors = calibration["single_qubit_p11_error"]
    for key, value in zero_errors.items():
        qubit = calibration.key_to_qubit(key)
        readout_errors[qubit] = 0.5 * (
            calibration.value_to_float(value)
            + calibration.value_to_float(one_errors[key])
        )
    return sq_errors, cz_errors, readout_errors


def device_graph(device: cirq.Device):
    """Return working qubits and couplers from a GridDevice."""
    metadata = device.metadata
    qubits = list(metadata.qubit_set)
    couplers = [tuple(pair) for pair in metadata.qubit_pairs]
    edges = set(couplers)
    edges.update((b, a) for a, b in couplers)
    return qubits, couplers, edges


def native_swap(a: cirq.Qid, b: cirq.Qid):
    """Yield a SWAP decomposed into Willow-native CNOT gates."""
    yield from ideation.native_cnot(a, b)
    yield from ideation.native_cnot(b, a)
    yield from ideation.native_cnot(a, b)


def route_onto_path(
    circuit: cirq.Circuit, path: list[cirq.GridQubit]
) -> cirq.Circuit:
    """Move logical states along ``path`` so every CZ uses a path edge.

    The SCS construction applies CZ gates between qubits that are not always
    neighbors. SWAP gates slide those states onto adjacent GridQubits, the CZ
    is applied there, and the original order is restored before the circuit ends.
    """
    phys_of = {qubit: index for index, qubit in enumerate(path)}
    logical_at = list(path)
    routed = cirq.Circuit()

    def emit_swap(i: int, j: int) -> None:
        routed.append(native_swap(path[i], path[j]))
        left, right = logical_at[i], logical_at[j]
        logical_at[i], logical_at[j] = right, left
        phys_of[left] = j
        phys_of[right] = i

    def bring_adjacent(q1: cirq.Qid, q2: cirq.Qid) -> None:
        while abs(phys_of[q1] - phys_of[q2]) > 1:
            origin = phys_of[q2]
            step = -1 if origin > phys_of[q1] else 1
            emit_swap(origin, origin + step)

    for op in circuit.all_operations():
        targets = op.qubits
        if len(targets) == 2:
            bring_adjacent(targets[0], targets[1])
            i, j = phys_of[targets[0]], phys_of[targets[1]]
            routed.append(op.gate.on(path[i], path[j]))
        elif len(targets) == 1:
            routed.append(op.gate.on(path[phys_of[targets[0]]]))
        else:
            raise ValueError(f"Unsupported operation {op}")

    for target in range(len(path)):
        while phys_of[path[target]] != target:
            current = phys_of[path[target]]
            step = -1 if current > target else 1
            emit_swap(current, current + step)

    if logical_at != path:
        raise RuntimeError("Routing did not restore the original qubit order.")
    return routed


def assert_local(circuit: cirq.Circuit, edges: set) -> None:
    """Raise if any two-qubit gate leaves the device coupling graph."""
    for op in circuit.all_operations():
        if len(op.qubits) == 2 and tuple(op.qubits) not in edges:
            raise RuntimeError(f"Non-local operation on the Willow QVM: {op}")


def with_basis_measurement(
    circuit: cirq.Circuit, qubits: list[cirq.GridQubit], basis: str
) -> cirq.Circuit:
    """Return a copy that measures ``qubits`` in the Z or X basis."""
    measured = circuit.copy()
    if basis == "x":
        measured.append(ideation.native_h(qubit) for qubit in qubits)
    elif basis != "z":
        raise ValueError(f"Unknown basis {basis}")
    measured.append(cirq.measure(*qubits, key="m"))
    return measured


def sample_qvm(
    engine: cirq_google.engine.SimulatedLocalEngine,
    circuit: cirq.Circuit,
    repetitions: int,
) -> np.ndarray:
    """Run ``circuit`` on the virtual processor and return measurement bits."""
    processor = engine.get_processor(PROCESSOR_ID)
    result = processor.get_sampler().run(circuit, repetitions=repetitions)
    return np.asarray(result.measurements["m"], dtype=int)


def dicke_statevector(num_qubits: int, num_excitations: int) -> np.ndarray:
    """Return the ideal Dicke state in Cirq's computational basis ordering."""
    state = np.zeros(2**num_qubits, dtype=np.complex64)
    amplitude = 1 / np.sqrt(math.comb(num_qubits, num_excitations))
    for excited in itertools.combinations(range(num_qubits), num_excitations):
        index = sum(2**qubit for qubit in excited)
        state[index] = amplitude
    return state


def state_fidelity(
    circuit: cirq.Circuit, path: list[cirq.GridQubit], num_excitations: int
) -> float:
    """Exact fidelity of ``circuit`` against the Dicke state on ``path``."""
    simulated = cirq.Simulator().simulate(circuit, qubit_order=path).final_state_vector
    ideal = dicke_statevector(len(path), num_excitations)
    return float(np.abs(np.vdot(ideal, simulated)) ** 2)


def check_z_bitstring_distribution(bitstrings: np.ndarray, num_excitations: int) -> float:
    """Challenge Z-basis score. Lower is closer to the ideal Dicke distribution."""
    n_shots, n_qubits = bitstrings.shape
    measured_excitations = np.sum(bitstrings, axis=1)
    fraction_in_subspace = np.mean(measured_excitations == num_excitations)
    if fraction_in_subspace == 0:
        return float("inf")

    chi2 = 0.0
    ideal_prob = 1 / math.comb(n_qubits, num_excitations)
    for indices in itertools.combinations(range(n_qubits), num_excitations):
        ideal_state = np.zeros(n_qubits)
        ideal_state[np.array(indices)] = 1
        match_count = np.sum(np.all(bitstrings == ideal_state, axis=1))
        expected_count = n_shots * fraction_in_subspace * ideal_prob
        chi2 += (match_count - expected_count) ** 2 / expected_count
    return ideal_prob * chi2 / fraction_in_subspace


def get_correlator_average(bitstrings: np.ndarray, correlator_size: int) -> float:
    correlator_sum = 0.0
    n_qubits = bitstrings.shape[1]
    for indices in itertools.combinations(range(n_qubits), correlator_size):
        correlator_sum += 1 - 2 * np.mean(np.sum(bitstrings[:, indices], axis=1) % 2)
    return correlator_sum / math.comb(n_qubits, correlator_size)


def check_x_bitstring_distribution(
    bitstrings: np.ndarray, num_excitations: int, max_correlator_size: int
) -> float:
    """Challenge X-basis correlator score. Lower is closer to ideal."""
    comb = lambda n, k: math.comb(n, k) if (k >= 0 and n >= k) else 0
    n_qubits = bitstrings.shape[1]
    score = 0.0
    for correlator_size in range(1, max_correlator_size + 1):
        expected_correlator = (
            comb(correlator_size, correlator_size // 2)
            * comb(n_qubits - correlator_size, num_excitations - correlator_size // 2)
            / comb(n_qubits, num_excitations)
            if correlator_size % 2 == 0
            else 0.0
        )
        measured_correlator = get_correlator_average(bitstrings, correlator_size)
        score += (
            (expected_correlator - measured_correlator) ** 2 / expected_correlator**2
            if expected_correlator != 0
            else (expected_correlator - measured_correlator) ** 2
        )
    return score


def overall_score(
    z_bitstrings: np.ndarray,
    x_bitstrings: np.ndarray,
    num_excitations: int,
    max_correlator_size: int,
) -> float:
    """Challenge score. Higher means the samples look more like the Dicke state."""
    total = check_z_bitstring_distribution(z_bitstrings, num_excitations)
    total += check_x_bitstring_distribution(
        x_bitstrings, num_excitations, max_correlator_size
    )
    if not np.isfinite(total) or total == 0:
        return 0.0 if total != 0 else float("inf")
    return 1 / total


def prepare_routed_circuit(
    path: list[cirq.GridQubit],
    num_qubits: int,
    num_excitations: int,
    edges: set,
    device: cirq.Device,
) -> tuple[cirq.Circuit, cirq.Circuit, float, float]:
    """Build, route, and device-validate a Dicke circuit.

    Returns the unrouted circuit, the routed circuit, and both fidelities.
    """
    logical = ideation.build_full_u_nk(path, num_qubits, num_excitations)
    routed = route_onto_path(logical, path)
    assert_local(routed, edges)
    device.validate_circuit(routed)
    return (
        logical,
        routed,
        state_fidelity(logical, path, num_excitations),
        state_fidelity(routed, path, num_excitations),
    )


def find_device_path(
    finder: ideation.MultiModePathfinder,
    qubits,
    couplers,
    sq_errors,
    cz_errors,
    readout_errors,
    num_qubits: int,
    num_excitations: int,
    mode: str,
):
    started = time.perf_counter()
    path, cost = finder.find_path(
        qubits,
        couplers,
        num_qubits,
        num_excitations,
        sq_errors,
        cz_errors,
        readout_errors,
        override_mode=mode,
    )
    elapsed = time.perf_counter() - started
    if path is None:
        raise RuntimeError(f"{mode} found no {num_qubits}-qubit path on {PROCESSOR_ID}.")
    return path, cost, elapsed


def run_suite(noiseless_shots: int = 400, noisy_shots: int = 80) -> list[dict]:
    """Place every constructed circuit on the Willow QVM and sample it."""
    print(f"Cirq {cirq.__version__} virtual processors: {cirq_google.engine.list_virtual_processors()}")
    noiseless_engine = load_willow_qvm(noisy=False)
    noisy_engine = load_willow_qvm(noisy=True)
    device = noiseless_engine.get_processor(PROCESSOR_ID).get_device()
    calibration = cirq_google.engine.load_median_device_calibration(PROCESSOR_ID)
    sq_errors, cz_errors, readout_errors = calibration_maps(calibration)
    qubits, couplers, edges = device_graph(device)
    finder = ideation.MultiModePathfinder()
    rows: list[dict] = []

    print(f"\nPathfinder on {PROCESSOR_ID} calibration (N=4, n=1)")
    for mode in PATH_MODES:
        path, cost, elapsed = find_device_path(
            finder, qubits, couplers, sq_errors, cz_errors, readout_errors, 4, 1, mode
        )
        _, routed, logical_fidelity, routed_fidelity = prepare_routed_circuit(
            path, 4, 1, edges, device
        )
        z_bits = sample_qvm(
            noiseless_engine, with_basis_measurement(routed, path, "z"), repetitions=noiseless_shots
        )
        subspace = float(np.mean(np.sum(z_bits, axis=1) == 1))
        print(
            f"  {mode:18} cost={cost:.4f} search={elapsed:.3f}s "
            f"moments={len(routed):4} fidelity={routed_fidelity:.6f} "
            f"noiseless Z weight-1 fraction={subspace:.3f}"
        )
        rows.append(
            {
                "stage": "pathfinder",
                "mode": mode,
                "N": 4,
                "n": 1,
                "moments": len(routed),
                "logical_fidelity": logical_fidelity,
                "routed_fidelity": routed_fidelity,
                "qvm": "noiseless",
                "z_subspace_fraction": subspace,
            }
        )

    print("\nNoiseless Willow QVM")
    for num_qubits, num_excitations in CIRCUIT_CASES:
        path, cost, elapsed = find_device_path(
            finder,
            qubits,
            couplers,
            sq_errors,
            cz_errors,
            readout_errors,
            num_qubits,
            num_excitations,
            "dijkstra",
        )
        _, routed, logical_fidelity, routed_fidelity = prepare_routed_circuit(
            path, num_qubits, num_excitations, edges, device
        )
        z_bits = sample_qvm(
            noiseless_engine,
            with_basis_measurement(routed, path, "z"),
            repetitions=noiseless_shots,
        )
        x_bits = sample_qvm(
            noiseless_engine,
            with_basis_measurement(routed, path, "x"),
            repetitions=noiseless_shots,
        )
        score = overall_score(z_bits, x_bits, num_excitations, max_correlator_size=min(4, num_qubits))
        subspace = float(np.mean(np.sum(z_bits, axis=1) == num_excitations))
        print(
            f"  N={num_qubits} n={num_excitations} path_cost={cost:.4f} "
            f"search={elapsed:.3f}s moments={len(routed):4} "
            f"fidelity={routed_fidelity:.6f} (unrouted {logical_fidelity:.6f}) "
            f"Z subspace={subspace:.3f} score={score:.4f}"
        )
        rows.append(
            {
                "stage": "noiseless",
                "mode": "dijkstra",
                "N": num_qubits,
                "n": num_excitations,
                "moments": len(routed),
                "logical_fidelity": logical_fidelity,
                "routed_fidelity": routed_fidelity,
                "qvm": "noiseless",
                "shots": noiseless_shots,
                "z_subspace_fraction": subspace,
                "overall_score": score,
            }
        )

    print("\nNoisy Willow QVM (median calibration noise)")
    for num_qubits, num_excitations in CIRCUIT_CASES:
        path, _, _ = find_device_path(
            finder,
            qubits,
            couplers,
            sq_errors,
            cz_errors,
            readout_errors,
            num_qubits,
            num_excitations,
            "dijkstra",
        )
        _, routed, _, routed_fidelity = prepare_routed_circuit(
            path, num_qubits, num_excitations, edges, device
        )
        started = time.perf_counter()
        z_bits = sample_qvm(
            noisy_engine,
            with_basis_measurement(routed, path, "z"),
            repetitions=noisy_shots,
        )
        x_bits = sample_qvm(
            noisy_engine,
            with_basis_measurement(routed, path, "x"),
            repetitions=noisy_shots,
        )
        elapsed = time.perf_counter() - started
        score = overall_score(z_bits, x_bits, num_excitations, max_correlator_size=min(4, num_qubits))
        subspace = float(np.mean(np.sum(z_bits, axis=1) == num_excitations))
        print(
            f"  N={num_qubits} n={num_excitations} moments={len(routed):4} "
            f"ideal_fidelity={routed_fidelity:.6f} "
            f"noisy Z subspace={subspace:.3f} score={score:.4f} "
            f"sample_time={elapsed:.2f}s"
        )
        rows.append(
            {
                "stage": "noisy",
                "mode": "dijkstra",
                "N": num_qubits,
                "n": num_excitations,
                "moments": len(routed),
                "routed_fidelity": routed_fidelity,
                "qvm": "noisy",
                "shots": noisy_shots,
                "z_subspace_fraction": subspace,
                "overall_score": score,
                "sample_time_s": elapsed,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noiseless-shots", type=int, default=400)
    parser.add_argument("--noisy-shots", type=int, default=80)
    args = parser.parse_args()
    run_suite(noiseless_shots=args.noiseless_shots, noisy_shots=args.noisy_shots)


if __name__ == "__main__":
    main()
