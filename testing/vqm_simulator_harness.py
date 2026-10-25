"""Noisy local simulator harness for the DuQuantum Dicke-state solution."""

from __future__ import annotations

import argparse
import itertools
import math
from collections.abc import Mapping

import cirq
import numpy as np

from ideation import (
    create_dicke_state,
    generate_qubit_error_rates,
    generate_two_qubit_pauli_error_rates,
    native_h,
    pick_single_qubit_dropouts,
)


ErrorMap = Mapping[object, float | tuple[float, float]]


def generate_seeded_error_maps(
    seed: int,
) -> tuple[dict, dict, dict, dict]:
    """Generate a reproducible set of synthetic challenge error maps."""
    rng = np.random.default_rng(seed)
    dropouts = pick_single_qubit_dropouts(0.01, rng=rng)
    sq_errors = generate_qubit_error_rates(
        4e-4,
        0.5,
        dropouts,
        2e-4,
        rng=rng,
    )
    cz_errors = generate_two_qubit_pauli_error_rates(
        single_qubit_dropouts=dropouts,
        rng=rng,
    )
    zero_readout_errors = generate_qubit_error_rates(
        5e-3,
        0.5,
        dropouts,
        2e-4,
        rng=rng,
    )
    one_readout_errors = generate_qubit_error_rates(
        1e-2,
        0.5,
        dropouts,
        3e-4,
        rng=rng,
    )
    return (
        sq_errors,
        cz_errors,
        zero_readout_errors,
        one_readout_errors,
    )


class VQMSimulatorHarness:
    """Noisy simulator and scoring harness using Willow-style error maps."""

    def __init__(
        self,
        sq_errors: ErrorMap,
        cz_errors: ErrorMap,
        zero_errors: ErrorMap,
        one_errors: ErrorMap,
        idle_phase_error: float = 0.0,
        simulator_backend: str = "auto",
        max_density_matrix_qubits: int = 10,
    ) -> None:
        self.sq_errors = sq_errors
        self.cz_errors = cz_errors
        self.zero_errors = zero_errors
        self.one_errors = one_errors
        self.idle_phase_error = idle_phase_error
        self.simulator_backend = simulator_backend
        self.max_density_matrix_qubits = max_density_matrix_qubits

        if not 0.0 <= idle_phase_error <= 1.0:
            raise ValueError("idle_phase_error must be between 0 and 1")
        if simulator_backend not in {"auto", "density_matrix", "trajectory"}:
            raise ValueError(
                "simulator_backend must be auto, density_matrix, or trajectory"
            )

    @staticmethod
    def _error_rate(
        error_map: ErrorMap,
        key: object,
        default: float,
    ) -> float:
        """Extract a rate from either a float or a (rate, uncertainty) pair."""
        value = error_map.get(key, default)
        if isinstance(value, tuple):
            value = value[0]
        return float(value)

    @staticmethod
    def _readout_channel(p01: float, p10: float) -> cirq.KrausChannel:
        """Return a channel with 0→1 probability p01 and 1→0 probability p10."""
        p01 = float(np.clip(p01, 0.0, 1.0))
        p10 = float(np.clip(p10, 0.0, 1.0))
        kraus_ops = [
            np.array([[np.sqrt(1.0 - p01), 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [np.sqrt(p01), 0.0]]),
            np.array([[0.0, 0.0], [0.0, np.sqrt(1.0 - p10)]]),
            np.array([[0.0, np.sqrt(p10)], [0.0, 0.0]]),
        ]
        return cirq.KrausChannel(kraus_ops)

    def build_noisy_circuit(
        self,
        circuit: cirq.Circuit,
        path_qubits: set[cirq.Qid],
    ) -> cirq.Circuit:
        """Insert gate noise and final asymmetric readout noise."""
        noisy_circuit = cirq.Circuit()

        for moment in circuit:
            noisy_circuit.append(moment)
            noise_operations = []
            active_qubits = set()
            for operation in moment.operations:
                active_qubits.update(operation.qubits)
                if len(operation.qubits) == 1:
                    qubit = operation.qubits[0]
                    error = self._error_rate(
                        self.sq_errors,
                        qubit,
                        0.001,
                    )
                    noise_operations.append(cirq.depolarize(error).on(qubit))
                elif len(operation.qubits) == 2:
                    first, second = operation.qubits
                    error = self._error_rate(
                        self.cz_errors,
                        (first, second),
                        self._error_rate(
                            self.cz_errors,
                            (second, first),
                            0.01,
                        ),
                    )
                    noise_operations.append(
                        cirq.depolarize(error, n_qubits=2).on(first, second)
                    )
            if self.idle_phase_error:
                for qubit in path_qubits.difference(active_qubits):
                    noise_operations.append(
                        cirq.phase_damp(self.idle_phase_error).on(qubit)
                    )
            if noise_operations:
                noisy_circuit.append(noise_operations)

        readout_operations = []
        for qubit in path_qubits:
            p01 = self._error_rate(self.zero_errors, qubit, 0.01)
            p10 = self._error_rate(self.one_errors, qubit, 0.01)
            readout_operations.append(
                self._readout_channel(p01, p10).on(qubit)
            )
        noisy_circuit.append(readout_operations)
        return noisy_circuit

    @staticmethod
    def score_z_distribution(
        samples: np.ndarray,
        num_qubits: int,
        num_excitations: int,
    ) -> float:
        """Apply the challenge's exact Z-basis scoring formula."""
        shots = samples.shape[0]
        valid_mask = np.sum(samples, axis=1) == num_excitations
        subspace_retention = float(np.mean(valid_mask))
        if subspace_retention == 0:
            return float("inf")

        valid_samples = samples[valid_mask]
        state_count = math.comb(num_qubits, num_excitations)
        expected_count = shots * subspace_retention / state_count
        bit_powers = 2 ** np.arange(num_qubits - 1, -1, -1)
        measured_indices = valid_samples.dot(bit_powers)
        all_counts = np.bincount(
            measured_indices,
            minlength=2**num_qubits,
        )
        valid_indices = [
            sum(1 << (num_qubits - 1 - index) for index in excitation_indices)
            for excitation_indices in itertools.combinations(
                range(num_qubits),
                num_excitations,
            )
        ]
        observed_counts = all_counts[valid_indices]
        chi_squared = np.sum(
            (observed_counts - expected_count) ** 2 / expected_count
        )
        ideal_probability = 1.0 / state_count
        return float(
            ideal_probability * chi_squared / subspace_retention
        )

    @staticmethod
    def _correlator_average(
        samples: np.ndarray,
        correlator_size: int,
    ) -> float:
        """Return the average of all weight-k X correlators."""
        correlator_sum = 0.0
        _, num_qubits = samples.shape
        for indices in itertools.combinations(
            range(num_qubits),
            correlator_size,
        ):
            parity = np.sum(samples[:, indices], axis=1) % 2
            correlator_sum += 1.0 - 2.0 * np.mean(parity)
        return float(
            correlator_sum / math.comb(num_qubits, correlator_size)
        )

    @classmethod
    def score_x_distribution(
        cls,
        samples: np.ndarray,
        num_excitations: int,
        max_correlator_size: int,
    ) -> float:
        """Apply the challenge's exact X-basis scoring formula."""

        def safe_combination(n: int, k: int) -> int:
            return math.comb(n, k) if 0 <= k <= n else 0

        _, num_qubits = samples.shape
        score = 0.0
        for correlator_size in range(1, max_correlator_size + 1):
            expected_correlator = (
                safe_combination(
                    correlator_size,
                    correlator_size // 2,
                )
                * safe_combination(
                    num_qubits - correlator_size,
                    num_excitations - correlator_size // 2,
                )
                / safe_combination(num_qubits, num_excitations)
                if correlator_size % 2 == 0
                else 0.0
            )
            measured_correlator = cls._correlator_average(
                samples,
                correlator_size,
            )
            difference = expected_correlator - measured_correlator
            score += (
                difference**2 / expected_correlator**2
                if expected_correlator != 0
                else difference**2
            )
        return float(score)

    def _select_simulator(
        self,
        num_qubits: int,
        seed: int | None,
    ) -> tuple[cirq.Sampler, str]:
        """Choose a fast backend without risking a huge density matrix."""
        backend = self.simulator_backend
        if backend == "auto":
            backend = (
                "density_matrix"
                if num_qubits <= self.max_density_matrix_qubits
                else "trajectory"
            )
        if backend == "density_matrix":
            return cirq.DensityMatrixSimulator(seed=seed), backend
        return cirq.Simulator(seed=seed), backend

    def evaluate_solution(
        self,
        circuit_fn,
        num_qubits: int,
        num_excitations: int,
        shots: int = 10_000,
        seed: int | None = None,
    ) -> dict[str, float | int | str]:
        """Run noisy Z- and X-basis evaluations for one Dicke-state case."""
        clean_circuit, qubit_set = circuit_fn(
            num_qubits=num_qubits,
            num_excitations=num_excitations,
            cz_pauli_error_rate=self.cz_errors,
            sq_pauli_error_rate=self.sq_errors,
            zero_errors=self.zero_errors,
            one_errors=self.one_errors,
        )
        ordered_path = sorted(qubit_set)

        noisy_z = self.build_noisy_circuit(clean_circuit, qubit_set)
        noisy_z.append(cirq.measure(*ordered_path, key="z"))

        x_basis_circuit = clean_circuit.copy()
        x_basis_circuit.append(native_h(qubit) for qubit in ordered_path)
        noisy_x = self.build_noisy_circuit(x_basis_circuit, qubit_set)
        noisy_x.append(cirq.measure(*ordered_path, key="x"))

        simulator, backend = self._select_simulator(num_qubits, seed)
        samples_z = simulator.run(
            noisy_z,
            repetitions=shots,
        ).measurements["z"]
        samples_x = simulator.run(
            noisy_x,
            repetitions=shots,
        ).measurements["x"]

        score_z = self.score_z_distribution(
            samples_z,
            num_qubits,
            num_excitations,
        )
        score_x = self.score_x_distribution(
            samples_x,
            num_excitations,
            max_correlator_size=num_qubits,
        )
        total_penalty = score_z + score_x
        if not np.isfinite(total_penalty):
            judge_score = 0.0
        elif total_penalty == 0:
            judge_score = float("inf")
        else:
            judge_score = 1.0 / total_penalty
        return {
            "N": num_qubits,
            "n": num_excitations,
            "shots": shots,
            "simulator_backend": backend,
            "idle_phase_error": self.idle_phase_error,
            "subspace_retention": float(
                np.mean(np.sum(samples_z, axis=1) == num_excitations)
            ),
            "score_z": score_z,
            "score_x": score_x,
            "estimated_judge_score": judge_score,
            "overall_score": judge_score,
        }


def main() -> None:
    """Run the current Dicke-state implementation through the harness."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", "-N", type=int, default=4)
    parser.add_argument("--num-excitations", "-n", type=int, default=2)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--error-seed",
        type=int,
        default=2026,
        help="Seed used to generate reproducible synthetic error maps",
    )
    parser.add_argument(
        "--idle-phase-error",
        type=float,
        default=0.0,
        help="Optional phase-damping probability per idle logical moment",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "density_matrix", "trajectory"),
        default="auto",
    )
    arguments = parser.parse_args()

    (
        sq_errors,
        cz_errors,
        zero_readout_errors,
        one_readout_errors,
    ) = generate_seeded_error_maps(arguments.error_seed)
    harness = VQMSimulatorHarness(
        sq_errors=sq_errors,
        cz_errors=cz_errors,
        zero_errors=zero_readout_errors,
        one_errors=one_readout_errors,
        idle_phase_error=arguments.idle_phase_error,
        simulator_backend=arguments.backend,
    )
    result = harness.evaluate_solution(
        create_dicke_state,
        num_qubits=arguments.num_qubits,
        num_excitations=arguments.num_excitations,
        shots=arguments.shots,
        seed=arguments.seed,
    )
    result["error_seed"] = arguments.error_seed

    print("VQM noisy-simulation report")
    print("=" * 40)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:>20}: {value:.8f}")
        else:
            print(f"{key:>20}: {value}")


if __name__ == "__main__":
    main()
