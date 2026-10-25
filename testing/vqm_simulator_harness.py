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
    cz_pauli_error_rate,
    native_h,
    one_errors,
    sq_pauli_error_rate,
    zero_errors,
)


ErrorMap = Mapping[object, float | tuple[float, float]]


class VQMSimulatorHarness:
    """Noisy simulator and scoring harness using Willow-style error maps."""

    def __init__(
        self,
        sq_errors: ErrorMap,
        cz_errors: ErrorMap,
        zero_errors: ErrorMap,
        one_errors: ErrorMap,
    ) -> None:
        self.sq_errors = sq_errors
        self.cz_errors = cz_errors
        self.zero_errors = zero_errors
        self.one_errors = one_errors

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
            for operation in moment.operations:
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

    def score_z_distribution(
        self,
        samples: np.ndarray,
        num_qubits: int,
        num_excitations: int,
    ) -> float:
        """Score Z-basis subspace retention and in-subspace uniformity."""
        shots = samples.shape[0]
        valid_mask = np.sum(samples, axis=1) == num_excitations
        subspace_retention = float(np.mean(valid_mask))
        if subspace_retention == 0:
            return 1e6

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
        return float((1.0 - subspace_retention) + 0.01 * chi_squared)

    @staticmethod
    def score_x_distribution(samples: np.ndarray) -> float:
        """Score full-register X-basis parity preservation."""
        eigenvalues = 1.0 - 2.0 * samples
        parity_correlation = np.mean(np.prod(eigenvalues, axis=1))
        return float(np.abs(1.0 - np.abs(parity_correlation)))

    def evaluate_solution(
        self,
        circuit_fn,
        num_qubits: int,
        num_excitations: int,
        shots: int = 10_000,
        seed: int | None = None,
    ) -> dict[str, float | int]:
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

        simulator = cirq.Simulator(seed=seed)
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
        score_x = self.score_x_distribution(samples_x)
        return {
            "N": num_qubits,
            "n": num_excitations,
            "shots": shots,
            "subspace_retention": float(
                np.mean(np.sum(samples_z, axis=1) == num_excitations)
            ),
            "score_z": score_z,
            "score_x": score_x,
            "overall_score": 1.0 / (score_z + score_x + 1e-8),
        }


def main() -> None:
    """Run the current Dicke-state implementation through the harness."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", "-N", type=int, default=4)
    parser.add_argument("--num-excitations", "-n", type=int, default=2)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12345)
    arguments = parser.parse_args()

    harness = VQMSimulatorHarness(
        sq_errors=sq_pauli_error_rate,
        cz_errors=cz_pauli_error_rate,
        zero_errors=zero_errors,
        one_errors=one_errors,
    )
    result = harness.evaluate_solution(
        create_dicke_state,
        num_qubits=arguments.num_qubits,
        num_excitations=arguments.num_excitations,
        shots=arguments.shots,
        seed=arguments.seed,
    )

    print("VQM noisy-simulation report")
    print("=" * 40)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:>20}: {value:.8f}")
        else:
            print(f"{key:>20}: {value}")


if __name__ == "__main__":
    main()
