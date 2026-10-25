"""Command line for Dicke preparation on Willow.

Simulation is the default. ``--hardware`` prints the processor summary and
does not submit a job; submission requires ``--submit`` as well, and this
process still refuses to call the Quantum Engine run API. That keeps a
mistyped flag from launching a paid Willow job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cirq
import cirq_google
import numpy as np

from dicke_willow.benchmarks import logical_metrics, write_benchmarks
from dicke_willow.calibration import error_rates_from_calibration
from dicke_willow.challenge import (
    circuit_report,
    create_dicke_state,
    create_dicke_state_adaptive,
)
from dicke_willow.compile import MOMENT_LIMIT, gate_counts, to_willow_gates
from dicke_willow.divide import prepare_balanced
from dicke_willow.scs import prepare_dicke_circuit
from dicke_willow.verify import fidelity, hamming_fraction, marginal_excitation


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Dicke state on Willow")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--device", default="willow", choices=("willow", "line"))
    parser.add_argument("--method", default="scs", choices=("scs", "balanced"))
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--qvm", action="store_true")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--processor-id", default="willow_pink")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args(argv)


def _line_circuit(n: int, k: int, method: str) -> tuple[cirq.Circuit, list[cirq.Qid]]:
    qubits = list(cirq.LineQubit.range(n))
    if method == "balanced":
        circuit, _region = prepare_balanced(qubits, k)
    else:
        circuit = prepare_dicke_circuit(qubits, k)
    return circuit, qubits


def _willow_circuit(n: int, k: int) -> tuple[cirq.Circuit, set]:
    return create_dicke_state_adaptive(n, k)


def _print_hardware_banner(processor_id: str, circuit: cirq.Circuit, qubits: set, repetitions: int) -> None:
    active = cirq_google.Willow105.metadata.qubit_set
    print(f"processor {processor_id}")
    print(f"number of active qubits {len(active)}")
    print(f"selected physical patch {sorted(qubits, key=str)}")
    counts = gate_counts(circuit)
    print(f"circuit depth {counts['moments']}")
    print(f"two-qubit gate count {counts['cz']}")
    print(f"number of repetitions {repetitions}")
    if counts["moments"] > MOMENT_LIMIT:
        print(f"warning: depth exceeds the {MOMENT_LIMIT}-moment hardware cutoff")


def _simulate(circuit: cirq.Circuit, qubits: list[cirq.Qid], k: int, repetitions: int) -> None:
    if len(qubits) <= 16:
        print(f"fidelity {fidelity(circuit, k, qubits):.12f}")
    else:
        print("fidelity skipped (n > 16)")
    sampler = cirq.Simulator()
    result = sampler.run(circuit + cirq.Circuit(cirq.measure(*qubits, key="m")), repetitions=repetitions)
    bits = result.measurements["m"]
    print(f"fraction of samples with Hamming weight k {hamming_fraction(bits, k):.6f}")
    print(f"per-bit excitation {np.array2string(marginal_excitation(bits), precision=3)}")
    ideal = k / len(qubits)
    print(f"ideal per-bit excitation {ideal:.6f}")


def _circuit_from_calibration(n: int, k: int, processor_id: str):
    """Place the stair using the virtual processor's own error rates."""
    engine = cirq_google.engine.create_default_noisy_quantum_virtual_machine(processor_id)
    calibration = engine.get_processor(processor_id).get_current_calibration()
    cz, single, zero, one = error_rates_from_calibration(calibration)
    circuit, qubits = create_dicke_state(n, k, cz, single, zero, one)
    cz_values = [rate for rate, _uncertainty in cz.values()]
    single_values = [rate for rate, _uncertainty in single.values()]
    print(f"calibration metrics {len(single)} qubits, {len(cz)} couplers")
    print(f"median single-qubit Pauli error {float(np.median(single_values)):.6g}")
    print(f"median CZ Pauli error {float(np.median(cz_values)):.6g}")
    print(f"selected path {sorted(qubits, key=str)}")
    return circuit, sorted(qubits)


def _qvm(circuit: cirq.Circuit, qubits: list[cirq.Qid], k: int, processor_id: str, repetitions: int) -> None:
    """Sample on cirq-google's bundled noisy virtual processor.

    The circuit must already be routed onto qubits of that processor.
    ``willow_pink`` rejects line qubits.
    """
    engine = cirq_google.engine.create_default_noisy_quantum_virtual_machine(processor_id)
    processor = engine.get_processor(processor_id)
    device = processor.get_device()
    print(f"qvm processor {processor_id}")
    print(f"number of active qubits {len(device.metadata.qubit_set)}")
    device.validate_circuit(circuit)
    ordered = sorted(qubits)
    measured = circuit + cirq.Circuit(cirq.measure(*ordered, key="m"))
    bits = engine.get_sampler(processor_id).run(measured, repetitions=repetitions).measurements["m"]
    print(f"fraction of samples with Hamming weight k {hamming_fraction(bits, k):.6f}")
    print(f"per-bit excitation {np.array2string(marginal_excitation(bits), precision=3)}")
    print(f"ideal per-bit excitation {k / len(qubits):.6f}")


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.benchmark:
        rows = []
        for n in range(2, min(args.n, 12) + 1):
            for k in range(1, min(4, n // 2) + 1):
                rows.append(logical_metrics(n, k))
                print(rows[-1])
        write_benchmarks(Path("benchmarks") / "logical_metrics.json", rows)
        return 0

    if args.qvm and args.device == "willow" and args.method == "scs":
        circuit, qubits = _circuit_from_calibration(args.n, args.k, args.processor_id)
        qubit_set = set(qubits)
    elif args.device == "willow" and args.method == "scs":
        circuit, qubit_set = _willow_circuit(args.n, args.k)
        qubits = sorted(qubit_set)
    else:
        circuit, qubits = _line_circuit(args.n, args.k, args.method)
        circuit = to_willow_gates(circuit)
        qubit_set = set(qubits)

    report = circuit_report(circuit, qubit_set, args.k)
    print(json.dumps({key: value for key, value in report.items() if key != "qubits"}))

    if args.hardware:
        _print_hardware_banner(args.processor_id, circuit, qubit_set, args.repetitions)
        if args.submit:
            print("refusing to submit: this driver does not call the Quantum Engine run API")
            return 2
        print("hardware execution was not submitted")

    if args.qvm:
        _qvm(circuit, qubits, args.k, args.processor_id, args.repetitions)
    elif args.simulate or not args.hardware:
        if args.n <= 16:
            _simulate(circuit, qubits, args.k, args.repetitions)
        else:
            print("structural checks only; n > 16")
            counts = gate_counts(circuit)
            print(f"moments {counts['moments']} cz {counts['cz']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
