"""Exactness / validity checks for dicke.create_dicke_state on fake calibrations.

Run: python test_dicke.py [max_N]
"""
import itertools
import math
import sys
import time

import cirq
import cirq_google
import numpy as np

from dicke import create_dicke_state

DEVICE = cirq_google.Willow105


def fake_calibration(seed=0, q_dropout=0.01, c_dropout=0.01):
    """Same generators as the challenge notebook, seeded."""
    rng = np.random.default_rng(seed)
    all_qubits = sorted(DEVICE.metadata.qubit_set)
    dropouts = {q for q in all_qubits if rng.random() < q_dropout}
    working = [q for q in all_qubits if q not in dropouts]

    def rates(median, unc):
        r = rng.lognormal(np.log(median), 0.5, len(working))
        return {q: (x, unc) for q, x in zip(working, r)}

    pairs = [tuple(sorted(p)) for p in DEVICE.metadata.qubit_pairs]
    pairs = sorted(p for p in pairs if not set(p) & dropouts and rng.random() > c_dropout)
    cz = {p: (x, 5e-4) for p, x in zip(pairs, rng.lognormal(np.log(2.5e-3), 0.5, len(pairs)))}
    return cz, rates(4e-4, 2e-4), rates(5e-3, 2e-4), rates(1e-2, 3e-4)


def dicke_statevector(N, n):
    psi = np.zeros(2**N)
    for exc in itertools.combinations(range(N), n):
        psi[sum(2**i for i in exc)] = 1
    return psi / np.linalg.norm(psi)


def fidelity(circuit, n, qubits):
    """Notebook's test_circuit, with keep_indices fixed to range(len(qubits))."""
    all_qubits = circuit.all_qubits() | set(qubits)
    order = sorted(qubits) + sorted(all_qubits - set(qubits))
    psi = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=order).final_state_vector
    if len(order) > len(qubits):
        psi = cirq.sub_state_vector(psi, keep_indices=range(len(qubits)))
    return abs(psi @ dicke_statevector(len(qubits), n)) ** 2


def check(N, n, cal):
    cz, sq, e0, e1 = cal
    t0 = time.time()
    circuit, qubits = create_dicke_state(N, n, cz, sq, e0, e1)
    dt = time.time() - t0
    DEVICE.validate_circuit(circuit)
    for op in circuit.all_operations():
        if len(op.qubits) == 2:
            assert op.gate == cirq.CZ and (op.qubits in cz or op.qubits[::-1] in cz), op
        else:
            assert isinstance(op.gate, cirq.PhasedXZGate) and op.qubits[0] in sq, op
    assert len(circuit) <= 1750
    ncz = sum(1 for op in circuit.all_operations() if len(op.qubits) == 2)
    f = fidelity(circuit, n, qubits) if N <= 16 else float("nan")
    return f, ncz, len(circuit), dt


if __name__ == "__main__":
    max_N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    cal = fake_calibration(seed=1)
    bad = 0
    print(" N  n | fidelity     | CZs | moments | build s")
    for N in range(1, max_N + 1):
        for n in range(0, N + 1):
            f, ncz, depth, dt = check(N, n, cal)
            ok = np.isclose(f, 1.0, atol=1e-9) or math.isnan(f)
            bad += not ok
            print(f"{N:2d} {n:2d} | {f:.12f} | {ncz:3d} | {depth:7d} | {dt:6.2f}{'' if ok else '  FAIL'}")
    print("failures:", bad)
