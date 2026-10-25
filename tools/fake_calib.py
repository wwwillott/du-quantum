"""Fake calibration maps, copied from the official notebook (plus a seeded wrapper)."""
import numpy as np
import cirq
import cirq_google


def _all_qubits():
    return np.array(sorted(cirq_google.Willow105.metadata.qubit_set))


def pick_single_qubit_dropouts(dropout_rate, rng):
    all_qubits = _all_qubits()
    is_dropout = rng.random(len(all_qubits)) < dropout_rate
    return set(all_qubits[is_dropout])


def generate_qubit_error_rates(median, spread, dropouts, uncertainty, rng):
    all_qubits = _all_qubits()
    is_working = np.array([q not in dropouts for q in all_qubits])
    working_qubits = all_qubits[is_working]
    error_rate = rng.lognormal(np.log(median), spread, len(working_qubits))
    return dict(zip(working_qubits, zip(error_rate, uncertainty * np.ones(len(working_qubits)))))


def generate_two_qubit_pauli_error_rates(median=2.5e-3, spread=0.5, dropout_rate=0.01,
                                         uncertainty=5e-4, rng=None, single_qubit_dropouts=None):
    if single_qubit_dropouts is None:
        single_qubit_dropouts = set()
    all_pairs = sorted(tuple(sorted(p)) for p in cirq_google.Willow105.metadata.qubit_pairs)
    all_pairs = np.array(all_pairs, dtype=object)
    no_qubit_dropouts = np.array([all(q not in single_qubit_dropouts for q in pair) for pair in all_pairs])
    is_working = (rng.random(len(all_pairs)) > dropout_rate) * no_qubit_dropouts
    working_pairs = [tuple(pair) for pair in all_pairs[is_working]]
    error_rate = rng.lognormal(np.log(median), spread, len(working_pairs))
    return dict(zip(working_pairs, zip(error_rate, uncertainty * np.ones(len(working_pairs)))))


def fake_calibration(seed=0, q_dropout=0.01, c_dropout=0.01, cz_median=2.5e-3, sq_median=4e-4,
                     e0_median=5e-3, e1_median=1e-2):
    """Return (cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors) like the notebook."""
    rng = np.random.default_rng(seed)
    drop = pick_single_qubit_dropouts(q_dropout, rng)
    sq = generate_qubit_error_rates(sq_median, 0.5, drop, 2e-4, rng)
    cz = generate_two_qubit_pauli_error_rates(cz_median, 0.5, c_dropout, 5e-4, rng, drop)
    z0 = generate_qubit_error_rates(e0_median, 0.5, drop, 2e-4, rng)
    z1 = generate_qubit_error_rates(e1_median, 0.5, drop, 3e-4, rng)
    return cz, sq, z0, z1
