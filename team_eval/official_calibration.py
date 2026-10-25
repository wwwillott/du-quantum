"""Official notebook fake calibration generators, copied verbatim from cell 11."""
import cirq
import cirq_google
import numpy as np
import hashlib
import json
from pathlib import Path

def pick_single_qubit_dropouts(
    dropout_rate: float,
    rng: np.random.Generator = np.random.default_rng()
) -> set[cirq.GridQubit]:
    """Pick some qubits to be broken.

    Args:
        dropout_rate: The probability that a qubit is broken.
        rng: The pseudo-random number generator

    Returns:
        The broken qubits.
    """
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_dropout = rng.random(len(all_qubits)) < dropout_rate
    return set(all_qubits[is_dropout])


def generate_qubit_error_rates(
    median: float,
    spread: float,
    dropouts: set[cirq.GridQubit],
    uncertainty: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> dict[cirq.GridQubit, tuple[float, float]]:
    """Generate fake single-qubit error rates.

    Args:
        median: The median error rate.
        spread: The spread of the lognormal distribution
        dropouts: Qubits to exclude
        uncertainty: The uncertainty of the measured error rates.
        rng: The pseudo-random number generator

    Returns:
        A dictionary from qubits to the measured error rate and statistical uncertainty.
    """

    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_working = np.array([q not in dropouts for q in all_qubits])
    working_qubits = all_qubits[is_working]
    error_rate = rng.lognormal(np.log(median), spread, len(working_qubits))
    return dict(
        zip(working_qubits, zip(error_rate, uncertainty * np.ones(len(working_qubits))))
    )


def generate_two_qubit_pauli_error_rates(
    median: float = 2.5e-3,
    spread: float = 0.5,
    dropout_rate: float = 0.01,
    uncertainty: float = 5e-4,
    rng: np.random.Generator = np.random.default_rng(),
    single_qubit_dropouts: set[cirq.GridQubit] = None,
) -> dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]]:
    """Generate fake two-qubit error rates.

    Args:
        median: The median error rate.
        spread: The spread of the lognormal distribution
        dropout_rate: The probability of a broken coupler.
        uncertainty: The uncertainty of the measured error rates.
        rng: The pseudo-random number generator
        single_qubit_dropouts: The single-qubit dropouts to skip.

    Returns:
        A dictionary from qubits to the measured error rate and statistical uncertainty.
    """
    if single_qubit_dropouts is None:
        single_qubit_dropouts = set()

    all_pairs = np.array(list(cirq_google.Willow105.metadata.qubit_pairs))
    no_qubit_dropouts = np.array(
        [np.all([q not in single_qubit_dropouts for q in pair]) for pair in all_pairs]
    )
    is_working = (rng.random(len(all_pairs)) > dropout_rate) * no_qubit_dropouts
    working_pairs = [tuple(pair) for pair in all_pairs[is_working]]
    error_rate = rng.lognormal(np.log(median), spread, len(working_pairs))
    return dict(
        zip(working_pairs, zip(error_rate, uncertainty * np.ones(len(working_pairs))))
    )


def seeded_calibration(seed: int) -> dict:
    """Generate one calibration with a single explicit RNG in notebook cell-12 order."""
    rng = np.random.default_rng(seed)
    single_qubit_dropouts = pick_single_qubit_dropouts(0.01, rng=rng)
    sq_pauli_error_rate = generate_qubit_error_rates(
        4e-4, 0.5, single_qubit_dropouts, 2e-4, rng=rng
    )
    cz_pauli_error_rate = generate_two_qubit_pauli_error_rates(
        single_qubit_dropouts=single_qubit_dropouts, rng=rng
    )
    zero_errors = generate_qubit_error_rates(
        5e-3, 0.5, single_qubit_dropouts, 2e-4, rng=rng
    )
    one_errors = generate_qubit_error_rates(
        1e-2, 0.5, single_qubit_dropouts, 3e-4, rng=rng
    )
    return dict(
        cz_pauli_error_rate=cz_pauli_error_rate,
        sq_pauli_error_rate=sq_pauli_error_rate,
        zero_errors=zero_errors,
        one_errors=one_errors,
    )


def _serializable(calibration: dict) -> dict:
    """Canonical coordinates plus rate/uncertainty; no Python hash dependence."""
    serialized = {}
    for name, rates in calibration.items():
        if name == "cz_pauli_error_rate":
            serialized[name] = [
                [[a.row, a.col], [b.row, b.col], float(rate), float(uncertainty)]
                for (a, b), (rate, uncertainty) in sorted(
                    (tuple(sorted(pair)), value) for pair, value in rates.items()
                )
            ]
        else:
            serialized[name] = [
                [[q.row, q.col], float(rate), float(uncertainty)]
                for q, (rate, uncertainty) in sorted(rates.items())
            ]
    return serialized


def calibration_fingerprint(calibration: dict) -> str:
    payload = json.dumps(_serializable(calibration), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calibration_path(seed: int) -> Path:
    return Path(__file__).resolve().parent / "calibrations" / f"seed_{seed}.json"


def write_calibration(seed: int, path=None, overwrite: bool = False) -> Path:
    """Materialize exactly once; all candidate workers should load this artifact."""
    path = calibration_path(seed) if path is None else Path(path)
    if path.exists() and not overwrite:
        return path
    calibration = seeded_calibration(seed)
    payload = dict(
        seed=int(seed),
        fingerprint=calibration_fingerprint(calibration),
        generator_source="DuQuantum_Google_Standard.ipynb cell 11 (verbatim)",
        call_order="dropouts, sq, cz, zero, one; one explicit default_rng(seed)",
        data=_serializable(calibration),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_calibration(seed: int, path=None) -> dict:
    """Load the shared saved calibration; fail rather than silently regenerate."""
    path = calibration_path(seed) if path is None else Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["seed"]) != int(seed):
        raise ValueError(f"Calibration seed mismatch in {path}")
    calibration = {}
    for name, entries in payload["data"].items():
        if name == "cz_pauli_error_rate":
            calibration[name] = {
                (cirq.GridQubit(*a), cirq.GridQubit(*b)): (float(rate), float(uncertainty))
                for a, b, rate, uncertainty in entries
            }
        else:
            calibration[name] = {
                cirq.GridQubit(*q): (float(rate), float(uncertainty))
                for q, rate, uncertainty in entries
            }
    if calibration_fingerprint(calibration) != payload["fingerprint"]:
        raise ValueError(f"Calibration fingerprint mismatch in {path}")
    return calibration
