"""Turn a Google processor calibration into the notebook's error maps.

``create_dicke_state`` expects four dictionaries of ``(error, uncertainty)``.
The bundled Willow calibration stores one point estimate per metric and no
uncertainty, so the second entry is 0.

Metric names on ``willow_pink``:

* ``single_qubit_rb_pauli_error_per_gate`` -> single-qubit Pauli error
* ``two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle`` -> CZ Pauli error
* ``single_qubit_p00_error`` -> readout error 0 -> 1
* ``single_qubit_p11_error`` -> readout error 1 -> 0
"""

from __future__ import annotations

import cirq


def _rate(value) -> float:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("calibration entry has no value")
        return float(value[0])
    return float(value)


def _qubit(key) -> cirq.Qid:
    if isinstance(key, tuple):
        return key[0]
    return key


def error_rates_from_calibration(calibration) -> tuple[dict, dict, dict, dict]:
    """Return ``(cz, single_qubit, zero, one)`` in the challenge's format."""

    def singles(metric: str) -> dict[cirq.Qid, tuple[float, float]]:
        return {_qubit(key): (_rate(value), 0.0) for key, value in calibration[metric].items()}

    cz: dict[tuple[cirq.Qid, cirq.Qid], tuple[float, float]] = {}
    metric = "two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle"
    for key, value in calibration[metric].items():
        left, right = key
        cz[(left, right)] = (_rate(value), 0.0)
    return (
        cz,
        singles("single_qubit_rb_pauli_error_per_gate"),
        singles("single_qubit_p00_error"),
        singles("single_qubit_p11_error"),
    )
