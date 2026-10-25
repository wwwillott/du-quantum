"""Exact-state, depth-scored Problem 2 integration."""
from functools import lru_cache
from pathlib import Path

import cirq

from . import portfolio


@lru_cache(maxsize=1)
def interference_candidates():
    """Saved experiments are candidates only after an independent exact-state check.

    Approximate free-XY and two-CP circuits are deliberately retained in the
    experiment report, but cannot be submitted to the exact-state challenge.
    """
    from .verify import unitary_fidelity
    root = Path(__file__).parents[1] / 'artifacts' / 'pruned-20261025T080258195319Z'
    candidates = {}
    for n, k, name in [(4, 2, 'star-depth6'), (4, 2, 'chain-4-2-depth6-cp1'),
                       (6, 3, 'chain-6-3-depth10-cp1')]:
        circuit = cirq.read_json(root / (name + '.cirq.json'))
        outputs = set(circuit.all_qubits())
        if unitary_fidelity(circuit, outputs, k)['fidelity'] < 1 - 1e-9:
            raise ValueError(f'Saved interference circuit failed verification: {name}')
        candidates.setdefault((n, k), []).append((circuit, outputs, name))
    return candidates


@lru_cache(maxsize=128)
def _selected(n, k):
    # Validate domain before entering the private, cached portfolio builder.
    circuit, outputs = portfolio.create_dicke_state_adaptive(n, k)
    choices = [(circuit, outputs, 'prior portfolio: ' + portfolio.method_for(n, k))]
    from problem2_submission import _judge_depth_improvement
    improvement = _judge_depth_improvement(n, k)
    if improvement is not None:
        c, q = improvement
        choices.append((c, q, 'Will 605b9a8 all-branch verified depth improvement'))
    choices.extend(interference_candidates().get((n, k), []))
    c, q, method = min(choices, key=lambda item: len(item[0]))
    return cirq.FrozenCircuit(c), frozenset(q), method


def create_dicke_state_adaptive(num_qubits, num_excitations):
    c, q, _ = _selected(num_qubits, num_excitations)
    return cirq.Circuit(c), set(q)


def method_for(n, k):
    return _selected(n, k)[2]
