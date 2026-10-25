"""Experimental challenge-2 API: keep the incumbent unless a prototype wins.

Uses the pinned local Will snapshot; this module is NOT a standalone submission.
Available variants: will, yu_one_round, yu_two_rounds, best_validated (default).
The forced Yu variants substitute only the three independently verified cases.
"""
from functools import lru_cache

import cirq
from adaptive_dicke_validation.audit import ROOT, load_source
from adaptive_dicke_validation.paper_prototype import build


@lru_cache(maxsize=1)
def incumbent_module():
    return load_source(ROOT / "snapshots/will-605b9a8/dicke_submission.py", "will_prototype_incumbent")


@lru_cache(maxsize=6)
def paper(n, k, rounds):
    circuit, outputs, _ = build(n, k, rounds, incumbent_module())
    return cirq.FrozenCircuit(circuit), frozenset(outputs)


def create_dicke_state_adaptive(n, k, variant="best_validated"):
    incumbent, outputs = incumbent_module().create_dicke_state_adaptive(n, k)
    if variant == "will" or (n, k) not in {(2, 1), (4, 1), (4, 2)}:
        return incumbent, outputs
    if variant in {"yu_one_round", "yu_two_rounds"}:
        candidate, target = paper(n, k, 1 if variant == "yu_one_round" else 2)
        return cirq.Circuit(candidate), set(target)
    if variant != "best_validated":
        raise ValueError(variant)
    choices = [(incumbent, outputs)] + [paper(n, k, rounds) for rounds in [1, 2]]
    circuit, target = min(choices, key=lambda pair: len(pair[0]))
    return cirq.Circuit(circuit), set(target)
