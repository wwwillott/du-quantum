"""Logical tokens and the physical qubits that currently hold them."""

from __future__ import annotations

from dataclasses import dataclass, field

import cirq


@dataclass(frozen=True, order=True)
class LogicalToken:
    kind: str
    index: int


@dataclass
class Placement:
    """Inverse bijection between tokens and the qubits on one corridor."""

    logical_to_physical: dict[LogicalToken, cirq.Qid] = field(default_factory=dict)
    physical_to_logical: dict[cirq.Qid, LogicalToken] = field(default_factory=dict)

    def check(self) -> None:
        if len(self.logical_to_physical) != len(self.physical_to_logical):
            raise AssertionError("placement maps have different sizes")
        for token, qubit in self.logical_to_physical.items():
            if self.physical_to_logical.get(qubit) != token:
                raise AssertionError(f"placement is not a bijection at {qubit}")

    def swap(self, a: cirq.Qid, b: cirq.Qid) -> None:
        token_a = self.physical_to_logical[a]
        token_b = self.physical_to_logical[b]
        self.physical_to_logical[a] = token_b
        self.physical_to_logical[b] = token_a
        self.logical_to_physical[token_a] = b
        self.logical_to_physical[token_b] = a
        self.check()
