"""The solid 7 by 8 patch of Willow105.

Yuan and Zhang, arXiv:2505.15413, state the grid theorem for an
``n1`` by ``n2`` rectangle with ``n1 <= n2``. Willow105 has no solid
7-row by 8-column block. Rows 2..9 and columns 4..10 are a solid
8 by 7 block. Transposing it gives ``Grid_{7,8}``: seven paper rows
along the physical columns, eight paper columns along the physical rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import cirq

from willow_dicke.graph import WillowGraph

# Physical origin of the patch. Paper row r sits on physical column
# ``COL0 + r``; paper column c sits on physical row ``ROW0 + c``.
ROW0 = 2
COL0 = 4
N1 = 7
N2 = 8


@dataclass(frozen=True)
class Rectangle:
    """A filled grid. ``rows[r][c]`` are adjacent when the indices are."""

    rows: tuple[tuple[cirq.GridQubit, ...], ...]

    def __post_init__(self) -> None:
        if not self.rows or not self.rows[0]:
            raise ValueError("rectangle is empty")
        width = len(self.rows[0])
        if any(len(row) != width for row in self.rows):
            raise ValueError("rectangle is ragged")

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def n(self) -> int:
        return self.height * self.width

    def qubits(self) -> tuple[cirq.GridQubit, ...]:
        return tuple(qubit for row in self.rows for qubit in row)

    def qubit_set(self) -> frozenset[cirq.GridQubit]:
        return frozenset(self.qubits())

    def edges(self) -> tuple[tuple[cirq.GridQubit, cirq.GridQubit], ...]:
        found = []
        for r, row in enumerate(self.rows):
            for c, qubit in enumerate(row):
                if c + 1 < self.width:
                    found.append((qubit, row[c + 1]))
                if r + 1 < self.height:
                    found.append((qubit, self.rows[r + 1][c]))
        return tuple(found)

    def sub_rows(self, start: int, stop: int) -> "Rectangle":
        return Rectangle(self.rows[start:stop])

    def sub_cols(self, start: int, stop: int) -> "Rectangle":
        return Rectangle(tuple(tuple(row[start:stop]) for row in self.rows))

    def snake(self) -> tuple[cirq.GridQubit, ...]:
        """Row-major snake. Even paper rows run left to right."""
        ordered = []
        for index, row in enumerate(self.rows):
            ordered.extend(row if index % 2 == 0 else tuple(reversed(row)))
        return tuple(ordered)


def abstract_rectangle(n1: int, n2: int) -> Rectangle:
    """An ``n1`` by ``n2`` grid of fresh ``GridQubit``s, ``n1`` the height."""
    if n1 < 1 or n2 < 1:
        raise ValueError("both sides must be positive")
    rows = tuple(
        tuple(cirq.GridQubit(row, col) for col in range(n2)) for row in range(n1)
    )
    return Rectangle(rows)


def willow_7x8_rectangle(willow: WillowGraph) -> Rectangle:
    """The unique solid ``Grid_{7,8}`` inside ``willow``.

    Raises ``AssertionError`` if any of the 56 sites or any grid coupler
    inside the patch is missing. No edge is added.
    """
    rows = []
    for paper_row in range(N1):
        row = []
        for paper_col in range(N2):
            row.append(cirq.GridQubit(ROW0 + paper_col, COL0 + paper_row))
        rows.append(tuple(row))
    rect = Rectangle(tuple(rows))
    problems = solidity_problems(rect, willow)
    if problems:
        raise AssertionError("; ".join(problems))
    return rect


def solidity_problems(rect: Rectangle, willow: WillowGraph) -> list[str]:
    """Qubits or couplers the rectangle needs and the device does not have."""
    problems = []
    if rect.height != N1 or rect.width != N2 or rect.n != N1 * N2:
        problems.append(f"expected a {N1} by {N2} rectangle, got {rect.height} by {rect.width}")
    missing_qubits = [qubit for qubit in rect.qubits() if qubit not in willow.qubits]
    if missing_qubits:
        problems.append(f"{len(missing_qubits)} patch qubits are not on the device")
    if len(rect.qubit_set()) != rect.n:
        problems.append("patch repeats a qubit")
    missing_edges = [(a, b) for a, b in rect.edges() if not willow.has_edge(a, b)]
    if missing_edges:
        first = missing_edges[0]
        problems.append(f"{len(missing_edges)} patch couplers are missing, first {first}")
    return problems


def adjacent(a: cirq.Qid, b: cirq.Qid) -> bool:
    return abs(a.row - b.row) + abs(a.col - b.col) == 1


def neighbors(qubit: cirq.GridQubit, allowed: set[cirq.GridQubit]) -> list[cirq.GridQubit]:
    """Grid neighbors inside ``allowed``, in a fixed compass order."""
    found = []
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        other = cirq.GridQubit(qubit.row + d_row, qubit.col + d_col)
        if other in allowed:
            found.append(other)
    return found


def assert_path(qubits: tuple[cirq.Qid, ...] | list[cirq.Qid]) -> None:
    for left, right in zip(qubits, qubits[1:]):
        if not adjacent(left, right):
            raise RuntimeError(f"{left} and {right} are not a grid edge")
