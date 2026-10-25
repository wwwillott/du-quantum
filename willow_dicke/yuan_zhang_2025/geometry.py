"""Willow rectangles used as the paper's grid.

Paper coordinates are the ones in Theorem 14: ``n1`` rows by ``n2`` columns
with ``n1 <= n2``. Willow105 has no solid 7-by-8 block in the device's own
row-column orientation. Rows 2..9 and columns 4..10 are a solid 8-by-7
block. Transposing it yields the paper's ``Grid_{7,8}``. Physical Willow
coordinates stay in this module. The compiler works in paper coordinates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cirq

from willow_dicke.device import load_static_willow
from willow_dicke.graph import WillowGraph
from willow_dicke.rectangle import Rectangle

# Primary solid patch from the specification, transposed into paper order.
PRIMARY_ROW0 = 2
PRIMARY_COL0 = 4
PAPER_HEIGHT = 7
PAPER_WIDTH = 8


@dataclass(frozen=True)
class GridPatch:
    """One filled rectangle, addressed both ways.

    ``paper_rows[r][c]`` is the Willow qubit for paper row ``r`` and paper
    column ``c``. ``rows`` and ``cols`` are the occupied physical coordinates.
    """

    name: str
    qubits: tuple[cirq.GridQubit, ...]
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    adjacency: dict[cirq.GridQubit, tuple[cirq.GridQubit, ...]]
    width: int
    height: int
    paper_rows: tuple[tuple[cirq.GridQubit, ...], ...]
    device_name: str

    def __post_init__(self) -> None:
        if self.height * self.width != len(self.qubits):
            raise ValueError("patch qubit count does not match its dimensions")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("patch repeats a qubit")

    def rectangle(self) -> Rectangle:
        return Rectangle(self.paper_rows)

    def edge_tuples(self) -> tuple[tuple[cirq.GridQubit, cirq.GridQubit], ...]:
        found = []
        for qubit, neighbors in self.adjacency.items():
            for other in neighbors:
                if (qubit.row, qubit.col) < (other.row, other.col):
                    found.append((qubit, other))
        return tuple(found)

    def physical_edge_counts(self) -> dict[str, int]:
        horizontal = 0
        vertical = 0
        for left, right in self.edge_tuples():
            if left.row == right.row:
                horizontal += 1
            elif left.col == right.col:
                vertical += 1
            else:
                raise AssertionError(f"{left} -- {right} is not a grid edge")
        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "total": horizontal + vertical,
        }

    def graph_hash(self) -> str:
        lines = [
            f"{a.row},{a.col}-{b.row},{b.col}"
            for a, b in self.edge_tuples()
        ]
        payload = "\n".join(lines).encode()
        return hashlib.sha256(payload).hexdigest()


def _paper_qubit(row0: int, col0: int, paper_row: int, paper_col: int) -> cirq.GridQubit:
    """Paper row runs along a physical column; paper column along a physical row."""
    return cirq.GridQubit(row0 + paper_col, col0 + paper_row)


def build_transposed_patch(
    willow: WillowGraph,
    *,
    name: str,
    row0: int,
    col0: int,
    paper_height: int = PAPER_HEIGHT,
    paper_width: int = PAPER_WIDTH,
) -> GridPatch:
    """Build a paper ``paper_height`` by ``paper_width`` patch and require every edge."""
    if paper_height > paper_width:
        raise ValueError("paper height must be the shorter side")
    paper_rows = tuple(
        tuple(
            _paper_qubit(row0, col0, paper_row, paper_col) for paper_col in range(paper_width)
        )
        for paper_row in range(paper_height)
    )
    qubits = tuple(qubit for row in paper_rows for qubit in row)
    missing = [qubit for qubit in qubits if qubit not in willow.qubits]
    if missing:
        raise AssertionError(f"{name} is missing {len(missing)} Willow qubits, first {missing[0]}")
    adjacency: dict[cirq.GridQubit, tuple[cirq.GridQubit, ...]] = {}
    for row_index, row in enumerate(paper_rows):
        for col_index, qubit in enumerate(row):
            neighbors = []
            if col_index + 1 < paper_width:
                neighbors.append(row[col_index + 1])
            if col_index - 1 >= 0:
                neighbors.append(row[col_index - 1])
            if row_index + 1 < paper_height:
                neighbors.append(paper_rows[row_index + 1][col_index])
            if row_index - 1 >= 0:
                neighbors.append(paper_rows[row_index - 1][col_index])
            neighbors.sort(key=lambda item: (item.row, item.col))
            for other in neighbors:
                if not willow.has_edge(qubit, other):
                    raise AssertionError(f"{name} coupler {qubit} -- {other} is not on {willow.name}")
            adjacency[qubit] = tuple(neighbors)
    physical_rows = tuple(range(row0, row0 + paper_width))
    physical_cols = tuple(range(col0, col0 + paper_height))
    return GridPatch(
        name=name,
        qubits=qubits,
        rows=physical_rows,
        cols=physical_cols,
        adjacency=adjacency,
        width=paper_width,
        height=paper_height,
        paper_rows=paper_rows,
        device_name=willow.name,
    )


def primary_willow_rectangle(willow: WillowGraph | None = None) -> GridPatch:
    """The specification's rows 2..9, columns 4..10, as a paper 7 by 8 grid."""
    willow = willow or load_static_willow()
    return build_transposed_patch(
        willow, name="willow_rows_2_9_cols_4_10", row0=PRIMARY_ROW0, col0=PRIMARY_COL0
    )


def shifted_willow_rectangle(willow: WillowGraph | None = None) -> GridPatch:
    """The alternative rows 3..10, columns 4..10."""
    willow = willow or load_static_willow()
    return build_transposed_patch(
        willow, name="willow_rows_3_10_cols_4_10", row0=3, col0=PRIMARY_COL0
    )


def abstract_patch(height: int, width: int) -> GridPatch:
    """An ideal rectangle on fresh grid qubits, already in paper orientation.

    If ``height > width`` the axes are swapped so the paper's ``n1 <= n2``
    convention holds. The caller's requested shape is preserved up to that
    transpose, which does not change the coupling graph.
    """
    if height < 1 or width < 1:
        raise ValueError("both sides must be positive")
    if height > width:
        height, width = width, height
    paper_rows = tuple(
        tuple(cirq.GridQubit(row, col) for col in range(width)) for row in range(height)
    )
    qubits = tuple(qubit for row in paper_rows for qubit in row)
    adjacency = {}
    for row_index, row in enumerate(paper_rows):
        for col_index, qubit in enumerate(row):
            neighbors = []
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other_row, other_col = row_index + d_row, col_index + d_col
                if 0 <= other_row < height and 0 <= other_col < width:
                    neighbors.append(paper_rows[other_row][other_col])
            neighbors.sort(key=lambda item: (item.row, item.col))
            adjacency[qubit] = tuple(neighbors)
    return GridPatch(
        name=f"abstract_{height}x{width}",
        qubits=qubits,
        rows=tuple(range(height)),
        cols=tuple(range(width)),
        adjacency=adjacency,
        width=width,
        height=height,
        paper_rows=paper_rows,
        device_name="abstract",
    )
