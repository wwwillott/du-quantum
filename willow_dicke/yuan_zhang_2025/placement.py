"""Where the unary registers sit, and the path Lemma 8 runs on.

Finite-size choice, not in the integer-tiling proof: the two ``k``-qubit
registers are Hamiltonian paths inside the children, joined by one cut edge,
rather than ``sqrt(k)`` by ``sqrt(k)`` blocks. A 7 by 8 grid does not admit
that tile for every ``k`` we benchmark. The path is exactly what Lemma 8
requires. Block shape would only matter for the permutation that follows,
and that permutation is computed from the actual endpoints.
"""

from __future__ import annotations

import cirq

from willow_dicke.rectangle import Rectangle, assert_path, neighbors
from willow_dicke.yuan_zhang_2025.partition import is_leaf, peel_applies, split_longer


def snake(region: Rectangle) -> tuple[cirq.GridQubit, ...]:
    path = region.snake()
    assert_path(path)
    return path


def leaf_port(region: Rectangle, k: int) -> tuple[cirq.GridQubit, ...]:
    """The ``k`` qubits at the end of the snake. Ones sit on the right."""
    if k > region.n:
        raise ValueError(f"port length {k} exceeds the region")
    if region.n == k:
        path = _path_covering(region)
        return path
    return snake(region)[-k:]


def _path_covering(region: Rectangle) -> tuple[cirq.GridQubit, ...]:
    """A Hamiltonian path. The leaf unitary is applied on this order."""
    return snake(region)


def entry_port(region: Rectangle, k: int) -> tuple[cirq.GridQubit, ...]:
    """Where the unary register must sit when this region's circuit starts.

    A peeled strip keeps that register on the top row, which is the right end
    of ``strip_path``. Every other leaf uses the right end of the snake.
    An internal node uses the right half of its Divide path.
    """
    if peel_applies(region, k):
        return tuple(region.rows[0][:k])
    if is_leaf(region, k):
        return leaf_port(region, k)
    sites = divide_sites(region, k)
    if sites is None:
        return leaf_port(region, k)
    _left, _right, _s1, s2 = sites
    return s2


def divide_sites(
    region: Rectangle, k: int
) -> tuple[Rectangle, Rectangle, tuple[cirq.GridQubit, ...], tuple[cirq.GridQubit, ...]] | None:
    """Left, right, and the two ``k``-qubit halves of the Lemma 8 path."""
    if peel_applies(region, k):
        if region.width == k:
            return None
        left = region.sub_cols(0, k)
        right = region.sub_cols(k, region.width)
        first = tuple(region.rows[0][:k])
        second = tuple(region.rows[0][k : 2 * k])
        assert_path(first + second)
        return left, right, first, second
    split = split_longer(region, k)
    if split is None:
        return None
    left, right, _axis = split
    s1, s2 = _registers_across_cut(left, right, k)
    return left, right, s1, s2


def _registers_across_cut(
    left: Rectangle, right: Rectangle, k: int
) -> tuple[tuple[cirq.GridQubit, ...], tuple[cirq.GridQubit, ...]]:
    """``S1`` ends on the cut and ``S2`` starts on it, so ``S1 + S2`` is a path."""
    right_qubits = right.qubit_set()
    cuts = []
    for qubit in sorted(left.qubits(), key=lambda item: (item.row, item.col)):
        for other in neighbors(qubit, right_qubits):
            cuts.append((qubit, other))
    for end, start in cuts:
        toward = _path_from(end, k, left.qubit_set())
        away = _path_from(start, k, right_qubits)
        if toward is None or away is None:
            continue
        s1 = tuple(reversed(toward))
        s2 = away
        assert_path(s1 + s2)
        return s1, s2
    raise RuntimeError(
        f"no {k}-qubit divide path across a {left.height}x{left.width} | "
        f"{right.height}x{right.width} cut"
    )


def _path_from(start: cirq.GridQubit, k: int, nodes: set[cirq.GridQubit]):
    """A simple ``k``-qubit path that starts at ``start``, or ``None``."""
    if start not in nodes or k < 1 or len(nodes) < k:
        return None

    def dfs(path: tuple[cirq.GridQubit, ...], used: set[cirq.GridQubit]):
        if len(path) == k:
            return path
        for nxt in neighbors(path[-1], nodes):
            if nxt in used:
                continue
            used.add(nxt)
            found = dfs((*path, nxt), used)
            if found is not None:
                return found
            used.remove(nxt)
        return None

    return dfs((start,), {start})


def strip_path(region: Rectangle) -> tuple[cirq.GridQubit, ...]:
    """Hamiltonian path whose last paper row is the top row, left to right.

    Used for a peeled strip of width ``k`` so the unary register, which
    occupies that top row, is the right end of the 2019 stair.
    """
    ordered = []
    for row_index in range(region.height - 1, -1, -1):
        row = list(region.rows[row_index])
        if row_index % 2 == 1:
            row.reverse()
        ordered.extend(row)
    path = tuple(ordered)
    assert_path(path)
    return path
