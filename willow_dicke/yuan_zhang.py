"""Theorem 14 of Yuan and Zhang, arXiv:2505.15413, on one rectangle.

The grid cases are the ones in Section 3.2, not the all-to-all ancilla
divide. ``Divide`` is Lemma 8 on a path of ``2k`` qubits. A region that
cannot be split into two parts of size at least ``k`` is the 2019 SCS
unitary (Lemma 7). When ``k * n1 < n2`` and ``k`` divides ``n2``, the
rectangle is peeled in strips of height ``n1`` and width ``k`` (Figure 3b).
Otherwise the longer side is split as evenly as the weight constraint
allows (Figure 3a, with the paper's ceiling rule).

After ``Divide``, the ``k`` qubits of each register are swapped onto the
input of the corresponding child. The swap stays inside that child and
off the other register.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque

import cirq

from dicke_willow.scs import scs_unitary_ops
from willow_dicke.circuit.lower import (
    assert_cz_and_single_qubit,
    bridge_along_path,
    expand_circuit,
    is_full_cz,
    lower_to_willow_cz,
)
from willow_dicke.device import load_static_willow
from willow_dicke.divide.logical_divide import build_logical_divide
from willow_dicke.rectangle import (
    N1,
    N2,
    Rectangle,
    assert_path,
    neighbors,
    willow_7x8_rectangle,
)
from willow_dicke.verify.block_sim import fixed_weight_error, replay

# ``circuit.unitary`` on a wider block is the simulator limit, not a
# limit of the recursion. Columns of the 7 by 8 patch are 7 qubits.
MAX_BLOCK = 16


def coherent_preparation(rect: Rectangle, k: int) -> tuple[list, tuple[cirq.GridQubit, ...]]:
    """Unitary trace and the ``k``-qubit port that holds ``|0^{k-ell} 1^ell>``.

    The trace has no ``X`` gates. Ones sit on the right of the port.
    """
    if k < 1 or k > rect.n:
        raise ValueError(f"k={k} is outside 1..n={rect.n}")
    builder = _Builder()
    if _peel(rect, k):
        port = tuple(rect.rows[0][:k])
        _emit_peel(rect, k, builder)
        return builder.trace, port
    split = _split_longer(rect, k)
    if split is None:
        path = rect.snake()
        port = path[-k:]
        _emit_leaf(rect, k, port, builder)
        return builder.trace, port
    left, right = split
    _s1, s2 = _divide_sites(left, right, k)
    _emit_balanced(rect, k, s2, builder)
    return builder.trace, s2


def with_excitation(
    body: list, port: tuple[cirq.GridQubit, ...], ell: int
) -> list:
    """Prefix ``X`` on the rightmost ``ell`` qubits of the port."""
    if ell < 0 or ell > len(port):
        raise ValueError(f"ell={ell} is outside 0..{len(port)}")
    if ell == 0:
        return list(body)
    return [("pauli_x", tuple(port[-ell:])), *body]


def logical_circuit(trace: list) -> cirq.Circuit:
    ops: list[cirq.Operation] = []
    for kind, payload, *rest in _events(trace):
        if kind == "pauli_x":
            ops.extend(cirq.X(qubit) for qubit in payload)
        elif kind == "swaps":
            ops.extend(cirq.SWAP(a, b) for a, b in payload)
        else:
            ops.extend(payload.all_operations())
            del rest
    return cirq.Circuit(ops)


def lowered_circuit(trace: list) -> cirq.Circuit:
    """PhasedXZ and CZ. Every two-qubit gate is bridged onto a grid edge first."""
    lowered = cirq.Circuit()
    for kind, payload, *rest in _events(trace):
        if kind == "pauli_x":
            lowered.append(cirq.Moment(cirq.X(qubit) for qubit in payload))
        elif kind == "swaps":
            for left, right in payload:
                assert_path((left, right))
                lowered += lower_to_willow_cz(cirq.Circuit(cirq.SWAP(left, right)))
        else:
            block = rest[0]
            assert_path(block)
            expanded = expand_circuit(payload)
            bridged = bridge_along_path(expanded, block)
            lowered += lower_to_willow_cz(bridged)
    return lowered


def assert_cz_on_patch(circuit: cirq.AbstractCircuit, rect: Rectangle, willow) -> None:
    """Every CZ is a coupler of ``willow`` whose endpoints lie in ``rect``."""
    assert_cz_and_single_qubit(circuit)
    patch = rect.qubit_set()
    for op in circuit.all_operations():
        if not is_full_cz(op):
            continue
        left, right = op.qubits
        if left not in patch or right not in patch or not willow.has_edge(left, right):
            raise AssertionError(f"CZ {left} -- {right} is outside the Willow patch")


def block_report(trace: list, qubits: list[cirq.Qid], k: int) -> dict:
    """Replay Divide, SCS, and SWAP blocks and compare with ``|D_k>``."""
    for kind, payload, *rest in _events(trace):
        if kind == "unitary" and len(rest[0]) > MAX_BLOCK:
            raise RuntimeError(
                f"block of {len(rest[0])} qubits is above the simulator limit {MAX_BLOCK}"
            )
        del payload
    amplitudes = replay(trace, qubits)
    report = fixed_weight_error(amplitudes, len(qubits), k)
    report["relative_phase_ok"] = _relative_phase_ok(amplitudes, k)
    report["passed"] = bool(report["passed"] and report["relative_phase_ok"])
    report["k"] = k
    report["n"] = len(qubits)
    return report


def verify_willow(k: int) -> dict:
    """Block-simulator fidelity and the CZ-edge check on the 7 by 8 patch."""
    if k < 1 or k > (N1 * N2) // 2:
        raise ValueError(f"k={k} is outside 1..{(N1 * N2) // 2}")
    willow = load_static_willow()
    rect = willow_7x8_rectangle(willow)
    body, port = coherent_preparation(rect, k)
    trace = with_excitation(body, port, k)
    report = block_report(trace, list(rect.qubits()), k)
    lowered = lowered_circuit(trace)
    assert_cz_on_patch(lowered, rect, willow)
    report["cz_on_patch"] = True
    report["moments"] = len(lowered)
    return report


def _peel(rect: Rectangle, k: int) -> bool:
    """Figure 3(b): ``k < n2 / n1`` and the long side is a whole number of strips."""
    return k * rect.height < rect.width and rect.width % k == 0


def _split_longer(rect: Rectangle, k: int):
    """Split the longer side so both children have at least ``k`` qubits.

    Returns ``(left, right)`` or ``None`` when no such cut exists. ``left``
    is the low-index side. The cut closest to half the qubits wins.
    """
    height, width = rect.height, rect.width
    if width >= height:
        axis, length, other = "cols", width, height
    else:
        axis, length, other = "rows", height, width
    need = math.ceil(k / other)
    if length < 2 * need:
        return None
    target = rect.n / 2

    def score(lines: int) -> tuple[float, int]:
        return (abs(lines * other - target), lines)

    lines = min(range(need, length - need + 1), key=score)
    if axis == "cols":
        return rect.sub_cols(0, lines), rect.sub_cols(lines, width)
    return rect.sub_rows(0, lines), rect.sub_rows(lines, height)


def _divide_sites(left: Rectangle, right: Rectangle, k: int):
    """``S1`` in ``left`` and ``S2`` in ``right``, joined by one grid edge.

    ``S1`` ends on that edge and ``S2`` starts on it, so ``S1 + S2`` is the
    Lemma 8 path. Ones sit on the right of each register.
    """
    right_qubits = right.qubit_set()
    cuts = []
    for qubit in sorted(left.qubits(), key=_coord):
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


def _emit_balanced(rect: Rectangle, k: int, placed: tuple[cirq.GridQubit, ...], builder: _Builder) -> None:
    """Weight ``placed`` is already a unary register, ones on the right."""
    split = _split_longer(rect, k)
    if split is None:
        _emit_leaf(rect, k, placed, builder)
        return
    left, right = split
    s1, s2 = _divide_sites(left, right, k)
    if placed != s2:
        builder.swaps(_route(placed, s2, rect.qubit_set()))
    divide = build_logical_divide(rect.n, left.n, k, list(s1) + list(s2))
    builder.unitary(divide, s1 + s2)
    _emit_balanced(left, k, s1, builder)
    _emit_balanced(right, k, s2, builder)


def _emit_leaf(rect: Rectangle, k: int, placed: tuple[cirq.GridQubit, ...], builder: _Builder) -> None:
    if rect.n == k:
        path = placed
        if set(path) != rect.qubit_set() or len(path) != k:
            raise RuntimeError("leaf register does not cover the region")
    else:
        path = rect.snake()
        port = path[-k:]
        if placed != port:
            builder.swaps(_route(placed, port, rect.qubit_set()))
    assert_path(path)
    ops = scs_unitary_ops(list(path), min(k, rect.n))
    builder.unitary(cirq.Circuit(ops), path)


def _emit_peel(rect: Rectangle, k: int, builder: _Builder) -> None:
    """Figure 3(b). The unary register starts on the first strip's top row."""
    if rect.width == k:
        path = _path_ending_on_top_row(rect)
        builder.unitary(cirq.Circuit(scs_unitary_ops(list(path), min(k, rect.n))), path)
        return
    first = tuple(rect.rows[0][:k])
    second = tuple(rect.rows[0][k : 2 * k])
    builder.swaps(_route(first, second, set(rect.rows[0])))
    strip = rect.sub_cols(0, k)
    rest = rect.sub_cols(k, rect.width)
    divide = build_logical_divide(rect.n, strip.n, k, list(first) + list(second))
    builder.unitary(divide, first + second)
    path = _path_ending_on_top_row(strip)
    if path[-k:] != first:
        raise RuntimeError("strip path does not end on the divide register")
    builder.unitary(cirq.Circuit(scs_unitary_ops(list(path), min(k, strip.n))), path)
    _emit_peel(rest, k, builder)


def _path_ending_on_top_row(rect: Rectangle) -> tuple[cirq.GridQubit, ...]:
    """Hamiltonian path whose last paper row is the top row, left to right."""
    ordered = []
    for row_index in range(rect.height - 1, -1, -1):
        row = list(rect.rows[row_index])
        if row_index % 2 == 1:
            row.reverse()
        ordered.extend(row)
    path = tuple(ordered)
    assert_path(path)
    return path


def _route(
    sources: tuple[cirq.GridQubit, ...],
    targets: tuple[cirq.GridQubit, ...],
    allowed: set[cirq.GridQubit],
) -> tuple[tuple[cirq.GridQubit, cirq.GridQubit], ...]:
    """Adjacent SWAPs taking ``sources[i]`` to ``targets[i]``.

    Tokens are distinguishable. Every other allowed qubit is a blank.
    The search is over placements of the tokens, so ``k`` stays small.
    """
    sources = tuple(sources)
    targets = tuple(targets)
    if sources == targets:
        return ()
    if len(sources) != len(targets) or len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
        raise RuntimeError("token route needs equal distinct registers")
    allowed = set(allowed)
    if not set(sources) <= allowed or not set(targets) <= allowed:
        raise RuntimeError("token route left its rectangle")
    previous: dict[tuple[cirq.GridQubit, ...], tuple[cirq.GridQubit, ...] | None] = {sources: None}
    step: dict[tuple[cirq.GridQubit, ...], tuple[cirq.GridQubit, cirq.GridQubit]] = {}
    queue: deque[tuple[cirq.GridQubit, ...]] = deque([sources])
    found = sources == targets
    while queue and not found:
        state = queue.popleft()
        occupied = set(state)
        for index, pos in enumerate(state):
            for nxt in neighbors(pos, allowed):
                if nxt in occupied:
                    continue
                moved = list(state)
                moved[index] = nxt
                nxt_state = tuple(moved)
                if nxt_state in previous:
                    continue
                previous[nxt_state] = state
                step[nxt_state] = (pos, nxt)
                if nxt_state == targets:
                    found = True
                    queue.clear()
                    break
                queue.append(nxt_state)
        if len(previous) > 2_000_000:
            break
    if targets not in previous:
        raise RuntimeError(
            f"no grid route for {len(sources)} tokens inside {len(allowed)} qubits"
        )
    swaps = []
    state = targets
    while previous[state] is not None:
        swaps.append(step[state])
        state = previous[state]
    swaps.reverse()
    return tuple(swaps)


def _relative_phase_ok(amplitudes: dict[int, complex], weight: int) -> bool:
    """True when every weight-``weight`` amplitude is the same complex number."""
    reference = None
    for basis, amplitude in amplitudes.items():
        if basis.bit_count() != weight or abs(amplitude) <= 1e-10:
            continue
        if reference is None:
            reference = amplitude
        elif abs(amplitude - reference) > 1e-6:
            return False
    return reference is not None or weight == 0


def _events(trace: list):
    for event in trace:
        yield event


def _coord(qubit: cirq.GridQubit) -> tuple[int, int]:
    return (qubit.row, qubit.col)


class _Builder:
    def __init__(self) -> None:
        self.trace: list = []

    def swaps(self, pairs) -> None:
        if pairs:
            self.trace.append(("swaps", tuple(pairs)))

    def unitary(self, circuit: cirq.AbstractCircuit, block) -> None:
        block = tuple(block)
        assert_path(block)
        if len(circuit) == 0:
            return
        self.trace.append(("unitary", circuit, block))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Theorem 14 on the Willow 7 by 8 patch.")
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        report = verify_willow(args.k)
    except ValueError as exc:
        print(exc)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
