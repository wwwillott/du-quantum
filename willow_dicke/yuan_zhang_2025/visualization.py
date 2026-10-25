"""Static pictures of the rectangle, the registers, and one SWAP schedule.

Colors are stable: the left unary register is blue, the right register is
orange, the cut is black, and idle qubits are gray. An illustrative basis
label is a single classical weight, never a coherent superposition.
"""

from __future__ import annotations

import html

import cirq

from willow_dicke.yuan_zhang_2025.compiler import CompiledRegion
from willow_dicke.yuan_zhang_2025.geometry import GridPatch


def _coord(qubit: cirq.GridQubit) -> tuple[int, int]:
    return (qubit.row, qubit.col)


def patch_svg(patch: GridPatch, compiled: CompiledRegion | None = None) -> str:
    """Paper-coordinate drawing. Physical Willow labels are printed under each site."""
    cell = 46
    pad = 28
    width = patch.width * cell + 2 * pad
    height = patch.height * cell + 2 * pad + 18
    roles: dict[cirq.GridQubit, str] = {}
    if compiled is not None and compiled.tree.divide_path:
        for qubit in compiled.tree.left_register:
            roles[qubit] = "#3b6ea5"
        for qubit in compiled.tree.right_register:
            roles[qubit] = "#d17b2a"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f4ee"/>',
    ]
    for row in patch.paper_rows:
        for qubit in row:
            for other in patch.adjacency[qubit]:
                if (qubit.row, qubit.col) < (other.row, other.col):
                    x1, y1 = _paper_xy(patch, qubit, cell, pad)
                    x2, y2 = _paper_xy(patch, other, cell, pad)
                    parts.append(
                        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#b7b1a6" stroke-width="2"/>'
                    )
    for row_index, row in enumerate(patch.paper_rows):
        for col_index, qubit in enumerate(row):
            x, y = pad + col_index * cell + cell / 2, pad + row_index * cell + cell / 2
            color = roles.get(qubit, "#e4e0d8")
            parts.append(f'<circle cx="{x}" cy="{y}" r="14" fill="{color}" stroke="#333" stroke-width="1"/>')
            parts.append(
                f'<text x="{x}" y="{y + 22}" text-anchor="middle" font-size="8" '
                f'font-family="sans-serif" fill="#444">{qubit.row},{qubit.col}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _paper_xy(patch: GridPatch, qubit: cirq.GridQubit, cell: int, pad: int) -> tuple[float, float]:
    for row_index, row in enumerate(patch.paper_rows):
        for col_index, site in enumerate(row):
            if site == qubit:
                return pad + col_index * cell + cell / 2, pad + row_index * cell + cell / 2
    raise KeyError(qubit)


def permutation_frames(compiled: CompiledRegion) -> str:
    """HTML frame sequence for the root's outgoing block permutations.

    Each frame is one SWAP layer. Qubits show the logical token they hold,
    starting from the identity placement. This is one illustrative routing,
    not a measured excitation pattern.
    """
    frames = []
    for certificate in compiled.certificates:
        if not certificate.region_id.startswith("root/"):
            continue
        tokens = {qubit: qubit for qubit in certificate.initial_mapping}
        frames.append(_frame(certificate.region_id, 0, tokens))
        for number, layer in enumerate(certificate.swap_layers, start=1):
            for left, right in layer:
                tokens[left], tokens[right] = tokens[right], tokens[left]
            frames.append(_frame(certificate.region_id, number, dict(tokens)))
    body = "\n".join(frames) or "<p>No root permutations.</p>"
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>SWAP frames</title>"
        "<style>body{font-family:sans-serif;background:#f7f4ee} .frame{margin:12px;padding:8px;"
        "background:white;display:inline-block;vertical-align:top}</style></head><body>"
        "<h1>Block-permutation frames</h1>"
        "<p>Each cell is a physical qubit. The text is the logical token currently there.</p>"
        f"{body}</body></html>"
    )


def _frame(region_id: str, number: int, tokens: dict) -> str:
    rows: dict[int, list] = {}
    for qubit, token in tokens.items():
        rows.setdefault(qubit.row, []).append((qubit.col, token))
    lines = []
    for row in sorted(rows):
        cells = []
        for col, token in sorted(rows[row]):
            cells.append(f"{token.row},{token.col}")
        lines.append(" ".join(cells))
    text = html.escape("\n".join(lines))
    return (
        f"<div class='frame'><h2>{html.escape(region_id)} layer {number}</h2>"
        f"<pre>{text}</pre></div>"
    )
