"""The two unary anchors sitting on a staging path."""

from __future__ import annotations

import cirq


def split_anchors(
    staging_path: tuple[cirq.GridQubit, ...], k: int
) -> tuple[tuple[cirq.GridQubit, ...], tuple[cirq.GridQubit, ...]]:
    """Left anchor is the A half. Right anchor is the B half.

    Before Divide, the incoming weight-k register occupies the left anchor.
    After Divide, the two output unary registers occupy these two anchors.
    """
    if len(staging_path) != 2 * k:
        raise ValueError(f"staging path length {len(staging_path)} is not 2k={2 * k}")
    return staging_path[:k], staging_path[k:]
