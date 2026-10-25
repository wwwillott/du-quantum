# Phase 0 audit

The 105-qubit compiler stays where it is. This package does not import it in order to modify it. What was reused, and what was left alone:

Reusable and used as-is:

- `willow_dicke.math.coefficients` and `dicke_willow.dicke_math.rotation_angle` for the split amplitudes and the Ry convention.
- `willow_dicke.divide.logical_divide.build_logical_divide` as Divide backend A, the reference.
- `dicke_willow.scs.scs_unitary_ops` as the Lemma 7 leaf. It is already nearest-neighbor on a path.
- `willow_dicke.circuit.lower.lower_to_willow_cz` for the final PhasedXZ + CZ pass.
- `willow_dicke.device.load_static_willow` and `cirq_google.Willow105` for the coupling graph. No edge is typed in by hand.
- `willow_dicke.rectangle.Rectangle` for paper-coordinate subrectangles and the snake.

Not reused as the default Divide:

- `bridge_along_path` walks each nonlocal elementary gate to a neighbor and walks it back. On the reference weight-distribution block that costs 402 CZ / 805 moments at `k=3` and 2025 CZ / 4051 moments at `k=4` (balanced `n=4k`, `m=2k`, before this package). That schedule is kept only as the reference column of the Divide benchmark.

The 2025 text's own grid circuit (`willow_dicke.yuan_zhang.py`) already splits rectangles the way Theorem 14 describes, but it still lowers Divide through that bridge, and it appends the left child before the right child. Both of those choices are replaced here.
