# Divide

`compile_divide_lnn(n, m, k, path)` builds one unitary on a path of `2k` qubits.

Register order matches `build_logical_divide`: `path[:k]` is the size-`m` side and starts at `|0^k>`, `path[k:]` holds the unary input. Ones sit on the right of each register. The same circuit is used for every weight `ell <= k`.

Steps, all nearest-neighbor:

1. Unary to one-hot on the right register by adjacent CNOTs.
2. For each one-hot value, bubble that qubit to the boundary, write the weighted unary state on the left register, bubble it back. The first rotation is a controlled-Ry. Each later rotation is controlled by the original one-hot bit and the previous target only. On this input subspace the previous target is `|1>` only when every earlier target is `|1>`, so the extra controls in the reference stair do not fire. Other weight branches stay idle because their one-hot bit is `|0>`.
3. One-hot back to unary.
4. CNOT the right-aligned ones, then cycle the right register one step for each unary `|1>` on the left. That is the same permutation as the one-hot controlled cycle in the 2022 figure, with `O(k^2)` Fredkins rather than `O(k^3)`.

A 3-qubit gate is gathered onto three consecutive vertices, decomposed, and gathered back. Elementary gates inside that decomposition are not bridged one by one across the path.

Backend A, used only for the equivalence test and the reference column, is `build_logical_divide` lowered by `bridge_along_path`.

Measured CZ moments after the Willow gate-set pass, balanced split `n=4k`, `m=2k`:

| k | reference moments | reference CZ | LNN moments | LNN CZ |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5 | 2 | 5 | 2 |
| 2 | 85 | 42 | 165 | 82 |
| 3 | 805 | 402 | 567 | 307 |
| 4 | 4051 | 2025 | 1265 | 756 |

At `k=2` the reference block is already nearly local (6 bridging SWAPs), and the generic CCRy and Fredkin decompositions cost more than they save. From `k=3` up, the reference's `k`-controlled rotations, each walked out and back, dominate, and the path schedule is smaller. The `O(k)` claim is the paper's asymptotic statement. These rows are the finite constants on this gate set. `k=5` is produced by `benchmark.divide_benchmark` and stored as a new JSON file rather than typed here in advance.
