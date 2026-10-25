# Paper mapping

Source: Pei Yuan and Shengyu Zhang, arXiv:2505.15413, Section 3.2.

| Paper | Code |
| --- | --- |
| Eq. (7), `Divide_k^{n,m}` | `mathematical_primitives.weight_distribution_amplitudes`, `divide_lnn.compile_divide_lnn` |
| Lemma 7, `U_k^n` on a path | `leaf.compile_leaf` via the 2019 SCS stair |
| Lemma 8, Divide on a path of `2k` qubits | `divide_lnn.compile_divide_lnn` |
| Eq. (8), balanced recursion | `compiler._compile` |
| Figure 3(a), split the long side | `partition.split_longer` |
| Figure 3(b), peel strips of width `k` when `k < n2/n1` | `partition.peel_applies` |
| `P(S, S')` after Divide | `permutation.route_block_exchange` |
| Corollary 15 | The depth we report is the compiled moment count, not the big-O expression |

Followed exactly:

- The input of every region is a unary register, ones on the right, and the unitary handles every weight up to its capacity. Nothing is conditioned on a measured weight.
- Divide acts on two `k`-qubit registers and is only called when both children have at least `k` qubits.
- After Divide, one register stays in the left child and the other is placed on the right child's entry. The children then run on disjoint qubits in the same moments.
- Leaves are the 2019 path unitary, including the one-qubit identity.

Finite-size adaptations, labeled as such in `partition.py`, `placement.py`, and `divide_lnn.py`:

- The proof takes ceilings and then pretends the tile sizes are integers. For 7 by 8 the cut is the integer closest to half the qubits among cuts that keep both sides at least `k` qubits wide.
- The registers are Hamiltonian paths of length `k` joined across one cut edge, not `sqrt(k)` by `sqrt(k)` blocks. Lemma 8 only needs the path.
- The 2025 paper cites the 2022 weight-distribution block for the `O(k)` path depth and does not draw that circuit. The implementation is the block from Bärtschi and Eidenbenz, rewritten so that every stair uses one or two controls (enough on the unary subspace) and so that the realignment is one controlled rotation per unary 1. Equivalence with the reference block is tested on every weight, including a superposition of two weights.
- `P(S, S')` is realized by odd-even transposition sort on the snake, which is a sorting network on grid edges. Lemma 5 only states that some `O(n2)`-depth circuit exists.

Not claimed:

- A few finite benchmarks are not a proof of Corollary 15.
- Device validation is not a proof of the amplitudes.
- Hamming-weight conservation is not fidelity. The small-grid tests compare the full statevector to `|D_ell>`.
