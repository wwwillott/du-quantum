# Algorithm

## State

\[
|D_k^n\rangle
=
\binom{n}{k}^{-1/2}
\sum_{|x|=k}|x\rangle.
\]

The input that every unitary below expects is the unary string \(|0^{n-\ell}1^\ell\rangle\), ones on the right. Preparing \(|D_k^n\rangle\) from \(|0^n\rangle\) is an \(X\) on each of the last \(k\) qubits, followed by the unitary. Because \(X^{\otimes n}|D_{n-k}^n\rangle = |D_k^n\rangle\), the circuits use \(\min(k, n-k)\) excitations and finish with a layer of \(X\) when that is the smaller one.

## Amplitudes

Vandermonde's identity is the divide step (Bärtschi & Eidenbenz 2022, Definition 2; Yuan & Zhang 2025 use the same split):

\[
|D_\ell^n\rangle
=
\sum_i
\sqrt{
\frac{\binom{m}{i}\binom{n-m}{\ell-i}}{\binom{n}{\ell}}
}
|D_i^m\rangle\,|D_{\ell-i}^{n-m}\rangle,
\]

\[
\max(0,\ell-(n-m)) \le i \le \min(\ell, m).
\]

`split_weights` multiplies exact `math.comb` integers. `split_amplitudes` takes the square root in floating point for small totals and via log-gamma past \(10^{12}\). The integer weights sum to \(\binom{n}{\ell}\), so the amplitudes are normalized. Tests check both facts.

A unary stair writes those amplitudes with

\[
\theta_i = 2\arccos\sqrt{p_i},
\qquad
p_i = \frac{x_i}{\sum_{j\ge i} x_j},
\qquad
x_i = \binom{m}{i}\binom{n-m}{\ell-i}.
\]

`cirq.ry(theta)` satisfies \(R_y(\theta)|0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle\), so this \(\theta_i\) stops with probability \(p_i\) and continues with probability \(1-p_i\). That is the same angle convention as Bärtschi & Eidenbenz 2019, Section 2.2.

## 2019 stair (the hardware circuit)

\(U_{n,k}\) is the product in Lemma 2 of arXiv:1904.07358:

\[
U_{n,k}
=
\prod_{\ell=2}^{k}(\mathrm{SCS}_{\ell,\ell-1}\otimes I^{\otimes n-\ell})
\cdot
\prod_{\ell=k+1}^{n}(I^{\otimes \ell-k-1}\otimes \mathrm{SCS}_{\ell,k}\otimes I^{\otimes n-\ell}),
\]

applied right to left, so \(\mathrm{SCS}_{n,k}\) runs first. Each SCS is gate (i) and then gates \((ii)_\ell\) for \(\ell = 2,\ldots,k\):

- gate (i), on the last two wires: \(R_y(2\arccos\sqrt{1/n})\), conjugated by CNOTs
- gate \((ii)_\ell\), on wires \(n-\ell\), \(n-\ell+1\), and \(n\): \(R_y(2\arccos\sqrt{\ell/n})\) as a doubly controlled rotation, conjugated by one CNOT

`logical_dicke_circuit` emits that stair. Controls are not all nearest neighbors. `route_on_path` compiles it to CZ + PhasedXZ and inserts SWAPs with `cirq.RouteCQC` on the path graph only. Every SWAP edge is a path edge, and the path edges are couplers from the device or from `cz_pauli_error_rate`. The final permutation of a Dicke state is still a Dicke state, which is why the notebook's sorted-qubit check passes.

Adjacent single-qubit PhasedXZ runs are merged before the moment count. Without that merge the \(N=20\) stair sits a few dozen moments over the notebook's limit of 1750; with it, both \((20,4)\) and \((20,10)\) compile to 1740 moments.

## 2025 balanced recursion

`build_region` splits a region of size \(n\) and max weight \(w\) only when \(n > w\). The right child is

\[
r = \max\big(w,\lceil n/2\rceil\big),
\]

clipped so the left child is nonempty. The right child is at least \(w\) qubits wide, so the incoming unary ones stay inside it. The left hot register is the rightmost \(\min(m, w)\) qubits of the left child; the right hot register is the rightmost \(\min(n-m, w)\) qubits of the right child.

`weight_distribution_ops` then:

1. Turns the unary weight on A into a one-hot with a CNOT stair.
2. For each one-hot value \(\ell\), writes the weighted unary state on B with the controlled angles \(\theta_i\) above.
3. Restores A to unary.
4. Cancels B's ones against the right-hand ones of A, converts B to one-hot, and cycles A's surviving ones to the right under that one-hot control (the controlled permutation in Figure 4 of arXiv:2207.09998).

Both children are then prepared by the same procedure. Their circuits are zipped into the same moments wherever the qubits are disjoint. A region with size at most \(w\) is finished by the 2019 unitary \(U_{s,s}\), which spreads every remaining unary weight.

On an abstract line, before any Willow routing, the balanced circuit is the shallower one for every \((n,k)\) in `benchmarks/logical_metrics.csv` except the smallest W states, where the two match. Those balanced CZ counts still include long-range gates. Routing them onto Willow is future measurement, not a number already claimed as hardware depth.

## Routing

`route_permutation` returns layers of SWAPs. Each layer is a matching. On a path the layers are odd-even transposition sort. On a general graph a misplaced token steps along a shortest path, and a SWAP is kept only when the sum of remaining distances does not increase. `apply_swap_layers` updates `logical_to_physical` and `physical_to_logical` and rejects a layer that touches a vertex twice.

## Depth

On a path the 2019 construction is \(O(kn)\) gates and, after the stair is pushed together, \(O(n)\) depth in units of SCS blocks (Theorem 1). Each block decomposes to a constant number of CZs, and the router adds SWAPs proportional to how far a control sits from the bottom wire. The compiled moment counts in the challenge path are the sum of those two, not the asymptotic expression alone.

The 2025 grid bound is \(O(k\log(n/k) + n_2)\) when the physical patch is an \(n_1\times n_2\) grid with \(n_1\le n_2\) and \(k \ge n_2/n_1\), and \(O(n_2)\) when \(k\) is smaller (Corollary 15). Reaching that bound on Willow requires the hot-register moves to stay inside the child rectangles. The recursion tree records the split \(m\), the hot registers, and the depth so those moves can be routed per region instead of across the whole chip.
