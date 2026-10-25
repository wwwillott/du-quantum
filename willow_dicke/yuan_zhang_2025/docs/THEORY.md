# Theory

The target family is the Dicke unitary `U_k^n`, not only the single state `|D_k^n>`. On the unary input `|0^{n-ell} 1^{ell}>` it produces `|D_ell^n>` for every `ell <= k`. Preparing `|D_k^n>` from `|0^n>` is an `X` on each of the last `k` qubits of the entry port, then that unitary.

A split `n = m + (n-m)` uses

```
alpha_i = sqrt( C(m, i) C(n-m, ell-i) / C(n, ell) )
```

with `i` ranging from `max(0, ell-(n-m))` to `min(ell, m)`. The integer form of this is Vandermonde's identity. The circuits use the 2019 angle `theta = 2 arccos sqrt(p)`, and Cirq's `ry(theta)` satisfies `ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>`.

The grid circuit is the one in Section 3.2 of arXiv:2505.15413:

```
balanced Divide -> structured block permutation -> parallel child recursion
```

When `k < n2/n1` and `k` divides `n2`, the same section peels strips instead of cutting in half. On the 7 by 8 patch that is only `k = 1`. Every larger benchmark weight uses the balanced cut.

Corollary 15 bounds the depth by `O(k log(n/k) + n2)` or `O(n2)` according to the same threshold. Those expressions count abstract layers under an `O(k)`-depth Divide. The numbers this package reports are Cirq moments of PhasedXZ and CZ. They are allowed to show a different finite-size slope. A matching slope on a handful of `k` values would still not be a proof of the corollary.
