# Routing

After Divide, the left register already lies in the left child and the right register lies in the right child. Each is sent to that child's entry port by a permutation of that child alone. The two permutations are disjoint and share moments.

The entry port is the right half of the child's own Divide path, or the right end of the leaf's Hamiltonian path. Ones stay on the right because the block map preserves order: source qubit `i` goes to destination qubit `i`.

Only the states that the register displaces move: each destination site that is not already a source is paired with a vacated source site, in coordinate order, and every other qubit stays where it is. The route does not look at the excitation count. Odd-even transposition sort along the snake implements that permutation. Each layer is a matching of snake edges, which are grid edges, and on the Willow patch those edges are device couplers. The certificate records the target map, the layers, and whether the final placement matches. A coordinate-order pairing of every idle site was measured and rejected: a one-qubit peel shift then dragged the rest of the rectangle and cost 289 moments inside the `k=1` root stage. The displaced-only map brings that root permutation to 19 moments.

The peel case adds one earlier permutation. Figure 3(b) leaves the surviving unary register on the strip that the next Divide would treat as its left register, while Eq. (7) wants the unary input on the right register. The compiler exchanges the top-row blocks before the next Divide. That exchange is the same weight-independent permutation, not a special case of the data.
