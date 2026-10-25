# Verification

Two obligations, kept separate.

Logical state. For every rectangle with `n <= 22`, the compiler is simulated with Cirq's statevector on every weight `ell <= k` and compared with `|D_ell^n>`. The overlap ignores a global phase. The same check is repeated after CZ lowering, so the gate decomposition is not trusted just because the logical circuit was right. A Divide equivalence test also feeds a superposition of two unary inputs, which fails if the relative phases between weights are wrong.

Compiled gates. Every final two-qubit gate is a full CZ on a grid edge. On the Willow rectangle the circuit is additionally passed to `Willow105.validate_circuit`. That check does not know about Dicke states.

What is not claimed for `n=56`. A dense statevector is refused above 22 qubits (`simulator.assert_simulable` estimates the footprint first). The 56-qubit certificate is the composition of three checked pieces: Divide matches the reference block on its `2k` qubits, each permutation certificate matches its target map, and every leaf is the 2019 unitary that the small-grid tests exercise directly. Weight conservation alone is not used as a fidelity.
