# Dicke states on Willow

Deterministic, ancilla-free preparation of

\[
|D_k^n\rangle
=
\frac{1}{\sqrt{\binom nk}}
\sum_{|x|=k}|x\rangle.
\]

The code is split the way the circuit is built:

| Module | Role |
| --- | --- |
| `dicke_willow/dicke_math.py` | Exact binomial weights and log-gamma amplitudes |
| `dicke_willow/weight_distribution.py` | One divide / weight-distribution block |
| `dicke_willow/divide.py` | Balanced recursion tree (Yuan & Zhang 2025) |
| `dicke_willow/scs.py` | 2019 Split & Cyclic Shift stair, logical and nearest-neighbor |
| `dicke_willow/topology.py` | Coupling graph from a Cirq device or from the challenge error maps |
| `dicke_willow/placement.py` | Noise-weighted path on that graph |
| `dicke_willow/routing.py` | Permutation routing as layers of disjoint SWAPs |
| `dicke_willow/compile.py` | PhasedXZ + CZ, dynamical decoupling, 1750-moment cap |
| `dicke_willow/challenge.py` | `create_dicke_state` and `create_dicke_state_adaptive` |
| `dicke_willow/verify.py` | Fidelity, Hamming weight, marginals |

`DuQuantum_Google_Standard.ipynb` imports those two functions. Its checker compares the statevector, in sorted-qubit order, with the little-endian Dicke vector. A Dicke state is the same in every qubit order, and the circuits here match that check to simulator precision.

## What the hardware question is allowed to run

- `cirq.PhasedXZGate` on qubits present in `sq_pauli_error_rate`
- `CZ` on pairs present in `cz_pauli_error_rate`
- no classical conditions, no postselection
- at most 1750 moments
- construction under a minute
- perfect state, or `NotImplementedError`

Qubits missing from the single-qubit or readout maps are treated as disabled. Couplers missing from the CZ map are not used. Problem 2 additionally allows mid-circuit measurement and feedforward; its verifier checks native operations, classical causality and complete readout diagonals.

## Which circuit is submitted

The hardware function places the 2019 SCS stair on a path and routes it with SWAPs along that path. On the measured error maps this stays exact through \(N=20\) and under 1750 moments (the \(N=20\) cases compile to 1740 moments after single-qubit merging). Dynamical decoupling is inserted when it does not cross that cap. Hardware submission is not automatic: `python main.py --hardware` prints the processor summary and stops.

Problem 2 now uses the independently verified depth portfolio in
[`interference_dicke/problem2`](interference_dicke/problem2/README.md), including
full-diagonal readout and exact feedforward where they reduce depth. The
standalone entry point is [`problem2_submission.py`](problem2_submission.py).
Reproduce the notebook's 100-instance depth score with
`python -m interference_dicke.problem2.evaluate`. Fully compiled interference
and Yu-paper experiments are recorded separately in that directory.

The 2025 balanced divide is implemented and checked exactly for \(D_1^4, D_2^4, D_2^6, D_3^6\). On an abstract line it is often the shallower of the two; see `benchmarks/logical_metrics.csv`. It is not the default hardware circuit, because its hot registers still have to be moved onto Willow couplers and that move is the quantity this project is measuring.

## Run

```bash
python main.py --n 16 --k 3 --device willow --simulate
python main.py --n 6 --k 3 --device line --method balanced --simulate
python main.py --n 8 --k 2 --benchmark
python -m pytest dicke_willow/tests/test_dicke.py
```

`--qvm` targets the bundled `willow_pink` virtual processor. `--hardware` never calls the Quantum Engine run API; `--submit` is refused on purpose.

## Interference research

The additive [`interference_dicke`](interference_dicke/README.md) experiments
study excitation-preserving exchange/phase control, with fixed-weight
simulation, analytic reachability bounds, compiled Willow comparisons,
synthetic noise and coherent multi-sector leaf checks. See
[`RESULTS.md`](interference_dicke/RESULTS.md) for measured results and limits.
Run `python -m pytest interference_dicke/tests -q` for independent checks.

## References

- Bärtschi & Eidenbenz, 2019. Deterministic Preparation of Dicke States. arXiv:1904.07358
- Bärtschi & Eidenbenz, 2022. Short-Depth Circuits for Dicke State Preparation. arXiv:2207.09998
- Yuan & Zhang, 2025. Depth-Efficient Quantum Circuit Synthesis for Deterministic Dicke State Preparation. arXiv:2505.15413
