# Research results

Evidence: analytical bounds are **Proven** in theory.md; circuit fidelities are **Verified numerically** against independent full-space Cirq simulation. Performance comparisons are **Observed** for the exact implementations and synthetic noise below, not hardware measurements or novelty claims.

All final counts apply the same state-aware pruning pass to candidates and baselines, then the repo CZ compiler (no dynamical decoupling), topology checks, and Willow105.validate_circuit. State-aware pruning is valid for this input only. Exact means numerical fidelity within 1e-9, not an analytic synthesis proof.

## Final consistent comparison

Noise: p1=0.0001, p2=0.001 depolarizing-channel probabilities; synthetic T1=100 μs, T2=80 μs; illustrative single/CZ duration 25/40 ns. Readout flips are separate classical probabilities. Durations are NOT calibrated Willow predictions. Same noise and compilation pass for every row.

| Circuit | n,l | Ideal F | Synthetic noisy F | CZ gates | Moment depth | Illustrative ns | Optimization s |
|---|---|---:|---:|---:|---:|---:|---:|
| chain-4-2-depth4-cp0 | 4,2 | 0.9714045208 | 0.94991964 | 8 | 13 | 415 | 0.0789 |
| chain-4-2-depth6-cp1 | 4,2 | 1.0000000000 | 0.95976392 | 15 | 23 | 740 | 0.0717 |
| SCS-4-2 | 4,2 | 1.0000000000 | 0.91751558 | 28 | 57 | 1845 | 0.0000 |
| recursive-4-2 | 4,2 | 1.0000000000 | 0.92455557 | 32 | 58 | 1885 | 0.0000 |
| star-depth6 | 4,2 | 1.0000000000 | 0.96854069 | 10 | 21 | 675 | 0.0220 |
| chain-6-3-depth6-cp0 | 6,3 | 0.9000000000 | 0.84131701 | 26 | 25 | 805 | 0.3059 |
| chain-6-3-depth6-cp2 | 6,3 | 0.9774687898 | 0.90596755 | 28 | 29 | 935 | 1.0910 |
| chain-6-3-depth10-cp1 | 6,3 | 1.0000000000 | 0.88684004 | 47 | 43 | 1390 | 0.3342 |
| SCS-6-3 | 6,3 | 1.0000000000 | 0.64032476 | 109 | 216 | 7020 | 0.0000 |
| recursive-6-3 | 6,3 | 1.0000000000 | 0.33331359 | 330 | 595 | 19390 | 0.0000 |

The star uses a different connected Willow subset; it is a geometry result, not a same-path comparison. All chain/SCS/recursive rows of a given size use exactly the same physical qubits and allowed path couplers. Recursive counts include the repository's explicit move-and-restore routing, not an optimal recursive compiler. The SCS/recursive implementations are verified here, but these results do not establish an advantage over the best possible literature circuits.

## Scientific decisions

- **Proven:** unrestricted open-chain XY+Z cannot exactly prepare any intermediate Dicke state (2≤l≤n−2). For D₂⁴ its exact maximum fidelity is 0.9714045207910317, independently checked with SVD, an explicit maximizing Slater, and optimized circuits.
- **Observed:** one CP did not exceed the ceiling in the four-layer schedule but succeeds in the six-layer schedule. Optimizer failure at four layers is not a proof of that schedule's best possible fidelity.
- **Observed:** D₃⁶ free XY attained 0.9 in both tested depths; this is an optimizer observation, NOT a certified global fidelity bound. One CP at ten layers reaches unit fidelity numerically. At six layers two CP gates improve ideal fidelity to 0.9774687898.
- **Observed:** under the stated synthetic noise, the shorter approximate six-layer/two-CP D₃⁶ circuit beats the ten-layer/one-CP numerically exact circuit. See the table; this is a finite Pareto comparison, not direct noise-aware continuous optimization.
- **Observed:** star XY+Z alone reaches D₂⁴; the chain obstruction is not a general Willow graph obstruction. A square remains at the ceiling in tested schedules, consistent with the fixed-parity ring restriction.
- Kernel timing: direct affected-pair gates are fastest in the measured isolated-XY task. Seven repetitions per method, median wall time, independently checked against expm and expm_multiply. Dense exponentials were capped at dimension 70; no full 105-qubit simulation or large optimizer sweep was run. This is not a general multi-term Hamiltonian benchmark.

## Missing capabilities and next experiment

Arbitrary FSim parameters were used as logical gates only. An independently calibrated fixed-native-FSim comparison is unavailable: this installed Willow105 record accepts full CZ for the comparison, and no processor-specific adjustable-pulse calibration was supplied. Hybrid recursion is unimplemented: a replacement block must be certified simultaneously on every child-weight sector and preserve branch phases. Symmetry-constrained control, direct noisy-objective optimization, crosstalk, leakage, live calibration, and 8+ qubit optimized preparation remain untested. Only small kernel benchmarks reach n=12.

The coherent multi-sector leaf experiment was subsequently completed; see below. Full recursive integration remains unimplemented. Next cheapest useful experiment: insert the saved leaf into one existing recursion node whose child registers can be mapped to a physical star; verify every populated branch and compare total routed CZ counts. A local leaf improvement is not yet a full hybrid advantage.

## Acceptance and coherent fidelity

| Circuit | Noisy F | P(valid sector) | F after ideal sector projection | P(valid after classical readout flips) |
|---|---:|---:|---:|---:|
| chain-4-2-depth4-cp0 | 0.94991964 | 0.98384687 | 0.96551574 | 0.94591071 |
| chain-4-2-depth6-cp1 | 0.95976392 | 0.97112031 | 0.98830588 | 0.93401070 |
| SCS-4-2 | 0.91751558 | 0.94422526 | 0.97171259 | 0.90887456 |
| recursive-4-2 | 0.92455557 | 0.95539192 | 0.96772388 | 0.91914973 |
| star-depth6 | 0.96854069 | 0.97647156 | 0.99187804 | 0.93903798 |
| chain-6-3-depth6-cp0 | 0.84131701 | 0.95036895 | 0.88525304 | 0.89729954 |
| chain-6-3-depth6-cp2 | 0.90596755 | 0.94659496 | 0.95708047 | 0.89387001 |
| chain-6-3-depth10-cp1 | 0.88684004 | 0.91557239 | 0.96861816 | 0.86566630 |
| SCS-6-3 | 0.64032476 | 0.75864474 | 0.84403769 | 0.72227702 |
| recursive-6-3 | 0.33331359 | 0.56344058 | 0.59156830 | 0.54311913 |

F_post assumes an ideal projective weight-sector test before readout. No postselection is inserted into the executable preparation circuit or proposed for the challenge interface. A classical noisy weight filter has different acceptance and does not implement that ideal projection. Noise trajectories use the full small-system density matrix because damping and depolarization leave the fixed-weight sector.

## Reproduction and artifact manifest

```powershell
python -m pytest interference_dicke/tests -q
python -m interference_dicke.experiment --noise
python -m interference_dicke.experiment --extended
python -m interference_dicke.geometry
python -m interference_dicke.noise_followup
python -m interference_dicke.prune
python -m interference_dicke.report
```

Run from the du-quantum repository root. Every run has a unique directory; prior artifacts are preserved. JSON stores schedules, angles, seeds, limits, trials/evaluations, timestamps, topology coordinates, versions, commit and measured timings. Source snapshots capture uncommitted implementations; saved Cirq JSON circuits are directly reloadable. The generated report selects the earliest small/extended milestone and latest geometry/pruned results. Wall-clock times vary across machines.

- [20261025T075710653153Z](artifacts/20261025T075710653153Z/results.json)
- [20261025T075915050796Z](artifacts/20261025T075915050796Z/results.json)
- [geometry-20261025T080010278842Z](artifacts/geometry-20261025T080010278842Z/results.json)
- [pruned-20261025T080258195319Z](artifacts/pruned-20261025T080258195319Z/results.json)

## Coherent multi-sector leaf feasibility

Unlike the state-specific pruned circuits above, these unitary blocks contain no initialization or state-aware pruning. One fixed schedule is checked on all five orthogonal unary inputs on the SAME physical star. The objective penalizes relative phases against the vacuum-sector phase, not just separate-sector fidelities. Analytic gradients were checked with finite differences; Cirq independently checks every compiled sector.

| Star block | CZ gates | Moment depth | Min sector fidelity | Max absolute relative phase (rad) | Common-phase objective loss |
|---|---:|---:|---:|---:|---:|
| XY+Z, 0 CP | 24 | 49 | 0.933012439851 | 4.78e-07 | 0.017 |
| XY+Z, 1 CP | 26 | 53 | 0.999999999997 | 7.51e-07 | 1.07e-12 |
| Existing SCS unary unitary, routed on same star | 250 | 500 | 1.000000000000 | 6.66e-15 | 2.22e-16 |

**Observed:** the one-CP star block meets the coherent local-leaf test with 26 CZ gates versus 250 for the repository SCS unitary with move-and-restore star routing. The zero-CP bounded optimization leaves sectors 1 and 3 at about 0.933 fidelity while sector 2 reaches unity: this is a concrete warning against single-sector substitution. It is not a proof that zero-CP coherent preparation is impossible at greater depth or under a different schedule. No global hybrid circuit, scaling law, or optimal routing claim follows from this leaf comparison.

Data and circuit files: [multisector-20261025T080630810392Z](artifacts/multisector-20261025T080630810392Z/results.json). Reproduce with `python -m interference_dicke.multisector`; two 12-second family budgets, four starts each. The source snapshot and JSON retain every angle, schedule, trial, common-phase metric, topology and environment.

## Figures

- [01_ideal_vs_depth](figures/01_ideal_vs_depth.png)
- [02_noisy_vs_depth](figures/02_noisy_vs_depth.png)
- [03_fidelity_vs_interactions](figures/03_fidelity_vs_interactions.png)
- [04_fidelity_vs_CZ](figures/04_fidelity_vs_CZ.png)
- [05_optimization_time](figures/05_optimization_time.png)
- [06_geometry](figures/06_geometry.png)
- [07_amplitudes_probabilities](figures/07_amplitudes_probabilities.png)
- [08_relative_phases](figures/08_relative_phases.png)
- [09_configuration_current](figures/09_configuration_current.png)
- [10_pareto_frontier](figures/10_pareto_frontier.png)

Theory and verified bibliography: [theory.md](theory.md). Current hypotheses: [hypothesis_tree.json](hypothesis_tree.json). Kernel timings, raw unpruned circuits, sensitivity across p₂∈{0,0.001,0.005}, and all optimizer trials remain in their original result files.
