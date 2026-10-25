# Independent validation: Kai, Will, and mid-circuit prototypes

Pinned sources: Will `605b9a86c931e66710ed31c4e1e5aa9ab08e78d7` (`will`); Kai `48115b9fbd9dd3d587fd8cfa01f0e58e00617638` (`kai/root-pipeline-v2`). Kai's unchanged portfolio also corresponds to main commit `913fcdf`. The current research branch pulled main; newer branch tips were fetched and archived into isolated local snapshots, not merged over one another. No remote branch was modified.

## Challenge 2: correct scoring and validation

Read the official notebook bonus cells 13–17. This is a **depth** competition, separate from the noisy Z/X hardware score. The displayed `score_solution` adds `N*k/len(circuit)`; every gate and measurement moment in every conditional branch counts in the static circuit depth. It does not score average execution time. The text says N≤15, whereas code loops N≤20. Kai's pushed handoff records an organizer clarification of N≤15; this review uses that range as primary and also records the literal code-range depths.

| Public source / experimental variant | Validated score, N=2…15 | Coverage |
|---|---:|---|
| kai-portfolio-913fcdf | 41.1036323434 | All 56 cases, all reachable branches verified |
| kai-v2-48115b9 | 41.1036323434 | All 56 cases, all reachable branches verified |
| will-605b9a8 | 41.3714032198 | All 56 cases, all reachable branches verified |
| Yu one-round prototype substituted into 3 cases | 39.0569045748 | 3 prototypes verified, remainder Will |
| Yu two-round prototype substituted into 3 cases | 39.0277216679 | 3 prototypes verified, remainder Will |
| Best validated portfolio | 37.1403219841 | Keeps Will; no prototype promotion |

Every accepted circuit passes native gate/edge checks, output cardinality, <60 s construction and a **full-device readout diagonal check** (including unused qubits). Each measurement is projected explicitly; classical corrections are resolved using Cirq's classical-data model. Every nonzero branch must yield the target up to phase, after tracing active ancillas by a statevector contraction. No postselection or sampled branch coverage is used. Numerical branches below 1e-15 joint probability are counted as discarded roundoff; total discarded mass is checked below 1e-10. Unitariy cases use an independent complex128 Cirq simulator. Full branch correctness of the auditor itself is tested with an intentionally missing correction.

Will's increase comes from D(7,3): 27→23 moments, and D(8,4): 23→21. D(7,3) has two measurement branches, both verified exact, with the entire readout diagonal measured. Kai v2's changes are in the hardware API and do not improve its public bonus API.

For the literal N≤20 loop, raw depth-only sums are:
- kai-v2-48115b9: **73.0787974470**. All returned circuits checked structurally; exact-state verification was capped at 16 active qubits. N=17…20 are **unverified**, so this is not a fully validated score.
- will-605b9a8: **73.4119278005**. All returned circuits checked structurally; exact-state verification was capped at 16 active qubits. N=17…20 are **unverified**, so this is not a fully validated score.

## Review findings

1. **Walkthrough benchmark claim is not reproduced by the committed CSV.** `WALKTHROUGH.md` claims runtime_anytime 9.645 vs shallow_incumbent 9.527. The actual ten rows per variant in `testing/vqm_shallow_vs_runtime.csv` sum to runtime_anytime **14.0802922131** vs shallow_incumbent **14.2035920630**. The claimed D(5,2), seed 7 score 2.437 is also absent (CSV has 1.3693454353 vs incumbent 1.2840875974). The writeup may describe another run, but that evidence is not in the pushed file. This does not invalidate the exact circuits.

2. **The new Z-dispersion surrogate has an extra factor of valid-sector retention.** For S shots, B valid shots, f=B/S and conditional shell probabilities q_i over M strings, the exact multinomial expectation is `E[Z | B] = (1-Σq_i²)/f + S Σ(q_i-1/M)²`. Thus the leading nonuniformity penalty is proportional to S/M times relative variance, not S*f/M. The first-order fixed-weight readout-log variance can approximate that variance, but its coefficient in `dicke_submission.py:1445` and `src/dicke_compiler/scoring.py:148` contains the extra f. An independent 20,000-trial count experiment checks the expectation at f=1,0.5,0.25. Treat this model as a ranking heuristic; the walkthrough's claim of an exact prefactor is too strong.

3. **The X surrogate penalizes correlators whose target expectation is zero.** For D(6,1), target X correlations at sizes 4 and 6 are zero. Uniform attenuation leaves them zero, and the official scorer adds zero population error for them. The new `for m in range(1,n//2+1)` relative-error sum still adds positive penalties. With attenuation η=0.9 per site, the correct population X penalty is **0.0361**; the surrogate sum is **0.3739147465**. Finite-shot variance remains a separate term. This can distort W-state ranking. Both scoring copies need the same correction before claiming judge-denominator accuracy.

4. **The older Bell measurement experiment's diagonal check is incomplete.** `testing/problem2_adaptive_experiments.py` reports `readout_line_valid=True` when helper and outputs use different diagonals, but it measures only one of seven device qubits on diagonal 8. Independent validation catches six missing qubits. The Bell state is correct on both branches; the prototype violates the challenge readout rule and is already deeper than the incumbent. It is not the public D(7,3) circuit, which passes.

5. **Kai v2 emits some CZs in the reverse order of the supplied dictionary key.** In the seed-101 D(10,3) test the physical edge is available and CZ is symmetric, so the state and undirected connectivity are correct. A checker requiring literal tuple-key membership could reject it. Normalize CZ endpoint order to an available key for compatibility; this costs no gates or depth. This is a compatibility warning, not evidence of an invalid physical coupling. `_v2_allowed` also accepts any CZPowGate rather than insisting on exponent 1; the measured smoke-test circuits all contain full CZs.

The first three findings concern **problem 1** ranking/documentation. They do not change the independently measured problem-2 depth scores above. No teammate source was edited to conceal or repair these findings.

## Hardware-API smoke checks

Five synthetic-calibration cases per file: (2,1), (4,2), (6,3), (8,4), (10,3), seed 101, using Kai's pushed calibration generator. This is a bounded smoke check, not a full noisy head-to-head or a hardware score. Exact fidelity, native gates, supplied qubits/readout maps, undirected couplers, runtime, and 1750-moment limit are independently checked.

| API source | Case | CZ | Depth | Exact fidelity | Build s | Literal-key warnings |
|---|---|---:|---:|---:|---:|---:|
| main-kai-v2-48115b9 | 2,1 | 1 | 3 | 1.000000000000 | 0.083 | 0 |
| main-kai-v2-48115b9 | 4,2 | 5 | 7 | 1.000000000000 | 0.409 | 0 |
| main-kai-v2-48115b9 | 6,3 | 14 | 18 | 1.000000000000 | 1.484 | 0 |
| main-kai-v2-48115b9 | 8,4 | 24 | 23 | 1.000000000000 | 12.639 | 0 |
| main-kai-v2-48115b9 | 10,3 | 45 | 45 | 1.000000000000 | 0.548 | 18 |
| main-will-605b9a8 | 2,1 | 1 | 3 | 1.000000000000 | 0.059 | 0 |
| main-will-605b9a8 | 4,2 | 5 | 7 | 1.000000000000 | 0.361 | 0 |
| main-will-605b9a8 | 6,3 | 14 | 18 | 1.000000000000 | 1.367 | 0 |
| main-will-605b9a8 | 8,4 | 24 | 23 | 1.000000000000 | 12.789 | 0 |
| main-will-605b9a8 | 10,3 | 57 | 51 | 1.000000000000 | 0.818 | 0 |

## Yu-paper prototype

[Yu, Muleady, Wang, Schine, Gorshkov & Childs, Efficient preparation of Dicke states, arXiv:2411.03428v1](https://arxiv.org/html/2411.03428v1) uses global rotations and collective magnetization measurements. A collective measurement reveals only Hamming weight, preserving coherence inside each weight sector; measuring every data qubit individually is not equivalent. The paper analyzes expected rounds of a repeat-until-success protocol, not the finite static depth scored here.

Implemented its rotation/QND-readout core with a Fourier Hamming-weight counter, actual adjacent CZ/PhasedXZ gates, full readout-line measurements, feedback angles, and the balanced-state reset threshold. For the first product-state rotation we use the exact small-system binomial optimum; off-balance feedback uses Eq.2. To satisfy finite exactness, exhausted trials trigger a conditioned reset and exact incumbent fallback. All failures are kept. Counter correctness is tested on every computational input, including relative phases, independently of the fallback. The symmetric-space Markov simulator records success probabilities and expected rounds separately from physical circuit costs.

| Target | Rounds | Counter depth | Full static depth | Incumbent depth | Prototype score term | Branches |
|---|---:|---:|---:|---:|---:|---:|
| D(2,1) | 1 | 54 | 72 | 3 | 0.02777778 | 3 |
| D(2,1) | 2 | 54 | 130 | 3 | 0.01538462 | 7 |
| D(4,1) | 1 | 230 | 280 | 7 | 0.01428571 | 13 |
| D(4,1) | 2 | 230 | 516 | 7 | 0.00775194 | 53 |
| D(4,2) | 1 | 230 | 328 | 7 | 0.02439024 | 11 |
| D(4,2) | 2 | 230 | 566 | 7 | 0.01413428 | 45 |

**Decision:** reject these prototypes for the score portfolio. Fourier-counter compilation, conservative move-and-restore routing, and mandatory fallback dominate small-system depth. This is a negative result for this concrete prototype, not a lower bound or a refutation of the paper's asymptotic algorithm. The paper's expected-round advantage cannot be substituted for `len(circuit)`. A practical next experiment is a much cheaper exact measurement/complement or measurement-uncompute gadget replacing a specific deep incumbent block; require all branches and the entire readout diagonal to pass before comparing the score.

## Reproduction

Source snapshots are pinned git archives; see `artifacts/manifest.json` for SHA256 and commits. They are ignored by git to avoid vendoring two repositories. Recreate them with `git archive` at the listed commits and extract into the paths under `snapshots/`. No package installation, processor job, or paid compute was performed.

```powershell
python -m adaptive_dicke_validation.audit --source adaptive_dicke_validation/snapshots/will-605b9a8/dicke_submission.py --label will-605b9a8 --nmax 20
python -m adaptive_dicke_validation.audit --source adaptive_dicke_validation/snapshots/kai-48115b9/final_v2/submission_v2.py --label kai-v2-48115b9 --nmax 20
python -m adaptive_dicke_validation.audit --source adaptive_dicke_validation/snapshots/kai-48115b9/dicke_submission.py --label kai-portfolio-913fcdf --nmax 15
python -m adaptive_dicke_validation.paper_prototype
python -m adaptive_dicke_validation.scoring_checks
python -m adaptive_dicke_validation.helper_checks
python -m pytest adaptive_dicke_validation/tests -q
python -m adaptive_dicke_validation.summarize
```

`portfolio.create_dicke_state_adaptive` is an experimental selectable API (Will, forced one/two-round Yu, or best validated), dependent on the pinned snapshot. It is not a standalone competition submission. Saved Cirq JSON circuits and per-case JSON results remain under artifacts. Original teammate files and earlier interference experiments are preserved.

Checks: Will's 26 package tests pass; five new independent auditor/counter tests pass. Kai's fast problem-1 count scorer matches the literal notebook functions in eight independently generated sample cases to ≤1e-15. No noisy-score improvement is claimed from that parity test.
