# Hypothesis-driven optimization, 2026-10-25

The new best measured submission is [`record_submission.py`](../record_submission.py).
It preserves Will's constructor and the existing P2 portfolio, then applies an
exact, circuit-signature-guarded Pauli frame pattern to D(8,4). The previous
[`recommended_submission.py`](../recommended_submission.py) is retained unchanged.

## Matched original-judge scores

| Submission | P1, K=N | P1, K=min(4,N) | P2, code N<=20 |
| --- | ---: | ---: | ---: |
| Previous Will portfolio | 23.773406667 | 24.049165046 | 82.323036395 |
| New guarded frame portfolio | **23.966727512** | **24.249253351** | **82.323036395** |
| Relative P1 improvement | **0.813%** | **0.832%** | unchanged |

P1 columns are **sums of per-case score medians**, not a single contest run or
the median of an aggregate. Each median uses 4,000 simulated independent trials
with the judges' **10,000 shots per basis**. Coverage is all 16 cases
N=2..8, k=1..floor(N/2), plus (9,1), (9,4), (10,1), (10,3), (10,5).
Unchanged cases reuse the incumbent's saved full-noise distributions and matched
seeds; the changed case is freshly simulated through the exported constructor.
The judges have not specified P1's full case set or correlator cutoff, so both
cutoffs are shown. P2's prose N<=15 instead gives 49.349903202; its executable
N<=20 loop gives the table's score. P2 was not improved in this round.

Full reproducible comparison, raw 10,000-shot QVM bitstrings, native circuit,
probabilities and export hash:
[`runs/record-20261025T100835002238Z/results.json`](runs/record-20261025T100835002238Z/results.json).

| D(8,4), matched 4,000-trial medians | Will | New | Change |
| --- | ---: | ---: | ---: |
| K=8 | 0.242980204 | **0.436301049** | **+79.56%** |
| K=4 | 0.247270603 | **0.447358908** | **+80.92%** |
| Native moments | 23 | 23 | unchanged |
| CZ count | 24 | 24 | unchanged |

One actual noisy QVM sample of the exported candidate scored **0.394987808**
(K=8) and **0.402918610** (K=4), using the literal original judge function.
This sample is separate from the median estimates. Exported construction took
10.80 seconds (60-second limit), ideal fidelity was 0.9999999999999996, and
device validation, native-gate/coupler checks and CircuitSerializer passed.

## Why this improves the score

The judge's Z penalty strongly amplifies unequal populations *within* the correct
Hamming-weight shell. A model based only on average gate errors missed the
coherent interference responsible for much of D(8,4)'s imbalance.

For any two local Pauli gates P around a CZ, choose the local Pauli correction
Q = CZ P-dagger CZ. Then Q CZ P equals the ideal CZ up to a global phase.
Changing these frames changes the orientation and accumulation of the physical
coherent errors while preserving the ideal Dicke state. Single-qubit frames fuse
into adjacent native PhasedXZ gates; the promoted circuit adds no CZs or depth.

We screened ten seeded patterns, then used at most 48 frame-coordinate proposals
starting from the best seed. Proposals were ordered by supplied coupler error.
The search minimized the **expected original judge denominator**, including
10,000-shot variance; an accepted step needed at least a 0.5% reduction.
This is a search proxy, not an assertion that expected inverse score equals the
inverse expected denominator. Final ranking uses fresh score distributions.

## Independent validation and decisions

[`hypotheses.json`](hypotheses.json) records mechanisms, bounded searches and
falsifiers. Every run has its own immutable timestamp directory and saved
circuits/probabilities; failed ideas remain available.

| Hypothesis | Result | Decision |
| --- | --- | --- |
| H1: correct cheap placement surrogate | Best cheap-model leads <=0.43%; does not explain coherent-noise gap | No promotion |
| H2: single-qubit fusion/repacking | No material robust lead | No promotion |
| H3: balanced terminal global X | D(4,2) screening lead collapsed to +0.43% in fresh validation | No promotion |
| H4: exact output symmetries for P2 depth | No strict depth gains on tested cases through N=15 | P2 unchanged |
| H5: seeded exact CZ Pauli frames | D(8,4) seed 3 validates at +57.8%; D(6,3) seeded patterns lose | Seed 3 becomes H6 starting point |
| H6: coordinate-refined frames | D(8,4) fresh validation +79.7%, unchanged depth/CZ count | **Promote D(8,4) only** |
| H5/H6 on D(7,3), D(8,3) | Nominal leads, including refined D(8,3) +38.4%, fail or lack independent placement validation | Keep experimental |
| H7: train frames across three placements | 24 proposals; fresh nominal ratio 0.994, reserved patches 0.978 and 1.007 | Reject |

The final D(8,4) refinement was checked with complex128 density matrices and
8,000 fresh shot trials; it improved K=8 by **79.7%**. Three physical placements
not used in its frame search improved by **70.8%, 63.1%, 60.4%**. These are
different calibrated patches, not independent real-device calibrations.

Holding stochastic/thermal noise fixed, coherent-angle scaling gave K=8 ratios:

| Coherent angle scale | New / Will |
| --- | ---: |
| -1 | 1.640 |
| 0 | 1.007 |
| 0.5 | 1.294 |
| 1.5 | 2.180 |

The near-unity zero-coherent result supports the proposed coherent-error
mechanism. These are diagnostic model stresses, not measured chip conditions.
The fixed-stochastic ablation deliberately freezes Cirq's residual depolarizing
error; simply replacing FSim angles otherwise reallocates the supplied Pauli
error budget. An earlier screening ablation used that reallocation and is not
used for this causal claim.

Evidence: [seed validation](runs/validation-20261025T095410289633Z/results.json),
[refinement](runs/refine-20261025T095636098346Z/results.json),
[independent refined validation](runs/validation-20261025T100047232801Z/results.json),
[rejected expansion validation](runs/validation-20261025T100058071510Z/results.json).
The final [H7 multi-placement run](runs/multiplacement-20261025T100606715901Z/results.json)
also failed its nominal improvement criterion; its more stable near-baseline
pattern did not retain the nominal-only D(8,3) gain.

## Simulation scope and portable submission

All noisy claims use Cirq Google's bundled **willow_pink** calibrated noise,
including coherent FSim, damping/dephasing, depolarization and asymmetric readout
noise. The bundled calibration is dated **2024-08-16**. We cannot run the physical
chip; this is a **local simulated record**, not a hardware or global optimum claim.

The export calls Will's calibration-aware placement code first. The frame pattern
is applied only when a coordinate-independent signature matches the tested
logical native circuit; other cases or changed circuit families fall back to
Will's output. It uses no hardcoded physical patch. The signature is a structural
guard, not a guarantee of improvement under every possible noise calibration.
P2 and all incumbent source are preserved verbatim as the export's prefix.

Screening uses complex64 density matrices and 64 score resamplings per strategy;
validation uses complex128, fresh random seeds and 3,000--12,000 resamplings.
Every individual trial still uses the official 10,000 shots. Tests cover exact
frame unitaries, symmetry, scorer expectation, fixed-stochastic ablations,
signature translation invariance and export/experiment agreement.
The completed verification run passed **11 tests**. Runtime versions were
Cirq/Cirq Google `1.8.0.dev20261023221525`, NumPy `2.5.0`, SciPy `1.18.1`.

## Reproduce

Run from the repository root in the existing Python environment:

```powershell
python -m pytest optimization_lab/tests/test_exact.py -q
python -m optimization_lab.qvm_screen --help
python -m optimization_lab.refine_frames --n 8 --k 4 --seed 3 --budget 48
python -m optimization_lab.validate --run optimization_lab/runs/refine-20261025T095636098346Z --refined
python -m optimization_lab.export --runs optimization_lab/runs/refine-20261025T095636098346Z
python -m optimization_lab.benchmark_record
```

The final two commands recreate the standalone candidate and audit it through
its actual API, including a noisy QVM sample scored by the original notebook.
New experiment outputs receive new timestamp directories.
