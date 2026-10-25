# NEW BEST Problem 1: note for Astra agent

2026-10-25. User-requested handoff on Kai's `kai/best-file` branch.

## Actionable result

We found a strongly validated improvement to **Will's D(8,4) native circuit**:
exact local Pauli frames around its CZ gates reduce coherent noise accumulation.
The matched original-judge median score rises **0.242980204 -> 0.436301049
(+79.56%, K=8)** with **23 moments and 24 CZ gates unchanged**.

This was compared against Will's `recommended_submission.py`, **not** the
reconciled `best_file.py` on this branch. Do not treat it as proof that this new
portfolio beats Kai's current file. Compare the D(8,4) candidate against the
current reconciled candidate on matched calibrated-noise simulations before
integrating it. Keep all other verified cases and the bonus implementation.

## Implementation and evidence, saved and pushed

All code, raw circuits, probabilities, QVM shot samples, negative results and
tests are in commit **831c14658aae8c015c7f7f112c1017e624c26762** on
`yuan-zhang-7x8`, in the same repository:

- [Standalone candidate: record_submission.py](https://github.com/wwwillott/du-quantum/blob/831c14658aae8c015c7f7f112c1017e624c26762/record_submission.py)
- [Full findings and reproduction commands](https://github.com/wwwillott/du-quantum/blob/831c14658aae8c015c7f7f112c1017e624c26762/optimization_lab/README.md)
- [Matched exported-API/QVM audit](https://github.com/wwwillott/du-quantum/blob/831c14658aae8c015c7f7f112c1017e624c26762/optimization_lab/runs/record-20261025T100835002238Z/results.json)
- [Fresh refined-candidate validation and held-out stresses](https://github.com/wwwillott/du-quantum/blob/831c14658aae8c015c7f7f112c1017e624c26762/optimization_lab/runs/validation-20261025T100047232801Z/results.json)
- [Frozen 24-pair frame pattern and source manifest](https://github.com/wwwillott/du-quantum/blob/831c14658aae8c015c7f7f112c1017e624c26762/optimization_lab/export_manifest.json)
- [Portable transformation and circuit-signature guard](https://github.com/wwwillott/du-quantum/blob/831c14658aae8c015c7f7f112c1017e624c26762/optimization_lab/runtime.py)

Retrieve selected files without merging the entire research branch, if useful:

```sh
git fetch origin
git show 831c146:record_submission.py > record_submission_candidate.py
git show 831c146:optimization_lab/README.md
```

## Scores and their exact meaning

| Metric | Previous Will portfolio | New guarded portfolio |
| --- | ---: | ---: |
| P1 matched 21-case sum, K=N | 23.773406667 | **23.966727512 (+0.813%)** |
| P1 matched 21-case sum, K=min(4,N) | 24.049165046 | **24.249253351 (+0.832%)** |
| D(8,4) median, K=8 | 0.242980204 | **0.436301049** |
| D(8,4) median, K=4 | 0.247270603 | **0.447358908** |

Each P1 aggregate is the **sum of per-case medians of 4,000 trials**, with the
original **10,000 shots in each Z and X basis per trial**. It is not a single
contest run, an expected score, or the median of an aggregate. Cases are all 16
N=2..8, k=1..floor(N/2), plus (9,1), (9,4), (10,1), (10,3), (10,5).
Unchanged cases reuse exact saved incumbent distributions and matched seeds;
D(8,4) is freshly simulated through the actual exported API. Full-noise density
distributions include readout noise. Both cutoffs are reported because the
original P1 test cases/cutoff are unspecified.

One actual 10,000-shot-per-basis noisy QVM run of the exported file, scored with
the original notebook function, returned **0.394987808 (K=8)** and
**0.402918610 (K=4)**. This is separate from the medians above.

## Mechanism and safe integration

For local Pauli P on the two CZ wires, Q = CZ P-dagger CZ is also a local Pauli;
Q CZ P equals CZ up to global phase. Frames are fused into adjacent PhasedXZ
gates. The ideal full unitary is preserved, while physical coherent errors
accumulate differently. The gain mainly reduces the judge's heavily weighted
nonuniformity within the correct Hamming-weight shell.

The tested export calls Will's calibration-aware constructor, then applies the
saved frame pattern only if its native circuit signature matches. There are no
hardcoded physical-qubit coordinates. Unmatched circuit families fall back to
Will. **Do not transplant the 24 frame pairs onto a different native circuit**;
the saved signature is `4d5f09484cfb538c933a747584f64a4bc54de9bb48e94bdc9998cb28ff387377`.
If Kai's selector chooses a different D(8,4) circuit, compare the saved candidate
as a separate exact template or redo frame search for that circuit.

Exported construction took 10.80 seconds; ideal fidelity was
0.9999999999999996, maximum amplitude residual 8.29e-12. Native gates, supplied
couplers/qubits, Willow device validation, CircuitSerializer and the 1750-moment
limit passed. Eleven focused tests passed. Recheck orientation and the current
branch's stricter contract checks when integrating.

## Robustness and rejected ideas

Fresh complex128 validation with 8,000 new shot trials gave **+79.7%** for
D(8,4). Three physical placements not used in frame search gave **+70.8%,
+63.1%, +60.4%**. These are different patches of one calibration, not independent
real-device calibrations. At coherent-angle scales -1, 0, 0.5, 1.5, keeping
thermal/stochastic noise fixed, new/Will score ratios were **1.640, 1.007,
1.294, 2.180**. The near-unity zero-coherent result supports the mechanism.

Search: ten seeded patterns, then at most 48 coordinate proposals; objective
was expected original denominator including finite-shot variance. Screening used
complex64 and 64 score draws; final ranking used complex128 and fresh seeds.
Every trial retained 10,000 shots. Do not replace official scoring with a lower
shot count. Expected inverse score and inverse expected denominator differ.

Retained negative results: simple fusion, terminal-X symmetry and cheap placement
surrogates did not yield robust promotions. D(7,3)/D(8,3) nominal leads failed
placement validation; even a 24-proposal search trained across three patches
did not retain a reliable D(8,3) gain. **Only D(8,4) was promoted.** No P2 depth
improvement was found in this round. Earlier P2 numbers 82.3230 (code N<=20)
and 49.3499 (prose N<=15) are from our different portfolio, not Kai's reported
52.5218 for N<=15; do not substitute them for this branch's bonus audit.

All noisy results are **simulated**, using Cirq Google's bundled `willow_pink`
calibration dated **2024-08-16**, not physical hardware. Versions:
Cirq/Cirq Google 1.8.0.dev20261023221525, NumPy 2.5.0, SciPy 1.18.1.
This is a measured local improvement, not proof of a universal optimum.
