# Original judges' notebook audit

**Recommendation under the calibrated Willow noisy QVM: Will for Problem 1,
and the combined depth portfolio for Problem 2.** Both APIs are packaged in
[`recommended_submission.py`](../recommended_submission.py). These are the best
tested choices under the stated simulation, not a proof of a universal optimum.

## Authoritative rules

The supplied original notebook is archived unchanged at
[`sources/judges_original.ipynb`](sources/judges_original.ipynb), SHA256
`ee1d7d53f64bde2e5f8da114d0bc198ea2666e40cc65ea4eadca2f97c062acd1`.
Its Problem 1 scoring function is AST-identical to the repository notebook.
Only the relevant function definitions are executed; document instructions
such as uploading to a form are not task authorization.

- **P1:** actual Z/X samples, 10000 shots per basis, score `1/(Z_penalty+X_penalty)`.
  We do not substitute fidelity, depth, a gate-count proxy or a surrogate cost
  for this function. All original zero-correlator cases are retained.
- P1 does **not** specify which (N,k) cases or `max_correlator_size=K` the judges
  will use. We report K=N and K=min(4,N) separately. Absolute ranking from the
  document alone is therefore impossible.
- **P2:** executable cell 16 sums `N*k/len(circuit)` over N=2..20, 100 cases.
  The prose separately describes N<=15 and one point per shallowest case. We
  report both range totals; they are not interchangeable competition rules.
- The original illustrative state checker has `range(qubits)` where `qubits`
  is a set, and P2 refers to `validate_circuit` without defining it. We supply
  explicit audited validation rather than silently treating those helpers as
  evidence that a circuit is correct.

## Fresh noisy Willow simulation

The backend is Google's `create_default_noisy_quantum_virtual_machine` for
`willow_pink`, using its bundled processor noise properties and median
calibration, timestamp **2024-08-16 08:11:17.630 UTC**. This is simulated Willow
noise, not a current live calibration. Complex128 density matrices avoid
trajectory-estimation error. Both Z and X circuits are executed through the
virtual engine with 10000 measurements, including noise on the X-basis rotations
and readout. The literal original scorer is applied to these bitstrings.

For stable comparison we also generate 4000 independent pairs of 10000-shot
draws from each exact noisy distribution. Fixed per-case seeds make identical
circuits receive identical statistics, regardless of earlier benchmark cases.
The table reports **sums of per-case median scores**. These are repeated-shot
performance summaries, not one official-shot aggregate. The actual one-run
score sums and bitstrings are retained in the JSON/NPZ artifacts.

Every main case with N=2..8 and 1<=k<=N//2 is included (16 cases). A matched
extension compares Will and latest Kai on D(9,1), D(9,4), D(10,1), D(10,3),
D(10,5), giving 21 cases total. No failed cases were dropped from these totals.

| Portfolio | 16 cases, K=N | 16 cases, K<=4 | 5 extra cases, K=N | 21-case total, K=N |
|---|---:|---:|---:|---:|
| **Will 8af9eb4** | **20.087912** | **20.194655** | **3.685495** | **23.773407** |
| Kai patched v2, 4163b40 | 19.665219 | 19.750184 | 3.492110 | 23.157329 |
| Current team/root submission | 19.665219 | 19.750184 | Not run | Not comparable |
| Repository SWAP-routed SCS | 11.570882 | 11.632052 | Not run | Not comparable |

Will leads latest Kai by about 2.66% on the matched 21-case K=N score summary,
and also leads with K<=4. Its strategy is state/size-aware selection of exact
short native circuits, calibration-aware placement, W broadcast trees,
coherent count decoders and echo/DD where useful. This does not mean every
individual Will circuit wins, nor that the oldest routed SCS implementation
is the best possible implementation of the literature algorithm.

The remote branch was pulled and all origin tips fetched. Latest Will's code
is AST-identical to the pinned 605b9a8 file actually evaluated; subsequent
commits add reports/plots only. Latest Kai changes only its large-state library
and final selector; the N<=8 base code path is unchanged from the tested v2.
The five larger cases use the **new 4163b40 source directly**.

As a sensitivity check, the previously saved multi-seed synthetic Pauli/T1/T2
benchmark ranks latest Kai first: **85.359226** versus reference A **84.983710**
and root B **84.734146**, on common available cases only (24 seed-101 and 23
seed-202 cases, each under two models). This was recomputed from raw rows,
not copied from the published totals. It differs from the calibrated QVM in
both noise model and calibration. It demonstrates why there is no justified
model-independent "absolute best" claim.

## Hardware compatibility checks

Before every P1 noisy run, we check the ideal target state, output cardinality,
available calibration/readout qubits, calibrated undirected couplers, full CZ
and PhasedXZ gates only, no feedback or measurement in preparation, Willow
device validation, actual `CircuitSerializer` serialization, <60 s construction,
and <=1750 moments. The X readout rotations are compiled to native PhasedXZ.

Additional full-vector checks of latest Kai cover D(12,6), D(16,8), D(20,1),
D(20,10). Their depths are 101,267,23,359, construction times <=2.50 s and
minimum fidelity 0.99999999999930. Those larger cases are **contract/ideal-state
checks**, not extra noisy score measurements. These checks establish compatibility
with the tested Willow device/calibration model; they do not certify every
future disabled-qubit map. The user confirmed simulation is the target; no
physical processor job was submitted.

The same four large ideal-state/contract checks also pass for the recommended
Will API: depths 101,283,19,375; construction <=3.13 s; minimum fidelity
0.99999999999904. They are recorded in `results/large_contract_will.json`.
The combined submission's P1 circuits for D(2,1) and D(8,4) additionally match
the scored Will circuits exactly (`packaged_p1_smoke.json`). In total, all 74
variant/case noisy-QVM runs completed successfully (148 basis circuits,
1480000 actual QVM measurement shots).

## Problem 2

| Portfolio | Literal N<=20 depth sum | N<=15 depth sum |
|---|---:|---:|
| **New combined portfolio** | **82.323036395** | **49.349903202** |
| Previous verified interference/depth integration | 82.152487741 | 49.179354547 |
| Will public bonus | <=73.411927801 | 41.371403220 |
| Kai public bonus | <=73.078797447 | 41.103632343 |

The new portfolio selects the per-case shallowest circuit among the tested
portfolios, preserving the fully verified 100-case construction and promoting
two independently checked Will circuits: D(7,3), 24->23 moments, and D(8,4),
23->21. Both branches of the former are exact. All readout diagonals, classical
causality and physical operations pass validation. The large cat/teleportation
case retains the earlier all-branch Clifford Choi certificate.

The last two full-range values are **depth-only upper bounds**, since their
large cases were not independently fully state-verified in the previous audit.
Their N<=15 scores have full state/branch validation. Even their upper bounds
are below the new verified score. The original 100-case state verification is
reused for unchanged constructions; the two replacements are freshly checked.
`package.py` additionally runs the literal original `score_solution` through
the packaged API, checking every returned circuit's structure, depth and time.

Interference and finite Yu collective-measurement prototypes remain worse
after full routing/measurement/fallback costs; they are not promoted just
because their local leaf or expected-round count looks favorable.

## Reproduction and evidence

From the repository root:

```powershell
python -m judges_audit.compare --candidate team --max-n 8
python -m judges_audit.compare --candidate will --max-n 8
python -m judges_audit.compare --candidate kai_v2 --max-n 8
python -m judges_audit.compare --candidate scs --max-n 8
python -m judges_audit.compare --candidate kai_latest --label kai_large --max-n 10 --cases 9:1,9:4,10:1,10:3,10:5
python -m judges_audit.compare --candidate will --label will_large --max-n 10 --cases 9:1,9:4,10:1,10:3,10:5
python -m judges_audit.rescore team will kai_v2 scs kai_large will_large
python -m judges_audit.hardware_preflight
python -m judges_audit.hardware_preflight --candidate will
python -m judges_audit.improve_depth
python -m judges_audit.report
python -m judges_audit.package
```

`results/summary.json` contains the table numbers. Per-case JSON, native circuit
JSON and NPZ files retain scores, source hashes, exact noisy probabilities and
all QVM bitstrings. `large_contract.json` records large ideal-state checks;
`problem2_best.json` records the complete depth portfolio and new branch checks;
`packaged_score.json` records the literal original P2 scorer execution and the
packaged file hash. The benchmark count/histogram scorer agrees with the literal
original scorer on every actual QVM sample; vectorized resampling is separately
cross-checked with that count formula.

Official backend documentation: [Google Quantum AI QVM](https://quantumai.google/cirq/simulate/quantum_virtual_machine).
