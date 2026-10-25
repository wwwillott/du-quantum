# Submission status (Kai's workspace, updated 2026-10-25)

All numbers below are **simulator proxies** (tools/sim.py trajectory noise model, official scorer, 10k shots per basis),
not hardware results. Higher is better; each instance is ~1 for good circuits, < 1 as N grows.

## Which file to submit

| File | What it is | Status |
|---|---|---|
| `final_v2/submission_v2.py` | **Recommended.** Team portfolio (= `dicke_submission.py` on `main`, unchanged for N ≤ 8) plus a library of exact, GPU/CPU-compressed op-list circuits for N ≥ 9, ranked against the portfolio circuit by one error estimate and verified noiselessly, with fallback to the portfolio circuit. Built by `tools/build_v2.py`. | Smoke-tested and noisy-sim scored (see below): best total, +0.1% over A. |
| `dicke_submission.py` (on `main`) | Team portfolio, candidate **A** in `team_eval/RESULTS.md`. Includes `create_dicke_state_adaptive` (bonus). | Fully evaluated, safest fallback. |
| `submission.py` | Root pipeline (`build_submission.py` inlines `dicke_core.py` + `dicke_main.py` + `data/templates` + `data/peep`), candidate **B**. | Fully evaluated; about tied with A. |

`FINAL_SUBMISSION.py` in the local workspace is an older copy (byte-identical to the 2026-10-24 19:33 build) and is
deliberately **not** pushed, to avoid confusion.

## Head-to-head (team_eval/RESULTS.md, partial run: N ≤ 11 scored)

Sum of per-instance medians over instances that all four candidates have scored (seed = calibration seed):

| Seed | Model | A portfolio | B root `submission.py` | C ColumnForge | D Astra |
|---:|:---|---:|---:|---:|---:|
| 101 | Pauli | **22.67** | 22.45 | 22.35 | 22.53 |
| 101 | HW | **20.11** | 20.01 | 19.14 | 20.02 |
| 202 | Pauli | **20.97** | 20.94 | 20.63 | 20.94 |
| 202 | HW | 19.08 | **19.18** | 18.26 | 18.94 |

A and B are within ~1% of each other. They win different instances: A is much better at D(8,4) (24 vs 45 CZ),
B is better at D(10,3) and D(10,4) (45/73 vs 57/89 CZ). v2 keeps A for N ≤ 8 and takes the lowest-error exact
circuit for N ≥ 9, which is meant to get the best of both.

## v2 smoke test (`python team_eval/smoke_v2.py 101 ...`, calibration seed 101)

Every circuit passes `Willow105.validate_circuit`, uses only calibrated CZ pairs and qubits, and has noiseless fidelity
1.000000000000. Build time is ≤ 15 s (the slowest is D(8,4) at 14 s).

| (N, n) | v1 = portfolio CZ / moments | v2 CZ / moments |
|---|---|---|
| 2,1 / 4,2 / 6,3 / 8,4 | 1/3, 5/7, 12/14, 24/23 | identical |
| 9,4 | 61 / 62 | 61 / 62 |
| 10,3 | 57 / 51 | **45 / 45** |
| 10,5 | 73 / 73 | 73 / 77 |
| 12,6 | 117 / 101 | 117 / 102 |
| 14,7 | 171 / 135 | 171 / 137 |
| 16,8 | 309 / 267 | 309 / 267 |
| 20,10 | 505 / 359 | 505 / 359 |

So far v2 has only produced a clear gain where the root pipeline beat the portfolio (D(10,3)). Elsewhere it ties.
The few extra moments at D(10,5), D(12,6) and D(14,7) come from the error estimate picking a different, equal-CZ
embedding.

## v2 noisy-sim score vs A and B (`team_eval/v2/eval_v2.py`)

The team_eval pipeline was used unchanged: the same saved calibrations, per-instance RNG seeds (which depend only
on seed, N, n and model, so a circuit identical to A's or B's gets an identical score), 2000 Pauli / 1000 HW
trajectories, 400 resamplings, and K = N. 40 of v2's 60 circuits (N ≤ 11) were byte-identical to an already-scored
A or B circuit and reuse that score; the other 20 were simulated. All 60 are valid and exact, and each builds in
≤ 15 s.

Sum of medians over the 24 instances per seed where A, B and v2 all have scores (N ≤ 11; seed 202 has no A/B
scores for N ≤ 5 yet). "Best of A/B" is the per-instance maximum, i.e. the ceiling for a perfect selector:

| Seed | Model | A portfolio | B `submission.py` | **v2** | best of A/B |
|---:|:---|---:|---:|---:|---:|
| 101 | Pauli | **22.668** | 22.454 | 22.645 | 22.800 |
| 101 | HW | 20.511 | 20.425 | **20.556** | 20.805 |
| 202 | Pauli | **22.546** | 22.437 | 22.408 | 22.900 |
| 202 | HW | 20.266 | 20.413 | **20.488** | 20.787 |
| all | total | 85.991 | 85.729 | **86.097** | 87.292 |

**Verdict:** v2 is the best single file, but only just. It wins the HW model (T1 and dephasing, the more realistic
one) on both seeds and is +0.1% over A in total. That is within the run-to-run differences of the proxy, so
in practice A and v2 are tied, and B is about 0.3% behind. For N ≤ 8, v2 is A.

Where v2's selector gains and loses (N ≥ 9; Pauli/HW medians):
- **Gains:** D(9,3) s101 0.859/0.760 against A 0.836/0.708 (45 vs 48 CZ). D(10,3) on both seeds (takes B's
  45-CZ circuit). D(10,4) and D(11,2) on s202 under HW.
- **Losses:** D(10,5) on both seeds: v2 scores 0.717/0.520 (s101) and 0.665/0.514 (s202), against A's
  0.743/0.552 and 0.709/0.545. On s202 it even picked B's 79-CZ circuit over A's 73-CZ one. Also D(11,5) and
  D(9,3) on s202 under Pauli, slightly.
- **Missed:** W states, where B beats A on some seeds (e.g. D(10,1) s202: B 1.155/1.090 vs A 1.036/1.057). v2
  keeps A there.

Closing the gap to "best of A/B" (+1.4%) needs a better ranking of candidates than `estimate_error`. The simplest
fix: always keep A's circuit for half-filled states (n = N/2) and allow B's W-state circuits.

## Reproduce

```
pip install --pre cirq cirq-google numpy networkx
python tools/build_v2.py                      # rebuild final_v2/submission_v2.py
python team_eval/smoke_v2.py 101 "10:5,12:6"  # validity / exactness / timing vs v1
python tools/head2head.py SEED NMAX           # noisy-sim score, submission.py vs another file
```

Not pushed (size or ownership): `astra/`, `bonus/`, `notes/` (Astra), `columnforge/` (71 MB), `record_attempt/`,
`outputs/`, `team_eval/{circuits,scores,sources}`, pickles. Their useful parts are already inlined in the portfolio.
