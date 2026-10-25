# Submission status (Kai's workspace, 2026-10-25 ~04:20 EDT)

All numbers below are **simulator proxies** (tools/sim.py trajectory noise model, official scorer, 10k shots per basis),
not hardware results. Higher is better; each instance is ~1 for good circuits, < 1 as N grows.

## Which file to submit

| File | What it is | Status |
|---|---|---|
| `final_v2/submission_v2.py` | **Recommended.** Team portfolio (= `dicke_submission.py` on `main`, unchanged for N ≤ 8) plus a library of exact, GPU/CPU-compressed op-list circuits for N ≥ 9, ranked against the portfolio circuit by one error estimate and verified noiselessly, with fallback to the portfolio circuit. Built by `tools/build_v2.py`. | Smoke-tested (see below); not yet in the full team_eval run. |
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
embedding. **Still to do:** a noisy-sim score for v2 (add it as candidate E in team_eval).

## Reproduce

```
pip install --pre cirq cirq-google numpy networkx
python tools/build_v2.py                      # rebuild final_v2/submission_v2.py
python team_eval/smoke_v2.py 101 "10:5,12:6"  # validity / exactness / timing vs v1
python tools/head2head.py SEED NMAX           # noisy-sim score, submission.py vs another file
```

Not pushed (size or ownership): `astra/`, `bonus/`, `notes/` (Astra), `columnforge/` (71 MB), `record_attempt/`,
`outputs/`, `team_eval/{circuits,scores,sources}`, pickles. Their useful parts are already inlined in the portfolio.
