# Submission status (Kai's workspace, updated 2026-10-25)

All numbers below are **simulator proxies** (tools/sim.py trajectory noise model, official scorer, 10k shots per basis),
not hardware results. Higher is better; each instance is ~1 for good circuits, < 1 as N grows.

## Which file to submit

| File | What it is | Status |
|---|---|---|
| `final_v2/submission_v2.py` | **Recommended.** Team portfolio (= `dicke_submission.py` on `main`, unchanged for N ≤ 8) plus a library of exact, GPU/CPU-compressed op-list circuits for N ≥ 9, ranked against the portfolio circuit by one error estimate and verified noiselessly, with fallback to the portfolio circuit. For half-filled states (n = N/2) it always returns the portfolio circuit; for W states it also ranks the root pipeline's broadcast tree. Built by `tools/build_v2.py`. | Smoke-tested and noisy-sim scored (see below): best total, +0.45% over A. |
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

## v2 smoke test (`python team_eval/smoke_v2.py 101 ...`, calibration seed 101, first v2 build)

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
The few extra moments at D(10,5), D(12,6) and D(14,7) came from the error estimate picking a different, equal-CZ
embedding. The patched build returns the portfolio circuit for all half-filled states, so those now match v1.

Patched build, seed 202, larger N (`smoke_v2.py 202 ...`). All circuits are valid and exact, and each builds in ≤ 4 s:
D(12,1) 21 CZ / 13 moments, D(15,1) 27 / 14, **D(20,1) 37 / 19 (v1: 68 moments)**, D(12,6) 117 / 101,
**D(13,6) 153 / 121 (v1: 185 / 193)**, D(16,8) 309 / 267, D(20,10) 505 / 359, **D(15,14) 27 / 19 (v1: 53 moments)**.

## v2 noisy-sim score vs A and B (`team_eval/v2/eval_v2.py`)

The team_eval pipeline was used unchanged: the same saved calibrations, per-instance RNG seeds (which depend only
on seed, N, n and model, so a circuit identical to A's or B's gets an identical score), 2000 Pauli / 1000 HW
trajectories, 400 resamplings, and K = N. For N ≤ 11 on both seeds: in the first v2 build, 40 of 60 circuits were
byte-identical to an A or B circuit and reused that score, and the other 20 were simulated. Every circuit in the
patched build is one of those already scored. All are valid and exact; the patched build takes ≤ 13 s per instance.

Sum of medians over the 24 instances per seed where every column has scores (N ≤ 11; seed 202 has no A/B scores for
N ≤ 5 yet). "Best of A/B" is the per-instance maximum, i.e. the ceiling for a perfect A/B selector:

| Seed | Model | A portfolio | B `submission.py` | v2 first build | **v2 patched** | best of A/B |
|---:|:---|---:|---:|---:|---:|---:|
| 101 | Pauli | **22.668** | 22.454 | 22.645 | 22.657 | 22.800 |
| 101 | HW | 20.511 | 20.425 | 20.556 | **20.574** | 20.805 |
| 202 | Pauli | 22.546 | 22.437 | 22.408 | **22.599** | 22.900 |
| 202 | HW | 20.266 | 20.413 | 20.488 | **20.548** | 20.787 |
| all | total | 85.992 | 85.729 | 86.097 | **86.379** | 87.291 |

**Verdict:** the patched v2 is the best file: +0.39 (+0.45%) over A and +0.65 over B in total, ahead on 3 of the 4
seed/model sums, and only 0.011 behind A on the fourth (s101 Pauli). The margins are small, so A remains a fine
fallback. For N ≤ 8, v2 is A.

What the patch changed (old → new, Pauli/HW medians):
- **Half-filling rule (n = N/2 → portfolio circuit):** D(10,5) s101 0.717/0.520 → 0.743/0.552, and D(10,5) s202
  0.665/0.514 → 0.709/0.545 (79 → 73 CZ).
- **W states (B's broadcast tree as a candidate):** D(10,1) s202 1.036/1.057 → 1.155/1.090 (34 → 15 moments), and
  D(11,1) s202 1.138/1.038 → 1.167/1.034. Against that, D(9,1) s101 1.187/1.099 → 1.173/1.085.

Remaining gaps, all from `estimate_error` choosing badly:
- **W states:** it picks the better W circuit in only 3 of 6 cases (it keeps A at s101 D(11,1) and s202 D(9,1)
  where B is better, and takes B at s101 D(9,1) where A is better). Always taking B for N ≥ 9 would add about
  +0.03 Pauli / +0.03 HW more on these 6 instances, but that would be tuning on the same small sample we measure on.
- **Odd N near half filling:** D(9,4) and D(11,5) can still lose a little to A. The rule above only covers even N.
- **Gap to best of A/B:** now 0.91 (about 1%).

## Reproduce

```
pip install --pre cirq cirq-google numpy networkx
python tools/build_v2.py                      # rebuild final_v2/submission_v2.py
python team_eval/smoke_v2.py 101 "10:5,12:6"  # validity / exactness / timing vs v1
python team_eval/v2/eval_v2.py 10             # noisy-sim score v2 with the team_eval pipeline (needs team_eval/scores cache for speed)
python tools/head2head.py SEED NMAX           # noisy-sim score, submission.py vs another file
```

Not pushed (size or ownership): `astra/`, `bonus/`, `notes/` (Astra), `columnforge/` (71 MB), `record_attempt/`,
`outputs/`, `team_eval/{circuits,scores,sources}`, pickles. Their useful parts are already inlined in the portfolio.
