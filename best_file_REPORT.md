# Reconciled DuQuantum submission

Submit **best_file.py**. It contains both notebook entry points and the available
AI prompt history, without project imports, external data, network access or
runtime fitting. Bonus is restricted to the user-confirmed **N<=15**.

SHA-256: `bb249a78550a0b00298c832865493944f2d35ad3a7c6dce2dfa541b9b9a5cc39`. Size: 1,835,305 bytes.
The original file/report are preserved under `reports/reconcile_20261025/sources/`.
The notebook and other original submissions were not changed.

## What changed

- Reconciled the original best file with Will commit
  [`1aaf4ffe29778615f247aa67baf412586a9ae7d4`](https://github.com/wwwillott/du-quantum/tree/1aaf4ffe29778615f247aa67baf412586a9ae7d4)
  and the generators recommended by judges-audit commit
  [`5d739d1c06169915bc6376799a18ca6a5e318770`](https://github.com/wwwillott/du-quantum/tree/5d739d1c06169915bc6376799a18ca6a5e318770/judges_audit).
  `compare.py` is a benchmark harness. Its `package.py` identifies the actual
  P1 source `judges_audit/sources/will_8af9eb4.py` and P2 source `problem2_submission.py`.
- Added **80 distinct accepted native templates**, for **249 total**. All 112
  remote source/case combinations were constructed on notebook seed 101; 107
  passed the strict amplitude check. Rejected: both sources' D(4,1) and D(8,3),
  plus the judges snapshot's D(9,4). Their residuals were 2.11e-9 to 6.09e-8;
  verified alternatives remain in the portfolio. Duplicate templates were removed.
- Bonus changes only **D(7,3), 24 to 23 moments**. It uses the remote complement
  construction with the original file's more precise D(8,4) parent. Both branches
  pass at residual 2.75e-15. The other 55 bonus circuits retain the original winners.

## Scoring update

The organizer now reports the harmonic mean of trial scores:

`1 / mean(1 / trial_score) = 1 / mean(Z_penalty + X_penalty)`.

Apply that average across repeated trials for each case, then sum the per-case
harmonic scores, consistent with the notebook and the screenshot's totals.

The previous audit's arithmetic-mean and median rankings cannot be treated as
rankings under this updated statistic. We independently checked the new
aggregation against the literal notebook scorer, including zero-valued ideal
correlators and finite-shot variance. The notebook itself was not rewritten.

The existing runtime selector already targets expected denominators for N<=8,
using the worst relative loss across all possible cutoffs. Above 8 it retains a
calibration-based gate/readout proxy; it does **not** run a full QVM simulation
inside the one-minute constructor. The reconciliation expands its verified
choices without pretending that this proxy guarantees the best hardware score.

The screenshot's preliminary cases D(10,3), D(10,5), D(15,4) were freshly compared
under Google's bundled noisy Willow QVM. Both remote sources return identical
circuits for those three cases on that calibration. The **final file returns
exactly the original best-file circuits** for all three, including their schedules
and physical layouts; `p1_benchmark/final_binding.json` binds them to this file hash.

Values below are final / Will-or-judges harmonic simulation scores. N=10 uses
exact noisy probabilities and finite-shot expectations; N=15 uses trajectory
estimates with batch-jackknife uncertainty. Every trial is 10,000 shots per basis.

| Case | Cutoff K=4 | Cutoff K=N | Method |
|---|---:|---:|---|
| D(10,3) | 0.693933 / 0.698679 | 0.670357 / 0.660937 | Exact density matrix |
| D(10,5) | 0.606075 / 0.598583 | 0.512295 / 0.491902 | Exact density matrix |
| D(15,4) | 0.550808 / 0.502061 | 0.439471 / 0.384895 | 512 trajectories per basis; uncertainty in receipt |
| Sum over the three cases | 1.850816 / 1.799322 | 1.622123 / 1.537733 | Includes N=15 trajectory uncertainty |

For D(10,3), the preferred circuit changes with the cutoff. The user confirmed
that the new cutoff was not supplied; no value is assumed and all K are retained
in the evidence. The organizer may also change the final (N,k) cases. These are
simulation comparisons on bundled calibration timestamp 2024-08-16, **not current
hardware measurements**. Trajectory and finite-trial uncertainty are recorded
separately; simulation precision does not remove noise-model uncertainty.

## Validation of this exact file

| Check | Result |
|---|---|
| P1, all 56 cases through N=15, notebook seeds 101 and 202 | **112/112 pass**, zero skips |
| Largest P1 amplitude residual | 1.71e-11 |
| Slowest construction in the P1 sweep | 8.650 seconds |
| P2, all 56 cases through N=15 | **56/56 pass**, zero skips |
| Bonus depth sum, sum N*k/moments | **52.559871754361** |
| Largest bonus numerical residual | 8.19e-14 |
| Complements, extremes, reversed edges, sparse/dropout maps, N=16..20 checks | **26/26 pass** |
| Deliberately invalid circuit/input regression checks | **26/26 pass** |
| Isolated standalone construction | **13 calls pass**, zero filesystem attempts after permitted dependency initialization |

The previous best bonus sum was 52.521828276100; Will is 41.371403219828 and the
judges recommendation is 49.349903201506 over the same confirmed 56-case domain.
The merged bonus is never deeper than any of the three inputs on that domain.

Checks enforce full CZ/PhasedXZ output operations, calibrated/physical qubits,
couplers, <=1750 moments, <60-second construction, output size, imports, and
complete complex128 amplitudes at tolerance 1e-10. Maximum construction in the
additional stress cases was 15.195 seconds (D(20,10)).
Numerical residual certification is distinct from a symbolic exactness proof.

Adaptive D(7,3) and D(11,5) are exhaustively checked on their actual returned
circuits. D(10,5) retains a circuit-bound deferred-measurement Clifford/Choi
certificate covering all 262,144 possible active readout strings. Full physical
readout diagonals and classical causality are checked separately. Cirq's device
validator rejects classical-control wrappers, so underlying native operations
are validated alongside the explicit control/readout checks.

The notebook helper's `range(qubits)` bug is corrected only in the verifier's
in-memory copy. Its undefined bonus `validate_circuit` is replaced by explicit
structural and state/branch checks. Bonus N>15 intentionally raises
NotImplementedError, consistent with the user's confirmed range.

## Reproduce

```powershell
python -B reports/reconcile_20261025/build_reconciled.py
python -B reports/reconcile_20261025/verify_reconciled.py --target best_file.py --mode main --max-n 15 --seeds 101,202 --output reports/reconcile_20261025/recheck_main
python -B reports/reconcile_20261025/verify_reconciled.py --target best_file.py --mode bonus --max-n 15 --output reports/reconcile_20261025/recheck_bonus
python -I -B reports/reconcile_20261025/final_stress/isolated.py
python -B reports/reconcile_20261025/scoring_audit/audit_scores.py
```

Use the new builder above; the older `reports/build_best_file.py` describes the
previous reconciliation. All current receipts, source hashes, rejected outputs,
scoring references and noise-model results are under `reports/reconcile_20261025/`.
