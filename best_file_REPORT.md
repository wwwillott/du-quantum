# DuQuantum reconciled submission

New Problem 1 research handoff for Astra:
[NEW_BEST_prob_1_note_for_astra_agent.md](NEW_BEST_prob_1_note_for_astra_agent.md).
It reports a validated D(8,4) improvement over Will's baseline, with implementation
and evidence links; comparison against this reconciled file remains to be done.

Submit **best_file.py**. It is self-contained and includes both notebook entry points and the available AI prompt logs in `AI_PROMPTS`. The separate best_file_PROMPTS.md is a readable copy. The original two submissions, root submission.py, and notebook were preserved unchanged.

SHA-256: `5a31b10eaa83b948a0be6940fcfa3c16f8da0d0a2fdc0931748c873c6edad1b2`. File size: 1,397,132 bytes.

## Validation of the exact final file

| Check | Result |
|---|---|
| Main, all 56 pairs N=2..15 on notebook calibration seeds 101 and 202 | **112/112 pass**, no skips |
| Maximum main amplitude residual in that sweep | 7.1e-14 |
| Slowest main construction in that sweep | 6.611 seconds |
| Bonus, all 56 pairs N=2..15 | **56/56 pass**, no skips |
| Bonus score, sum N*k/moments over N<=15 | **52.521828276100** |
| Maximum bonus numerical residual | 8.19e-14 |
| Slowest bonus construction | 0.170 seconds |
| Additional extreme/complement/dropout/orientation/larger-state checks | **26/26 pass** |
| Deliberately invalid gate/state/input regression checks | **26/26 pass** |
| Isolated constructors with runtime file reads prohibited | **Pass**, zero file reads |

The amplitude acceptance threshold is 1e-10 in complex128, with output normalization checked separately. This tests the full state including relative phases, not just a rounded fidelity. Finite precision certification is not a symbolic exactness proof. Tests include the notebook's own state helper with its `range(qubits)` bug corrected to `range(len(qubits))` **in the test harness only**. The notebook itself was not edited.

All main circuits use only full CZ gates on supplied couplers and finite PhasedXZ rotations on supplied, physical Willow qubits. CZ operands use a literal orientation present in the input dictionary. Outputs have exactly N qubits; main circuits contain no measurements, feedforward, or ancillas. Every returned main circuit receives a fresh structural/device check and full noiseless amplitude check. Illegal or unsupported requests raise NotImplementedError. Circuits remain within 1750 moments. Main supports valid requests through 20 with an analytic fallback; the larger-domain checks are representative, not an exhaustive two-calibration sweep. The largest additional measured construction time was 21.825s, including D(20,10).

Tested with Cirq and Cirq-Google 1.8.0.dev20261023221525. Imports are limited to Cirq/Cirq-Google, their NumPy/SymPy dependencies, and Python's standard library. No fitting, local source imports, data-file reads, network calls, or hardware access occur during construction.

## What was reconciled

The main entry point keeps verified native templates from both requested source files and selects placements using the supplied calibration. Source schedules and complete native operations are retained. This removes expensive online fitting and nested portfolio searches from the submission path. If defects prevent any saved patch from embedding, an analytic path construction is tried.

Five raw V3 outputs passed its loose fidelity check despite amplitude errors above 1e-10: D(8,3), D(10,2), D(10,3), D(10,5), and D(11,2). Their original versions are excluded. Offline precision repairs preserved each circuit's CZ topology, gate count, and moment count, and reached independently measured residuals between 2.85e-15 and 1.43e-14. All five original/repair pairs are covered by rejection/acceptance regression tests. Repaired circuits are embedded as complete templates; their nominal echo pulses must not be removed independently.

For N<=8, the selector compares predicted 10,000-shot notebook denominators under independent gate-Pauli and asymmetric-readout noise, considering every possible X-correlator cutoff. It minimizes worst relative loss because the notebook does not specify the cutoff. For larger cases, it uses calibrated gate/readout costs. These are estimates, not an exact characterization of Willow noise or a guarantee of the highest hardware score.

On the tested calibration, 15 of 16 small cases improve or tie both compliant source outputs at every cutoff under this deterministic model. D(4,2) trades a 0.0408% increase at cutoff 4 for improvements at cutoffs 1..3. This is why the report does not claim universal per-cutoff nonregression.

An exploratory larger-case check used 128 noise trajectories and 400 resamplings of 10,000 shots per basis, with the notebook statistic and cutoff K=N:

| Case | Revised model score | Reconciled model score | Reconciled CZ / moments |
|---|---:|---:|---:|
| D(10,3) | 0.693708 | 0.760597 | 37 / 37 |
| D(10,5) | 0.565435 | 0.565435 | 73 / 73 |
| D(15,4) | 0.469810 | 0.469810 | 119 / 81 |

These harmonic simulated scores are exploratory and have sampling/model uncertainty. They are **not real Willow results**. The repaired 71-CZ D(10,5) is retained as an alternative; the calibrated ranking prefers the revised 73-CZ circuit for the tested map. A different calibration may select a different valid template. The final main function and template portfolio preserve the circuits used for these scores.

## Bonus

The user confirmed **N<=15**, overriding the notebook's inconsistent executable loop through 20. The final function returns NotImplementedError outside the requested bonus range, which the original scorer handles by skipping those cases. Both `create_dicke_state_adaptive` (notebook/root spelling) and `create_dicke_states_adaptive` (user spelling) are available.

The bonus retains root submission.py for 55 cases and replaces its 23-moment D(8,4) with the verified 21-moment revised schedule. Root alone scores 52.389323100116; the combined portfolio scores 52.521828276100.

Three cases use measurement/feedforward: D(7,3), D(10,5), and D(11,5). Full physical readout diagonals, classical-key causality, allowed native gates, and Willow couplers were checked. D(7,3) and D(11,5) have exhaustive two-branch verification; D(10,5) uses a circuit-bound Choi/channel certificate covering all 262,144 readout strings. No sampling or postselection substitutes for this proof. Controlled X/Z were converted to exactly matrix-identical PhasedXZ representations without changing depth. Cirq's ordinary Willow validator rejects classical-control wrappers, so the bonus audit validates underlying native operations and separately proves feedforward/readout correctness.

## Reproduce and evidence

```powershell
python -B reports/reconcile_audit/verify_final.py --target best_file.py --mode main --max-n 15 --seeds 101,202 --output reports/recheck_main
python -B reports/reconcile_audit/verify_final.py --target best_file.py --mode bonus --max-n 15 --output reports/recheck_bonus
python -I -B reports/check_best_isolated.py
```

Detailed final receipts: reports/reconcile_audit/final_main/, reports/reconcile_audit/final_bonus/, reports/reconcile_audit/stress_cases.json, reports/reconcile_audit/guard_cases.json, and reports/reconcile_isolation.json. Numerical repair scripts and residual histories are in reports/reconcile_audit/. Source comparisons and noise-model results are in reports/reconcile_main/. All-branch bonus certificates and the selected portfolio are in reports/reconcile_bonus/.
