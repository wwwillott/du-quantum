# DuQuantum 2026 — Google Quantum AI Dicke-state challenge

Team: Kai Sandberg, Will, Ben. 24-hour hackathon that started Sat 2026-10-24.
This file is the handoff from a claude.ai chat session; read it first.

## The task (full spec: DuQuantum_Google_Standard.ipynb)
Implement `create_dicke_state(num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors) -> (cirq.Circuit, set[GridQubit])`.
- Gates: CZ only on pairs in `cz_pauli_error_rate`; `cirq.PhasedXZGate` on qubits in `sq_pauli_error_rate`; optional mid-circuit reset. No classically conditioned gates.
- Must be an exact Dicke state when noiseless, else raise `NotImplementedError`. < 1 minute to build, ≤ 1750 moments.
- Judged on Willow hardware: 10,000 Z-basis and 10,000 X-basis shots per instance; `overall_score = 1/(Z + X)`, summed over instances.
- Bonus: `create_dicke_state_adaptive(N, n)` with mid-circuit measurement + feedforward, scored on moment count only (classically verified, not run on hardware).

## Key findings so far
Full literature review (Claude Docs): https://claude.ai/artifact/GosXrb6QLN4ryWzUiMQ7p7

Scorer analysis (analysis/score_analysis.py, analysis/trials.py; Pauli/classical noise model):
- With M = C(N,n): Z = χ²/(M·f), E[χ²] ≈ (M−1) + S·f·mean(δ²), S = 10,000, f = fraction of Z shots with correct Hamming weight, δ = relative non-uniformity in the shell.
- Large M (≳20): score ≈ f. Perfect D(10,5) median 1.01; with 2% bit flips 0.79. Hamming-weight retention is the lever.
- Small M (≤6): shot-noise lottery (1/Z heavy-tailed for M = 2, 3); systematic amplitude non-uniformity is amplified by S/M. A 2% amplitude spread roughly halves the median at N = 2–4. Exact angles + minimal circuits matter most here.
- X term is small unless dephasing is large and `max_correlator_size` is large for half-filled states (weight-N parity is GHZ-like).

Literature in one line each:
- Bärtschi & Eidenbenz 2019 (arXiv:1904.07358): O(kn) gates, O(n) depth, linear chain — the base construction.
- Bärtschi & Eidenbenz 2022 (arXiv:2207.09998): O(k log(n/k)) all-to-all, O(√(nk)) grid.
- Yuan & Zhang 2025 (arXiv:2505.15413): O(k log(n/k) + n₂) on n₁×n₂ grid; Ω(n₂) lower bound — on Willow, depth ~ patch side length.
- Wang et al. 2026 (arXiv:2608.24084): D(8,2) on a 2×4 superconducting patch, 31 CZs, fidelity 0.631; CZ compilation degraded badly with dead couplers.
- Adaptive (bonus): Buhrman et al. Quantum 2024 (LAQCC, O(1) W states), Luo/Chen/Ma 2026 (arXiv:2608.01144, probabilistic), measurement-based fan-out (arXiv:2406.07611, 2408.03064).
- Rosmanis 2026 (arXiv:2601.15171) does NOT give a Dicke circuit: in DQI he picks binomial weights so Σₖ wₖ|D_k⟩ is a product state. Doesn't help a single fixed-weight Dicke state.

## Strategy (priority order)
1. Noise/defect-aware placement on the supplied error map (no published Dicke work does this). Expected readout cost per qubit = ((N−n)/N)·e₀ + (n/N)·e₁.
2. Exploit permutation symmetry: final qubit order is irrelevant → drop order-restoring SWAPs, relabel instead of routing.
3. Gate-minimal circuits for small N (direct search for N ≤ 6).
4. Maximise f: create |1⟩s late, keep n ≤ N/2 form, minimise CZs.
5. Dynamical decoupling on idle qubits (`cirq.add_dynamical_decoupling`, `cirq_google` `DynamicalDecouplingTag`).
6. Bonus: grid divide-and-conquer with long-range steps replaced by measurement-based fan-out; mid-circuit measurements must cover whole readout diagonals (row+col const); score code sums N·n/len(circuit) over N = 2–20.
- NOT allowed: pre-distorting amplitudes to cancel readout bias (must be exact when noiseless).

## Organizer clarifications (from Kai, 2026-10-24)
- Hardware instances: N up to what they can track with statevector; n ≤ N/2 (n ≥ 1).
- Bonus: N ≤ 15 only (not 20).
- Circuits not exact without noise are discarded entirely.
- Submitted circuit must end without measurements; the 1750-moment limit is checked before they append basis-change/measurement gates.
- Submission cutoff: Sunday 2026-10-25 morning.

## Open questions (ask organizers)
- Which (N, n) instances and which `max_correlator_size` are used for hardware scoring?
- Bonus: text says N ≤ 15 and one point per instance; code loops N = 2–20 and sums N·n/len(circuit).
- Notebook bug: `test_circuit` has `keep_indices = range(qubits)` (a set → TypeError); use `range(len(qubits))`.

## Files
- DuQuantum_Google_Standard.ipynb — official problem notebook (unmodified).
- willow_graph.ipynb — `plot_willow(...)` draws available qubits / allowed CZ pairs in spec-sheet layout; `to_networkx(...)` builds the hardware graph with error attributes; examples incl. fake calibration, readout lines, greedy patch.
- data/willow105_layout.json — 105 qubits / 182 couplers parsed from Cirq's `willow_pink` device spec (what `cirq_google.Willow105` is built from). Useful for checks without cirq.
- analysis/score_analysis.py, analysis/trials.py — Monte-Carlo of the official scorer (numpy only; run trials.py from inside analysis/).
- dicke.py — baseline `create_dicke_state` (see Baseline below). test_dicke.py — exactness/validity/timing checks (`python test_dicke.py [max_N]`).
- PROMPTS.md — log of prompts given to AI (the challenge requires submitting these). Keep appending.

## Environment
`pip install --pre cirq cirq-google matplotlib networkx` — the `--pre` matters: `cirq_google.Willow105` only exists in the dev builds (tested with cirq 1.8.0.dev20261023221525); stable 1.7.0 does not have it. willow_graph.ipynb runs cleanly against it, and data/willow105_layout.json matches `Willow105.metadata` exactly.

## Baseline (done, 2026-10-24): dicke.py
Nearest-neighbour Bärtschi–Eidenbenz on a hardware path. `create_dicke_state` is implemented and exact.
- Construction: start |0^{N-k}1^k⟩; SCS_m is a left-to-right sweep over pairs (j, j+1). In sorted unary the "hole" left of the 1-block is locally identifiable, so:
  - first pair of each window: Givens 01 → √(l/m)·01 + √(1−l/m)·10 (2 CZ);
  - later pairs: 3-qubit gate on consecutive (c, a, b) = Givens if c=0, SWAP if c=1, built as G(φ₂)·CZ(c,a)·G(φ₁)·Z_c with φ₁+φ₂ = γ, φ₁−φ₂ = π/2 (5 CZ). Exact on the only inputs that occur (000, 001, 011, 101, 111). A numerical search found no 4-CZ version and needed 2 CZ for the Givens.
- n > N/2 builds D(N, N−n) and flips all qubits at the end. Placement: beam search for a path minimising Σ(CZ uses per edge × CZ error) + readout cost (`best_path`).
- CZ counts: W state 2(N−1); D(8,2) 44 (Wang et al. got 31); D(10,5) 148; D(16,8) 415. Depth ≈ 4N for W (sequential, no pipelining benefit).
- `cirq.optimize_for_target_gateset` gave zero CZ savings and was 10× slower, so compile only merges 1q gates to PhasedXZ and stratifies 1q/2q moments.
- test_dicke.py: all (N, n) with N ≤ 12 exact (fidelity 1 to 1e-12), pass `Willow105.validate_circuit`, use only allowed CZ pairs / PhasedXZ on allowed qubits; N ≤ 40 builds in < 10 s and stays under 1750 moments.

## Superseded as the main line (2026-10-24 ~16:40)
A parallel session ("Willow chip optimization competition", owns dicke_core.py, dicke_main.py, build_submission.py → submission.py, tools/, data/templates/, logs/) has a stronger exact pipeline: pruned B&E with the same NN gadget idea, a broadcast-tree W (2N−3 CZ), template library, ALAP + DD, and a trajectory noise sim + fast scorer in tools/sim.py. The Codex "Astra" team works in astra/ (owns bonus/, notes/). Same calibration (test_dicke.fake_calibration(seed=1)), CZ count dicke.py → submission.py: D(4,2) 16→5, D(6,3) 45→35, D(8,2) 44→40, D(8,4) 89→70, D(10,5) 148→117, D(12,6) 222→176. Keep dicke.py only as an independent cross-check; don't edit the other sessions' files.

Open items that session listed as not yet covered: (1) 2D / divide-and-conquer construction with lower depth for N ≥ 10; (2) DD / noise-model ablations.

## Main line status (Willow chip optimization session, 2026-10-24 ~19:30)
- Submission: `python build_submission.py` -> submission.py (inlines dicke_core.py + dicke_main.py + data/templates + data/peep). Test: `python tools/run_submission.py SEED NMAX NTRAJ`; head-to-head vs another file: `python tools/head2head.py SEED NMAX`.
- Constructions: W broadcast tree (2N-3 CZ); numerically synthesized small templates; support-pruned NN B&E with 5/4/3-CZ gadgets (4-CZ when input 111 unreachable); `dc2_template` divide&conquer (hypergeometric unary seed + 2 parallel B&E blocks).
- General optimizer: tools/peephole.py = state-aware peephole re-synthesis (window only needs to map the Schmidt vectors of its actual input state; prune CZs; splice). Outputs data/peep/*.json, auto-included by the build.
- Bonus lives in bonus/ (Astra). astra/ is an independent competing team — do not read/copy their code.
