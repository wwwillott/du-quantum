# Experimental demonstrations of Dicke / W-state preparation on real hardware

Notation: D(N,k) = N qubits, Hamming weight k; W_N = D(N,1). "2q" = two-qubit gates.
Values marked **[unverified]** come from prior knowledge of the paper and were not re-checked against the primary text in this session; treat them as approximate until checked.

## Q1. Demonstrations by platform, with the largest states and their fidelities

### Takeaway
Best certified gate-based Dicke results today: superconducting D(8,2) at F = 0.736 (Zuchongzhi-architecture chip, AshN gates, 23 entanglers) or 0.631 (CZ-only, 31 CZ); superconducting D(10,5) at F ≈ 0.26 (IQM Emerald, 84 2q gates, symmetry-based DFE); trapped-ion D(10,5) at a fidelity lower bound ≥ 0.46 (Quantinuum H1-2, about 85 CNOT). Photonic states peak at six qubits (D(6,3), F = 0.56–0.65). Atomic-ensemble and BEC "Dicke" states are far larger (10^3–10^4 atoms), but they are certified only through entanglement depth, not fidelity.

### Cited Findings

**Dense table (gate-based and photonic)**

| Year | Platform / group | State | Method | # 2q gates | Fidelity | Certification | Source |
|---|---|---|---|---|---|---|---|
| 2026 | Superconducting, chip "with the same hardware architecture as Zuchongzhi 3.0" (105 transmons, 182 couplers); 2×4 patch | D(8,2) | B&E-type short-depth circuit for 2×4 lattice, recompiled with Mirror-SABRE; CZ-only vs AshN native set (CZ, iSWAP, √iSWAP-family, SWAP, B-like) | 31 CZ (CZ scheme); 23 entanglers (AshN) | 0.631 ± 0.001 (CZ); 0.736 ± 0.001 (AshN) | Full QST (all 3^8 Pauli bases, 3584 shots each), linear inversion, readout-corrected; also ⟨J_z²⟩, ⟨J_x²+J_y²⟩, PPT-mixture GME witness | [Wang et al., arXiv:2608.24084](https://arxiv.org/pdf/2608.24084) |
| 2026 | Superconducting, IQM Emerald (54 q, via Amazon Braket) | D(5,2), D(6,3), D(8,3), D(10,5) | Collision model (shuttle ancillas, partial-SWAP collisions + phase layer), compared against "published" literature circuits | 21–936 native 2q | D(5,2) 0.714 (21 gates); D(6,3) 0.525–0.544 (33–51 gates); D(8,3) 0.446 (51 gates); D(10,5) 0.261 ± 0.080 (84 gates); the 201-gate "published" D(10,5) arm gave −0.001 | Permutation-symmetry direct fidelity estimation (Pauli X/Y/Z terms grouped into symmetry orbits; 24–84 circuits instead of 3^N); per-qubit readout correction | [Vu et al., arXiv:2608.30176](https://arxiv.org/html/2608.30176v1) |
| 2022–25 | Trapped ion, Quantinuum H1-2 (12 q) | All D(N,k), N ≤ 10, k ≤ N/2 | Divide-and-conquer B&E on LNN; 5K(N−K) − 3(N+K) + 5 CNOT (≈ 85 for D(10,5)) | up to ~85 CNOT | Lower bound 0.46 for D(10,5) (emulator predicted 0.22) | Somma et al. lower bound from only 3 global settings (X, Y, Z); upper bounds from measured success probability and Hellinger fidelity | [Aktar et al., arXiv:2210.03048](https://arxiv.org/pdf/2210.03048) |
| 2022 | Superconducting, IBM Q Sydney / Montreal (heavy-hex) | D(n,k), n ≤ 6 | Divide-and-conquer + improved B&E unitaries on LNN | D(4,2) 10 CNOT; D(6,3) 32–33 CNOT | D(4,2) 0.87 (previous best on IBM 0.53) | Full state tomography, maximum likelihood; with/without readout mitigation | [Aktar et al., arXiv:2112.12435 / IEEE TQE 2022](https://arxiv.org/pdf/2112.12435) |
| 2020 | IBM ibmqx2 | D(4,2) | Mukherjee–Maitra–Gaurav–Roy deterministic circuit (12 CNOT, depth 21) | 12 CNOT | 0.53 | Tomography | [Mukherjee et al., arXiv:2007.01681](https://arxiv.org/pdf/2007.01681); fidelity and CNOT figures as quoted in [Aktar 2022](https://arxiv.org/pdf/2112.12435) |
| 2018–19 | IBM | W_2 … W_5 | Cruz et al. log-depth / linear W circuits | O(n) | 0.95 (W_2) down to 0.75 (W_5) | Tomography | [Cruz et al., arXiv:1807.05572, Adv. Quantum Technol. 2, 1900015](https://arxiv.org/abs/1807.05572), values as quoted in [Aktar 2022](https://arxiv.org/pdf/2112.12435) |
| 2025 | IBM ibm_quebec | W_3 (resource state) | Gate circuit | small | 0.958 ± 0.007 (readout-mitigated) | Fidelity estimate | [arXiv:2508.09110 (random party distillation)](https://arxiv.org/pdf/2508.09110) |
| 2010 | Superconducting phase qubits (UCSB, Martinis group) | W_3 (and GHZ_3) | Analog: resonant swap interaction on simultaneously coupled qubits, timed so the excitation is shared equally | n/a (analog) | **[unverified]** ≈ 0.78 (W), ≈ 0.62 (GHZ) | Full QST + entanglement witnesses | [Neeley et al., Nature 467, 570 (2010), arXiv:1004.4246](https://arxiv.org/abs/1004.4246v1) |
| 2005 | Trapped ions, ⁴⁰Ca⁺ (Innsbruck) | W_4 … W_8 | Deterministic laser-pulse sequences on a linear Paul-trap string | n/a (sideband pulses) | **[unverified]** W_8 ≈ 0.72 | Full QST; the 8-ion density matrix took more than 650,000 measurements | [Häffner et al., Nature 438, 643 (2005)](https://www.nature.com/articles/nature04279); [Innsbruck summary](https://www.uibk.ac.at/en/exphys/research/quantumoptics/news/scalable-multiparticle-entanglement-of-trapped-ions/) |
| 2004 | Trapped ions (Innsbruck) | W_3 | Laser pulses | — | **[unverified]** ≈ 0.83 | QST | Roos et al., Science 304, 1478 (2004), doi:10.1126/science.1097522 **[unverified]** |
| 2009 | Photonics (Zeilinger/Walther, Vienna) | D(6,3); projected D(4,·), D(5,·), GHZ_4, W_4 | SPDC multi-photon source + linear optics | n/a | 0.56 ± 0.02 | Fidelity + genuine 6-qubit entanglement witness (violated by 4σ) | [Prevedel et al., PRL 103, 020503](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.020503) |
| 2009 | Photonics (Weinfurter, Munich) | D(6,3) | SPDC + linear optics | n/a | 0.654 ± 0.024 | Two-setting witness, −0.422 ± 0.148 | [Wieczorek et al., PRL 103, 020504](https://link.aps.org/doi/10.1103/PhysRevLett.103.020504) (values from search snippet; full text not opened) |
| 2007 | Photonics (Munich) | D(4,2) | SPDC + linear optics | n/a | 0.844 ± 0.008 | Full QST | [Kiesel et al., PRL 98, 063604, arXiv:quant-ph/0606234](https://arxiv.org/pdf/quant-ph/0606234) |

**Wang et al. 2026 (arXiv:2608.24084) in detail**
- Authors: Hanyi Wang, …, Feng Wu, Hantao Sun, Jianxin Chen (20 authors; affiliations include the Quantum Research Institute of China Telecom, HKUST and Zhongguancun Lab); dated August 26, 2026 — [arXiv:2608.24084](https://arxiv.org/pdf/2608.24084)
- Device: "a superconducting quantum processor with the same hardware architecture as Zuchongzhi 3.0, comprising 105 frequency-tunable transmon qubits and 182 tunable couplers" (the same qubit and coupler counts as Willow105) — [arXiv:2608.24084 SM §I](https://arxiv.org/pdf/2608.24084)
- Circuit: based on a CZ circuit from their ref. [13], "a short-depth construction for a perfect 2×4 rectangular lattice optimized for nearest-neighbor interactions"; recompiled per topology with Mirror-SABRE — [SM §V](https://arxiv.org/pdf/2608.24084)
- Topologies: eight in total (five 2×4 variants and three 3×3 variants), including Defect(G25), Defect(G12/G46), Defect(G13/G25/G47) and Defect(G13/G25/G68) — [SM §V, Table S9](https://arxiv.org/pdf/2608.24084)
- Defect-free 2×4 results: CZ scheme F = 0.631 ± 0.001, ⟨J_z²⟩ = 3.79 ± 0.04, ⟨J_x²+J_y²⟩ = 12.39 ± 0.13. AshN scheme F = 0.736 ± 0.001, ⟨J_z²⟩ = 3.91, ⟨J_x²+J_y²⟩ = 13.64 ± 0.13. Ideal values are 4 and 16. AshN needed "only 23 entangling gates compared with 31 for the CZ-based realization" — [main text](https://arxiv.org/pdf/2608.24084)
- GME witness Tr(W·ρ), where negative values certify GME: AshN −0.0344 (perfect), −0.0140, −0.0360, −0.0089, −0.0119 across the defect configurations; CZ +0.0593 (perfect), then +0.0997, +0.1383, +0.1441, +0.1945. CZ never certified GME. "The fidelity of the CZ-prepared Dicke state deteriorates rapidly as defective couplers are introduced"; with AshN, three defective couplers cost only about 0.04 in fidelity — [main text, Table S9](https://arxiv.org/pdf/2608.24084)
- Hardware numbers on the 2×4 patch: CZ XEB errors 0.353–0.860% per coupler (average 0.550%); other AshN gates average 0.615%. T1 33–57 µs. T2-Ramsey only 5.2–21.5 µs. Readout error for |0⟩ is 0.5–1.9%, for |1⟩ 0.9–2.2%. Gate durations: CZ 48 ns, iSWAP 44 ns, SWAP 60 ns — [SM Tables S7, S8](https://arxiv.org/pdf/2608.24084)
- Mitigation: tensor-product single-qubit readout-error correction (matrix inversion) on all results; no post-selection reported — [SM §V](https://arxiv.org/pdf/2608.24084)
- Compilation study (the same paper, Bärtschi–Eidenbenz Dicke circuits among the benchmarks): D(8,2) CZ-compiled on constrained topologies has 2q-count ratios of about 1.4–2.2× the all-to-all count; overall geometric-mean overhead is 1.65× (count) and 1.86× (depth) for CZ, versus 1.13× and 1.24× for AshN — [SM Table S2](https://arxiv.org/pdf/2608.24084)

**Vu et al. 2026, IQM Emerald (arXiv:2608.30176)**
- Device calibration: 2q gate error 0.153–8.206% (median 0.349%), median T1 54.7 µs, median T2 16.6 µs. One session, 88,656 shots — [arXiv:2608.30176](https://arxiv.org/html/2608.30176v1)
- Full table (F_unconditional / P_K / F/P_K, where P_K = probability of the correct Hamming weight): D(5,2) 21 gates: 0.714 / 0.825 / 0.866. D(5,2) 465 gates: 0.087 / 0.363 / 0.238. D(6,3) 33 gates: 0.525 / 0.668 / 0.786. D(8,3) 51 gates: 0.446 / 0.652 / 0.684. D(8,3) 114 gates: 0.179 / 0.444 / 0.402. D(10,5) 84 gates: 0.261 / 0.501 / 0.521. D(10,5) 201 gates: −0.001 / 0.227 / −0.006 — [arXiv:2608.30176](https://arxiv.org/html/2608.30176v1)
- Error attribution: P_K decays steeply with depth, attributed to decoherence plus 2q errors. A fit gives an effective 2q gate time of 100 ns; an alternative fit needs calibrated gate errors scaled by λ = 2.4 — [arXiv:2608.30176](https://arxiv.org/html/2608.30176v1)

**Aktar et al., IBM 2022 (arXiv:2112.12435)**
- CNOT counts (LNN) for the states run: D(2,1) 1, D(3,1) 3, D(4,1) 5, D(5,1) 7, D(6,1) 9, D(4,2) 10, D(5,2) 17, D(6,2) 24, D(6,3) 32/33 — [arXiv:2112.12435 Figs. 11–12](https://arxiv.org/pdf/2112.12435)
- Their D(4,2) circuit uses 10 CNOT at depth 11, against the previous 12 CNOT at depth 21 — [arXiv:2112.12435](https://arxiv.org/pdf/2112.12435)
- Average readout error was 0.0508 on Sydney and 0.0203 on Montreal — [arXiv:2112.12435 §4.3](https://arxiv.org/pdf/2112.12435)

**Aktar et al., Quantinuum 2022–25 (arXiv:2210.03048)**
- Dicke circuits ran on the 12-qubit H1-2; GHZ circuits ran up to G_20 on the 20-qubit H1-1, with a G_20 bound ≥ 0.73. The authors claim their bounds "match or exceed exact fidelity records recently achieved on IBM's superconducting systems for much smaller states |D_6^3⟩ and |G_5⟩" — [arXiv:2210.03048](https://arxiv.org/pdf/2210.03048)
- Shots were chosen as max(150, …) per setting so that the 68% CI half-width is ≤ 0.1. Hardware bounds came out better than the H1-2E emulator's (0.46 vs 0.22 for D(10,5)) — [arXiv:2210.03048](https://arxiv.org/pdf/2210.03048)
- For large D(N, N/2), the authors argue an approximate GHZ-based or product-state preparation can score better on fidelity bounds. This is not relevant to us, because exactness is required — [arXiv:2210.03048](https://arxiv.org/pdf/2210.03048)

**Neutral atoms / ensembles** (no gate-based Dicke circuits found)
- Searches found no Rydberg-tweezer-array experiment reporting a Dicke-state or W-state fidelity. The closest hit is a 2026 Rydberg-ring optimal-control preprint on symmetric states, which looks theory-oriented — [search result arXiv:2604.18457](https://arxiv.org/html/2604.18457v1)
- Ensemble results, all **[unverified]**: Haas et al., Science 344, 180 (2014), a heralded W-like state of 41 atoms in a fiber cavity. McConnell et al., Nature 519, 439 (2015), a heralded W-like state of about 3,000 atoms with a negative Wigner function. Lücke et al., PRL 112, 155304 (2014), a twin-Fock (D(N, N/2)-like) BEC state of about 8,000 atoms, certified by entanglement depth from ⟨J_z²⟩ and ⟨J²⟩. These are certified only through entanglement depth or witnesses; no per-state fidelity is given.

### Inferences
- Largest certified gate-based Dicke state per platform: superconducting D(10,5) at F ≈ 0.26 (IQM) and D(8,2) at F = 0.74 (Zuchongzhi-type); trapped ions D(10,5) at F ≥ 0.46 (H1-2); photonics D(6,3) at F = 0.56–0.65. The largest W states are W_8 for ions (2005), W_5 for IBM superconducting, and W_3 for analog superconducting. No published Google Sycamore or Willow Dicke-state demonstration turned up in the searches.
- Effective fidelity per 2q gate (F^(1/#2q), lumping in readout, idle and 1q errors): Wang CZ 0.631^(1/31) ≈ 0.985; Wang AshN ≈ 0.987; IBM D(4,2) 0.87^(1/10) ≈ 0.986; IQM D(10,5) 0.261^(1/84) ≈ 0.984; H1-2 D(10,5) bound 0.46^(1/85) ≈ 0.991. So across platforms, about 1.5% of fidelity is lost per 2q gate. That is 3–5× the quoted per-gate XEB or RB error, which points to idle decoherence (T2-Ramsey of 5–20 µs) and readout as large hidden contributors.

### Gaps
- Exact fidelities for Häffner W_8, Neeley W_3 and Roos W_3 were not re-verified (paywall, or abstract only).
- Hume et al. 2009 (NIST, "Preparation of Dicke states in an ion chain") was not found in this session.
- No NMR Dicke or W demonstration was found or verified.
- No Zuchongzhi- or Rigetti-specific W-state paper was found. No Google Sycamore or Willow Dicke paper was found.
- Wang et al.'s ref. [13] (the 2×4 CZ circuit) could not be resolved from the supplement text; it is most likely Bärtschi & Eidenbenz 2022, arXiv:2207.09998.
- No experimental Dicke prep with mid-circuit measurement and feedforward was found (relevant to the bonus).
- No Rydberg-array Dicke-state fidelity was found.
- The exact numeric D(6,3) fidelity on IBM Montreal (Aktar 2022) is shown only in figures that did not extract as text.

## Q2. How fidelity was certified, what errors dominated, and Dicke-specific mitigation

### Takeaway
Certification has moved from full QST (3^N settings: Kiesel, Häffner, Aktar-IBM, Wang) to symmetry-exploiting estimators: Somma lower bounds from 3 global settings, permutation-orbit DFE, and collective-spin observables plus PPT witnesses. The dominant error channel is leaving the Hamming-weight shell. P_K (shell retention) falls fastest with 2q count, and the post-selected ratio F/P_K stays much higher. Readout (|1⟩ errors about 2× |0⟩ errors), routing or compilation overhead, and dead couplers are the other clear contributors. The only mitigation used in practice is readout-matrix inversion. Hamming-weight post-selection appears as a diagnostic (F/P_K) but is not reported as a fidelity.

### Cited Findings
- Somma-type lower bound: S_Jz (computational-basis projector onto the j_z eigenvalue, which equals the measured success probability) plus S_J² = (J² − N(N+2))/(4N), measured in X, Y and Z (three settings). The Hellinger fidelity and measured success probability are upper bounds, and the gap between them and the true fidelity grows with CNOT count — [Aktar et al., arXiv:2210.03048](https://arxiv.org/pdf/2210.03048)
- On IBM, "Hellinger fidelity and measured success probability are good approximations for quantum fidelity, particularly up to 4 qubits and 10 CNOTs; unfortunately, the gap becomes larger for larger circuits" — [Aktar et al. 2022](https://arxiv.org/pdf/2112.12435)
- Fidelity falls "largely linear[ly]" with CNOT count. Noise-adaptive compilation is a notable exception because it "sometimes drastically increases CNOT counts." Readout mitigation adds about +0.1 absolute fidelity regardless of circuit complexity, with the biggest relative gain at large n and small k. Montreal was more stable than Sydney — [Aktar et al. 2022](https://arxiv.org/pdf/2112.12435)
- Wang: full QST over 3^8 bases × 3584 shots with an unbiased linear estimator, then readout inversion. Dicke quality was quantified with ⟨J_z²⟩ (shell and population structure) and ⟨J_x²+J_y²⟩ (coherence), and a Bergmann–Gühne PPT-mixture witness W_{N,k} = Σ_{i=0}^{2k} Π_i − |D⟩⟨D| — [arXiv:2608.24084 SM](https://arxiv.org/pdf/2608.24084)
- Wang: the deficit in ⟨J_x²+J_y²⟩ (12.39 vs 16 for CZ) is much larger in relative terms than the deficit in ⟨J_z²⟩ (3.79 vs 4). Coherence loss dominates over shell leakage in that metric. The fidelity gap between AshN and CZ is attributed to compilation overhead (23 vs 31 entanglers), even though each AshN gate is slightly noisier (0.615% vs 0.550%) — [arXiv:2608.24084](https://arxiv.org/pdf/2608.24084)
- IQM: P_K (shell retention) ranges from 0.82 (21 gates) down to 0.23 (201 gates), while F/P_K stays around 0.87 for the shallow D(5,2). Deep literature circuits (465–936 gates, from naive compilation) gave near-zero fidelity — [arXiv:2608.30176](https://arxiv.org/html/2608.30176v1)
- Analog W-state preparation, for example by adiabatic passage, is "quite sensitive to the dephasing" in prior superconducting proposals — [Sci. Rep. 6, 36737](https://www.nature.com/articles/srep36737)
- Dynamical decoupling: no Dicke experiment found in this search reported using DD.

### Inferences
- For the hackathon scorer, P_K is exactly the f term (Z-basis shell fraction). Hardware data say P_K decays faster with gate count than coherence quality, so the best way to raise f is minimising 2q count and duration. IQM's two D(6,3) arms illustrate this: 33 vs 51 gates gave P_K 0.668 vs 0.672 (within noise), but at 114 vs 51 gates for D(8,3), P_K dropped from 0.65 to 0.44.
- Readout asymmetry (|1⟩ error about 2× |0⟩ on Zuchongzhi-type hardware) supports the k ≤ N/2 form, and supports placing excitations on qubits with low e₁.
- Wang's dead-coupler result is direct evidence that routing around defects with CZ-only SWAPs is expensive. On Willow (CZ-only), choosing a defect-free patch or path and relabeling instead of routing should matter more than any gate-level trick.

### Gaps
- No Dicke-specific use of symmetry verification, Hamming-weight post-selection as an actual reported mitigation, or DD was found in an experimental paper. These remain untested in the literature.
- No Willow-generation (sub-0.3% CZ) Dicke data exist to calibrate expected P_K for D(10,5)-scale circuits.

## Q3. What fidelity-vs-N trends say about which construction to prefer on noisy hardware

### Takeaway
Across IBM, IQM, Zuchongzhi-type and Quantinuum data, fidelity tracks the compiled 2q count (and depth) almost exponentially, at roughly 1.5% per 2q gate. Constructions that reduce compiled (not logical) 2q count on the actual connectivity always win: divide-and-conquer B&E on LNN (IBM), collision or shallow arms (IQM; 2–2.4× fewer gates turned F from ≈0 into 0.26–0.45), and richer native gates (Zuchongzhi: 31 → 23 gates, 0.63 → 0.74).

### Cited Findings
- On IQM, the same target with 84 vs 201 2q gates gave F 0.26 vs 0.00 for D(10,5), and 51 vs 114 gates gave F 0.45 vs 0.18 for D(8,3) — [arXiv:2608.30176](https://arxiv.org/html/2608.30176v1)
- On IBM, D(4,2) went from F = 0.53 at 12 CNOT, depth 21 to F = 0.87 at 10 CNOT, depth 11 (the latter also on newer hardware and with readout mitigation) — [Aktar 2022](https://arxiv.org/pdf/2112.12435)
- On Zuchongzhi-type hardware, eliminating routing overhead (23 vs 31 entanglers) raised F by 0.105 and made the circuit robust to up to three dead couplers. CZ fidelity degraded rapidly with defects — [arXiv:2608.24084](https://arxiv.org/pdf/2608.24084)
- On Quantinuum H1-2, the fidelity lower bound decreases with CNOT count, while the upper bounds (measured success probability and Hellinger) decrease more slowly — [arXiv:2210.03048](https://arxiv.org/pdf/2210.03048)
- When compiled to constrained 1D or 2D CZ topologies, Bärtschi–Eidenbenz Dicke circuits incur about 1.4–3× 2q overhead relative to all-to-all, and the overhead grows with N (e.g., for n = 64, k = 21) — [arXiv:2608.24084 SM Table S2 and large-circuit table](https://arxiv.org/pdf/2608.24084)

### Inferences
- On Willow (CZ only, nearest-neighbour grid), prefer constructions native to the grid that need no SWAP routing: the NN-gadget B&E path, divide-and-conquer with blocks on adjacent rows, or a broadcast-tree W. Choose the patch to avoid bad couplers rather than routing around them.
- Depth matters as well as count, given T2-Ramsey of 5–20 µs on comparable transmon chips. Parallel divide-and-conquer blocks (Aktar; Yuan & Zhang grid bounds) should beat a single long B&E chain at N ≥ 8 when the 2q count is similar.
- Expected order of magnitude for Willow (inference only): with an effective ~1–1.5% loss per CZ, a 117-CZ D(10,5) circuit implies P_K perhaps 0.3–0.5. Every 10 CZs saved is worth roughly a 10–15% relative gain in f.

### Gaps
- There is no head-to-head hardware comparison of linear B&E vs divide-and-conquer vs grid-optimal (Bärtschi–Eidenbenz 2022, Yuan–Zhang 2025) constructions at fixed N on the same chip.
- There is no published hardware study of fidelity vs N at fixed k for N > 10 on any gate-based platform.
