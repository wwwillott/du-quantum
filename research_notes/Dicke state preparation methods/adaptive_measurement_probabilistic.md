# Adaptive / measurement-based / probabilistic preparation of Dicke and W states

Scope: methods that use mid-circuit measurement, classical feedforward, ancillas, or post-selection to prepare W and Dicke states. Purely unitary constructions appear only as pointers. Context: the hackathon bonus scores an **exact** D(N,n), N ≤ 15, built with mid-circuit measurement plus feedforward on a square grid. It is judged on **moment count only**, and the circuit is classically verified.

Labels used throughout:
- **E-det**: exact and deterministic.
- **E-post**: exact only when a heralded outcome occurs (post-selection or repeat-until-success).
- **Approx**: approximate.

Only E-det methods (feedforward corrections, no discarded branches) can satisfy the bonus as stated.

Note on the brief's citations: arXiv:2406.07611 is not a fan-out paper. It is a readout-error-mitigation paper for mid-circuit measurement and feedforward. The measurement-based fan-out paper is arXiv:2408.03064 (Bäumer & Woerner). Both are covered below.

---

## Q1. Buhrman, Folkertsma, Loff, Neumann et al., "State preparation by shallow circuits using feed forward" (LAQCC), Quantum 8, 1552 (2024), arXiv:2307.14840

### Takeaway
LAQCC is a model with constant quantum depth on a nearest-neighbour grid, plus intermediate measurements whose outcomes feed a log-depth classical computation that controls later gates. In it the authors give:
- an exact, deterministic, constant-depth W state on an n × log n grid of O(n log n) qubits;
- an exact, deterministic, constant-depth Dicke state for k = O(√n), using O(n² log n) qubits;
- a log-depth (LAQCC-LOG) Dicke state for any k, using poly(n) qubits.

The constructions are asymptotic and have large constants. The authors' own n = 4, k = 2 "filling" example alone uses 24 qubits.

### Cited Findings
- **Model.** Qubits sit on a grid with nearest-neighbour interactions. Circuits have constant depth with intermediate measurements, and a classical controller runs log-depth computations on the outcomes to control later operations. This gives long-range interactions that plain constant-depth circuits cannot reach. — [arXiv:2307.14840 abstract](https://arxiv.org/abs/2307.14840)
- **Venue.** Quantum 8, 1552 (2024), DOI 10.22331/q-2024-12-09-1552. The latest arXiv version is v5 (4 Dec 2024). — [arXiv:2307.14840](https://arxiv.org/abs/2307.14840)
- **Uniform superposition over q values (Thm 4.1).** This is the basic primitive.
  - Prepare all 2^⌈log q⌉ index values.
  - Flag the "good" values y < q with a Greaterthan gate, which needs O(log² q) qubits.
  - Because the good fraction is known and ≥ 1/2, an exact amplitude-amplification step (Lemma 3.14, Exact₀ gate plus Hadamards) with O(1) iterations makes it deterministic.
  - Result: depth O(1), O(⌈log₂ q⌉²) qubits, works for any q. — [arXiv:2307.14840v5 HTML](https://arxiv.org/html/2307.14840v5)
- **W state (Thm 4.5), "uncompress, then compress".**
  1. Put a log n-qubit index register in the uniform superposition (1/√n) Σᵢ |i⟩.
  2. *Uncompress*: fan out the index bits to copies placed next to each system qubit. Use parallel Equalᵢ gates to flip system qubit i when the index equals i, giving Σᵢ |i⟩|eᵢ⟩. Then fan out again to clean up the copies.
  3. *Compress*: erase the index register, using the one-hot system register as control. This uses Hadamards, fan-out, CZ and fan-out again (phase kickback).
  - Result: exact, deterministic, O(1) depth, O(n log n) qubits on an n × log n grid. Each index copy sits next to its column of system qubits. — [arXiv:2307.14840v5 HTML](https://arxiv.org/html/2307.14840v5)
- **Fan-out is the only source of "long range" in LAQCC.** It is built from measurement plus Pauli corrections (Clifford-ladder / teleportation style), and fan-out is its own inverse. Parity is fan-out conjugated by Hadamards. — [arXiv:2307.14840v5 HTML](https://arxiv.org/html/2307.14840v5)
- **Comparison with older LOCC work.** The authors say the earlier W-state circuit of Piroli–Styliaris–Cirac (2021) needs sequential measurement and correction rounds, so it is linear-depth in this model. — [arXiv:2307.14840v5 HTML](https://arxiv.org/html/2307.14840v5)
- **Dicke state for small k, k = O(√n) (Section 4.3, Thm 4.10).**
  1. *Filling*: run Uncompress on k index registers in parallel. This gives a superposition over all k-tuples of positions.
  2. *Filtering*: tuples with a repeated index are "birthday collisions" (two 1s on the same qubit cancel). Flag them, then remove them with exact boosting (Lemma 3.14). The good amplitude satisfies |α|² > e^(−2k²/n), a constant when k = O(√n). The paper says this makes the protocol deterministic.
  3. *Ordering*: compare index registers pairwise with Greaterthan gates and **measure** the comparison ancillas. The outcomes fix a permutation, which is then applied.
  4. *Cleaning*: compute Hamming weights of system-register suffixes and uncompute each index register with Hadamard-basis phase kicks.
  - Result: O(n² log n) qubits, O(1) depth. It fails for larger k because of the birthday paradox. The lemma statement and proof give inconsistent qubit counts. — [arXiv:2307.14840v5 HTML](https://arxiv.org/html/2307.14840v5)
- **Dicke state for any k (Section 4.4, Thm 4.21).**
  - Prepare a uniform superposition over factoradic numbers.
  - Map it to weight-k bitstrings with a logspace conversion (factoradic to combinatorial number system). This conversion is in TC¹, hence LAQCC-LOG.
  - Uncompute the label with the inverse map, which is in TC⁰.
  - Result: poly(n) qubits, O(log n) depth. — [arXiv:2307.14840v5 HTML](https://arxiv.org/html/2307.14840v5)
- **Cost accounting.** Luo–Chen–Ma point out that LAQCC's Table 5 assigns width O(n log n) to exact Hamming-weight gates, but the cited Takahashi–Tani lemma gives O(n²). — [arXiv:2608.01144v1 HTML](https://arxiv.org/html/2608.01144v1)
- **No concrete gate counts.** The paper gives asymptotic bounds only, with no explicit gate counts or constant factors. The Fig. 7 filling example for n = 4, k = 2 uses 24 qubits: 16 index qubits plus 8 system qubits. — [arXiv:2307.14840v5 HTML](https://arxiv.org/html/2307.14840v5)
- **Also in the paper.** Many-body scar states (Dicke-like, with a Fibonacci "no adjacent 1s" constraint) for k = O(√n), using O(n² log n) qubits and a filter-and-boost step. — [arXiv:2307.14840v5 HTML](https://arxiv.org/html/2307.14840v5)

### Inferences
- **The W-state idea is the most transferable piece for the bonus.** Make a small binary index register in an exact uniform superposition over N values. Decode it to one-hot. Then erase the index by **measuring it in the X basis and fixing the resulting phases with Z gates on the system qubits**: since the system is one-hot, the phase (−1)^{s·i} is a function of which qubit holds the 1. That is an exact, deterministic disentangling step costing one measurement moment plus one feedforward moment. This mirrors the paper's "phase kicks in the Hadamard basis" cleanup, but the explicit measure-and-correct form is my reading, not a quoted construction.
- **Uniform superposition over N values for N ≤ 15 needs at most 4 index qubits** and can be made exactly with plain rotations (Bärtschi-style or controlled-rotation splitting). The LAQCC "deterministic via Greaterthan + boosting" machinery is only needed for the asymptotic constant-depth claim.
- **The small-k Dicke route is too ancilla-heavy for a 105-qubit chip at N = 15, n ≤ 7.** It needs k index registers of ~4 qubits plus a fan-out copy per system qubit, pairwise comparators, and Hamming-weight cleanup. Also k = 7 is not "≪ √15", so collision filtering would have a modest success amplitude to amplify. Expect it to lose to a well-pipelined unitary circuit on moment count.

### Gaps
- No explicit moment, CZ or ancilla counts for concrete small instances (N ≤ 15). The paper is asymptotic.
- The exact number of measurement rounds per protocol is not stated. The fan-out implementation choice (Clifford ladder) determines it.

---

## Q2. Luo–Chen–Ma (arXiv:2608.01144), and arXiv:2406.07611 / arXiv:2408.03064

### Takeaway
- **Luo–Chen–Ma** give an exact but **probabilistic** (success ≥ 1/k) constant-depth adaptive Dicke protocol with O(n² + k² log² n + kn log n log log n) ancillas, assuming all-to-all connectivity. It is E-post, so it does not directly meet a deterministic exactness requirement.
- **Bäumer & Woerner (arXiv:2408.03064)** give the practical constant-depth, measurement-based fan-out and long-range CNOT on a **1D line**: about 5 CNOT layers, n ancillas, one measurement round and one feedforward round. This is the realistic building block for grid hardware.
- **arXiv:2406.07611** is a readout-error-mitigation scheme for mid-circuit measurements with feedforward, not a state-preparation method.

### Cited Findings
**Luo, Chen, Ma, "Constant-depth adaptive preparation of Dicke and symmetric states" (arXiv:2608.01144, 2 Aug 2026)**
- **Primitive: the uniform subset superposition (USS).** This is a superposition over all k-subsets of {0,…,n−1}, stored as k sorted binary indices. It is prepared in constant adaptive depth with O(k² log² n) ancillas and success probability ≥ 1/k. — [arXiv:2608.01144 abstract](https://arxiv.org/abs/2608.01144)
- **How the USS step works, in plain words.** — [arXiv:2608.01144v1 HTML](https://arxiv.org/html/2608.01144v1)
  1. Make k independent uniform index registers over {0,…,η−1}, using Buhrman et al.'s Lemma 1, each with O(log² n) ancillas.
  2. Compute all pairwise "greater-than" bits and **measure them**. This tells you the ranking of the k random indices.
  3. Add classically chosen offsets (Add gates) so that ties and non-strict orderings become a strictly increasing list. This lands in USS over a range of size η + k − 1 − d, where d is the measured "descent number".
  4. η is chosen in advance so that the most likely d gives exactly range n.
  5. Success means "the measured d hit the planned value". Its probability is ≥ 1/k, and numerics suggest about 1/O(√k).
- **Dicke step: USS → D(n,k).**
  1. Index-controlled excitation: write 1s at the k listed positions, using Hadamards, fan-out, parallel phase-Equal gates, fan-out and Hadamards.
  2. Erase the index registers using dyadic-block Hamming weights, prefix ranks, one-hot encodings and Z-phase tricks.
  - Result: O(1) adaptive depth, O(n² + k² log² n + kn log n log log n) ancillas, success ≥ 1/k, and exact when it succeeds. The only heralded measurement layer is the comparison layer. The fan-outs use their own deterministic measurements. — [arXiv:2608.01144v1 HTML](https://arxiv.org/html/2608.01144v1)
- **Boosting by parallel copies (Corollary 2).** Run R = ⌈cα⌉ copies in parallel, with α = 1/p ≤ k, and keep a successful one. Failure probability is ≤ e^(−c), at the same depth. This is still not deterministic. — [arXiv:2608.01144v1 HTML](https://arxiv.org/html/2608.01144v1)
- **Symmetric states (Thm 2 / Cor 3).** A "lifting" framework coherently combines clean **unitary** Dicke circuits. Combined with Joshi & Vasconcelos's constant-depth unitary Dicke circuit (arXiv:2601.10693, with global fan-out and O(n²√log n) ancillas), this gives exact constant-depth arbitrary symmetric states with O(n³√log n) ancillas. The adaptive Thm 1 protocol cannot be plugged in, because measurement records can correlate with the weight label. — [arXiv:2608.01144v1 HTML](https://arxiv.org/html/2608.01144v1)
- **Their comparison table (Table 1)** (E = exact, A = approximate; depths marked * exclude classical processing). — [arXiv:2608.01144v1 HTML](https://arxiv.org/html/2608.01144v1)

  | Method | Type | Connectivity | Depth | Ancillas | Exact? |
  |---|---|---|---|---|---|
  | Bärtschi–Eidenbenz | Unitary | 1D | O(n) | 0 | E |
  | Bärtschi–Eidenbenz | Unitary | All-to-all | O(k log(n/k)) | 0 | E |
  | Bärtschi–Eidenbenz | Unitary | Grid | O(√(nk)) | 0 | E |
  | Liu–Childs–Gottesman | Unitary | All-to-all | O(log³n · loglog n) | O(n log n) | E |
  | Liu–Childs–Gottesman | Probabilistic | All-to-all | O(log n · loglog n) | O(n log n) | E |
  | Yu et al. | Adaptive | All-to-all | polylog(n) | O(log n) | E |
  | LAQCC (W) | LAQCC | Local grid | O(1)* | O(n log n) | E |
  | LAQCC (small-k Dicke) | LAQCC | Local grid | O(1)* | O(n² log n) | E |
  | LAQCC (any k) | LAQCC | 1D | O(log n)* | Ω(n²) | E |
  | Farrell et al., arXiv:2505.03111 | LOCC | Grid | O(1)* | 0 | A |
  | Piroli–Styliaris–Cirac 2024 | LOCC | Grid | O(log k)* or O(1)* | O(1) or O(log k) | A |
  | Joshi–Vasconcelos | Unitary (global fan-out) | All-to-all | O(1) | O(n²√log n) | E |

- **Related references they cite.**
  - Gretta, Gupta, Joshi, "Polylogarithmic-weight Dicke states in QAC⁰…", arXiv:2604.15298.
  - Zi, Nie, Sun, "Constant-depth circuits for arbitrary state preparation via measurement and feedback", arXiv:2503.16208.
  - Liu, Chen, Ma, "State complexity and phase identification in adaptive quantum circuits", arXiv:2509.17014.
  - Source: [arXiv:2608.01144v1 references](https://arxiv.org/html/2608.01144v1)

**Bäumer & Woerner, "Measurement-Based Long-Range Entangling Gates in Constant Depth" (arXiv:2408.03064, Phys. Rev. Research 7, 023120 (2025))**
- **Goal.** Constant-depth fan-out and long-range CNOT using mid-circuit measurement and feedforward, needing only a 1D line. — [arXiv:2408.03064](https://arxiv.org/abs/2408.03064)
- **Fan-out (1 control, n targets).** — [arXiv:2408.03064 HTML](https://arxiv.org/html/2408.03064)
  - Layout: n+1 system qubits alternating with n ancillas on a line (2n+1 qubits).
  - Preferred version: about 3n−1 CNOTs in 5 CNOT layers, n mid-circuit measurements in one round, then one feedforward round.
  - Alternative: about 4n−2 CNOTs, depth 4, 2n−1 measurements in two rounds.
  - Unitary fan-out on a line, for comparison: 2n−1 CNOT layers and 2n−1 CNOTs.
- **Long-range CNOT (skipping n−1 qubits).** — [arXiv:2408.03064 HTML](https://arxiv.org/html/2408.03064)
  - Preferred version: n ancillas, about 4n−2 CNOTs, 7 CNOT layers, one measurement round.
  - Unitary version: about 2n layers and 4n−3 CNOTs.
- **Comparison with other fan-outs.** Buhrman et al.'s fan-out needs 2n ancillas, 2 rounds, depth 6 and about 6n CNOTs on a line. Piroli et al.'s needs n ancillas, 2 rounds, depth about 12 and about 7n CNOTs, and assumes a ladder. — [arXiv:2408.03064 HTML](https://arxiv.org/html/2408.03064)
- **Hardware.** — [arXiv:2408.03064 HTML](https://arxiv.org/html/2408.03064)
  - Device: ibm_kyiv (127-qubit Eagle). Fan-out ran on a 101-qubit line (51 system qubits); long-range CNOT on a 75-qubit line.
  - Feedforward was emulated: a 654 ns delay was inserted and corrections applied in post-processing.
  - Measurement-based beat unitary for fan-out from about 7 system qubits and for long-range CNOT from about 9. The paper's conclusion says 6 and 8, so it is internally inconsistent.

**Koh, Koh, Thompson, "Readout Error Mitigation for Mid-Circuit Measurements and Feedforward" (arXiv:2406.07611, PRX Quantum 7, 010317 (2026))**
- A general error-mitigation protocol for circuits with any number of feedforward layers. It adds no circuit depth or two-qubit gates. It gave up to about 60% error reduction on superconducting hardware, on resets, shallow GHZ preparation and multi-stage teleportation. It does not prepare W or Dicke states. — [arXiv:2406.07611](https://arxiv.org/abs/2406.07611)

### Inferences
- **Moment cost of measurement-based fan-out on Willow in Cirq** (my estimate): each CNOT is H·CZ·H, so the 5 CNOT layers cost about 5 CZ moments plus about 5–6 single-qubit moments, then 1 measurement moment and 1 conditioned-Pauli moment. That is roughly 12–13 moments regardless of n. A unitary grid fan-out over a 15-qubit patch has graph radius only about 3–5, so a unitary BFS-tree fan-out is about 2 × radius ≈ 8–10 moments. **For N ≤ 15 on a grid, measurement-based fan-out rarely pays off.** It pays off only for long 1D-like distances.
- **Luo–Chen–Ma is E-post (success ≥ 1/k), so it cannot be the bonus construction as is.** Its classical "measure the ranking, then fix offsets" trick is a nice example of *converting a measurement outcome into a deterministic correction*. It does not remove randomness in d.
- **Ancilla counts are far too large for a 105-qubit device at N = 15.** The n² term alone is 225.

### Gaps
- Luo–Chen–Ma's explicit constant depth (number of layers) and any small-n gate counts were not in the parts I read.
- I did not get Bäumer & Woerner's exact feedforward correction rule. It is standard teleportation-style: an X correction on targets from the parity of earlier measurements.

---

## Q3. Piroli–Styliaris–Cirac LOCC-assisted circuits; Tantivasadakarn et al.; Smith et al.; constant-depth Dicke papers

### Takeaway
Piroli–Styliaris–Cirac (PRL 133, 230401 (2024), arXiv:2403.07604) prepare W and Dicke states **approximately** in system-size-independent depth:
1. Start from a binomial product state.
2. Measure the excitation number modulo 2^ℓ with a QFT-based ancilla register.
3. Post-select (repeat until success).

Infidelity drops doubly exponentially in ℓ. Their earlier 2021 W protocol (PRL 127, 220503) is exact but needs sequential LOCC rounds, so O(N) time. The IBM 2026 constant-depth W demo (arXiv:2609.08925) follows the same binomial-product-then-project pattern.

### Cited Findings
**Piroli, Styliaris, Cirac, "Approximating many-body quantum states with quantum circuits and measurements", PRL 133, 230401 (2024), arXiv:2403.07604**
- **Main claim.** Relaxing exactness saves substantial resources. For W and Dicke states, the depth and the ancillas per site do not grow with system size. — [arXiv:2403.07604](https://arxiv.org/abs/2403.07604)
- **Gadget V.** A controlled unitary using one ancilla per site, depth 6. Its feedforward is mid-circuit Z measurements plus Pauli corrections, and it is deterministic. — [arXiv:2403.07604v3 HTML](https://arxiv.org/html/2403.07604v3)
- **Excitation-number measurement.** Attach ℓ ancillas. Apply controlled phases exp(i2πN_e/2^x), built from gadget V. Then apply an inverse QFT on the ancillas and read out. This returns N_e mod 2^ℓ at depth O(ℓ). — [arXiv:2403.07604v3 HTML](https://arxiv.org/html/2403.07604v3)
- **W state (Approx, E-post style).** — [arXiv:2403.07604v3 HTML](https://arxiv.org/html/2403.07604v3)
  1. Prepare each qubit as √(1−δ/N)|0⟩ + √(δ/N)|1⟩ (depth 1).
  2. Measure the parity of the excitation number (one ancilla).
  3. Keep the state if the parity is odd; otherwise restart.
  - Infidelity ≤ δ²/4 (leftover weight-3, 5, … components). Success probability > δ/2. Choosing δ ~ √ε gives O(1/√ε) repetitions.
- **Dicke state, M excitations.** — [arXiv:2403.07604v3 HTML](https://arxiv.org/html/2403.07604v3)
  1. Prepare the product state (√(1−p)|0⟩ + √p|1⟩)^⊗N with p = M/N.
  2. Measure N_e mod 2^ℓ.
  3. Post-select on N_e ≡ M.
  - Success ≥ 1/√(8πM). Infidelity is about e^(−2^ℓ).
  - With depth about log₂N, the excitation number is measured exactly, which makes the state **exact after post-selection**.
  - Constant-depth variant: copy each site bit into ℓ−1 ancillas with fan-out, which costs O(ℓ) ancillas per site.
  - Deterministic variant: amplitude amplification gives depth O(M^(1/4) ℓ²) with no repetitions.
- **Connectivity.** Nearest-neighbour chain layers with ancillas attached to sites. No 2D analysis. — [arXiv:2403.07604v3 HTML](https://arxiv.org/html/2403.07604v3)
- **Earlier work (PSC21, PRL 127, 220503 (2021)).** It uses depth-O(1) circuits but "requires sequential use of LOCC", so O(N) preparation time. — [arXiv:2403.07604v3 HTML](https://arxiv.org/html/2403.07604v3). Buhrman et al. make the same point. — [arXiv:2307.14840v5 HTML](https://arxiv.org/html/2307.14840v5)

**Smith, Pokharel, Maurya, Takita (IBM), "Resilience Beyond the Light Cone: Error-Detected Primitives for Practical Dynamic Circuits", arXiv:2609.08925 (8 Sep 2026)**
- **W protocol.** — [arXiv:2609.08925 HTML](https://arxiv.org/html/2609.08925)
  - Apply RY on every qubit to get a binomial product state. They deliberately use a smaller angle, θ = −2 sin⁻¹(√(0.3/N)), trading success probability for fidelity.
  - Project onto weight 1 with one round of "distribute–collapse", i.e. one round of mid-circuit measurement and feedforward. Failed shots are discarded.
  - Layout: a 1D ancilla line with dangling data qubits, 3N qubits in total. Depth is constant.
- **Dicke protocol.** — [arXiv:2609.08925 HTML](https://arxiv.org/html/2609.08925)
  - Phase estimation of (n̂ − k)/τ, bit by bit with no feedforward between bits, each bit a distribute–collapse Hadamard test.
  - τ = O(√k), so O(log k) rounds. Exact preparation needs O(log N) bits.
  - Ancillas: O(N) reused, or O(N log k) for O(1) depth.
  - Success about O(1/√k) for k ≪ N, about O(1/√N) at half filling.
- **Error detection.** ZZ checks or XX/YY "implicit" checks on the distributed GHZ-like control. Flagged shots are post-selected away. — [arXiv:2609.08925 HTML](https://arxiv.org/html/2609.08925)

**Pointers not fetched in this session (from memory; verify before citing)**
- Tantivasadakarn, Thorngren, Vishwanath, Verresen, "Long-range entanglement from measuring symmetry-protected topological phases", arXiv:2112.01519. It uses measurement plus feedforward to make GHZ and topological orders in constant depth. It does not give W or Dicke states directly.
- Smith, Crane, Wiebe, Girvin, "Deterministic constant-depth preparation of the AKLT state…" and "Constant-depth preparation of matrix product states with adaptive quantum circuits" (arXiv:2404.16083). These cover MPS with Abelian-symmetry-correctable tensors. W and Dicke states have bond dimension that grows with n and are not in the "correctable" class as far as I know.
- Piroli, Styliaris, Cirac 2021, "Quantum circuits assisted by local operations and classical communication: transformations and phases of matter", PRL 127, 220503. I believe it is arXiv:2103.13367.

### Inferences
- **All of the "constant-depth Dicke via LOCC" results (PSC24, IBM 2026, Farrell et al.) are Approx or E-post.** Exactness needs the full weight register (about log₂N bits), and determinism needs amplitude amplification, which brings back depth growing with M. None of them is a direct template for a deterministic exact bonus circuit.
- **Measure-and-correct does not work for Dicke states** the way it does for GHZ states. A wrong weight outcome cannot be fixed with Pauli corrections, because changing the weight of a symmetric state needs a non-Pauli, permutation-symmetric operation.

### Gaps
- PSC21 internals (how the sequential W protocol works) were not retrieved.
- Farrell–Zemlevskiy–Illa–Preskill (arXiv:2505.03111), listed as constant-depth LOCC W on a grid (approximate), was not read.

---

## Q4. Measuring total Hamming weight / collective angular momentum (QPE, counting registers), symmetrization registers

### Takeaway
There are two families:
- **Global weight measurement plus adaptive global rotation** (Yu et al., PRL 136 (2026), arXiv:2411.03428). It is exact when the right outcome appears. It loops a random number of times, O(log n) rounds expected, with polylog depth and ancillas.
- **Ancilla symmetrization** (Liu–Childs–Gottesman, arXiv:2411.04019). It turns a sorted list into a symmetric superposition in Õ(log n) depth with O(n log n) ancillas, and is exact.

Neither gives a fixed, constant-length deterministic circuit.

### Cited Findings
- **Yu, Muleady, Wang, Schine, Gorshkov, Childs, "Efficient preparation of Dicke states" (arXiv:2411.03428; PRL, DOI 10.1103/9gjk-rgql).** It uses only global mid-circuit projective measurements and adaptively chosen global rotations, with polylog depth and polylog ancillas. — [arXiv:2411.03428](https://arxiv.org/abs/2411.03428)
- **Algorithm 1 for D(n, n/2), step by step.** — [arXiv:2411.03428 HTML](https://arxiv.org/html/2411.03428)
  1. Start in |j, j⟩ = |0…0⟩.
  2. Rotate all qubits by exp(−iθ_m J_y), with θ_m = arcsin(m/j).
  3. Measure the collective J_z (that is, the Hamming weight).
  4. If |m| > √j, reset.
  5. Repeat until m = 0. The state is then **exactly** |j, 0⟩.
  - Expected O(log j) iterations. It terminates with probability 1, but there is no fixed bound on the number of rounds.
  - A gate-model weight measurement needs about ⌊log n⌋+1 counter ancillas, or O(log² n) depth with none.
  - Cavity-QED version: O(1) per collective measurement with zero ancillas, but coupling must scale as √n.
- **Yu et al.'s comparison.** Low-weight constructions have depth Ω(n^{1/4}) at weight n/2. One reference reaches O(log n) depth with O(n² log n) ancillas. Liu–Childs–Gottesman reach O(log n) with O(n log n). — [arXiv:2411.03428 HTML](https://arxiv.org/html/2411.03428)
- **Liu, Childs, Gottesman, "Low-depth quantum symmetrization" (arXiv:2411.04019).** — [arXiv:2411.04019](https://arxiv.org/abs/2411.04019)
  - Symmetrizes a sorted list, repeats allowed, into an equal superposition of its permutations.
  - Classical input: Õ(log n) depth, O(n log n + log m) ancillas.
  - Superposed input: Õ(log³ n) depth.
  - Gives Dicke states of any weight in Õ(log n) depth.
  - Luo–Chen–Ma's table lists a probabilistic variant at O(log n · loglog n) depth and a unitary one at O(log³ n · loglog n), both exact and all-to-all. — [arXiv:2608.01144v1 HTML](https://arxiv.org/html/2608.01144v1)
- **IBM 2026 Hamming-weight projection.** Phase estimation of n̂ = Σ(1−Z_j)/2, bit by bit, with distribute–collapse (measurement-based) controls. — [arXiv:2609.08925 HTML](https://arxiv.org/html/2609.08925)
- **AC/DC (arXiv:2412.07969).** An automated dynamic-circuit compiler that includes W and Dicke examples using one ancilla with mid-circuit measurement. Seen only in a search snippet, not read. — [arXiv:2412.07969](https://arxiv.org/pdf/2412.07969)

### Inferences
- **For the bonus, any "measure weight, retry if wrong" scheme needs a fixed number of unrolled attempts**, and then success is < 1. A circuit "exact when noiseless" would be violated on the failure branch, so these are out unless the failure branch can be deterministically repaired.
- **The useful idea to borrow is the counter register.** A log₂(N+1)-qubit register holding the weight lets you uncompute/disentangle by X-measurement plus diagonal phase corrections. Those corrections are only cheap when the phase is a product of single-qubit Z's on the system (as in the one-hot / W case).

### Gaps
- I did not verify Liu–Childs–Gottesman's success probability for the probabilistic variant, or its connectivity.
- Barenco et al. symmetrization (1997) and quantum Schur sampling / Schur transform approaches were not researched in this session. No sources gathered.

---

## Q5. Dicke via projection from product states: success probabilities

### Takeaway
Projecting the binomial product state (√(1−p)|0⟩+√p|1⟩)^⊗N with p = k/N onto weight k gives an **exact** D(N,k) whenever the weight outcome equals k (E-post). The success probability is C(N,k)p^k(1−p)^{N−k} ≈ 1/√(2πNp(1−p)), which is ≥ 1/√(8πk) per PSC24. For the bonus sizes it is about 0.2–0.4.

### Cited Findings
- **Bound and mechanism.** Success ≥ 1/√(8πM), and the projected component is proportional to the Dicke state. — [arXiv:2403.07604v3 HTML](https://arxiv.org/html/2403.07604v3)
- **Scaling.** About O(1/√k) for k ≪ N, about O(1/√N) at half filling. — [arXiv:2609.08925 HTML](https://arxiv.org/html/2609.08925)

### Inferences
- **My own arithmetic for the exact binomial value:**
  - D(15,7): C(15,7)·(7/15)^7·(8/15)^8 ≈ 6435 × 3.16e−5 ≈ 0.20. The Stirling estimate 1/√(2π·15·(7/15)(8/15)) gives ≈ 0.21.
  - W, N = 15, p = 1/15: (14/15)^14 ≈ 0.38.
- **This route is not deterministic.** It would need a deterministic fix-up for each wrong weight w ≠ k, which is as hard as preparing the Dicke state in the first place. So it is not usable for an exactness-checked bonus unless the rules accept heralded post-selection. They appear not to, per CLAUDE.md: "Circuits not exact without noise are discarded."

### Gaps
- None beyond the general point that no source gives a deterministic repair for wrong-weight outcomes.

---

## Q6. 2025–2026 hardware results for W/Dicke states with dynamic circuits

### Takeaway
The clearest hardware result is IBM's 2026 constant-depth W states up to N = 20 on ibm_boston (Heron r3). With error detection the fidelities are 0.87, 0.79, 0.69 and 0.59 for N = 5, 10, 15, 20, with 86% down to 46% of shots kept. The protocol is approximate and post-selected. I found no hardware demonstration of a dynamic-circuit (exact) Dicke state with k ≥ 2.

### Cited Findings
- **W-state results.** — [arXiv:2609.08925 HTML](https://arxiv.org/html/2609.08925)
  - N = 5, 10, 15, 20 with error detection: F = 0.87, 0.79, 0.69, 0.59, gains of 0.06–0.21 over the dynamic baseline.
  - Acceptance from 0.86 down to 0.46. 60 qubits used for N = 20.
- **Long-range Bell pair at distance 100.** F = 0.59 with error detection, versus 0.39 for the dynamic baseline and 0.33 for the unitary circuit. — [arXiv:2609.08925 HTML](https://arxiv.org/html/2609.08925)
- **Timing in their noise model.** 2q gate 70 ns, mid-circuit measurement 1 μs, feedforward 650 ns. Measurement-plus-feedforward is about 24 two-qubit-gate durations. — [arXiv:2609.08925 HTML](https://arxiv.org/html/2609.08925)
- **Earlier hardware.** Bäumer & Woerner's measurement-based fan-out on ibm_kyiv (101-qubit line), with feedforward emulated in post-processing. — [arXiv:2408.03064 HTML](https://arxiv.org/html/2408.03064)
- **Superconducting readout mitigation.** Up to about 60% error reduction for feedforward circuits. — [arXiv:2406.07611](https://arxiv.org/abs/2406.07611)
- **Search result.** A targeted search turned up mostly theory, not a clearly identified hardware Dicke (k ≥ 2) dynamic-circuit experiment. — [search: Yu et al. JQI PDF](https://hub.jqi.umd.edu/sites/default/files/2026-01/9gjk-rgql.pdf), [AC/DC arXiv:2412.07969](https://arxiv.org/pdf/2412.07969)

### Inferences
- **Hardware here is irrelevant to the bonus**, which is scored on moment count only. On real hardware, though, a measurement-plus-feedforward step costs about 24 two-qubit-gate times on IBM, so "constant depth" in moments understates wall-clock cost.
- **Bonus design suggestion** (inference, not from a paper): an E-det W/Dicke circuit for N ≤ 15 should use measurement only where it removes a long sequential chain. Candidates:
  - (a) X-measure-and-phase-correct to disentangle a binary or one-hot index register (LAQCC compress step);
  - (b) measurement-based fan-out or long-range CNOT to replace long SWAP chains, using about 12–13 Cirq moments each;
  - (c) measurement-based uncomputation of AND/Toffoli-like ancillas (X-measure, then a CZ/Z phase fix-up).

  Everything probabilistic (binomial projection, Luo–Chen–Ma USS, Yu et al. loop, PSC24) fails the exactness requirement as written.

### Gaps
- No Quantinuum or Google hardware Dicke results with mid-circuit measurement were found within the tool-call budget.
- Absolute W fidelities without error detection, and the per-shot measurement counts, were only partly recovered (the remaining HTML was not read).
