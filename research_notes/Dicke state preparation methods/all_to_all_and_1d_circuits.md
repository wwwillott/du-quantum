# Deterministic unitary circuits for exact Dicke states D(N,k) (and W states): all-to-all and 1D nearest-neighbour

Notation: D(N,k) = |D^N_k>, N qubits, Hamming weight k. Everything below assumes k <= N/2 (for k > N/2, prepare D(N, N-k) and flip every qubit with X; this costs no 2-qubit gates). One CNOT = one CZ plus two Hadamards, so every CNOT count below is also a CZ count on Willow (single-qubit gates are free to merge into PhasedXZ).

## Q1. Bärtschi & Eidenbenz 2019 (arXiv:1904.07358): the split-and-cyclic-shift (SCS) recursion, gate counts, and the linear chain

### Takeaway
B&E 2019 builds D(N,k) from the input |0^(N-k) 1^k> by applying N-1 "split & cyclic shift" blocks. Each block moves the right amount of amplitude so that the last qubit is 1 with probability k/N, then hands the rest to a smaller problem. It uses no ancillas, is exact, needs O(kN) gates and O(N) depth, and works on a plain linear nearest-neighbour (LNN) chain. Written out naively it costs 5(Nk - k(k+1)/2 - N + 1) + 4(N-1) CNOTs. Later papers cut this roughly in half (see Q2/Q5).

### Cited Findings
- **The key identity (plain words):** a Dicke state splits on its last qubit. |D^n_l> = sqrt(l/n)·|D^(n-1)_(l-1)>|1> + sqrt((n-l)/n)·|D^(n-1)_l>|0>. So you first decide the last qubit with the correct weight, then recurse on the remaining n-1 qubits. — [B&E 2019, arXiv HTML](https://arxiv.org/html/1904.07358v1)
- **The Dicke-state unitary U_{n,k}:** a single circuit that maps |0^(n-l) 1^l> to D(n,l) for every l <= k, not just for l = k. It is the product of SCS_{l,k} blocks for l = n down to k+1, then SCS_{l,l-1} for l = k down to 2. — [B&E 2019](https://arxiv.org/html/1904.07358v1)
- **SCS_{n,k} (split & cyclic shift):** it acts on the last k+1 qubits. For each 1 <= l <= k it maps |0^(k+1-l) 1^l> to sqrt(l/n)·(same string) + sqrt((n-l)/n)·|0^(k-l) 1^l 0>. "Split" means two branches; "cyclic shift" means the 1-block slides one position left. The all-0 and all-1 inputs are left unchanged. — [B&E 2019](https://arxiv.org/html/1904.07358v1)
- **Gates inside SCS_{n,k}:** gate (i) is a 2-qubit gate on qubits (n-1, n): a CNOT-conjugated controlled-Ry with angle 2·arccos(sqrt(1/n)). Gates (ii)_l for 2 <= l <= k are 3-qubit gates on (n-l, n-l+1, n): a doubly-controlled Ry with angle 2·arccos(sqrt(l/n)), conjugated by CNOTs. Each gate touches exactly one of the k input patterns. So each SCS block is 1 two-qubit gate plus k-1 three-qubit gates, laid out as a "staircase". — [B&E 2019](https://arxiv.org/html/1904.07358v1)
- **Counts:** (n-1) two-qubit gates (i) and (n-k)(k-1) + sum_{i=3..k}(i-2) = kn - k²/2 + O(n) three-qubit gates (ii). A CCRy costs 4 CNOT + 4 Ry, and one CNOT cancels with the conjugating CNOT, so each (ii) costs 5 CNOT. The paper's bound is "at most 5kn+O(n) CNOT-gates and 4kn+O(n)" Ry gates. — [B&E 2019](https://arxiv.org/html/1904.07358v1)
- **Exact closed form (computed by Mukherjee et al. for the B&E circuit):** 5(nk - k(k+1)/2 - n + 1) + 4(n-1) CNOTs and 4(nk - k(k+1)/2 - n + 1) + 2(n-1) single-qubit gates. — [Mukherjee et al., arXiv:2007.01681 (Table I)](https://arxiv.org/abs/2007.01681)
- **Depth O(n):** consecutive SCS staircases can be "pushed into each other", so about ⌊(k+1)/3⌋-1 gates from later blocks run in parallel with each gate (ii)_k. This gives total depth O(kn)/O(k) = O(n). — [B&E 2019](https://arxiv.org/html/1904.07358v1)
- **Linear chain:** every 3-qubit gate is on three neighbouring wires, except that each one involves the bottom qubit n. B&E move qubit n up the chain with SWAPs so it sits next to the pair it needs. They do this in blocks of about k consecutive SCS unitaries, each costing O(k²) gates and O(k) depth, then sift back down. The total stays O(kn) gates and O(n) depth on LNN. The sift-down step is only there to stop the gate count growing to O(n²). The abstract says "All of these properties even hold for Linear Nearest Neighbor architectures." — [B&E 2019](https://arxiv.org/html/1904.07358v1)
- **Ancillas:** none, and no measurement. — [B&E 2019](https://arxiv.org/abs/1904.07358)
- **Extensions:** (a) arbitrary symmetric states sum_l α_l e^{iφ_l} D(n,l), using O(n²) gates and O(n) depth, also on LNN. First load the amplitudes onto the unary "staircase" inputs with a linear-depth controlled-Ry stair, then apply U_{n,n}. (b) Run in reverse, the circuit gives compression into ⌈log(n+1)⌉ qubits: O(n²) size, Õ(n) depth. — [B&E 2019](https://arxiv.org/html/1904.07358v1)
- **Venue:** FCT 2019, LNCS pp. 126–139, DOI 10.1007/978-3-030-25027-0_9. — [arXiv:1904.07358 abstract page](https://arxiv.org/abs/1904.07358)

### Inferences
- Plain-words picture: start with k ones packed at the right end. Sweep a "hole" leftward through the 1-block. At each step a small rotation decides whether a 1 moves out by one site. Only the 0/1 boundary and the end qubit are ever active, which is why every gate is local except the moving end qubit.
- On a chain where the SCS gates are written directly as nearest-neighbour gates (as this repo's dicke.py does with "sorted unary": the boundary is locally identifiable), you can avoid the SWAP-sifting completely. dicke.py's measured counts (W 2(N-1), D(8,2) 44) match Mukherjee's improved formula 5Nk-5k²-2N exactly at D(8,2) = 44. So dicke.py is essentially "Mukherjee-level" on a chain.

### Gaps
- B&E 2019 gives no explicit depth constant and no exact LNN CNOT count including the SWAP overhead. The 5·C(N-1,2) + 2(N-1) LNN count for U_{N,N}, quoted in Aktar et al. 2022/2025 (Q2), is the closest explicit figure.

## Q2. Earlier and related deterministic constructions (W states, Mukherjee 2020, Aktar 2022 divide-and-conquer, B&E 2022 short-depth)

### Takeaway
For W states, the standard is Cruz et al.: a tree of "B(p)" two-qubit blocks giving 2N-3 CNOTs, in either linear or log2 N depth. For general k, three LANL/ISI works cut B&E's constant factor: Mukherjee 2020 (5Nk-5k²-2N CNOT), Aktar 2022 divide-and-conquer (LNN, and a 2-row "ladder" variant with 5k(N-k)-3(N+k)+5 CNOT), and B&E 2022 (all-to-all depth O(k log(N/k)), with explicit records D(8,2) = 27 CNOT all-to-all and 31 on a grid). All are exact, use no ancillas and no measurements.

### Cited Findings
**Cruz et al. 2019, "Efficient quantum algorithms for GHZ and W states, and implementation on the IBM quantum computer" (arXiv:1807.05572, Adv. Quantum Technol. 2019)**
- Building block B(p): a controlled-G(p) rotation followed by an inverted CNOT. B(p)|00> = |00>, B(p)|10> = sqrt(p)|10> + sqrt(1-p)|01>. In plain words it splits one excitation between two qubits with weight p. — [Cruz et al.](https://arxiv.org/pdf/1807.05572)
- **Linear W_N:** apply B(1/N), B(1/(N-1)), ..., B(1/2) along a chain, which is N-1 blocks in N-1 time steps. This is nearest-neighbour. — [Cruz et al.](https://arxiv.org/pdf/1807.05572)
- **Log-depth W_N:** a binary "dichotomy tree". The excitation is split in half (B(1/2) for powers of 2), then each half splits in parallel, so the depth is about log2 N layers. For general N, use the best split n = ⌊n/2⌋ + ⌈n/2⌉. For N = 2^m the implemented version has 4N-5 gates and 4·log2(N)-2 steps. The first qubit needs about log2 N CNOT partners, so it needs more connectivity than a chain. The first block's control can be dropped because the top qubit starts in |1>. — [Cruz et al.](https://arxiv.org/pdf/1807.05572)
- Their controlled-G(p) can be written with 2 CNOTs (4-gate identity) or with 1 CNOT (3-gate identity: U3, CNOT, U3). — [Cruz et al.](https://arxiv.org/pdf/1807.05572)
- Aktar et al. list Cruz's W-state CNOT counts as 1, 3, 5, 7, 9 for N = 2..6, i.e. 2N-3. — [Aktar et al., arXiv:2112.12435 (Table 2)](https://arxiv.org/abs/2112.12435)

**Mukherjee, Maitra, Gaurav, Roy 2020, "Preparing Dicke states on a quantum computer" (IEEE TQE 1, 2020; arXiv:2007.01681, titled "On actual preparation of Dicke state on a quantum computer")**
- Core tricks: (1) Each 2-qubit SCS piece only ever sees the inputs |00>, |01>, |10>, so it can be any matching unitary. They show it needs only 1 CNOT plus 2 single-qubit gates, i.e. 3 CNOT per block instead of 4. (2) Many of the 2-qubit and 3-qubit sub-gates in the early SCS blocks act as identity on the states that actually occur, so they are deleted: (k-1) of the 2-qubit pieces and (k-2)(k-1)/2 of the 3-qubit pieces. (3) In some 3-qubit gates one control is known to be |1> on every live branch, so the gate collapses to a 2-qubit gate costing 3 CNOT, or even to 1 Ry + 1 CNOT. — [Mukherjee et al.](https://arxiv.org/abs/2007.01681) (full text read from the arXiv PDF)
- **Result:** 5nk - 5k² - 2n CNOTs and 4nk - 4k² - 2n + 1 single-qubit gates for k > 1. For k = 1 the W state uses 2n-2 CNOT and 2n-2 single-qubit gates, the same as the linear W circuit of Cruz et al. — [Mukherjee et al.](https://arxiv.org/abs/2007.01681)
- **Paper's Table II (B&E vs improved CNOT pairs):** k=2: n=4 (22,12), 5 (31,20), 6 (40,28), 7 (49,36), 8 (58,44). k=3: n=4 (27,7), 5 (41,20), 6 (55,33), 7 (69,46), 8 (83,59). k=4: n=5 (46,10), 6 (65,28), 7 (84,46), 8 (103,64). — [Mukherjee et al.](https://arxiv.org/abs/2007.01681)
- They also analyse the "CNOT map" (the interaction graph) of the circuit, to fit it onto IBM coupling maps. — [Mukherjee et al.](https://arxiv.org/abs/2007.01681); [IEEE TQE page](https://tqe.ieee.org/2020/12/01/preparing-dicke-states-on-a-quantum-computer/)
- Aktar et al. note that the Mukherjee/B&E gate reductions need "either much higher than LNN connectivity or the introduction of large constant factors". — [Aktar et al., arXiv:2112.12435](https://arxiv.org/abs/2112.12435)

**Aktar, Bärtschi, Badawy, Eidenbenz 2022, "A Divide-and-Conquer Approach to Dicke State Preparation" (IEEE TQE 3:3101816, 2022; arXiv:2112.12435)**
- **Core idea in plain words:** first "divide" the k excitations between two halves of n/2 qubits. Build the superposition over k1+k2 = k with weights C(n1,k1)·C(n2,k2) in unary on the first half, using controlled-Ry gates with angles from x_i/s_i. Here x_i = C(n1,i)C(n2,k-i) and s_i are suffix sums. A CRy whose target is still a classical 0/1 costs only 1 CNOT. Then "subtract" k1 from the second half's unary k with 1 CNOT per pair. Finally "conquer" both halves in parallel with B&E-style Dicke unitaries, one big-endian and one little-endian (mirror image). — [Aktar et al.](https://arxiv.org/abs/2112.12435)
- **Costs:** divide is O(k) depth and O(k) gates on a ladder (2×n/2), or O(k) depth and O(k²) gates on LNN. Conquer is O(n) depth and O(kn) gates. — [Aktar et al.](https://arxiv.org/abs/2112.12435)
- **LNN CNOT counts (paper's Table 2):** W: 1, 3, 5, 7, 9 for N = 2..6 (2N-3). D(4,2) 10, D(5,2) 17, D(6,2) 24, D(6,3) 32. The paper compares these with the previous best (Mukherjee) of 12, 20, 28, 33. D(4,2): "10 CNOT gates and has a depth of 11 vs. previously best known values of 12 CNOT gates and a depth of 21". — [Aktar et al.](https://arxiv.org/abs/2112.12435)
- **D(6,3) on LNN, two variants:** (a) reversed block order: divide 15 CNOT (CNOT depth 8) plus conquer 9, total 33 CNOT, CNOT depth 17. (b) interleaved blocks: divide 14 plus conquer 9, total 32 CNOT, CNOT depth 20. Trick: SWAPs and CRy+SWAP whose target is still classical cost only 2 CNOT. — [Aktar et al.](https://arxiv.org/abs/2112.12435)
- **Ladder counts:** Table 2 also has a "ladder" column. The extracted text gives 7, 14, 21 for k=2 (N = 4, 5, 6) and 23 for D(6,3), but the PDF layout is garbled. These values match the closed-form formula in the next item, which supports this reading. — [Aktar et al.](https://arxiv.org/abs/2112.12435)
- **Follow-up closed form (Aktar, Bärtschi, Badawy, Eidenbenz, "Scalable Experimental Bounds for Entangled Quantum State Fidelities", arXiv:2210.03048, Quantinuum H1 experiments):** they start from the LNN U_{N,N} with 5·C(N-1,2) + 2(N-1) CNOTs. They reduce it using both an upper and a lower bound on the Hamming weight, then add divide-and-conquer whose divide step uses 2K-1 CNOT "on an all-to-all connectivity". The circuit is also written with X-flips (U_{0,K} = X·U_{N-K,N}·X). Total two-qubit gates: **5K(N-K) - 3(N+K) + 5 CNOTs**, linear depth O(N), O(KN) gates. Example circuit: D(9,3) from parallel U^4 and U^5 blocks. — [Aktar et al. 2022/2025, arXiv:2210.03048](https://arxiv.org/abs/2210.03048)

**Bärtschi & Eidenbenz 2022, "Short-Depth Circuits for Dicke State Preparation" (IEEE QCE 2022; arXiv:2207.09998)**
- **Core idea in plain words:** generalise SCS to a "weight distribution block" WDB^{n,m}_k. It takes a unary weight l <= k on n qubits and splits it into weight i on m qubits and weight l-i on n-m qubits, with amplitude sqrt(C(m,i)C(n-m,l-i)/C(n,l)). It acts only on the last k qubits of each part, so it touches 2k qubits. Steps: unary→one-hot, controlled addition of a weighted superposition into the second register (controlled-Ry stair), back to unary, then unary subtraction built from Fredkin (controlled-SWAP) stairs. — [B&E 2022](https://arxiv.org/abs/2207.09998)
- **WDB cost:** O(k) depth and O(k²) gates, no ancillas. On LNN over the 2k qubits it needs an extra SWAP network (Lemma 2), still O(k) depth. SCS_{n,k} is the special case WDB^{n,1}_k. Cruz's log-depth W tree is WDB^{n,m}_1 applied recursively. — [B&E 2022](https://arxiv.org/abs/2207.09998)
- **All-to-all (Theorem 1):** split the n qubits into ⌊n/k⌋ sets of k qubits. Build a balanced tree by union-by-size, which has height O(log(n/k)). Apply WDBs level by level in parallel, then finish with U^k_k on each set. Result: **depth O(k log(n/k)), O(kn) 2-qubit gates, no ancillas**. — [B&E 2022](https://arxiv.org/abs/2207.09998)
- **Grid (pointer only):** depth O(sqrt(nk)) on grids of size Ω(sqrt(n/s)) × O(sqrt(ns)), O(kn) gates. — [B&E 2022](https://arxiv.org/abs/2207.09998)
- **Concrete records:** D(8,2) on all-to-all uses WDB^{8,4}_2, two WDB^{4,2}_2 and four U^2_2, for **27 CNOTs**. The paper says this "improves on previous work with an explicit CNOT gate count of 44" (Mukherjee). On a grid the same scheme uses **31 CNOTs**, beating an "extrapolated lower bound of 35" for the grid version of Aktar's scheme. — [B&E 2022](https://arxiv.org/abs/2207.09998)
- B&E conjecture their depths are optimal up to constants for constant k on all-to-all, and for any k on sqrt(n/k) × sqrt(nk) grids. — [B&E 2022](https://arxiv.org/abs/2207.09998)

**Older background (from the comparison tables of B&E 2022 and Aktar 2022)**
- Childs et al. 2000 (quant-ph/0012104): probabilistic Dicke preparation by measuring the Hamming weight of a product state into O(log n) ancillas; O(n polylog n) gates. Not deterministic. — [B&E 2022, Table I](https://arxiv.org/abs/2207.09998); [Aktar et al. Table 1](https://arxiv.org/abs/2112.12435)
- Kaye & Mosca 2004: symmetric states with O(n polylog(n/ε)) gates and O(log(n/ε)) ancillas, on LNN. Bacon, Harrow, Chuang 2004 Schur transform: O(n polylog(n/ε)) gates, O(log(n/ε)) ancillas. Plesch & Bužek 2009: symmetric states with O(n²) gates and O(n²) depth, no ancillas. All three are approximate to precision ε or have quadratic depth, and all are superseded by B&E for exact Dicke states. — [B&E 2022, Table I](https://arxiv.org/abs/2207.09998); [Aktar et al. Table 1](https://arxiv.org/abs/2112.12435)
- Diker 2016 (arXiv:1606.09290), "Deterministic construction of arbitrary W states with quadratically increasing number of two-qubit gates": W states only (arbitrary amplitudes), not general Dicke states. I only confirmed it through a reference listing. — search result citing the reference list (see Gaps)

### Inferences
- Ranking by CNOT count for exact D(N,k), from worst to best: B&E 2019 naive, then Mukherjee 2020 (about 5Nk), then Aktar ladder/divide (about 5k(N-k)), then B&E 2022 WDB tree (D(8,2): 27). The bigger gains for small k come from dividing early: the conquer blocks each pay about 5·k·(N/2 - k) instead of 5·k·(N-k).
- The 2×(N/2) ladder in Aktar 2022 is a subgraph of Willow's square grid. So Aktar-style "divide across the two rows, then conquer each row as a chain" fits Willow with no SWAPs. Example counts: D(4,2) 7, D(6,3) 23. Check before use: the 2210.03048 text says the 2K-1 divide CNOTs are "on an all-to-all connectivity", while the TQE paper attributes its low counts to the ladder.
- Cruz's log-depth W tree needs the root to talk to about log2 N partners. On a grid, the broadcast tree already used in this repo (2N-3 CZ) is the same count as Cruz/Aktar (2N-3), so there is little more to gain in CNOT count for W.

### Gaps
- I could not open the Diker 2016 abstract (arXiv:1606.09290) to confirm its gate count. Its title says "quadratically increasing", which would be worse than linear 2N-3.
- No exact explicit CNOT formula is given for B&E 2022's all-to-all construction in general, only big-O and the D(8,2) = 27 and 31 examples.
- I did not read the IEEE versions; the arXiv versions were used.

## Q3. Newer constructions, 2023–2026

### Takeaway
Since 2022, the unitary, no-measurement results that matter are: (1) Yuan & Zhang 2025, with all-to-all depth O(log k · log(N/k) + k), using idle qubits as internal scratch space, and matching depth lower bounds; (2) exact-CNOT synthesis (Wang, Tan, Cong, De Micheli 2024), giving the fewest known CNOTs for tiny cases, e.g. D(4,2) = 6 and D(6,3) = 25 on all-to-all; (3) structural reformulations (MPS view, qudit/spin-s Dicke) that do not lower qubit CNOT counts. Constant-depth unitary schemes exist only with unbounded fan-in gates (QAC0), which Willow does not have. I found no 2026 paper with a new exact LNN Dicke circuit using fewer CNOTs.

### Cited Findings
- **Yuan & Zhang 2025, "Depth-Efficient Quantum Circuit Synthesis for Deterministic Dicke State Preparation" (arXiv:2505.15413):**
  - All-to-all depth O(log(k)·log(n/k) + k), improving B&E 2022's O(k log(n/k)). Grid results (pointer): O(k log(n/k) + n2) for k >= n2/n1, and O(n2) otherwise. — [arXiv:2505.15413](https://arxiv.org/abs/2505.15413)
  - The circuit uses no measurement and no external ancilla. In Theorem 13, though, qubits that are idle at layer j of the WDB tree serve as ancillas, N_j = n/2^(j-1) - 2k of them. These speed up the unary↔one-hot conversion and the Toffoli/CNOT parts of each divide unitary to O(log k)-ish depth. If k = Θ(n), it falls back to B&E 2019's O(n) circuit. The paper reports depth only, not gate counts. — [Yuan & Zhang PDF](https://arxiv.org/pdf/2505.15413)
- **Wang, Tan, Cong, De Micheli 2024, "Quantum State Preparation Using an Exact CNOT Synthesis Formulation" (DATE 2024; arXiv:2401.01009):** an exact shortest-path or SAT-style search over {CNOT, Ry, CRy, MCRy} circuits. It claims to be "the first design automation algorithm to surpass manual design, reducing the best CNOT numbers to prepare a Dicke state by 2×". Table IV, comparing their result with the Mukherjee manual design: D(3,1) 4 vs 4; D(4,1) 7 vs 7; **D(4,2) 6 vs 12**; D(5,1) 10 vs 10; D(5,2) 16 vs 20; D(6,1) 13 vs 13; D(6,2) 22 vs 28; **D(6,3) 25 vs 33**. Code: github.com/Nozidoali/quantum-xyz. The circuits are irregular and do not generalise. — [arXiv:2401.01009](https://arxiv.org/abs/2401.01009); [DATE proceedings PDF](https://past.date-conference.com/proceedings-archive/2024/DATA/417_pdf_upload.pdf)
- **Raveh & Nepomechie, "Dicke states as matrix product states" (PRA 110, 052438, 2024; arXiv:2408.04729):** writes the Dicke state as an MPS and turns it into a sequential deterministic circuit, which they relate to B&E's recursion. They also give minimal-bond-dimension MPS for higher-spin and qudit Dicke states. — [PRA](https://link.aps.org/doi/10.1103/PhysRevA.110.052438)
- **Nepomechie & Raveh, "Qudit Dicke state preparation" (QIC 24, 2024; arXiv:2301.04989)** and **Nepomechie, Ravanini, Raveh, "Spin-s Dicke states and their preparation" (Adv. Quantum Technol. 2024; arXiv:2402.03233):** deterministic, ancilla-free qudit generalisations of B&E. The spin-s gate count scales as s·k·(2sn-k). Not relevant to qubit hardware except as a cross-check. — [arXiv:2301.04989](https://arxiv.org/pdf/2301.04989); [arXiv:2402.03233](https://arxiv.org/pdf/2402.03233)
- **Joshi & Vasconcelos 2026, "Constant-Depth Unitary Preparation of Dicke States" (arXiv:2601.10693):** gives constant-depth exact Dicke circuits, but only with unbounded fan-in CZ (QAC0) plus polynomial ancillas, and with quantum FAN-OUT for arbitrary weight. It targets neutral-atom and ion global gates. The abstract says unitary preparation in the standard circuit model is limited to logarithmic depth. — [arXiv:2601.10693](https://arxiv.org/abs/2601.10693)
- **Patamawisut et al. 2025, "Quantum Circuit Design for Decoded Quantum Interferometry" (arXiv:2504.18334):** uses plain B&E SCS with 10ℓ-6 CNOTs per SCS block. This is an unoptimised B&E and worse than Mukherjee, so it is not useful as a count source. — [arXiv:2504.18334](https://arxiv.org/abs/2504.18334) (full text extracted)
- **Pointers outside scope (other researchers cover these):** adaptive constant-depth Dicke (arXiv:2608.01144); collision-model Dicke on superconducting hardware (arXiv:2608.30176); AshN-gate D(8,2) on a 2×4 patch (arXiv:2608.24084); global-control W/Dicke on rings (arXiv:2512.19545); LCU block encodings via Dicke states (arXiv:2507.20887). — [search results](https://arxiv.org/abs/2505.15413) (titles seen in search listings only)

### Inferences
- For a hackathon at N <= ~20, the asymptotic depth improvements (B&E 2022, Yuan–Zhang) matter less than constant factors. Yuan–Zhang's gains come from ancilla-assisted log-depth CNOT and Toffoli circuits that have large constants and need all-to-all connectivity, so they are probably not worth implementing for Willow.
- The exact-synthesis numbers (D(4,2) = 6 CNOT) assume all-to-all connectivity and the gate set {CNOT, Ry, CRy=2, MCRy}. A search over CZ plus arbitrary single-qubit gates on a fixed connected subgraph can beat that. The repo's submission.py already reaches D(4,2) with 5 CZ.

### Gaps
- I found no 2023–2026 paper that gives a general closed-form CNOT count below 5k(N-k)-3(N+k)+5 for exact D(N,k) on LNN or a ladder. My searches cannot rule out a recent preprint I missed.
- I found no Schur–Weyl or quantum-Schur-transform based exact Dicke circuit with competitive CNOT counts. The Schur-transform route (Bacon–Harrow–Chuang and follow-ups) is approximate to ε and ancilla-heavy.

## Q4. Lower bounds on depth and gate count

### Takeaway
Depth lower bounds come from light-cone arguments: every pair of qubits in a Dicke state is entangled, so their backward light cones must overlap. That gives Ω(log N) on all-to-all, Ω(N) on a 1D chain, and Ω(n2) on an n1×n2 grid (n1 <= n2). B&E 2019 is therefore depth-optimal up to constants on a chain. No tight Dicke-specific CNOT-count lower bound was found; the only exact optima are computational, for tiny N.

### Cited Findings
- Yuan & Zhang, Theorem 20: any standard (2-qubit gate, no measurement) circuit preparing D(n,k) needs depth Ω(log n) with all-to-all connectivity, Ω(n2) on an n1×n2 grid, and **Ω(n) on a path**. Proof: if the depth is too small, two qubits have disjoint light cones and are therefore unentangled. — [Yuan & Zhang PDF](https://arxiv.org/pdf/2505.15413)
- Yuan & Zhang conjecture that Ω(k) is also a lower bound on all-to-all, which is still open. — [arXiv:2505.15413](https://arxiv.org/pdf/2505.15413)
- B&E 2022 conjecture, "based on (perhaps misleading) insights from classical lower bounds", that their depths are optimal up to constants for constant k on all-to-all and for sqrt(n/k)×sqrt(nk) grids. On such grids the depth matches the graph diameter. — [B&E 2022](https://arxiv.org/abs/2207.09998)
- Joshi & Vasconcelos: in the standard circuit model, unitary Dicke preparation is limited to logarithmic depth. Constant depth needs measurement plus feed-forward, or unbounded fan-in gates. — [arXiv:2601.10693](https://arxiv.org/abs/2601.10693)
- Mukherjee et al. prove small local optimality facts: their 2-qubit transformation T1(θ) needs at least 1 CNOT and 2 single-qubit gates, and a certain 2-qubit gate needs at least 3 elementary gates. These are per-gadget bounds, not a whole-circuit bound. — [Mukherjee et al.](https://arxiv.org/abs/2007.01681)
- Wang et al. 2024's exact synthesis is optimal within its gate library and cost model for the states it solved, with N <= 6 for Dicke. — [arXiv:2401.01009](https://arxiv.org/abs/2401.01009)

### Inferences
- A simple counting bound: every qubit must be entangled with the rest, so the 2-qubit interaction graph must be connected, which needs at least N-1 two-qubit gates. The W-state constructions (2N-3) are within a factor of 2 of this. For D(N,k) with k ~ N/2 nobody has a matching bound, so the Θ(kN) gap is open. (This bound is my own reasoning, not from a source.)
- On Willow, a patch of side length L forces depth Ω(L) for any exact unitary circuit, as the CLAUDE.md note on Yuan–Zhang already says.

### Gaps
- I found no published lower bound on the CNOT/CZ count of exact D(N,k) beyond trivial connectivity arguments and small exact searches.

## Q5. Exactness, and counts for small cases D(4,2), D(6,3), D(8,2), D(8,4), D(10,5)

### Takeaway
Every construction above is exact when noiseless (deterministic, no post-selection), except Childs et al. (probabilistic) and the Kaye–Mosca / Schur-transform approaches (ε-approximate). The best published CNOT counts are D(4,2) 6, D(6,3) 23–25, D(8,2) 27 (all-to-all) or 31 (grid), D(8,4) about 49, and D(10,5) about 85 (ladder/divide-and-conquer formula). The table below gives sources and connectivity.

### Cited Findings
| State | B&E 2019 (formula) | Mukherjee 2020 (5Nk-5k²-2N) | Aktar 2022 LNN | Aktar ladder / 5K(N-K)-3(N+K)+5 | B&E 2022 | Exact synthesis 2024 (all-to-all) |
|---|---|---|---|---|---|---|
| W_N = D(N,1) | 5(N-1-1-N+1)+4(N-1) = 4N-4 by the formula; Mukherjee's text says 3N-3 before its W tweak | 2N-2 | 2N-3 | 2N-3 | (Cruz tree, 2N-3) | D(4,1) 7, D(6,1) 13 (they compare against Mukherjee's general formula, not 2N-3) |
| D(4,2) | 22 | 12 | 10 (depth 11) | 7 | – | **6** |
| D(6,3) | 55 | 33 | 32 (CNOT depth 20) / 33 (depth 17) | 23 | – | 25 |
| D(8,2) | 58 | 44 | – | 35 | **27** all-to-all; **31** grid | – |
| D(8,4) | 103 | 64 | – | 49 | – | – |
| D(10,5) | 166 | 105 | – | 85 | – | – |

Sources for the table:
- B&E 2019 column: formula 5(nk - k(k+1)/2 - n + 1) + 4(n-1) and Table II pairs (22, 55, 58, 103 printed in the paper; 166 computed from the formula). — [Mukherjee et al.](https://arxiv.org/abs/2007.01681)
- Mukherjee column: formula and Table II (12, 33, 44, 64 printed; 105 computed). — [Mukherjee et al.](https://arxiv.org/abs/2007.01681)
- Aktar LNN: 10 / 32 / 33 printed. — [Aktar et al. arXiv:2112.12435](https://arxiv.org/abs/2112.12435)
- Ladder / formula column: formula 5K(N-K)-3(N+K)+5 from [arXiv:2210.03048](https://arxiv.org/abs/2210.03048). The values 7 and 23 also appear in the ladder column of [arXiv:2112.12435](https://arxiv.org/abs/2112.12435); 35, 49 and 85 are computed from the formula.
- B&E 2022: 27 and 31 printed. — [arXiv:2207.09998](https://arxiv.org/abs/2207.09998)
- Exact synthesis: 6 and 25 printed. — [arXiv:2401.01009](https://arxiv.org/abs/2401.01009)
- Exactness: B&E 2019, Mukherjee, Aktar and B&E 2022 are deterministic and ancilla-free. Childs 2000 is probabilistic; Kaye–Mosca and Bacon–Harrow–Chuang are ε-approximate. — [B&E 2022 Table I](https://arxiv.org/abs/2207.09998); [Aktar Table 1](https://arxiv.org/abs/2112.12435)

### Inferences
- **Compared with this repo's current pipeline** (CLAUDE.md, same fake calibration): submission.py has D(4,2) 5, D(6,3) 35, D(8,2) 40, D(8,4) 70, D(10,5) 117 CZ. The literature suggests real headroom for k >= 2 at N >= 6. D(6,3): 23 (ladder formula) vs 35. D(8,2): 27–31 (B&E 2022 WDB tree, grid version 31) vs 40. D(8,4): 49 vs 70. D(10,5): 85 vs 117. The main lever is an Aktar-style "divide k across two rows of a 2×(N/2) ladder, then conquer both rows in parallel as mirrored chains", or B&E 2022's recursive WDB tree. These also halve the depth. The repo's dc2_template is already this kind of construction, so compare its CZ counts against 5K(N-K)-3(N+K)+5.
- **Caveat on counting:** these papers count CNOT with arbitrary single-qubit gates. The repo's 5-CZ gadget corresponds to the 5-CNOT CCRy-based gate (ii), and the 3/4-CZ gadgets correspond to Mukherjee's "known-|1>-control" reductions. Applying the "SWAP / CRy on a classical target costs 1–2 CNOT" tricks from Aktar is a cheap win wherever a qubit is still in a known basis state.
- **For the smallest instances (N <= 6)**, exact numerical synthesis beats every hand construction, and a CZ-native search on the actual Willow subgraph is the right tool. This matches CLAUDE.md strategy item 3.

### Gaps
- The ladder-column values in the 2112.12435 table are reconstructed from garbled PDF text, and they agree with the 2210.03048 formula. The 2210.03048 text, however, says the divide uses "all-to-all" connectivity. Whether 5K(N-K)-3(N+K)+5 is achievable on a ladder or only on all-to-all should be checked by building D(6,3) (23 CNOT) directly.
- No published explicit counts were found for D(10,5) or D(8,4) with B&E 2022's WDB tree. Only D(8,2) is given.
- The W-state row's B&E-formula entry is ambiguous: Mukherjee's text says B&E-style W costs 3N-3 CNOT before the tweak, while plugging k = 1 into its Table I formula gives 4N-4. Use 2N-3 as the practical value.
