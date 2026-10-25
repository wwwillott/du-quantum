# Dicke / W-state preparation on restricted 2D connectivity (square grids, heavy-hex) and connectivity-aware compilation

Notation: D(n,k) = Dicke state on n qubits with k excitations (Hamming weight k). U^n_k = "Dicke state unitary" mapping unary input |0^{n-l}1^l> to D(n,l) for every l <= k. WDB^{n,m}_k = "weight distribution block" (defined below).

## Q1. Bärtschi & Eidenbenz 2022 (arXiv:2207.09998, IEEE QCE'22, pp. 87–96): the 2D grid construction and its O(sqrt(nk)) depth

### Takeaway
Cut the grid into tiles of k qubits, each threaded by a short snake (1D path). A divide-and-conquer gadget called a weight distribution block (WDB) splits "how many 1s go where" between two neighbouring tiles. Run the WDBs down the first column of tiles, then along every row in parallel, then run small 1D Dicke unitaries inside each tile. A WDB costs O(k) depth and O(k^2) gates, and there are about sqrt(n/k) sequential WDB steps per direction, so depth is O(k·sqrt(n/k)) = O(sqrt(nk)) and total gates are O(kn), with no ancillas. They also give hand-tuned D(8,2) circuits: 31 CNOTs on a grid and 27 CNOTs all-to-all.

### Cited Findings
**Headline results**
- Depth O(k log(n/k)) for all-to-all connectivity. Depth O(k sqrt(n/k)) = O(sqrt(nk)) on grids of size Ω(sqrt(n/s)) × O(sqrt(ns)) with 1 ≤ s ≤ k. Both use O(kn) gates and no ancillas. The previous state of the art was O(n) depth on LNN for arbitrary k (the FCT 2019 construction). — [arXiv:2207.09998](https://arxiv.org/abs/2207.09998)
- "Grid-like" topologies are explicitly in scope. The paper defines a topology (e.g. heavy-hex) as grid-like if a grid maps onto it canonically with at most constant-factor overhead in depth and 2-qubit gate count. It names IBMQ, Google and Rigetti devices as targets, but all proofs assume Cartesian grids. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)

**Main idea in plain words**
- Many small 1D Dicke unitaries U^k_k run in parallel, each spreading some number l_i ≤ k of 1s inside its own block of k qubits. The hard part is preparing the correctly weighted superposition of (l_1, l_2, ...) with Σ l_i = k. The amplitudes must follow products of binomials (e.g. D(6,3) = (U^3_3 ⊗ U^3_3) applied to Σ_l sqrt(C(3,3−l)·C(3,l)/C(6,3)) |0^l 1^{3−l}>|0^{3−l}1^l>). — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)
- WDB^{n,m}_k definition: it takes any unary-encoded weight l ≤ k on n qubits and splits it into a correctly weighted superposition of unary weights on two parts of size m and n−m. The weights are hypergeometric: sqrt(C(m,i)·C(n−m,l−i)/C(n,l)). It generalises two older schemes. The log-depth W-state scheme of Cruz et al. is the k = 1 case. The Aktar et al. two-half split is the special case that accepts only l = k. The SCS gate of the 2019 linear construction is WDB^{n,1}_k. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)
- Key locality observation: WDB^{n,m}_k acts only on the last k qubits of each part (only k qubits per side can carry the 1s). On hardware, those two k-qubit sets just need to be neighbours along an LNN path spanning the 2k qubits. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)

**WDB circuit (Lemma 2)**
- The WDB has four steps:
  1. Convert register 1 from unary to one-hot encoding.
  2. Controlled-add a superposition of weights i ≤ l into register 2 (unary), using a stair of controlled θ-rotations with angles θ = 2·arccos sqrt(x_i/s_i), where x_i = C(m,i)·C(n−m,l−i) and s_i are suffix sums.
  3. Convert register 1 back to unary.
  4. Subtract i from register 1, using a stair of Fredkin gates (each Fredkin = Toffoli + 2 CNOT).
- The stairs can be partly parallelised, giving depth O(k) and O(k^2) gates. On LNN, an extra SWAP network costing O(k) depth and O(k^2) gates keeps the controls next to their targets. Reversing a register's qubit order costs another O(k)-depth SWAP network. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)

**Grid layout (Section V.B, Fig. 5)**
- Tiling: a sqrt(n/s) × sqrt(ns) grid is cut into sqrt(n/k) × sqrt(n/k) rectangles of size sqrt(k/s) × sqrt(ks), each holding k qubits. Inside each rectangle a snake-shaped LNN path is chosen so that its head/tail touches the head/tail of the snake in each neighbouring rectangle.
- Tiers: in the leftmost column, a rectangle's tier is the number of qubits in its lower-right quadrant. Elsewhere it is the number of qubits in its row to the right.
- Order of operations: blocks WDB^{x, x−sqrt(nk)}_k pass weight top-to-bottom down the leftmost column (sqrt(n/k) sequential steps). Then blocks WDB^{y, y−k}_k pass weight left-to-right along each row, with all rows in parallel. Finally, U^k_k runs in each rectangle in parallel.
- Cost: each phase is O(k·sqrt(n/k)) depth. Gates are O(k^2·sqrt(n/k)) + O(k^2·n/k) = O(kn).
- Source: [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)
- Better variant used in the full proof, which also handles non-integer square roots: use k-qubit snakes only in the leftmost column. Each row to the right becomes one long snake of Θ(sqrt(nk)) qubits. After the column distribution, run one U^{O(sqrt(nk))}_k on each (short snake + its row) pair, all rows in parallel. If the two snakes are glued in the wrong order, first reverse the short snake with a SWAP network. Depth O(sqrt(nk)), gates O(kn). This avoids unused qubits inside partly filled rectangles, which would otherwise block interactions and have to be swapped out of the way. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)
- Suggested constant-factor improvement: inject the weight in a central rectangle and distribute it up/down the column, then to left and right half-rows. This cuts depth by about 4×. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)

**Optimality**
- On Θ(sqrt(n/k)) × Θ(sqrt(nk)) grids the depth matches the grid diameter asymptotically, so the authors conjecture it is optimal up to constants there. They note that teleportation / mid-circuit feed-forward could get around the limit, effectively by shrinking the graph diameter. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)
- All-to-all case: qubit sets are arranged as a union-by-size tree of height O(log(n/k)) and WDBs are applied to undo it. The authors relate this to minimum broadcast time and conjecture optimality for constant k. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)

**Concrete small circuits**
- D(8,2) all-to-all (Fig. 6): WDB^{8,4}_2 (accepts only input weight 2), then two WDB^{4,2}_2, then four U^2_2. Total 27 CNOTs. The paper says this beats the previous explicit count of 44 (their ref [25]). — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)
- D(8,2) on a grid (Fig. 7): same skeleton, but the WDB^{4,2}_2 blocks are combined with a reversal of two wires, and two of the four U^2_2 are oriented opposite to the others. Total 31 CNOTs, versus an "extrapolated lower bound of 35" for grids using the Aktar et al. scheme. Quirk links for both circuits are in the paper. The authors say these use "hand-tuned non-scalable" WDB implementations. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)
- Symmetry facts: D(n, n−k) = X^{⊗n} D(n,k). Because the input is unary, the same circuits also prepare any symmetric state supported on weights ≤ k. Run in reverse, they compress such states into ceil(log(k+1)) qubits. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)

### Inferences
- For our project, the D(8,2) grid circuit (31 CNOT = 31 CZ-equivalent) is a concrete benchmark. Our dicke.py uses 44 CZ and submission.py 40 CZ for D(8,2). The skeleton to copy is: WDB that accepts only l = k (it starts from a fixed weight, so it is cheaper), then two-level halving, then U^2_2 leaves, with wire reversals absorbed by orienting the leaves differently. That last step is permutation-symmetry relabelling in practice.
- Willow-relevant sizes (N ≲ 20 on a compact patch) put us in the small-constant regime. Asymptotic depth hardly matters there; the CZ count of the WDB gadgets (Fredkin stairs are expensive) dominates. The paper itself says its scalable WDBs are not resource-efficient and the small circuits were hand-tuned.
- The WDB locality fact (only k qubits per side matter) is what lets a WDB sit on a 2k-qubit path. For W states (k = 1) a WDB is just a two-qubit split, which is why tree-shaped W circuits embed directly in the hardware graph.

### Gaps
- I could not read the Fig. 6/7 gate-level details beyond the rotation-angle labels in the extracted text: the angles are θ/2 with arguments sqrt(5/6), sqrt(1/5), sqrt(6/28), sqrt(6/22) and sqrt(1/2). The Quirk links would give the exact circuits.
- No hardware experiment in this paper.
- Heavy-hex is claimed as grid-like but no explicit heavy-hex embedding or constant is given.

## Q2. Yuan & Zhang 2025 (arXiv:2505.15413, "Depth-Efficient Quantum Circuit Synthesis for Deterministic Dicke State Preparation"): O(k log(n/k) + n2) on an n1×n2 grid and the Ω(n2) lower bound

### Takeaway
They replace Bärtschi–Eidenbenz's lopsided "column then rows" split with a balanced halving recursion on the grid: split the region into left/right halves, divide the weight between two adjacent blocks at the midpoint, move one block into place, and recurse on both halves in parallel. The divide steps cost O(k) per level over log(n/k) levels. The block moves form a geometric series that sums to O(n2). The matching Ω(n2) lower bound is a light-cone argument: all pairs of qubits in a Dicke state are entangled, so on a grid the circuit needs depth on the order of the long side before far-apart qubits can share a light cone.

### Cited Findings
**Headline results**
- All-to-all: depth O(log(k)·log(n/k) + k), improving O(k log(n/k)). Grid n1×n2 with n1 ≤ n2: depth O(k log(n/k) + n2) when k ≥ n2/n1, and O(n2) when k < n2/n1. Lower bounds: Ω(log n) for all-to-all and Ω(n2) for the grid. — [arXiv:2505.15413](https://arxiv.org/abs/2505.15413)
- The grid construction is asymptotically optimal when k = O(n2/log n1). The circuits are deterministic, with no measurements and no ancillas on the grid (ancillas appear only in the all-to-all lemmas used to cut depth). — [arXiv:2505.15413 (HTML)](https://arxiv.org/html/2505.15413)

**Building blocks**
- Three subroutines:
  - Divide unitary on two adjacent k-qubit blocks forming a 2k-qubit path: depth O(k), size O(k^2).
  - Permutation unitary that swaps the contents of two blocks anywhere on the grid: depth O(n2).
  - Path-constrained Dicke unitary: depth O(n), size O(nk).
- Source: [arXiv:2505.15413 (HTML)](https://arxiv.org/html/2505.15413)

**Case k ≥ n2/n1**
- The grid is cut into n/k blocks of k qubits, each a sub-grid of about sqrt(n1·k/n2) × sqrt(n2·k/n1).
- Recursion:
  1. Split the region into left/right halves.
  2. Apply the divide unitary to the two adjacent blocks straddling the midpoint.
  3. Permute the second block to the top-left corner of the right half (cost O(n2/2^i) at level i).
  4. Recurse on both halves in parallel.
- Divides add up to O(k log(n/k)); permutations form a geometric series summing to O(n2). The authors credit the gain over BE22 to balanced recursion and more parallelism, partly offset by the cost of moving excitations to the midpoint. The level-by-level bookkeeping is compressed in the text.
- Source: [arXiv:2505.15413 (HTML)](https://arxiv.org/html/2505.15413)

**Case k < n2/n1 (wide, thin grid)**
- Blocks are 1×k strips. Repeatedly:
  1. Divide off the weight belonging to the first column (n1·k qubits).
  2. Run the Dicke unitary U^{n1·k} on that column in parallel with continuing on the remaining columns.
- Total depth is about (n2/k)·O(k) + T(n1·k) = O(n2).
- Source: [arXiv:2505.15413 (HTML)](https://arxiv.org/html/2505.15413)

**Lower bound**
- Light-cone principle: if two qubits' backward light cones do not intersect, a product input leaves them unentangled. Every pair of qubits in a Dicke state is entangled, which forces large depth. — [arXiv:2505.15413 (HTML)](https://arxiv.org/html/2505.15413)
- Grid version: on a grid the backward light cone grows like O(t^2) until it spans the short side n1, then like O(n1·t), and covers everything only after t > n2.
- If d = o(n2), one can pick cones from the upper-left and lower-right corners that stay disjoint and of size o(n). The circuit then factorises into a tensor product, which contradicts the sign/coefficient structure of the Dicke reduced state.
- The path case (n1 = 1) gives Ω(n). The grid proof is only sketched ("by the same discussion above").
- Source: [arXiv:2505.15413 (HTML)](https://arxiv.org/html/2505.15413)

**Open problems and missing numbers**
- The authors conjecture Ω(k) depth even for all-to-all; if true, their constructions would be optimal up to a log factor. — [arXiv:2505.15413 (HTML)](https://arxiv.org/html/2505.15413)
- Only asymptotic results (Table 1). No constant factors, no total grid gate count, no numerical benchmarks, no hardware. — [arXiv:2505.15413 (HTML)](https://arxiv.org/html/2505.15413)

### Inferences
- In plain terms, the depth floor on a grid is set by the long side of the patch, not by N. This favours compact, near-square patches: on Willow, a 4×5 patch for N = 20 has n2 = 5.
- At hackathon sizes the light-cone bound is a handful of layers, far below our circuits' actual depths (BE-style ~4N). The bound does not constrain us; CZ count and idle time do.
- The "permutation unitary" that moves block contents across the grid costs SWAPs. For a Dicke state the final qubit order does not matter, so a practical implementation can often relabel instead of physically moving blocks. That works whenever the block whose contents would be moved has not yet been entangled with anything other than its divide partner. This is my inference, not stated in the paper.

### Gaps
- No explicit gate count or constant factors, so it is impossible to say from the paper whether the balanced recursion beats BE22 at N ≤ 20.
- The full grid lower-bound proof is only sketched.

## Q3. Wang et al. 2026 (arXiv:2608.24084, "Lifting connectivity bottlenecks in superconducting quantum processors via enriched native two-qubit gates"): D(8,2) on a 2×4 patch, 31 CZ

### Takeaway
The CZ baseline is exactly the hand-tuned 31-CNOT grid circuit from Bärtschi–Eidenbenz 2022 (their Fig. 7). For every lattice variant, including ones with dead couplers, they recompiled it with Mirror-SABRE, a router that does qubit mapping and SWAP routing together and absorbs SWAPs. Their contribution is the "AshN" control scheme, which offers a richer native two-qubit gate set on each coupler (CZ, iSWAP, √iSWAP, SWAP, and a B-like gate). Logical gates and routing SWAPs merge into single native gates, so D(8,2) drops to 23 entangling gates with fidelity 0.736. The CZ version reaches 0.631 on the perfect 2×4 patch. The lesson about defects: CZ-only compilation degrades quickly as couplers are removed, because every extra SWAP costs 3 CZ. The richer gate set is almost insensitive (−0.04 fidelity with three defective couplers).

### Cited Findings
**Setup**
- Authors: Hanyi Wang, Jingzhe Guo, …, Jianxin Chen (20 authors), submitted 25 Aug 2026.
- AshN gives rich two-qubit control on each nearest-neighbour pair, so "a logical interaction and the qubit routing it needs" merge into one native operation.
- Across seven benchmark circuits, AshN cut two-qubit gate count versus CZ compilation by geometric means of 45.2% (1D topologies) and 43.7% (2D).
- Source: [arXiv:2608.24084](https://arxiv.org/abs/2608.24084)

**D(8,2) results**
- Gate counts: AshN 23 entangling gates vs CZ 31.
- Fidelity on the defect-free 2×4: AshN 0.736 ± 0.001, CZ 0.631 ± 0.001.
- Collective spin: ⟨Jz²⟩ = 3.91 (AshN) vs 3.79 (CZ), ideal 4. ⟨Jx²+Jy²⟩ = 13.64 vs 12.39, ideal 16.
- GME witness: −0.0344 (AshN, certified) vs +0.0593 (CZ, not certified).
- Source: [arXiv:2608.24084 (HTML)](https://arxiv.org/html/2608.24084)
- Device: same architecture as Zuchongzhi 3.0 (105 transmons, 182 tunable couplers). Average CZ XEB error 0.550%. The other native gates average 0.615%, attributed to more complex pulses and crosstalk from simultaneous XY drives. — [arXiv:2608.24084 (HTML)](https://arxiv.org/html/2608.24084)

**Compilation**
- The base circuit is the CZ construction from Bärtschi et al. 2022, described as a "short-depth construction for a perfect 2×4 rectangular lattice". It was recompiled per topology with Mirror-SABRE (joint mapping + SWAP routing).
- Eight topologies were tested: five 2×4 variants and three 3×3 variants. The 3×3 variants place 8 qubits on a 9-site patch.
- Source: [arXiv:2608.24084 (HTML)](https://arxiv.org/html/2608.24084)

**Defect results**
- Table S9 lists defect sets G25; G12/G46; G13/G25/G47; G13/G25/G68. Witness values (±0.0008), AshN / CZ:

  | Configuration | AshN | CZ |
  |---|---|---|
  | Perfect | −0.0344 | +0.0593 |
  | G25 | −0.0140 | +0.0997 |
  | G12/G46 | −0.0360 | +0.1383 |
  | G13/G25/G47 | −0.0089 | +0.1441 |
  | G13/G25/G68 | −0.0119 | +0.1945 |

- So CZ worsens steadily as defects are added, while AshN stays certified in every case.
- Source: [arXiv:2608.24084 (HTML)](https://arxiv.org/html/2608.24084)
- CZ fidelity "deteriorates rapidly" with defective couplers; AshN drops only about 0.04 with three defective couplers. — [arXiv:2608.24084 (HTML)](https://arxiv.org/html/2608.24084)
- Per-coupler errors differ by native gate. Example: on G25, iSWAP error is 0.319% vs CZ 0.410%. — [arXiv:2608.24084 (HTML)](https://arxiv.org/html/2608.24084)

### Inferences
- In our challenge only CZ + PhasedXZ are allowed, so AshN is not usable. The transferable lessons:
  1. 31 CZ is the published state of the art for D(8,2) on a 2×4 grid.
  2. A generic router (SABRE-type) applied to a circuit hand-tuned for a perfect lattice pays heavily for each missing coupler, roughly 3 CZ per SWAP.
  3. Building the circuit directly on whatever connected patch is available, relabelling qubits instead of routing (allowed by permutation symmetry), is the CZ-only answer to dead couplers. No paper I found does this explicitly.
- A CZ fidelity of 0.631 at 0.55% CZ error with 31 CZ is consistent with error budgets dominated by 2-qubit gates plus readout/idling. Willow-class CZ errors (~0.3% or less) on a 31–40 CZ circuit would give a similar order of fidelity.

### Gaps
- Per-topology CZ/SWAP counts after Mirror-SABRE and per-defect fidelities are not in the parts of the text I could read. Only witness values were available.
- The exact coupler labels for the "Defect247" / "Defect45" figure names are unconfirmed: one read of the text reported both labels, another found only Defect247 in the supplement.

## Q4. Other relevant papers: W states on grids/trees, heavy-hex Dicke/W, hardware experiments, routing/mapping, lower bounds

### Takeaway
Little published work targets Dicke states on real 2D hardware graphs beyond BE22, Yuan–Zhang and Wang et al. The IBM-hardware papers (Mukherjee et al. 2020; Aktar et al. 2022) simply embed LNN circuits into heavy-hex. For W states, the measurement-free route is a tree of two-qubit "split" gates (Cruz et al. log-depth W; the WDB with k = 1) embedded in the device graph. Logarithmic depth on a grid is only known with mid-circuit feed-forward. No paper I found exploits permutation symmetry for SWAP-free routing or does noise-aware placement for Dicke states. Only GHZ papers do error-weighted tree/qubit selection.

### Cited Findings
**Aktar, Bärtschi, Badawy, Eidenbenz, IEEE TQE 3 (2022), arXiv:2112.12435, divide-and-conquer**
- Method: split the Hamming weight across two halves of n/2 qubits (fixed total weight k), then prepare each half with improved U^{n/2}_k.
- Hardware: up to n = 6 on IBM Quantum Sydney and Montreal, with "significantly higher" fidelity than Mukherjee et al. 2020 and Cruz et al. 2019.
- Connectivity: because heavy-hex is sparse, the circuits are implemented for LNN topologies.
- Source: [arXiv:2112.12435](https://arxiv.org/abs/2112.12435)
- A search-result summary says this approach cuts CNOT counts by up to 30% over earlier work. — [arXiv:2112.12435 (PDF)](https://arxiv.org/pdf/2112.12435) (figure from the search engine summary; not verified in the full text)
- BE22 notes that this "fixed weight split" is not recursive, so it gives no asymptotic depth gain, but it did lower gate count and raise IBMQ fidelity. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)

**Mukherjee, Maitra, Gaurav, Roy, "Preparing Dicke states on a quantum computer", IEEE TQE 2020, arXiv:2007.01681**
- Starting point: the B&E 2019 circuits are made of "partially defined" 2- and 3-qubit transformations, which only need to act correctly on the inputs that actually occur.
- They choose among the compatible unitaries to cut CNOT and single-qubit gate counts for 2 ≤ k ≤ n−2, and claim the most efficient deterministic circuit at the time. For k = 1 they match Cruz et al.'s linear W circuit.
- Ran D(4,2) on IBM ibmqx2 and saw lower error than the baseline.
- They define a "CNOT map" graph with n−k−1 independent modifications that redistribute CNOTs (different hardware embeddings and error profiles), and argue the optimisations reduce architectural constraints.
- Source: [arXiv:2007.01681](https://ar5iv.arxiv.org/html/2007.01681)

**Log-depth W / Dicke on grids requires measurements**
- The LAQCC model (Buhrman et al., arXiv:2307.14840) gives log-depth grid nearest-neighbour preparation of W states and Dicke states up to k = O(sqrt(n)), but uses mid-circuit measurements + classical feed-forward. Out of scope here except as a pointer. — [arXiv:2307.14840](https://arxiv.org/pdf/2307.14840)
- A 1D-chain W protocol (arXiv:2206.09929) reaches O(log N) with measurements. Without measurements its depth is T = N/2 + log2 N − 1, i.e. linear in N. — [arXiv:2206.09929](https://arxiv.org/pdf/2206.09929) (via search-result summary)
- BE22's comparison table lists Cruz et al. 2018 W states at O(log n) depth, O(n) CNOTs, all-to-all. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)

**Heavy-hex W states: 2505.03111 (scattering in QFT using W states)**
- W states are prepared on IBM heavy-hex devices (104 qubits of ibm_marrakesh, up to 5,589 two-qubit gates), extending earlier protocols with mid-circuit measurement and feed-forward. — [arXiv:2505.03111](https://arxiv.org/abs/2505.03111)
- A search summary describes the paper as viewing heavy-hex as "a linear chain plus some connections between segments", with those links used as ancillas to cut depth. — [arXiv:2505.03111 (PDF)](https://arxiv.org/pdf/2505.03111) (from the search-result summary; not verified in the full text)

**Single-step W states on a 2D lattice: Romeiro, Roy, …, Filipp, arXiv:2605.18962 (analog; pointer only)**
- Engineered simultaneous interactions spread one excitation across a lattice, so entangling time "scales only with the largest dimension".
- 6-qubit W in a 3×2 lattice: 83.9 ± 1.0% fidelity in 99 ns. 7-qubit chain: 79.6 ± 1.3% in 264 ns.
- Source: [arXiv:2605.18962](https://arxiv.org/abs/2605.18962)
- The heavy-hex example arranges qubits into distance-based layers; the dynamics stay in the span of layer superposition states. — [arXiv:2605.18962 (PDF)](https://arxiv.org/pdf/2605.18962) (from the search-result summary)

**Collision-model Dicke states on IQM Emerald (square-lattice superconducting): Vu, Nguyen, Özdemir, Müstecaplıoğlu, Ozaydin, arXiv:2608.30176 (approximate, numerically optimised; pointer only)**
- Partial-swap collisions with optimised angles. K shuttle qubits carry excitations.
- Routing on hardware added a median 25% extra two-qubit gates (90th percentile 56.7%) over 150 transpiled circuits.
- They chose a 14-qubit region of the 54-qubit Emerald by minimising summed gate error, with per-state layouts.
- Measured fidelities with one shallow round: D(5,2) ≈ 0.714, D(6,3) ≈ 0.525, D(8,3) ≈ 0.446, D(10,5) ≈ 0.261. Deep, optimised arms did much worse: shallow beats "better ideal fidelity".
- Source: [arXiv:2608.30176 (HTML)](https://arxiv.org/html/2608.30176v1)

**Noise-aware placement precedent (GHZ, not Dicke): arXiv:2312.15170**
- Tree-type minimal-depth GHZ circuits on heavy-hex devices of up to 414 qubits. Qubit subsets are chosen by weighting edges with error rates.
- The search summary notes performance on other topologies is untested.
- Source: [arXiv:2312.15170](https://arxiv.org/pdf/2312.15170) (from the search-result summary)
- The metriq-gym GHZ benchmark builds GHZ from a BFS spanning tree of the device graph, later switched to a binary dissemination tree. — [metriq-gym PR #736](https://github.com/unitaryfoundation/metriq-gym/pull/736) (from the search-result summary)

**Lower bounds summary**
- Light-cone: Ω(n2) on an n1×n2 grid, Ω(n) on a path, Ω(log n) all-to-all. — [arXiv:2505.15413](https://arxiv.org/abs/2505.15413)
- BE22 conjectures their grid depth is optimal on Θ(sqrt(n/k)) × Θ(sqrt(nk)) grids because it matches the graph diameter. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)

### Inferences
- W states on hardware: the measurement-free W circuit is a broadcast tree of two-qubit "split" gates (k = 1 WDB, ≈2 CZ each). On a hardware graph, depth is the broadcast time from a chosen root, roughly the patch radius. This matches our submission's broadcast-tree W with 2N−3 CZ. Choosing the root at the patch centre and the tree by edge error mirrors the GHZ error-weighted spanning-tree methods.
- Defect-aware, SWAP-free Dicke construction seems to be open in the literature. The papers either assume a perfect patch (BE22, Yuan–Zhang) or route generically (Wang et al. Mirror-SABRE; IQM collision-model transpilation with +25% median overhead). This supports our strategy of building directly on the error map.

### Gaps
- I found no measurement-free Dicke/W paper with an explicit heavy-hex embedding and gate counts. The heavy-hex W items found use feed-forward or analog dynamics.
- I found no Google Sycamore/Willow Dicke-state experiment.
- I found no paper that explicitly uses permutation symmetry to remove routing SWAPs for Dicke circuits.
- I did not verify Della Chiara et al. (arXiv:2507.20887, Dicke states for LCU block encodings). The abstract gives no connectivity details.

## Q5. Practical tricks across these papers (qubit placement, avoiding SWAPs, permutation symmetry)

### Takeaway
The useful tricks in the literature are:
- Work only on the "active" k qubits of each side (WDB locality).
- Thread snakes through tiles so that block endpoints are adjacent.
- Reverse wire order by flipping leaf orientation instead of SWAP networks.
- Use D(n, n−k) = X^{⊗n} D(n,k).
- Exploit partially defined unitaries (only the reachable inputs matter).
- Pick low-error hardware regions.
The gap left open is defect-aware, relabel-instead-of-route placement.

### Cited Findings
- The WDB acts only on the last k qubits of each part, so qubits are arranged so those two k-sets are LNN neighbours. Snakes in adjacent tiles are joined head-to-tail. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)
- Order reversal costs an O(k)-depth SWAP network in general. In the hand-tuned D(8,2) grid circuit it is instead absorbed by "a reversal of two wires" and by orienting two of the four U^2_2 leaves opposite to the others, for 31 CNOT. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)
- Unused qubits inside partly filled tiles block interactions and would need to be swapped away. BE22 avoids this with long row-snakes instead of full tiles. — [arXiv:2207.09998 (PDF)](https://arxiv.org/pdf/2207.09998)
- Choosing among the unitaries that agree on the reachable inputs reduces CNOT counts and architectural constraints. — [arXiv:2007.01681](https://ar5iv.arxiv.org/html/2007.01681)
- Generic routers over a fixed circuit cost a lot: +25% median two-qubit gates (IQM collision model); CZ fidelity collapses with coupler defects (Wang et al.). — [arXiv:2608.30176](https://arxiv.org/html/2608.30176v1); [arXiv:2608.24084](https://arxiv.org/html/2608.24084)
- Region selection by minimising summed gate error is used in practice. — [arXiv:2608.30176](https://arxiv.org/html/2608.30176v1)

### Inferences
- For our Willow pipeline, ranked by expected payoff at N ≤ 20:
  1. Build a 2-level BE22-style split (fixed-weight WDB^{N, N/2}_k accepting only l = k, then 1D leaves) directly on a compact low-error patch. Choose snakes so leaf endpoints are adjacent, and relabel outputs rather than swap.
  2. Use the 31-CZ D(8,2) grid circuit as a target / template check.
  3. For W states, use an error-weighted, centre-rooted broadcast tree.
  The asymptotic O(sqrt(nk)) and O(k log(n/k) + n2) schemes mainly pay off for larger N than the hardware instances likely use.

### Gaps
- No published CZ counts for scalable (non-hand-tuned) WDB gadgets at small k. BE22 calls its scalable WDBs not resource-efficient. Concrete constants would need our own synthesis.
