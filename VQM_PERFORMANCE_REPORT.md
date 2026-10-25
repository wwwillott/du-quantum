# Dicke State Performance & VQM Simulation Report

This report presents empirical benchmarks across all 56 symmetric Dicke targets $2 \le N \le 15, 1 \le k \le \lfloor N/2 \rfloor$ using Google Willow hardware constraints and Cirq's calibrated Virtual Quantum Machine (VQM).

---

## 1. Runtime Guarantee: Under 60 Seconds on All Targets

The competition requires that circuit generation executes in under **60 seconds** per $(N, k)$ target.

Across all 56 targets:
- **Maximum Synthesis Time**: **`4.50s`** (for $D(8, 4)$ with full multi-candidate graph exploration).
- **Median Synthesis Time**: **`1.24s`**.
- **95th-Percentile (p95)**: **`2.53s`**.
- **Safety Margin**: Over **$13.3\times$ faster** than the 60.0s deadline in the worst case, and **$>48\times$ faster** on average.

![Synthesis Runtime vs 60-Second Competition Limit](/Users/willott/.gemini/antigravity-cli/brain/6779596d-9ecf-47f9-923f-023e9bc53e6c/construction_time_margin.png)

> [!NOTE]
> The internal anytime scheduler enforces a hard deadline (`_TEAM_RUNTIME_DEADLINE = started + 50.0`), guaranteeing that a valid incumbent is returned safely before the competition evaluator times out.

---

## 2. Circuit Depth & CZ Gate Count Reductions

Our portfolio selects among specialized analytic constructions, pre-optimized native templates, and pruned ladder embeddings depending on the target family.

![Circuit Depth and CZ Scaling](/Users/willott/.gemini/antigravity-cli/brain/6779596d-9ecf-47f9-923f-023e9bc53e6c/circuit_efficiency_metrics.png)

### Key Architectural Advantages:
1. **$W$-States ($k=1$ and anti-$W$ $k=N-1$)**:
   - **Branching $W$-Tree**: Replaces the $O(N)$ depth of the standard Bärtschi-Eidenbenz linear chain with an optimal **$O(\log N)$** tree depth.
   - For $N=15, k=1$: Depth drops from **54 moments down to 14 moments** ($74\%$ depth reduction!), drastically reducing idle phase damping ($T_2$).
2. **Balanced States ($k = N/2$)**:
   - Our native Willow grid templates and first-column Givens rewrites eliminate redundant reverse-sweep swaps.
   - **$D(4,2)$**: 5 CZs in 7 moments (vs. 9 CZs in 16 moments on the line, **$-44\%$ CZs**).
   - **$D(6,3)$**: 14 CZs in 18 moments (vs. 29 CZs in 45 moments on the line, **$-52\%$ CZs**).
   - **$D(8,4)$**: 24 CZs in 23 moments (vs. 61 CZs in 87 moments on the line, **$-61\%$ CZs**).

---

## 3. Calibrated VQM Noisy Simulation Results

We evaluated representative targets under realistic Willow noise channels (single-qubit depolarizing, coupler CZ Pauli errors, asymmetric readout noise $0\to 1$ and $1\to 0$, and idle dephasing $T_\phi$):

![VQM Simulated Hardware Judge Score by State](/Users/willott/.gemini/antigravity-cli/brain/6779596d-9ecf-47f9-923f-023e9bc53e6c/vqm_judge_score_comparison.png)

### Observations from the Judge Score Breakdown:
- **Small-$N$ Dominance**: For small $N$ ($N \le 4$), within-subspace uniformity dominates. On $D(4,1)$, our state-aware selection achieves an estimated judge score of **`13.72`** (vs. `12.89` for baseline).
- **Penalty Shift from $\text{Score}_Z$ to $\text{Score}_X$**:
  - For $W$-states ($k=1$), $\text{Score}_X$ is negligible ($< 1\%$ of penalty), and $\text{Score}_Z$ is dominated by the $0\to 1$ false-excitation rate.
  - For balanced states ($k \approx N/2$), as $N$ grows ($D(6,3), D(8,4)$), $\text{Score}_X$ multi-qubit correlator damping increases, requiring our shallower native templates to preserve high-order parity correlations.

---

## 4. Benchmark Data Summary Table

| Target $D(N, k)$ | Our Moments | Baseline Moments | Our CZs | Baseline CZs | Synthesis Time | VQM Judge Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$D(2, 1)$** | **3** | 3 | **1** | 1 | 0.05s | **8.193** |
| **$D(3, 1)$** | **6** | 7 | **3** | 3 | 0.45s | **5.972** |
| **$D(4, 1)$** | **7** | 11 | **5** | 5 | 0.48s | **13.715** |
| **$D(4, 2)$** | **7** | 16 | **5** | 9 | 0.30s | **1.803** |
| **$D(5, 1)$** | **11** | 15 | **7** | 7 | 0.92s | **1.849** |
| **$D(5, 2)$** | **14** | 22 | **9** | 13 | 0.64s | **1.160** |
| **$D(6, 1)$** | **10** | 19 | **9** | 9 | 0.97s | **1.965** |
| **$D(6, 3)$** | **18** | 45 | **14** | 29 | 1.30s | **0.967** |
| **$D(7, 1)$** | **10** | 23 | **11** | 11 | 1.16s | **0.471** |
| **$D(7, 3)$** | **27** | 59 | **21** | 38 | 1.84s | **1.282** |
| **$D(8, 1)$** | **11** | 27 | **13** | 13 | 1.71s | **1.111** |
| **$D(8, 4)$** | **23** | 87 | **24** | 61 | 4.50s | **0.818** |
| **$D(10, 1)$** | **15** | 35 | **17** | 17 | 1.74s | — |
| **$D(10, 5)$** | **73** | 133 | **73** | 105 | 1.24s | — |
| **$D(12, 6)$** | **101** | 179 | **117** | 161 | 1.82s | — |
| **$D(14, 7)$** | **135** | 225 | **171** | 229 | 2.65s | — |
| **$D(15, 1)$** | **14** | 54 | **27** | 27 | 2.23s | — |
| **$D(15, 7)$** | **155** | 243 | **218** | 258 | 1.71s | — |
