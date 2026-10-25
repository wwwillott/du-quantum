# Walkthrough: State- and $N$-Aware Error Shifting

We implemented state- and $N$-aware error modeling across [`dicke_submission.py`](file:///Users/willott/School/Hackathons/DuQuantum/Google/du-quantum/dicke_submission.py), [`src/dicke_compiler/scoring.py`](file:///Users/willott/School/Hackathons/DuQuantum/Google/du-quantum/src/dicke_compiler/scoring.py), and [`testing/problem1_variants.py`](file:///Users/willott/School/Hackathons/DuQuantum/Google/du-quantum/testing/problem1_variants.py).

## Changes Made

### 1. Authoritative Submission Artifact ([`dicke_submission.py`](file:///Users/willott/School/Hackathons/DuQuantum/Google/du-quantum/dicke_submission.py))
- **Replaced static log-loss summation in `_robust_calibration_cost`** with an analytical judge surrogate directly predicting the official denominator $\mathbb{E}[\text{Score}_Z + \text{Score}_X]$:
  - **Z-Subspace Retention Floor**: Tracks physical circuit moment durations ($42\text{ ns}$ CZ, $25\text{ ns}$ 1Q) and active-wire $T_1$ relaxation scaled by excitation density $p = k/N$:
    $$L_{T_1} = \frac{k}{N} \sum_{q \in \text{outputs}} \frac{T_{\text{active}}(q)}{T_1}, \quad \text{subspace floor } \approx \left(1 - \frac{1}{\binom{N}{k}}\right) \frac{1}{f_Z}$$
  - **Dicke-Weighted Readout Dispersion**: Replaced the static `1e-3 * bounded_dispersion` with the exact multinomial chi-squared variance prefactor:
    $$\text{disp\_scale} = \frac{S \cdot f_Z}{\binom{N}{k}} \cdot \frac{k(N-k)}{N(N-1)}$$
    This correctly penalizes non-uniform readout for small $\binom{N}{k}$ (e.g. $D(4,2), D(5,2), W$-states) while allowing $f_Z$ and $\text{Score}_X$ to dominate for large balanced states.
  - **Symmetric X-Readout and Correlator Scaling**:
    - Modeled symmetric X-basis readout loss $-(1 - (e_0 + e_1))$ and final Hadamard gate error $\frac{4}{3} e_{\text{sq}}$.
    - Added idle dephasing $T_\phi$ scaled by superposition coherence $4p(1-p) = \frac{4k(N-k)}{N^2}$.
    - Evaluated all $\lfloor N/2 \rfloor$ even-weight correlators $\ell = 2, 4, \dots, 2\lfloor N/2 \rfloor$:
      $$\widehat{\text{Score}}_X = \sum_{m=1}^{\lfloor N/2 \rfloor} \left(1 - \exp\left(-\frac{2m}{N} L_X\right)\right)^2 + (\bar{\Delta}_{\text{ro}})^2$$

### 2. Reusable Compiler Package ([`src/dicke_compiler/scoring.py`](file:///Users/willott/School/Hackathons/DuQuantum/Google/du-quantum/src/dicke_compiler/scoring.py))
- Implemented [`projected_judge_penalty`](file:///Users/willott/School/Hackathons/DuQuantum/Google/du-quantum/src/dicke_compiler/scoring.py#L66-L168) and exported it in [`src/dicke_compiler/__init__.py`](file:///Users/willott/School/Hackathons/DuQuantum/Google/du-quantum/src/dicke_compiler/__init__.py) so that `quicke-states` users can inspect both raw survival probability and $(N, k)$-shifted judge penalties.

### 3. Variant Harness ([`testing/problem1_variants.py`](file:///Users/willott/School/Hackathons/DuQuantum/Google/du-quantum/testing/problem1_variants.py))
- Updated `circuit_calibration_cost` to delegate directly to `_robust_calibration_cost` from `dicke_submission.py`, keeping benchmark rankings aligned with live submission selection.

---

## Verification Results

### Unit Tests and Linter
- **`pytest`**: All 26 tests in `tests/test_api.py`, `tests/test_passes.py`, `tests/test_templates.py`, and `tests/test_verification.py` passed (`26 passed in 29.28s`).
- **`ruff check`**: All checks passed with zero errors or warnings.

### Large-$N$ Scaling ($10 \le N \le 15$)
Ran structural verification across $W$-states and balanced states under synthetic calibration maps:
- **$(10, 1)$**: `runtime_anytime` selected a lower-cost candidate ($1.076$ vs. $1.093$).
- **$(15, 7)$**: `runtime_anytime` selected a lower-cost candidate ($6.383$ vs. $6.434$ on seed 2026, and $6.467$ vs. $6.548$ on seed 7).
- **Runtime**: Median construction time was $1.72\text{s}$ (max $2.37\text{s}$), well below the 60s judge deadline.

### Noisy VQM Simulation Benchmark
Simulated across stratified test cases $(4,2), (5,2), (6,1), (6,3)$ with full Pauli, depolarizing, and asymmetric readout channels:
- **Total Estimated Score**: `runtime_anytime` achieved **`9.645`** vs. `shallow_incumbent` **`9.527`**.
- **Score Stability**: `runtime_anytime` had lower score standard deviation across seeds (`0.837` vs. `0.882`).
- On $(5, 2)$ seed 7, `runtime_anytime` reduced depth from 18 to 17, CZ count from 13 to 10, increased retention to $96.1\%$, and raised the judge score from `1.778` to `2.437`.
