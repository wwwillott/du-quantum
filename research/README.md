# Research papers

Papers for Dicke-state preparation on Google Willow105 (native gates: `cirq.PhasedXZGate`, `cirq.CZ`; noise-aware layout + dynamical decoupling).

## 1. Primary algorithmic paper (must-read)

| File | Citation |
|------|----------|
| `01_Baertschi_Eidenbenz_2019_Deterministic_Preparation_of_Dicke_States.pdf` | Bärtschi & Eidenbenz (2019). *Deterministic Preparation of Dicke States*. [arXiv:1904.07358](https://arxiv.org/abs/1904.07358) |

Benchmark for preparing \(\lvert D_k^n\rangle\): linear-depth \(O(n)\) circuits via splitter / Givens-rotation networks; most deterministic Dicke compilers build on this.

## 2. Adaptive circuits & mid-circuit measurements (bonus)

| File | Citation |
|------|----------|
| `02a_Baertschi_Eidenbenz_2022_Short_Depth_Circuits_Dicke.pdf` | Bärtschi & Eidenbenz (2022). *Short-Depth Circuits for Dicke State Preparation*. [arXiv:2207.09998](https://arxiv.org/abs/2207.09998) |
| `02c_Buhrman_etal_2023_State_Prep_Shallow_FeedForward.pdf` | Buhrman et al. (2023). *State preparation by shallow circuits using feed forward*. [arXiv:2307.14840](https://arxiv.org/abs/2307.14840) |
| `02b_Yu_etal_2024_Efficient_Preparation_Dicke_MidCircuit.pdf` | Yu et al. (2024). *Efficient preparation of Dicke states*. [arXiv:2411.03428](https://arxiv.org/abs/2411.03428) |

- **02a** — short-depth deterministic unitary circuits (all-to-all / grid); improves on the 2019 linear-depth result.
- **02c** — mid-circuit measurement + classical feed-forward (LAQCC) for W and Dicke states; constant / log depth depending on \(k\). Closest match to the “adaptive / mid-circuit” bonus (`create_dicke_state_adaptive`).
- **02b** — adaptive mid-circuit collective measurements + feedback; polylog depth / ancillas.

> Note: no arXiv paper by *Piveteau* matches this Dicke mid-circuit topic; **02c** is the 2023 feed-forward / mid-circuit reference usually intended for that bullet.

## 3. Noise-aware mapping & grid routing

| File | Citation |
|------|----------|
| `03a_Li_Ding_Xie_2019_SABRE_Qubit_Mapping.pdf` | Li, Ding & Xie (2019). *Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices* (SABRE). [arXiv:1809.02573](https://arxiv.org/abs/1809.02573) |
| `03b_Murali_etal_2019_Noise_Adaptive_Compiler_Mappings.pdf` | Murali et al. (2019). *Noise-Adaptive Compiler Mappings for Noisy Intermediate-Scale Quantum Computers*. [arXiv:1901.11054](https://arxiv.org/abs/1901.11054) |

- **03a** — SABRE SWAP-based bidirectional routing for arbitrary coupling maps.
- **03b** — calibration / error-rate–aware initial layout and routing (complements SABRE for Willow105 `sq_pauli_error_rate`, `cz_pauli_error_rate`, readout errors).

## 4. Dynamical decoupling & native gate / architecture (Google Quantum AI)

| File | Citation |
|------|----------|
| `04a_Google_2019_Quantum_Supremacy_Sycamore_SI.pdf` | Google AI Quantum et al. (2019). Supplementary info for *Quantum supremacy using a programmable superconducting processor* (Sycamore). [arXiv:1910.11333](https://arxiv.org/abs/1910.11333) |
| `04b_Google_2024_Willow_QEC_Below_Threshold.pdf` | Google Quantum AI (2024). *Quantum error correction below the surface code threshold* (Willow). [arXiv:2408.13687](https://arxiv.org/abs/2408.13687) |
| `04c_Google_Willow_Spec_Sheet.pdf` | Google Quantum AI. *Willow Spec Sheet* (105 qubits, CZ / PhasedXZ-relevant metrics). [PDF](https://quantumai.google/site-assets/downloads/willow-spec-sheet.pdf) |

Practical Cirq DD / spin-echo guidance (XY4-style idle filling): [Circuit optimization, gate alignment, and spin echoes](https://quantumai.google/cirq/tutorials/google/spin_echoes) and [`cirq.add_dynamical_decoupling`](https://quantumai.google/reference/python/cirq/add_dynamical_decoupling).
