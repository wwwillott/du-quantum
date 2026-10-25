# Head-to-head evaluation methodology

These results are CPU simulator proxies. They are not Willow hardware results.

## Scope and frozen submissions

The requested matrix is all 36 pairs with `N=2..12` and `1<=n<=N//2`, on calibration seeds 101 and 202, for four candidates: 288 construction records. Construction prioritizes `N=6..12`. Each candidate runs in its own subprocess; the supervisor limits active construction and scoring workers to five. The interpreter is `C:/Users/kaisa/miniforge3/python.exe`. GPU use is disabled and the `dqgpu` environment is not used. All evaluation code and output are under `team_eval/`.

Candidates are copied once into `team_eval/sources/` and imported from those frozen copies. Construction records retain each copy's SHA-256, circuit SHA-256, calibration fingerprint, source path, construction time, and import time. Import time is separate from the requested construction time. Already persisted records are reused on resume.

| Candidate | Requested source | Frozen copy SHA-256 |
|---|---|---|
| A, reference | `team_eval/will_dicke_submission.py` | `e618641a745a98b2f63ec7e080d525ca2bf8a94c94ab3e6875a334dd8dc7f920` |
| B | `submission.py` | `93576dfc4bada22fa7b3e4742bd1eee29e807f41892091a31d46414df2fadb65` |
| C | `columnforge/noise_aware/rewrites/submission_scheduled.py` | `5745ac827fd33e638ddbf19c1f8002281f885c332fe53cd20e105f3e0c074bc7` |
| D | `astra/submission.py` | `cfbe6a6095226aaa6e147889f2ee92e6e208e0fb9ad9ad3a1f63d3dd23825aa5` |

The official notebook SHA-256 observed for this run is `360fe891a73b0e402dfd1cff1730ba8a3c83c5eff77f0894efb7eb6473e5df6a`. The observed `tools/sim.py` SHA-256 is `39a53de0428c2503ab65407b85af3f7d9d72b8f10716b7030758758f95c8cd3a`. Each scoring result records the simulator source hash actually imported; the simulator file itself is not edited or copied by this evaluation.

At the documented CPU-backend selection, the scoring helper `score_circuit.py` SHA-256 is `e09591bf9aac441bc01a7404e38ff26caca2fc220c2abaf870855362a69ac2cc`, and the corresponding `backend.json` SHA-256 is `d1bec6b16f2f8c2413744e3c747545dc46cdd53e41c4e421d325ee3c76990ce5`. These are provenance snapshots; existing cache entries assume fixed scoring settings and are not automatically invalidated by source-file changes.

## Shared calibration and canonical ordering

`official_calibration.py` contains the notebook's three fake-calibration generator functions verbatim. A direct source-substring check passes. For each seed, one `np.random.default_rng(seed)` is explicitly passed to every generator in notebook call order:

1. Single-qubit dropouts, probability 0.01.
2. Single-qubit Pauli error, median `4e-4`, spread `0.5`, uncertainty `2e-4`.
3. Two-qubit Pauli error with notebook defaults: median `2.5e-3`, spread `0.5`, coupler dropout probability `0.01`, uncertainty `5e-4`, respecting the single-qubit dropouts.
4. Readout zero errors, median `5e-3`, spread `0.5`, uncertainty `2e-4`.
5. Readout one errors, median `1e-2`, spread `0.5`, uncertainty `3e-4`.

The notebook iterates Willow metadata frozensets. Generation is performed under `PYTHONHASHSEED=0`, then saved once to `calibrations/seed_101.json` and `calibrations/seed_202.json`. Every builder and scorer loads these shared files and verifies their fingerprint. Each constructor receives its own dictionary copies.

Persistence canonicalizes dictionary insertion order by sorting qubit coordinates and sorting CZ pair endpoints. Thus, an original generator CZ tuple `(b,a)` may load as `(a,b)`, and insertion order differs from metadata-frozenset order. No qubit identity, undirected coupler, numerical error rate, uncertainty, or dropout is changed. All candidates receive the same canonical representation. This common ordering can affect tie-breaking in an order-sensitive candidate, so results should be reproduced from the saved calibrations, not from an independently regenerated calibration dictionary.

| Seed | Available qubits | Available couplers | Canonical calibration SHA-256 |
|---|---:|---:|---|
| 101 | 105 | 181 | `3e78cf23ad04e115b3dfacc7071885dc63f1d668394387076ec750f2195decac` |
| 202 | 103 | 173 | `bc3b6d6ad048a333cfa6a07cafbf30268039ae1f621ea3e449e26a17489a1f42` |

Both fingerprints reproduced in a second isolated Python process. `calibration_repro_check.json` preserves that evidence. This check establishes equality of canonical physical maps, not equality of original CZ tuple orientation or insertion order.

## Exactness and validity

Every emitted circuit is saved with `cirq.to_json`. The noiseless Dicke fidelity is calculated in complex128 on the returned state qubits, with all other circuit qubits treated as ancillas and traced out. Unitary circuits use the projected statevector norm; circuits containing reset use a density matrix and partial trace. A circuit passes only when `fidelity > 1 - 1e-8`.

Validity checks require exactly N returned GridQubits in a set, calibrated state qubits, full CZ gates only on calibrated undirected pairs, PhasedXZ only on calibrated qubits, allowed resets only on calibrated qubits, no measurements or unsupported operations, at most 1750 moments, and successful `cirq_google.Willow105.validate_circuit`.

Construction exceeding 60 seconds is recorded as a failure flag. The circuit and any available proxy scores are still retained for diagnosis; a numerical proxy score does not erase a timing, validity, or exactness failure. `NotImplementedError` and other exceptions are recorded with their exception text. Invalid or non-exact circuits are excluded from scoring by the supervisor.

## Simulation and scoring

The scoring entry point is `score_circuit.py`. It calls `tools/sim.py:simulate`, followed by `tools/sim.py:score_stats`, using `K=N`, 10,000 shots per basis, and 400 independent shot resamplings. It reports both the median and mean. The resamplings quantify shot variability conditional on the estimated trajectory distributions; they do not eliminate finite-trajectory Monte Carlo uncertainty.

| Model | Trajectories | Hardware settings |
|---|---:|---|
| `pauli` | 2000 | `HW(T1=1e9, Tphi_white=0, T2star=0)` |
| `hw` | 1000 | Default `HW()` |

Default `HW()` uses T1=68 microseconds, white-dephasing Tphi=200 microseconds, quasi-static T2*=30 microseconds, 25 ns single-qubit moments, 42 ns CZ moments, and 160 ns reset moments. Residual ZZ and coherent over-rotation/phase errors are zero by default. The `pauli` choice retains the existing simulator implementation; its finite but extremely long T1 is effectively disabled at these gate durations.

Trajectory and resampling RNGs are independent and deterministic. For each purpose the seed is the first eight SHA-256 digest bytes, interpreted little-endian, of `duquantum-head-to-head-v1:{seed}:{N}:{n}:{model}:{purpose}`, where purpose is `trajectories` or `resampling`. Candidate identity and scheduling order do not enter this seed. Different circuit structures consume random numbers differently; equal RNG seeds do not imply identical fault locations.

The reference arithmetic backend is NumPy. The selected CPU Numba backend replaces only Batch arithmetic kernels while retaining the original simulator noise operations, RNG draws, and scorer. `backend.json` records comparisons against NumPy on N=6 and N=12 circuits, under both noise models with matching RNG seeds and 64 trajectories: maximum absolute probability difference `9.367506770274758e-17`. These small comparison runs validate the arithmetic replacement; reportable circuit scores still use the full trajectory counts above. Each score records the backend actually used. BLAS/OpenMP worker thread counts are limited to one.

Byte-identical emitted circuit JSON can reuse a score only when returned qubits, calibration fingerprint, N, n, and calibration seed also match. This saves repeated scoring of identical circuits across candidates. The current cache assumes the fixed settings above; simulator/scorer source hashes and sampling parameters are recorded provenance rather than cache-key components. Changing simulator semantics, scorer arithmetic, calibration construction, or sample counts requires a new cache. Candidate source snapshots remain frozen, and each record retains their source hash even when an identical circuit shares a score with another candidate.

## Official-score parity

`score_parity_check.py` executes the official notebook scoring cell unchanged and compares it with `tools.sim.score_from_counts`. Twelve sampled count sets cover N=2,3,4,6,8,12, ideal Dicke distributions and noisy nonuniform distributions, all with K=N and 10,000 shots per basis. Every score, Z term, and X term passes tolerance; maximum absolute difference is `2.2737367544323206e-13`. Raw evidence is in `score_parity_results.json`.

The documented edge difference is zero observed Z shots in the correct-weight subspace: the official notebook returns NaN through division by zero, whereas `tools.sim` returns score zero by convention. This edge is separate from the twelve passing comparisons.

## Incremental output and recovery

Construction records and model scores persist incrementally. `results.json` and `RESULTS.md` may therefore represent a partial matrix; missing results must not be counted as zero or represented as complete. Per-model statistics can appear before both models finish. JSON publication uses atomic replacement, with up to 30 attempts separated by 0.1 seconds on Windows `PermissionError` to tolerate brief shared-reader locks. Workers already running before this retry change retain their previously loaded code.
