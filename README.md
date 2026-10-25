# Quicke States

`quicke-states` is a Cirq library for verified Dicke-state circuit synthesis,
optimization, hardware placement, and reproducible benchmarking.

The compiler is deliberately explicit about equivalence:

- **unitary-safe passes** preserve a circuit on every input;
- **state-preparation optimization** may replace a circuit when both prepare
  the same declared Dicke state from the all-zero input;
- unsupported large or arbitrary circuits are reported as unverified rather
  than silently treated as exact.

## Installation

```bash
pip install quicke-states
```

For a source checkout, Willow support and development tools are optional:

```bash
pip install -e ".[willow,dev]"
```

## Compile a Dicke state

```python
from dicke_compiler import compile_dicke

result = compile_dicke(8, 4, objective="depth")

print(result.circuit)
print(result.after_metrics)
print(result.verification)
```

The default output uses `cirq.LineQubit` with full CZ and
`cirq.PhasedXZGate` operations.

## Optimize an existing preparation

```python
from dicke_compiler import DickeTarget, optimize_dicke

result = optimize_dicke(
    existing_circuit,
    DickeTarget(8, 4),
    output_qubits=ordered_output_qubits,
    equivalence="state_preparation",
    objective="cz_depth",
)
```

The result records its selected family, applied passes, before/after metrics,
runtime, and verification evidence.

## Register a research template

`CircuitTemplate` and `TemplateRegistry` let researchers add verified
state-preparation constructions without modifying compiler internals. The
compiler searches bounded graph embeddings, verifies the template when
feasible, and compares it with analytic generators under the selected
objective.

The packaged registry covers all 56 symmetric challenge targets
`2 <= N <= 15`, `1 <= k <= floor(N/2)`. Each compact, offline-generated
template retains its source revision, a SHA-256 provenance hash, and its
known-valid Willow placement. The compiler validates the asset at import,
uses bounded fallback embedding on other coupling graphs, and still compares
templates with analytic generators. These are the best constructions
currently known to this project, not proofs of global optimality.

The analytic generator portfolio includes:

- `line_shallow`, the default reachable-subspace line construction;
- `line_basic`, an independent line baseline without shallow row rewrites;
- `scs_path`, an independent split-and-cyclic-shift reference family;
- topology-aware `w_tree` and `anti_w_tree` families selected automatically
  for one-excitation and one-hole targets.

The SCS family is intentionally opt-in because it is a useful independent
reference but is usually deeper after nearest-neighbor routing:

```python
result = compile_dicke(6, 3, methods=("scs_path",))
```

Current Willow comparison results are stored in
[`benchmarks/incumbent_comparison.csv`](benchmarks/incumbent_comparison.csv).
The reusable compiler matches the incumbent's 7-moment `D(4,2)` and 13-moment
`D(6,3)` circuits. Its selected `D(8,4)` template uses 21 moments instead of
23 at the same CZ count. These are reference results, not claims of global
optimality.

## Use a coupling graph and current calibration

```python
from dicke_compiler import (
    CalibrationData,
    HardwareTarget,
    compile_dicke,
)

calibration = CalibrationData(
    single_qubit=sq_errors,
    two_qubit=cz_errors,
    readout_zero=zero_to_one_errors,
    readout_one=one_to_zero_errors,
)
hardware = HardwareTarget.from_edges(
    device_qubits,
    device_couplers,
    calibration=calibration,
)

result = compile_dicke(
    6,
    3,
    hardware=hardware,
    objective="projected_survival",
)
```

Projected survival is an uncertainty-adjusted independent gate/readout proxy.
It is useful for ranking candidates on current calibration data, but it is not
presented as a measured hardware fidelity or competition score.

For Google Willow:

```python
from dicke_compiler import compile_dicke, willow_target

result = compile_dicke(6, 3, hardware=willow_target())
```

## Command line

```bash
dicke-compile 6 3 --objective depth --print-circuit
```

## Guarantees and current limits

- Exact complex128 statevector verification defaults to absolute tolerance
  `1e-8` for up to 15 qubits.
- Built-in analytic generators can return an analytic provenance report above
  the exact-simulation limit.
- Version 0.1 supports deterministic unitary state preparation. Adaptive
  measurement/feed-forward synthesis is not yet part of the stable API.
- Candidate search is bounded and does not claim globally minimum depth.
- State-aware row templates are valid on their documented reachable
  subspaces, not as arbitrary full-unitary replacements.

## Research basis

The line synthesis is based on:

> A. Bärtschi and S. Eidenbenz, “Deterministic Preparation of Dicke States,”
> arXiv:1904.07358 (2019).

Additional papers and hardware references are catalogued in
[`research/README.md`](research/README.md).

## Development

```bash
pytest
ruff check src tests
python benchmarks/benchmark_compiler.py --max-qubits 15 --domain half
```

Competition submission artifacts remain separate from the package. New
compiler results should be independently benchmarked before being copied into
those artifacts.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
