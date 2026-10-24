# Willow embedding

## Device

Static development uses `cirq_google.Willow105`: 105 `GridQubit`s and 182 couplers. The coupling graph is read from `device.metadata.qubit_pairs`. No edge is added by hand.

Live hardware is not queried from this repository. `topology.graph_from_error_rates` builds the same kind of graph from the dictionaries the notebook passes in:

- a qubit is absent if it is missing from `sq_pauli_error_rate`, `zero_errors`, or `one_errors`
- a coupler is absent if it is missing from `cz_pauli_error_rate`

Those absences are the disabled qubits and couplers. The circuit is checked against that graph with `assert_challenge_gates` before it is returned. `Willow105.validate_circuit` is the check for `create_dicke_state_adaptive`, which has no error map and therefore uses the full static device.

## Placement

The hardware circuit is a path, because the 2019 stair is a path algorithm.

`place_path` is a beam search (width 48, 24 quiet starting qubits). The cost of path edge \(i\) is

\[
(\text{CZ error of that coupler}) \times (\text{number of logical multi-qubit gates whose span covers edge } i)
\]

plus each qubit's single-qubit Pauli error and its readout error at excitation fraction \(k/n\),

\[
(1-k/n)\,\varepsilon_0 + (k/n)\,\varepsilon_1.
\]

The adaptive question has no calibration. `snake_path` walks Willow rows and reverses every other row, keeping a step only when the coupler exists. A 16-qubit example from the static device, written by the benchmark script:

```text
q(0, 6) q(0, 7) q(0, 8)
q(1, 8) q(1, 7) q(1, 6) q(1, 5)
q(2, 5) q(2, 6) q(2, 7) q(2, 8) q(2, 9) q(2, 10)
q(3, 10) q(3, 9) q(3, 8)
```

`benchmarks/willow_snake_16.svg` is that path. The noise-aware path for a real error map is a different 16-tuple; it is chosen at call time from the dictionaries, not stored here.

## Logical-to-physical map

For the challenge circuit the logical stair index \(i\) starts on path qubit \(i\). Routing SWAPs move the computational state. The returned qubit set is the whole path. Because \(|D_k^n\rangle\) is symmetric, the notebook's sorted-coordinate order is a legal order to test against.

For the balanced tree, `DickeRegion.qubits` is the logical slice, `hot_left` and `hot_right` are the unary registers the divide block touches, and `children` are the two regions that run in parallel after the divide. `routing.describe_route` prints one region's SWAP layers in the form

```text
recursion level 2
region: 24 physical qubits
before:
weight register A -> [...]
routing layer 0:
SWAP q(r,c)<->q(r',c')
```

## Compilation

1. Logical gates: \(R_y\), CNOT, controlled-\(R_y\), SWAP.
2. `cirq.CZTargetGateset` rewrites them to `PhasedXZGate` and `CZPowGate`.
3. `RouteCQC` on the path graph inserts SWAPs. SWAPs are compiled again, to three CZs plus PhasedXZ.
4. Adjacent single-qubit gates on the same qubit are merged and compiled back, which is what brings \(N=20\) from about 1900 moments to 1740.
5. `cirq.add_dynamical_decoupling` fills idle slots. The result is compiled again. If that exceeds 1750 moments, the undecoupled circuit is the one returned.

`python main.py --hardware` prints the processor id, the number of active Willow105 qubits, the selected path, the moment count, the CZ count, and the repetitions. It does not submit a job.
