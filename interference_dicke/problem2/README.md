# Problem 2: exact states under the provided depth score

The [original-judges audit](../../judges_audit/README.md) subsequently promotes
two Will circuits and raises the verified score to **82.323036395**. The
standalone `problem2_submission.py` and notebook wrapper include them. The
82.152487741 result below documents the previous integration and its evidence.

The notebook entry point now selects this depth portfolio. Its complete,
independently checked score is **82.152487741**, across all **100** instances
with no skips. The lowest verified fidelity is **0.9999999999998976**; the
maximum measured construction time is **8.45 seconds** and the largest circuit
has **217 moments**. Per-case evidence is in [evaluation.json](evaluation.json).
The standalone file is [problem2_submission.py](../../problem2_submission.py);
it exports `create_dicke_state_adaptive(N, k)` without importing this repository
or reading artifact files. Its scored builder matches `portfolio.py`; a
concurrent workspace update also added product-state boundary handling outside
the scored domain. The notebook wrapper also admits the saved interference
experiments; none beats that file's
selected circuit. No hardware run or hardware-fidelity score is claimed.

## Scoring and scope

We use the actual `score_solution` cell in
`DuQuantum_Google_Standard.ipynb`:

```python
sum(N * k / len(circuit) for N in range(2, 21)
    for k in range(1, N // 2 + 1))
```

That is **100 instances**, including N=16..20. The nearby prose describes a
different N<=15, point-per-shallowest-instance competition. These are different
metrics. All score claims here refer to the provided executable sum, with
finite compiled moment counts, and no skipped or postselected cases.

The reusable portfolio is prior workspace work: the `_BONUS_SRC` literal in
`adaptive_dicke_validation/snapshots/kai-48115b9/submission.py`, attributed there
to the Astra bonus implementation. It was extracted using `ast.literal_eval`,
not by importing or executing the large submission snapshot. Changes extend
its public domain from 15 to 20 qubits, expand its cache, and fix its otherwise
out-of-range balanced adaptive layout calls at N=16,18,20. The greater-than-50
result comes from integrating and extending that portfolio, **not from claiming
a new interference or Yu speedup**.

## Applying the interference experiments

The saved free-chain circuits at F=0.9714 and F=0.9, and the approximate
six-qubit two-CP circuit, cannot replace an exact Dicke preparation. The
admission check independently verifies the exact chain/star candidates and
compares their actual compiled depths. They are deeper than the selected
portfolio circuits (D(4,2): 7 moments; D(6,3): 13).

`hybrid.py` also performs the previously missing complete coherent integration:
a hypergeometric weight-distribution block followed by two four-qubit star
leaves, each using the same certified five-sector, common-phase unitary. The
right-aligned unary registers are reversed to match the leaf input convention.
Initialization and state-specific pruning are never inserted into a coherent
leaf. Both the hybrid and SCS comparison use the same prefix and physical graph.

| Full D(8,4) preparation | Moments | CZ | Ideal fidelity | Score contribution |
|---|---:|---:|---:|---:|
| Coherent interference leaves | 6882 | 3491 | 0.9999999999953 | 0.004650 |
| Repository SCS leaves, same routing | 7308 | 3918 | 0.99999999999999 | 0.004379 |
| Selected existing portfolio | 23 | 38 | 1 to numerical precision | 1.391304 |

The local interference improvement survives composition, saving 427 CZ and
426 moments, but naive weight-distribution routing dominates. The hybrid is
also over the 1750-moment cap and is excluded. This is evidence about the
implemented routing, not a lower bound on possible hybrids.

## Applying the mid-circuit paper

We checked Yu et al., *Efficient preparation of Dicke states*,
[arXiv:2411.03428](https://arxiv.org/abs/2411.03428), including Algorithm 1 and
Eq. (2). Its measurement is a **coherent collective weight measurement**,
not measurement of individual data bits. Its logarithmic bound concerns an
expected number of collective rounds, which is not `len(circuit)`.

`paper.py` implements the symmetric-space rotation matrix and geometric
feedback policy. It records the probability of failure after a finite number
of rounds rather than declaring the remaining branches successful. Those
round counts are explicitly ineligible for the challenge depth score.

`yu_circuit.py` is an actual finite Willow circuit for D(4,2):

1. Apply the global binomial-mode rotation.
2. Coherently accumulate weight into three Fourier ancillas and apply inverse
   QFT, using full CZ and single-qubit gates after routing.
3. Measure the complete physical readout diagonal of the ancillas, keeping
   every data wire on other diagonals.
4. On an unsuccessful weight w, map D_w back to unary(w), change it to
   unary(2), and decode it to D_2 with an exact conditioned unitary.

This exact completion is our finite substitute for the paper's retry loop,
not the paper's asymptotic algorithm. All five outcomes are retained and
independently verified. Their probabilities are
`[1/16, 4/16, 6/16, 4/16, 1/16]`; every branch has target fidelity 1 to numerical
precision. With routing, readout and every conditional correction counted,
the circuit has **1149 moments**, contribution **0.006962576**, versus 7 moments
and contribution 1.142857 for the selected portfolio. It therefore loses the
given score despite exploiting the correct collective-measurement primitive.

## Independent verification

`verify.py` applies native gates to full complex128 vectors independently of
Cirq's simulator, checks output fidelity and residuals, and traces any
ancillas. A separate test compares its amplitudes with Cirq. Every selected
measurement/complement branch is enumerated without postselection.

The large D(10,5) cat/teleportation circuit has 38 active wires, so it is not
validated by pretending a 20-qubit simulation includes all ancillas. Instead:

- The 11-wire non-Clifford core is checked by a full vector, including that
  temporary ancillas return to zero.
- An arbitrary-input Choi state checks the actual Clifford resource and
  suffix against the desired coherent complement/teleportation isometry.
- Clifford tableau signs are affine in independent measurement bits; all-zero
  and each basis-bit assignment certify all **262144** outcomes of its
  **18** independent bits in **19** checks. Conditional corrections are Pauli
  gates controlled by linear parity expressions.
- The production scheduler retains shared-wire dependencies, complete readout
  blocks and classical key causality. Allowed diagonal commutations are exact.

Every scored row checks physical Willow operations, complete readout diagonals,
causal controls, construction under a minute, and fidelity above `1 - 1e-9`.
Unitaries additionally pass `Willow105.validate_circuit`. Conditional operations
are checked through their native underlying operations, because the device's
hardware validator does not model the bonus classical controller.

## Reproduce

From the repository root:

```powershell
python -m interference_dicke.problem2.evaluate
python -m interference_dicke.problem2.research
python -m pytest interference_dicke/tests dicke_willow/tests/test_dicke.py -q
```

`evaluation.json` contains one row per scored case, exact formula, timings,
methods, versions and portfolio SHA256. `clifford_certificate.json` records the
additional zero-ancilla check. `experiments/results.json` and its Cirq JSON files
retain the full interference and finite-Yu comparisons. Experiment scripts
overwrite only these dedicated Problem 2 result files; the original interference
research artifacts and submission snapshots are preserved.

The standalone scored implementation comes from `portfolio.py`; preserve its
additional product-state boundary handling when regenerating it. Its cold
D(20,10) build was separately checked:
217 moments, 20 outputs, 6.74 seconds, valid Willow connectivity.
