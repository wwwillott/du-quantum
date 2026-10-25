# Scope and numerical controls

## Questions and budgets

1. Does chain XY saturate the analytically certified D₂⁴ Slater ceiling?
   Two through eight matching layers; four seeded starts; max 350 L-BFGS-B
   iterations/700 function calls per start, with 12 s shared per candidate.
   Saturation confirms the implementation and constructive attainability;
   exceeding the ceiling signals an implementation/convention error.
2. Does one CP lift it, after full CZ compilation on the same path?
   The same depths/seeds and budgets; direct coherent fidelity and native
   count, not uniform sampling, determine the outcome.
3. Does the effect extend to n=6? Two depths and at most two CP gates,
   with the same per-candidate limits; only after the first milestone.
4. Does graph geometry matter? Four-site physical square and star,
   three depths each. Star success can falsify a universal extension of the
   chain restriction; square failure alone cannot prove impossibility.
5. Does reduced compiled cost survive noise? Small exact density matrices,
   synthetic independent gate channels plus damping/dephasing during every
   compiled moment, including idle qubits. Readout is a separate classical
   bit-flip model. Only the n=4 unpruned sensitivity test varies p₂; n=6 and
   pruned comparisons use one default setting. None is calibrated hardware.
6. Is a promising star leaf coherent across recursive weight branches?
   Two twelve-layer schedules, zero/one CP; all five sectors, common vacuum
   phase, at most 24 s objective evaluations total. The analytical derivative
   of this common-phase objective has a finite-difference test. Compiler and
   verification overhead is outside the optimizer's evaluation budget.

## Independent checks

Fixed-sector Hamiltonians are compared with independently tensor-built full
Hamiltonians, including exchange, fields, interaction, and the commutator
with N. Sparse and dense propagation checks use the same one-edge Hamiltonian.
Gate sequences are checked against Cirq in complex128 with explicit bit-order
conversion. Both ordinary and multi-sector gradients are checked with central
finite differences. The exact Slater ceiling is checked by SVD and a direct
maximizing decomposable state. Random XY chain circuits satisfy Plucker.
State-aware pruning is checked against the whole prepared state up to phase.
Noiseless density evolution and analytic amplitude-damping acceptance are
tested. Every saved native candidate passes topology and gate-set validation
and agrees with its logical ideal fidelity within 1e-9.

The n=4/n=6 full-space validations are deliberate independent checks;
optimization itself uses sectors of size binomial(n,l), never a dense full
Hamiltonian. Baseline pruning uses the full tiny state because baseline
intermediate gates can leave the excitation sector.

## Physical comparison limits

Static Willow105 topology is loaded through the existing repository API.
CZ/PhasedXZ compilation uses the repository compiler with decoupling disabled
for every method. Exact full-CZ gates and allowed physical couplers are checked
with both the repository checker and Willow105.validate_circuit. No native
arbitrary FSim control, pulse calibration, crosstalk schedule certification,
leakage model, live gate duration, or processor availability is assumed.
Validation certifies the static software device model, not live executability.

State-preparation tables apply a common state-aware elimination pass to every
method. Multi-sector blocks never use that pass. Recursive path/star baselines
include existing move-and-restore SWAP routing, which is valid but may be
suboptimal. An improvement against these implementations is not an asymptotic
or best-literature claim. Exact describes numerical state agreement, not an
algebraic proof of optimized angle values. Timing results are machine-specific;
two measured system sizes do not define a scaling law.

## Deferred work

Full hybrid integration, systematic calibration/parameter robustness, direct
noisy-objective optimization, symmetry-constrained ansatze, fixed-native pulse
compilation, optimized n>=8 preparation and 105-qubit studies require further
evidence. The analytic obstruction, fair small-system resource comparisons,
and coherent local-leaf result already satisfy useful research milestones.
