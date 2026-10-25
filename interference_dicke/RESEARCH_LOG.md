# Research log — 2026-10-25

1. Inspected existing `dicke_willow` and `willow_dicke` modules. Reused their
   topology, compiler, SCS, recursion, routing and gate-validation functions.
   Initial tracked/untracked bytecode changes predated this work; preserved.
2. Derived excitation conservation, current equations, Dyson-history phases,
   free-chain Slater restriction and exact D₂⁴ bound. Derived a connected
   graph Laplacian ground-state construction, overlap-conservation caveat,
   locality bound and necessary branch coherence for recursion.
3. Implemented cached fixed-weight direct gates and analytic adjoint gradients.
   An optional threadpoolctl import was unavailable; removed the dependency
   and set BLAS thread environment variables for the primary experiment.
4. Completed n=4 sanity/obstruction/interaction milestone; saved raw runs.
   One CP at sufficient depth overcame the proven chain ceiling.
5. Compiled and verified SCS and routed recursion on identical Willow paths.
   Extended selectively to n=6; one CP again attained unit numerical fidelity.
6. Tested actual Willow square/star geometries. Star XY+Z prepares D₂⁴;
   square stayed at the ceiling. Preserved every negative shallow result.
7. Evaluated synthetic noise, including out-of-sector losses. At n=6 the
   shorter imperfect circuit beat the longer exact circuit under this model.
8. Applied identical state-aware pruning to every method. This lowered both
   candidate and baseline counts; final comparisons use the improved counts.
9. Generated ten figures, result tables, JSON and hypotheses. Independently
   verified preparation, gradients, Hamiltonians, noise and topology. Existing
   relevant SCS/recursion tests passed as well.
10. Followed the evidence with a joint-sector star-leaf experiment. Zero-CP
    trial succeeded on weight 2 but failed the common-unitary test at weights
    1 and 3. One CP achieved all-sector fidelities within ~3e-12 of unity
    with a common phase and 26 compiled CZ gates. Same-star routed SCS unitary
    used 250 CZ gates. This is a reusable candidate leaf, not a complete
    recursive integration or scaling result.

All run IDs, source snapshots, parameters and compiled circuits are in
`artifacts/`; RESULTS.md links the exact evidence. No prior experiment source
was overwritten. No hardware jobs, external spending, or large sweeps ran.
