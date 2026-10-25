# Interference-based Dicke preparation

Small, reproducible experiments alongside the existing project. Start with
[RESULTS.md](RESULTS.md) for measurements, [theory.md](theory.md) for proofs,
and [hypothesis_tree.json](hypothesis_tree.json) for open questions.

From the `du-quantum` root:

```powershell
python -m pytest interference_dicke/tests -q
python -m interference_dicke.experiment --noise
python -m interference_dicke.experiment --extended
python -m interference_dicke.geometry
python -m interference_dicke.noise_followup
python -m interference_dicke.prune
python -m interference_dicke.multisector
python -m interference_dicke.report
```

The first command validates mathematics, derivatives and simulator agreement.
The first experiment is the compact n=4 milestone; only then run the n=6
extension. Kernel timing alone reaches n=12. Full-state/density computations
are restricted to small independent checks and noise that leaves fixed weight.
No hardware submission, network service, or paid compute is used.

Code uses the repository's installed NumPy, SciPy, Cirq, cirq-google,
Matplotlib and NetworkX; [requirements-reproduced.txt](requirements-reproduced.txt)
records measured versions of core packages. The original requirements.txt is
untouched (its NumPy pin differs from the actually installed environment).
Each run records the Python version and OS; Python 3.14.4 was used here.

Files are deliberately compact:

- `core.py`: bitmask sectors, sparse Hamiltonians, affected-pair gates,
  analytic adjoint gradients and bounded optimization.
- `hardware.py`: reuse of Willow topology, repo CZ compiler, device checks,
  and synthetic density-matrix noise.
- `experiment.py`: kernel measurements, free/interacting path circuits,
  verified SCS and routed recursive baselines, incremental JSON and circuits.
- `geometry.py`: physical square/star experiments.
- `prune.py`: equal input-specific pruning for all state-preparation methods.
- `multisector.py`: common-phase unitary leaf test on all five weight sectors.
- `report.py`: ten figures, tables, comparison JSON and hypothesis updates.

Artifacts are append-only unique run directories. Reports select the earliest
small/extended milestone and latest geometry/pruned/multisector results so
the original evidence is retained. Use the JSON paths in RESULTS.md to select
an exact run. Parameters, circuits, source snapshots and optimizer budgets
are preserved, including negative results. Do not confuse state-specific
pruning with a unitary that works coherently inside recursion.
