# Validate the pushed submissions and experiment with challenge 2

Read [VALIDATION.md](VALIDATION.md) for the score comparison, review findings,
paper prototype, scope of validation and exact reproduction commands.

Pinned commits:

- Will `605b9a86c931e66710ed31c4e1e5aa9ab08e78d7`, branch `will`.
- Kai `48115b9fbd9dd3d587fd8cfa01f0e58e00617638`, branch `kai/root-pipeline-v2`.
- Kai portfolio from `913fcdfdacf3165f9f5f7a5676277c09e44d3fa0`, main.

The snapshots are ignored by Git. Recreate them locally from these commits:

```powershell
New-Item -ItemType Directory -Force adaptive_dicke_validation/snapshots
git archive --format=zip --output=adaptive_dicke_validation/snapshots/will.zip 605b9a86c931e66710ed31c4e1e5aa9ab08e78d7
Expand-Archive -LiteralPath adaptive_dicke_validation/snapshots/will.zip -DestinationPath adaptive_dicke_validation/snapshots/will-605b9a8
git archive --format=zip --output=adaptive_dicke_validation/snapshots/kai.zip 48115b9fbd9dd3d587fd8cfa01f0e58e00617638 dicke_submission.py final_v2 submission.py dicke_core.py dicke_main.py build_submission.py SUBMISSION_STATUS.md CLAUDE.md team_eval tools data test_dicke.py dicke.py analysis
Expand-Archive -LiteralPath adaptive_dicke_validation/snapshots/kai.zip -DestinationPath adaptive_dicke_validation/snapshots/kai-48115b9
```

Use `git fetch origin` first if the commits are not available locally. Run
from the repository root. The existing Python environment already contains
all dependencies; no new installations were required.

Minimal entry points:

```powershell
python -m pytest adaptive_dicke_validation/tests -q
python -m adaptive_dicke_validation.paper_prototype
python -m adaptive_dicke_validation.summarize
```

For selectable experiments:

```python
from adaptive_dicke_validation.portfolio import create_dicke_state_adaptive

circuit, outputs = create_dicke_state_adaptive(4, 2)  # keeps shortest verified
circuit, outputs = create_dicke_state_adaptive(4, 2, variant="yu_one_round")
```

This API depends on the pinned local snapshot and is not a standalone
submission. Every forced paper prototype includes a finite exact fallback;
there is no postselection. To evaluate a new candidate, run `structure` and
`verify_all_branches` from `audit.py`, then compare `N*k/len(circuit)` against
the incumbent. Count the entire static circuit, not average repeat-until-success
rounds. Full readout diagonals must be checked separately from Cirq's device.

Artifacts include all 56 primary-domain circuits per submission, full branch
validation, scores, source hashes, bounded hardware smoke checks, scorer parity,
counterexamples to the surrogate, and the six failed-to-improve paper circuits.
These are local simulations; no hardware execution or hardware scores are claimed.
