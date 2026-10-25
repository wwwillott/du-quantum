"""End-to-end check of submission.create_dicke_state: rules, exactness, runtime, simulated score.
usage: python tools/run_submission.py [seed] [Nmax] [ntraj] [instances like 4:2,6:3]
"""
import sys, os, time, importlib, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cirq, cirq_google
import submission as S
from sim import simulate, score_stats, check_exact, HW
from fake_calib import fake_calibration

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
Nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ntraj = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
if len(sys.argv) > 4:
    inst = [tuple(map(int, s.split(':'))) for s in sys.argv[4].split(',')]
else:
    inst = [(N, n) for N in range(2, Nmax + 1) for n in range(1, N // 2 + 1)]
calib = fake_calibration(seed)
cz, sq, z0, z1 = calib
tot = 0
rows = []
for N, n in inst:
    t0 = time.time()
    c, qs = S.create_dicke_state(N, n, cz, sq, z0, z1)
    dt = time.time() - t0
    S.check_allowed(c, cz, sq)
    cirq_google.Willow105.validate_circuit(c)
    nq = len(c.all_qubits())
    F = check_exact(c, qs, n) if nq <= 20 else float('nan')
    ncz = sum(1 for op in c.all_operations() if len(op.qubits) == 2)
    if nq <= 14:
        PZ, PX, _ = simulate(c, qs, cz, sq, z0, z1, hw=HW(), n_traj=ntraj)
        st = score_stats(PZ, PX, N, n, reps=200)
    else:
        st = dict(median=float('nan'), mean=float('nan'), f=float('nan'), Xmed=float('nan'))
    tot += st['median'] if not math.isnan(st['median']) else 0
    print(f'N={N:2d} n={n:2d} F={F:.10f} cz={ncz:3d} moments={len(c):3d} t={dt:5.1f}s  '
          f'score med={st["median"]:.3f} mean={st["mean"]:.3f} f={st["f"]:.3f} X={st["Xmed"]:.4f}', flush=True)
print('sum of medians', round(tot, 3))
