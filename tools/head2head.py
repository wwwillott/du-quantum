import sys, os, time, importlib.util, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np, cirq, cirq_google
from sim import simulate, score_stats, check_exact, HW
from fake_calib import fake_calibration

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

ours = load(os.path.join(ROOT, 'submission.py'), 'ours')
theirs = load(os.path.join(ROOT, 'astra', 'submission.py'), 'theirs')
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
Nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 10
cz, sq, z0, z1 = fake_calibration(seed)
tot = {'ours': 0.0, 'theirs': 0.0}
inst = [tuple(map(int, x.split(':'))) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else     [(N, n) for N in range(2, Nmax + 1) for n in range(1, N // 2 + 1)]
for N, n in inst:
    if True:
        row = []
        for name, mod in (('ours', ours), ('theirs', theirs)):
            t0 = time.time()
            c, qs = mod.create_dicke_state(N, n, cz, sq, z0, z1)
            dt = time.time() - t0
            F = check_exact(c, qs, n)
            ncz = sum(1 for op in c.all_operations() if len(op.qubits) == 2)
            PZ, PX, _ = simulate(c, qs, cz, sq, z0, z1, hw=HW(seed=99), n_traj=int(os.environ.get("NTRAJ", "600")))
            st = score_stats(PZ, PX, N, n, reps=400, rng=np.random.default_rng(7))
            tot[name] += st['median']
            row.append(f'{name}: med={st["median"]:.3f} f={st["f"]:.3f} X={st["Xmed"]:.3f} cz={ncz} mom={len(c)} F={F:.8f} t={dt:.1f}s')
        print(f'D({N},{n})  ' + ' | '.join(row), flush=True)
print('TOTAL medians', {k: round(v, 3) for k, v in tot.items()})
