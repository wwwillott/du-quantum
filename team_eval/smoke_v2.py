"""Smoke test for final_v2/submission_v2.py: validity, exactness, timing, CZ count vs the v1 portfolio."""
import sys, time, importlib.util
sys.path.insert(0, 'tools')
import numpy as np, cirq, cirq_google
from fake_calib import fake_calibration
from sim import check_exact
t = time.time()
spec = importlib.util.spec_from_file_location('v2', 'final_v2/submission_v2.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
print('import', round(time.time() - t, 1), 's', flush=True)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 101
inst = [tuple(map(int, s.split(':'))) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 else [(10, 3), (10, 5), (12, 6), (9, 4), (11, 8)]
cz, sq, z0, z1 = fake_calibration(seed)
for N, n in inst:
    t = time.time(); c1, q1 = S._team_v1_create_dicke_state(N, n, cz, sq, z0, z1); t1 = time.time() - t
    t = time.time(); c, qs = S.create_dicke_state(N, n, cz, sq, z0, z1); t2 = time.time() - t
    cirq_google.Willow105.validate_circuit(c)
    ok = S._v2_allowed(c, cz, sq)
    F = check_exact(c, qs, n)
    ncz = lambda c: sum(1 for op in c.all_operations() if len(op.qubits) == 2)
    print(f'D({N},{n}) v1: {ncz(c1)} CZ {len(c1)} mom {t1:.1f}s | v2: {ncz(c)} CZ {len(c)} mom {t2:.1f}s allowed={ok} F={F:.12f}', flush=True)
