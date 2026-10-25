import sys, time, os; sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'tools')
import numpy as np, cirq, cirq_google
import submission as S
from sim import check_exact
from fake_calib import fake_calibration
worst = (0, None); fails = []; ok = 0; maxmom = 0
for seed in (11, 23):
    cz, sq, z0, z1 = fake_calibration(seed, q_dropout=0.02, c_dropout=0.02)
    for N in range(2, 21):
        for n in range(1, N // 2 + 1):
            t0 = time.time()
            try:
                c, qs = S.create_dicke_state(N, n, cz, sq, z0, z1)
            except Exception as e:
                fails.append((seed, N, n, repr(e)[:80])); continue
            dt = time.time() - t0
            if dt > worst[0]: worst = (dt, (seed, N, n))
            try:
                S.check_allowed(c, cz, sq); cirq_google.Willow105.validate_circuit(c)
                assert len(c) <= 1750 and len(qs) == N
                assert not any(cirq.is_measurement(o) for o in c.all_operations())
                if len(c.all_qubits()) <= 16:
                    F = check_exact(c, qs, n); assert abs(F - 1) < 1e-8, F
                ok += 1; maxmom = max(maxmom, len(c))
            except Exception as e:
                fails.append((seed, N, n, repr(e)[:80]))
print(f'hardware entry: {ok} OK, failures={fails}, worst build={worst[0]:.1f}s at {worst[1]}, max moments={maxmom}', flush=True)
