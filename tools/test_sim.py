import sys, time
sys.path.insert(0, 'tools')
import numpy as np, cirq
from sim import *
from fake_calib import fake_calibration
cz, sq, z0, z1 = fake_calibration(0)
pairs = sorted(cz, key=lambda k: cz[k][0])
a, b = pairs[0]
c = cirq.Circuit([cirq.ry(np.pi/2)(a), cirq.X(b), cirq.CNOT(a, b)])
c = cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset())
c = cirq.merge_single_qubit_gates_to_phxz(c)
c = cirq.drop_empty_moments(c)
print(c)
print('F', check_exact(c, {a, b}, 1))
for hw in [HW(), HW(T2star=0, Tphi_white=0, T1=1e9)]:
    t = time.time()
    PZ, PX, info = simulate(c, {a, b}, cz, sq, z0, z1, hw=hw, n_traj=4000)
    print(time.time() - t, PZ, PX)
    print(score_stats(PZ, PX, 2, 1, reps=2000))
