import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cirq, math
import submission as S
from sim import simulate, score_stats, HW, score_from_counts
from fake_calib import fake_calibration
cz, sq, z0, z1 = fake_calibration(0)
N, n = int(sys.argv[1]), int(sys.argv[2])
c, qs = S.create_dicke_state(N, n, cz, sq, z0, z1)
# rebuild without DD: find template by re-running selection is complex; strip X-pairs not possible -> build variants via flag
variants = {
 'base': HW(),
 'noT2*': HW(T2star=0),
 'noTphi': HW(Tphi_white=0),
 'noDeph': HW(T2star=0, Tphi_white=0),
 'noT1': HW(T1=1e9),
 'noGate': HW(gate_noise_scale=0),
 'noRO': HW(readout_scale=0),
}
for name, hw in variants.items():
    PZ, PX, _ = simulate(c, qs, cz, sq, z0, z1, hw=hw, n_traj=600)
    st = score_stats(PZ, PX, N, n, reps=100)
    print(f'{name:8s} med={st["median"]:.3f} f={st["f"]:.3f} Z={st["Zmed"]:.3f} X={st["Xmed"]:.4f}', flush=True)
# X by k for base
PZ, PX, _ = simulate(c, qs, cz, sq, z0, z1, hw=HW(), n_traj=600)
cx = np.random.default_rng(1).multinomial(10000, PX / PX.sum())
for K in [2, 4, 6, N]:
    print('K', K, score_from_counts(np.random.default_rng(2).multinomial(10000, PZ / PZ.sum()), cx, N, n, K))
