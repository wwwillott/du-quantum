import sys, time; sys.path.insert(0, 'tools')
import gpu_compress as g
import numpy as np
N, n, k, ops = g.load_ops('data/peep/10_5_dc2_cz79.json')
czs, i, p = g.ops_to_slots(k, ops)
rng = np.random.default_rng(0)
for j in [0, 40, 78]:
    t = time.time()
    c2, i2, p2, F = g.try_delete(N, n, k, czs, i, p, j, rng, restarts=1)
    print(j, 'ok' if c2 else 'fail', F, f'{time.time()-t:.1f}s', flush=True)
