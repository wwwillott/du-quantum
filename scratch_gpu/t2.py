import sys, time; sys.path.insert(0, 'tools')
import numpy as np, cpu_compress as cc, gpu_compress as g
N, n, k, ops = g.load_ops('data/peep_gpu/start/10_5_astra_cz73.json')
czs, i, p = g.ops_to_slots(k, ops)
rng = np.random.default_rng(0)
for j in [5, 30, 60]:
    t = time.time(); r = cc._worker((N, n, k, czs, i, p, j, 1, 1)); print(j, r[1] is not None, r[4], f'{time.time()-t:.1f}s', flush=True)
