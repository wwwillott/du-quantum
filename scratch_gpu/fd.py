import sys, time; sys.path.insert(0, 'tools')
import numpy as np
import cpu_compress as cc, gpu_compress as g
N, n, k, ops = g.load_ops('data/peep/10_5_dc2_cz79.json')
czs, i, p = g.ops_to_slots(k, ops)
c = cc.Circ(N, n, k, czs)
th = g.slots_to_params(k, czs, i, p) + np.random.default_rng(1).normal(0, 0.1, 3 * (k + 2 * len(czs)))
L, gr = c.loss_grad(th)
t = time.time(); [c.loss_grad(th) for _ in range(10)]; print('loss_grad', (time.time() - t) / 10 * 1e3, 'ms')
for idx in [0, 5, 3 * k + 1, 3 * k + 4, len(th) - 1, len(th) - 5]:
    e = np.zeros_like(th); e[idx] = 1e-6
    fd = (c.loss_grad(th + e)[0] - c.loss_grad(th - e)[0]) / 2e-6
    print(idx, gr[idx], fd)


