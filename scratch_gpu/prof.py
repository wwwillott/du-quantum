import sys, time; sys.path.insert(0, 'tools')
import gpu_compress as g
import torch, numpy as np
for src in ['data/peep/6_3_dc2_cz22.json', 'data/peep/10_5_dc2_cz79.json', 'data/peep/12_6_dc2_cz129.json']:
    N, n, k, ops = g.load_ops(src)
    czs, i, p = g.ops_to_slots(k, ops)
    th = g.slots_to_params(k, czs, i, p)
    for dt in (torch.complex64,):
        m = g.Model(N, n, k, czs, dt)
        for B in (8, 64):
            T = torch.tensor(np.tile(th, (B, 1)), dtype=m.rdtype, device=g.dev, requires_grad=True)
            mk = torch.ones((B, len(czs)), dtype=m.rdtype, device=g.dev)
            for rep in range(3):
                torch.cuda.synchronize(); t0 = time.time()
                L = m.loss(T, mk); L.sum().backward()
                torch.cuda.synchronize()
            print(src, k, len(czs), dt, B, f'{(time.time()-t0)*1e3:.1f} ms/step', flush=True)
    m2 = g.Model(N, n, k, czs, torch.complex128)
    t0 = time.time(); g.lm_polish(m2, th, iters=3); print('  LM 3 iters', time.time() - t0, flush=True)
