"""Sanity checks for gpu_compress conversions and torch model (dqgpu env)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_compress as g
import torch

src = sys.argv[1] if len(sys.argv) > 1 else 'data/peep/6_3_dc2_cz22.json'
N, n, k, ops = g.load_ops(src)
czs, i, p = g.ops_to_slots(k, ops)
print(N, n, k, len(czs), sorted(set(tuple(sorted(c)) for c in czs)))
print('infid slots', g.infid_np(N, n, k, czs, i, p))
th = g.slots_to_params(k, czs, i, p)
i2, p2 = g.params_to_slots(k, czs, th)
print('infid roundtrip', g.infid_np(N, n, k, czs, i2, p2))
m = g.Model(N, n, k, czs, torch.complex128)
print('torch loss', m.loss(torch.tensor(th[None], device=g.dev)).item())
c3, i3, p3 = g.delete_cz(k, czs, i2, p2, 0)
m3 = g.Model(N, n, k, c3, torch.complex128)
th3 = g.slots_to_params(k, c3, i3, p3)
print('after delete 0: np', g.infid_np(N, n, k, c3, i3, p3), 'torch', m3.loss(torch.tensor(th3[None], device=g.dev)).item())
th4, L4 = g.lm_polish(m3, th3, iters=100, verbose=True, time_limit=60)
print('LM after deletion', L4)
