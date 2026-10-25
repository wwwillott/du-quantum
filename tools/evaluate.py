"""Evaluate templates / full create_dicke_state on fake calibrations with the noisy simulator."""
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cirq
import dicke_core as dc
from sim import simulate, score_stats, check_exact, HW
from fake_calib import fake_calibration


def load_template(path):
    d = json.load(open(path))
    N, edges, seq, P, mode = d['N'], d['edges'], d['seq'], np.array(d['params']), d['mode']
    ops = []

    def layer(Pl):
        for q in range(N):
            if mode == 'real':
                ops.append(('u', q, dc.ry(Pl[q][0])))
            else:
                a, t, b = Pl[q]
                ops.append(('u', q, dc.rz(b) @ dc.ry(t) @ dc.rz(a)))

    layer(P[0])
    for t, e in enumerate(seq):
        a, b = edges[e]
        ops.append(('cz', a, b))
        layer(P[t + 1])
    return dict(k=N, ops=ops, state_nodes=list(range(N)), N=N, n=d['n'])


def run_template(t, calib, hw=HW(), n_traj=2000, dd=True, reps=400):
    cz, sq, z0, z1 = calib
    N, n = t['N'], t['n']
    adj = dc.hw_graph(cz, sq)
    nw = dc.node_weights(t)
    node_cost = lambda node, q: dc.readout_cost(q, n / N, z0, z1) + nw[node] * sq[q][0]
    cost, mapping = dc.embed(t, adj, node_cost)
    c = dc.build_circuit(t, mapping, dd=dd)
    qs = {mapping[i] for i in t['state_nodes']}
    F = check_exact(c, qs, n)
    PZ, PX, info = simulate(c, qs, cz, sq, z0, z1, hw=hw, n_traj=n_traj)
    st = score_stats(PZ, PX, N, n, reps=reps)
    st.update(F=F, depth=len(c), ncz=sum(1 for op in c.all_operations() if len(op.qubits) == 2), cost=cost)
    return st, c


if __name__ == '__main__':
    calib = fake_calibration(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    t = load_template(sys.argv[1])
    for dd in (False, True):
        st, c = run_template(t, calib, dd=dd)
        print('dd' if dd else 'nodd', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()})
    print(c)
