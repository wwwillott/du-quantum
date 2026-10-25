"""Verify the merged submission's bonus circuits and compare per instance with the competitor's file.
usage: python tools/verify_bonus_merged.py"""
import importlib.util, os, sys, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np, cirq, cirq_google
from sim import check_exact


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ours = load(os.path.join(ROOT, 'submission.py'), 'ours_v')
theirs = load(os.path.join(ROOT, 'astra', 'submission.py'), 'theirs_v')
devq = cirq_google.Willow105.metadata.qubit_set
win = tie = loss = 0; tot = 0.0; rows = []; bad = []; adapt = []
for N in range(2, 16):
    for n in range(1, N // 2 + 1):
        c, qs = ours.create_dicke_state_adaptive(N, n)
        a = len(c); b = len(theirs.create_dicke_state_adaptive(N, n)[0]); tot += N * n / a
        meas = any(isinstance(o.gate, cirq.MeasurementGate) for o in c.all_operations())
        if meas:
            for m in c:
                mq = {q for o in m.operations if isinstance(o.gate, cirq.MeasurementGate) for q in o.qubits}
                if mq:
                    dg = {q.row + q.col for q in mq}
                    if not all({q for q in devq if q.row + q.col == x} <= mq for x in dg) or any(q.row + q.col in dg for q in qs):
                        bad.append((N, n, 'diagonal rule'))
            d = cirq.drop_terminal_measurements(cirq.defer_measurements(c))
            if len(d.all_qubits()) <= 24:
                out = sorted(qs); order = out + sorted(set(d.all_qubits()) - set(out))
                psi = cirq.final_state_vector(d, qubit_order=order, dtype=np.complex64).reshape(2 ** N, -1)
                tgt = np.zeros(2 ** N)
                for cc in itertools.combinations(range(N), n):
                    tgt[sum(1 << (N - 1 - i) for i in cc)] = 1
                F = float(np.sum(np.abs((tgt / np.linalg.norm(tgt)) @ psi) ** 2))
                adapt.append((N, n, round(F, 6)))
                if abs(F - 1) > 1e-5:
                    bad.append((N, n, F))
            else:
                adapt.append((N, n, 'too big for local check'))
        elif len(c.all_qubits()) <= 18:
            F = check_exact(c, qs, n)
            if abs(F - 1) > 1e-8:
                bad.append((N, n, F))
        rows.append((N, n, a, b)); win += a < b; tie += a == b; loss += a > b
print(f'MERGED bonus sum={tot:.3f} | vs competitor: wins {win}, ties {tie}, losses {loss}')
print('ties:', [(N, n, a) for N, n, a, b in rows if a == b])
print('losses:', [(N, n, a, b) for N, n, a, b in rows if a > b])
print('adaptive checks:', adapt)
print('failures:', bad)
