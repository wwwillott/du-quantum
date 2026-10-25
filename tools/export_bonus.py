"""Export current bonus circuits (unitary, no ancilla) to JSON for gpu_mprune.py; and rebuild/verify pruned ones.
usage: python tools/export_bonus.py export N n   |   python tools/export_bonus.py verify file.json"""
import sys, os, json, math, itertools, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cirq, cirq_google
from gpu_mprune import zyz, u3_mat
from sim import check_exact


def load_sub():
    s = importlib.util.spec_from_file_location('subx', 'submission.py'); S = importlib.util.module_from_spec(s); s.loader.exec_module(S); return S


def export(N, n):
    S = load_sub()
    c, qs = S.create_dicke_state_adaptive(N, n)
    outs = sorted(qs); anc = sorted(set(c.all_qubits()) - set(outs)); q = outs + anc
    idx = {x: i for i, x in enumerate(q)}
    moments = []
    for m in c:
        ops = []
        for op in m.operations:
            if len(op.qubits) == 2:
                ops.append(['cz', idx[op.qubits[0]], idx[op.qubits[1]]])
            else:
                ops.append(['u', idx[op.qubits[0]], [float(x) for x in zyz(cirq.unitary(op))]])
        moments.append(ops)
    d = dict(N=N, n=n, qubits=[[x.row, x.col] for x in q], outputs=list(range(N)), moments=moments)
    fn = f'logs/bonus_export_{N}_{n}.json'
    json.dump(d, open(fn, 'w'))
    return fn


def rebuild(d):
    q = [cirq.GridQubit(r, c) for r, c in d['qubits']]
    moments = []
    for mo in d['moments']:
        ops = []
        for op in mo:
            if op[0] == 'cz':
                ops.append(cirq.CZ(q[op[1]], q[op[2]]))
            else:
                ops.append(cirq.PhasedXZGate.from_matrix(u3_mat(op[2])).on(q[op[1]]))
        moments.append(cirq.Moment(ops))
    return cirq.Circuit(moments), set(q[:d['N']])


if __name__ == '__main__':
    if sys.argv[1] == 'export':
        print(export(int(sys.argv[2]), int(sys.argv[3])))
    else:
        d = json.load(open(sys.argv[2]))
        c, qs = rebuild(d)
        cirq_google.Willow105.validate_circuit(c)
        print(sys.argv[2], 'moments', len(c), 'F=%.14f' % check_exact(c, qs, d['n']))
