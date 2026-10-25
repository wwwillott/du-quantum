"""Export a submission's create_dicke_state circuit as a peep-format op list (start point for compression).

usage: python tools/export_ops.py MODULE_PATH N n [seed] [outdir]
Writes {outdir}/{N}_{n}_{tag}_cz{m}.json with ops on nodes 0..k-1 (state qubits first, sorted).
"""
import importlib.util, json, os, sys
import numpy as np
import cirq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_calib import fake_calibration


def circuit_to_ops(circuit, state_qubits):
    sq = sorted(state_qubits)
    anc = sorted(set(circuit.all_qubits()) - set(sq))
    order = sq + anc
    idx = {q: i for i, q in enumerate(order)}
    ops = []
    for op in circuit.all_operations():
        qs = [idx[q] for q in op.qubits]
        if len(qs) == 2:
            assert isinstance(op.gate, cirq.CZPowGate) and abs(op.gate.exponent - 1) < 1e-12, op
            ops.append(['cz', qs[0], qs[1]])
        else:
            U = cirq.unitary(op)
            ops.append(['u', qs[0], [[float(z.real), float(z.imag)] for z in U.ravel()]])
    return len(sq), len(order), ops


if __name__ == '__main__':
    path, N, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    outdir = sys.argv[5] if len(sys.argv) > 5 else 'data/peep_gpu/start'
    os.makedirs(outdir, exist_ok=True)
    spec = importlib.util.spec_from_file_location('sub', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cz, sq, z0, z1 = fake_calibration(seed)
    c, qs = mod.create_dicke_state(N, n, cz, sq, z0, z1)
    Nq, k, ops = circuit_to_ops(c, qs)
    assert Nq == N
    ncz = sum(1 for o in ops if o[0] == 'cz')
    tag = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(outdir, f'{N}_{n}_{tag}_cz{ncz}.json')
    json.dump(dict(N=N, n=n, k=k, ops=ops, tag=tag, ncz=ncz), open(out, 'w'))
    print(out, 'k', k, 'cz', ncz, 'moments', len(c))
