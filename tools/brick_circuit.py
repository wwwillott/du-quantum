"""Turn data/brick/*.json solutions into Willow circuits; verify exactness + device validity."""
import sys, os, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cirq, cirq_google
from sim import check_exact

DEV = cirq_google.Willow105
QS = set(DEV.metadata.qubit_set)
PAIRS = set(frozenset(p) for p in DEV.metadata.qubit_pairs)


def ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def place(cells, classes):
    for r0 in range(-2, 14):
        for c0 in range(-2, 14):
            q = [cirq.GridQubit(r0 + r, c0 + c) for r, c in cells]
            if not all(x in QS for x in q):
                continue
            if all(frozenset((q[a], q[b])) in PAIRS for cl in classes for a, b in cl):
                return q
    return None


def build(d):
    q = place(d['cells'], d['classes'])
    P = np.array(d['params'])
    moments = []

    def oneq(L):
        ops = []
        for i in range(d['N']):
            U = ry(P[L][i][0]) if d['mode'] == 'real' else None
            ops.append(cirq.PhasedXZGate.from_matrix(U).on(q[i]))
        return cirq.Moment(ops)

    moments.append(oneq(0))
    for t, c in enumerate(d['seq']):
        moments.append(cirq.Moment([cirq.CZ(q[a], q[b]) for a, b in d['classes'][c]]))
        moments.append(oneq(t + 1))
    return cirq.Circuit(moments), set(q)


if __name__ == '__main__':
    best = {}
    for f in glob.glob('data/brick/*.json'):
        d = json.load(open(f))
        key = (d['N'], d['n'])
        c, qs = build(d)
        DEV.validate_circuit(c)
        F = check_exact(c, qs, d['n'])
        ok = abs(F - 1) < 1e-9
        print(f'{os.path.basename(f)}: moments={len(c)} F={F:.14f} {"OK" if ok else "BAD"}')
        if ok and (key not in best or len(c) < best[key][0]):
            best[key] = (len(c), f)
    print({f'{k[0]},{k[1]}': v[0] for k, v in sorted(best.items())})
