"""Assemble submission.py = dicke_core.py + template library + dicke_main.py (single file, cirq/numpy/stdlib only)."""
import glob, json, re

core = open('dicke_core.py', encoding='utf8').read()
main = open('dicke_main.py', encoding='utf8').read()
lib = {}
for f in sorted(glob.glob('data/templates/*.json')):
    d = json.load(open(f))
    if d['infidelity'] > 1e-11:
        continue
    key = (d['N'], d['n'])
    entry = dict(edges=d['edges'], seq=d['seq'], params=d['params'], mode=d['mode'], shape=d['shape'], L=d['L'])
    lib.setdefault(key, []).append(entry)
# keep, per (N,n,shape), only the minimal-L entries (ties kept, max 3)
for key in lib:
    best = {}
    for e in lib[key]:
        best.setdefault(e['shape'], []).append(e)
    keep = []
    for shape, es in best.items():
        Lmin = min(e['L'] for e in es)
        keep += [e for e in es if e['L'] == Lmin][:3]
    lib[key] = keep
libsrc = 'LIBRARY = ' + repr({k: v for k, v in sorted(lib.items())}) + '\n'

# ---- peephole-optimised op lists (data/peep): keep the 2 lowest-CZ per (N, n), re-verified exactly
import sys
import numpy as np
sys.path.insert(0, 'tools')
import dicke_core as dc
from prune_be import zyz
from synth import dicke_vec
peep = {}
for f in glob.glob('data/peep/*.json'):
    d = json.load(open(f))
    peep.setdefault((d['N'], d['n']), []).append(d)
PEEP = {}
for key, ds in sorted(peep.items()):
    ds.sort(key=lambda d: d['ncz'])
    out_ = []
    for d in ds[:2]:
        ops = []
        for op in d['ops']:
            if op[0] == 'cz':
                ops.append(('cz', op[1], op[2]))
            else:
                U = np.array([complex(r, i) for r, i in op[2]]).reshape(2, 2)
                a_, t_, b_ = zyz(U)
                ops.append(('u', op[1], (float(a_), float(t_), float(b_))))
        tops = [('u', o[1], dc.rz(o[2][2]) @ dc.ry(o[2][1]) @ dc.rz(o[2][0])) if o[0] == 'u' else o for o in ops]
        F = abs(np.vdot(dicke_vec(key[0], key[1]), dc.template_state(dict(k=d['k'], ops=tops)))) ** 2
        if abs(F - 1) > 1e-10:
            print('PEEP reconstruction failed', key, F)
            continue
        out_.append((d['k'], ops))
    if out_:
        PEEP[key] = out_
libsrc += 'PEEP = ' + repr(PEEP) + '\n'
print('peep CZ counts:', {k: [sum(1 for o in ops if o[0] == 'cz') for _, ops in v] for k, v in PEEP.items()})
target = "LIBRARY = {}   # (N, n) -> list of template dicts {edges, seq, params, mode}; filled by build_submission.py\nPEEP = {}\n"
assert target in main
main = main.replace(target, libsrc)
header = '''"""DuQuantum 2026 - Google Quantum AI Dicke-state challenge submission.
Team: Kai Sandberg, Will, Ben (+ Claude / GPT-Astra).
Exact Dicke-state circuits (CZ + PhasedXZ only) with calibration-aware placement, ALAP scheduling and
dynamical decoupling. Entry points: create_dicke_state(...), create_dicke_state_adaptive(...).
"""
'''
out = header + core + '\n\n' + main
open('submission.py', 'w', encoding='utf8').write(out)
print('wrote submission.py', len(out), 'bytes; library keys:', sorted(lib))

# ---------------------------------------------------------------- bonus (adaptive) entry point
bonus_src = open('bonus/bonus.py', encoding='utf8').read()
brick = {}
for f in glob.glob('data/brick/*.json'):
    d = json.load(open(f))
    if d.get('loss', 1) > 1e-12:
        continue
    key = (d['N'], d['n'])
    if key not in brick or d['d'] < brick[key]['d']:
        brick[key] = dict(cells=d['cells'], classes=d['classes'], seq=d['seq'], params=d['params'], mode=d['mode'],
                          d=d['d'], N=d['N'], n=d['n'], anc=d.get('anc', False))
bonus_part = '''

# ==================================================================== BONUS: adaptive circuits
import types as _types
_BONUS_SRC = ''' + repr(bonus_src) + '''
_bonus_mod = _types.ModuleType('dicke_bonus')
exec(compile(_BONUS_SRC, 'dicke_bonus', 'exec'), _bonus_mod.__dict__)
BRICK = ''' + repr(brick) + '''


def _brick_circuit(d):
    import cirq_google
    dev = cirq_google.Willow105
    qset = set(dev.metadata.qubit_set)
    pairs = set(frozenset(p) for p in dev.metadata.qubit_pairs)
    for r0 in range(-2, 14):
        for c0 in range(-2, 14):
            q = [cirq.GridQubit(r0 + r, c0 + c) for r, c in d['cells']]
            if all(x in qset for x in q) and all(frozenset((q[a], q[b])) in pairs for cl in d['classes'] for a, b in cl):
                break
        else:
            continue
        break
    P = d['params']
    def layer(L):
        ops = []
        for i in range(len(q)):
            if d['mode'] == 'real':
                U = ry(P[L][i][0])
            else:
                a_, t_, b_ = P[L][i]
                U = rz(b_) @ ry(t_) @ rz(a_)
            ops.append(cirq.PhasedXZGate.from_matrix(U).on(q[i]))
        return cirq.Moment(ops)
    moments = [layer(0)]
    for t, c in enumerate(d['seq']):
        moments.append(cirq.Moment([cirq.CZ(q[a], q[b]) for a, b in d['classes'][c]]))
        moments.append(layer(t + 1))
    return cirq.Circuit(moments), set(q[:d['N']])


def create_dicke_state_adaptive(num_qubits, num_excitations):
    """Shallowest verified circuit per instance: adaptive/unitary constructions (bonus module) or a
    depth-minimal numerically synthesised brickwork circuit."""
    N, n = int(num_qubits), int(num_excitations)
    cands = []
    try:
        cands.append(_bonus_mod.create_dicke_state_adaptive(N, n))
    except NotImplementedError:
        pass
    if (N, n) in BRICK:
        cands.append(_brick_circuit(BRICK[(N, n)]))
    if not cands:
        raise NotImplementedError
    return min(cands, key=lambda cq: len(cq[0]))
'''
out = out + bonus_part
open('submission.py', 'w', encoding='utf8').write(out)
print('appended bonus; brick instances:', sorted(brick))
