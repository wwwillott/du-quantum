"""Build final_v2/submission_v2.py = team portfolio (team_eval/will_dicke_submission.py, unchanged)
+ a library of exact compressed op-list circuits for N > 8 (data/peep_gpu from tools/cpu_compress.py and
tools/gpu_compress.py, plus the root pipeline's data/peep).

For N > 8 the new create_dicke_state embeds every library circuit on the calibrated chip, ranks them together
with the portfolio's own circuit by one error estimate (the root pipeline's estimate_error), verifies the
winner noiselessly, and falls back to the portfolio circuit on any failure or time pressure.
usage: python tools/build_v2.py
"""
import glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_compress import load_ops, ops_to_slots, infid_np

BASE = 'team_eval/will_dicke_submission.py'
OUT = 'final_v2/submission_v2.py'
KEEP = 3          # lowest-CZ circuits kept per (N, n)
NMIN = 9


def library():
    files = [f for f in glob.glob('data/peep_gpu/*.json')] + glob.glob('data/peep/*.json') + \
        glob.glob('data/peep_gpu/start/*.json')
    best = {}
    for f in files:
        N, n, k, ops = load_ops(f)
        if N < NMIN:
            continue
        ncz = sum(1 for o in ops if o[0] == 'cz')
        best.setdefault((N, n), []).append((ncz, f, k, ops))
    lib = {}
    for key, lst in sorted(best.items()):
        lst.sort(key=lambda x: x[0])
        out, seen = [], set()
        for ncz, f, k, ops in lst:
            if ncz in seen:
                continue
            czs, init, post = ops_to_slots(k, ops)
            F = infid_np(key[0], key[1], k, czs, init, post)
            if abs(F) > 1e-11:
                print('  reject (not exact)', f, F)
                continue
            seen.add(ncz)
            enc = []
            for o in ops:
                if o[0] == 'cz':
                    enc.append(('cz', o[1], o[2]))
                else:
                    enc.append(('u', o[1], tuple(complex(round(z.real, 17), round(z.imag, 17)) for z in o[2].ravel())))
            out.append(dict(k=k, ncz=ncz, src=os.path.basename(f), ops=enc))
            if len(out) >= KEEP:
                break
        lib[key] = out
        print(key, [(e['ncz'], e['src']) for e in out])
    return lib


DISPATCH = r'''

# ==================================================================== v2: compressed-circuit library for N > 8
# Exact op-list circuits on abstract nodes (state nodes 0..N-1), found offline by global CZ deletion + refit
# (tools/cpu_compress.py, tools/gpu_compress.py) or by the root pipeline's peephole search. Each entry was
# checked noiselessly (1-F < 1e-11) at build time; the chosen embedding is re-verified at run time.
_V2_LIB = __V2_LIB__


def _v2_template(entry, N, n):
    ops = []
    for o in entry['ops']:
        if o[0] == 'cz':
            ops.append(o)
        else:
            ops.append(('u', o[1], np.array(o[2], dtype=complex).reshape(2, 2)))
    return dict(k=entry['k'], ops=ops, state_nodes=list(range(N)), N=N, n=n)


def _v2_verify(circuit, outputs, k):
    outs = sorted(outputs)
    anc = sorted(set(circuit.all_qubits()) - set(outs))
    n = len(outs)
    if len(outs) + len(anc) > 20:
        return False
    state = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=outs + anc).final_state_vector
    state = state.reshape(2 ** n, -1)
    target = np.array([int(i).bit_count() == k for i in range(1 << n)], dtype=np.complex128)
    target /= np.linalg.norm(target)
    amp = target.conj() @ state
    return abs(float(np.sum(np.abs(amp) ** 2)) - 1) < 1e-9


def _v2_allowed(circuit, cz, sq):
    pairs = {frozenset(p) for p in cz}
    for op in circuit.all_operations():
        if len(op.qubits) == 2:
            if not isinstance(op.gate, cirq.CZPowGate) or frozenset(op.qubits) not in pairs:
                return False
        elif not isinstance(op.gate, cirq.PhasedXZGate) or op.qubits[0] not in sq:
            return False
    return len(circuit) <= 1750


def create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit,cirq.GridQubit],tuple[float,float]],
    sq_pauli_error_rate: dict[cirq.GridQubit,tuple[float,float]],
    zero_errors: dict[cirq.GridQubit,tuple[float,float]],
    one_errors: dict[cirq.GridQubit,tuple[float,float]],
) -> tuple[cirq.Circuit,set[cirq.GridQubit]]:
    """Team portfolio, plus (for N > 8) the compressed exact circuit library ranked by a calibrated
    error estimate. Only CZ and PhasedXZ gates; no measurements, external files or runtime fitting."""
    started = time.monotonic()
    N, k = num_qubits, num_excitations
    args = cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors
    base = _team_v1_create_dicke_state(N, k, *args)
    kk = min(k, N - k) if isinstance(N, int) and isinstance(k, int) else k
    if not isinstance(N, int) or N <= 8 or 2 * kk == N:
        # Half filling: the portfolio circuit beat every library pick in the noisy sim (team_eval/v2).
        return base
    if kk != 1 and (N, kk) not in _V2_LIB:
        return base
    flip = kk != k
    try:
        adj = _peer_hw_graph(cz_pauli_error_rate, sq_pauli_error_rate)
        qcost = lambda q: _peer_readout_cost(q, k / N, zero_errors, one_errors) + 2 * float(sq_pauli_error_rate[q][0])
        scored = [(_peer_estimate_error(base[0], base[1], k, *args, dd=True), 'base', base[0], set(base[1]))]
        if kk == 1:
            # W states: also rank the root pipeline's broadcast-tree circuit (2N-3 CZ, shallow).
            try:
                pc, pq = _peer_create_dicke_state(N, k, *args)
                scored.append((_peer_estimate_error(pc, set(pq), k, *args, dd=True), 'peer_w', pc, set(pq)))
            except Exception:
                pass
        for entry in _V2_LIB.get((N, kk), []):
            if time.monotonic() - started > 25:
                break
            t = _v2_template(entry, N, kk)
            res = _peer_embed(t, adj, lambda node, q: qcost(q), beam=800 if N > 11 else 1500, return_all=True)
            if not res:
                continue
            for r in _peer__diverse(res, 2):
                mapping = r[1]
                for dd in (True, False):
                    c = _peer_build_circuit(t, mapping, dd=dd)
                    qs = [mapping[i] for i in t['state_nodes']]
                    if flip:
                        c = _peer__append_flip(c, qs)
                    e = _peer_estimate_error(c, set(qs), k, *args, dd=dd)
                    scored.append((e, entry['src'], c, set(qs)))
        scored.sort(key=lambda x: x[0])
        for e, name, c, qs in scored:
            if name == 'base':
                return base
            if time.monotonic() - started > 45:
                return base
            if _v2_allowed(c, cz_pauli_error_rate, sq_pauli_error_rate) and _v2_verify(c, qs, k):
                return c, qs
    except Exception:
        return base
    return base
'''


def main():
    src = open(BASE, encoding='utf8').read()
    idx = [m.start() for m in re.finditer(r'^def create_dicke_state\(\n', src, flags=re.M)]
    assert idx, 'final create_dicke_state not found'
    last = idx[-1]
    src = src[:last] + 'def _team_v1_create_dicke_state(\n' + src[last + len('def create_dicke_state(\n'):]
    lib = library()
    out = src + DISPATCH.replace('__V2_LIB__', repr(lib))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w', encoding='utf8').write(out)
    print('wrote', OUT, len(out), 'bytes;', sum(len(v) for v in lib.values()), 'library circuits for', len(lib), 'instances')


if __name__ == '__main__':
    main()
