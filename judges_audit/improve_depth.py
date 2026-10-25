"""Promote only independently verified, strictly shallower saved circuits."""
import json
from pathlib import Path
import cirq
from .compare import load, ROOT, SOURCES
from adaptive_dicke_validation.audit import structure, verify_all_branches


def main():
    module = load(SOURCES['will'], 'depth_reference_will')
    additions, checks = {}, []
    for n, k in [(7, 3), (8, 4)]:
        circuit, outputs = module.create_dicke_state_adaptive(n, k)
        assert not structure(circuit, outputs, n)
        result = verify_all_branches(circuit, outputs, k, max_qubits=20)
        assert result['status'] == 'verified'
        additions[n,k] = dict(circuit_json=cirq.to_json(circuit, indent=None),
                              outputs=[(q.row,q.col) for q in sorted(outputs)])
        checks.append(dict(n=n,k=k,moments=len(circuit),verification=result,
                           source_sha256=__import__('hashlib').sha256(SOURCES['will'].read_bytes()).hexdigest()))
    destination = ROOT/'problem2_submission.py'
    source = destination.read_text(encoding='utf8')
    marker = '# JUDGES ORIGINAL NOTEBOOK DEPTH IMPROVEMENTS'
    if marker in source:
        source = source[:source.index(marker)].rstrip()+'\n'
    suffix = '\n\n'+marker+'\n'
    suffix += '# Verified all-branch Will 605b9a8 circuits; all moments counted.\n'
    suffix += '_JUDGE_DEPTH_DATA = '+repr(additions)+'\n\n'
    suffix += '''@functools.lru_cache(maxsize=2)
def _judge_depth_improvement(n, k):
    data = _JUDGE_DEPTH_DATA.get((n,k))
    if data is None:
        return None
    circuit = cirq.read_json(json_text=data['circuit_json'])
    return cirq.FrozenCircuit(circuit), frozenset(cirq.GridQubit(*q) for q in data['outputs'])


_JUDGE_PREVIOUS_ADAPTIVE = create_dicke_state_adaptive


def create_dicke_state_adaptive(num_qubits, num_excitations):
    candidate = _judge_depth_improvement(num_qubits, num_excitations)
    if candidate is not None:
        c,q = candidate
        return cirq.Circuit(c),set(q)
    return _JUDGE_PREVIOUS_ADAPTIVE(num_qubits,num_excitations)
'''
    destination.write_text(source+suffix,encoding='utf8')
    old = json.loads((ROOT/'interference_dicke/problem2/evaluation.json').read_text())
    rows = []
    for row in old['rows']:
        row = dict(row)
        found = next((r for r in checks if (r['n'],r['k']) == (row['n'],row['k'])), None)
        if found:
            row.update(moments=found['moments'],score=row['n']*row['k']/found['moments'],
                       verification=found['verification'],method='Will 605b9a8 independently verified improvement')
        rows.append(row)
    result = dict(score_N20=sum(r['score'] for r in rows),score_N15=sum(r['score'] for r in rows if r['n']<=15),
                  rows=rows,new_checks=checks,source='problem2_submission.py',
                  score_definition='Exact original notebook cell 16: sum N*k/len(circuit)',
                  complete=True)
    (ROOT/'judges_audit/results').mkdir(exist_ok=True)
    (ROOT/'judges_audit/results/problem2_best.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({key:result[key] for key in ('score_N20','score_N15','new_checks')},indent=2),flush=True)


if __name__ == '__main__':
    main()
