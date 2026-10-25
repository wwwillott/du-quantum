"""Package the calibrated-QVM leader and verified depth leader in one file."""
import ast
import hashlib
import json
import time
from .compare import ROOT, OUT, ORIGINAL, load


def main():
    main_source = (ROOT/'judges_audit/sources/will_8af9eb4.py').read_text(encoding='utf8')
    bonus_source = (ROOT/'problem2_submission.py').read_text(encoding='utf8')
    content = '# Best tested calibrated-Willow-QVM P1 + independently verified P2.\n'
    content += '# Simulation results and scope: judges_audit/README.md.\n'+main_source
    content += '\n\n# Isolated depth module prevents compiler/global-name collisions.\n'
    content += 'import types as _judged_types\n'
    content += '_judged_bonus = _judged_types.ModuleType("judged_depth_portfolio")\n'
    content += 'exec(compile('+repr(bonus_source)+', "judged_depth_portfolio", "exec"), _judged_bonus.__dict__)\n'
    content += 'create_dicke_state_adaptive = _judged_bonus.create_dicke_state_adaptive\n'
    path = ROOT/'recommended_submission.py'
    path.write_text(content,encoding='utf8')
    module = load(path,'recommended_checked')
    # Run the literal original score_solution with an explicit validator: the
    # original notebook refers to validate_circuit but never defines it.
    reference = json.loads((OUT/'problem2_best.json').read_text())
    expected = {(r['n'],r['k']):r for r in reference['rows']}
    from interference_dicke.problem2.verify import validate_structure
    checked = []
    def validate(circuit,k,qubits):
        n = len(qubits)
        validate_structure(circuit,qubits,n)
        assert len(circuit) == expected[n,k]['moments']
        checked.append(dict(n=n,k=k,moments=len(circuit)))
    def create(n,k):
        start = time.perf_counter()
        result = module.create_dicke_state_adaptive(n,k)
        assert time.perf_counter()-start < 60
        return result
    scope = dict(create_dicke_state_adaptive=create,validate_circuit=validate)
    notebook = json.loads(ORIGINAL.read_text(encoding='utf8'))
    node = ast.parse(''.join(notebook['cells'][16]['source']))
    exec(compile(node,'original_judges_cell_16','exec'),scope)
    score = scope['score_solution']()
    assert len(checked) == 100 and abs(score-reference['score_N20']) < 1e-10
    (OUT/'packaged_score.json').write_text(json.dumps(dict(score=score,rows=checked,
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        state_evidence='Unchanged prior verified construction; two replacements independently checked on every branch',
        original_score_solution_executed=True),indent=2),encoding='utf8')
    print('Packaged original-notebook score:',score,'all',len(checked),'cases',flush=True)


if __name__ == '__main__':
    main()
