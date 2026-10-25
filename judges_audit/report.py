"""Summarize fresh calibrated-QVM and original-notebook depth results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ast
from .compare import ROOT, OUT, ORIGINAL


def main():
    candidates = {}
    for name in ('team','will','kai_v2','scs','kai_large','will_large'):
        data = json.loads((OUT/(name+'.json')).read_text())
        assert data['complete'] and all(r['accepted'] for r in data['rows'])
        rows = data['rows']
        candidates[name] = dict(cases=len(rows),
            observed_N=sum(r['scores'][str(r['n'])] for r in rows),
            median_sum_N=sum(r['shot_statistics'][str(r['n'])]['median'] for r in rows),
            median_sum_4=sum(r['shot_statistics'][str(min(4,r['n']))]['median'] for r in rows),
            min_fidelity=min(r['verification']['fidelity'] for r in rows),
            max_construction_seconds=max(r['construction_seconds'] for r in rows),
            max_depth=max(r['moments'] for r in rows))
    depth = json.loads((OUT/'problem2_best.json').read_text())
    old = json.loads((ROOT/'interference_dicke/problem2/evaluation.json').read_text())
    compare = dict()
    for name in ('will-605b9a8','kai-v2-48115b9'):
        records = json.loads((ROOT/'adaptive_dicke_validation/artifacts'/name/'results.json').read_text())['records']
        compare[name] = dict(score_N20=sum(r['n']*r['k']/r['depth'] for r in records),
            score_N15=sum(r['n']*r['k']/r['depth'] for r in records if r['n']<=15))
    original = json.loads(ORIGINAL.read_text())
    repository = json.loads((ROOT/'DuQuantum_Google_Standard.ipynb').read_text())
    assert ast.dump(ast.parse(''.join(original['cells'][7]['source']))) == ast.dump(ast.parse(''.join(repository['cells'][7]['source'])))
    summary = dict(original_notebook_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
        p1=candidates, p2_best=dict(score_N20=depth['score_N20'],score_N15=depth['score_N15']),
        p2_previous=dict(score_N20=old['score'],score_N15=sum(r['score'] for r in old['rows'] if r['n']<=15)),
        p2_references=compare,
        qvm_calibration_utc=datetime.fromtimestamp(1723795877.630,timezone.utc).isoformat(),
        physical_hardware=False,score_ast_matches_repository=True,
        limitations=['P1 test cases and K are not specified by original notebook',
                     'P2 prose N<=15 disagrees with code N<=20',
                     'Noise model and calibration can change rankings; no universal optimum established'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
