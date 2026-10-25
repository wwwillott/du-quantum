import json
from pathlib import Path
root = Path(__file__).resolve().parent
r = json.loads((root / 'results.json').read_text())
print('WORKERS', r['metadata']['active_workers'])
print('PROGRESS', len(r['records']), 'scored', sum(bool(x.get('models')) for x in r['records']),
      'complete', sum(x.get('status') == 'complete' for x in r['records']))
for x in r['records']:
    if x.get('failures'):
        print('FAILURE', {k: x.get(k) for k in ('candidate', 'seed', 'N', 'n', 'status', 'failures', 'construction_seconds', 'exception')})
for c in 'ABCD':
    rows = [x for x in r['records'] if x['candidate'] == c]
    print(c, len(rows), 'build max', max((x.get('construction_seconds', 0) for x in rows), default=0))
for seed in (101, 202):
    for model in ('pauli', 'hw'):
        maps = {c: {(x['N'], x['n']): x for x in r['records'] if x['candidate'] == c and x['seed'] == seed
                    and x.get('exactness', {}).get('passed') and x.get('validity', {}).get('passed')
                    and x.get('construction_seconds', 999) <= 60 and model in x.get('models', {})} for c in 'ABCD'}
        common = set.intersection(*(set(maps[c]) for c in 'ABCD'))
        sums = {c: round(sum(maps[c][i]['models'][model]['median'] for i in common), 6) for c in 'ABCD'}
        print('COMMON', seed, model, len(common), sums)
for p in sorted((root / 'logs').glob('score_pool_*.log')):
    lines = p.read_text().splitlines()
    if lines:
        print(p.name, lines[-1][:350])
