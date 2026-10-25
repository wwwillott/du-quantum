"""Read-only artifact integrity audit and final report assembly (writes team_eval only)."""
import sys
sys.dont_write_bytecode = True
import ast
import hashlib
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_evaluation import collect, atomic_json
from render_results import render

def main():
    rows = collect()
    current = json.loads((HERE / 'results.json').read_text())
    findings = []
    expected = {(c, seed, N, n) for c in 'ABCD' for seed in (101, 202)
                for N in range(2, 13) for n in range(1, N // 2 + 1)}
    actual = {(r['candidate'], r['seed'], r['N'], r['n']) for r in rows}
    if actual != expected or len(rows) != 288:
        findings.append('missing_or_duplicate_records')
    score_keys = set()
    backends = {}
    missing_models = 0
    for row in rows:
        label = [row[k] for k in ('candidate', 'seed', 'N', 'n')]
        if not row.get('exactness', {}).get('passed') or not row.get('exactness', {}).get('fidelity', 0) > 1 - 1e-8:
            findings.append([label, 'exactness'])
        if not row.get('validity', {}).get('passed') or not row.get('validity', {}).get('willow105') or row.get('moments', 99999) > 1750:
            findings.append([label, 'validity'])
        # Worker hashes cirq.to_json text before Windows text-mode CRLF expansion.
        circuit_file = ROOT / row['circuit_path']
        if hashlib.sha256(circuit_file.read_text(encoding='utf-8').encode('utf-8')).hexdigest() != row['circuit_sha256']:
            findings.append([label, 'circuit_hash'])
        row['circuit_file_sha256'] = hashlib.sha256(circuit_file.read_bytes()).hexdigest()
        source = HERE / 'sources' / f"{row['candidate']}_submission.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['source_sha256']:
            findings.append([label, 'source_hash'])
        calibration = json.loads((HERE / 'calibrations' / f"seed_{row['seed']}.json").read_text())
        if calibration['fingerprint'] != row['calibration_sha256']:
            findings.append([label, 'calibration_hash'])
        if any(m not in row.get('models', {}) for m in ('pauli', 'hw')):
            missing_models += 1
            continue
        score_keys.add(row['score_cache_key'])
        score = json.loads((ROOT / row['score_artifact_path']).read_text())
        backends[score['backend']] = backends.get(score['backend'], 0) + 1
        if score['K'] != row['N'] or score['shots_per_basis'] != 10000 or score['resamplings'] < 400:
            findings.append([label, 'sampling_settings'])
        for model, minimum in (('pauli', 2000), ('hw', 1000)):
            stats = score['models'][model]
            if stats['n_traj'] < minimum or not all(math.isfinite(stats[x]) for x in ('median', 'mean')):
                findings.append([label, 'model_stats', model])
            if max(abs(stats[x] - 1) for x in ('pz_sum', 'px_sum')) > 1e-9:
                findings.append([label, 'probability_sum', model])
    syntax = []
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'))
        syntax.append(path.name)
    smoke = json.loads((HERE / 'cli_smoke.json').read_text()) if (HERE / 'cli_smoke.json').exists() else None
    smoke_deltas = None
    if smoke is not None:
        baseline = next(r for r in rows if (r['candidate'], r['seed'], r['N'], r['n']) == ('A', 101, 2, 1))
        smoke_deltas = {model: {stat: smoke['models'][model][stat] - baseline['models'][model][stat]
                              for stat in ('median', 'mean')} for model in ('pauli', 'hw')}
        if any(abs(d) > 1e-10 for model in smoke_deltas.values() for d in model.values()):
            findings.append('cli_smoke_mismatch')
    inventory = []
    inventory_error = None
    try:
        import psutil
        for process in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmd = ' '.join(process.info['cmdline'] or [])
                if any(f'team_eval{sep}{name}' in cmd for sep in ('/', '\\')
                       for name in ('evaluate_worker.py', 'score_job.py', 'score_batch.py', 'timing_retry.py', 'score_circuit.py')):
                    inventory.append({'pid': process.pid, 'cmdline': cmd})
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
    except Exception as exc:
        inventory_error = str(exc)
    audit = {'checked_unix': time.time(), 'records': len(rows), 'missing_models': missing_models,
             'unique_scored_circuits': len(score_keys), 'backend_record_counts': backends,
             'findings': findings, 'passed': not findings and missing_models == 0,
             'syntax_checked': syntax, 'cli_smoke_deltas': smoke_deltas,
             'worker_inventory': inventory, 'worker_inventory_error': inventory_error,
             'minimum_fidelity': min(r.get('exactness', {}).get('fidelity', 1) for r in rows),
             'max_moments': max(r.get('moments', 0) for r in rows)}
    atomic_json(HERE / 'audit_results.json', audit)
    print(json.dumps({**audit, 'findings': findings[:12]}, indent=2))
    if '--finalize' in sys.argv:
        if not audit['passed']:
            raise RuntimeError('Refusing finalization before all artifact checks pass')
        current['records'] = rows
        current['metadata'].update(updated_unix=time.time(), final_audit='team_eval/audit_results.json',
                                   score_backend_evidence='team_eval/backend.json',
                                   pool_worker_limit=4, auxiliary_worker_limit=1, max_worker_processes=5,
                                   circuit_sha256_encoding='UTF-8 Cirq JSON with normalized LF; circuit_file_sha256 hashes saved bytes',
                                   complete=True)
        if (HERE / 'timing_retry_results.json').exists():
            current['metadata']['timing_retry'] = json.loads((HERE / 'timing_retry_results.json').read_text())
        atomic_json(HERE / 'results.json', current)
        render(current, HERE / 'RESULTS.md')

if __name__ == '__main__':
    main()
