"""Resumable bounded evaluator supervisor; all files remain under team_eval/."""
import os
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import ctypes

class AttachedProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
    def poll(self):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, self.pid)
        if not handle:
            self.returncode = 0
            return 0
        code = ctypes.c_ulong()
        okay = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        self.returncode = None if okay and code.value == 259 else (code.value if okay else 1)
        return self.returncode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = 'C:/Users/kaisa/miniforge3/python.exe'
CANDIDATES = {'A': 'team_eval/will_dicke_submission.py', 'B': 'submission.py',
              'C': 'columnforge/noise_aware/rewrites/submission_scheduled.py', 'D': 'astra/submission.py'}

def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(30):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.1)

def collect():
    records = []
    for path in sorted((HERE / 'records').glob('*.json')):
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
            key = score_key(row)
            score_path = HERE / 'scores' / f'{key}.json'
            if key and score_path.exists():
                score = json.loads(score_path.read_text(encoding='utf-8'))
                row['models'] = score.get('models', {})
                row['score_cache_key'] = key
                row['score_backend'] = score.get('backend')
                row['score_artifact_path'] = str(score_path.relative_to(ROOT)).replace('\\', '/')
                row['probabilities_path'] = score.get('probabilities_path')
                row['score_simulator_sha256'] = score.get('simulator_source_sha256', score.get('simulator_sha256', score.get('tools_sim_sha256')))
                if all(x in row['models'] for x in ('pauli', 'hw')):
                    row['status'] = 'complete'
                if score.get('error'):
                    row['failures'] = row.get('failures', []) + ['simulation_error']
                    row['simulation_error'] = score['error']
            records.append(row)
        except (OSError, ValueError):
            continue
    return records

def score_key(row):
    if 'circuit_sha256' not in row:
        return None
    identity = {k: row[k] for k in ('circuit_sha256', 'qubits', 'calibration_sha256', 'N', 'n', 'seed')}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        raise ValueError('worker cap is 5')
    for name in ('sources', 'records', 'scores', 'logs', 'circuits'):
        (HERE / name).mkdir(exist_ok=True)
    for key, source in CANDIDATES.items():
        path = HERE / 'sources' / f'{key}_submission.py'
        if not path.exists():
            path.write_bytes((ROOT / source).read_bytes())
    env = os.environ.copy()
    env.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='')
    active = []
    previous = json.loads((HERE / 'results.json').read_text(encoding='utf-8')) if (HERE / 'results.json').exists() else {}
    for task in previous.get('metadata', {}).get('active_workers', []):
        proc = AttachedProcess(task['pid'])
        if proc.poll() is None:
            log = (HERE / 'logs' / 'attached.log').open('a', encoding='utf-8')
            active.append({'name': task['name'], 'proc': proc, 'log': log})
            print(f"reattached {task['name']} pid={task['pid']}", flush=True)
    build_queue = [key for key in CANDIDATES if f'build_{key}' not in {x['name'] for x in active}]
    attempted_scores = set()
    started = previous.get('metadata', {}).get('created_unix', time.time())
    initial = {'created_unix': started, 'seeds': [101, 202], 'instances_per_seed': 36,
               'expected_records': 288, 'python': PYTHON, 'max_workers': args.workers,
               'noise_models': {'pauli': {'T1': 1e9, 'Tphi_white': 0, 'T2star': 0, 'n_traj': 2000},
                                'hw': {'defaults': 'tools.sim.HW()', 'n_traj': 1000}},
               'shots_per_basis': 10000, 'shot_resamplings': 400, 'max_correlator_size': 'N',
               'candidates': CANDIDATES, 'simulator_proxy_only': True,
               'exactness_threshold': 1 - 1e-8, 'construction_limit_seconds': 60,
               'tools_sim_sha256': hashlib.sha256((ROOT / 'tools' / 'sim.py').read_bytes()).hexdigest(),
               'deduplication': 'Reuse only byte-identical circuit JSON, same qubits, calibration, N,n,seed.'}
    (HERE / 'supervisor.pid').write_text(str(os.getpid()))
    last = None
    while True:
        for task in list(active):
            if task['proc'].poll() is not None:
                task['log'].close()
                active.remove(task)
                print(f"finished {task['name']} exit={task['proc'].returncode}", flush=True)
        while build_queue and len(active) < args.workers:
            candidate = build_queue.pop(0)
            log = (HERE / 'logs' / f'build_{candidate}.log').open('a', encoding='utf-8')
            proc = subprocess.Popen([PYTHON, '-B', '-u', str(HERE / 'evaluate_worker.py'), '--candidate', candidate],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active.append({'name': f'build_{candidate}', 'proc': proc, 'log': log})
        records = collect()
        if not args.build_only and (HERE / 'score_job.py').exists():
            for row in sorted(records, key=lambda r: (0 if r['N'] >= 6 else 1, r['N'], r['n'], r['seed'], r['candidate'])):
                if len(active) >= args.workers:
                    break
                if row['status'] != 'constructed' or not row.get('exactness', {}).get('passed') or not row.get('validity', {}).get('passed'):
                    continue
                key = score_key(row)
                if key in attempted_scores:
                    continue
                if f"score_{row['candidate']}_s{row['seed']}_{row['N']}_{row['n']}" in {t['name'] for t in active}:
                    continue
                attempted_scores.add(key)
                job = HERE / 'scores' / f'{key}.job.json'
                atomic_json(job, row)
                log = (HERE / 'logs' / f'score_{key}.log').open('a', encoding='utf-8')
                proc = subprocess.Popen([PYTHON, '-B', '-u', str(HERE / 'score_job.py'), str(job)],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                active.append({'name': f"score_{row['candidate']}_s{row['seed']}_{row['N']}_{row['n']}",
                               'proc': proc, 'log': log})
        metadata = dict(initial, updated_unix=time.time(), elapsed_seconds=time.time() - started,
                        active_workers=[{'name': x['name'], 'pid': x['proc'].pid} for x in active])
        result = {'metadata': metadata, 'records': records}
        atomic_json(HERE / 'results.json', result)
        if (HERE / 'render_results.py').exists():
            try:
                from render_results import render
                render(result, HERE / 'RESULTS.md')
            except Exception as exc:
                print(f'report render: {exc}', flush=True)
        progress = (len(records), sum(r['status'] == 'complete' for r in records),
                    sum(bool(r.get('failures')) for r in records))
        if progress != last:
            print(f'progress records={progress[0]}/288 complete={progress[1]} failures={progress[2]} active={len(active)}', flush=True)
            last = progress
        if not active and not build_queue:
            if args.build_only or len(records) == 288:
                break
            # A builder failed before creating all records; surface the incomplete run.
            print('No workers remain; inspect build logs for missing records.', flush=True)
            break
        time.sleep(3)

if __name__ == '__main__':
    main()
