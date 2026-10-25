"""Take over existing workers safely and finish with four persistent CPU scorers."""
import os
import sys
sys.dont_write_bytecode = True
import ctypes
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_evaluation import AttachedProcess, atomic_json, collect, score_key, PYTHON
from render_results import render

def main():
    previous = json.loads((HERE / 'results.json').read_text())
    old_pid = int((HERE / 'supervisor.pid').read_text())
    if old_pid != os.getpid() and AttachedProcess(old_pid).poll() is None:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(1, False, old_pid)
        if not handle:
            raise RuntimeError(f'Cannot stop own previous supervisor {old_pid}')
        kernel.TerminateProcess(handle, 0)
        kernel.CloseHandle(handle)
        print('stopped previous supervisor only', old_pid, flush=True)
        # Its child workers remain running and are adopted below.
        previous = json.loads((HERE / 'results.json').read_text())
    (HERE / 'supervisor.pid').write_text(str(os.getpid()))
    metadata = previous['metadata']
    metadata['pool_mode'] = 'persistent CPU workers; imports and JIT reused'
    metadata['max_workers'] = 4
    metadata['score_parity'] = json.loads((HERE / 'score_parity_results.json').read_text())
    active = []
    for task in metadata.get('active_workers', []):
        proc = AttachedProcess(task['pid'])
        if proc.poll() is not None:
            continue
        item = dict(task, proc=proc)
        if task['name'].startswith('score_') and not task['name'].startswith('score_pool_'):
            for row in previous['records']:
                name = f"score_{row['candidate']}_s{row['seed']}_{row['N']}_{row['n']}"
                if name == task['name']:
                    key = score_key(row)
                    lock = HERE / 'scores' / f'{key}.lock'
                    lock.write_text(str(proc.pid))
                    item['lock'] = lock
                    break
        active.append(item)
        print('adopted', task, flush=True)
    env = os.environ.copy()
    env.update({key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='')
    seq = 0
    last = None
    while True:
        for lock in (HERE / 'scores').glob('*.lock'):
            try:
                content = lock.read_text()
                if content.strip():
                    dead = AttachedProcess(int(content)).poll() is not None
                else:
                    dead = time.time() - lock.stat().st_mtime > 60
                if dead:
                    lock.unlink(missing_ok=True)
            except (ValueError, OSError):
                pass
        for task in list(active):
            if task['proc'].poll() is not None:
                print('finished', task['name'], task['proc'].returncode, flush=True)
                if 'log' in task:
                    task['log'].close()
                if 'lock' in task:
                    task['lock'].unlink(missing_ok=True)
                active.remove(task)
        records = collect()
        remaining = [r for r in records if r['status'] in ('building', 'verifying') or
                     (r['status'] == 'constructed' and r.get('exactness', {}).get('passed') and
                      r.get('validity', {}).get('passed') and not r.get('simulation_error'))]
        while len(active) < 4 and (remaining or len(records) < 288):
            seq += 1
            name = f'score_pool_{os.getpid()}_{seq}'
            log = (HERE / 'logs' / f'{name}.log').open('a', encoding='utf-8')
            proc = subprocess.Popen([PYTHON, '-B', '-u', str(HERE / 'score_batch.py')], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active.append(dict(name=name, pid=proc.pid, proc=proc, log=log))
            print('launched', name, proc.pid, flush=True)
        metadata.update(updated_unix=time.time(), elapsed_seconds=time.time() - metadata['created_unix'],
                        active_workers=[{'name': t['name'], 'pid': t['proc'].pid} for t in active])
        result = {'metadata': metadata, 'records': records}
        atomic_json(HERE / 'results.json', result)
        try:
            render(result, HERE / 'RESULTS.md')
        except PermissionError:
            pass
        progress = (len(records), sum(r['status'] == 'complete' for r in records), len(remaining))
        if progress != last:
            print('PROGRESS records/complete/remaining', progress, flush=True)
            last = progress
        if not active and len(records) == 288 and not remaining:
            break
        time.sleep(3)

if __name__ == '__main__':
    main()
