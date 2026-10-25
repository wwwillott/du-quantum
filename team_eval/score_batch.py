"""Persistent CPU scoring worker; shared file claims avoid duplicate circuits."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
print('worker starting imports', os.getpid(), flush=True)
from score_circuit import score_circuit, _write_result
from run_evaluation import collect, score_key, atomic_json, AttachedProcess
print('worker imports complete', os.getpid(), flush=True)

def main():
    backend_path = HERE / 'backend.json'
    while True:
        records = collect()
        records.sort(key=lambda r: (0 if r['N'] >= 6 else 1, r['N'], r['n'], r['seed'], r['candidate']))
        did_work = False
        for row in records:
            if row.get('status') != 'constructed' or not row.get('exactness', {}).get('passed') or not row.get('validity', {}).get('passed'):
                continue
            key = score_key(row)
            output = HERE / 'scores' / f'{key}.json'
            lock = HERE / 'scores' / f'{key}.lock'
            if output.exists():
                prior = json.loads(output.read_text())
                if all(m in prior.get('models', {}) for m in ('pauli', 'hw')) or prior.get('error'):
                    continue
            if lock.exists():
                continue  # Only the supervisor reclaims stale claims.
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, 'w') as file:
                file.write(str(os.getpid()))
            did_work = True
            label = f"{row['candidate']} s{row['seed']} D({row['N']},{row['n']})"
            try:
                # Recheck after claim in case another worker just completed it.
                if output.exists():
                    prior = json.loads(output.read_text())
                    if all(m in prior.get('models', {}) for m in ('pauli', 'hw')):
                        continue
                atomic_json(HERE / 'scores' / f'{key}.job.json', row)
                backend = json.loads(backend_path.read_text())['backend'] if backend_path.exists() else 'numpy'
                print('START', label, backend, flush=True)
                result = score_circuit(ROOT / row['circuit_path'], row['qubits'], row['N'], row['n'], row['seed'],
                                       backend=backend, output_path=output, probabilities_path=output.with_suffix('.npz'),
                                       progress=lambda model, stats: print(label, model, json.dumps(stats), flush=True))
                _write_result(output, result)
            except Exception as exc:
                result = json.loads(output.read_text()) if output.exists() else {'models': {}}
                result.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
                _write_result(output, result)
                print('FAILED', label, traceback.format_exc(), flush=True)
            finally:
                lock.unlink(missing_ok=True)
            break
        if not did_work:
            if len(records) == 288 and all(r.get('status') in ('complete', 'failed') or
                (r.get('status') == 'constructed' and (not r.get('exactness', {}).get('passed') or not r.get('validity', {}).get('passed') or r.get('simulation_error')))
                for r in records):
                break
            time.sleep(2)
    print('worker complete', os.getpid(), flush=True)

if __name__ == '__main__':
    main()
