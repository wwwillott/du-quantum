"""One isolated scoring job, invoked by run_evaluation.py."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from score_circuit import score_circuit, _write_result

def main():
    job_path = Path(sys.argv[1])
    row = json.loads(job_path.read_text(encoding='utf-8'))
    output_path = job_path.with_name(job_path.name.replace('.job.json', '.json'))
    probabilities = output_path.with_suffix('.npz')
    backend_path = HERE / 'backend.json'
    backend = json.loads(backend_path.read_text())['backend'] if backend_path.exists() else 'numpy'
    try:
        result = score_circuit(ROOT / row['circuit_path'], row['qubits'], row['N'], row['n'], row['seed'],
                               backend=backend, output_path=output_path, probabilities_path=probabilities,
                               progress=lambda model, stats: print(json.dumps({'model': model, **stats}), flush=True))
        _write_result(output_path, result)
    except Exception as exc:
        result = json.loads(output_path.read_text(encoding='utf-8')) if output_path.exists() else {'models': {}}
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        _write_result(output_path, result)
        raise

if __name__ == '__main__':
    main()
