"""Isolated candidate construction and strict verification. Writes only team_eval/."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import numpy as np
import cirq
import cirq_google
from official_calibration import load_calibration, calibration_fingerprint

CANDIDATES = {'A': 'team_eval/will_dicke_submission.py', 'B': 'submission.py',
              'C': 'columnforge/noise_aware/rewrites/submission_scheduled.py', 'D': 'astra/submission.py'}

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(30):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.1)

def validate(circuit, state_qubits, N, calibration):
    errors = []
    if len(state_qubits) != N:
        errors.append(f'returned {len(state_qubits)} state qubits, expected {N}')
    if not isinstance(state_qubits, set):
        errors.append('returned qubits are not a set')
    if any(not isinstance(q, cirq.GridQubit) for q in state_qubits):
        errors.append('state qubits must be GridQubits')
    if not set(state_qubits) <= set(calibration['sq_pauli_error_rate']):
        errors.append('state qubit outside SQ calibration')
    if len(circuit) > 1750:
        errors.append(f'{len(circuit)} moments exceeds 1750')
    pairs = {frozenset(pair) for pair in calibration['cz_pauli_error_rate']}
    cz_count = 0
    for op in circuit.all_operations():
        gate = op.gate
        if cirq.is_measurement(op):
            errors.append(f'measurement: {op}')
        elif isinstance(gate, cirq.CZPowGate):
            cz_count += 1
            if not np.isclose(float(gate.exponent) % 2, 1, rtol=0, atol=1e-12):
                errors.append(f'non-full CZ: {op}')
            if frozenset(op.qubits) not in pairs:
                errors.append(f'CZ on uncalibrated pair: {op.qubits}')
        elif isinstance(gate, cirq.PhasedXZGate):
            if op.qubits[0] not in calibration['sq_pauli_error_rate']:
                errors.append(f'PhasedXZ on uncalibrated qubit: {op.qubits}')
        elif isinstance(gate, cirq.ResetChannel):
            if op.qubits[0] not in calibration['sq_pauli_error_rate']:
                errors.append(f'reset on uncalibrated qubit: {op.qubits}')
        else:
            errors.append(f'unsupported gate: {op}')
    try:
        cirq_google.Willow105.validate_circuit(circuit)
        willow = True
    except Exception as exc:
        willow = False
        errors.append(f'Willow105: {type(exc).__name__}: {exc}')
    return {'passed': not errors, 'errors': errors, 'willow105': willow}, cz_count

def exactness(circuit, state_qubits, N, n):
    state = sorted(state_qubits)
    ancillas = sorted(set(circuit.all_qubits()) - set(state))
    order = state + ancillas
    target = np.array([1 / math.sqrt(math.comb(N, n)) if i.bit_count() == n else 0
                       for i in range(2 ** N)], dtype=np.complex128)
    if any(isinstance(op.gate, cirq.ResetChannel) for op in circuit.all_operations()):
        rho = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(
            circuit, qubit_order=order).final_density_matrix
        reduced = cirq.partial_trace(rho.reshape((2,) * (2 * len(order))), list(range(N)))
        fidelity = float(np.real(target.conj() @ reduced.reshape(2 ** N, 2 ** N) @ target))
    else:
        psi = cirq.final_state_vector(circuit, qubit_order=order, dtype=np.complex128)
        amplitudes = target.conj() @ psi.reshape(2 ** N, -1)
        fidelity = float(np.sum(np.abs(amplitudes) ** 2))
    return {'fidelity': fidelity, 'passed': bool(fidelity > 1 - 1e-8),
            'threshold': 1 - 1e-8, 'ancilla_count': len(ancillas), 'dtype': 'complex128'}

def run(candidate, instances, seeds):
    source = HERE / 'sources' / f'{candidate}_submission.py'
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        source.write_bytes((ROOT / CANDIDATES[candidate]).read_bytes())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    module_spec = importlib.util.spec_from_file_location(f'eval_candidate_{candidate}', source)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    started = time.perf_counter()
    module_spec.loader.exec_module(module)
    import_seconds = time.perf_counter() - started
    for N, n in instances:
        for seed in seeds:
            key = f'{candidate}_s{seed}_N{N:02d}_n{n:02d}'
            row_path = HERE / 'records' / f'{key}.json'
            if row_path.exists():
                old = json.loads(row_path.read_text(encoding='utf-8'))
                if old.get('status') in ('constructed', 'complete', 'failed'):
                    continue
            calibration = load_calibration(seed)
            record = {'candidate': candidate, 'seed': seed, 'N': N, 'n': n,
                      'source_path': CANDIDATES[candidate], 'source_sha256': digest,
                      'import_seconds': import_seconds, 'calibration_sha256': calibration_fingerprint(calibration),
                      'status': 'building', 'build_started_unix': time.time(),
                      'failures': [], 'models': {}}
            atomic_json(row_path, record)
            start = time.perf_counter()
            try:
                circuit, qubits = module.create_dicke_state(N, n, **{k: dict(v) for k, v in calibration.items()})
                record['construction_seconds'] = time.perf_counter() - start
                if record['construction_seconds'] > 60:
                    record['failures'].append('construction_over_60_seconds')
                record['moments'] = len(circuit)
                record['qubits'] = [[int(q.row), int(q.col)] for q in sorted(qubits)]
                record['all_qubit_count'] = len(set(circuit.all_qubits()) | set(qubits))
                circuit_path = HERE / 'circuits' / f'{key}.json'
                circuit_path.parent.mkdir(parents=True, exist_ok=True)
                circuit_text = cirq.to_json(circuit)
                circuit_path.write_text(circuit_text, encoding='utf-8')
                record['circuit_path'] = str(circuit_path.relative_to(ROOT)).replace('\\', '/')
                record['circuit_sha256'] = hashlib.sha256(circuit_text.encode()).hexdigest()
                record['validity'], record['cz_count'] = validate(circuit, qubits, N, calibration)
                if not record['validity']['passed']:
                    record['failures'].append('invalid')
                record['status'] = 'verifying'
                atomic_json(row_path, record)
                record['exactness'] = exactness(circuit, qubits, N, n)
                if not record['exactness']['passed']:
                    record['failures'].append('non_exact')
                record['status'] = 'constructed'
            except Exception as exc:
                record.setdefault('construction_seconds', time.perf_counter() - start)
                record['status'] = 'failed'
                record['failures'].append(type(exc).__name__)
                record['exception'] = str(exc)
                record['traceback'] = traceback.format_exc()
            atomic_json(row_path, record)
            print(json.dumps({k: record.get(k) for k in ('candidate', 'seed', 'N', 'n', 'status',
                  'construction_seconds', 'cz_count', 'moments', 'exactness', 'failures')}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', choices=list(CANDIDATES), required=True)
    parser.add_argument('--seeds', default='101,202')
    parser.add_argument('--instances', default='')
    args = parser.parse_args()
    instances = [tuple(map(int, x.split(':'))) for x in args.instances.split(',')] if args.instances else [
        (N, n) for N in [6, 7, 8, 9, 10, 11, 12, 2, 3, 4, 5] for n in range(1, N // 2 + 1)]
    run(args.candidate, instances, list(map(int, args.seeds.split(','))))
