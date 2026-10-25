"""Calibrated Willow QVM benchmark; never report this as physical hardware.

python -m judges_audit.compare [--max-n 8] [--candidate team|will|kai_v2|scs]
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import time

import cirq
import cirq_google
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'judges_audit' / 'results'
ORIGINAL = ROOT / 'judges_audit/sources/judges_original.ipynb'
SOURCES = {
    'team': ROOT/'dicke_submission.py',
    'will': ROOT/'adaptive_dicke_validation/snapshots/will-605b9a8/dicke_submission.py',
    'kai_v2': ROOT/'adaptive_dicke_validation/snapshots/kai-48115b9/final_v2/submission_v2.py',
    'scs': ROOT/'dicke_willow/challenge.py',
    'kai_latest': ROOT/'judges_audit/sources/kai_4163b40.py',
}


def official():
    notebook = json.loads(ORIGINAL.read_text(encoding='utf8'))
    nodes = ast.parse(''.join(notebook['cells'][7]['source'])).body
    assert all(isinstance(node, ast.FunctionDef) for node in nodes)
    scope = dict(np=np, math=math, itertools=itertools)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'judges_original_cell_7', 'exec'), scope)
    return scope


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def calibration_maps(cal):
    def rates(name):
        return {key[0]: (float(value[0]), 0.) for key, value in cal[name].items()}
    return ({tuple(key): (float(value[0]), 0.) for key, value in cal['two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle'].items()},
            rates('single_qubit_rb_pauli_error_per_gate'), rates('single_qubit_p00_error'), rates('single_qubit_p11_error'))


def check(circuit, outputs, n, k, maps, seconds, device):
    cz, sq, zero, one = maps
    assert seconds < 60 and len(circuit) <= 1750
    assert len(outputs) == n and set(outputs) <= set(sq) & set(zero) & set(one)
    for op in circuit.all_operations():
        assert not op.classical_controls and not cirq.is_measurement(op)
        assert set(op.qubits) <= set(sq)
        assert isinstance(op.gate, cirq.PhasedXZGate) or op.gate == cirq.CZ
        if op.gate == cirq.CZ:
            assert op.qubits in cz or op.qubits[::-1] in cz
    device.validate_circuit(circuit)
    cirq_google.CircuitSerializer().serialize(circuit)
    from interference_dicke.problem2.verify import unitary_fidelity
    result = unitary_fidelity(circuit, outputs, k)
    assert result['fidelity'] > 1-1e-9
    return result


def fast_score(z, x, n, k, max_k):
    """Algebraic count/histogram form, checked against the literal functions."""
    weights = np.array([i.bit_count() for i in range(1 << n)])
    shell = z[weights == k]
    fraction = shell.sum()/z.sum()
    if not fraction:
        return 0.
    m = math.comb(n, k)
    expected = shell.sum()/m
    penalty_z = np.sum((shell-expected)**2/expected)/(m*fraction)
    hist = np.bincount(weights, weights=x, minlength=n+1)/x.sum()
    penalty_x = 0.
    def comb(a, b):
        return math.comb(a, b) if 0 <= b <= a else 0
    for ell in range(1, max_k+1):
        ideal = comb(ell, ell//2)*comb(n-ell, k-ell//2)/comb(n,k) if ell%2 == 0 else 0.
        correlations = [sum((-1)**j*comb(w,j)*comb(n-w,ell-j) for j in range(ell+1))/comb(n,ell) for w in range(n+1)]
        measured = np.dot(hist, correlations)
        penalty_x += (ideal-measured)**2/(ideal**2 if ideal else 1)
    return float(1/(penalty_z+penalty_x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-n', type=int, default=8)
    parser.add_argument('--candidate', choices=SOURCES, required=True)
    parser.add_argument('--cases', help='Explicit benchmark cases, e.g. 9:1,10:3')
    parser.add_argument('--label', help='Separate result label for a bounded extension')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    scorer = official()
    module = load(SOURCES[args.candidate], 'benchmark_'+args.candidate)
    engine = cirq_google.engine.create_default_noisy_quantum_virtual_machine(
        'willow_pink', simulator_class=cirq.DensityMatrixSimulator, dtype=np.complex128, seed=20261025)
    processor = engine.get_processor('willow_pink')
    maps = calibration_maps(processor.get_current_calibration())
    sampler = processor._sampler
    while hasattr(sampler, '_sampler'):
        sampler = sampler._sampler
    noise = sampler.noise
    data = dict(candidate=args.candidate, source=str(SOURCES[args.candidate]),
                source_sha256=hashlib.sha256(SOURCES[args.candidate].read_bytes()).hexdigest(),
                judges_notebook_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                backend='Google calibrated noisy Willow QVM; exact density matrices',
                physical_hardware=False, calibration_timestamp=processor.get_current_calibration().timestamp,
                shots_per_basis=10000, bootstrap_repetitions=400, rows=[])
    label = args.label or args.candidate
    destination = OUT/(label+'.json')
    rng = np.random.default_rng(43210)
    for n in range(2, args.max_n+1):
        for k in range(1, n//2+1):
            if args.cases and f'{n}:{k}' not in args.cases.split(','):
                continue
            started = time.perf_counter()
            try:
                before = time.perf_counter()
                circuit, outputs = module.create_dicke_state(n, k, *maps)
                seconds = time.perf_counter()-before
                verified = check(circuit, outputs, n, k, maps, seconds, processor.get_device())
                order = sorted(outputs) + sorted(circuit.all_qubits()-set(outputs))
                bits, probabilities = {}, {}
                for basis in ('z', 'x'):
                    measured = cirq.Circuit(circuit)
                    if basis == 'x':
                        measured += cirq.Circuit(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H)).on(q) for q in sorted(outputs))
                    measured += cirq.Circuit(cirq.measure(*sorted(outputs), key='result'))
                    bits[basis] = engine.get_sampler('willow_pink').run(measured, repetitions=10000).measurements['result']
                    # Retain every noise channel associated with readout; only
                    # remove the final ideal projective measurements.
                    expanded = cirq.Circuit(noise.noisy_moments(measured, order))
                    without_measurement = cirq.Circuit(op for op in expanded.all_operations() if not cirq.is_measurement(op))
                    rho = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(without_measurement, qubit_order=order).final_density_matrix
                    probabilities[basis] = np.real(np.diag(rho)).reshape(1 << n, -1).sum(axis=1)
                    probabilities[basis] = np.maximum(probabilities[basis], 0)
                    probabilities[basis] /= probabilities[basis].sum()
                scores, bootstrap = {}, {}
                counts = {b: np.bincount(bits[b] @ (1 << np.arange(n-1,-1,-1)), minlength=1 << n) for b in bits}
                # K is unspecified by the judges; report both common choices.
                for max_k in sorted({min(4,n), n}):
                    reference = float(scorer['overall_score'](bits['z'], bits['x'], k, max_k))
                    assert np.isclose(reference, fast_score(counts['z'],counts['x'],n,k,max_k), rtol=1e-10, atol=1e-10)
                    trials = [fast_score(rng.multinomial(10000,probabilities['z']),rng.multinomial(10000,probabilities['x']),n,k,max_k) for _ in range(400)]
                    scores[str(max_k)] = reference
                    bootstrap[str(max_k)] = dict(median=float(np.median(trials)), mean=float(np.mean(trials)),
                        q025=float(np.quantile(trials,.025)), q975=float(np.quantile(trials,.975)))
                row = dict(n=n,k=k,accepted=True,moments=len(circuit),cz=sum(op.gate == cirq.CZ for op in circuit.all_operations()),
                    construction_seconds=seconds,verification=verified,scores=scores,shot_statistics=bootstrap,
                    z_sector_retention=float(probabilities['z'][[i.bit_count()==k for i in range(1 << n)]].sum()))
                cirq.to_json(circuit, OUT/f'{label}-{n}-{k}.cirq.json')
                np.savez_compressed(OUT/f'{label}-{n}-{k}.npz',PZ=probabilities['z'],PX=probabilities['x'],z_bits=bits['z'],x_bits=bits['x'])
                print(args.candidate,n,k,'depth',len(circuit),'scores',scores,'seconds',round(time.perf_counter()-started,2),flush=True)
            except Exception as exc:
                row = dict(n=n,k=k,accepted=False,error=type(exc).__name__+': '+str(exc))
                import traceback
                row['traceback'] = traceback.format_exc()
                print(args.candidate,n,k,row['error'],flush=True)
            data['rows'].append(row)
            destination.write_text(json.dumps(data,indent=2),encoding='utf8')
    data['complete'] = True
    destination.write_text(json.dumps(data,indent=2),encoding='utf8')


if __name__ == '__main__':
    main()
