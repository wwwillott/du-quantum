"""Score final_v2/submission_v2.py with the team_eval pipeline (same calibrations, RNG seeds, trajectory counts).

Stage 1 builds every v2 circuit and checks validity/exactness. If a circuit's score-cache key already exists in
team_eval/scores (i.e. the circuit is byte-identical to one A or B produced), that score is reused; otherwise the
circuit is scored here. Writes only to this script's directory.
usage (from the repo root): python <this file> [workers]
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, json, time, hashlib, importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'team_eval')); sys.path.insert(0, str(ROOT))
SEEDS = (101, 202)
INST = [(N, n) for N in range(2, 12) for n in range(1, N // 2 + 1)]


def key_of(row):
    ident = {k: row[k] for k in ('circuit_sha256', 'qubits', 'calibration_sha256', 'N', 'n', 'seed')}
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()


def build():
    import cirq
    from official_calibration import load_calibration, calibration_fingerprint
    from evaluate_worker import validate, exactness
    spec = importlib.util.spec_from_file_location('v2', ROOT / 'final_v2' / 'submission_v2.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rows = []
    for seed in SEEDS:
        cal = load_calibration(seed)
        for N, n in INST:
            t = time.perf_counter()
            c, qs = mod.create_dicke_state(N, n, **{k: dict(v) for k, v in cal.items()})
            dt = time.perf_counter() - t
            text = cirq.to_json(c)
            path = OUT / 'circuits' / f'E_s{seed}_N{N:02d}_n{n:02d}.json'
            path.parent.mkdir(exist_ok=True); path.write_text(text, encoding='utf-8')
            row = dict(seed=seed, N=N, n=n, construction_seconds=dt, moments=len(c),
                       qubits=[[int(q.row), int(q.col)] for q in sorted(qs)],
                       circuit_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       calibration_sha256=calibration_fingerprint(cal), circuit_path=str(path))
            row['validity'], row['cz_count'] = validate(c, qs, N, cal)
            row['exactness'] = exactness(c, qs, N, n)
            row['key'] = key_of(row)
            cached = ROOT / 'team_eval' / 'scores' / f"{row['key']}.json"
            if cached.exists() and 'pauli' in json.loads(cached.read_text()).get('models', {}):
                row['models'] = json.loads(cached.read_text())['models']; row['score_source'] = 'cache'
            rows.append(row)
            print(f"built s{seed} D({N},{n}) cz={row['cz_count']} mom={len(c)} t={dt:.1f}s "
                  f"valid={row['validity']['passed']} F={row['exactness']['fidelity']:.10f} "
                  f"{row.get('score_source', 'to-score')}", flush=True)
    return rows


def score_job(row):
    from official_calibration import load_calibration
    from score_circuit import score_circuit
    res = score_circuit(row['circuit_path'], json.dumps(row['qubits']), row['N'], row['n'], row['seed'],
                        calibration=load_calibration(row['seed']), backend='numba', check_circuit=False)
    return row['key'], res['models']


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = build()
    (OUT / 'rows.json').write_text(json.dumps(rows, indent=1))
    todo = [r for r in rows if 'models' not in r]
    print(f'{len(rows) - len(todo)} cached, {len(todo)} to score', flush=True)
    by_key = {r['key']: r for r in rows}
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(score_job, r) for r in sorted(todo, key=lambda r: -r['N'])]
        for f in as_completed(futs):
            k, models = f.result()
            r = by_key[k]; r['models'] = models; r['score_source'] = 'scored'
            print(f"scored s{r['seed']} D({r['N']},{r['n']}) pauli={models['pauli']['median']:.5f} "
                  f"hw={models['hw']['median']:.5f}", flush=True)
            (OUT / 'rows.json').write_text(json.dumps(rows, indent=1))
    print('done', flush=True)
