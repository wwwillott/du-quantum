"""Run the notebook's actual depth score and independent state checks.

python -m interference_dicke.problem2.evaluate
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cirq
import numpy as np
from . import create_dicke_state_adaptive, method_for, interference_candidates
from .verify import unitary_fidelity, adaptive_branches, validate_structure
from .clifford_certificate import certify
from .paper import feedback_statistics


def main():
    output = Path(__file__).parent / 'evaluation.json'
    data = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                formula='sum(N*k/len(circuit) for N=2..20, k=1..N//2)',
                cirq=cirq.__version__, numpy=np.__version__, rows=[], score=0.,
                portfolio_sha256=hashlib.sha256((Path(__file__).parent/'portfolio.py').read_bytes()).hexdigest())
    for n in range(2, 21):
        for k in range(1, n//2+1):
            start = time.perf_counter()
            circuit, outputs = create_dicke_state_adaptive(n, k)
            seconds = time.perf_counter()-start
            assert seconds < 60, (n, k, seconds)
            structure = validate_structure(circuit, outputs, n)
            method = method_for(n, k)
            if circuit.has_measurements() and 'cat complement' in method:
                verification = certify(n)
            elif circuit.has_measurements():
                verification = adaptive_branches(circuit, outputs, k)
                assert abs(verification['probability_sum']-1) < 1e-9
            else:
                verification = unitary_fidelity(circuit, outputs, k)
            assert verification['fidelity'] > 1-1e-9, (n, k, verification)
            row = dict(n=n, k=k, moments=len(circuit), score=n*k/len(circuit),
                       construction_seconds=seconds, method=method,
                       verification=verification, **structure)
            data['rows'].append(row)
            data['score'] += row['score']
            output.write_text(json.dumps(data, indent=2), encoding='utf8')
            print(f'{n:2} {k:2} depth={len(circuit):3} F={verification["fidelity"]:.12f} total={data["score"]:.6f} build={seconds:.2f}s', flush=True)
    data['paper_model'] = [feedback_statistics(n, n//2) for n in (4, 8, 12, 20)]
    data['interference_comparison'] = [dict(n=n, k=k, name=name, moments=len(c),
        selected_moments=next(r['moments'] for r in data['rows'] if (r['n'],r['k']) == (n,k)))
        for (n,k), candidates in interference_candidates().items() for c,q,name in candidates]
    assert len(data['rows']) == 100 and data['score'] > 50
    data['complete'] = True
    output.write_text(json.dumps(data, indent=2), encoding='utf8')
    print(f'PASS: 100 instances; official depth score {data["score"]:.9f}', flush=True)


if __name__ == '__main__':
    main()
