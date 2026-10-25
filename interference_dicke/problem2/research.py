"""Save the fully compiled hybrid and finite Yu experiment with verification."""
import json
from pathlib import Path
import cirq
from .hybrid import experiment
from .yu_circuit import build
from .verify import adaptive_branches, validate_structure


def main():
    root = Path(__file__).parent / 'experiments'
    root.mkdir(exist_ok=True)
    candidates, results = experiment()
    for name, (c, q) in candidates.items():
        cirq.to_json(c, root / (name+'.cirq.json'))
    circuit, outputs = build()
    result = adaptive_branches(circuit, outputs, 2)
    assert result['fidelity'] > 1-1e-9
    assert abs(result['probability_sum']-1) < 1e-9
    results['finite_yu_round'] = dict(**result, **validate_structure(circuit, outputs, 4),
                                     moments=len(circuit), score=8/len(circuit))
    cirq.to_json(circuit, root / 'finite_yu_round.cirq.json')
    (root/'results.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
