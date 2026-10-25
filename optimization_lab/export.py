"""Export guarded exact-frame improvements without replacing the incumbent."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import cirq
from judges_audit.compare import ROOT
from .runtime import _record_signature


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',nargs='+',required=True)
    args=parser.parse_args();specs={}
    for entry in args.runs:
        path=Path(entry);data=json.loads((path/'results.json').read_text())
        assert data['complete']
        n,k=data.get('n',8),data.get('k',4)
        baseline=cirq.read_json(ROOT/f'judges_audit/results/will-{n}-{k}.cirq.json')
        specs[n,k]=dict(signature=_record_signature(baseline),frames=data['best_frames'],
                        experiment=str(path.relative_to(ROOT) if path.is_absolute() else path),
                        hypothesis=data['hypothesis'])
    base_path=ROOT/'recommended_submission.py'
    base=base_path.read_text(encoding='utf8')
    content=base+'\n\n# EXACT PAULI FRAME POSTPROCESSOR; experiments in optimization_lab/README.md\n'
    content+=(ROOT/'optimization_lab/runtime.py').read_text(encoding='utf8')
    content+='\n_RECORD_FRAME_SPECS = '+repr(specs)+'\n'
    content+='''
_RECORD_BASE_HARDWARE = create_dicke_state


def create_dicke_state(num_qubits, num_excitations, cz_pauli_error_rate,
                       sq_pauli_error_rate, zero_errors, one_errors):
    circuit, outputs = _RECORD_BASE_HARDWARE(num_qubits, num_excitations,
        cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    spec = _RECORD_FRAME_SPECS.get((num_qubits,num_excitations))
    if spec is None or _record_signature(circuit) != spec['signature']:
        return circuit, outputs
    result = _record_apply(circuit,spec['frames'])
    if len(result)>1750:
        return circuit, outputs
    return result, outputs
'''
    destination=ROOT/'record_submission.py'
    destination.write_text(content,encoding='utf8')
    manifest=dict(file='record_submission.py',file_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        incumbent_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        specs=[dict(n=n,k=k,**data) for (n,k),data in specs.items()],
        p2='Unchanged delegate and entire incumbent source preserved verbatim as prefix',
        performance_scope='Calibrated Willow QVM; guarded matched circuit families. No universal hardware record claimed.')
    (ROOT/'optimization_lab/export_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('Exported',destination,'guarded cases',list(specs),flush=True)


if __name__=='__main__':main()
