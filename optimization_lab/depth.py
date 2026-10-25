"""H4: exact output symmetries and native scheduling, strict depth wins only."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import json
import time
import cirq
import numpy as np
from judges_audit.compare import ROOT
from interference_dicke.problem2.verify import unitary_fidelity,validate_structure
from problem2_submission import create_dicke_state_adaptive
from .methods import will,fuse,global_x


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-n',type=int,default=12)
    args=parser.parse_args()
    output=ROOT/'optimization_lab/runs'/('depth-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    data=dict(hypothesis='H4',records=[],delta_score=0.,settings=vars(args))
    for n in range(2,args.max_n+1):
        for k in range(1,n//2+1):
            start=time.perf_counter()
            circuit,outputs=create_dicke_state_adaptive(n,k)
            if circuit.has_measurements():
                data['records'].append(dict(n=n,k=k,skipped='adaptive branch phases need separate proof'))
                continue
            best=circuit;method='incumbent'
            variants=[('unchanged',circuit)]
            for angle in (np.pi/2,-np.pi/2,np.pi):
                rz=cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.rz(angle)))
                variants.append((f'collective_z_{angle}',fuse(circuit+cirq.Circuit(rz(q) for q in outputs))))
            if n==2*k:
                variants.append(('collective_x',global_x(circuit,outputs)))
            for label,variant in variants:
                optimized=will().optimize_native_depth(fuse(variant))
                if len(optimized)<len(best):
                    verify=unitary_fidelity(optimized,outputs,k)
                    assert verify['fidelity']>1-1e-9
                    validate_structure(optimized,outputs,n)
                    best=optimized;method=label
            gain=n*k*(1/len(best)-1/len(circuit))
            result=dict(n=n,k=k,before=len(circuit),after=len(best),method=method,score_gain=gain,
                        seconds=time.perf_counter()-start)
            if gain>0:
                result['verification']=unitary_fidelity(best,outputs,k)
                cirq.to_json(best,output/f'{n}-{k}.cirq.json')
                print('DEPTH WIN',result,flush=True)
            data['records'].append(result);data['delta_score']+=gain
            (output/'results.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    data['complete']=True
    (output/'results.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    print('DEPTH TOTAL GAIN',data['delta_score'],output,flush=True)


if __name__=='__main__':
    main()
