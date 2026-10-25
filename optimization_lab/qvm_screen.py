"""H2/H3/H5: low-repetition calibrated-QVM screening with exact ideal checks."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import json
import time
import cirq
import cirq_google
import numpy as np
from judges_audit.compare import ROOT,calibration_maps,check
from .qvm import model,probabilities,stats,penalties
from .methods import variants
from .frames import twirl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',default='4:2,6:3,8:4')
    parser.add_argument('--seeds',type=int,default=8)
    args=parser.parse_args()
    output=ROOT/'optimization_lab/runs'/('qvm-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    data=dict(hypotheses=['H2','H3','H5'],stage='complex64 full-QVM screening; 64 resamplings, no promotion',
              settings=vars(args),records=[])
    noise=model();maps=calibration_maps(cirq_google.engine.load_median_device_calibration('willow_pink'))
    for item in args.cases.split(','):
        n,k=map(int,item.split(':'))
        incumbent=cirq.read_json(ROOT/f'judges_audit/results/will-{n}-{k}.cirq.json')
        outputs=set(incumbent.all_qubits())
        candidates=list(variants(incumbent,outputs,k))
        candidates.extend((f'pauli_seed_{seed}',twirl(incumbent,seed)) for seed in range(args.seeds))
        baseline=None
        for label,circuit in candidates:
            start=time.perf_counter()
            verification=check(circuit,outputs,n,k,maps,0.,cirq_google.Willow105)
            p=probabilities(circuit,outputs,noise)
            measured=stats(p,n,k)
            if baseline is None:
                baseline=measured
            row=dict(n=n,k=k,variant=label,moments=len(circuit),statistics=measured,
                     ratio={key:measured[key]['median']/baseline[key]['median'] for key in measured},
                     verification=verification,penalties=penalties(p,n,k),seconds=time.perf_counter()-start)
            name=f'{n}-{k}-{label}'
            cirq.to_json(circuit,output/(name+'.cirq.json'))
            np.savez_compressed(output/(name+'.npz'),PZ=p['z'],PX=p['x'])
            row['circuit_file']=name+'.cirq.json';data['records'].append(row)
            (output/'results.json').write_text(json.dumps(data,indent=2),encoding='utf8')
            print(n,k,label,'ratio',round(row['ratio'][str(n)],4),'depth',len(circuit),'seconds',round(row['seconds'],2),flush=True)
        # Attribution ablation changes the noise, not the candidate circuit.
        p=probabilities(incumbent,outputs,model(0.))
        data.setdefault('coherent_error_ablation',[]).append(dict(n=n,k=k,statistics=stats(p,n,k),penalties=penalties(p,n,k)))
        (output/'results.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    data['complete']=True
    (output/'results.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    print('RUN',output,flush=True)


if __name__=='__main__':
    main()
