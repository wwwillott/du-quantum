"""H6: bounded, deterministic exact-frame coordinate search."""
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
from .qvm import model,probabilities,expected_denominator,stats,penalties
from .frames import with_frames


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=int,default=48);parser.add_argument('--seed',type=int,default=3)
    parser.add_argument('--n',type=int,default=8);parser.add_argument('--k',type=int,default=4)
    args=parser.parse_args()
    n,k=args.n,args.k
    output=ROOT/'optimization_lab/runs'/('refine-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    baseline=cirq.read_json(ROOT/f'judges_audit/results/will-{n}-{k}.cirq.json')
    outputs=set(baseline.all_qubits());gates=[op for op in baseline.all_operations() if op.gate==cirq.CZ]
    maps=calibration_maps(cirq_google.engine.load_median_device_calibration('willow_pink'))
    errors={frozenset(edge):value[0] for edge,value in maps[0].items()}
    order=sorted(range(len(gates)),key=lambda i:(-errors[frozenset(gates[i].qubits)],-i))
    frames=np.random.default_rng(args.seed).integers(0,4,size=(len(gates),2))
    noise=model();circuit=with_frames(baseline,frames)
    p=probabilities(circuit,outputs,noise)
    best=expected_denominator(p,n,k)
    data=dict(hypothesis='H6',n=n,k=k,settings=vars(args),initial_denominator=best,evaluations=[],accepted=[],best_frames=frames.tolist())
    count=0
    for site in order:
        origin=frames.copy()
        for mask in ((1,0),(0,1),(1,1)):
            if count>=args.budget:break
            proposal=origin.copy();proposal[site]^=np.array(mask)
            changed=with_frames(baseline,proposal)
            start=time.perf_counter();p=probabilities(changed,outputs,noise)
            objective=expected_denominator(p,n,k);count+=1
            result=dict(evaluation=count,site=site,mask=mask,denominator=objective,seconds=time.perf_counter()-start)
            data['evaluations'].append(result)
            if objective<best*.995:
                best=objective;frames=proposal;circuit=changed
                data['accepted'].append(result);data['best_frames']=frames.tolist()
                cirq.to_json(circuit,output/'best.cirq.json')
                print('FRAME IMPROVEMENT',count,'denominator',round(best,6),'ratio',round(data['initial_denominator']/best,4),flush=True)
            data['best_denominator']=best
            (output/'results.json').write_text(json.dumps(data,indent=2),encoding='utf8')
        if count>=args.budget:break
    verification=check(circuit,outputs,n,k,maps,0.,cirq_google.Willow105)
    p=probabilities(circuit,outputs,model(),dtype=np.complex128)
    base=probabilities(baseline,outputs,model(),dtype=np.complex128)
    final=stats(p,n,k,repetitions=12000,seed=99003)
    reference=stats(base,n,k,repetitions=12000,seed=99003)
    data.update(complete=True,verification=verification,statistics=final,baseline_statistics=reference,
        ratio={key:final[key]['median']/reference[key]['median'] for key in final},
        penalties=penalties(p,n,k),baseline_penalties=penalties(base,n,k),
        final_expected_denominator=expected_denominator(p,n,k),moments=len(circuit),cz=len(gates))
    cirq.to_json(circuit,output/'best.cirq.json')
    np.savez_compressed(output/'probabilities.npz',PZ=p['z'],PX=p['x'],baseline_PZ=base['z'],baseline_PX=base['x'])
    (output/'results.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    print('FINAL RECORD',data['ratio'],output,flush=True)


if __name__=='__main__':main()
