"""Fresh complex128 validation, coherent-angle stress, unseen placements."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import cirq
import cirq_google
import numpy as np
from judges_audit.compare import ROOT,calibration_maps,check,official
from .qvm import model,model_fixed_stochastic,probabilities,stats,penalties
from .methods import placements,corrected_cost


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--refined',action='store_true')
    args=parser.parse_args();source=Path(args.run)
    screened=json.loads((source/'results.json').read_text())
    if args.refined:
        n,k=screened.get('n',8),screened.get('k',4)
        screened=dict(records=[
            dict(n=n,k=k,variant='unchanged',circuit_file=str(ROOT/f'judges_audit/results/will-{n}-{k}.cirq.json'),
                 statistics=screened['baseline_statistics'],ratio={str(n):1.}),
            dict(n=n,k=k,variant='refined_frames',circuit_file='best.cirq.json',
                 statistics=screened['statistics'],ratio=screened['ratio'])])
    output=ROOT/'optimization_lab/runs'/('validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    maps=calibration_maps(cirq_google.engine.load_median_device_calibration('willow_pink'))
    data=dict(source_run=str(source),stage='complex128, fresh resampling seeds, 8000 trials per nominal candidate',
              records=[],holdouts=[],screen_did_not_use_validation_seeds=True,
              source_hashes={name:hashlib.sha256((ROOT/'optimization_lab'/name).read_bytes()).hexdigest()
                             for name in ('methods.py','frames.py','qvm.py','validate.py')})
    def save():
        (output/'results.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    def evaluate(circuit,qs,n,k,noise,reps=8000):
        verification=check(circuit,qs,n,k,maps,0.,cirq_google.Willow105)
        p=probabilities(circuit,qs,noise,dtype=np.complex128)
        return p,dict(statistics=stats(p,n,k,repetitions=reps,seed=77377),
                      penalties=penalties(p,n,k),verification=verification)
    for n,k in sorted({(r['n'],r['k']) for r in screened['records']}):
        rows=[r for r in screened['records'] if (r['n'],r['k'])==(n,k)]
        ordered=sorted(rows,key=lambda r:r['statistics'][str(n)]['median'],reverse=True)
        chosen=[r for r in ordered[:2] if r['ratio'][str(n)]>1.05]
        if not chosen:
            data.setdefault('rejected',[]).append(dict(n=n,k=k,reason='No screened candidate exceeds preregistered 5% lead'))
            save();continue
        baseline=cirq.read_json(source/next(r['circuit_file'] for r in rows if r['variant']=='unchanged'))
        qs=set(baseline.all_qubits())
        pbase,bbase=evaluate(baseline,qs,n,k,model())
        validated=[]
        for row in chosen:
            c=cirq.read_json(source/row['circuit_file'])
            p,record=evaluate(c,qs,n,k,model())
            record.update(n=n,k=k,variant=row['variant'],baseline=bbase,
                ratio={key:record['statistics'][key]['median']/bbase['statistics'][key]['median'] for key in record['statistics']},
                moments=len(c),cz=sum(op.gate==cirq.CZ for op in c.all_operations()))
            name=f'{n}-{k}-{row["variant"]}'
            cirq.to_json(c,output/(name+'.cirq.json'))
            np.savez_compressed(output/(name+'.npz'),PZ=p['z'],PX=p['x'],baseline_PZ=pbase['z'],baseline_PX=pbase['x'])
            record['circuit_file']=name+'.cirq.json'
            validated.append((record,c));data['records'].append(record);save()
            print('VALIDATED',n,k,row['variant'],record['ratio'],flush=True)
        record,winner=max(validated,key=lambda t:t[0]['statistics'][str(n)]['median'])
        if min(record['ratio'].values())<1.05:
            continue
        for scale in (-1.,0.,.5,1.5):
            noise=model_fixed_stochastic(scale)
            _,b=evaluate(baseline,qs,n,k,noise,reps=3000)
            _,v=evaluate(winner,qs,n,k,noise,reps=3000)
            result=dict(n=n,k=k,test='coherent_angle_stress',scale=scale,variant=record['variant'],
                baseline=b,changed=v,ratio={key:v['statistics'][key]['median']/b['statistics'][key]['median'] for key in v['statistics']})
            data['holdouts'].append(result);save();print('STRESS',n,k,scale,result['ratio'],flush=True)
        pool=[]
        for c,outputs,placement in placements(baseline,qs,maps):
            if outputs==qs:
                continue
            pool.append((corrected_cost(c,outputs,k,*maps),c,outputs,placement))
        pool.sort(key=lambda p:p[0]);seen=set();selected=[]
        for item in pool:
            if frozenset(item[2]) not in seen:
                seen.add(frozenset(item[2]));selected.append(item)
            if len(selected)==3:break
        for _,placed,new_outputs,placement in selected:
            def move(q):
                return cirq.GridQubit(placement['sx']*(q.col if placement['swap'] else q.row)+placement['dr'],
                    placement['sy']*(q.row if placement['swap'] else q.col)+placement['dc'])
            changed=winner.transform_qubits(move)
            _,b=evaluate(placed,new_outputs,n,k,model(),reps=3000)
            _,v=evaluate(changed,new_outputs,n,k,model(),reps=3000)
            result=dict(n=n,k=k,test='unseen_physical_placement',placement=placement,variant=record['variant'],
                baseline=b,changed=v,ratio={key:v['statistics'][key]['median']/b['statistics'][key]['median'] for key in v['statistics']})
            data['holdouts'].append(result);save();print('PLACEMENT HOLDOUT',n,k,result['ratio'],flush=True)
    data['complete']=True;save();print('RUN',output,flush=True)


if __name__=='__main__':main()
