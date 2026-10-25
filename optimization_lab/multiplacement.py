"""H7: learn a frame pattern on three placements; reserve two other patches."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import json
import cirq
import cirq_google
import numpy as np
from judges_audit.compare import ROOT,calibration_maps,check
from .methods import placements,corrected_cost
from .frames import with_frames
from .qvm import model,model_fixed_stochastic,probabilities,expected_denominator,stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=int,default=24)
    args=parser.parse_args();n,k=8,3
    output=ROOT/'optimization_lab/runs'/('multiplacement-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    baseline=cirq.read_json(ROOT/'judges_audit/results/will-8-3.cirq.json');qs=set(baseline.all_qubits())
    maps=calibration_maps(cirq_google.engine.load_median_device_calibration('willow_pink'))
    pool=sorted([(corrected_cost(c,q,k,*maps),c,q,p) for c,q,p in placements(baseline,qs,maps) if q!=qs],key=lambda t:t[0])
    selected=[];seen=set()
    for item in pool:
        if frozenset(item[2]) not in seen:
            seen.add(frozenset(item[2]));selected.append(item)
        if len(selected)==5:break
    configurations=[(baseline,qs,None)]+[(item[1],item[2],item[3]) for item in selected]
    train=configurations[:3];holdout=configurations[4:6]
    noise=model()
    denominators=[expected_denominator(probabilities(c,q,noise),n,k) for c,q,_ in train]
    gates=[op for op in baseline.all_operations() if op.gate==cirq.CZ]
    data=dict(hypothesis='H7',n=n,k=k,settings=vars(args),training_placements=[p for _,_,p in train],
        reserved_placements=[p for _,_,p in holdout],evaluations=[],accepted=[],holdouts=[])
    def save(): (output/'results.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    def transformed(frames,configuration):
        _,q,placement=configuration
        c=with_frames(baseline,frames)
        if placement:
            c=c.transform_qubits(lambda x:cirq.GridQubit(placement['sx']*(x.col if placement['swap'] else x.row)+placement['dr'],
                placement['sy']*(x.row if placement['swap'] else x.col)+placement['dc']))
        return c,q
    def objective(frames):
        ratios=[]
        for configuration,denominator in zip(train,denominators):
            c,q=transformed(frames,configuration)
            ratios.append(expected_denominator(probabilities(c,q,noise),n,k)/denominator)
        return float(np.mean(ratios)+.5*max(ratios)),ratios
    candidates=[np.zeros((len(gates),2),int)]+[np.random.default_rng(seed).integers(0,4,size=(len(gates),2)) for seed in (5,7)]
    evaluated=[(objective(frames),frames) for frames in candidates]
    (best,ratios),frames=min(evaluated,key=lambda t:t[0][0])
    frames=frames.copy();data['initial_objective']=best
    errors={frozenset(edge):v[0] for edge,v in maps[0].items()}
    order=sorted(range(len(gates)),key=lambda i:(-errors[frozenset(gates[i].qubits)],-i))
    count=0
    for site in order:
        origin=frames.copy()
        for mask in ((1,0),(0,1),(1,1)):
            if count>=args.budget:break
            proposed=origin.copy();proposed[site]^=np.array(mask)
            cost,ratios=objective(proposed);count+=1
            row=dict(evaluation=count,site=site,mask=mask,objective=cost,normalized_denominators=ratios)
            data['evaluations'].append(row)
            if cost<best*.995:
                best=cost;frames=proposed;data['accepted'].append(row)
                print('MULTIPLACEMENT IMPROVEMENT',count,cost,ratios,flush=True)
            data.update(best_objective=best,best_frames=frames.tolist());save()
        if count>=args.budget:break
    c=with_frames(baseline,frames)
    check(c,qs,n,k,maps,0.,cirq_google.Willow105)
    cirq.to_json(c,output/'best.cirq.json')
    for label,configuration,test_noise in [('nominal',configurations[0],model()),
            ('coherent_sign',configurations[0],model_fixed_stochastic(-1.))]+[(f'heldout_patch_{i+4}',config,model()) for i,config in enumerate(holdout)]:
        base,outputs,_=configuration
        changed,_=transformed(frames,configuration)
        check(changed,outputs,n,k,maps,0.,cirq_google.Willow105)
        base_p=probabilities(base,outputs,test_noise,dtype=np.complex128)
        p=probabilities(changed,outputs,test_noise,dtype=np.complex128)
        b=stats(base_p,n,k,repetitions=8000,seed=88117);s=stats(p,n,k,repetitions=8000,seed=88117)
        result=dict(test=label,statistics=s,baseline=b,ratio={key:s[key]['median']/b[key]['median'] for key in s})
        data['holdouts'].append(result);save();print('MULTIPLACEMENT VALIDATION',label,result['ratio'],flush=True)
        if label=='nominal':
            np.savez_compressed(output/'probabilities.npz',PZ=p['z'],PX=p['x'],baseline_PZ=base_p['z'],baseline_PX=base_p['x'])
            data.update(statistics=s,baseline_statistics=b,ratio=result['ratio'])
    data['complete']=True;save();print('RUN',output,flush=True)


if __name__=='__main__':main()
