"""Screen exact transformations with inexpensive Pauli/readout scoring."""
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
from judges_audit.compare import ROOT, calibration_maps, check
from .methods import will, candidate_pool


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',default='4:2,5:2,6:3,7:3,8:3,8:4')
    parser.add_argument('--shortlist',type=int,default=12)
    args=parser.parse_args()
    output=ROOT/'optimization_lab/runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    data=dict(stage='cheap screening only; no promotion',created_utc=datetime.now(timezone.utc).isoformat(),
              hypotheses=['H1','H2','H3'],settings=vars(args),records=[],
              source_sha256=hashlib.sha256((ROOT/'optimization_lab/methods.py').read_bytes()).hexdigest())
    maps=calibration_maps(cirq_google.engine.load_median_device_calibration('willow_pink'))
    for item in args.cases.split(','):
        n,k=map(int,item.split(':')); start=time.perf_counter()
        incumbent=cirq.read_json(ROOT/f'judges_audit/results/will-{n}-{k}.cirq.json')
        outputs=set(incumbent.all_qubits())
        family=[('will',incumbent,outputs)]
        # Hypothesis: a depth-optimized native family may be preferable on some
        # placements even when it was not selected by the hardware surrogate.
        from problem2_submission import create_dicke_state_adaptive
        p2,p2outputs=create_dicke_state_adaptive(n,k)
        if not p2.has_measurements() and p2.all_qubits()==p2outputs:
            family.append(('p2_unitary',p2,p2outputs))
        baseline=will().small_proxy(incumbent,sorted(outputs),k,*maps)
        records=[]
        for family_name,circuit,qubits in family:
            pool,enumerated=candidate_pool(circuit,qubits,k,maps,args.shortlist)
            for rank,(cost,name,c,qs,placement) in enumerate(pool):
                proxy=will().small_proxy(c,sorted(qs),k,*maps)
                label=f'{n}-{k}-{family_name}-{rank:02d}-{name}'
                result=check(c,qs,n,k,maps,0.,cirq_google.Willow105)
                cirq.to_json(c,output/(label+'.cirq.json'))
                records.append(dict(id=label,family=family_name,variant=name,
                    placement=placement,enumerated=enumerated,analytic_denominator=cost,
                    proxy_denominator=proxy,proxy_ratio=baseline/proxy,moments=len(c),
                    cz=sum(op.gate==cirq.CZ for op in c.all_operations()),verification=result,
                    circuit_file=label+'.cirq.json'))
        records.sort(key=lambda r:r['proxy_denominator'])
        data['records'].append(dict(n=n,k=k,baseline_proxy=baseline,candidates=records,
                                    seconds=time.perf_counter()-start))
        (output/'screen.json').write_text(json.dumps(data,indent=2),encoding='utf8')
        print(n,k,'best proxy ratio',round(records[0]['proxy_ratio'],4),records[0]['id'],
              'depth',records[0]['moments'],'seconds',round(time.perf_counter()-start,2),flush=True)
    data['complete']=True
    (output/'screen.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    print('RUN',output,flush=True)


if __name__=='__main__':
    main()
