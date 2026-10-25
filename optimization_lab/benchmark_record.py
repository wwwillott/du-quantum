"""Validate exported circuit construction and measure the matched 21-case score."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import json
from datetime import datetime,timezone
import hashlib
import time
import cirq
import cirq_google
import numpy as np
from judges_audit.compare import ROOT,calibration_maps,load,check,official
from .qvm import probabilities,model,stats,penalties
from .frames import with_frames


def main():
    module=load(ROOT/'record_submission.py','tested_record_submission')
    previous=load(ROOT/'recommended_submission.py','prior_recommended_submission')
    manifest=json.loads((ROOT/'optimization_lab/export_manifest.json').read_text())
    destination=ROOT/'optimization_lab/runs'/('record-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True)
    engine=cirq_google.engine.create_default_noisy_quantum_virtual_machine('willow_pink',simulator_class=cirq.DensityMatrixSimulator,
        dtype=np.complex128,seed=66129)
    device=engine.get_processor('willow_pink').get_device()
    maps=calibration_maps(engine.get_processor('willow_pink').get_current_calibration())
    scorer=official();changed={}
    data=dict(manifest=manifest,changed_cases=[],p2_smoke=[],comparison=[],physical_hardware=False)
    for spec in manifest['specs']:
        n,k=spec['n'],spec['k'];start=time.perf_counter()
        circuit,outputs=module.create_dicke_state(n,k,*maps)
        seconds=time.perf_counter()-start
        baseline,base_outputs=previous.create_dicke_state(n,k,*maps)
        expected=with_frames(baseline,spec['frames'])
        assert circuit==expected and outputs==base_outputs,'Guard unexpectedly failed on measured calibration'
        verify=check(circuit,outputs,n,k,maps,seconds,device)
        p=probabilities(circuit,outputs,model(),dtype=np.complex128)
        measured={}
        for basis in ('z','x'):
            sampling=cirq.Circuit(circuit)
            if basis=='x':
                gate=cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H))
                sampling+=cirq.Circuit(gate(q) for q in sorted(outputs))
            sampling+=cirq.Circuit(cirq.measure(*sorted(outputs),key='m'))
            measured[basis]=engine.get_sampler('willow_pink').run(sampling,repetitions=10000).measurements['m']
        scores={str(K):float(scorer['overall_score'](measured['z'],measured['x'],k,K)) for K in sorted({n,min(4,n)})}
        record=dict(n=n,k=k,construction_seconds=seconds,moments=len(circuit),cz=sum(op.gate==cirq.CZ for op in circuit.all_operations()),
            verification=verify,actual_qvm_10000_shot_scores=scores,penalties=penalties(p,n,k))
        data['changed_cases'].append(record);changed[n,k]=p
        cirq.to_json(circuit,destination/f'{n}-{k}.cirq.json')
        np.savez_compressed(destination/f'{n}-{k}.npz',PZ=p['z'],PX=p['x'],z_bits=measured['z'],x_bits=measured['x'])
        print('EXPORTED QVM PASS',record,flush=True)
    for n,k in [(2,1),(4,2),(6,3),(8,1)]:
        a,qa=module.create_dicke_state(n,k,*maps);b,qb=previous.create_dicke_state(n,k,*maps)
        assert a==b and qa==qb
    for n,k in [(7,3),(8,4),(20,10)]:
        a,qa=module.create_dicke_state_adaptive(n,k);b,qb=previous.create_dicke_state_adaptive(n,k)
        assert a==b and qa==qb
        data['p2_smoke'].append(dict(n=n,k=k,moments=len(a),unchanged=True))
    totals={choice:dict(baseline=0.,record=0.) for choice in ('N','4')}
    for label in ('will','will_large'):
        previous_rows=json.loads((ROOT/'judges_audit/results'/f'{label}.json').read_text())['rows']
        for row in previous_rows:
            n,k=row['n'],row['k']
            old=np.load(ROOT/f'judges_audit/results/{label}-{n}-{k}.npz')
            old_p=dict(z=old['PZ'],x=old['PX'])
            fresh=changed.get((n,k),old_p)
            b=stats(old_p,n,k,repetitions=4000,seed=43210)
            r=stats(fresh,n,k,repetitions=4000,seed=43210)
            # stats() uses the same documented seed rule as judges_audit.rescore.
            assert np.isclose(b[str(n)]['median'],row['shot_statistics'][str(n)]['median'],atol=1e-10)
            data['comparison'].append(dict(n=n,k=k,baseline=b,record=r,changed=(n,k) in changed))
            for choice in totals:
                key=str(n if choice=='N' else min(4,n))
                totals[choice]['baseline']+=b[key]['median'];totals[choice]['record']+=r[key]['median']
    data['totals']=totals
    data['relative_gain']={choice:totals[choice]['record']/totals[choice]['baseline']-1 for choice in totals}
    data['source_sha256']=hashlib.sha256((ROOT/'record_submission.py').read_bytes()).hexdigest()
    data['p2_score_unchanged']=82.3230363949385
    data['complete']=True
    (destination/'results.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    print('MATCHED RECORD',totals,data['relative_gain'],destination,flush=True)


if __name__=='__main__':main()
