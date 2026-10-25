import cirq
import cirq_google
import numpy as np
import pytest
from optimization_lab.frames import twirl
from optimization_lab.methods import global_x,corrected_cost,will
from optimization_lab.qvm import properties_fixed_stochastic
from judges_audit.compare import ROOT
from interference_dicke.problem2.verify import unitary_fidelity


@pytest.mark.parametrize('seed',[0,1,3,9])
def test_cz_frames_preserve_full_unitary(seed):
    a,b=cirq.LineQubit.range(2)
    c=cirq.Circuit(cirq.H(a),cirq.ry(.71)(b),cirq.CZ(a,b),cirq.rx(.26)(a),cirq.CZ(a,b),cirq.rz(.4)(b))
    native=cirq.optimize_for_target_gateset(c,gateset=cirq.CZTargetGateset())
    changed=twirl(native,seed)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(native),cirq.unitary(changed),atol=1e-12)
    assert sum(op.gate==cirq.CZ for op in native.all_operations())==sum(op.gate==cirq.CZ for op in changed.all_operations())


def test_balanced_output_flip_preserves_target():
    c=cirq.read_json(ROOT/'judges_audit/results/will-4-2.cirq.json')
    qs=set(c.all_qubits())
    assert unitary_fidelity(global_x(c,qs),qs,2)['fidelity']>1-1e-12


def test_ablation_does_not_reallocate_error_budget():
    original=cirq_google.engine.load_device_noise_properties('willow_pink')
    changed=properties_fixed_stochastic(0.)
    assert changed._depolarizing_error==original._depolarizing_error
    assert all(np.allclose(cirq.unitary(g),np.eye(4),atol=1e-14) for g in changed.fsim_errors.values())
    assert changed.t1_ns==original.t1_ns and changed.readout_errors==original.readout_errors


def test_zero_target_relative_x_penalty_removed():
    qs=sorted(cirq_google.Willow105.metadata.qubit_set)[:6]
    sq={q:(.001,0.) for q in qs};zero={q:(.01,0.) for q in qs};one={q:(.02,0.) for q in qs}
    c=cirq.Circuit(cirq.PhasedXZGate(x_exponent=.5,z_exponent=0,axis_phase_exponent=0)(q) for q in qs)
    assert corrected_cost(c,set(qs),1,{},sq,zero,one)<will()._robust_calibration_cost(c,set(qs),1,{},sq,zero,one)


@pytest.mark.parametrize('n,k',[(4,2),(6,1)])
def test_expected_judge_denominator_matches_shot_monte_carlo(n,k):
    from optimization_lab.qvm import expected_denominator
    from judges_audit.rescore import score_batch
    rng=np.random.default_rng(345)
    pz=np.array([2.0 if i.bit_count()==k else .01 for i in range(1<<n)])
    pz*=rng.uniform(.98,1.02,len(pz));pz/=pz.sum()
    px=rng.uniform(.5,1.5,len(pz));px/=px.sum()
    z=rng.multinomial(10000,pz,size=6000)
    x=rng.multinomial(10000,px,size=6000)
    observed=1/score_batch(z,x,n,k,n)
    predicted=expected_denominator(dict(z=pz,x=px),n,k)
    assert abs(observed.mean()-predicted)<5*observed.std()/np.sqrt(len(observed))


def test_export_signature_tracks_gates_but_not_physical_coordinates():
    from optimization_lab.runtime import _record_signature
    a,b=cirq.GridQubit.rect(1,2)
    circuit=cirq.optimize_for_target_gateset(cirq.Circuit(cirq.ry(.71)(a),cirq.CZ(a,b)),gateset=cirq.CZTargetGateset())
    translated=circuit.transform_qubits(lambda q:cirq.GridQubit(q.row+3,q.col+4))
    assert _record_signature(circuit)==_record_signature(translated)
    changed=circuit+cirq.Circuit(cirq.PhasedXZGate(x_exponent=.25,z_exponent=0,axis_phase_exponent=0)(a))
    assert _record_signature(circuit)!=_record_signature(changed)
    assert _record_signature(cirq.Circuit(cirq.measure(a))) is None


def test_export_runtime_matches_frozen_experiment():
    import json
    from optimization_lab.frames import with_frames
    from optimization_lab.runtime import _record_apply
    result=json.loads((ROOT/'optimization_lab/runs/refine-20261025T095636098346Z/results.json').read_text())
    circuit=cirq.read_json(ROOT/'judges_audit/results/will-8-4.cirq.json')
    assert _record_apply(circuit,result['best_frames'])==with_frames(circuit,result['best_frames'])
