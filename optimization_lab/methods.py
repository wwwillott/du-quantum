"""Exact transformations and judge-aware calibration ranking."""
import functools
import inspect
import math
from pathlib import Path
import time

import cirq
import cirq_google
import numpy as np
from judges_audit.compare import ROOT, load


@functools.lru_cache(maxsize=1)
def will():
    return load(ROOT/'judges_audit/sources/will_8af9eb4.py','lab_will')


def fuse(circuit):
    c = cirq.merge_single_qubit_gates_to_phxz(circuit,atol=1e-13)
    c = cirq.drop_negligible_operations(c,atol=1e-13)
    return cirq.Circuit(c.all_operations())


def global_x(circuit,outputs):
    return fuse(cirq.Circuit(circuit)+cirq.Circuit(cirq.PhasedXZGate(x_exponent=1,z_exponent=0,axis_phase_exponent=0).on(q) for q in outputs))


@functools.lru_cache(maxsize=1)
def _corrected_function():
    module = will()
    source=inspect.getsource(module._robust_calibration_cost)
    assert source.count('(shots * f_z / M)')==1
    assert source.count('        ell = 2 * m\n')==1
    source=source.replace('def _robust_calibration_cost(', 'def corrected(')
    source=source.replace('(shots * f_z / M)','(shots / M)')
    source=source.replace('        ell = 2 * m\n','        ell = 2 * m\n        if not 0 <= k-ell//2 <= n-ell:\n            continue\n')
    namespace=dict(module.__dict__)
    exec(compile(source,'corrected_calibration_surrogate','exec'),namespace)
    return namespace['corrected']


def corrected_cost(circuit,outputs,k,*maps):
    """Fix two established errors; other terms remain screening heuristics."""
    return float(_corrected_function()(circuit,outputs,k,*maps))


def placements(circuit,outputs,maps):
    """All grid rotations/reflections/translations that fit the supplied maps."""
    cz,sq,zero,one=maps
    available=set(sq)&set(zero)&set(one)
    edges={frozenset(edge) for edge in cz}
    original=sorted(circuit.all_qubits())
    seen=set()
    for swap in (False,True):
        for sx,sy in ((1,1),(1,-1),(-1,1),(-1,-1)):
            xy=[(sx*(q.col if swap else q.row),sy*(q.row if swap else q.col)) for q in original]
            for anchor in sorted(available):
                dr,dc=anchor.row-xy[0][0],anchor.col-xy[0][1]
                moved=tuple(cirq.GridQubit(r+dr,c+dc) for r,c in xy)
                if moved in seen or not set(moved)<=available:
                    continue
                mapping=dict(zip(original,moved))
                if any(frozenset(mapping[q] for q in op.qubits) not in edges for op in circuit.all_operations() if len(op.qubits)==2):
                    continue
                seen.add(moved)
                yield circuit.transform_qubits(mapping),{mapping[q] for q in outputs},dict(swap=swap,sx=sx,sy=sy,dr=dr,dc=dc)


def variants(circuit,outputs,k):
    yield 'unchanged',cirq.Circuit(circuit)
    merged=fuse(circuit)
    if merged!=circuit:
        yield 'fused',merged
    if len(outputs)==2*k:
        yield 'terminal_global_x',global_x(circuit,outputs)


def candidate_pool(circuit,outputs,k,maps,shortlist=12):
    """Use cheap ranking before expensive small-system proxy evaluations."""
    ranked=[]
    for name,base in variants(circuit,outputs,k):
        for placed,qs,placement in placements(base,outputs,maps):
            score=corrected_cost(placed,qs,k,*maps)
            ranked.append((score,name,placed,qs,placement))
    ranked.sort(key=lambda r:r[0])
    # Diversity by physical support and transformation. Retain untransformed
    # candidates so a placement gain is not confused with pulse simplification.
    selected=[]; seen=set()
    for cost,name,c,qs,placement in ranked:
        key=(name,frozenset(qs))
        if key in seen:
            continue
        seen.add(key); selected.append((cost,name,c,qs,placement))
        if len(selected)>=shortlist:
            break
    for name,c in variants(circuit,outputs,k):
        selected.append((corrected_cost(c,outputs,k,*maps),name,c,set(outputs),dict(original_placement=True)))
    return selected,len(ranked)
