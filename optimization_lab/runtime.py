"""Small, portable exact-frame adapter; embedded verbatim in record exports."""
import hashlib as _record_hashlib
import json as _record_json
import numpy as _record_np
import cirq as _record_cirq


def _record_signature(circuit):
    wires={};description=[]
    for moment_index,moment in enumerate(circuit):
        for op in moment:
            indices=[]
            for q in op.qubits:
                if q not in wires:wires[q]=len(wires)
                indices.append(wires[q])
            if op.gate==_record_cirq.CZ:
                gate='CZ'
            elif isinstance(op.gate,_record_cirq.PhasedXZGate):
                u=_record_cirq.unitary(op)
                pivot=u.ravel()[int(_record_np.argmax(abs(u)))]
                u=u/(pivot/abs(pivot))
                values=_record_np.concatenate((u.real.ravel(),u.imag.ravel()))
                gate=[0. if abs(float(v))<1e-11 else round(float(v),10) for v in values]
            else:return None
            description.append((moment_index,indices,gate))
    return _record_hashlib.sha256(_record_json.dumps(description,separators=(',',':')).encode()).hexdigest()


def _record_apply(circuit,frames):
    paulis=[_record_np.eye(2),_record_cirq.unitary(_record_cirq.X),_record_cirq.unitary(_record_cirq.Y),_record_cirq.unitary(_record_cirq.Z)]
    z=paulis[3];operations=[];index=0
    for op in circuit.all_operations():
        if op.gate!=_record_cirq.CZ:
            operations.append(op);continue
        a,b=op.qubits;ia,ib=frames[index];index+=1
        pa,pb=paulis[ia],paulis[ib]
        qa=pa@(z if ib in (1,2) else paulis[0]);qb=pb@(z if ia in (1,2) else paulis[0])
        operations.extend([_record_cirq.PhasedXZGate.from_matrix(pa)(a),_record_cirq.PhasedXZGate.from_matrix(pb)(b),op,
            _record_cirq.PhasedXZGate.from_matrix(qa)(a),_record_cirq.PhasedXZGate.from_matrix(qb)(b)])
    assert index==len(frames)
    result=_record_cirq.merge_single_qubit_gates_to_phxz(_record_cirq.Circuit(operations),atol=1e-13)
    result=_record_cirq.drop_negligible_operations(result,atol=1e-13)
    return _record_cirq.Circuit(result.all_operations())
