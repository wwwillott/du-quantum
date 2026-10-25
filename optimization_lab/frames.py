"""H5: two-sided CZ Pauli frames with exact ideal-unitary preservation."""
import cirq
import numpy as np
from .methods import fuse


def twirl(circuit,seed):
    rng=np.random.default_rng(seed)
    count=sum(op.gate==cirq.CZ for op in circuit.all_operations())
    return with_frames(circuit,rng.integers(0,4,size=(count,2)))


def with_frames(circuit,frames):
    frames=list(frames)
    assert len(frames)==sum(op.gate==cirq.CZ for op in circuit.all_operations())
    paulis=[np.eye(2),cirq.unitary(cirq.X),cirq.unitary(cirq.Y),cirq.unitary(cirq.Z)]
    z=paulis[3]
    operations=[]
    index=0
    for op in circuit.all_operations():
        if op.gate!=cirq.CZ:
            operations.append(op)
            continue
        a,b=op.qubits
        ia,ib=map(int,frames[index]);index+=1
        assert 0<=ia<4 and 0<=ib<4
        pa,pb=paulis[ia],paulis[ib]
        # Conjugating X/Y through CZ contributes Z on the opposite wire.
        qa=pa@(z if ib in (1,2) else paulis[0])
        qb=pb@(z if ia in (1,2) else paulis[0])
        operations.extend([cirq.PhasedXZGate.from_matrix(pa)(a),cirq.PhasedXZGate.from_matrix(pb)(b),op,
                           cirq.PhasedXZGate.from_matrix(qa)(a),cirq.PhasedXZGate.from_matrix(qb)(b)])
    return fuse(cirq.Circuit(operations))
