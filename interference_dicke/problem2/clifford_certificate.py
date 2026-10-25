"""All-branch Choi certificate for the large cat/teleportation suffix.

For a fixed Clifford circuit with Pauli measurement and linear-parity Pauli
corrections, tableau supports are outcome independent and signs are affine in
the independent measurement bits. The zero outcome and each basis vector
therefore certify every outcome, including phases on arbitrary core inputs.
"""
import cirq
import numpy as np
from . import portfolio
from .verify import unitary_fidelity


class Bits(np.random.RandomState):
    def __init__(self, one=None):
        super().__init__(0)
        self.one, self.calls = one, 0

    def randint(self, low, high=None, size=None, dtype=int):
        assert low == 2 and high is None and size is None
        result = int(self.calls == self.one)
        self.calls += 1
        return result


def certify(n):
    d = portfolio.balanced_candidate(n)
    core_outputs = sorted(d['core_outputs'])
    refs = [cirq.NamedQubit(f'ref_{i}') for i in range(n-1)]
    prep = cirq.Circuit()
    for old, ref in zip(core_outputs, refs):
        prep.append([cirq.H(old), cirq.CNOT(old, ref)])
    # Independent arbitrary input (its Choi state) replaces the non-Clifford
    # core. The resource and the entire suffix remain the actual native gates.
    prep += d['cat']
    prep += d['suffix']
    # Idle readout filler measurements may have been packed into early
    # moments, so extract by operation kind rather than a moment offset.
    original = list(d['unscheduled_circuit'].all_operations())
    prep.append(cirq.Moment(op for op in original if cirq.is_measurement(op)))
    prep.append(cirq.Moment(op for op in original if op.classical_controls))
    # Invert the expected isometry: (|1>|psi> + |0>X^all|psi>)/sqrt(2).
    root = d['root']
    prep.append(cirq.X(root))
    prep.append(cirq.CNOT(root, dest) for dest in d['destinations'].values())
    prep.append(cirq.H(root))
    for old, ref in zip(core_outputs, refs):
        dest = d['destinations'][old]
        prep.append([cirq.CNOT(dest, ref), cirq.H(dest)])
    checked = [root] + list(d['destinations'].values()) + refs
    prep.append(cirq.measure(*checked, key='choi_zero'))
    order = sorted(prep.all_qubits())

    def run(one=None):
        rng = Bits(one)
        data = cirq.ClassicalDataDictionaryStore()
        args = cirq.StabilizerSimulationState(state=cirq.CliffordTableau(len(order)),
                                              qubits=order, prng=rng, classical_data=data)
        for op in prep.all_operations():
            cirq.act_on(op, args)
        assert args.classical_data.get_int(cirq.MeasurementKey('choi_zero')) == 0
        return rng.calls

    bits = run()
    for i in range(bits):
        assert run(i) == bits
    core = unitary_fidelity(d['core'], d['core_outputs'], n//2-1)
    assert core['fidelity'] > 1-1e-9
    assert core['ancilla_zero_probability'] > 1-1e-9
    return dict(fidelity=core['fidelity'], core=core, independent_bits=bits,
                branches_certified=2**bits, choi_basis_checks=bits+1,
                certificate='Clifford Choi isometry; affine Pauli measurement signs',
                scheduling='shared-wire dependencies and causal measurement keys preserved')
