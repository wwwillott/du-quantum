"""A finite, exact Willow translation of one Yu measurement round (N=4).

The paper's unbounded retry loop cannot be assigned a finite len(circuit).
We finish every failed outcome with an exact coherent weight-to-target map.
This deliberately counts QFT, routing, readout and ALL conditioned branches.
It is a comparison candidate, not a claim of paper-level depth on Willow.
"""
from math import asin, sqrt
import cirq
import cirq_google
import networkx as nx
import sympy
from dicke_willow.compile import finalize
from dicke_willow.scs import scs_unitary_ops
from willow_dicke.circuit.lower import bridge_along_path


def build(k=2):
    data = [cirq.GridQubit(0, 6), cirq.GridQubit(0, 7),
            cirq.GridQubit(0, 8), cirq.GridQubit(1, 8)]
    ancillas = [cirq.GridQubit(2, 8), cirq.GridQubit(3, 7), cirq.GridQubit(4, 6)]
    graph = nx.Graph()
    graph.add_edges_from(cirq_google.Willow105.metadata.qubit_pairs)
    def compile_route(logical):
        native = finalize(logical, decouple=False)
        ops = []
        for op in native.all_operations():
            if len(op.qubits) == 2:
                ops.extend(bridge_along_path(cirq.Circuit(op), nx.shortest_path(graph, *op.qubits)).all_operations())
            else:
                ops.append(op)
        return finalize(cirq.Circuit(ops), decouple=False)

    logical = cirq.Circuit(cirq.ry(2*asin(sqrt(k/4))).on_each(*data))
    logical.append(cirq.H.on_each(*ancillas))
    # Coherently write weight into the Fourier register, without learning
    # which data strings have that weight. Big-endian QFT convention.
    for j, ancilla in enumerate(ancillas):
        logical.append(cirq.CZPowGate(exponent=2**(-j)).on(ancilla, q) for q in data)
    logical += cirq.Circuit(cirq.decompose((cirq.qft(*ancillas))**-1))
    circuit = compile_route(logical)
    lines = {q.row+q.col for q in ancillas}
    measured = sorted(q for q in graph if q.row+q.col in lines)
    keys = [f'weight_{j}' for j in range(3)]
    key_for = dict(zip(ancillas, keys))
    circuit.append(cirq.Moment(cirq.measure(q, key=key_for.get(q, f'fill_{q.row}_{q.col}')) for q in measured))
    decoders = {w: compile_route(cirq.Circuit(scs_unitary_ops(data, w))) for w in range(5)}
    # D_w --U_w^-1--> unary(w) --X--> unary(k) --U_k--> D_k.
    # No branch is discarded and the success branch has no further quantum gates.
    for w in range(5):
        if w == k:
            continue
        condition = cirq.SympyCondition(sympy.And(*[
            sympy.Eq(sympy.Symbol(key), (w >> (2-j)) & 1) for j, key in enumerate(keys)]))
        correction = cirq.Circuit(cirq.inverse(decoders[w]))
        correction.append(cirq.X(data[i]) for i in range(4) if (i >= 4-w) != (i >= 4-k))
        correction += decoders[k]
        correction = finalize(correction, decouple=False)
        # Moment boundaries explicitly retain measurement -> control causality.
        circuit += cirq.Circuit(cirq.Moment(op.with_classical_controls(condition) for op in moment) for moment in correction)
    return circuit, set(data)
