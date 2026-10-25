"""Coherent interference leaves inside a full eight-qubit preparation.

Both leaves use all five certified columns, with one common phase. No
state-specific pruning is applied to a coherent leaf.
"""
from pathlib import Path
import cirq
import networkx as nx

from dicke_willow.compile import finalize
from dicke_willow.scs import scs_unitary_ops
from dicke_willow.weight_distribution import weight_distribution_ops
from willow_dicke.circuit.lower import bridge_along_path
from .verify import unitary_fidelity, validate_structure


def experiment():
    source = Path(__file__).parents[1] / 'artifacts' / 'multisector-20261025T080630810392Z'
    leaf = cirq.read_json(source / 'star-all-weights-cp1.cirq.json')
    a = [cirq.GridQubit(0, 7), cirq.GridQubit(0, 6), cirq.GridQubit(0, 8), cirq.GridQubit(1, 7)]
    b = [cirq.GridQubit(q.row+2, q.col) for q in a]
    outputs = set(a+b)
    graph = nx.Graph()
    graph.add_nodes_from(outputs)
    graph.add_edges_from((x, y) for x in outputs for y in outputs if x.is_adjacent(y))
    # The repository's distributor uses right-aligned unary registers; the
    # interference leaf consumes left-aligned unary registers.
    prefix = cirq.Circuit(cirq.X.on_each(*b))
    prefix += cirq.Circuit(weight_distribution_ops(a[::-1], b[::-1], n=8, m=4, k=4))
    native = finalize(prefix, decouple=False)
    routed = cirq.Circuit()
    for op in native.all_operations():
        routed += (bridge_along_path(cirq.Circuit(op), nx.shortest_path(graph, *op.qubits))
                   if len(op.qubits) == 2 else cirq.Circuit(op))
    prefix = finalize(routed, decouple=False)
    candidates, records = {}, {}
    for name in ('interference_leaf', 'scs_leaf'):
        suffix = cirq.Circuit()
        for sites in (a, b):
            if name == 'interference_leaf':
                local = leaf.transform_qubits(dict(zip(a, sites)))
            else:
                raw = finalize(cirq.Circuit(scs_unitary_ops(sites[::-1], 4)), decouple=False)
                local = cirq.Circuit()
                for op in raw.all_operations():
                    local += (bridge_along_path(cirq.Circuit(op), nx.shortest_path(graph.subgraph(sites), *op.qubits))
                              if len(op.qubits) == 2 else cirq.Circuit(op))
                local = finalize(local, decouple=False)
            suffix.append(local.all_operations())
        circuit = finalize(prefix + suffix, decouple=False)
        result = unitary_fidelity(circuit, outputs, 4)
        assert result['fidelity'] > 1-1e-9
        records[name] = dict(**result, **validate_structure(circuit, outputs, 8),
                             moments=len(circuit), score=32/len(circuit),
                             cz=sum(op.gate == cirq.CZ for op in circuit.all_operations()))
        candidates[name] = (circuit, outputs)
    return candidates, records
