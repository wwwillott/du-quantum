"""Yu et al., arXiv:2411.03428v1: collective rotation + QND weight readout.

Finite challenge adaptation: one/two rounds followed by an exact conditioned
fallback. The original repeat-until-success algorithm has unbounded worst-case
depth and cannot be scored as a finite exact circuit without this adaptation.
Global weight is measured with a Fourier counter, NOT individual data readout.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cirq
import cirq_google
import networkx as nx
import numpy as np
from scipy.linalg import expm
import sympy

from adaptive_dicke_validation.audit import ROOT, load_source, structure, verify_all_branches
from willow_dicke.circuit.lower import bridge_along_path


def geometric_angle(n, target, current):
    if current == target:
        return 0.
    if current == 0:
        # Exact optimum for the first product-state rotation; a declared
        # small-n angle variant of the paper's approximate geometric choice.
        return 2 * math.asin(math.sqrt(target / n))
    j = n / 2
    m, mt = j - current, j - target
    if mt == 0:
        return math.asin(np.clip(m / j, -1, 1))
    r0sq = j * (j + 1)
    rm, rt = math.sqrt(r0sq - m*m), math.sqrt(r0sq - mt*mt)
    return math.asin(np.clip((m * rt - mt * rm) / r0sq, -1, 1))


def markov(n, k, rounds=8):
    generator = np.zeros((n + 1, n + 1))
    for w in range(n):
        v = .5 * np.sqrt((n - w) * (w + 1))
        generator[w + 1, w], generator[w, w + 1] = v, -v
    effective = [0 if n == 2*k and abs(n/2 - w) > math.sqrt(n/2) and w != k else w
                 for w in range(n + 1)]
    transition = np.column_stack([expm(geometric_angle(n, k, source) * generator)[:, source] ** 2
                                  for source in effective])
    distribution = np.eye(n + 1)[:, 0]
    success = []
    for _ in range(rounds):
        distribution = transition @ distribution
        success.append(float(distribution[k]))
    transient = [w for w in range(n + 1) if w != k]
    times = np.linalg.solve(np.eye(n) - transition[np.ix_(transient, transient)].T, np.ones(n))
    return {"n": n, "k": k, "transition_matrix": transition.tolist(),
            "success_by_round": success, "expected_rounds": float(times[transient.index(0)]),
            "angle_policy": "paper Eq.2; balanced asin(m/j) and Algorithm-1 reset threshold; exact product-state first angle"}


def patch_layout(n):
    device = cirq_google.Willow105
    vertices = set(device.metadata.qubit_set)
    side = 2 if n == 2 else 3
    for r in range(10):
        for c in range(10):
            patch = {cirq.GridQubit(r + i, c + j) for i in range(side) for j in range(side)}
            if not patch <= vertices:
                continue
            parity = (r + c) % 2
            data = sorted(q for q in patch if (q.row + q.col) % 2 == parity)[:n]
            ancillas = sorted(q for q in patch if (q.row + q.col) % 2 != parity)[:math.ceil(math.log2(n + 1))]
            if len(data) == n and len(ancillas) == math.ceil(math.log2(n + 1)):
                graph = nx.Graph()
                graph.add_nodes_from(patch)
                graph.add_edges_from(tuple(pair) for pair in device.metadata.qubit_pairs if set(pair) <= patch)
                if nx.is_connected(graph):
                    return data, ancillas, graph
    raise ValueError("no suitable physical patch")


def native_route(circuit, graph):
    target = cirq.CZTargetGateset(allow_partial_czs=False)
    native = cirq.optimize_for_target_gateset(circuit, gateset=target)
    routed = cirq.Circuit()
    for op in native.all_operations():
        if len(op.qubits) == 2:
            routed += bridge_along_path(cirq.Circuit(op), nx.shortest_path(graph, *op.qubits))
        else:
            routed.append(op)
    return cirq.optimize_for_target_gateset(routed, gateset=target)


def weight_counter(data, ancillas):
    m = len(ancillas)
    circuit = cirq.Circuit(cirq.H(q) for q in ancillas)
    for b, q in enumerate(ancillas):
        circuit.append(cirq.CZPowGate(exponent=2 ** (b + 1) / 2**m).on(d, q) for d in data)
    circuit.append(cirq.qft(*ancillas[::-1], inverse=True))
    return circuit


def build(n, k, rounds, baseline_module):
    data, ancillas, graph = patch_layout(n)
    circuit = cirq.Circuit()
    counter = native_route(weight_counter(data, ancillas), graph)
    device = set(cirq_google.Willow105.metadata.qubit_set)
    previous = None
    for iteration in range(rounds):
        if iteration:
            circuit.append(cirq.Moment(cirq.reset(q) for q in ancillas))
            for current in range(n + 1):
                condition = cirq.SympyCondition(sympy.Eq(sympy.Symbol(previous), current))
                source = current
                if n == 2*k and current != k and abs(n/2 - current) > math.sqrt(n/2):
                    circuit.append(cirq.Moment(cirq.reset(q).with_classical_controls(condition) for q in data))
                    source = 0
                angle = geometric_angle(n, k, source)
                if abs(angle) < 1e-14:
                    continue
                circuit.append(cirq.Moment(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.ry(angle))).on(q)
                                          .with_classical_controls(condition) for q in data))
        else:
            angle = geometric_angle(n, k, 0)
            circuit.append(cirq.Moment(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.ry(angle))).on(q) for q in data))
        circuit += counter
        key = f"weight_{iteration}"
        diagonal_ids = {q.row + q.col for q in ancillas}
        spectators = sorted(q for q in device if q.row + q.col in diagonal_ids and q not in ancillas)
        assert not set(data) & set(spectators)
        circuit.append(cirq.Moment([cirq.measure(*ancillas[::-1], key=key)] +
                                  [cirq.measure(q, key=f"idle_{iteration}_{q.row}_{q.col}") for q in spectators]))
        previous = key
    # Unlike postselection, these failures are retained and corrected exactly.
    failure = cirq.SympyCondition(sympy.Ne(sympy.Symbol(previous), k))
    circuit.append(cirq.Moment(cirq.reset(q).with_classical_controls(failure) for q in data))
    baseline, outputs = baseline_module.create_dicke_state_adaptive(n, k)
    if baseline.has_measurements() or baseline.all_qubits() != set(outputs):
        raise ValueError("prototype fallback requires a small unitary baseline")
    mapping = dict(zip(sorted(outputs), data))
    fallback = native_route(baseline.transform_qubits(mapping), graph)
    for moment in fallback:
        circuit.append(cirq.Moment(op.with_classical_controls(failure) for op in moment))
    return circuit, set(data), {"counter_depth": len(counter), "fallback_depth": len(fallback),
                                "rounds": rounds, "ancillas": [[q.row, q.col] for q in ancillas],
                                "data": [[q.row, q.col] for q in data],
                                "algorithm": "bounded Yu collective-readout protocol with deterministic fallback"}


def main():
    source = ROOT / "snapshots" / "will-605b9a8" / "dicke_submission.py"
    module = load_source(source)
    output = ROOT / "artifacts" / ("yu-prototype-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=True)
    results = {"paper": "https://arxiv.org/html/2411.03428v1", "records": [],
               "markov": [markov(n, n // 2) for n in [2, 4, 6, 10, 15, 20]],
               "scoring": "worst-case static circuit moments, including all fallback and conditional gates"}
    for n, k in [(2, 1), (4, 1), (4, 2)]:
        for rounds in [1, 2]:
            start = perf_counter()
            circuit, outputs, info = build(n, k, rounds, module)
            verification = verify_all_branches(circuit, outputs, k, max_qubits=12)
            violations = structure(circuit, outputs, n)
            baseline, _ = module.create_dicke_state_adaptive(n, k)
            record = {"n": n, "k": k, **info, "depth": len(circuit), "baseline_depth": len(baseline),
                      "score_term": n*k / len(circuit), "baseline_score_term": n*k / len(baseline),
                      "verification": verification, "violations": violations, "seconds": perf_counter() - start,
                      "promote": not violations and verification["status"] == "verified" and len(circuit) < len(baseline)}
            results["records"].append(record)
            cirq.to_json(circuit, output / f"{n}-{k}-rounds{rounds}.cirq.json")
            (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Yu D({n},{k}) rounds={rounds} depth={len(circuit)} baseline={len(baseline)} "
                  f"verification={verification} violations={violations}", flush=True)


if __name__ == "__main__":
    main()
