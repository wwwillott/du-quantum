"""<=30 s total optimization: coherent four-qubit star leaf feasibility.

One common schedule must map every unary weight to its Dicke state, with a
common global phase. This tests a necessary local hybrid requirement;
it does not implement a full recursive circuit. Budgets are 12 s/family.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import cirq
import networkx as nx
import numpy as np
from scipy.optimize import minimize
from dicke_willow.scs import scs_unitary_ops
from dicke_willow.topology import willow105_graph
from dicke_willow.verify import ideal_dicke_vector
from willow_dicke.circuit.lower import bridge_along_path
from interference_dicke.core import Sector, schedule_for
from interference_dicke.hardware import compile_validate, metrics


def overlaps_gradients(sectors, schedule, parameters):
    overlaps, derivatives = [], []
    for sector in sectors:
        states = [sector.initial]
        for gate, angle in zip(schedule, parameters):
            states.append(sector.apply(states[-1], gate, angle))
        overlaps.append(np.vdot(sector.target, states[-1]))
        adjoint = sector.target.copy()
        derivative = np.empty(len(schedule), complex)
        for k in reversed(range(len(schedule))):
            derivative[k] = np.vdot(adjoint, sector.apply(states[k], schedule[k], parameters[k], derivative=True))
            adjoint = sector.apply(adjoint, schedule[k], -parameters[k])
        derivatives.append(derivative)
    return np.array(overlaps), np.array(derivatives)


def objective(sectors, schedule, parameters):
    overlaps, derivatives = overlaps_gradients(sectors, schedule, parameters)
    # Vacuum has unit-modulus overlap and fixes the allowed common phase.
    loss = 1 - np.mean(np.real(overlaps[1:] * np.conj(overlaps[0])))
    gradient = -np.mean(np.real(derivatives[1:] * np.conj(overlaps[0]) +
                               overlaps[1:, None] * np.conj(derivatives[0])), axis=0)
    return float(loss), gradient


def verify(circuit, qubits):
    overlaps = []
    simulator = cirq.Simulator(dtype=np.complex128)
    for l in range(5):
        # Site order input |1^l 0^(4-l)>; Cirq big endian.
        index = ((1 << l) - 1) << (4 - l)
        state = simulator.simulate(circuit, qubit_order=qubits, initial_state=index).final_state_vector
        overlaps.append(np.vdot(ideal_dicke_vector(4, l), state))
    overlap = np.array(overlaps)
    return {"fidelities": (abs(overlap) ** 2).tolist(),
            "relative_phases": np.angle(overlap * np.conj(overlap[0])).tolist(),
            "common_phase_loss": float(1 - np.mean(np.real(overlap[1:] * np.conj(overlap[0]))))}


def main():
    root = Path(__file__).parent
    geometry_path = sorted((root / "artifacts").glob("geometry-*/results.json"))[-1]
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    star = next(r for r in geometry["records"] if r["geometry"] == "star")
    qubits = [cirq.GridQubit(*xy) for xy in star["qubits"]]
    edges = [tuple(e) for e in star["edges"]]
    sectors = [Sector(4, l, edges) for l in range(5)]
    output = root / "artifacts" / ("multisector-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    for name in ["core.py", "multisector.py", "hardware.py"]:
        shutil.copyfile(root / name, output / name)
    reference = json.loads(sorted((root / "artifacts").glob("pruned-*/results.json"))[-1].read_text(encoding="utf-8"))
    data = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "environment": reference["environment"],
            "git_commit": reference["git_commit"], "topology": {"qubits": star["qubits"], "edges": star["edges"]},
            "source": str(geometry_path.resolve()), "records": [],
            "question": "Does one star schedule prepare all five unary weights with a common phase?"}
    for cp in [0, 1]:
        schedule = schedule_for(sectors[2], 12, cp)
        rng = np.random.default_rng(42)
        start = perf_counter()
        best = {"loss": 2., "parameters": None}
        trials, evaluations = [], 0

        class Expired(Exception):
            pass

        def fun(x):
            nonlocal evaluations
            if perf_counter() - start > 12:
                raise Expired
            loss, grad = objective(sectors, schedule, x)
            evaluations += 1
            if loss < best["loss"]:
                best.update(loss=loss, parameters=x.tolist())
            return loss, grad

        for _ in range(4):
            try:
                result = minimize(fun, rng.uniform(-np.pi, np.pi, len(schedule)), jac=True,
                                  method="L-BFGS-B", options={"maxiter": 350, "maxfun": 700, "gtol": 1e-8, "ftol": 1e-13})
                trials.append({"iterations": int(result.nit), "evaluations": int(result.nfev), "message": str(result.message)})
            except Expired:
                trials.append({"message": "12-second budget exhausted"})
                break
            if best["loss"] < 1e-11:
                break
        from interference_dicke.hardware import circuit_for
        # Drop initialization: this candidate is a unitary block for unary inputs.
        vacuum = sectors[0]
        logical = circuit_for(vacuum, schedule, best["parameters"], qubits)
        native = compile_validate(logical, qubits)
        measured = verify(native, qubits)
        if abs(measured["common_phase_loss"] - best["loss"]) > 1e-9:
            raise AssertionError("multi-sector independent verification failed")
        record = {"cp_count": cp, "matching_layers": 12, "schedule": schedule,
                  "parameters": best["parameters"], "common_phase_loss": best["loss"],
                  "verification": measured, "compiled_CZ": metrics(native), "device_validated": True,
                  "seconds": perf_counter() - start, "evaluations": evaluations, "trials": trials,
                  "seed": 42, "budget": {"starts": 4, "maxiter": 350, "maxfun": 700, "seconds_per_family": 12}}
        data["records"].append(record)
        cirq.to_json(native, output / f"star-all-weights-cp{cp}.cirq.json")
        (output / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"CP={cp}, loss={best['loss']:.8g}, F={measured['fidelities']}, CZ={record['compiled_CZ']['two_qubit_gates']}", flush=True)
    # Route the existing full SCS unitary on the SAME star, not just a path.
    baseline = cirq.Circuit(scs_unitary_ops(qubits[::-1], 4))
    graph = nx.Graph(); graph.add_edges_from((qubits[i], qubits[j]) for i, j in edges)
    routed = cirq.Circuit()
    for op in baseline.all_operations():
        if len(op.qubits) == 1:
            routed.append(op)
        else:
            path = nx.shortest_path(graph, *op.qubits)
            routed += bridge_along_path(cirq.Circuit(op), path)
    native = compile_validate(routed, qubits)
    data["baseline"] = {"method": "SCS full unary unitary, same star, move-and-restore routing",
                        "compiled_CZ": metrics(native), "verification": verify(native, qubits)}
    cirq.to_json(native, output / "SCS-all-weights.cirq.json")
    (output / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Baseline: {data['baseline']}", flush=True)


if __name__ == "__main__":
    main()
