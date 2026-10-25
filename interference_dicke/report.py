"""Rebuild figures, evidence summary and hypothesis tree from saved runs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from interference_dicke.core import Sector

ROOT = Path(__file__).parent


def main():
    all_runs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted((ROOT / "artifacts").glob("*/results.json"))]
    small_path, small = next((p, d) for p, d in all_runs if any("noise" in r for r in d.get("experiments", [])))
    extended_path, extended = next((p, d) for p, d in all_runs if any(r["n"] == 6 for r in d.get("experiments", [])))
    geometry_path, geometry = sorted(((p, d) for p, d in all_runs if p.parent.name.startswith("geometry-")), key=lambda x: str(x[0]))[-1]
    pruned_path, pruned = sorted(((p, d) for p, d in all_runs if p.parent.name.startswith("pruned-")), key=lambda x: str(x[0]))[-1]
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.grid": True,
                         "axes.spines.top": False, "axes.spines.right": False})

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    records = pruned["records"]
    four = [r for r in records if r["n"] == 4 and r["id"].startswith("chain-")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for cp in [0, 1]:
        rows = [r for r in four if r["cp_count"] == cp]
        ax.plot([r["matching_layers"] for r in rows], [r["ideal_fidelity"] for r in rows], "o-", label=f"{cp} CP")
    ax.axhline((3 + 2 * np.sqrt(2)) / 6, color="gray", ls=":", label="Proven Slater ceiling")
    ax.set(xlabel="Scheduled matching layers (includes inactive initial layer)", ylabel="Ideal coherent fidelity", title="Dicke D₂⁴: exchange vs one interaction")
    ax.legend(); finish(fig, "01_ideal_vs_depth")

    fig, ax = plt.subplots(figsize=(7, 4))
    for cp in [0, 1]:
        rows = [r for r in four if r["cp_count"] == cp]
        ax.plot([r["compiled_CZ"]["depth"] for r in rows], [r["noise"]["fidelity"] for r in rows], "o-", label=f"{cp} CP")
    for r in records:
        if r["n"] == 4 and r["id"].startswith(("SCS", "recursive")):
            ax.scatter(r["compiled_CZ"]["depth"], r["noise"]["fidelity"], marker="s", label=r["id"])
    ax.set(xlabel="Compiled moment depth after common state-aware pruning", ylabel="Noisy coherent fidelity", title="D₂⁴ — synthetic noise, p₂=0.001")
    ax.legend(); finish(fig, "02_noisy_vs_depth")

    fig, ax = plt.subplots(figsize=(7, 4))
    for depth in [6, 10]:
        rows = [r for r in records if r["n"] == 6 and r.get("matching_layers") == depth]
        ax.plot([r["cp_count"] for r in rows], [r["ideal_fidelity"] for r in rows], "o-", label=f"{depth} matching layers")
    ax.set(xlabel="Conditional-phase gates", ylabel="Ideal fidelity", title="D₃⁶: marginal value of interactions", xticks=[0, 1, 2])
    ax.legend(); finish(fig, "03_fidelity_vs_interactions")

    fig, ax = plt.subplots(figsize=(7, 4))
    for cp in [0, 1]:
        rows = [r for r in four if r["cp_count"] == cp]
        ax.plot([r["compiled_CZ"]["two_qubit_gates"] for r in rows], [r["ideal_fidelity"] for r in rows], "o-", label=f"{cp} CP")
    ax.set(xlabel="Compiled CZ gates", ylabel="Ideal fidelity", title="D₂⁴: physical gate-set cost after pruning")
    ax.legend(); finish(fig, "04_fidelity_vs_CZ")

    fig, ax = plt.subplots(figsize=(7, 4))
    for n, rows in [(4, small["experiments"]), (6, extended["experiments"])]:
        vals = [r["optimization"]["seconds"] for r in rows if r["n"] == n and r["l"] > 1]
        ax.scatter([n] * len(vals), vals, alpha=.6)
        ax.scatter(n, np.median(vals), marker="_", s=300, color="black")
    ax.set(xlabel="Qubits", ylabel="Optimization seconds", title="Two measured sizes; no scaling fit", xticks=[4, 6])
    finish(fig, "05_optimization_time")

    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in ["square", "star"]:
        rows = [r for r in geometry["records"] if r["geometry"] == kind]
        ax.plot([r["matching_layers"] for r in rows], [r["compiled_fidelity"] for r in rows], "o-", label=kind)
    rows = [r for r in small["experiments"] if r["l"] == 2 and r["cp_count"] == 0]
    ax.plot([r["matching_layers"] for r in rows], [r["compiled_fidelity"] for r in rows], "o--", label="chain")
    ax.set(xlabel="Scheduled matching layers (different widths by geometry)", ylabel="Ideal fidelity", title="Geometry: XY+Z only, n=4,l=2")
    ax.legend(); finish(fig, "06_geometry")

    chosen = next(r for r in small["experiments"] if r["id"] == "chain-4-2-depth6-cp1")
    sector = Sector(4, 2, chosen["topology"]["edges"])
    schedule = [(kind, tuple(sites)) for kind, sites in chosen["schedule"]]
    parameters = chosen["optimization"]["parameters"]
    state = sector.evolve(schedule, parameters)
    state *= np.exp(-1j * np.angle(np.vdot(sector.target, state)))
    labels = [format(int(x), "04b")[::-1] for x in sector.masks]
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].bar(labels, abs(state), color="#267c9b")
    axes[0].axhline(1 / np.sqrt(6), ls="--", color="black", label="Target amplitude")
    axes[0].set(ylabel="Amplitude magnitude", title="Final D₂⁴ configuration structure (site order 0→3)")
    axes[0].legend()
    axes[1].bar(labels, abs(state) ** 2, color="#5b9e6f")
    axes[1].set(ylabel="Probability", xlabel="Configuration")
    finish(fig, "07_amplitudes_probabilities")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(labels, np.angle(state)); ax.set(ylabel="Relative phase (radians)", xlabel="Configuration", ylim=(-.01, .01), title="Global phase aligned to target overlap")
    finish(fig, "08_relative_phases")

    # Actual instantaneous currents during the largest-flow single XY gate.
    state = sector.initial.copy()
    best = None
    for gate, angle in zip(schedule, parameters):
        if gate[0] == "xy":
            mid = sector.apply(state, gate, angle / 2)
            a, b = sector.pairs[gate[1]]
            currents = 2 * angle * np.imag(np.conj(mid[a]) * mid[b])
            score = float(np.sum(abs(currents)))
            if best is None or score > best[0]:
                best = score, mid, a, b, currents, gate[1]
        state = sector.apply(state, gate, angle)
    _, mid, a, b, currents, active_edge = best
    graph = nx.Graph(); graph.add_nodes_from(range(6))
    for pair_a, pair_b in sector.pairs.values():
        graph.add_edges_from(zip(pair_a.tolist(), pair_b.tolist()))
    pos = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(7, 5))
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="lightgray")
    nodes = nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=abs(mid) ** 2, cmap="Blues", node_size=900, vmin=0, vmax=.5)
    nx.draw_networkx_labels(graph, pos, labels=dict(enumerate(labels)), ax=ax)
    for i, j, current in zip(a, b, currents):
        if abs(current) > 1e-8:
            source, target = (int(j), int(i)) if current > 0 else (int(i), int(j))
            nx.draw_networkx_edges(nx.DiGraph([(source, target)]), pos, ax=ax,
                                   arrows=True, arrowsize=25, width=2 + 6 * abs(current), edge_color="#d66b29", node_size=900)
    fig.colorbar(nodes, ax=ax, label="Instantaneous configuration probability")
    ax.set_title(f"Actual XY gate midpoint current, physical edge {active_edge}\nArrow width ∝ |dp/ds|; normalized gate time s∈[0,1]")
    ax.axis("off"); finish(fig, "09_configuration_current")

    candidates = [r for r in records if r["n"] == 4 and "noise" in r and r["id"] != "star-depth6"]
    pareto = [r for r in candidates if not any(
        s["compiled_CZ"]["two_qubit_gates"] <= r["compiled_CZ"]["two_qubit_gates"] and
        s["noise"]["fidelity"] >= r["noise"]["fidelity"] and
        (s["compiled_CZ"]["two_qubit_gates"] < r["compiled_CZ"]["two_qubit_gates"] or s["noise"]["fidelity"] > r["noise"]["fidelity"])
        for s in candidates)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([r["compiled_CZ"]["two_qubit_gates"] for r in candidates], [r["noise"]["fidelity"] for r in candidates], label="Measured circuits")
    pareto.sort(key=lambda r: r["compiled_CZ"]["two_qubit_gates"])
    ax.plot([r["compiled_CZ"]["two_qubit_gates"] for r in pareto], [r["noise"]["fidelity"] for r in pareto], "o-", label="Measured Pareto frontier")
    ax.set(xlabel="Compiled CZ gates after common pruning", ylabel="Synthetic noisy fidelity", title="D₂⁴ path: finite candidate set, not a global optimum")
    ax.legend(); finish(fig, "10_pareto_frontier")

    lines = ["# Research results", "", "Evidence: analytical bounds are **Proven** in theory.md; circuit fidelities are **Verified numerically** against independent full-space Cirq simulation. Performance comparisons are **Observed** for the exact implementations and synthetic noise below, not hardware measurements or novelty claims.", "",
             "All final counts apply the same state-aware pruning pass to candidates and baselines, then the repo CZ compiler (no dynamical decoupling), topology checks, and Willow105.validate_circuit. State-aware pruning is valid for this input only. Exact means numerical fidelity within 1e-9, not an analytic synthesis proof.", "",
             "## Final consistent comparison", "",
             "Noise: p1=0.0001, p2=0.001 depolarizing-channel probabilities; synthetic T1=100 μs, T2=80 μs; illustrative single/CZ duration 25/40 ns. Readout flips are separate classical probabilities. Durations are NOT calibrated Willow predictions. Same noise and compilation pass for every row.", "",
             "| Circuit | n,l | Ideal F | Synthetic noisy F | CZ gates | Moment depth | Illustrative ns | Optimization s |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    selected_ids = ["chain-4-2-depth4-cp0", "chain-4-2-depth6-cp1", "SCS-4-2", "recursive-4-2", "star-depth6",
                    "chain-6-3-depth6-cp0", "chain-6-3-depth6-cp2", "chain-6-3-depth10-cp1", "SCS-6-3", "recursive-6-3"]
    for label in selected_ids:
        r = next(r for r in records if r["id"] == label)
        noise = r["noise"]
        lines.append(f"| {label} | {r['n']},{r['l']} | {r['ideal_fidelity']:.10f} | {noise['fidelity']:.8f} | {r['compiled_CZ']['two_qubit_gates']} | {r['compiled_CZ']['depth']} | {noise['duration_ns_illustrative']:.0f} | {r['optimization_seconds']:.4f} |")
    lines += ["", "The star uses a different connected Willow subset; it is a geometry result, not a same-path comparison. All chain/SCS/recursive rows of a given size use exactly the same physical qubits and allowed path couplers. Recursive counts include the repository's explicit move-and-restore routing, not an optimal recursive compiler. The SCS/recursive implementations are verified here, but these results do not establish an advantage over the best possible literature circuits.", "",
              "## Scientific decisions", "",
              "- **Proven:** unrestricted open-chain XY+Z cannot exactly prepare any intermediate Dicke state (2≤l≤n−2). For D₂⁴ its exact maximum fidelity is 0.9714045207910317, independently checked with SVD, an explicit maximizing Slater, and optimized circuits.",
              "- **Observed:** one CP did not exceed the ceiling in the four-layer schedule but succeeds in the six-layer schedule. Optimizer failure at four layers is not a proof of that schedule's best possible fidelity.",
              "- **Observed:** D₃⁶ free XY attained 0.9 in both tested depths; this is an optimizer observation, NOT a certified global fidelity bound. One CP at ten layers reaches unit fidelity numerically. At six layers two CP gates improve ideal fidelity to 0.9774687898.",
              "- **Observed:** under the stated synthetic noise, the shorter approximate six-layer/two-CP D₃⁶ circuit beats the ten-layer/one-CP numerically exact circuit. See the table; this is a finite Pareto comparison, not direct noise-aware continuous optimization.",
              "- **Observed:** star XY+Z alone reaches D₂⁴; the chain obstruction is not a general Willow graph obstruction. A square remains at the ceiling in tested schedules, consistent with the fixed-parity ring restriction.",
              "- Kernel timing: direct affected-pair gates are fastest in the measured isolated-XY task. Seven repetitions per method, median wall time, independently checked against expm and expm_multiply. Dense exponentials were capped at dimension 70; no full 105-qubit simulation or large optimizer sweep was run. This is not a general multi-term Hamiltonian benchmark.", "",
              "## Missing capabilities and next experiment", "",
              "Arbitrary FSim parameters were used as logical gates only. An independently calibrated fixed-native-FSim comparison is unavailable: this installed Willow105 record accepts full CZ for the comparison, and no processor-specific adjustable-pulse calibration was supplied. Hybrid recursion is unimplemented: a replacement block must be certified simultaneously on every child-weight sector and preserve branch phases. Symmetry-constrained control, direct noisy-objective optimization, crosstalk, leakage, live calibration, and 8+ qubit optimized preparation remain untested. Only small kernel benchmarks reach n=12.", "",
              "The coherent multi-sector leaf experiment was subsequently completed; see below. Full recursive integration remains unimplemented. Next cheapest useful experiment: insert the saved leaf into one existing recursion node whose child registers can be mapped to a physical star; verify every populated branch and compare total routed CZ counts. A local leaf improvement is not yet a full hybrid advantage.", "",
              "## Acceptance and coherent fidelity", "", "| Circuit | Noisy F | P(valid sector) | F after ideal sector projection | P(valid after classical readout flips) |", "|---|---:|---:|---:|---:|"]
    for label in selected_ids:
        r = next(r for r in records if r["id"] == label); z = r["noise"]
        lines.append(f"| {label} | {z['fidelity']:.8f} | {z['p_valid']:.8f} | {z['f_post']:.8f} | {z['observed_valid_after_readout']:.8f} |")
    lines += ["", "F_post assumes an ideal projective weight-sector test before readout. No postselection is inserted into the executable preparation circuit or proposed for the challenge interface. A classical noisy weight filter has different acceptance and does not implement that ideal projection. Noise trajectories use the full small-system density matrix because damping and depolarization leave the fixed-weight sector.", "", "## Reproduction and artifact manifest", "", "```powershell", "python -m pytest interference_dicke/tests -q", "python -m interference_dicke.experiment --noise", "python -m interference_dicke.experiment --extended", "python -m interference_dicke.geometry", "python -m interference_dicke.noise_followup", "python -m interference_dicke.prune", "python -m interference_dicke.report", "```", "", "Run from the du-quantum repository root. Every run has a unique directory; prior artifacts are preserved. JSON stores schedules, angles, seeds, limits, trials/evaluations, timestamps, topology coordinates, versions, commit and measured timings. Source snapshots capture uncommitted implementations; saved Cirq JSON circuits are directly reloadable. The generated report selects the earliest small/extended milestone and latest geometry/pruned results. Wall-clock times vary across machines.", ""]
    for path in [small_path, extended_path, geometry_path, pruned_path]:
        relative = path.relative_to(ROOT).as_posix(); lines.append(f"- [{path.parent.name}]({relative})")
    multi_runs = [(p, d) for p, d in all_runs if p.parent.name.startswith("multisector-")]
    if multi_runs:
        multi_path, multi = sorted(multi_runs, key=lambda x: str(x[0]))[-1]
        lines += ["", "## Coherent multi-sector leaf feasibility", "",
                  "Unlike the state-specific pruned circuits above, these unitary blocks contain no initialization or state-aware pruning. One fixed schedule is checked on all five orthogonal unary inputs on the SAME physical star. The objective penalizes relative phases against the vacuum-sector phase, not just separate-sector fidelities. Analytic gradients were checked with finite differences; Cirq independently checks every compiled sector.", "",
                  "| Star block | CZ gates | Moment depth | Min sector fidelity | Max absolute relative phase (rad) | Common-phase objective loss |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in multi["records"]:
            v = r["verification"]
            lines.append(f"| XY+Z, {r['cp_count']} CP | {r['compiled_CZ']['two_qubit_gates']} | {r['compiled_CZ']['depth']} | {min(v['fidelities']):.12f} | {max(abs(x) for x in v['relative_phases']):.3g} | {v['common_phase_loss']:.3g} |")
        b = multi["baseline"]; v = b["verification"]
        lines.append(f"| Existing SCS unary unitary, routed on same star | {b['compiled_CZ']['two_qubit_gates']} | {b['compiled_CZ']['depth']} | {min(v['fidelities']):.12f} | {max(abs(x) for x in v['relative_phases']):.3g} | {v['common_phase_loss']:.3g} |")
        lines += ["", "**Observed:** the one-CP star block meets the coherent local-leaf test with 26 CZ gates versus 250 for the repository SCS unitary with move-and-restore star routing. The zero-CP bounded optimization leaves sectors 1 and 3 at about 0.933 fidelity while sector 2 reaches unity: this is a concrete warning against single-sector substitution. It is not a proof that zero-CP coherent preparation is impossible at greater depth or under a different schedule. No global hybrid circuit, scaling law, or optimal routing claim follows from this leaf comparison.", "",
                  f"Data and circuit files: [{multi_path.parent.name}]({multi_path.relative_to(ROOT).as_posix()}). Reproduce with `python -m interference_dicke.multisector`; two 12-second family budgets, four starts each. The source snapshot and JSON retain every angle, schedule, trial, common-phase metric, topology and environment."]
    lines += ["", "## Figures", ""]
    for png in sorted(figures.glob("*.png")):
        lines.append(f"- [{png.stem}](figures/{png.name})")
    lines += ["", "Theory and verified bibliography: [theory.md](theory.md). Current hypotheses: [hypothesis_tree.json](hypothesis_tree.json). Kernel timings, raw unpruned circuits, sensitivity across p₂∈{0,0.001,0.005}, and all optimizer trials remain in their original result files."]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hypotheses = [
        ("H1", "Interference reduces routing overhead", "inconclusive", "Path circuits need no SWAP routing, but no same-algorithm routing ablation isolates the source of the benefit.", "Measure explicit SWAP and controlled-distribution costs with the same topology/compiler."),
        ("H2", "Free XY has reachability limitations", "supported", "Proven for open chains; exact D₂⁴ bound verified independently. Star is a counterexample to extending the chain claim to all graphs.", "Joint-sector star leaf test; distinguish ring parity from branched connectivity."),
        ("H3", "Few interactions suffice", "supported", "One CP achieves unit numerical fidelity for D₂⁴ and D₃⁶ at sufficient tested depth; four-layer D₂⁴ is a negative result.", "Require common-sector phase coherence before generalizing."),
        ("H4", "Native FSim provides an advantage", "inconclusive", "Arbitrary FSim compiled to CZ; no calibrated fixed-native comparison exists.", "Obtain an actual processor gate-set record and fixed-pulse compiler."),
        ("H5", "Symmetry reduces search cost", "untested", "Collective symmetry can conserve target overlap and obstruct preparation; no fair constrained-parameter optimization measured.", "Test input-compatible reflection constraints at equal budgets."),
        ("H6", "Short approximate beats deep exact under noise", "supported", "Observed at n=6 for synthetic channel model after equal pruning/compilation, not calibrated hardware.", "Sweep T1/T2 and coherent parameter errors; reversal would limit this observation."),
        ("H7", "Hybrid scales better", "untested", "Single-sector success does not certify coherent recursive substitution.", "Optimize one star block on all populated weight sectors, budget <=30 seconds.")]
    tree = {"updated": pruned["timestamp_utc"], "evidence_scope": "small-system numerical research, not hardware execution", "hypotheses": []}
    for key, title, status, evidence, next_test in hypotheses:
        if key == "H7" and multi_runs:
            status = "inconclusive"
            evidence = "A common-phase, all-weight star leaf with one CP is numerically verified at 26 CZ versus the same-star routed SCS unitary at 250 CZ. Full recursive integration/scaling remains untested. Zero-CP trial fails the coherent leaf test within the bounded schedule."
            next_test = "Substitute the verified leaf into one physically mapped recursion node; compare full routed cost and branch coherence."
        tree["hypotheses"].append({"id": key, "hypothesis": title, "status": status, "evidence": evidence,
                                   "mathematical_evidence": {"H1": "No routing advantage theorem.", "H2": "Complete derivation in theory.md: Plucker obstruction and exact D2^4 Slater bound.", "H3": "CP introduces nonquadratic fermion interactions; no general sufficiency theorem.", "H4": "Exchange and CP commute on the same pair and combine algebraically into FSim; pulse availability is a separate question.", "H5": "Permutation-symmetric target-eigenstate controls can conserve target overlap; symmetry is not automatically helpful.", "H6": "No general noise ordering theorem.", "H7": "The binomial split and common-sector phase conditions are necessary for coherent substitution."}[key],
                                   "numerical_evidence": evidence,
                                   "counterexamples_or_limits": {"H1": "Routing, decomposition and state-specific gate elimination are confounded in total cost differences.", "H2": "The star prepares D2^4 with XY+Z alone; the chain theorem is not graph-general.", "H3": "Four-layer one-CP D2^4 trial did not improve fidelity; zero-CP star single-sector success does not imply all-sector success.", "H4": "No fixed-native-FSim experiment was performed.", "H5": "No structured-parameter benchmark was performed.", "H6": "At n=4 the deeper exact chain circuit beats the shorter approximate one at p2=.001; the ordering depends on noise and candidate.", "H7": "The zero-CP star trial fails common-sector preparation despite unit weight-two fidelity; no full recursive integration exists."}[key],
                                   "experiment_runs": [p.parent.name for p in [small_path, extended_path, geometry_path, pruned_path]],
                                   "next_cheapest_experiment": next_test,
                                   "falsifier": {"H1": "Optimized candidates lose their cost advantage under equally optimized routing.", "H2": "An independently verified open-chain quadratic circuit violates the Plucker bound (would contradict the assumptions or implementation).", "H3": "Additional intermediate states require interaction counts growing too rapidly to retain the cost advantage.", "H4": "Fixed native pulse compilation is no cheaper than CZ at equal fidelity.", "H5": "Symmetry constraints reduce attainable fidelity or do not reduce wall time.", "H6": "The ordering reverses across measured/calibrated error ranges.", "H7": "A coherent multi-sector leaf is no cheaper than the existing exact unitary."}[key]})
    (ROOT / "hypothesis_tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    (ROOT / "comparison.json").write_text(json.dumps({"source": str(pruned_path), "records": records,
                                                       "pareto_ids_n4_path": [r["id"] for r in pareto]}, indent=2), encoding="utf-8")
    (ROOT / "requirements-reproduced.txt").write_text("\n".join(f"{name}=={version}" for name, version in extended["environment"]["packages"].items()) + "\n", encoding="utf-8")
    print(ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
