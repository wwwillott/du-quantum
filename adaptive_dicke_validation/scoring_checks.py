"""Verify scorer parity and audit the new surrogate/walkthrough claims."""
import ast
import collections
import csv
import itertools
import json
import math

import numpy as np
from adaptive_dicke_validation.audit import ROOT, load_source


def official_functions():
    notebook = json.loads((ROOT.parent / "DuQuantum_Google_Standard.ipynb").read_text(encoding="utf-8"))
    names = {"overall_score", "check_z_bitstring_distribution", "check_x_bitstring_distribution", "get_correlator_average"}
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "def overall_score" not in source:
            continue
        nodes.extend(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names)
    scope = {"np": np, "math": math, "itertools": itertools}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "official_notebook_score", "exec"), scope)
    return scope


def main():
    official = official_functions()
    fast = load_source(ROOT / "snapshots/kai-48115b9/tools/sim.py", "kai_scorer")
    rng = np.random.default_rng(42)
    parity = []
    for n, k in [(2, 1), (4, 2), (6, 1), (6, 3)]:
        table = ((np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.int8)
        for label in ["uniform_noise", "ideal_weight_Z"]:
            pz = np.ones(2**n)
            if label == "ideal_weight_Z":
                pz = np.array([int(i).bit_count() == k for i in range(2**n)], float)
            pz /= pz.sum()
            px = rng.uniform(size=2**n); px /= px.sum()
            cz, cx = rng.multinomial(10000, pz), rng.multinomial(10000, px)
            z, x = np.repeat(table, cz, axis=0), np.repeat(table, cx, axis=0)
            ref = [official["overall_score"](z, x, k, n), official["check_z_bitstring_distribution"](z, k),
                   official["check_x_bitstring_distribution"](x, k, n)]
            actual = fast.score_from_counts(cz, cx, n, k, n)[:3]
            parity.append({"n": n, "k": k, "case": label, "maximum_error": float(np.max(abs(np.array(ref) - actual))),
                           "passed": bool(np.allclose(ref, actual, atol=1e-10, rtol=1e-10))})
    # Exact conditional multinomial expectation: E[Z|B]=
    # (1-sum q_i^2)/f + S*sum(q_i-1/M)^2, f=B/S.
    m, shots = 6, 10000
    q = np.array([.20, .13, .17, .16, .18, .16])
    dispersion = float(np.sum((q - 1/m) ** 2))
    checks = []
    for fraction in [1., .5, .25]:
        counts = rng.multinomial(int(shots * fraction), q, size=20000)
        pearson = np.sum((counts - shots * fraction / m) ** 2 / (shots * fraction / m), axis=1)
        observed = float(np.mean(pearson / (m * fraction)))
        exact = (1 - np.sum(q*q)) / fraction + shots * dispersion
        checks.append({"fraction_valid": fraction, "observed_mean_Z": observed, "exact_conditional_mean_Z": float(exact),
                       "standard_error": float(np.std(pearson / (m * fraction)) / np.sqrt(len(counts))),
                       "true_dispersion_leading_term": shots * dispersion,
                       "extra_f_factor_prediction": shots * fraction * dispersion})
    # Counterexample to applying a relative error to zero target correlators.
    n, k, eta = 6, 1, .9
    expected = []
    for ell in [2, 4, 6]:
        value = (math.comb(ell, ell//2) * math.comb(n - ell, k - ell//2) / math.comb(n, k)
                 if 0 <= k - ell//2 <= n - ell else 0.)
        expected.append(value)
    actual_x = sum((1 - eta**ell)**2 for ell, value in zip([2, 4, 6], expected) if value != 0)
    surrogate_x = sum((1 - eta**ell)**2 for ell in [2, 4, 6])
    rows = list(csv.DictReader((ROOT / "snapshots/will-605b9a8/testing/vqm_shallow_vs_runtime.csv").open()))
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(float(row["estimated_judge_score"]))
    result = {"scorer_parity": parity, "all_parity_pass": all(r["passed"] for r in parity),
              "multinomial_dispersion_checks": checks,
              "zero_correlator_counterexample": {"n": n, "k": k, "attenuation_per_site": eta,
                                                 "ideal_even_correlators": expected,
                                                 "correct_noiseless_population_X_penalty_after_attenuation": actual_x,
                                                 "surrogate_even_sum": surrogate_x},
              "committed_csv_totals": {key: {"rows": len(values), "sum_score": sum(values)} for key, values in grouped.items()},
              "walkthrough_claimed_totals": {"runtime_anytime": 9.645, "shallow_incumbent": 9.527},
              "scope": "surrogate findings concern problem 1 ranking; challenge 2 uses moment counts"}
    output = ROOT / "artifacts/scoring_checks.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
