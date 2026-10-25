"""Audit Will's experimental helpers separately from the public submission."""
import json
import sys

import cirq
from adaptive_dicke_validation.audit import ROOT, load_source, structure, verify_all_branches


def main():
    snapshot = ROOT / "snapshots/will-605b9a8"
    sys.path.insert(0, str(snapshot / "src"))
    sys.path.insert(0, str(snapshot))
    load_source(snapshot / "dicke_submission.py", "dicke_submission")
    bell = load_source(snapshot / "testing/problem2_adaptive_experiments.py", "will_bell")
    circuit, outputs = bell.build_measurement_assisted_dicke_2_1()
    result = {"bell_helper": {"structure_violations": structure(circuit, outputs, 2),
                              "verification": verify_all_branches(circuit, outputs, 1),
                              "reported_readout_line_valid": bell.evaluate_experiment().readout_line_valid},
              "scope": "experimental helpers only; not a rejection of the public accepted portfolio"}
    (ROOT / "artifacts/helper_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
