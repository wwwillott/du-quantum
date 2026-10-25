"""Render incremental Dicke head-to-head results without importing candidates.

Usage: python team_eval/render_results.py [team_eval/results.json]
       [--output team_eval/RESULTS.md]

The orchestrator can also call ``render(results, output_path)`` after each row.
Scores are ranked only after exactness, validity, and build-time checks pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


CANDIDATES = {
    "A": "team_eval/will_dicke_submission.py (reference)",
    "B": "submission.py",
    "C": "columnforge/noise_aware/rewrites/submission_scheduled.py",
    "D": "astra/submission.py",
}
MODELS = ("pauli", "hw")
INSTANCES = tuple((N, n) for N in range(2, 13) for n in range(1, N // 2 + 1))
EXACTNESS_THRESHOLD = 1 - 1e-8
ARTIFACTS = Path(__file__).resolve().parent


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value: Any, digits: int = 5) -> str:
    if value is None:
        return "—"
    if not _finite(value):
        return str(value)
    return f"{value:.{digits}f}"


def _safe(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _passed(record: dict, field: str) -> bool:
    check = record.get(field)
    return isinstance(check, dict) and check.get("passed") is True


def _eligible(record: dict | None) -> bool:
    if not record or not _passed(record, "exactness") or not _passed(record, "validity"):
        return False
    fidelity = record["exactness"].get("fidelity")
    moments = record.get("moments")
    seconds = record.get("construction_seconds")
    return (_finite(fidelity) and fidelity > EXACTNESS_THRESHOLD
            and _finite(moments) and 0 <= moments <= 1750
            and _finite(seconds) and 0 <= seconds <= 60)


def _score(record: dict | None, model: str) -> float | None:
    if not _eligible(record):
        return None
    stats = record.get("models", {}).get(model, {})
    value = stats.get("median") if isinstance(stats, dict) else None
    return float(value) if _finite(value) else None


def _fully_scored(record: dict | None) -> bool:
    return all(_score(record, model) is not None for model in MODELS)


def _failure_reasons(record: dict) -> list[str]:
    reasons = []
    failures = record.get("failures") or []
    if isinstance(failures, (str, dict)):
        failures = [failures]
    reasons.extend(_safe(item) for item in failures if item)
    for field, label in (("exactness", "non-exact"), ("validity", "invalid")):
        check = record.get(field)
        numeric_failure = (field == "exactness" and isinstance(check, dict)
                           and _finite(check.get("fidelity")) and check["fidelity"] <= EXACTNESS_THRESHOLD)
        if isinstance(check, dict) and (check.get("passed") is False or numeric_failure):
            detail = check.get("errors") or check.get("error")
            if field == "exactness":
                detail = f"fidelity={check.get('fidelity', 'unknown')}"
            reasons.append(label + (f": {_safe(detail)}" if detail else ""))
    seconds = record.get("construction_seconds")
    if _finite(seconds) and seconds > 60:
        reasons.append(f"construction >60 s ({seconds:.3f} s)")
    moments = record.get("moments")
    if _finite(moments) and moments > 1750:
        reasons.append(f"invalid: {moments} moments exceeds 1750")
    for key in ("error_type", "exception_type", "error", "exception"):
        if record.get(key):
            reasons.append(_safe(record[key]))
    if record.get("status") == "failed" and not reasons:
        reasons.append("failed (no error detail recorded)")
    return list(dict.fromkeys(reasons))


def _winners(values: dict[str, float | None]) -> list[str]:
    available = {candidate: value for candidate, value in values.items() if value is not None}
    if not available:
        return []
    best = max(available.values())
    return [candidate for candidate, value in available.items() if math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-14)]


def _winner_label(winners: list[str], records: dict[str, dict | None], available: int) -> str:
    if not winners:
        return "pending"
    labels = [f"{candidate} ({records[candidate].get('cz_count', '?')} CZ)" for candidate in winners]
    return ", ".join(labels) + (" †" if available < len(CANDIDATES) else "")


def _cell(record: dict | None, best: dict[str, list[str]], candidate: str) -> str:
    if record is None:
        return "pending"
    cz, moments = record.get("cz_count", "—"), record.get("moments", "—")
    scores = []
    for model in MODELS:
        stats = record.get("models", {}).get(model, {})
        value = stats.get("median") if isinstance(stats, dict) else None
        formatted = _number(value)
        if candidate in best[model]:
            formatted = f"**{formatted}**"
        elif value is not None and not _eligible(record):
            formatted += " ×"
        scores.append(formatted)
    cell = f"{cz} / {moments} / {' / '.join(scores)}"
    if _failure_reasons(record):
        cell += " ⚠"
    elif not _fully_scored(record):
        cell += f" ({_safe(record.get('status', 'pending'))})"
    return cell


def _artifact(name: str) -> dict | None:
    """Read only small verification JSON; never import the simulator here."""
    try:
        value = json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _metadata_lines(metadata: dict) -> list[str]:
    methodology_status = "" if (ARTIFACTS / "METHODOLOGY.md").exists() else " (document pending)"
    lines = [f"Reproduction and caveats: [METHODOLOGY.md](METHODOLOGY.md){methodology_status}. Re-score saved circuits with [score_circuit.py](score_circuit.py).", ""]
    seeds = metadata.get("seeds", [101, 202])
    calibration_links = ", ".join(f"[seed {int(seed)}](calibrations/seed_{int(seed)}.json)" for seed in seeds)
    lines.extend([f"**Canonical calibrations:** all candidates receive copies of the same saved dictionaries ({calibration_links}). Re-scoring also loads these dictionaries; it does not generate a fresh map. This preserves the exact qubit/coupler assignments as well as RNG seed and notebook generator call order.", ""])
    parity = _artifact("score_parity_results.json")
    calibration_check = _artifact("calibration_repro_check.json")
    if calibration_check:
        checks = calibration_check.get("calibrations", {})
        passed = [str(seed) for seed in seeds if checks.get(str(seed), {}).get("equal") is True]
        lines.extend([f"Calibration regeneration matches saved dictionary fingerprints for **{len(passed)}/{len(seeds)} seeds** under `PYTHONHASHSEED={calibration_check.get('pythonhashseed', 'unrecorded')}`. [Reproducibility evidence](calibration_repro_check.json).", ""])
    else:
        lines.extend(["Calibration fingerprint reproduction check: pending. [Evidence file](calibration_repro_check.json).", ""])
    if parity:
        cases = parity.get("cases", [])
        passed = sum(case.get("passed") is True for case in cases)
        maximum = parity.get("maximum_absolute_difference")
        difference = f"{maximum:.3g}" if _finite(maximum) else "unrecorded"
        generator_status = "passed" if parity.get("generators_verbatim") is True else "not passed or unrecorded"
        lines.extend([f"**Official scorer parity:** {passed}/{len(cases)} sampled count sets passed, including ideal and noisy nonuniform distributions; largest absolute difference across score/Z/X was {difference}. The notebook-generator verbatim source check {generator_status}. [Scorer parity summary and individual cases](score_parity_results.json).", ""])
        edge_case = parity.get("edge_case")
        if edge_case:
            lines.extend([f"Recorded scorer edge case: {_safe(edge_case.get('note', edge_case))}", ""])
    else:
        lines.extend(["Official scorer parity: pending. [Scorer parity summary](score_parity_results.json).", ""])
    backend = _artifact("backend.json")
    if backend:
        maximum = backend.get("max_absolute_probability_difference")
        difference = f"{maximum:.3g}" if _finite(maximum) else "unrecorded"
        case_count = len(backend.get("benchmarks", []))
        lines.extend([f"**CPU acceleration:** scorer backend `{_safe(backend.get('backend', 'unrecorded'))}` uses {_safe(backend.get('scope', 'scope not recorded'))}. On {case_count} recorded circuit/model comparisons with identical seeds, the largest probability difference from the original NumPy backend was {difference}. This is a spot-check of implementation agreement, not an independent hardware-model validation. Earlier NumPy score files may coexist with Numba score files; see individual scoring artifacts for recorded backend identifiers. [Acceleration parity and timings](backend.json).", ""])
    else:
        lines.extend(["CPU Numba acceleration parity: pending; no passed acceleration check is assumed. [Acceleration evidence](backend.json).", ""])
    notes = metadata.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    if notes:
        lines.extend(["Run notes:", ""] + [f"- {_safe(note)}" for note in notes] + [""])
    rest = {key: value for key, value in metadata.items() if key not in ("notes",)}
    if rest:
        lines.extend(["Recorded run configuration:", "", "```json", json.dumps(rest, indent=2, ensure_ascii=False, default=str), "```", ""])
    return lines


def render(results: dict, output_path: str | Path) -> str:
    """Write a complete Markdown snapshot; return its contents.

    Input records use candidate/seed/N/n keys, exactness and validity dictionaries,
    and ``models[pauli|hw][median|mean]``. Missing rows remain explicitly pending.
    If a rerun left duplicate keys, the last record is the current observation.
    """
    metadata = results.get("metadata", {})
    records = results.get("records", [])
    index: dict[tuple[str, int, int, int], dict] = {}
    for record in records:
        try:
            key = (record["candidate"], int(record["seed"]), int(record["N"]), int(record["n"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key[0] in CANDIDATES and (key[2], key[3]) in INSTANCES:
            index[key] = record
    seed_values = metadata.get("seeds", [101, 202])
    seeds = list(dict.fromkeys(int(seed) for seed in seed_values))
    if not seeds:
        seeds = [101, 202]
    expected = len(CANDIDATES) * len(seeds) * len(INSTANCES)
    ready = sum(_fully_scored(record) for key, record in index.items() if key[1] in seeds)
    seen = sum(key[1] in seeds for key in index)
    failure_rows = [(key, row, _failure_reasons(row)) for key, row in index.items() if key[1] in seeds and _failure_reasons(row)]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Dicke-state submission head-to-head", "",
        "**Simulator proxies only; these are not hardware results.** Higher scores are better.", "",
        f"Snapshot: {now}. Eligible circuits with both models scored: **{ready}/{expected}**; recorded circuits: {seen}/{expected}; circuits with reported failures: {len(failure_rows)}.", "",
        "Candidates:", "",
        *[f"- **{candidate}** — `{source}`" for candidate, source in CANDIDATES.items()], "",
        "Each calibration seed uses the same dictionaries for every candidate. Requested settings are K=N; 10,000 shots per basis; at least 2,000 Pauli trajectories and 1,000 HW trajectories; at least 400 shot resamplings. Actual settings and score-formula spot-check results are recorded below when supplied by the runner.", "",
        "Eligibility requires recorded noiseless exactness and validity passes, finite fidelity strictly >1−1e−8, at most 1,750 moments, and construction time ≤60 seconds. A missing check is pending. Non-finite medians are not ranked. All score sums below sum instance medians, not pooled shots or per-resampling sums. Raw means, fidelity, build time, validation detail, and circuit paths are in `results.json`.", "",
        "**Partial-data rules:** † marks a provisional winner with fewer than four eligible scores. × marks an unranked score from an ineligible circuit. ⚠ points to the failures table. Bold scores are per-instance maxima among eligible available results; ties are retained. Small-state scores can be heavy-tailed, so reported means and medians need not agree.", "",
        "## Score sums and coverage", "",
        "Available sums can have different coverage and must not be compared directly. Common sums use only instances with an eligible score for **all four** candidates under that seed/model. A complete all-instance sum requires 36/36.", "",
        "| Seed | Candidate | Pauli available sum | Pauli coverage | Pauli common sum | Common coverage | HW available sum | HW coverage | HW common sum | Common coverage |",
        "|---:|:---|---:|:---:|---:|:---:|---:|:---:|---:|:---:|",
    ]
    for seed in seeds:
        common = {model: [(N, n) for N, n in INSTANCES if all(_score(index.get((c, seed, N, n)), model) is not None for c in CANDIDATES)] for model in MODELS}
        for candidate in CANDIDATES:
            cells = []
            for model in MODELS:
                available = [_score(index.get((candidate, seed, N, n)), model) for N, n in INSTANCES]
                values = [value for value in available if value is not None]
                common_values = [_score(index.get((candidate, seed, N, n)), model) for N, n in common[model]]
                cells.extend([_number(sum(values)) if values else "—", f"{len(values)}/36", _number(sum(common_values)) if common_values else "—", f"{len(common_values)}/36"])
            lines.append(f"| {seed} | {candidate} | " + " | ".join(cells) + " |")
    lines.append("")
    for seed in seeds:
        lines.extend([
            f"## Per-instance results — seed {seed}", "",
            "Each candidate cell: **CZ / moments / Pauli median / HW median**.", "",
            "| N | n | A (reference) | B (root) | C (ColumnForge) | D (Astra) | Pauli winner | HW winner |",
            "|---:|---:|:---|:---|:---|:---|:---|:---|",
        ])
        for N, n in INSTANCES:
            row = {candidate: index.get((candidate, seed, N, n)) for candidate in CANDIDATES}
            scores = {model: {candidate: _score(record, model) for candidate, record in row.items()} for model in MODELS}
            best = {model: _winners(scores[model]) for model in MODELS}
            cells = [_cell(row[candidate], best, candidate) for candidate in CANDIDATES]
            winners = [_winner_label(best[model], row, sum(value is not None for value in scores[model].values())) for model in MODELS]
            lines.append(f"| {N} | {n} | " + " | ".join(cells + winners) + " |")
        lines.append("")
    lines.extend([
        "## Per-instance winner across calibration seeds", "",
        "A candidate enters this table only when both models are eligible and scored for every requested seed. Winners maximize the arithmetic mean of the seed medians. CZ counts follow seed order " + ", ".join(map(str, seeds)) + ". A † winner remains provisional until all four candidates have paired complete results.", "",
        "| N | n | Paired candidates | Pauli winner (CZ by seed) | Mean Pauli median | HW winner (CZ by seed) | Mean HW median |",
        "|---:|---:|:---|:---|---:|:---|---:|",
    ])
    win_counts: dict[str, Counter] = {model: Counter() for model in MODELS}
    complete_comparisons = 0
    for N, n in INSTANCES:
        paired = {candidate: [index.get((candidate, seed, N, n)) for seed in seeds] for candidate in CANDIDATES}
        paired = {candidate: rows for candidate, rows in paired.items() if all(_fully_scored(record) for record in rows)}
        complete_comparisons += len(paired) == len(CANDIDATES)
        cells = []
        for model in MODELS:
            values = {candidate: sum(_score(record, model) for record in rows) / len(seeds) for candidate, rows in paired.items()}
            winners = _winners(values)
            if winners:
                labels = [f"{candidate} ({'/'.join(str(record.get('cz_count', '?')) for record in paired[candidate])})" for candidate in winners]
                cells.extend([", ".join(labels) + (" †" if len(paired) < len(CANDIDATES) else ""), _number(values[winners[0]])])
                if len(paired) == len(CANDIDATES):
                    win_counts[model].update(winners)
            else:
                cells.extend(["pending", "—"])
        lines.append(f"| {N} | {n} | {','.join(paired) or '—'} | " + " | ".join(cells) + " |")
    lines.extend(["", f"Complete four-candidate paired comparisons: {complete_comparisons}/36. Winner counts (ties count for each candidate): Pauli " + ", ".join(f"{c}={win_counts['pauli'][c]}" for c in CANDIDATES) + "; HW " + ", ".join(f"{c}={win_counts['hw'][c]}" for c in CANDIDATES) + ".", "", "## Failures and validation", ""])
    if failure_rows:
        lines.extend(["| Seed | N | n | Candidate | Construction seconds | Reasons |", "|---:|---:|---:|:---|---:|:---|"])
        for (candidate, seed, N, n), record, reasons in sorted(failure_rows, key=lambda item: (item[0][1], item[0][2], item[0][3], item[0][0])):
            lines.append(f"| {seed} | {N} | {n} | {candidate} | {_number(record.get('construction_seconds'), 3)} | {'; '.join(reasons)} |")
    else:
        lines.append("No failures recorded in the current snapshot. Pending cases are not presumed valid or exact.")
    retry = _artifact("timing_retry_results.json")
    if retry:
        identity = ("The retry produced byte-identical circuit JSON and the same returned qubits."
                    if retry.get("same_circuit_sha256") is True and retry.get("same_qubits") is True
                    else "Circuit identity on retry is not fully confirmed.")
        verification = ("Noiseless exactness and validity passed again."
                        if _passed(retry, "exactness") and _passed(retry, "validity")
                        else "Retry exactness/validity verification is incomplete or failed.")
        original = index.get((retry.get("candidate"), retry.get("seed"), retry.get("N"), retry.get("n")), {})
        raw_available = all(_finite(original.get("models", {}).get(model, {}).get("median")) for model in MODELS)
        score_note = ("Scores for this identical circuit are available as unranked raw results in `results.json`."
                      if raw_available else "Any scores for this circuit remain unranked raw results in `results.json`.")
        lines.extend(["", f"**Timing retry:** candidate {_safe(retry.get('candidate', '?'))}, seed {_safe(retry.get('seed', '?'))}, D({_safe(retry.get('N', '?'))},{_safe(retry.get('n', '?'))}) built in {_number(retry.get('original_construction_seconds'), 3)} s initially and {_number(retry.get('construction_seconds'), 3)} s in an isolated retry. {identity} {verification} The primary strict sums and rankings retain the original >60 s failure; this retry does not change eligibility. {score_note} [Timing retry evidence](timing_retry_results.json).", ""])
    lines.extend(["", "## Run configuration and verification", ""])
    lines.extend(_metadata_lines(metadata))
    if not metadata:
        lines.extend(["Run metadata has not yet been supplied.", ""])
    text = "\n".join(lines).rstrip() + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("RESULTS.md"))
    args = parser.parse_args()
    results = json.loads(args.input.read_text(encoding="utf-8"))
    render(results, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
