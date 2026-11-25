"""Validate accuracy and anomaly-detection results against their schemas and check their numbers.

For every ``*.json`` with ``accuracy_schema_version`` (image classification,
``scripts/eval_accuracy.py``) or ``anomaly_schema_version`` (anomaly detection,
``scripts/eval_anomaly.py``) in the given files or directories (searched
recursively): the schema (``schemas/accuracy-result.v<version>.schema.json`` or
``schemas/anomaly-result.v<version>.schema.json``), then for classification

- run results: the file is named ``<run_id>.json``; every accuracy equals
  correct/total; every 95% interval is the Wilson interval of its counts and
  contains the accuracy; top-5 correct >= top-1 correct; all runs in one
  directory use the same dataset manifest;
- comparisons: every compared run exists in the same directory with the
  recorded status and predictions SHA256; top1_a / top1_b are the runs'
  accuracies and top1_delta their difference; the 2x2 counts add up to the
  image count and match both runs' correct counts; the agreement is at least
  the both-correct fraction; the p-value is McNemar's exact test of the counts.

and for anomaly detection

- run results: the file is named ``<run_id>.json``; the confusion counts add up
  to the test windows and the labelled windows; precision, recall, F1 and the
  false-positive rate are those of the counts; no validation window exceeds the
  threshold; every labelled range of the dataset has an event entry, and each
  event's detection point, delay and detected flag follow from its first
  flagged window; all runs in one directory use the same dataset block;
- comparisons: every compared run exists with the recorded status and score
  SHA256; F1, ROC-AUC and detected events are the runs' own; the correctness
  counts add up to the test windows and match each run's tp + tn; the windows
  flagged by only one run are exactly the discordant windows; the flag
  agreement follows from them; the p-value is McNemar's exact test.

Standard library and jsonschema only, so it runs anywhere (CI included).
Exit status: 0 all valid, 1 a check failed, 2 usage error.

Usage:
    python scripts/validate_accuracy.py FILE_OR_DIR [...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.accuracy_stats import mcnemar_exact, wilson_interval  # noqa: E402

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
VERSION_KEY = "accuracy_schema_version"
ANOMALY_KEY = "anomaly_schema_version"
# Version key -> schema file prefix.
SCHEMAS = {VERSION_KEY: "accuracy-result", ANOMALY_KEY: "anomaly-result"}


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def version_key(doc: dict) -> str | None:
    return next((k for k in SCHEMAS if k in doc), None)


def schema_errors(doc: dict) -> list[str]:
    import jsonschema

    key = version_key(doc)
    path = SCHEMA_DIR / f"{SCHEMAS[key]}.v{doc.get(key)}.schema.json"
    if not path.is_file():
        return [f"unsupported {key} {doc.get(key)!r}"]
    validator = jsonschema.Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))
    best = jsonschema.exceptions.best_match(validator.iter_errors(doc))
    return [] if best is None else [f"schema: {'/'.join(map(str, best.absolute_path)) or '<root>'}: "
                                    f"{best.message}"]


def proportion_errors(name: str, p: dict) -> list[str]:
    out = []
    if p["correct"] > p["total"] or not close(p["accuracy"], p["correct"] / p["total"]):
        out.append(f"{name}: accuracy {p['accuracy']} is not {p['correct']}/{p['total']}")
    low, high = wilson_interval(p["correct"], p["total"])
    if not (close(p["ci95"]["low"], low) and close(p["ci95"]["high"], high)):
        out.append(f"{name}: ci95 is not the Wilson interval ({low:.6f}, {high:.6f})")
    if not p["ci95"]["low"] <= p["accuracy"] <= p["ci95"]["high"]:
        out.append(f"{name}: accuracy outside its interval")
    return out


def run_errors(path: Path, doc: dict) -> list[str]:
    out = []
    if path.stem != doc["run_id"]:
        out.append(f"file name should be {doc['run_id']}.json")
    if doc["metrics"]:
        t1, t5 = doc["metrics"]["top1"], doc["metrics"]["top5"]
        out += proportion_errors("top1", t1) + proportion_errors("top5", t5)
        if t5["correct"] < t1["correct"]:
            out.append("top5 correct is below top1 correct")
        if t1["total"] != doc["dataset"]["image_count"]:
            out.append("metrics total differs from the dataset image count")
    return out


def comparison_errors(doc: dict, runs: dict[str, dict]) -> list[str]:
    out = []
    for run_id, info in doc["runs"].items():
        r = runs.get(run_id)
        if r is None:
            out.append(f"runs: {run_id} has no result in this directory")
            continue
        sha = r["predictions"]["top5_indices"]["sha256"] if r["predictions"] else None
        if info != {"status": r["status"], "top5_indices_sha256": sha}:
            out.append(f"runs: {run_id} does not match its result")
    for c in doc["comparisons"]:
        name = c["name"]
        a, b = runs.get(c["a"]), runs.get(c["b"])
        if a is None or b is None or not a["metrics"] or not b["metrics"]:
            out.append(f"{name}: compares a run without a successful result")
            continue
        ta, tb = a["metrics"]["top1"], b["metrics"]["top1"]
        if not (close(c["top1_a"], ta["accuracy"]) and close(c["top1_b"], tb["accuracy"])):
            out.append(f"{name}: top1_a/top1_b differ from the run results")
        if not close(c["top1_delta"], c["top1_a"] - c["top1_b"]):
            out.append(f"{name}: top1_delta is not top1_a - top1_b")
        n = c["counts"]
        total = doc["dataset"]["image_count"]
        if sum(n.values()) != total:
            out.append(f"{name}: counts do not add up to {total}")
        if n["both_correct"] + n["only_a_correct"] != ta["correct"] or \
                n["both_correct"] + n["only_b_correct"] != tb["correct"]:
            out.append(f"{name}: counts do not match the runs' correct counts")
        if c["top1_agreement"] + 1e-12 < n["both_correct"] / total:
            out.append(f"{name}: agreement below the both-correct fraction")
        if not close(c["p_value"], mcnemar_exact(n["only_a_correct"], n["only_b_correct"])):
            out.append(f"{name}: p_value is not McNemar's exact test of the counts")
    return out


def anomaly_run_errors(path: Path, doc: dict) -> list[str]:
    out = []
    if path.stem != doc["run_id"]:
        out.append(f"file name should be {doc['run_id']}.json")
    m, ds, th = doc["metrics"], doc["dataset"], doc["threshold"]
    if m is None:
        return out
    c = m["at_threshold"]
    if c["tp"] + c["fp"] + c["fn"] + c["tn"] != m["test_windows"] or \
            m["test_windows"] != ds["test_windows"]:
        out.append("confusion counts do not add up to the test windows")
    if c["tp"] + c["fn"] != m["positives"] or m["positives"] != ds["test_anomalous_windows"]:
        out.append("tp + fn is not the number of labelled test windows")
    flagged = c["tp"] + c["fp"]
    expected = {"precision": c["tp"] / flagged if flagged else 0.0,
                "recall": c["tp"] / m["positives"] if m["positives"] else 0.0,
                "f1": 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if c["tp"] else 0.0,
                "false_positive_rate": c["fp"] / (c["fp"] + c["tn"]) if c["fp"] + c["tn"] else 0.0}
    for key, value in expected.items():
        if not close(c[key], value):
            out.append(f"at_threshold {key} {c[key]} does not follow from the counts")
    if th["validation_windows"] != ds["validation_windows"]:
        out.append("threshold validation_windows differs from the dataset")
    ev = m["events"]
    if [[e["start"], e["stop"]] for e in ev["ranges"]] != ds["anomaly_ranges"] or \
            ev["total"] != len(ev["ranges"]):
        out.append("events do not list exactly the dataset's anomaly ranges")
    if ev["detected"] != sum(e["detected"] for e in ev["ranges"]):
        out.append("events detected is not the number of detected ranges")
    for e in ev["ranges"]:
        where = f"event [{e['start']}, {e['stop']})"
        if e["detected"] != (e["flagged_windows"] > 0) or e["flagged_windows"] > e["windows"]:
            out.append(f"{where}: detected / flagged_windows inconsistent")
        if e["detected"]:
            point = e["first_flagged_window_start"] + ds["window_length"] - 1
            if e["detection_point"] != point or e["delay_points"] != point - e["start"] or \
                    not close(e["delay_fraction_of_range"],
                              e["delay_points"] / (e["stop"] - e["start"])) or \
                    not close(e["delay_minutes"],
                              e["delay_points"] * ds["sampling_interval_minutes"]):
                out.append(f"{where}: detection point or delay does not follow from the first "
                           "flagged window")
            if e["first_flagged_window_start"] + ds["window_length"] <= e["start"] or \
                    e["first_flagged_window_start"] >= e["stop"]:
                out.append(f"{where}: the first flagged window does not overlap the range")
    return out


def anomaly_comparison_errors(doc: dict, runs: dict[str, dict]) -> list[str]:
    out = []
    for run_id, info in doc["runs"].items():
        r = runs.get(run_id)
        if r is None:
            out.append(f"runs: {run_id} has no result in this directory")
            continue
        sha = r["scores"]["test"]["sha256"] if r["scores"] else None
        if info != {"status": r["status"], "test_scores_sha256": sha}:
            out.append(f"runs: {run_id} does not match its result")
    total = doc["dataset"]["test_windows"]
    for c in doc["comparisons"]:
        name = c["name"]
        a, b = runs.get(c["a"]), runs.get(c["b"])
        if a is None or b is None or not a["metrics"] or not b["metrics"]:
            out.append(f"{name}: compares a run without a successful result")
            continue
        ma, mb = a["metrics"], b["metrics"]
        own = {"f1_a": ma["at_threshold"]["f1"], "f1_b": mb["at_threshold"]["f1"],
               "roc_auc_a": ma["roc_auc"], "roc_auc_b": mb["roc_auc"]}
        if any(not close(c[k], v) for k, v in own.items()) or \
                (c["events_detected_a"], c["events_detected_b"]) != \
                (ma["events"]["detected"], mb["events"]["detected"]):
            out.append(f"{name}: F1, ROC-AUC or events differ from the run results")
        if not close(c["f1_delta"], c["f1_a"] - c["f1_b"]):
            out.append(f"{name}: f1_delta is not f1_a - f1_b")
        n = c["counts"]
        if sum(n.values()) != total:
            out.append(f"{name}: counts do not add up to {total}")
        ca, cb = ma["at_threshold"], mb["at_threshold"]
        if n["both_correct"] + n["only_a_correct"] != ca["tp"] + ca["tn"] or \
                n["both_correct"] + n["only_b_correct"] != cb["tp"] + cb["tn"]:
            out.append(f"{name}: counts do not match the runs' correct windows (tp + tn)")
        if (c["flagged_a"], c["flagged_b"]) != (ca["tp"] + ca["fp"], cb["tp"] + cb["fp"]) or \
                c["flagged_only_a"] - c["flagged_only_b"] != c["flagged_a"] - c["flagged_b"]:
            out.append(f"{name}: flagged counts do not match the runs")
        discordant = c["flagged_only_a"] + c["flagged_only_b"]
        if discordant != n["only_a_correct"] + n["only_b_correct"]:
            out.append(f"{name}: windows flagged by one run are not the discordant windows")
        if not close(c["flag_agreement"], 1 - discordant / total):
            out.append(f"{name}: flag_agreement does not follow from the discordant windows")
        if not close(c["p_value"], mcnemar_exact(n["only_a_correct"], n["only_b_correct"])):
            out.append(f"{name}: p_value is not McNemar's exact test of the counts")
    return out


def validate_anomaly(docs: dict[Path, dict], issues: dict[Path, list[str]]) -> None:
    by_dir: dict[Path, dict[str, dict]] = {}
    for path, doc in docs.items():
        if issues[path] or doc["kind"] != "run":
            continue
        issues[path] += anomaly_run_errors(path, doc)
        by_dir.setdefault(path.parent, {})[doc["run_id"]] = doc
    for directory in {p.parent for p in docs}:
        datasets = [json.dumps(d["dataset"], sort_keys=True) for p, d in docs.items()
                    if p.parent == directory and not issues[p]]
        if len(set(datasets)) > 1:
            for path in (p for p in docs if p.parent == directory):
                issues[path].append("results in this directory use different datasets")
    for path, doc in docs.items():
        if not issues[path] and doc["kind"] == "comparisons":
            issues[path] += anomaly_comparison_errors(doc, by_dir.get(path.parent, {}))


def validate_directory(files: list[Path]) -> dict[Path, list[str]]:
    docs, anomaly = {}, {}
    issues: dict[Path, list[str]] = {}
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues[path] = [f"cannot read: {exc}"]
            continue
        if not isinstance(doc, dict) or version_key(doc) is None:
            continue  # not an accuracy or anomaly document
        (docs if version_key(doc) == VERSION_KEY else anomaly)[path] = doc
        issues[path] = schema_errors(doc)
    validate_anomaly(anomaly, issues)
    by_dir: dict[Path, dict[str, dict]] = {}
    for path, doc in docs.items():
        if issues[path] or doc["kind"] != "run":
            continue
        issues[path] += run_errors(path, doc)
        by_dir.setdefault(path.parent, {})[doc["run_id"]] = doc
    for directory, runs in by_dir.items():
        manifests = {r["dataset"]["manifest_sha256"] for r in runs.values()}
        if len(manifests) > 1:
            for path in (p for p, d in docs.items() if p.parent == directory):
                issues[path].append("runs in this directory use different dataset manifests")
    for path, doc in docs.items():
        if not issues[path] and doc["kind"] == "comparisons":
            issues[path] += comparison_errors(doc, by_dir.get(path.parent, {}))
    return {p: issues[p] for p in issues}


def collect(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted(p.rglob("*.json"))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, metavar="FILE_OR_DIR")
    args = parser.parse_args(argv)
    try:
        files = collect(args.paths)
    except FileNotFoundError as exc:
        print(f"error: no such file or directory: {exc}", file=sys.stderr)
        return 2
    results = validate_directory(files)
    if not results:
        print("No accuracy or anomaly-detection results found.")
        return 0
    failed = 0
    for path, issues in sorted(results.items()):
        failed += bool(issues)
        print(f"{'FAIL' if issues else 'PASS'} {path}")
        for i in issues:
            print(f"  {i}")
    print(f"\n{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
