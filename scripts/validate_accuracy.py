"""Validate accuracy results against their schema and check that their numbers add up.

For every ``*.json`` with ``accuracy_schema_version`` in the given files or
directories (searched recursively): the schema
(``schemas/accuracy-result.v<version>.schema.json``), then

- run results: the file is named ``<run_id>.json``; every accuracy equals
  correct/total; every 95% interval is the Wilson interval of its counts and
  contains the accuracy; top-5 correct >= top-1 correct; all runs in one
  directory use the same dataset manifest;
- comparisons: every compared run exists in the same directory with the
  recorded status and predictions SHA256; top1_a / top1_b are the runs'
  accuracies and top1_delta their difference; the 2x2 counts add up to the
  image count and match both runs' correct counts; the agreement is at least
  the both-correct fraction; the p-value is McNemar's exact test of the counts.

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


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def schema_errors(doc: dict) -> list[str]:
    import jsonschema

    path = SCHEMA_DIR / f"accuracy-result.v{doc.get(VERSION_KEY)}.schema.json"
    if not path.is_file():
        return [f"unsupported {VERSION_KEY} {doc.get(VERSION_KEY)!r}"]
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


def validate_directory(files: list[Path]) -> dict[Path, list[str]]:
    docs = {}
    issues: dict[Path, list[str]] = {}
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues[path] = [f"cannot read: {exc}"]
            continue
        if not isinstance(doc, dict) or VERSION_KEY not in doc:
            continue  # not an accuracy document
        docs[path] = doc
        issues[path] = schema_errors(doc)
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
        print("No accuracy results found.")
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
