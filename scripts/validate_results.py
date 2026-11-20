"""Validate EdgeBench benchmark result files.

Checks each file against schemas/benchmark-result.schema.json, then applies
consistency rules that JSON Schema cannot express (percentile ordering,
sample-count thresholds, throughput vs. mean latency, trial counts vs. the run
configuration, and the raw latency file). See docs/results-schema.md.

Usage:
    python scripts/validate_results.py FILE_OR_DIR [FILE_OR_DIR ...]
    python scripts/validate_results.py benchmarks/results/published

Directories are searched recursively for *.json files. Exits with status 1 if
any file fails, 2 on usage errors.

Platform-independent: needs only Python 3.10+ and jsonschema, so result files
can be validated on any OS or CPU architecture (e.g. Linux CI runners).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import jsonschema
from jsonschema.exceptions import best_match

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "benchmark-result.schema.json"

NIL_UUID = "00000000-0000-0000-0000-000000000000"
EXAMPLE_WARNING_CODE = "illustrative_example"
RAW_CSV_HEADER = ["trial_index", "iteration_index", "latency_ms"]

# Minimum sample counts below which a statistic must be null (and above which it must be set).
MIN_COUNT = {"stddev_ms": 2, "p95_ms": 20, "p99_ms": 100}
REL_TOL = 1e-3


@dataclass(frozen=True)
class Issue:
    path: str  # JSON Pointer into the document, or "" for the whole file
    message: str

    def __str__(self) -> str:
        return f"{self.path or '/'}: {self.message}"


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def make_validator(schema: dict[str, Any] | None = None) -> jsonschema.Draft202012Validator:
    schema = schema if schema is not None else load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


def pointer(parts: Iterable[Any]) -> str:
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def schema_issues(doc: Any, validator: jsonschema.Draft202012Validator) -> list[Issue]:
    supported = validator.schema["properties"]["schema_version"]["enum"]
    if isinstance(doc, dict) and isinstance(doc.get("schema_version"), str):
        version = doc["schema_version"]
        if version not in supported:
            major = version.split(".", 1)[0]
            if major != supported[-1].split(".", 1)[0]:
                return [Issue("/schema_version",
                              f"unsupported major version {version!r}; this schema supports {supported}")]
            return [Issue("/schema_version",
                          f"version {version!r} is not known to this schema (supports {supported}); "
                          "update schemas/benchmark-result.schema.json")]

    issues = []
    for error in validator.iter_errors(doc):
        for leaf in _most_specific(error):
            issues.append(Issue(pointer(leaf.absolute_path), _message(leaf)))
    return sorted(set(issues), key=lambda i: (i.path, i.message))


def _most_specific(error: jsonschema.ValidationError) -> list[jsonschema.ValidationError]:
    """Descend into oneOf/anyOf sub-errors to the most deeply nested ones.

    Nullable blocks are written as oneOf [null, object]. When the object branch
    fails, the top-level error ("... is not valid under any of the given
    schemas") and the null branch ("... is not of type 'null'") are both
    unhelpful; the deepest sub-errors point at the actual problems.
    """
    if not error.context:
        return [error]
    deepest = max(len(e.absolute_path) for e in error.context)
    candidates = [e for e in error.context if len(e.absolute_path) == deepest]
    if deepest == len(error.absolute_path):
        # No branch got further than the others (e.g. a wrong scalar type): pick the best one.
        candidates = [best_match(candidates)]
    return [leaf for e in candidates for leaf in _most_specific(e)]


def _message(error: jsonschema.ValidationError, limit: int = 80) -> str:
    """Error message with long instance values shortened."""
    instance = repr(error.instance)
    if len(instance) > limit and instance in error.message:
        short = f"{type(error.instance).__name__} value" if isinstance(error.instance, (dict, list)) else instance[:limit] + "..."
        return error.message.replace(instance, short)
    return error.message


# ---------------------------------------------------------------------------
# Consistency rules
# ---------------------------------------------------------------------------

def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-9)


def stats_issues(stats: dict[str, Any], batch_size: int, base: str) -> list[Issue]:
    issues = []
    count = stats["count"]
    for key, minimum in MIN_COUNT.items():
        value = stats[key]
        if count < minimum and value is not None:
            issues.append(Issue(f"{base}/{key}", f"must be null when count < {minimum} (count = {count})"))
        if count >= minimum and value is None:
            issues.append(Issue(f"{base}/{key}", f"must be set when count >= {minimum} (count = {count})"))

    lo, hi = stats["min_ms"], stats["max_ms"]
    if lo > hi:
        issues.append(Issue(f"{base}/min_ms", f"min_ms ({lo}) > max_ms ({hi})"))
    if not lo <= stats["mean_ms"] <= hi:
        issues.append(Issue(f"{base}/mean_ms", "mean_ms must lie between min_ms and max_ms"))
    chain = [("min_ms", lo)] + [(k, stats[k]) for k in ("p50_ms", "p95_ms", "p99_ms") if stats[k] is not None]
    chain.append(("max_ms", hi))
    for (ka, a), (kb, b) in zip(chain, chain[1:]):
        if a > b:
            issues.append(Issue(f"{base}/{kb}", f"{kb} ({b}) is smaller than {ka} ({a})"))

    mean = stats["mean_ms"]
    if mean > 0 and not _close(stats["inferences_per_sec"], 1000.0 / mean):
        issues.append(Issue(f"{base}/inferences_per_sec",
                            f"expected 1000 / mean_ms = {1000.0 / mean:.6g}, got {stats['inferences_per_sec']}"))
    expected_items = stats["inferences_per_sec"] * batch_size
    if not _close(stats["items_per_sec"], expected_items):
        issues.append(Issue(f"{base}/items_per_sec",
                            f"expected inferences_per_sec * batch_size = {expected_items:.6g}, "
                            f"got {stats['items_per_sec']}"))
    return issues


def timing_issues(doc: dict[str, Any]) -> list[Issue]:
    timing, config = doc["timing"], doc["config"]
    if timing is None:
        return []
    issues = []
    batch = config["batch_size"]
    trials = timing["trials"]

    indices = [t["trial_index"] for t in trials]
    if indices != list(range(len(trials))):
        issues.append(Issue("/timing/trials", f"trial_index values must be 0..{len(trials) - 1} in order, got {indices}"))
    for i, trial in enumerate(trials):
        base = f"/timing/trials/{i}/stats"
        issues += stats_issues(trial["stats"], batch, base)
        if trial["stats"]["count"] != config["measured_iterations"]:
            issues.append(Issue(f"{base}/count",
                                f"count ({trial['stats']['count']}) != config.measured_iterations "
                                f"({config['measured_iterations']})"))

    if doc["status"] == "success" and len(trials) != config["trials"]:
        issues.append(Issue("/timing/trials", f"{len(trials)} trials recorded, config.trials = {config['trials']}"))

    aggregate = timing["aggregate"]
    if aggregate is not None:
        issues += stats_issues(aggregate["stats"], batch, "/timing/aggregate/stats")
        total = sum(t["stats"]["count"] for t in trials)
        if trials and aggregate["stats"]["count"] != total:
            issues.append(Issue("/timing/aggregate/stats/count",
                                f"count ({aggregate['stats']['count']}) != sum of trial counts ({total})"))
        variation = aggregate["trial_variation"]
        means = [t["stats"]["mean_ms"] for t in trials]
        if means:
            if not _close(variation["trial_mean_min_ms"], min(means)):
                issues.append(Issue("/timing/aggregate/trial_variation/trial_mean_min_ms",
                                    f"expected min of trial means = {min(means)}"))
            if not _close(variation["trial_mean_max_ms"], max(means)):
                issues.append(Issue("/timing/aggregate/trial_variation/trial_mean_max_ms",
                                    f"expected max of trial means = {max(means)}"))
        for key in ("trial_mean_stddev_ms", "trial_mean_cv_ratio"):
            if len(trials) < 2 and variation[key] is not None:
                issues.append(Issue(f"/timing/aggregate/trial_variation/{key}", "must be null with fewer than 2 trials"))
            if len(trials) >= 2 and variation[key] is None:
                issues.append(Issue(f"/timing/aggregate/trial_variation/{key}", "must be set with 2 or more trials"))
    return issues


def raw_issues(doc: dict[str, Any], result_path: Path | None) -> list[Issue]:
    raw = doc["raw"]
    if raw is None or result_path is None:
        return []
    path = result_path.parent / raw["latencies_file"]
    if not path.is_file():
        return [Issue("/raw/latencies_file", f"file not found: {path}")]
    issues = []
    if raw["sha256"] is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != raw["sha256"]:
            issues.append(Issue("/raw/sha256", f"does not match {path.name} (actual {actual})"))
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != RAW_CSV_HEADER:
            return issues + [Issue("/raw/latencies_file", f"{path.name}: header must be {','.join(RAW_CSV_HEADER)}")]
        rows = list(reader)
    for n, row in enumerate(rows, start=2):
        try:
            if len(row) != 3 or float(row[2]) < 0:
                raise ValueError
            int(row[0]), int(row[1])
        except ValueError:
            issues.append(Issue("/raw/latencies_file", f"{path.name}: line {n} is not 'int,int,non-negative number'"))
            break
    timing = doc["timing"]
    if timing is not None and timing["aggregate"] is not None:
        expected = timing["aggregate"]["stats"]["count"]
        if len(rows) != expected:
            issues.append(Issue("/raw/latencies_file",
                                f"{path.name}: {len(rows)} rows, expected {expected} (aggregate count)"))
    return issues


def consistency_issues(doc: dict[str, Any], result_path: Path | None = None) -> list[Issue]:
    issues = []
    if doc["run_id"] == NIL_UUID and not any(w["code"] == EXAMPLE_WARNING_CODE for w in doc["warnings"]):
        issues.append(Issue("/run_id", f"the nil UUID is reserved for examples with a '{EXAMPLE_WARNING_CODE}' warning"))
    issues += timing_issues(doc)
    issues += raw_issues(doc, result_path)
    return issues


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def validate_document(doc: Any, validator: jsonschema.Draft202012Validator,
                      result_path: Path | None = None) -> list[Issue]:
    """Return all problems with a parsed result document (empty list if valid)."""
    issues = schema_issues(doc, validator)
    if issues:
        return issues  # consistency rules assume a schema-valid document
    return consistency_issues(doc, result_path)


def validate_file(path: Path, validator: jsonschema.Draft202012Validator) -> list[Issue]:
    try:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, UnicodeDecodeError) as exc:
        return [Issue("", f"cannot read file: {exc}")]
    except json.JSONDecodeError as exc:
        return [Issue("", f"invalid JSON: {exc}")]
    return validate_document(doc, validator, path)


def collect_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
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
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH, help="schema file (default: %(default)s)")
    args = parser.parse_args(argv)

    try:
        files = collect_files(args.paths)
    except FileNotFoundError as exc:
        print(f"error: no such file or directory: {exc}", file=sys.stderr)
        return 2
    if not files:
        print("No JSON files found.")
        return 0

    validator = make_validator(load_schema(args.schema))
    failed = 0
    for path in files:
        issues = validate_file(path, validator)
        if issues:
            failed += 1
            print(f"FAIL {path}")
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"PASS {path}")
    print(f"\n{len(files) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
