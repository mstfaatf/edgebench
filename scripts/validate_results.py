"""Validate EdgeBench benchmark result files.

Each file is checked against the schema of the version it declares in
schema_version (schemas/benchmark-result.v<version>.schema.json), then against
consistency rules that JSON Schema cannot express (percentile ordering,
sample-count thresholds, throughput vs. mean latency, trial counts vs. the run
configuration, the raw latency file, for 1.1.0 the profiling, precision,
EP context cache and loaded-module rules, and for 1.2.0 the effective precision,
unsupported node, batch equivalence, cache and profiled-pass rules). See
docs/results-schema.md.

Problems are errors or warnings. A file with errors fails; a file with only
warnings passes unless --strict is given.

Usage:
    python scripts/validate_results.py FILE_OR_DIR [FILE_OR_DIR ...]
    python scripts/validate_results.py --strict benchmarks/results/published

Directories are searched recursively for *.json files. Accuracy results (files
with accuracy_schema_version or anomaly_schema_version, written by
scripts/eval_accuracy.py and scripts/eval_anomaly.py) are reported
as skipped: scripts/validate_accuracy.py checks them. Exits with status 1 if
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
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import jsonschema
from jsonschema.exceptions import best_match

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "schemas"
SCHEMA_FILE_RE = re.compile(r"^benchmark-result\.v(\d+\.\d+\.\d+)\.schema\.json$")

NIL_UUID = "00000000-0000-0000-0000-000000000000"
EXAMPLE_WARNING_CODE = "illustrative_example"
RAW_CSV_HEADER = ["trial_index", "iteration_index", "latency_ms"]

# Minimum sample counts below which a statistic must be null (and above which it must be set).
MIN_COUNT = {"stddev_ms": 2, "p95_ms": 20, "p99_ms": 100}
REL_TOL = 1e-3


ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    path: str  # JSON Pointer into the document, or "" for the whole file
    message: str
    severity: str = ERROR
    code: str = ""  # rule identifier for consistency rules, e.g. "profiled_timing"

    def __str__(self) -> str:
        suffix = f" [{self.code}]" if self.code else ""
        return f"{self.severity}: {self.path or '/'}: {self.message}{suffix}"


def errors(issues: Iterable[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity == ERROR]


def warnings_of(issues: Iterable[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity == WARNING]


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def schema_path(version: str) -> Path:
    return SCHEMA_DIR / f"benchmark-result.v{version}.schema.json"


def supported_versions(schema_dir: Path = SCHEMA_DIR) -> list[str]:
    """Versions that have a schema file, oldest first."""
    found = [m.group(1) for f in schema_dir.glob("benchmark-result.v*.schema.json")
             if (m := SCHEMA_FILE_RE.match(f.name))]
    return sorted(found, key=version_key)


def load_schema(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def make_validator(schema: dict[str, Any]) -> jsonschema.Draft202012Validator:
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


class Validators:
    """Schema validators by schema_version, loaded on first use.

    With `forced_schema`, every document is checked against that schema
    regardless of the version it declares.
    """

    def __init__(self, schema_dir: Path = SCHEMA_DIR, forced_schema: dict[str, Any] | None = None):
        self.schema_dir = schema_dir
        self.versions = supported_versions(schema_dir)
        self._forced = make_validator(forced_schema) if forced_schema is not None else None
        self._cache: dict[str, jsonschema.Draft202012Validator] = {}

    @property
    def forced(self) -> bool:
        return self._forced is not None

    def for_version(self, version: str) -> jsonschema.Draft202012Validator:
        if self._forced is not None:
            return self._forced
        if version not in self._cache:
            self._cache[version] = make_validator(
                load_schema(self.schema_dir / f"benchmark-result.v{version}.schema.json"))
        return self._cache[version]


def pointer(parts: Iterable[Any]) -> str:
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def schema_issues(doc: Any, validators: Validators) -> list[Issue]:
    if not isinstance(doc, dict):
        return [Issue("", "a result document must be a JSON object")]
    version = doc.get("schema_version")
    supported = ", ".join(validators.versions)
    if not validators.forced:
        if not isinstance(version, str):
            what = "missing" if version is None else f"not a string ({version!r})"
            return [Issue("/schema_version", f"schema_version is {what}; supported versions: {supported}")]
        if version not in validators.versions:
            return [Issue("/schema_version",
                          f"unsupported schema_version {version!r}; supported versions: {supported}")]
    validator = validators.for_version(version if isinstance(version, str) else "")

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


def stats_issues(stats: dict[str, Any], batch_size: int | None, base: str) -> list[Issue]:
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
    if batch_size is None:  # unknown in a failed result: nothing to compare with
        return issues
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


def _windows_path_key(path: str) -> str:
    """Normalized form of a Windows path for comparison: separators, extended-length prefix, case."""
    p = path.replace("/", "\\")
    if p.startswith("\\\\?\\UNC\\"):
        p = "\\\\" + p[8:]
    elif p.startswith("\\\\?\\"):
        p = p[4:]
    return str(PureWindowsPath(p)).rstrip("\\").casefold()


def _windows_path_within(path: str, directory: str) -> bool:
    p, d = _windows_path_key(path), _windows_path_key(directory)
    return p == d or p.startswith(d + "\\")


CACHE_STATES_WITH_FILES = ("generated", "loaded", "regenerated")

# Consistency rules for schema 1.1.0 and later: code -> severity (see docs/results-schema.md).
V1_1_RULES = {
    "profiled_timing": ERROR,
    "accelerator_without_accelerator_ep": ERROR,
    "fp16_precision_not_recorded": ERROR,
    "cpu_threads_unset": WARNING,
    "load_not_checked": WARNING,
    "module_outside_bundle": ERROR,
    "device_selection_ambiguous": ERROR,
    "ep_context_files_missing": ERROR,
    "ep_context_state_mismatch": ERROR,
    "session_init_not_applicable": ERROR,
    "qnn_validation_missing": ERROR,
    "rejected_nodes_but_verified": ERROR,
    "missing_1_1_0_key": WARNING,
}

# Consistency rules for schema 1.2.0 and later.
V1_2_RULES = {
    "npu_compute_fp32": ERROR,
    "fp16_compute_without_option": ERROR,
    "cpu_compute_not_stored": ERROR,
    "effective_precision_stored_mismatch": ERROR,
    "unsupported_nodes_mismatch": ERROR,
    "batch_not_equivalent_unflagged": ERROR,
    "batch_not_equivalent_unfounded": ERROR,
    "batch_rounding_note_missing": ERROR,
    "batch_rounding_note_unfounded": ERROR,
    "batch_prediction_counts_mismatch": ERROR,
    "cache_hit_without_cache": ERROR,
    "cache_outputs_not_verified": ERROR,
    "cache_outputs_differ": ERROR,
    "cache_key_missing": ERROR,
    "cache_key_id_mismatch": ERROR,
    "profiled_pass_without_accelerator_ep": ERROR,
    "qnn_partition_count_mismatch": ERROR,
    "missing_1_2_0_key": WARNING,
}

BATCH_NOT_EQUIVALENT_WARNING = "batch_not_equivalent"
BATCH_ROUNDING_NOTE = "batch_rounding_only"
STRICT_REFUSAL_KIND = "strict_mode_refusal"


def _resolve(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    while isinstance(node, dict) and "$ref" in node:
        node = root["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    return node


def _object_schema(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any] | None:
    """The object branch of a schema node (directly, or inside oneOf/anyOf)."""
    node = _resolve(node, root)
    if "properties" in node:
        return node
    for branch in node.get("oneOf", []) + node.get("anyOf", []):
        found = _object_schema(branch, root)
        if found is not None:
            return found
    return None


def _items_schema(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any] | None:
    node = _resolve(node, root)
    if "items" in node:
        return node["items"]
    for branch in node.get("oneOf", []) + node.get("anyOf", []):
        found = _items_schema(branch, root)
        if found is not None:
            return found
    return None


def missing_keys(doc: Any, schema: dict[str, Any], root: dict[str, Any],
                 path: tuple = ()) -> list[str]:
    """JSON Pointers of every key the schema defines that `doc` omits (null values count
    as present; the contents of null and free-form maps are not inspected)."""
    found: list[str] = []
    if isinstance(doc, dict):
        obj = _object_schema(schema, root)
        if obj is None:
            return found
        for key, sub in obj["properties"].items():
            if key not in doc:
                found.append(pointer(path + (key,)))
            else:
                found += missing_keys(doc[key], sub, root, path + (key,))
    elif isinstance(doc, list):
        items = _items_schema(schema, root)
        if items is not None:
            for i, value in enumerate(doc):
                found += missing_keys(value, items, root, path + (i,))
    return found


_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def _schema_of_version(version: str) -> dict[str, Any]:
    if version not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[version] = load_schema(schema_path(version))
    return _SCHEMA_CACHE[version]


def missing_key_issues(doc: dict[str, Any], schema: dict[str, Any]) -> list[Issue]:
    """The C++ engine writes every key of the version it declares; other writers may
    omit keys added in a minor version (they mean null). Each missing key is reported
    with the version that added it (missing_1_1_0_key, missing_1_2_0_key)."""
    if doc["tool"].get("implementation") != "cpp":
        return []
    missing = missing_keys(doc, schema, schema)
    from_1_1 = set(missing)
    if version_key(doc["schema_version"]) >= (1, 2, 0):
        older = _schema_of_version("1.1.0")
        from_1_1 = set(missing_keys(doc, older, older))
    issues = []
    for p in missing:
        version, code = ("1.1.0", "missing_1_1_0_key") if p in from_1_1 else ("1.2.0", "missing_1_2_0_key")
        severity = V1_1_RULES.get(code) or V1_2_RULES[code]
        issues.append(Issue(p, f"key defined by the {version} schema is missing; the C++ engine "
                               "writes every key", severity, code))
    return issues


def v1_1_issues(doc: dict[str, Any]) -> list[Issue]:
    """Rules for fields added in schema 1.1.0. Fields that are absent are not checked."""
    issues: list[Issue] = []

    def report(path: str, message: str, code: str) -> None:
        issues.append(Issue(path, message, V1_1_RULES[code], code))

    config, timing, runtime, execution = doc["config"], doc["timing"], doc["runtime"], doc["execution"]
    requested = config["requested"]
    device = requested["device_type"]
    timed = timing is not None and len(timing["trials"]) > 0
    provider = runtime.get("execution_provider") if runtime is not None else None
    ep_name = provider["name"] if provider is not None else None

    # Profiled sessions are never timed.
    profiling = config.get("profiling")
    if timed and profiling is not None and (profiling["ort"] or profiling["qnn_level"] != "off"):
        report("/config/profiling",
              f"timing.trials were measured with profiling on (ort={profiling['ort']}, "
              f"qnn_level={profiling['qnn_level']!r}); profiled sessions must not be timed. "
              "Accelerator time from a separate profiled pass goes in timing.accelerator",
              "profiled_timing")

    # Accelerator time needs an accelerator.
    if timing is not None and timing.get("accelerator") is not None and (
            device == "CPU" or ep_name == "CPUExecutionProvider"):
        report("/timing/accelerator",
              f"accelerator timing recorded for a {ep_name or device} run; it requires an accelerator "
              "execution provider", "accelerator_without_accelerator_ep")

    # FP16 on the NPU must be requested and recorded as such.
    if (device == "NPU" and config["ep_options"].get("enable_htp_fp16_precision") == "1"
            and requested["compute_precision"] != "fp16"):
        report("/config/requested/compute_precision",
              f"enable_htp_fp16_precision is set, so the compute precision must be recorded as 'fp16' "
              f"(got {requested['compute_precision']!r})", "fp16_precision_not_recorded")

    # CPU latency depends strongly on the thread count.
    if timed and device == "CPU" and config["intra_op_threads"] is None:
        report("/config/intra_op_threads",
                "timed CPU run with intra_op_threads left at the ONNX Runtime default; "
                "set and record the thread count explicitly", "cpu_threads_unset")

    # ONNX Runtime loading.
    load_check = runtime["onnxruntime"].get("load_check") if runtime is not None else None
    if doc["tool"].get("implementation") == "cpp" and load_check == "not_checked":
        report("/runtime/onnxruntime/load_check",
                "C++ engine result without a verified ONNX Runtime load", "load_not_checked")
    # A run that stopped because of a foreign module records that module on purpose.
    error = doc.get("error") or {}
    stopped_for_module = doc["status"] == "failed" and error.get("code") == "module_outside_bundle"
    if load_check == "full_path_verified" and not stopped_for_module:
        library = runtime["onnxruntime"]["library_path"]
        if library is None:
            report("/runtime/onnxruntime/library_path",
                  "load_check 'full_path_verified' requires the loaded library path", "module_outside_bundle")
        else:
            bundle = str(PureWindowsPath(library.replace("/", "\\")).parent)
            for i, module in enumerate(runtime.get("loaded_modules") or []):
                if not _windows_path_within(module["path"], bundle):
                    report(f"/runtime/loaded_modules/{i}/path",
                          f"{module['name']} was loaded from outside the bundle directory {bundle}",
                          "module_outside_bundle")

    # Device selection.
    ep_device = runtime["ep_device"] if runtime is not None else None
    selection = ep_device.get("selection") if ep_device is not None else None
    if selection is not None and doc["status"] == "success" and selection["candidate_count"] != 1:
        report("/runtime/ep_device/selection/candidate_count",
              f"{selection['candidate_count']} devices matched; a run must select exactly one",
              "device_selection_ambiguous")

    # EP context cache.
    cache = config["ep_context_cache"]
    cache_on = cache is not None and cache["enabled"]
    state = cache.get("state") if cache is not None else None
    if state is not None:
        if state in CACHE_STATES_WITH_FILES and not cache.get("files"):
            report("/config/ep_context_cache/files",
                  f"state {state!r} but no cache files are listed", "ep_context_files_missing")
        if (state == "disabled") == cache["enabled"]:
            report("/config/ep_context_cache/state",
                  f"state {state!r} contradicts enabled={cache['enabled']}", "ep_context_state_mismatch")
        if (state == "loaded" and timing is not None
                and timing["session_init"]["ep_context_state"] not in ("cached", None)):
            report("/timing/session_init/ep_context_state",
                  f"must be 'cached' when the cache was loaded (got "
                  f"{timing['session_init']['ep_context_state']!r})", "ep_context_state_mismatch")
    if timing is not None:
        init = timing["session_init"]
        # --init-breakdown (1.2.0) builds and loads per-trial contexts with the cache off.
        contexts_used = cache_on or config.get("init_breakdown") is True
        if init.get("context_build_ms") is not None and not cache_on:
            report("/timing/session_init/context_build_ms",
                  "must be null when no EP context cache is enabled", "session_init_not_applicable")
        if init.get("cached_ms") is not None and not contexts_used:
            report("/timing/session_init/cached_ms",
                  "must be null when no EP context was used", "session_init_not_applicable")
        # Per-trial values (1.2.0) follow the same rule.
        for i, trial in enumerate(timing["trials"]):
            for key in ("context_build_ms", "cached_init_ms"):
                if (trial.get("session_init") or {}).get(key) is not None and not contexts_used:
                    report(f"/timing/trials/{i}/session_init/{key}",
                          "must be null when no EP context was used",
                          "session_init_not_applicable")
        if init.get("context_build_ms") is not None and state == "loaded":
            report("/timing/session_init/context_build_ms",
                  "must be null when an existing cache was loaded (nothing was built)",
                  "session_init_not_applicable")

    # Placement evidence.
    if execution is not None:
        validation = execution.get("qnn_validation")
        if execution["verification_method"] == "qnn_validation_log" and validation is None:
            report("/execution/qnn_validation",
                  "verification_method 'qnn_validation_log' requires execution.qnn_validation",
                  "qnn_validation_missing")
        if validation is not None and execution["full_device_execution_verified"] is True and (
                validation["rejected_nodes"] or any(p["failed_nodes"] for p in validation["passes"])):
            report("/execution/full_device_execution_verified",
                  "true although execution.qnn_validation reports rejected nodes",
                  "rejected_nodes_but_verified")
    return issues


def v1_2_issues(doc: dict[str, Any]) -> list[Issue]:
    """Rules for fields added in schema 1.2.0. Fields that are absent are not checked."""
    issues: list[Issue] = []

    def report(path: str, message: str, code: str) -> None:
        issues.append(Issue(path, message, V1_2_RULES[code], code))

    config, timing, runtime, execution = doc["config"], doc["timing"], doc["runtime"], doc["execution"]
    device = config["requested"]["device_type"]
    provider = runtime.get("execution_provider") if runtime is not None else None
    ep_name = provider["name"] if provider is not None else None
    cpu_run = device == "CPU" or ep_name == "CPUExecutionProvider"
    execution = execution or {}

    # Effective precision.
    precision = execution.get("effective_precision")
    if precision is not None:
        base = "/execution/effective_precision"
        stored, compute, source = precision["stored"], precision["compute"], precision["source"]
        if doc["model"] is not None and stored != doc["model"]["stored_precision"]:
            report(f"{base}/stored", f"stored {stored!r} differs from model.stored_precision "
                                     f"{doc['model']['stored_precision']!r}",
                   "effective_precision_stored_mismatch")
        if device == "NPU" and compute == "fp32":
            report(f"{base}/compute", "the NPU has no FP32 compute; an FP32 model runs in FP16 "
                                      "(enable_htp_fp16_precision) or not at all", "npu_compute_fp32")
        if compute == "fp16" and stored == "fp32" and (
                source != "enable_htp_fp16_precision"
                or config["ep_options"].get("enable_htp_fp16_precision") != "1"):
            report(f"{base}/source", "FP16 compute of an FP32 model requires source "
                                     "'enable_htp_fp16_precision' and that EP option set to '1' "
                                     f"(source {source!r}, option "
                                     f"{config['ep_options'].get('enable_htp_fp16_precision')!r})",
                   "fp16_compute_without_option")
        if device == "CPU" and compute != stored:
            report(f"{base}/compute", f"a CPU run computes in the stored precision ({stored!r}), "
                                      f"not {compute!r}", "cpu_compute_not_stored")

    # Unsupported nodes: exactly on strict-mode refusals.
    error = doc.get("error")
    if error is not None:
        refusal = error.get("kind") == STRICT_REFUSAL_KIND
        nodes = error.get("unsupported_nodes")
        if refusal and nodes is None:
            report("/error/unsupported_nodes", "a strict-mode refusal must list the unsupported "
                                               "nodes (an empty list if none were named)",
                   "unsupported_nodes_mismatch")
        if not refusal and nodes is not None:
            report("/error/unsupported_nodes", f"must be null unless error.kind is "
                                               f"{STRICT_REFUSAL_KIND!r} (got {error.get('kind')!r})",
                   "unsupported_nodes_mismatch")

    # Batch equivalence: a warning if predictions differ from batch 1, a note if only
    # the last bits do.
    equivalence = execution.get("batch_equivalence")
    warned = any(w["code"] == BATCH_NOT_EQUIVALENT_WARNING for w in doc["warnings"])
    noted = any(n["code"] == BATCH_ROUNDING_NOTE for n in doc.get("notes", []))
    equivalence = equivalence or {}
    predictions_differ = equivalence.get("predictions_equal_to_b1") is False
    rounding_only = (equivalence.get("bitwise_equal_to_b1") is False
                     and equivalence.get("predictions_equal_to_b1") is True)
    base = "/execution/batch_equivalence"
    if predictions_differ and not warned:
        report(f"{base}/predictions_equal_to_b1",
               f"a result whose predictions differ from batch 1 needs a "
               f"'{BATCH_NOT_EQUIVALENT_WARNING}' warning", "batch_not_equivalent_unflagged")
    if warned and not predictions_differ:
        report(f"{base}/predictions_equal_to_b1",
               f"a '{BATCH_NOT_EQUIVALENT_WARNING}' warning requires predictions_equal_to_b1 "
               "false", "batch_not_equivalent_unfounded")
    if rounding_only and not noted:
        report(f"{base}/bitwise_equal_to_b1",
               "outputs that differ from batch 1 only in their bits (same predictions) need a "
               f"'{BATCH_ROUNDING_NOTE}' note", "batch_rounding_note_missing")
    if noted and not rounding_only:
        report(f"{base}/bitwise_equal_to_b1",
               f"a '{BATCH_ROUNDING_NOTE}' note requires bitwise_equal_to_b1 false and "
               "predictions_equal_to_b1 true", "batch_rounding_note_unfounded")
    differing = equivalence.get("predictions_differing")
    equal = equivalence.get("predictions_equal_to_b1")
    if isinstance(differing, int) and isinstance(equal, bool) and equal != (differing == 0):
        report(f"{base}/predictions_differing",
               f"predictions_equal_to_b1 is {equal} but {differing} prediction(s) differ",
               "batch_prediction_counts_mismatch")

    # EP context cache.
    cache = config["ep_context_cache"]
    if cache is not None:
        if cache.get("hit") is True and not cache["enabled"]:
            report("/config/ep_context_cache/hit", "a cache hit requires an enabled cache",
                   "cache_hit_without_cache")
        if (cache.get("mode") in ("build", "use") and doc["status"] == "success"
                and cache.get("verified_equal_outputs") is None):
            report("/config/ep_context_cache/verified_equal_outputs",
                  "a run that builds or uses a cache must record whether cached and uncached "
                  "sessions give the same outputs", "cache_outputs_not_verified")
        if cache.get("verified_equal_outputs") is False and doc["status"] == "success":
            report("/config/ep_context_cache/verified_equal_outputs",
                   "cached and uncached sessions gave different outputs; such a run must fail",
                   "cache_outputs_differ")
        if cache.get("mode") in ("build", "use") and (
                cache.get("key_id") is None or cache.get("key") is None) and \
                doc["status"] == "success":
            report("/config/ep_context_cache/key_id",
                   "a run that builds or uses a cache must record its key", "cache_key_missing")
        key = cache.get("key")
        if cache.get("key_id") is not None and key is not None and "session_config" in key:
            # A key with every 1.2.0 key field: its id can be recomputed.
            canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != cache["key_id"]:
                report("/config/ep_context_cache/key_id",
                       "is not the SHA256 of the key object serialized with sorted keys and no "
                       "whitespace", "cache_key_id_mismatch")
            model = doc.get("model")
            if model is not None and key["model_sha256"] != model["sha256"]:
                report("/config/ep_context_cache/key/model_sha256", "differs from model.sha256",
                       "cache_key_id_mismatch")
            if key.get("ep_options") is not None and key["ep_options"] != config["ep_options"]:
                report("/config/ep_context_cache/key/ep_options", "differs from config.ep_options",
                       "cache_key_id_mismatch")

    # The profiled pass measures an accelerator.
    profiled = timing.get("profiled_pass") if timing is not None else None
    if profiled is not None and profiled["enabled"] and cpu_run:
        report("/timing/profiled_pass", f"profiled accelerator pass recorded for a "
                                        f"{ep_name or device} run; it requires an accelerator "
                                        "execution provider", "profiled_pass_without_accelerator_ep")

    # QNN partitions: the QNN EP's node count in the placement.
    qnn = runtime.get("qnn") if runtime is not None else None
    assignment = execution.get("node_assignment")
    if qnn is not None and qnn.get("partition_count") is not None and assignment is not None:
        placed = assignment["nodes_by_ep"].get("QNNExecutionProvider", 0)
        if qnn["partition_count"] != placed:
            report("/runtime/qnn/partition_count",
                  f"{qnn['partition_count']} partitions, but node_assignment places {placed} "
                  "nodes on QNNExecutionProvider", "qnn_partition_count_mismatch")
    return issues


def consistency_issues(doc: dict[str, Any], result_path: Path | None = None,
                       schema: dict[str, Any] | None = None) -> list[Issue]:
    issues = []
    if doc["run_id"] == NIL_UUID and not any(w["code"] == EXAMPLE_WARNING_CODE for w in doc["warnings"]):
        issues.append(Issue("/run_id", f"the nil UUID is reserved for examples with a '{EXAMPLE_WARNING_CODE}' warning"))
    issues += timing_issues(doc)
    issues += raw_issues(doc, result_path)
    if version_key(doc["schema_version"]) >= (1, 1, 0):
        issues += v1_1_issues(doc)
        if version_key(doc["schema_version"]) >= (1, 2, 0):
            issues += v1_2_issues(doc)
        if schema is not None:
            issues += missing_key_issues(doc, schema)
    return issues


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def validate_document(doc: Any, validators: Validators | None = None,
                      result_path: Path | None = None) -> list[Issue]:
    """Return all errors and warnings for a parsed result document (empty list if clean)."""
    validators = validators if validators is not None else Validators()
    issues = schema_issues(doc, validators)
    if issues:
        return issues  # consistency rules assume a schema-valid document
    schema = validators.for_version(doc["schema_version"]).schema
    return consistency_issues(doc, result_path, schema)


def validate_file(path: Path, validators: Validators | None = None) -> list[Issue]:
    try:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, UnicodeDecodeError) as exc:
        return [Issue("", f"cannot read file: {exc}")]
    except json.JSONDecodeError as exc:
        return [Issue("", f"invalid JSON: {exc}")]
    return validate_document(doc, validators, path)


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


def is_accuracy_document(path: Path) -> bool:
    """An accuracy result (scripts/eval_accuracy.py), which has its own schema and validator."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and ("accuracy_schema_version" in doc
                                      or "anomaly_schema_version" in doc)


def make_output_safe() -> None:
    """Never fail on printing: replace characters the console encoding cannot show.

    On Windows, stdout of a child process (or a console with a legacy code page)
    may use a code page such as cp1252, which cannot encode every path.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, metavar="FILE_OR_DIR")
    parser.add_argument("--strict", action="store_true", help="fail on warnings as well as errors")
    parser.add_argument("--schema", type=Path,
                        help="validate every file against this schema file instead of the one "
                             "matching its schema_version")
    make_output_safe()
    args = parser.parse_args(argv)

    try:
        files = collect_files(args.paths)
    except FileNotFoundError as exc:
        print(f"error: no such file or directory: {exc}", file=sys.stderr)
        return 2
    if not files:
        print("No JSON files found.")
        return 0

    validators = Validators(forced_schema=load_schema(args.schema) if args.schema else None)
    failed = warned = skipped = 0
    for path in files:
        if is_accuracy_document(path):
            skipped += 1
            print(f"SKIP {path} (accuracy result: checked by scripts/validate_accuracy.py)")
            continue
        issues = validate_file(path, validators)
        n_errors, n_warnings = len(errors(issues)), len(warnings_of(issues))
        warned += n_warnings > 0
        fails = n_errors > 0 or (args.strict and n_warnings > 0)
        failed += fails
        status = "FAIL" if fails else ("WARN" if n_warnings else "PASS")
        print(f"{status} {path}")
        for issue in sorted(issues, key=lambda i: i.severity != ERROR):
            print(f"  {issue}")
    mode = " (strict: warnings fail)" if args.strict else ""
    print(f"\n{len(files) - failed - skipped} passed, {failed} failed, {warned} with warnings"
          + (f", {skipped} skipped" if skipped else "") + mode)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
