"""Check a CPU/NPU matrix (scripts/run_matrix.py, or an `edgebench suite` directory) and print
its sanity table.

MATRIX_DIR is a directory with the plan.json of scripts/run_matrix.py, or a suite
directory with a suite-manifest.json (docs/suite.md); its runs are read as plan
entries (settings per run). For every run:

- the edgebench call exited 0 and its result validates in strict mode
  (``validate_results.py``: schema and consistency rules, warnings fail);
- the result is the planned configuration: model file and SHA256 (models/manifest.json),
  provider, batch size, trials, iterations and warm-up;
- the effective precision: CPU computes what is stored (FP32 or INT8); NPU INT8
  as stored (int8, source model); NPU FP16 is an FP32 file computed in FP16
  (source enable_htp_fp16_precision, recorded in ep_options);
- batch equivalence equals models/batch_equivalence.json for the file, provider,
  compute precision and compile-relevant EP options (batch 1: equal by
  definition), and a batch_not_equivalent warning appears exactly where the
  summary says predictions differ (NPU INT8 image models at batch 8 with the
  default graph finalization; not with --htp-graph-finalization-mode 1);
- the graph finalization option appears in ep_options exactly in the runs that set it;
- every module check found no module outside the bundle, and every node ran on
  the requested provider;
- NPU runs: one QNN partition; the cache entry was built by the run just before
  (a successful --ep-cache build result of the same key) and the timed run
  loaded it (hit, state loaded); cached and compiled sessions gave equal outputs;
  the init breakdown has a cold, a build and a cached value per trial; the
  profiled pass's median accelerator time is not above the median timed latency
  (in a suite, each only for runs whose configuration asks for it: init "cached",
  init_breakdown, profile_accelerator);
- no warning or note outside EXPECTED below.

Every unexpected item is listed and makes the check fail (exit 1). Then the
sanity table (median of the trials' p50 latencies per model, batch and
configuration, NPU accelerator time, cold and cached initialization, the cost of
finalization mode 1) and the implausible cases it flags. These numbers are local
checks of the engine, not published results.

Standard library and jsonschema only; runs anywhere.

Usage:
    python scripts/check_matrix.py MATRIX_DIR [--no-table]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_results as vr  # noqa: E402
from common.model_registry import MODELS_DIR, read_manifest  # noqa: E402

BATCH_EQUIVALENCE_JSON = MODELS_DIR / "batch_equivalence.json"
NOT_COMPILE_RELEVANT = ("backend_path", "htp_performance_mode", "enable_htp_fp16_precision")
FINALIZATION_OPTION = "htp_graph_finalization_optimization_mode"

# (kind, code, path) -> who may carry it. Anything else is unexpected.
EXPECTED = {
    # Neither the CPU EP nor the QNN EP reports a driver version for its device.
    ("warning", "value_unavailable", "/runtime/ep_device/driver_version"): "all runs",
    # The QNN EP does not report the SoC model (docs/npu.md).
    ("warning", "value_unavailable", "/runtime/qnn/soc_model"): "NPU runs",
    # Where models/batch_equivalence.json says the predictions differ from batch 1.
    ("warning", "batch_not_equivalent", "/execution/batch_equivalence/predictions_equal_to_b1"):
        "runs whose batch-equivalence entry has different predictions",
    # Every EP context load: the context stores VTCM 0 (docs/npu.md).
    ("note", "ep_context_recompilation_suggested", "/config/ep_context_cache"): "NPU runs",
    # Where the summary says only the last bits differ.
    ("note", "batch_rounding_only", "/execution/batch_equivalence/bitwise_equal_to_b1"):
        "runs whose batch-equivalence entry differs in bits only",
}


def compile_options(ep_options: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in ep_options.items() if k not in NOT_COMPILE_RELEVANT}


def expected_equivalence(summary: dict, sha: str, provider: str, compute: str,
                         options: dict[str, str], batch: int) -> dict[str, Any] | None:
    """What the engine must record (the fields of execution.batch_equivalence)."""
    if batch == 1:
        return {"bitwise_equal_to_b1": True, "bitwise_basis": "batch_size_1",
                "predictions_equal_to_b1": True, "predictions_basis": "bitwise",
                "predictions_compared": None, "predictions_differing": None}
    for e in summary["entries"]:
        if (e["sha256"], e["provider"], e["compute_precision"], e["ep_compile_options"]) == \
                (sha, provider, compute, options):
            return {"bitwise_equal_to_b1": e["bitwise_equal_to_b1"],
                    "bitwise_basis": "reference_outputs",
                    **{k: e[k] for k in ("predictions_equal_to_b1", "predictions_basis",
                                         "predictions_compared", "predictions_differing")}}
    return None


def load(root: Path, rel: str | None) -> tuple[dict | None, Path | None]:
    if rel is None:
        return None, None
    path = root / rel
    return (json.loads(path.read_text(encoding="utf-8")), path) if path.is_file() else (None, path)


def strict_issues(path: Path) -> list[str]:
    return [str(i) for i in vr.validate_file(path)]  # errors and warnings: strict


def module_problems(doc: dict) -> list[str]:
    details = doc["execution"]["verification_details"]
    if not isinstance(details, dict):
        return ["execution.verification_details has no module checks"]
    checks = details.get("module_checks") or []
    out = [f"modules outside the bundle at {c['stage']}: {c['modules_outside_bundle']}"
           for c in checks if c["modules_outside_bundle"]]
    if not checks:
        out.append("no module checks recorded")
    return out


def check_run(run: dict, root: Path, plan: dict, summary: dict,
              manifest: dict[str, Any]) -> tuple[list[str], dict | None]:
    problems: list[str] = []
    add = problems.append
    if run["exit_code"] != 0:
        add(f"edgebench exited {run['exit_code']}")
    doc, path = load(root, run.get("result"))
    if doc is None:
        add("no result file")
        return problems, None
    problems += [f"validator: {i}" for i in strict_issues(path)]
    if doc["status"] != "success":
        add(f"status {doc['status']}: {doc.get('error')}")
        return problems, doc
    s = run.get("settings") or plan["settings"]
    cfg = doc["config"]
    entry = manifest[run["file"]]
    if doc["model"]["sha256"] != entry.sha256:
        add("model SHA256 is not the manifest's")
    want = {"batch_size": run["batch"], "trials": s["trials"],
            "measured_iterations": s["iterations"], "warmup_iterations": s["warmup"]}
    got = {k: cfg[k] for k in want}
    if got != want:
        add(f"config {got} is not the plan's {want}")
    npu = run["provider"] == "npu"
    ep_name = "QNNExecutionProvider" if npu else "CPUExecutionProvider"
    if (doc["runtime"]["execution_provider"] or {}).get("name") != ep_name:
        add(f"execution provider {doc['runtime']['execution_provider']}")

    # Effective precision.
    ep = doc["execution"]["effective_precision"]
    fp16 = run["compute_precision"] == "fp16"
    want_ep = {"stored": run["stored_precision"], "compute": run["compute_precision"],
               "source": "enable_htp_fp16_precision" if fp16 else "model"}
    if ep != want_ep:
        add(f"effective precision {ep}, expected {want_ep}")
    if fp16 != (cfg["ep_options"].get("enable_htp_fp16_precision") == "1"):
        add("enable_htp_fp16_precision does not match the FP16 configuration")

    # Compile options and batch equivalence.
    options = compile_options(cfg["ep_options"])
    if options != run["ep_compile_options"]:
        add(f"compile-relevant EP options {options}, planned {run['ep_compile_options']}")
    be = doc["execution"]["batch_equivalence"]
    want_be = expected_equivalence(summary, entry.sha256, run["provider"], run["compute_precision"],
                                   options, run["batch"])
    if want_be is None:
        add("the file is not in models/batch_equivalence.json for these options")
    elif {k: be.get(k) for k in want_be} != want_be:
        add(f"batch equivalence {be}, the summary says {want_be}")
    warned = {w["code"] for w in doc["warnings"]}
    noted = {n["code"] for n in doc.get("notes", [])}
    differ = want_be is not None and want_be["predictions_equal_to_b1"] is False
    rounding = (want_be is not None and want_be["bitwise_equal_to_b1"] is False
                and want_be["predictions_equal_to_b1"] is True)
    if ("batch_not_equivalent" in warned) != differ:
        add(f"batch_not_equivalent warning {'missing' if differ else 'unexpected'}")
    if ("batch_rounding_only" in noted) != rounding:
        add(f"batch_rounding_only note {'missing' if rounding else 'unexpected'}")

    # Warnings and notes outside the expected list.
    for kind, items in (("warning", doc["warnings"]), ("note", doc.get("notes", []))):
        for w in items:
            key = (kind, w["code"], w["path"])
            if key not in EXPECTED:
                add(f"unexpected {kind} [{w['code']}] at {w['path']}: {w['message'][:160]}")
            elif w["code"] == "value_unavailable" and w["path"] == "/runtime/qnn/soc_model" \
                    and not npu:
                add("soc_model value_unavailable on a CPU run")
    if npu and run.get("cached", True) and "ep_context_recompilation_suggested" not in noted:
        add("no ep_context_recompilation_suggested note on a run that loads an EP context")

    # Placement and modules.
    if doc["execution"]["full_device_execution_verified"] is not True:
        add("full device execution not verified")
    problems += module_problems(doc)

    if npu:
        qnn = doc["runtime"]["qnn"]
        if qnn["partition_count"] != 1:
            add(f"{qnn['partition_count']} QNN partitions")
        cache = cfg["ep_context_cache"]
    if npu and run.get("cached", True):
        build_doc, _ = load(root, run["cache_build"]["result"])
        if run["cache_build"]["exit_code"] != 0 or build_doc is None:
            add("the cache build run failed")
        else:
            _, build_path = load(root, run["cache_build"]["result"])
            problems += [f"cache build validator: {i}" for i in strict_issues(build_path)]
            bc = build_doc["config"]["ep_context_cache"]
            if bc["state"] not in ("generated", "regenerated") or bc["verified_equal_outputs"] \
                    is not True:
                add(f"cache build state {bc['state']}, verified {bc['verified_equal_outputs']}")
            if bc["key_id"] != cache["key_id"]:
                add("the timed run's cache key is not the one the build run built")
        if (cache["mode"], cache["hit"], cache["state"]) != ("use", True, "loaded"):
            add(f"cache mode {cache['mode']}, hit {cache['hit']}, state {cache['state']}")
        if cache["verified_equal_outputs"] is not True:
            add("cached and compiled outputs not verified equal")
    if npu and run.get("init_breakdown", True):
        init = doc["timing"]["session_init"]["summary"]
        for k in ("cold_ms", "context_build_ms", "cached_init_ms"):
            if not init.get(k) or init[k]["count"] != s["trials"]:
                add(f"init breakdown {k} has no value per trial: {init.get(k)}")
    if npu and run.get("profile_accelerator", True):
        pp = doc["timing"]["profiled_pass"]
        if not pp or not pp["enabled"] or not pp.get("accelerator_ms"):
            add("no profiled pass")
        elif pp["accelerator_ms"]["median_ms"] > doc["timing"]["aggregate"]["stats"]["p50_ms"]:
            add("median accelerator time above the median timed latency")
    return problems, doc


def median_p50(doc: dict) -> float:
    return statistics.median(t["stats"]["p50_ms"] for t in doc["timing"]["trials"])


def table(rows: list[tuple[dict, dict]]) -> list[str]:
    """The sanity table and the implausible cases it shows."""
    by = {(r["model"], r["batch"], r["config"]): d for r, d in rows if d and d["status"] == "success"}
    configs = ["cpu_fp32", "cpu_int8", "npu_int8", "npu_fp16"]
    out = ["", "Sanity table (local checks, NOT published numbers): median of the trials' p50 "
               "latency in ms; NPU: accelerator median (profiled pass), cold / cached init "
               "medians in ms",
           f"{'model':<20}{'b':>3}" + "".join(f"{c:>11}" for c in configs)
           + f"{'npu8 acc':>10}{'fp16 acc':>10}{'npu8 cold/cached':>19}{'fp16 cold/cached':>19}"]
    flags = []
    models = sorted({k[0] for k in by}, key=lambda m: (m == "conv1d_autoencoder", m))
    for m in models:
        for b in (1, 4, 8):
            cells = [by.get((m, b, c)) for c in configs]
            if not any(cells):
                continue
            line = f"{m:<20}{b:>3}" + "".join(
                f"{median_p50(d):>11.3f}" if d else f"{'-':>11}" for d in cells)
            for d in cells[2:]:
                pp = d and d["timing"]["profiled_pass"]
                line += f"{pp['accelerator_ms']['median_ms']:>10.3f}" if pp else f"{'-':>10}"
            for d in cells[2:]:
                init = d and d["timing"]["session_init"]["summary"]
                line += (f"{init['cold_ms']['median_ms']:>11.0f} /{init['cached_init_ms']['median_ms']:>5.0f}"
                         if init and init.get("cold_ms") and init.get("cached_init_ms")
                         else f"{'-':>19}")
            out.append(line)
            cpu32, cpu8, npu8, npu16 = (median_p50(d) if d else None for d in cells)
            image = m != "conv1d_autoencoder"
            if image and npu8 and cpu8 and npu8 > cpu8:
                flags.append(f"{m} b{b}: NPU INT8 ({npu8:.3f} ms) slower than CPU INT8 ({cpu8:.3f})")
            if npu8 and npu16 and npu16 < npu8:
                flags.append(f"{m} b{b}: NPU FP16 ({npu16:.3f} ms) faster than NPU INT8 ({npu8:.3f})")
            if cpu8 and cpu32 and cpu8 > cpu32:
                flags.append(f"{m} b{b}: CPU INT8 ({cpu8:.3f} ms) slower than CPU FP32 ({cpu32:.3f})")
    extra = [(k, d) for k, d in by.items() if k[2] not in configs]
    if extra:
        out.append("")
        for (m, b, c), d in sorted(extra):
            line = f"{m} b{b} {c}: p50 {median_p50(d):.3f} ms"
            base = by.get((m, b, "npu_int8" if c.startswith("npu") and "finalization" in c
                           else c.replace("_folded", "")))
            if c == "npu_int8_finalization1" and base:
                cost = median_p50(d) / median_p50(base) - 1
                line += f" vs default finalization {median_p50(base):.3f} ms ({cost:+.0%})"
            elif base:
                line += f" vs external normalization {median_p50(base):.3f} ms"
            out.append(line)
    out.append("")
    out += ["Implausible:"] + [f"  {f}" for f in flags] if flags else ["Implausible: none flagged"]
    return out


def plan_from_suite(manifest: dict) -> dict:
    """A suite manifest's runs as plan.json entries (settings per run)."""
    runs = []
    for r in manifest["runs"]:
        c = r["configuration"]
        label = c.get("label", r["label"])
        prefix = f"{c['model']}_b{c['batch']}_"
        runs.append({
            "run_id": f"{label}_rep{r['repetition']}", "model": c["model"],
            "config": label[len(prefix):] if label.startswith(prefix) else label,
            "file": c["model_file"], "batch": c["batch"], "provider": c["provider"],
            "stored_precision": c["precision"], "compute_precision": c["compute_precision"],
            "ep_compile_options": c["ep_compile_options"] or {},
            "settings": {"trials": c["trials"], "iterations": c["iterations"],
                         "warmup": c["warmup"]},
            "cached": c["init"] == "cached", "init_breakdown": c["init_breakdown"],
            "profile_accelerator": c["profile_accelerator"],
            "exit_code": r["exit_code"] if r["exit_code"] is not None else -1,
            "result": r["result_file"],
            "cache_build": {"exit_code": (r["cache_build"] or {}).get("exit_code"),
                            "result": (r["cache_build"] or {}).get("result_file")},
        })
    doctor = {}
    for attempt in manifest["preflight"][-1:]:
        for check in attempt["checks"]:
            if check["name"] == "doctor":
                doctor = check["details"].get("checks", {})
    return {"settings": None, "doctor": {"checks": doctor}, "runs": runs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--no-table", action="store_true")
    args = parser.parse_args(argv)
    vr.make_output_safe()
    plan_path = args.matrix_dir / "plan.json"
    suite_path = args.matrix_dir / "suite-manifest.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    elif suite_path.is_file():
        plan = plan_from_suite(json.loads(suite_path.read_text(encoding="utf-8")))
    else:
        print(f"error: neither {plan_path} nor {suite_path} found", file=sys.stderr)
        return 2
    summary = json.loads(BATCH_EQUIVALENCE_JSON.read_text(encoding="utf-8"))
    manifest = {f.file: f for f in read_manifest()[0]}
    rows, failed = [], 0
    for run in plan["runs"]:
        problems, doc = check_run(run, args.matrix_dir, plan, summary, manifest)
        rows.append((run, doc))
        failed += bool(problems)
        print(f"{'FAIL' if problems else 'OK  '} {run['run_id']}")
        for p in problems:
            print(f"     {p}")
    doctor = plan.get("doctor", {}).get("checks", {})
    print(f"\n{len(rows) - failed} of {len(rows)} runs as expected, {failed} with unexpected items; "
          f"doctor before the runs: " + ", ".join(f"{k} {v}" for k, v in doctor.items()
                                                   if k in ("power", "background_load")))
    if not args.no_table:
        print("\n".join(table(rows)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
