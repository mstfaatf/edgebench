"""Check that MobileNetV2 runs entirely on the Snapdragon NPU, and how well.

A feasibility check for the QNN HTP (NPU) backend from Python, not a benchmark.
For each configuration below it

1. **places** the model: the QNN EP device of type NPU is selected explicitly,
   CPU fallback is disabled (``session.disable_cpu_ep_fallback=1``) and the
   session is created with verbose logging, from which the QNN EP's per-node
   validation results and ONNX Runtime's final node placement are parsed. A
   one-image ORT profile confirms which EP executed each node at run time. If
   a strict session cannot be created, the error is recorded, a non-strict
   session is created to show which nodes would fall back, and the script
   moves on;
2. measures **accuracy** on the ``eval`` split (500 images, shared
   preprocessing): top-1 over 1000 classes and over the 10 Imagenette classes,
   agreement with CPU FP32, and for NPU runs agreement with and max abs logit
   difference from the same model on the CPU;
3. measures **timing**: session creation without an EP context cache (cold),
   generation of an EP context model, session creation from that cached
   context (also strict, outputs checked to be identical), and steady-state
   latency of single-image ``Run`` calls (warmup, then timed runs). CPU and
   NPU configurations are interleaved with a cooldown between them. CPU timing
   sessions use a fixed intra-op thread count (``--cpu-threads``, default 4):
   with ORT's default of one thread per core, single-image latency on this
   machine swings by an order of magnitude under light background load.

Configurations: CPU EP with the FP32 model and the three INT8 QDQ models; QNN
NPU with the three INT8 QDQ models; QNN NPU with the FP32 model and
``enable_htp_fp16_precision=1``. As a control, the unpruned per-tensor INT8
model is also placed in strict mode. All NPU sessions use
``htp_performance_mode=burst``.

Results go to ``benchmarks/results/local/feasibility/npu_check_mobilenetv2.json``,
with session logs and profiles next to it in ``npu_check_logs/``. EP context
caches are written to ``cache/qnn/``. Run it plugged in; the power source and
Windows power mode are recorded.

Usage:
    python scripts/npu_check.py [--warmup N] [--runs N] [--init-reps N] [--cooldown S]
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402

MODEL_DIR_REL = Path("models") / "mobilenetv2"
FP32 = "mobilenetv2_fp32_b1.onnx"
INT8 = {
    "int8_pt": "mobilenetv2_int8qdq_b1.onnx",
    "int8_pc_u8w": "mobilenetv2_int8qdq_pc_b1.onnx",
    "int8_pc_s8w": "mobilenetv2_int8qdq_pc_s8w_b1.onnx",
}
CONTROL_MODEL = Path("unpruned") / INT8["int8_pt"]
HTP_PERFORMANCE_MODE = "burst"
FP16_OPTIONS = {"enable_htp_fp16_precision": "1"}
EVAL_SPLIT = "eval"

# key, EP, model file, extra QNN provider options, CPU config of the same model
CONFIGS = [
    ("cpu_fp32", "cpu", FP32, None, None),
    *[(f"cpu_{k}", "cpu", f, None, None) for k, f in INT8.items()],
    *[(f"npu_{k}", "npu", f, {}, f"cpu_{k}") for k, f in INT8.items()],
    ("npu_fp16", "npu", FP32, FP16_OPTIONS, "cpu_fp32"),
]
# Interleave CPU and NPU so slow drifts (thermals, background load) hit both alike.
TIMING_ORDER = ["cpu_fp32", "npu_fp16", "cpu_int8_pt", "npu_int8_pt",
                "cpu_int8_pc_u8w", "npu_int8_pc_u8w", "cpu_int8_pc_s8w", "npu_int8_pc_s8w"]
REFERENCE = "cpu_fp32"
DLL_FILTER = re.compile(r"onnxruntime|qnn", re.IGNORECASE)


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile, as in docs/results-schema.md."""
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def latency_stats(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    mean = sum(s) / len(s)
    std = math.sqrt(sum((x - mean) ** 2 for x in s) / (len(s) - 1)) if len(s) > 1 else 0.0
    return {"count": len(s), "mean_ms": mean, "p50_ms": percentile(s, 50),
            "p95_ms": percentile(s, 95), "min_ms": s[0], "max_ms": s[-1], "stddev_ms": std}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--init-reps", type=int, default=3, help="session creations per init measurement")
    parser.add_argument("--cooldown", type=float, default=5.0, help="seconds between timing configs")
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="intra-op threads for CPU timing sessions (0 = ORT default, all cores); "
                             "the default thread pool is very sensitive to background load")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    import numpy as np
    import onnx
    import onnxruntime as ort
    import onnxruntime_qnn

    from common import accuracy as acc
    from common import ort_qnn
    from common.preprocess import IMAGENETTE_ROOT, REPO_ROOT, load_image, read_split

    model_dir = REPO_ROOT / MODEL_DIR_REL
    out_path = (args.output or REPO_ROOT / "benchmarks" / "results" / "local" / "feasibility"
                / "npu_check_mobilenetv2.json").resolve()
    log_dir = out_path.parent / "npu_check_logs"
    cache_dir = REPO_ROOT / "cache" / "qnn"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)

    missing = [f for f in [FP32, *INT8.values(), str(CONTROL_MODEL)] if not (model_dir / f).is_file()]
    if missing:
        print(f"missing models in {model_dir}: {missing}\n"
              "run scripts/export_mobilenetv2.py and scripts/quantize_qnn.py first", file=sys.stderr)
        return 1

    power = ort_qnn.power_state()
    print(f"power: {power}")
    if power["power_source"] != "ac":
        print("WARNING: not plugged in; timing numbers will not be comparable", file=sys.stderr)

    # ---------------------------------------------------------------- devices
    plugin_path = ort_qnn.register_qnn()
    devices = ort_qnn.list_devices()
    print("\nEP devices:")
    for d in devices:
        info = ort_qnn.describe_device(d)
        print(f"  {info['ep_name']:<22} ep_vendor={info['ep_vendor']:<10} "
              f"type={info['device_type']:<4} vendor={info['device_vendor']!r} "
              f"vendor_id=0x{info['device_vendor_id']:x} device_id=0x{info['device_id']:x} "
              f"metadata={info['device_metadata']}")
    try:
        npu = ort_qnn.select_qnn_npu(devices)
    except ort_qnn.NoNpuDeviceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"selected NPU device: {ort_qnn.describe_device(npu)['device_metadata']}")

    base_qnn_options = {
        "backend_path": onnxruntime_qnn.get_qnn_htp_path(),
        "htp_performance_mode": HTP_PERFORMANCE_MODE,
    }

    def qnn_options(extra: dict | None) -> dict:
        return {**base_qnn_options, **(extra or {})}

    images = read_split(EVAL_SPLIT)
    labels = acc.labels_for(images)
    print(f"\npreprocessing {len(images)} eval images")
    inputs = [load_image(IMAGENETTE_ROOT / p) for p in images]
    fixed_input = inputs[0]

    def run_all(session) -> np.ndarray:
        name = session.get_inputs()[0].name
        return np.stack([session.run(None, {name: x})[0][0] for x in inputs])

    def cpu_session(model: Path, **kw) -> ort.InferenceSession:
        so = ort.SessionOptions()
        for k, v in kw.items():
            setattr(so, k, v)
        return ort.InferenceSession(str(model), sess_options=so, providers=["CPUExecutionProvider"])

    def npu_session(model: Path, extra: dict | None, *, config=None, log_stem: str) -> tuple:
        """Create a quiet, strict QNN NPU session; return (session or None, error or None, log).

        Only session creation is captured to the log; at error severity nothing is logged
        per Run().
        """
        so = ort_qnn.qnn_npu_session_options(npu, qnn_options(extra), strict=True, config=config)
        log = log_dir / f"{log_stem}.log"
        with ort_qnn.capture_native_stderr(log):
            try:
                return ort.InferenceSession(str(model), sess_options=so), None, log
            except Exception as exc:  # noqa: BLE001 - recorded and reported
                return None, f"{type(exc).__name__}: {exc}", log

    def profile_providers(session) -> dict:
        """Run one image with profiling on and count executed nodes per EP."""
        name = session.get_inputs()[0].name
        session.run(None, {name: fixed_input})
        path = Path(session.end_profiling())
        events = json.loads(path.read_text(encoding="utf-8"))
        by_ep: dict[str, int] = {}
        nodes: dict[str, str] = {}
        for e in events:
            if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
                ep = e.get("args", {}).get("provider", "?")
                node = e["name"][: -len("_kernel_time")]
                if node not in nodes:
                    nodes[node] = ep
                    by_ep[ep] = by_ep.get(ep, 0) + 1
        return {"profile_file": path.name, "executed_nodes_by_ep": by_ep, "executed_nodes": nodes}

    def verbose_probe(model: Path, extra: dict | None, stem: str, *, strict=True,
                      profile=True) -> dict:
        """Create a verbose session, inspect it, and release it, all while capturing its log.

        A verbose session logs on every Run(), so it is never used for accuracy or timing.
        """
        so = ort_qnn.qnn_npu_session_options(npu, qnn_options(extra), strict=strict, log_severity=0)
        if profile:
            so.enable_profiling = True
            so.profile_file_prefix = str(log_dir / f"{stem}_profile")
        log = log_dir / f"{stem}.log"
        out: dict = {"log_file": log.name}
        with ort_qnn.capture_native_stderr(log):
            try:
                session = ort.InferenceSession(str(model), sess_options=so)
            except Exception as exc:  # noqa: BLE001 - recorded and reported
                session, out["error"] = None, f"{type(exc).__name__}: {exc}"
            if session is not None:
                # Snapshot while the session is alive: the HTP backend DLLs are unloaded with it.
                out["loaded_modules"] = ort_qnn.loaded_modules(DLL_FILTER)
                if profile:
                    out["profile"] = profile_providers(session)
            out["session_created"] = session is not None
            session = None
            gc.collect()
        report = ort_qnn.parse_placement_log(log.read_text(encoding="utf-8", errors="replace"))
        out.update(report.as_dict())
        out["all_nodes_on_qnn"] = bool(
            out["session_created"] and report.all_on_qnn
            and (not profile or set(out["profile"]["executed_nodes_by_ep"]) == {ort_qnn.QNN_EP}))
        return out

    def place(key: str, model: Path, extra: dict | None) -> dict:
        """Strict placement report; on failure, a non-strict diagnosis of what falls back."""
        result = verbose_probe(model, extra, f"{key}_placement")
        result["strict_session_created"] = result.pop("session_created")
        result["strict_error"] = result.pop("error", None)
        if not result["strict_session_created"]:
            result["non_strict_diagnosis"] = verbose_probe(
                model, extra, f"{key}_placement_nonstrict", strict=False, profile=False)
        return result

    def npu_logits(model: Path, extra: dict | None, stem: str) -> tuple:
        """Quiet strict session over the eval split; return (logits or None, error or None)."""
        session, error, _ = npu_session(model, extra, log_stem=stem)
        if session is None:
            return None, error
        out = run_all(session)
        session = None
        gc.collect()
        return out, None

    # ---------------------------------------------------------------- placement + accuracy
    results: dict[str, dict] = {}
    logits: dict[str, np.ndarray] = {}
    for key, ep, filename, extra, cpu_ref in CONFIGS:
        model = model_dir / filename
        entry = {"execution_provider": "CPUExecutionProvider" if ep == "cpu" else ort_qnn.QNN_EP,
                 "model": filename, "cpu_reference": cpu_ref}
        print(f"\n[{key}] {filename}")
        if ep == "cpu":
            session = cpu_session(model)
        else:
            entry["provider_options"] = qnn_options(extra)
            entry["session_config"] = dict(ort_qnn.STRICT_CONFIG)
            entry["placement"] = p = place(key, model, extra)
            final = p["qnn_get_capability_passes"][-1] if p["qnn_get_capability_passes"] else {}
            print(f"  strict session: {'OK' if p['strict_session_created'] else 'FAILED'}"
                  + (f" ({p['strict_error']})" if p["strict_error"] else ""))
            print(f"  ORT placement: {p['ort_node_placements']}; QNN validation (final pass): "
                  f"{final.get('passed_nodes')} nodes passed, {final.get('failed_nodes')} failed"
                  + (f"; executed: {p['profile']['executed_nodes_by_ep']}" if "profile" in p else ""))
            if p["qnn_rejections"]:
                print("  rejections:\n    " + "\n    ".join(p["qnn_rejections"][:20]))
            if not p["strict_session_created"]:
                results[key] = entry
                continue
        if ep == "cpu":
            logits[key] = run_all(session)
            session = None
        else:
            out, error = npu_logits(model, extra, f"{key}_accuracy")
            if out is None:
                entry["accuracy_error"] = error
                results[key] = entry
                print(f"  accuracy session FAILED: {error}")
                continue
            logits[key] = out
        entry["accuracy"] = acc.accuracy(logits[key], labels)
        gc.collect()
        results[key] = entry

    for key, entry in results.items():
        if key not in logits:
            continue
        entry["agreement_with_cpu_fp32"] = (None if key == REFERENCE else
                                            acc.agreement(logits[key], logits[REFERENCE], labels))
        ref = entry["cpu_reference"]
        if ref and ref in logits:
            entry["agreement_with_cpu_same_model"] = acc.agreement(logits[key], logits[ref], labels)
            entry["max_abs_logit_diff_vs_cpu_same_model"] = acc.max_abs_diff(logits[key], logits[ref])

    print("\n[control] unpruned per-tensor INT8 model, strict")
    control = place("control_unpruned_int8_pt", model_dir / CONTROL_MODEL, {})
    control["model"] = str(CONTROL_MODEL).replace("\\", "/")
    if control["strict_session_created"] and "npu_int8_pt" in logits:
        control_logits, error = npu_logits(model_dir / CONTROL_MODEL, {}, "control_accuracy")
        if control_logits is None:
            control["accuracy_error"] = error
        else:
            control["identical_to_pruned_npu_logits"] = bool(
                np.array_equal(control_logits, logits["npu_int8_pt"]))
            control["max_abs_logit_diff_vs_pruned_npu"] = acc.max_abs_diff(
                control_logits, logits["npu_int8_pt"])
    print(f"  strict session: {'OK' if control['strict_session_created'] else 'FAILED'}; "
          f"placement {control['ort_node_placements']}; "
          f"identical to pruned: {control.get('identical_to_pruned_npu_logits')}")

    dlls = sorted({m for r in [*results.values(), {"placement": control}]
                   for m in r.get("placement", {}).get("loaded_modules", [])}, key=str.lower)
    ort_core = [m for m in dlls if re.search(r"onnxruntime(\.dll|_pybind11_state)", m, re.I)]

    print("\nloaded ONNX Runtime / QNN modules (with an NPU session alive):")
    for m in dlls:
        print(f"  {m}")

    # ---------------------------------------------------------------- timing
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    config_by_key = {c[0]: c for c in CONFIGS}
    timing_order = [k for k in TIMING_ORDER if k in logits]
    print(f"\ntiming ({args.init_reps} session creations per init measurement, "
          f"{args.warmup} warmup + {args.runs} timed runs, {args.cooldown}s cooldown), "
          f"order: {timing_order}")

    def timed_create(factory) -> tuple:
        gc.collect()
        ws0 = ort_qnn.working_set_bytes()
        t0 = time.perf_counter()
        session = factory()
        t1 = time.perf_counter()
        ws1 = ort_qnn.working_set_bytes()
        return session, (t1 - t0) * 1000, ws1 - ws0

    def steady_state(session) -> dict:
        name = session.get_inputs()[0].name
        feed = {name: fixed_input}
        for _ in range(args.warmup):
            session.run(None, feed)
        samples = []
        for _ in range(args.runs):
            t0 = time.perf_counter_ns()
            session.run(None, feed)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        return latency_stats(samples)

    def check_npu(result: tuple) -> object:
        session, error, log = result
        if session is None:
            raise RuntimeError(f"{error} (log: {log.name})")
        return session

    for i, key in enumerate(timing_order):
        if i:
            time.sleep(args.cooldown)
        _, ep, filename, extra, _ = config_by_key[key]
        model = model_dir / filename
        timing: dict = {}
        print(f"\n[{key}] timing")
        try:
            cold_ms, cold_ws = [], []
            session = None
            for rep in range(args.init_reps):
                session = None
                gc.collect()
                if ep == "cpu":
                    session, ms, ws = timed_create(lambda: cpu_session(
                        model, intra_op_num_threads=args.cpu_threads))
                else:
                    session, ms, ws = timed_create(lambda: check_npu(npu_session(
                        model, extra, log_stem=f"{key}_cold_{rep}")))
                cold_ms.append(ms)
                cold_ws.append(ws)
            timing["cold_init_ms"] = cold_ms
            timing["cold_init_working_set_delta_bytes"] = cold_ws
            print(f"  cold init (ms): {[round(x, 1) for x in cold_ms]}")
            timing["steady_state"] = steady_state(session)
            s = timing["steady_state"]
            print(f"  steady state: mean {s['mean_ms']:.3f} ms, p50 {s['p50_ms']:.3f}, "
                  f"p95 {s['p95_ms']:.3f}, min {s['min_ms']:.3f}")
            session = None

            if ep == "npu":
                ctx_path = cache_dir / f"{key}_ctx.onnx"
                ctx_config = {"ep.context_enable": "1", "ep.context_file_path": str(ctx_path)}
                _, gen_ms, _ = timed_create(lambda: check_npu(npu_session(
                    model, extra, config=ctx_config, log_stem=f"{key}_ctx_generate")))
                gc.collect()
                ctx_model = onnx.load(str(ctx_path), load_external_data=False)
                ctx_files = sorted(p.name for p in cache_dir.glob(f"{key}_ctx*"))
                timing["ep_context"] = {
                    "session_config": {**ort_qnn.STRICT_CONFIG, **{
                        "ep.context_enable": "1",
                        "ep.context_file_path": str(ctx_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    }},
                    "generation_session_ms": gen_ms,
                    "files": {f: (cache_dir / f).stat().st_size for f in ctx_files},
                    "context_model_op_types": sorted({n.op_type for n in ctx_model.graph.node}),
                    "context_model_node_count": len(ctx_model.graph.node),
                }
                cached_ms, cached_ws = [], []
                cached = None
                for rep in range(args.init_reps):
                    cached = None
                    gc.collect()
                    cached, ms, ws = timed_create(lambda: check_npu(npu_session(
                        ctx_path, extra, log_stem=f"{key}_cached_{rep}")))
                    cached_ms.append(ms)
                    cached_ws.append(ws)
                timing["cached_init_ms"] = cached_ms
                timing["cached_init_working_set_delta_bytes"] = cached_ws
                cached_logits = run_all(cached)
                cached = None
                gc.collect()
                cached_probe = verbose_probe(ctx_path, extra, f"{key}_cached_placement")
                timing["cached_session"] = {
                    "strict_session_created": cached_probe["session_created"],
                    "error": cached_probe.get("error"),
                    "log_file": cached_probe["log_file"],
                    "ort_node_placements": cached_probe["ort_node_placements"],
                    "executed_nodes_by_ep": cached_probe.get("profile", {}).get("executed_nodes_by_ep"),
                    "all_nodes_on_qnn": cached_probe["all_nodes_on_qnn"],
                    "identical_outputs_to_uncached": bool(np.array_equal(cached_logits, logits[key])),
                    "max_abs_logit_diff_vs_uncached": acc.max_abs_diff(cached_logits, logits[key]),
                }
                print(f"  EP context generation: {gen_ms:.1f} ms; files {ctx_files}; "
                      f"context model ops {timing['ep_context']['context_model_op_types']}")
                print(f"  cached init (ms): {[round(x, 1) for x in cached_ms]}; strict "
                      f"{'OK' if timing['cached_session']['all_nodes_on_qnn'] else 'NOT all on QNN'}; "
                      f"placement {timing['cached_session']['ort_node_placements']}; identical outputs: "
                      f"{timing['cached_session']['identical_outputs_to_uncached']}")
        except Exception as exc:  # noqa: BLE001 - recorded and reported
            timing["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {timing['error']}")
        results[key]["timing"] = timing
        gc.collect()

    power_after = ort_qnn.power_state()

    # ---------------------------------------------------------------- output
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "feasibility check; latency numbers are a sanity check, not a benchmark",
        "versions": {
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "onnxruntime_qnn": next((d.ep_metadata.get("version") for d in devices
                                     if d.ep_name == ort_qnn.QNN_EP), None),
            "numpy": np.__version__,
            "onnx": onnx.__version__,
        },
        "machine": {"processor": platform.processor(), "platform": platform.platform()},
        "power_before": power,
        "power_after": power_after,
        "dll_paths": {"qnn_plugin_registered": plugin_path,
                      "qnn_htp_backend": base_qnn_options["backend_path"],
                      "onnxruntime_core": ort_core,
                      "loaded_modules_with_npu_session": dlls},
        "ep_devices": [ort_qnn.describe_device(d) for d in devices],
        "selected_npu_device": ort_qnn.describe_device(npu),
        "qnn_provider_options_base": base_qnn_options,
        "strict_session_config": ort_qnn.STRICT_CONFIG,
        "dataset": {"split": EVAL_SPLIT, "count": len(images),
                    "images_from_imagenet_val": labels.from_imagenet_val, "note": acc.DATASET_NOTE},
        "metrics": acc.METRIC_DEFINITIONS,
        "timing_method": {
            "init": "wall time of InferenceSession construction, repeated init_reps times",
            "steady_state": "single-image Run() on a fixed preprocessed eval image, "
                            "time.perf_counter_ns around each call",
            "warmup_runs": args.warmup, "timed_runs": args.runs, "init_reps": args.init_reps,
            "cooldown_s": args.cooldown, "order": timing_order,
            "percentiles": "nearest-rank",
            "cpu_timing_session_options": {"intra_op_num_threads": args.cpu_threads},
            "cpu_accuracy_session_options": "defaults",
        },
        "control_unpruned_strict": control,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n== Accuracy (500 eval images) ==")
    print(f"{'config':<18} {'top1/1000':>9} {'top1/10':>8} {'agreeFP32':>10} {'agreeCPU':>9} "
          f"{'maxdiffCPU':>10}")
    for key, r in results.items():
        if "accuracy" not in r:
            print(f"{key:<18} {'FAILED':>9}")
            continue
        a, f, c = r["accuracy"], r.get("agreement_with_cpu_fp32"), r.get("agreement_with_cpu_same_model")
        print(f"{key:<18} {a['top1_1000']:>9.3f} {a['top1_imagenette10']:>8.3f} "
              f"{(f'{f['top1_1000']:.3f}' if f else '-'):>10} "
              f"{(f'{c['top1_1000']:.3f}' if c else '-'):>9} "
              f"{(f'{r['max_abs_logit_diff_vs_cpu_same_model']:.4f}' if c else '-'):>10}")
    print("\n== Timing (sanity check, not a benchmark) ==")
    print(f"{'config':<18} {'cold init ms':>13} {'cached init ms':>15} {'mean':>8} {'p50':>8} "
          f"{'p95':>8} {'min':>8}")
    for key in timing_order:
        t = results[key].get("timing", {})
        if "steady_state" not in t:
            print(f"{key:<18} FAILED: {t.get('error')}")
            continue
        s = t["steady_state"]
        cold = sorted(t["cold_init_ms"])[len(t["cold_init_ms"]) // 2]
        cached = (f"{sorted(t['cached_init_ms'])[len(t['cached_init_ms']) // 2]:.1f}"
                  if "cached_init_ms" in t else "-")
        print(f"{key:<18} {cold:>13.1f} {cached:>15} {s['mean_ms']:>8.3f} {s['p50_ms']:>8.3f} "
              f"{s['p95_ms']:>8.3f} {s['min_ms']:>8.3f}")
    print(f"\n(init: median of {args.init_reps}) power: {power.get('power_source')}, "
          f"mode {power.get('power_mode')}, plan {power.get('power_plan')}")
    print(f"wrote {out_path}")

    npu_ok = all(results[k].get("placement", {}).get("all_nodes_on_qnn") for k in results
                 if k.startswith("npu_"))
    return 0 if npu_ok else 1


if __name__ == "__main__":
    sys.exit(main())
