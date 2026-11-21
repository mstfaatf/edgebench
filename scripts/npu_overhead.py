"""Break NPU latency into accelerator time and host-side overhead, C++ vs Python.

Times the per-channel s8w INT8 MobileNetV2 on the NPU (QNN EP, strict mode,
``htp_performance_mode=burst``, cached EP context) in four ways:

- ``cpp_npu_run``: ``tools/ort_probe`` with ``Session::Run`` and Ort::Value inputs,
- ``cpp_npu_iobinding``: ``tools/ort_probe`` with IoBinding and preallocated buffers,
- ``py_npu_run``: Python ``InferenceSession.run`` with a NumPy input,
- ``py_npu_iobinding``: Python IoBinding with preallocated OrtValues,

and, as a sanity check against Python, the s8w and FP32 models on the CPU EP at
4 intra-op threads from both languages. All NPU sessions load the same EP
context model (generated once by ort_probe in ``cache/qnn/``), so they run the
same compiled HTP graph. Every configuration uses the same input tensor (the
first reference input from ``tools/ort_probe/make_reference.py``).

Protocol, as in ``scripts/common/npu_harness.py``: 20 warmup + 200 timed runs per
configuration, 3 rounds with the order rotated each round, a cooldown between
configurations. Medians over rounds are reported.

Accelerator time comes from separate runs with the QNN EP profiler at
``profiling_level=basic`` (profiling slows the host side down a lot, so wall
times from those runs are not used). Host-side overhead is the median wall-clock
p50 minus the median accelerator execute time per inference.

Latency here is model execution only: preprocessing (decode, resize,
normalization) is not timed.

Results go to ``benchmarks/results/local/feasibility/npu_overhead_mobilenetv2.json``.

Usage:
    python scripts/npu_overhead.py [--probe PATH] [--rounds N] [--runs N] [--warmup N]
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.cli import add_timing_args  # noqa: E402
from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402

CPU_THREADS = 4
NPU_CONFIGS = ["cpp_npu_run", "cpp_npu_iobinding", "py_npu_run", "py_npu_iobinding"]
CPU_CONFIGS = ["cpp_cpu_s8w", "py_cpu_s8w", "cpp_cpu_fp32", "py_cpu_fp32"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_timing_args(parser, rounds=3)
    parser.add_argument("--probe", type=Path, default=None, help="ort_probe.exe path")
    parser.add_argument("--profile-runs", type=int, default=200,
                        help="inferences per QNN-profiled run")
    args = parser.parse_args()

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    import numpy as np
    import onnxruntime as ort
    import onnxruntime_qnn

    from common import ort_qnn
    from common.npu_harness import latency_stats, summarize_timing
    from common.preprocess import REPO_ROOT
    from common.qnn_log import parse_qnn_profile_csv

    probe = (args.probe or REPO_ROOT / "build" / "arm64-release" / "bin" / "Release"
             / "ort_probe.exe").resolve()
    ref_dir = REPO_ROOT / "data" / "ort_probe_reference"
    s8w = REPO_ROOT / "models" / "mobilenetv2" / "mobilenetv2_int8qdq_pc_s8w_b1.onnx"
    fp32 = REPO_ROOT / "models" / "mobilenetv2" / "mobilenetv2_fp32_b1.onnx"
    ctx = REPO_ROOT / "cache" / "qnn" / "overhead_s8w_ctx.onnx"
    out_path = Path(args.output or REPO_ROOT / "benchmarks" / "results" / "local" / "feasibility"
                    / "npu_overhead_mobilenetv2.json").resolve()
    work = out_path.parent / "npu_overhead"
    work.mkdir(parents=True, exist_ok=True)
    for p, hint in [(probe, "build the C++ tools (docs/setup.md)"),
                    (ref_dir / "index.tsv", "run tools/ort_probe/make_reference.py"),
                    (s8w, "run scripts/quantize_qnn.py")]:
        if not p.is_file():
            print(f"missing {p}: {hint}", file=sys.stderr)
            return 1

    power = ort_qnn.power_state()
    print(f"power: {power}")
    x = np.fromfile(ref_dir / "inputs" / "normalized" / "00.f32", dtype="<f4").reshape(1, 3, 224, 224)

    def probe_cmd(model: Path, target: str, *extra: str) -> list[str]:
        return [str(probe), "--model", str(model), "--target", target,
                "--threads", str(CPU_THREADS), "--reference-dir", str(ref_dir), *extra]

    def run_probe(cmd: list[str], json_path: Path) -> dict:
        res = subprocess.run(cmd + ["--json", str(json_path)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ort_probe failed ({res.returncode}): {res.stderr[-2000:]}")
        return json.loads(json_path.read_text(encoding="utf-8"))

    # EP context generated once by the C++ probe; every NPU configuration loads it.
    print(f"\ngenerating EP context {ctx.relative_to(REPO_ROOT)} with ort_probe")
    gen = run_probe(probe_cmd(s8w, "npu", "--ep-context", "on", "--context-file", str(ctx)),
                    work / "generate.json")
    print(f"  generation {gen['init']['context_generation_ms']:.0f} ms, "
          f"cached init {gen['init']['cached_ms'][0]:.0f} ms")

    ort_qnn.register_qnn()
    npu = ort_qnn.select_qnn_npu(ort_qnn.list_devices())
    qnn_options = {"backend_path": onnxruntime_qnn.get_qnn_htp_path(), "htp_performance_mode": "burst"}

    def py_session(target: str, model: Path, extra_qnn: dict | None = None, tag: str = "py"):
        if target == "cpu":
            so = ort.SessionOptions()
            so.intra_op_num_threads = CPU_THREADS
            return ort.InferenceSession(str(model), sess_options=so,
                                        providers=["CPUExecutionProvider"])
        so = ort_qnn.qnn_npu_session_options(npu, {**qnn_options, **(extra_qnn or {})})
        so.intra_op_num_threads = CPU_THREADS
        with ort_qnn.capture_native_stderr(work / f"{tag}.log"):
            return ort.InferenceSession(str(model), sess_options=so)

    def py_loop(session, iobinding: bool, warmup: int, runs: int) -> list[float]:
        name = session.get_inputs()[0].name
        if iobinding:
            out = session.get_outputs()[0]
            y = np.empty([d for d in out.shape], dtype=np.float32)
            io = session.io_binding()
            io.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(x))
            io.bind_ortvalue_output(out.name, ort.OrtValue.ortvalue_from_numpy(y))
            call = lambda: session.run_with_iobinding(io)  # noqa: E731
        else:
            feed = {name: x}
            call = lambda: session.run(None, feed)  # noqa: E731
        for _ in range(warmup):
            call()
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter_ns()
            call()
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        return samples

    def measure(key: str, tag: str) -> dict:
        timing = ["--warmup", str(args.warmup), "--runs", str(args.runs)]
        if key.startswith("cpp_npu"):
            mode = "iobinding" if key.endswith("iobinding") else "run"
            r = run_probe(probe_cmd(s8w, "npu", "--ep-context", "on", "--context-file", str(ctx),
                                    "--reuse-context", "--run-mode", mode, *timing),
                          work / f"{tag}.json")
            return {"steady_state": r["latency"], "cached_init_ms": r["init"]["cached_ms"]}
        if key.startswith("cpp_cpu"):
            model = s8w if key.endswith("s8w") else fp32
            r = run_probe(probe_cmd(model, "cpu", *timing), work / f"{tag}.json")
            return {"steady_state": r["latency"], "cold_init_ms": r["init"]["cold_ms"]}
        if key.startswith("py_npu"):
            session = py_session("npu", ctx, tag=tag)
        else:
            session = py_session("cpu", s8w if key.endswith("s8w") else fp32)
        samples = py_loop(session, key.endswith("iobinding"), args.warmup, args.runs)
        session = None
        gc.collect()
        return {"steady_state": latency_stats(samples)}

    # Interleave so CPU and NPU, C++ and Python alternate.
    order = ["cpp_npu_run", "py_cpu_s8w", "py_npu_run", "cpp_cpu_s8w", "cpp_npu_iobinding",
             "py_cpu_fp32", "py_npu_iobinding", "cpp_cpu_fp32"]
    results: dict[str, dict] = {k: {"timing_rounds": []} for k in order}
    print(f"\ntiming: {args.rounds} rounds, {args.warmup} warmup + {args.runs} runs, "
          f"{args.cooldown}s cooldown, order {order} (rotated each round)")
    first = True
    for r in range(args.rounds):
        shift = r % len(order)
        for key in order[shift:] + order[:shift]:
            if not first:
                time.sleep(args.cooldown)
            first = False
            t = measure(key, f"{key}_r{r + 1}")
            t["round"] = r + 1
            # summarize_timing expects cold_init_ms; record what is available.
            t.setdefault("cold_init_ms", [])
            results[key]["timing_rounds"].append(t)
            s = t["steady_state"]
            print(f"  round {r + 1} {key:<18} mean {s['mean_ms']:.4f}  p50 {s['p50_ms']:.4f}  "
                  f"p95 {s['p95_ms']:.4f}  min {s['min_ms']:.4f} ms")
    for key in order:
        results[key]["timing_summary"] = summarize_timing(results[key]["timing_rounds"])

    # Accelerator time per inference from QNN basic profiling, per NPU mode.
    print(f"\nQNN basic profiling, {args.profile_runs} inferences per mode")
    for key in NPU_CONFIGS:
        csv_path = work / f"{key}_qnn_profile.csv"
        csv_path.unlink(missing_ok=True)
        if key.startswith("cpp"):
            mode = "iobinding" if key.endswith("iobinding") else "run"
            run_probe(probe_cmd(s8w, "npu", "--ep-context", "on", "--context-file", str(ctx),
                                "--reuse-context", "--run-mode", mode, "--warmup", "0",
                                "--runs", str(args.profile_runs), "--qnn-profiling", "basic",
                                "--qnn-profile-file", str(csv_path)), work / f"{key}_profile.json")
        else:
            session = py_session("npu", ctx, {"profiling_level": "basic",
                                              "profiling_file_path": str(csv_path)},
                                 tag=f"{key}_profile")
            py_loop(session, key.endswith("iobinding"), 0, args.profile_runs)
            session = None
            gc.collect()
        prof = parse_qnn_profile_csv(csv_path.read_text(encoding="utf-8", errors="replace"))
        prof.pop("op_cycles_median")
        results[key]["qnn_profile_basic"] = prof
        print(f"  {key:<18} accelerator {prof['accelerator_execute_us']} us, QNN accelerator "
              f"{prof['qnn_accelerator_execute_us']} us, RPC {prof['rpc_execute_us']} us, "
              f"QNN execute {prof['qnn_execute_us']} us ({prof['inferences']} inferences)")

    for key in NPU_CONFIGS:
        r = results[key]
        wall_us = r["timing_summary"]["steady_p50_ms_median"] * 1000
        accel_us = r["qnn_profile_basic"]["accelerator_execute_us"]
        r["overhead"] = {"wall_p50_us": wall_us, "accelerator_execute_us": accel_us,
                         "host_overhead_us": wall_us - accel_us,
                         "host_overhead_share": (wall_us - accel_us) / wall_us}

    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "feasibility check; latency numbers are a sanity check, not a benchmark",
        "onnxruntime_python": ort.__version__,
        "onnxruntime_cpp": gen.get("onnxruntime_version"),
        "power_before": power, "power_after": ort_qnn.power_state(),
        "probe": str(probe),
        "model": s8w.relative_to(REPO_ROOT).as_posix(),
        "ep_context_model": ctx.relative_to(REPO_ROOT).as_posix(),
        "qnn_provider_options": qnn_options,
        "strict_session_config": ort_qnn.STRICT_CONFIG,
        "cpu_intra_op_threads": CPU_THREADS,
        "input": "data/ort_probe_reference/inputs/normalized/00.f32",
        "timing_method": {
            "warmup_runs": args.warmup, "timed_runs": args.runs, "rounds": args.rounds,
            "cooldown_s": args.cooldown, "order": order,
            "round_order": "order rotated left by one position per round",
            "summary": "medians over rounds", "percentiles": "nearest-rank",
            "scope": "model execution only (Run call); preprocessing not timed",
            "accelerator_time": "median 'Accelerator (execute) time' from separate runs with "
                                "QNN profiling_level=basic",
            "host_overhead": "median wall p50 minus accelerator execute time",
        },
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n== NPU (s8w, strict, cached EP context): wall vs accelerator (us) ==")
    print(f"{'config':<18} {'wall p50':>9} {'wall mean':>10} {'accel':>7} {'QNN acc':>8} "
          f"{'RPC':>6} {'QNN exec':>9} {'host ovh':>9} {'share':>6}")
    for key in NPU_CONFIGS:
        r = results[key]
        t, p, o = r["timing_summary"], r["qnn_profile_basic"], r["overhead"]
        print(f"{key:<18} {o['wall_p50_us']:>9.1f} {t['steady_mean_ms_median'] * 1000:>10.1f} "
              f"{p['accelerator_execute_us']:>7.1f} {p['qnn_accelerator_execute_us']:>8.1f} "
              f"{p['rpc_execute_us']:>6.1f} {p['qnn_execute_us']:>9.1f} "
              f"{o['host_overhead_us']:>9.1f} {100 * o['host_overhead_share']:>5.1f}%")
    print(f"\n== CPU sanity check ({CPU_THREADS} threads, ms, medians over rounds) ==")
    for key in CPU_CONFIGS:
        t = results[key]["timing_summary"]
        print(f"{key:<18} mean {t['steady_mean_ms_median']:.3f}  p50 {t['steady_p50_ms_median']:.3f}  "
              f"p95 {t['steady_p95_ms_median']:.3f}  min {t['steady_min_ms_median']:.3f}  "
              f"p50 per round {[round(v, 3) for v in t['steady_p50_ms_per_round']]}")
    print(f"\npower: {power.get('power_source')}, mode {power.get('power_mode')}; wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
