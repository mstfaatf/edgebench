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

Each run writes ``benchmarks/results/local/feasibility/npu_check_mobilenetv2_<UTC
timestamp>.json`` (copied to ``npu_check_mobilenetv2_latest.json``), with session
logs and profiles in the matching ``npu_check_mobilenetv2_<timestamp>/`` directory. EP context
caches are written to ``cache/qnn/``. Run it plugged in; the power source and
Windows power mode are recorded.

The machinery is shared with ``scripts/compare_aihub.py`` and lives in
``scripts/common/npu_harness.py``.

Usage:
    python scripts/npu_check.py [--warmup N] [--runs N] [--init-reps N] [--cooldown S]
                                [--rounds N] [--cpu-threads N] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.cli import add_timing_args, harness_kwargs  # noqa: E402
from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402
from common.results_io import FEASIBILITY_DIR, new_run  # noqa: E402

FP32 = "mobilenetv2_fp32_b1.onnx"
INT8 = {
    "int8_pt": "mobilenetv2_int8qdq_b1.onnx",
    "int8_pc_u8w": "mobilenetv2_int8qdq_pc_b1.onnx",
    "int8_pc_s8w": "mobilenetv2_int8qdq_pc_s8w_b1.onnx",
}
CONTROL_MODEL = Path("unpruned") / INT8["int8_pt"]
FP16_OPTIONS = {"enable_htp_fp16_precision": "1"}
REFERENCE = "cpu_fp32"
# Interleave CPU and NPU so slow drifts (thermals, background load) hit both alike.
TIMING_ORDER = ["cpu_fp32", "npu_fp16", "cpu_int8_pt", "npu_int8_pt",
                "cpu_int8_pc_u8w", "npu_int8_pc_u8w", "cpu_int8_pc_s8w", "npu_int8_pc_s8w"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_timing_args(parser)
    args = parser.parse_args()

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    import numpy as np

    from common import accuracy as acc
    from common import ort_qnn
    from common.npu_harness import NpuHarness, RunConfig
    from common.preprocess import REPO_ROOT

    model_dir = REPO_ROOT / "models" / "mobilenetv2"
    missing = [f for f in [FP32, *INT8.values(), str(CONTROL_MODEL)] if not (model_dir / f).is_file()]
    if missing:
        print(f"missing models in {model_dir}: {missing}\n"
              "run scripts/export_mobilenetv2.py and scripts/quantize_qnn.py first", file=sys.stderr)
        return 1

    configs = [
        RunConfig("cpu_fp32", "cpu", model_dir / FP32),
        *[RunConfig(f"cpu_{k}", "cpu", model_dir / f) for k, f in INT8.items()],
        *[RunConfig(f"npu_{k}", "npu", model_dir / f, cpu_reference=f"cpu_{k}")
          for k, f in INT8.items()],
        RunConfig("npu_fp16", "npu", model_dir / FP32, qnn_options=FP16_OPTIONS,
                  cpu_reference="cpu_fp32"),
    ]

    out = new_run(Path(args.output_dir) if args.output_dir else FEASIBILITY_DIR,
                  "npu_check_mobilenetv2")
    h = NpuHarness(out.run_dir, REPO_ROOT / "cache" / "qnn",
                   **harness_kwargs(args))
    try:
        h.setup()
    except ort_qnn.NoNpuDeviceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    results, logits = h.evaluate(configs, REFERENCE)

    print("\n[control] unpruned per-tensor INT8 model, strict")
    control_cfg = RunConfig("control_unpruned_int8_pt", "npu", model_dir / CONTROL_MODEL)
    control = h.place(control_cfg)
    control["model"] = CONTROL_MODEL.as_posix()
    if control["strict_session_created"] and "npu_int8_pt" in logits:
        control_logits, error = h.npu_logits(control_cfg, "control_accuracy")
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

    h.time_configs(configs, TIMING_ORDER, results, logits, rounds=args.rounds)

    summary = {**h.environment({**results, "control": {"placement": control}}),
               "control_unpruned_strict": control, "results": results}
    print("\nloaded ONNX Runtime / QNN modules (with an NPU session alive):")
    for m in summary["dll_paths"]["loaded_modules_with_npu_session"]:
        print(f"  {m}")
    out.write(summary)
    h.print_tables(results)
    print(f"wrote {out.summary} (and {out.latest.name})")

    npu_ok = all(r.get("placement", {}).get("all_nodes_on_qnn")
                 for k, r in results.items() if k.startswith("npu_"))
    return 0 if npu_ok else 1


if __name__ == "__main__":
    sys.exit(main())
