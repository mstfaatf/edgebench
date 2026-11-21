"""Compare our INT8 MobileNetV2 models with Qualcomm AI Hub's w8a8 model on the NPU.

Runs Qualcomm AI Hub's pre-quantized MobileNetV2 (``scripts/get_aihub_baseline.py``)
through exactly the same strict NPU setup as our own models
(``scripts/common/npu_harness.py``): QNN NPU device selected by type, CPU
fallback disabled, placement checked three ways, accuracy on the same
``eval`` split, and cold/cached init and steady-state latency measured in the
same run, rotating the order of the NPU models every round.

The AI Hub model is a different network instance: it was trained from
``tonylins/pytorch-mobilenet-v2`` weights rather than torchvision's, takes a
uint8 RGB image in [0, 255] (it normalizes internally) and returns uint8
logits. It sees the same pixels as our models (shared resize/crop); only the
final tensor formatting differs (see ``uint8_io_adapter``). Because the
weights differ, AI Hub's float ONNX model is also evaluated on the CPU, so
that each quantized model can be compared with its own float model.

Configurations:

- CPU: our FP32, per-tensor INT8, per-channel s8w INT8; AI Hub float, AI Hub w8a8
- NPU (strict): our per-tensor INT8, our per-channel s8w INT8, AI Hub w8a8

The summary (including a graph comparison from ``common/qdq_inspect.py``) is
written to ``benchmarks/results/local/feasibility/aihub_comparison_mobilenetv2.json``.

``--qnn-profile`` additionally runs each NPU model with the QNN EP's detailed
profiler and records per-op HTP cycles (where the time goes on the NPU).

Usage:
    python scripts/compare_aihub.py [--rounds N] [--warmup N] [--runs N] [--cooldown S]
                                    [--qnn-profile]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.cli import add_timing_args, harness_kwargs  # noqa: E402
from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402

OURS = {
    "fp32": "mobilenetv2_fp32_b1.onnx",
    "int8_pt": "mobilenetv2_int8qdq_b1.onnx",
    "int8_pc_s8w": "mobilenetv2_int8qdq_pc_s8w_b1.onnx",
}
REFERENCE = "cpu_fp32"
NPU_ORDER = ["npu_int8_pt", "npu_aihub_w8a8", "npu_int8_pc_s8w"]
# Each quantized model's own float model, for "agreement with own float".
OWN_FLOAT = {
    "cpu_int8_pt": "cpu_fp32", "cpu_int8_pc_s8w": "cpu_fp32",
    "npu_int8_pt": "cpu_fp32", "npu_int8_pc_s8w": "cpu_fp32",
    "cpu_aihub_w8a8": "cpu_aihub_float", "npu_aihub_w8a8": "cpu_aihub_float",
}


def heldout_accuracy(h, cpu_configs: list) -> dict:
    """CPU top-1 on every Imagenette val image that comes from ImageNet val.

    Most Imagenette images are ImageNet train images, which both sets of weights
    were trained on; these are the ones neither has seen. Too few for precise
    numbers, but enough to see whether a gap on the eval split is a train-set effect.
    """
    import numpy as np

    from common import accuracy as acc
    from common.preprocess import IMAGENETTE_ROOT, load_image

    images = sorted(p.relative_to(IMAGENETTE_ROOT).as_posix()
                    for p in (IMAGENETTE_ROOT / "val").glob("*/ILSVRC2012_val_*"))
    labels = acc.labels_for(images)
    unit = [load_image(IMAGENETTE_ROOT / p, normalized=False) for p in images]
    out = {"count": len(images), "selection": "Imagenette val files named ILSVRC2012_val_*",
           "results": {}}
    for cfg in cpu_configs:
        session = h.cpu_session(cfg.model)
        name = session.get_inputs()[0].name
        logits = [cfg.adapter.to_logits(session.run(None, {name: cfg.adapter.make_input(x)})[0])
                  for x in unit]
        out["results"][cfg.key] = acc.accuracy(np.stack(logits), labels)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_timing_args(parser, rounds=3)
    parser.add_argument("--qnn-profile", action="store_true",
                        help="also record per-op HTP cycles with the QNN detailed profiler")
    args = parser.parse_args()

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    import onnx

    from common import accuracy as acc
    from common import ort_qnn
    from common.npu_harness import UNIT_FLOAT, NpuHarness, RunConfig, uint8_io_adapter
    from common.preprocess import REPO_ROOT
    from common.qdq_inspect import describe
    from get_aihub_baseline import model_path as aihub_model

    ours_dir = REPO_ROOT / "models" / "mobilenetv2"
    aihub_w8a8, aihub_float = aihub_model("w8a8"), aihub_model("float")
    missing = [p for p in [*(ours_dir / f for f in OURS.values()), aihub_w8a8, aihub_float]
               if not p.is_file()]
    if missing:
        print("missing models: " + ", ".join(str(p) for p in missing) + "\n"
              "run scripts/export_mobilenetv2.py, scripts/quantize_qnn.py and "
              "scripts/get_aihub_baseline.py first", file=sys.stderr)
        return 1

    u8 = uint8_io_adapter(aihub_w8a8)
    configs = [
        RunConfig("cpu_fp32", "cpu", ours_dir / OURS["fp32"]),
        RunConfig("cpu_int8_pt", "cpu", ours_dir / OURS["int8_pt"]),
        RunConfig("cpu_int8_pc_s8w", "cpu", ours_dir / OURS["int8_pc_s8w"]),
        RunConfig("cpu_aihub_float", "cpu", aihub_float, adapter=UNIT_FLOAT),
        RunConfig("cpu_aihub_w8a8", "cpu", aihub_w8a8, adapter=u8),
        RunConfig("npu_int8_pt", "npu", ours_dir / OURS["int8_pt"], cpu_reference="cpu_int8_pt"),
        RunConfig("npu_int8_pc_s8w", "npu", ours_dir / OURS["int8_pc_s8w"],
                  cpu_reference="cpu_int8_pc_s8w"),
        RunConfig("npu_aihub_w8a8", "npu", aihub_w8a8, adapter=u8, cpu_reference="cpu_aihub_w8a8"),
    ]

    out_path = Path(args.output or REPO_ROOT / "benchmarks" / "results" / "local" / "feasibility"
                    / "aihub_comparison_mobilenetv2.json").resolve()
    h = NpuHarness(out_path.parent / "aihub_comparison_logs", REPO_ROOT / "cache" / "qnn",
                   **harness_kwargs(args))
    try:
        h.setup()
    except ort_qnn.NoNpuDeviceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    results, logits = h.evaluate(configs, REFERENCE)
    for key, own in OWN_FLOAT.items():
        if key in logits and own in logits:
            results[key]["own_float_model"] = own
            results[key]["agreement_with_own_float"] = acc.agreement(logits[key], logits[own],
                                                                     h.labels)
    if "cpu_aihub_float" in logits:
        results["cpu_aihub_float"]["agreement_with_reference_note"] = (
            "different trained weights from the reference; agreement measures weight "
            "differences, not quantization")

    heldout = heldout_accuracy(h, [c for c in configs if c.ep == "cpu"])

    h.time_configs(configs, NPU_ORDER, results, logits, rounds=args.rounds)

    if args.qnn_profile:
        for cfg in configs:
            if cfg.key in NPU_ORDER and cfg.key in logits:
                print(f"\n[{cfg.key}] QNN detailed profile")
                results[cfg.key]["qnn_op_profile"] = prof = h.qnn_op_profile(cfg)
                print(f"  accelerator execute {prof['accelerator_execute_us_median']} us, "
                      f"{prof['htp_op_count']} HTP ops; top: "
                      + ", ".join(f"{o['op']} {100 * o['share']:.1f}%" for o in prof["top_ops"][:4]))

    graphs = {}
    for name, path in [("ours_int8_pt", ours_dir / OURS["int8_pt"]),
                       ("ours_int8_pc_s8w", ours_dir / OURS["int8_pc_s8w"]),
                       ("aihub_w8a8", aihub_w8a8)]:
        graphs[name] = describe(onnx.load(str(path), load_external_data=False),
                                base_dir=str(path.parent))
    sidecar = REPO_ROOT / "models" / "aihub" / "mobilenet_v2_w8a8.json"
    summary = {**h.environment(results),
               "aihub_model": json.loads(sidecar.read_text(encoding="utf-8"))["source"]
               if sidecar.is_file() else None,
               "graph_comparison": graphs, "heldout_imagenet_val_cpu": heldout,
               "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    h.print_tables(results)

    print("\n== Agreement with own float model (1000 / 10) ==")
    for key in OWN_FLOAT:
        a = results.get(key, {}).get("agreement_with_own_float")
        if a:
            print(f"{key:<20} {a['top1_1000']:.3f} / {a['top1_imagenette10']:.3f}  "
                  f"(vs {results[key]['own_float_model']})")
    print(f"\n== Held-out check: {heldout['count']} Imagenette val images from ImageNet val, CPU ==")
    for key, a in heldout["results"].items():
        print(f"{key:<20} top-1 {a['top1_1000']:.3f} / {a['top1_imagenette10']:.3f} "
              f"({a['correct_1000']} / {a['correct_imagenette10']} correct)")
    print("\n== Graphs ==")
    for name, g in graphs.items():
        print(f"{name:<18} nodes {g['node_count']:>4}, compute {g['compute_op_types']}, "
              f"weights {g['weights']['type']} per-channel {g['weights']['per_channel']}, "
              f"activations {g['activation_quantize_types']}, I/O {g['graph_input_type']}"
              f"/{g['graph_output_type']}")
    print(f"wrote {out_path}")

    npu_ok = all(r.get("placement", {}).get("all_nodes_on_qnn")
                 for k, r in results.items() if k.startswith("npu_"))
    return 0 if npu_ok else 1


if __name__ == "__main__":
    sys.exit(main())
