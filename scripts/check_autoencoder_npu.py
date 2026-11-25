"""Check, before training, that the autoencoder designs run fully on the NPU as INT8.

TensorFlow does not run on this machine, so the trained Keras model cannot be
exported here. QNN HTP op support can be checked without it: this script builds
untrained ONNX stand-ins of every candidate layout (``common/autoencoder.py``:
the Conv1D design in three tf2onnx-like layouts, the Dense design as
MatMul + Add and as Gemm) at batch sizes 1, 4 and 8, quantizes them with the
default INT8 recipe (``common/onnx_quantize.py``; calibration on the 200
training windows of the split manifest; batch 4 and 8 reuse the batch-1
ranges), prunes them, and then for every file:

- creates a verbose QNN HTP session in strict mode (CPU fallback disabled) and
  records what the QNN EP validated or rejected (``common/qnn_log.py``);
  if the strict session is refused, a non-strict session shows which nodes
  would fall back;
- runs 32 validation windows on the NPU and on the CPU (FP32 and INT8) and
  records the differences, and whether batch 4 and 8 give the same outputs
  as batch 1 on the NPU.

The weights are random, so the outputs mean nothing; only op support, node
counts and agreement between devices are of interest. Files and logs go to
a new run directory under ``benchmarks/results/local/autoencoder/``.

Usage:
    python scripts/check_autoencoder_npu.py [--layout NAME ...]
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.platform_guard import require_native_arm64  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "local" / "autoencoder"
BATCHES = (1, 4, 8)
CHECK_WINDOWS = 32


def cpu_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def npu_options(log_severity: int, strict: bool):
    from common import ort_qnn

    ort_qnn.register_qnn()
    npu = ort_qnn.select_qnn_npu(ort_qnn.list_devices())
    return ort_qnn.qnn_npu_session_options(
        npu, {"backend_path": str(ort_qnn.htp_backend_path()), "htp_performance_mode": "burst"},
        strict=strict, log_severity=log_severity)


def placement(path: Path, log: Path, strict: bool) -> dict:
    """Verbose session: created or refused, and the placement facts from its log."""
    import onnxruntime as ort

    from common import ort_qnn

    out: dict = {"log_file": log.name}
    with ort_qnn.capture_native_stderr(log):
        try:
            session = ort.InferenceSession(str(path), sess_options=npu_options(0, strict))
            out["session_created"], out["error"] = True, None
        except Exception as exc:  # noqa: BLE001 - recorded and reported
            session, out["session_created"], out["error"] = None, False, f"{type(exc).__name__}: {exc}"
        session = None
        gc.collect()
    report = ort_qnn.parse_placement_log(log.read_text(encoding="utf-8", errors="replace"))
    final = report.qnn_passes[-1] if report.qnn_passes else {}
    out.update({
        "all_nodes_on_qnn": bool(out["session_created"] and report.all_on_qnn),
        "ort_node_placements": report.ort_placements,
        "qnn_validated_nodes": final.get("passed_nodes"),
        "qnn_validated_units": final.get("passed_units"),
        "qnn_failed_nodes": final.get("failed_nodes"),
        "qnn_rejections": report.rejections,
        "unsupported_nodes": report.unsupported_nodes,
    })
    return out


def run_batches(session, x, batch: int):
    import numpy as np

    name = session.get_inputs()[0].name
    return np.concatenate([session.run(None, {name: x[i:i + batch]})[0]
                           for i in range(0, len(x), batch)])


def check_layout(layout: str, data, run_dir: Path) -> dict:
    import numpy as np
    import onnx

    from common import autoencoder as ae
    from common import onnx_quantize as oq
    from common.onnx_prune import prune
    from common.timeseries import evenly_spaced

    design = ae.STANDINS[layout]
    models = run_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    work = run_dir / "work" / layout
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "calibration_ranges.json"
    calibration = [ae.model_inputs(design, data.calibration[i:i + 1])
                   for i in range(len(data.calibration))]
    picks = evenly_spaced(list(range(len(data.windows["validation"]))), CHECK_WINDOWS)
    x = ae.model_inputs(design, data.windows["validation"][picks])

    result: dict = {"layout": layout, "design": design, "batches": {}}
    npu_b1 = None
    for batch in BATCHES:
        fp32 = models / f"{layout}_fp32_b{batch}.onnx"
        onnx.save(ae.build_standin(layout, batch), str(fp32))
        unpruned = work / f"{layout}_int8_unpruned_b{batch}.onnx"
        changed, _ = oq.quantize_qdq(fp32, unpruned,
                                     oq.array_reader(ae.INPUT_NAME, calibration)
                                     if batch == 1 else None, cache, work)
        proto = onnx.load(str(unpruned))
        report = prune(proto)
        int8 = models / f"{layout}_int8_b{batch}.onnx"
        onnx.save(proto, str(int8))

        fp32_out = run_batches(cpu_session(fp32), x, batch)
        cpu_out = run_batches(cpu_session(int8), x, batch)
        entry: dict = {
            "fp32_size_bytes": fp32.stat().st_size,
            "int8_size_bytes": int8.stat().st_size,
            "fp32_ops": dict(sorted(Counter(n.op_type for n in onnx.load(str(fp32)).graph.node).items())),
            "int8_nodes": len(proto.graph.node),
            "int8_ops": dict(sorted(Counter(n.op_type for n in proto.graph.node).items())),
            "qnn_preprocess_changed": changed,
            "pruned": report.as_dict(),
            "cpu_int8_vs_fp32_max_abs": float(np.abs(cpu_out - fp32_out).max()),
        }
        strict = placement(int8, run_dir / "logs" / f"{layout}_b{batch}_strict.log", strict=True)
        entry["strict"] = strict
        if strict["session_created"]:
            import onnxruntime as ort

            session = ort.InferenceSession(str(int8), sess_options=npu_options(3, True))
            npu_out = run_batches(session, x, batch)
            again = run_batches(session, x, batch)
            session = None
            gc.collect()
            entry["npu_vs_cpu_int8_max_abs"] = float(np.abs(npu_out - cpu_out).max())
            entry["npu_vs_fp32_max_abs"] = float(np.abs(npu_out - fp32_out).max())
            entry["npu_repeat_identical"] = bool(np.array_equal(npu_out, again))
            if batch == 1:
                npu_b1 = npu_out
            elif npu_b1 is not None:
                entry["npu_identical_to_b1"] = bool(np.array_equal(npu_out, npu_b1))
                entry["npu_vs_b1_max_abs"] = float(np.abs(npu_out - npu_b1).max())
        else:
            entry["non_strict"] = placement(
                int8, run_dir / "logs" / f"{layout}_b{batch}_non_strict.log", strict=False)
        result["batches"][f"b{batch}"] = entry
    result["all_batches_on_qnn"] = all(b["strict"]["all_nodes_on_qnn"]
                                       for b in result["batches"].values())
    return result


def print_table(results: list[dict]) -> None:
    print(f"\n{'layout':18} {'batch':>5} {'strict':>7} {'on QNN':>7} {'INT8 nodes':>10} "
          f"{'QNN nodes':>9} {'NPU-CPU':>9} {'=b1':>5}  ops")
    for r in results:
        for b, e in r["batches"].items():
            s = e["strict"]
            same = e.get("npu_identical_to_b1")
            print(f"{r['layout']:18} {b:>5} {'ok' if s['session_created'] else 'REFUSED':>7} "
                  f"{str(s['all_nodes_on_qnn']):>7} {e['int8_nodes']:>10} "
                  f"{str(s['qnn_validated_nodes']):>9} "
                  f"{e.get('npu_vs_cpu_int8_max_abs', float('nan')):9.4f} "
                  f"{'-' if same is None else str(same):>5}  "
                  + ", ".join(f"{k} {v}" for k, v in s["qnn_validated_units"].items()
                              if s["qnn_validated_units"]))
            for line in s["qnn_rejections"] + [str(u) for u in s["unsupported_nodes"]]:
                print(f"{'':18}   rejected: {line}")


def main() -> int:
    from common import autoencoder as ae

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", nargs="+", choices=sorted(ae.STANDINS),
                        default=list(ae.STANDINS), help="stand-in layouts to check (default: all)")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    require_native_arm64()

    import onnx
    import onnxruntime as ort

    from common import ort_qnn
    from common.results_io import new_run
    from common.timeseries import load_split_windows

    data = load_split_windows()
    out = new_run(args.output_dir, "npu_architecture_check")
    run_dir = out.make_run_dir()
    results = []
    for layout in args.layout:
        print(f"checking {layout}", flush=True)
        results.append(check_layout(layout, data, run_dir))
    print_table(results)
    summary = {
        "description": "QNN HTP strict-mode check of untrained INT8 autoencoder stand-ins "
                       "(random weights; outputs are not meaningful)",
        "environment": {"onnxruntime": ort.__version__, "onnx": onnx.__version__,
                        **ort_qnn.qnn_versions(), "power": ort_qnn.power_state()},
        "calibration": {"windows": int(len(data.calibration)), "source": "split manifest"},
        "check_windows": CHECK_WINDOWS,
        "run_dir": run_dir.name,
        "layouts": results,
    }
    path = out.write(summary)
    print(f"\nwrote {path}")
    return 0 if all(r["all_batches_on_qnn"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
