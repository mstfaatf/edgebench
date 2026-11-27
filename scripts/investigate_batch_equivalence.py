"""Why the NPU INT8 image models compute different outputs at batch 8 than at batch 1.

Regenerates the evidence ``docs/npu.md`` ("Batch size and INT8 outputs") cites
and writes it to ``docs/findings/npu_int8_batch_size.json``. Nothing under
``models/`` is changed: the extra batch sizes are built in a work directory
(default ``tmp/batch_investigation/``, not committed).

Steps (QNN HTP, strict mode, burst, as ``make_references.py``):

1. Threshold. INT8 files (external normalization, pruned) at batch sizes 2, 3,
   5, 6, 7 and 16 are built with the shipped recipe (``common/onnx_quantize.py``;
   the batch-1 calibration ranges are reused, so every batch size has the shipped
   scales and zero points; checked, as is that the b1 and b4 graphs come out
   byte-identical to the shipped files). Each batch size runs the 8 reference
   images (repeated to fill the batch) and every item is compared bit for bit
   with the batch-1 reference output. The largest uint8 activation of each
   graph is listed next to it (the device's VTCM is 8 MiB).
2. Where it starts. ResNet-18: every activation (the output of each
   activation DequantizeLinear) is made the end of a truncated model, run at b1
   and at the first diverging batch size and compared in units of its
   quantization scale. All three models: sub-models of the classifier tail
   (GlobalAveragePool, + Flatten, + Gemm, and the Gemm alone) at b1 to b8, fed
   with uint8 activations.
3. Magnitude. Output differences in output LSBs; the final Gemm emulated in
   exact integer arithmetic, and the NPU outputs at b1 and b8 compared with the
   exactly rounded result.
4. Options. ``vtcm_mb``, ``htp_graph_finalization_optimization_mode`` and, as
   a control, ``htp_performance_mode`` at b1, b6, b7 and b8; the option that
   makes b8 equal to b1 is checked on all 1,000 eval-quick images, and its
   latency cost measured (interleaved, 10 rounds of 50 runs, medians; a quick
   check, not a benchmark). The QNN graph the execution provider hands the
   backend (``dump_json_qnn_graph``) is compared at b6 and b7.

Takes about 20 minutes. Latencies and the environment block change between
runs; every other value is deterministic.

Usage:
    python scripts/investigate_batch_equivalence.py [--work-dir DIR] [--output FILE]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common.model_registry import MODELS, MODELS_DIR, relative_path  # noqa: E402

OUTPUT = REPO_ROOT / "docs" / "findings" / "npu_int8_batch_size.json"
WORK_DIR = REPO_ROOT / "tmp" / "batch_investigation"
FINDINGS_VERSION = "1.0.0"
IMAGE_MODELS = ("mobilenetv2", "resnet18", "efficientnet_lite0")
SHIPPED_BATCHES = (1, 4, 8)
EXTRA_BATCHES = (2, 3, 5, 6, 7, 16)
THRESHOLD_BATCHES = (1, 2, 3, 4, 5, 6, 7, 8, 16)
TAIL_BATCHES = (1, 2, 4, 6, 7, 8)
OPTION_BATCHES = (1, 6, 7, 8)
SCAN_MODEL = "resnet18"
VTCM_BYTES = 8 * 1024 * 1024
GEMM_VECTORS = 64
FIN1 = {"htp_graph_finalization_optimization_mode": "1"}
OPTION_CONFIGS = {
    "default": {},
    "vtcm_mb=2": {"vtcm_mb": "2"},
    "vtcm_mb=4": {"vtcm_mb": "4"},
    "vtcm_mb=8": {"vtcm_mb": "8"},
    "htp_graph_finalization_optimization_mode=0": {"htp_graph_finalization_optimization_mode": "0"},
    "htp_graph_finalization_optimization_mode=1": FIN1,
    "htp_graph_finalization_optimization_mode=2": {"htp_graph_finalization_optimization_mode": "2"},
    "htp_graph_finalization_optimization_mode=3": {"htp_graph_finalization_optimization_mode": "3"},
    "htp_performance_mode=balanced": {"htp_performance_mode": "balanced"},
    "htp_performance_mode=power_saver": {"htp_performance_mode": "power_saver"},
    "htp_performance_mode=high_performance": {"htp_performance_mode": "high_performance"},
}
LATENCY_ROUNDS, LATENCY_WARMUP, LATENCY_RUNS = 10, 10, 50


def log(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------------------------- models and sessions
def model_path(work: Path, name: str, batch: int) -> Path:
    if batch in SHIPPED_BATCHES:
        return MODELS_DIR / relative_path(name, "int8", batch)
    return work / "models" / f"{name}_int8_b{batch}.onnx"


def build_models(work: Path, name: str) -> dict[str, Any]:
    """INT8 files at the extra batch sizes, made with the shipped recipe."""
    import onnx

    from common import onnx_export as ox
    from common import onnx_quantize as oq
    from common.onnx_prune import prune

    spec = MODELS[name]
    torch_model = ox.load_torch_model(spec, download=False)
    (work / "models").mkdir(parents=True, exist_ok=True)
    reproduced = []
    protos = {}
    with tempfile.TemporaryDirectory(prefix="edgebench-batch-") as tmp_dir:
        tmp = Path(tmp_dir)
        cache = tmp / "calibration_ranges.json"
        for batch in (1, 4, *EXTRA_BATCHES):
            fp32 = tmp / f"fp32_b{batch}.onnx"
            fp32.write_bytes(ox.export_fp32(spec, batch, torch_model))
            stripped = tmp / f"stripped_b{batch}.onnx"
            oq.strip_metadata(fp32, stripped)
            quantized = tmp / f"int8_b{batch}.onnx"
            reader = oq._reader(spec, oq.calibration_entries(), True) if batch == 1 else None
            oq.quantize_qdq(stripped, quantized, reader, cache, tmp)
            proto = onnx.load(str(quantized))
            prune(proto)
            protos[batch] = proto
            if batch in SHIPPED_BATCHES:
                shipped = onnx.load(str(MODELS_DIR / relative_path(name, "int8", batch)))
                if oq.graph_without_metadata(proto) == oq.graph_without_metadata(shipped):
                    reproduced.append(batch)
            else:
                onnx.save(proto, str(model_path(work, name, batch)))
    shipped_b1 = onnx.load(str(MODELS_DIR / relative_path(name, "int8", 1)))
    differences = oq.quant_param_differences(
        {0: shipped_b1, **{b: protos[b] for b in EXTRA_BATCHES}})
    return {"shipped_graphs_reproduced": reproduced,
            "quant_params_identical_to_shipped_b1": not differences,
            "quant_param_count": len(oq.quant_params(shipped_b1))}


def session(path: Path, extra: dict[str, str] | None = None) -> Any:
    """Strict QNN HTP session with make_references.py's options plus ``extra``."""
    import onnxruntime as ort

    from common.ort_qnn import (htp_backend_path, list_devices, qnn_npu_session_options,
                                register_qnn, select_qnn_npu)

    register_qnn()
    options = {"backend_path": str(htp_backend_path()), "htp_performance_mode": "burst",
               **(extra or {})}
    so = qnn_npu_session_options(select_qnn_npu(list_devices()), options, strict=True)
    return ort.InferenceSession(str(path), sess_options=so)


def run_items(sess: Any, items: list, batch: int) -> list:
    """First output per item; items are grouped into batches, the last one filled
    cyclically from the start."""
    import numpy as np

    name = sess.get_inputs()[0].name
    got: dict[int, Any] = {}
    n = len(items)
    for r in range(max(1, -(-n // batch))):
        ids = [(r * batch + k) % n for k in range(batch)]
        y = sess.run(None, {name: np.concatenate([items[i] for i in ids])})[0]
        for k, i in enumerate(ids):
            got.setdefault(i, y[k:k + 1])
    return [got[i] for i in range(n)]


def run_file(path: Path, items: list, batch: int, extra: dict[str, str] | None = None):
    import numpy as np

    s = session(path, extra)
    try:
        return np.concatenate(run_items(s, items, batch))
    finally:
        del s
        gc.collect()


def lsb_histogram(a: Any, b: Any, scale: float) -> dict[str, int]:
    import numpy as np

    d = np.round((a.astype(np.float64) - b.astype(np.float64)) / scale).astype(int)
    return {str(k): int(v) for k, v in sorted(Counter(d.ravel().tolist()).items())}


# ------------------------------------------------------------------- graph facts
class Graph:
    def __init__(self, path: Path) -> None:
        import onnx
        from onnx import numpy_helper

        self.proto = onnx.load(str(path))
        g = self.proto.graph
        self.inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        self.producer = {o: n for n in g.node for o in n.output}
        self.nodes = list(g.node)

    def dq_scale(self, dq_output: str) -> float:
        return float(self.inits[self.producer[dq_output].input[1]])

    def activation_cuts(self) -> list[str]:
        """Outputs of the DequantizeLinear nodes of activations (not of weights), in
        graph order."""
        q_out = {n.output[0] for n in self.nodes
                 if n.op_type == "QuantizeLinear" and n.input[0] not in self.inits}
        return [n.output[0] for n in self.nodes
                if n.op_type == "DequantizeLinear" and n.input[0] in q_out]

    def consumer_q(self, tensor: str) -> Any:
        return next(n for n in self.nodes if n.op_type == "QuantizeLinear" and n.input[0] == tensor)

    def consumer_dq(self, tensor: str) -> Any:
        return next(n for n in self.nodes
                    if n.op_type == "DequantizeLinear" and n.input[0] == tensor)


def largest_activation(path: Path) -> dict[str, Any]:
    """The largest uint8 activation (QuantizeLinear output) of the graph."""
    import numpy as np
    import onnx
    from onnx import shape_inference

    g = shape_inference.infer_shapes(onnx.load(str(path))).graph
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in (*g.value_info, *g.input, *g.output)}
    inits = {i.name for i in g.initializer}
    sizes = {}
    for n in g.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in inits:
            shape = shapes.get(n.output[0]) or shapes.get(n.input[0])
            if shape and all(shape):
                sizes[n.output[0]] = (int(np.prod(shape)), shape)
    name = max(sizes, key=lambda k: sizes[k][0])
    return {"tensor": name, "shape": sizes[name][1], "bytes": sizes[name][0]}


# ------------------------------------------------------------------- steps
def reference_images() -> list:
    import numpy as np

    from common.reference_outputs import REFERENCE_DIR

    folder = REFERENCE_DIR / "inputs" / "images-normalized-ef8b6be5"
    return [np.fromfile(folder / f"{i:02d}.f32", dtype=np.float32).reshape(1, 3, 224, 224)
            for i in range(8)]


def reference_outputs_b1(name: str) -> list:
    import numpy as np

    from common.reference_outputs import REFERENCE_DIR

    record = json.loads((REFERENCE_DIR / "outputs" / name / f"{name}_int8_b1" / "npu.json")
                        .read_text(encoding="utf-8"))
    outputs = sorted((o for o in record["outputs"] if o["input_set"].startswith("images-")),
                     key=lambda o: o["run"])
    return [np.fromfile(REFERENCE_DIR / o["file"], dtype=np.float32).reshape(1, -1)
            for o in outputs]


def threshold(work: Path, name: str, images: list, reference: list) -> dict[str, Any]:
    import numpy as np

    out_scale = Graph(model_path(work, name, 1)).dq_scale("logits")
    rows, outputs = {}, {}
    for batch in THRESHOLD_BATCHES:
        path = model_path(work, name, batch)
        y = run_file(path, images, batch)
        again = run_file(path, images, batch)
        outputs[batch] = y
        equal = sum(bool(np.array_equal(y[i:i + 1], reference[i])) for i in range(len(images)))
        act = largest_activation(path)
        rows[str(batch)] = {"items_equal_b1_reference": equal, "items": len(images),
                            "repeat_identical": bool(np.array_equal(y, again)),
                            "largest_activation": act,
                            "largest_activation_exceeds_vtcm": act["bytes"] > VTCM_BYTES}
        log(f"  {name} b{batch}: {equal}/{len(images)} equal to b1, largest activation "
            f"{act['bytes']:,} B")
    first = next((b for b in THRESHOLD_BATCHES if rows[str(b)]["items_equal_b1_reference"] == 0),
                 None)
    s8 = session(model_path(work, name, 8))
    feed = s8.get_inputs()[0].name
    reversed_order = s8.run(None, {feed: np.concatenate(images[::-1])})[0][::-1]
    del s8
    gc.collect()
    return {
        "output_scale": out_scale,
        "batches": rows,
        "first_diverging_batch": first,
        "diverging_batches_identical": all(np.array_equal(outputs[b], outputs[first])
                                           for b in THRESHOLD_BATCHES if b >= first),
        "b8_independent_of_position_in_batch": bool(np.array_equal(reversed_order, outputs[8])),
        "output_lsb_histogram_b8_minus_b1": lsb_histogram(outputs[8], outputs[1], out_scale),
        "top1_changes_b8_reference_images": int((outputs[8].argmax(1)
                                                 != outputs[1].argmax(1)).sum()),
    }


def layer_scan(work: Path, name: str, batch: int, images: list) -> dict[str, Any]:
    """Every activation as the end of a truncated model, b1 vs ``batch``."""
    import onnx.utils

    graph = Graph(model_path(work, name, 1))
    folder = work / "cuts" / name
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, tensor in enumerate(graph.activation_cuts()):
        values = {}
        for b in (1, batch):
            path = folder / f"cut{index:03d}_b{b}.onnx"
            onnx.utils.extract_model(str(model_path(work, name, b)), str(path), ["input"], [tensor])
            values[b] = run_file(path, images, b)
        hist = lsb_histogram(values[batch], values[1], graph.dq_scale(tensor))
        differing = sum(v for k, v in hist.items() if k != "0")
        rows.append({"index": index, "tensor": tensor, "elements": int(values[1].size),
                     "differing": differing, "lsb_histogram": hist})
        log(f"  {name} cut {index} {tensor}: {differing} differ")
    first = next((r["tensor"] for r in rows if r["differing"]), None)
    return {"model": name, "batch": batch, "cut_points": len(rows), "first_diverging": first,
            "rows": rows}


def tail(work: Path, name: str, images: list) -> dict[str, Any]:
    """Classifier tail sub-models at TAIL_BATCHES, b1 as the base."""
    import numpy as np
    import onnx.utils

    graph = Graph(model_path(work, name, 1))
    pool = next(n for n in graph.nodes if n.op_type == "GlobalAveragePool")
    pool_in_dq = graph.producer[pool.input[0]]
    pool_in_q = pool_in_dq.input[0]
    sx, zx = float(graph.inits[pool_in_dq.input[1]]), int(graph.inits[pool_in_dq.input[2]])
    pool_dq = graph.consumer_dq(graph.consumer_q(pool.output[0]).output[0]).output[0]
    flatten = next(n for n in graph.nodes if n.op_type == "Flatten")
    flatten_q = graph.consumer_q(flatten.output[0])
    flatten_dq = graph.consumer_dq(flatten_q.output[0]).output[0]
    folder = work / "tail" / name
    folder.mkdir(parents=True, exist_ok=True)
    # the real uint8 activations entering the tail, from the b1 file
    cut = folder / "to_pool_input_b1.onnx"
    onnx.utils.extract_model(str(model_path(work, name, 1)), str(cut), ["input"],
                             [pool_in_dq.output[0]])
    real = run_file(cut, images, 1)
    pool_items = [(np.round(real[i:i + 1] / sx) + zx).astype(np.uint8) for i in range(len(real))]
    channels = graph.inits[graph.producer[
        next(n for n in graph.nodes if n.op_type == "Gemm").input[1]].input[0]].shape[1]
    rng = np.random.default_rng(1)
    vectors = rng.integers(0, 256, size=(GEMM_VECTORS, channels), dtype=np.uint8)
    gemm_items = [vectors[i:i + 1] for i in range(GEMM_VECTORS)]
    variants = {
        "global_average_pool": ([pool_in_q], [pool_dq], pool_items),
        "global_average_pool_flatten": ([pool_in_q], [flatten_dq], pool_items),
        "tail_to_logits": ([pool_in_q], ["logits"], pool_items),
        "gemm_only": ([flatten_q.output[0]], ["logits"], gemm_items),
    }
    result = {}
    for variant, (inputs, outputs, items) in variants.items():
        scale = graph.dq_scale(outputs[0])
        base = None
        result[variant] = {}
        for b in TAIL_BATCHES:
            path = folder / f"{variant}_b{b}.onnx"
            onnx.utils.extract_model(str(model_path(work, name, b)), str(path), inputs, outputs)
            y = run_file(path, items, b)
            base = y if base is None else base
            result[variant][str(b)] = lsb_histogram(y, base, scale)
        log(f"  {name} {variant}: {result[variant]}")
    return result


def gemm_emulation(work: Path, name: str) -> dict[str, Any]:
    """The Gemm-only sub-model at b1 and b8 vs exact integer arithmetic."""
    import numpy as np

    graph = Graph(model_path(work, name, 1))
    gemm = next(n for n in graph.nodes if n.op_type == "Gemm")
    xdq, wdq, bdq = (graph.producer[t] for t in gemm.input)
    sx, zx = np.float64(graph.inits[xdq.input[1]]), int(graph.inits[xdq.input[2]])
    wq = graph.inits[wdq.input[0]].astype(np.int64)
    zw = np.asarray(graph.inits[wdq.input[2]], dtype=np.int64).reshape(-1, 1)
    sw = graph.inits[wdq.input[1]].astype(np.float64)
    bias = graph.inits[bdq.input[0]].astype(np.int64)
    oq = graph.consumer_q(gemm.output[0])
    sy, zy = np.float64(graph.inits[oq.input[1]]), int(graph.inits[oq.input[2]])
    rng = np.random.default_rng(1)
    xs = rng.integers(0, 256, size=(GEMM_VECTORS, wq.shape[1]), dtype=np.uint8)
    items = [xs[i:i + 1] for i in range(GEMM_VECTORS)]
    exact = sx * sw[None, :] * ((xs.astype(np.int64) - zx) @ (wq - zw).T + bias[None, :]) / sy + zy
    inside = (exact > 0.5) & (exact < 254.5)  # not clipped to the uint8 range
    rounded = np.rint(exact)
    result = {"vectors": GEMM_VECTORS, "outputs_compared": int(inside.sum()),
              "bias_scale_is_input_times_weight_scale": bool(np.allclose(
                  graph.inits[bdq.input[1]].astype(np.float64), sx * sw, rtol=1e-6))}
    q = {}
    for batch, extra, label in ((1, None, "b1"), (8, None, "b8"), (8, FIN1, "b8_finalization_1")):
        y = run_file(work / "tail" / name / f"gemm_only_b{batch}.onnx", items, batch, extra)
        q[label] = np.round(y.astype(np.float64) / sy) + zy
        err = (q[label] - exact)[inside]
        result[label] = {
            "error_vs_exact_mean_lsb": float(err.mean()),
            "error_vs_exact_std_lsb": float(err.std()),
            "error_vs_exact_min_lsb": float(err.min()),
            "error_vs_exact_max_lsb": float(err.max()),
            "minus_exactly_rounded_histogram": {
                str(int(k)): int(v) for k, v in zip(*np.unique(
                    (q[label] - rounded)[inside], return_counts=True))},
        }
    result["b8_equals_b1"] = bool(np.array_equal(q["b8"], q["b1"]))
    result["b8_finalization_1_equals_b1"] = bool(np.array_equal(q["b8_finalization_1"], q["b1"]))
    log(f"  {name} gemm: b1 mean error {result['b1']['error_vs_exact_mean_lsb']:.3f} LSB, "
        f"b8 {result['b8']['error_vs_exact_mean_lsb']:.3f} LSB")
    return result


def options(work: Path, name: str, images: list, reference: list) -> dict[str, Any]:
    import numpy as np

    result = {}
    for label, extra in OPTION_CONFIGS.items():
        result[label] = {}
        for batch in OPTION_BATCHES:
            y = run_file(model_path(work, name, batch), images, batch, extra)
            result[label][str(batch)] = sum(bool(np.array_equal(y[i:i + 1], reference[i]))
                                            for i in range(len(images)))
        log(f"  {name} {label}: {result[label]}")
    return result


def eval_quick_and_latency(work: Path, name: str, cache: Any) -> dict[str, Any]:
    import numpy as np

    from common import imagenet_eval as ie

    spec = MODELS[name]
    x = ie.model_input(cache, 0, len(cache), "external", spec.mean, spec.std)
    logits = {}
    for label, batch, extra in (("default_b1", 1, {}), ("default_b8", 8, {}),
                                ("finalization_1_b1", 1, FIN1), ("finalization_1_b8", 8, FIN1)):
        s = session(model_path(work, name, batch), extra)
        feed = s.get_inputs()[0].name
        logits[label] = np.concatenate([s.run(None, {feed: x[i:i + batch]})[0]
                                        for i in range(0, len(x), batch)])
        del s
        gc.collect()

    def compare(label: str) -> dict[str, int]:
        a, b = logits[label], logits["default_b1"]
        return {"images": len(a),
                "bitwise_equal": int(sum(np.array_equal(u, v) for u, v in zip(a, b))),
                "top1_differing": int((a.argmax(1) != b.argmax(1)).sum())}

    eval_quick = {label: compare(label)
                  for label in ("default_b8", "finalization_1_b1", "finalization_1_b8")}
    sessions = {}
    for label, batch, extra in (("default_b1", 1, {}), ("finalization_1_b1", 1, FIN1),
                                ("default_b4", 4, {}), ("default_b8", 8, {}),
                                ("finalization_1_b8", 8, FIN1)):
        sessions[label] = (session(model_path(work, name, batch), extra), batch)
    times: dict[str, list[float]] = {k: [] for k in sessions}
    for rnd in range(LATENCY_ROUNDS):
        for label in (list(sessions) if rnd % 2 == 0 else list(sessions)[::-1]):
            s, batch = sessions[label]
            feed = {s.get_inputs()[0].name: x[:batch]}
            for _ in range(LATENCY_WARMUP):
                s.run(None, feed)
            for _ in range(LATENCY_RUNS):
                t0 = time.perf_counter()
                s.run(None, feed)
                times[label].append((time.perf_counter() - t0) * 1e3)
    del sessions
    gc.collect()
    latency = {label: {"median_ms": float(np.median(v)), "runs": len(v)}
               for label, v in times.items()}
    log(f"  {name} eval-quick {eval_quick}; latency "
        f"{ {k: round(v['median_ms'], 3) for k, v in latency.items()} }")
    return {"eval_quick": eval_quick, "latency": latency}


def qnn_graph_difference(work: Path, name: str) -> list[str]:
    """Paths that differ between the QNN graphs (JSON) the EP builds for the Gemm-only
    sub-model at b6 and b7."""
    graphs = []
    for batch in (6, 7):
        folder = work / "qnn_json" / f"{name}_gemm_only_b{batch}"
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.json"):
            old.unlink()
        s = session(work / "tail" / name / f"gemm_only_b{batch}.onnx",
                    {"dump_json_qnn_graph": "1", "json_qnn_graph_dir": str(folder)})
        del s
        gc.collect()
        files = [p for p in folder.glob("*.json") if not p.name.endswith("_tensor_log.json")]
        graphs.append(json.loads(files[0].read_text(encoding="utf-8")))
    differences: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append(f"{path}/{key}")
                else:
                    walk(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (u, v) in enumerate(zip(a, b)):
                walk(u, v, f"{path}[{i}]")
        elif a != b:
            differences.append(f"{path}: {a} -> {b}")

    walk(graphs[0], graphs[1], "")
    nodes = sorted({v.get("type") for v in graphs[0]["graph"]["nodes"].values()})
    log(f"  QNN graph nodes {nodes}; differences b6 -> b7: {differences}")
    return differences


def environment() -> dict[str, Any]:
    import onnxruntime as ort

    from common.ort_qnn import power_state, qnn_versions

    return {"onnxruntime": ort.__version__, **qnn_versions(), "power": power_state()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR,
                        help="where the extra models and sub-models go (default: "
                             "tmp/batch_investigation)")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="findings JSON (default: docs/findings/npu_int8_batch_size.json)")
    args = parser.parse_args(argv)

    from common.platform_guard import NonNativeInterpreterError, require_native_arm64

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2
    from common import imagenet_eval as ie
    from common.manifests import EVAL_QUICK_MANIFEST

    work = args.work_dir.resolve()
    images = reference_images()
    started = time.monotonic()
    findings: dict[str, Any] = {
        "findings_version": FINDINGS_VERSION,
        "description": "NPU (QNN HTP) INT8 image models: outputs at batch sizes other than 1; "
                       "written by scripts/investigate_batch_equivalence.py",
        "environment": environment(),
        "vtcm_bytes": VTCM_BYTES,
        "reference_images": len(images),
        "models": {},
    }
    eval_cache = ie.build_cache(ie.load_eval_set("eval-quick", EVAL_QUICK_MANIFEST))
    for name in IMAGE_MODELS:
        log(f"{name}: building INT8 files at b{', b'.join(map(str, EXTRA_BATCHES))}")
        entry: dict[str, Any] = {"build": build_models(work, name)}
        if not entry["build"]["quant_params_identical_to_shipped_b1"] or \
                entry["build"]["shipped_graphs_reproduced"] != [1, 4]:
            print(f"error: {name}: the rebuilt files do not match the shipped recipe",
                  file=sys.stderr)
            return 1
        reference = reference_outputs_b1(name)
        log(f"{name}: threshold")
        entry["threshold"] = threshold(work, name, images, reference)
        log(f"{name}: classifier tail")
        entry["tail"] = tail(work, name, images)
        entry["gemm_emulation"] = gemm_emulation(work, name)
        log(f"{name}: options")
        entry["options_items_equal_b1_reference"] = options(work, name, images, reference)
        entry.update(eval_quick_and_latency(work, name, eval_cache))
        findings["models"][name] = entry
    first = findings["models"][SCAN_MODEL]["threshold"]["first_diverging_batch"]
    log(f"{SCAN_MODEL}: every activation, b1 vs b{first}")
    findings["layer_scan"] = layer_scan(work, SCAN_MODEL, first, images)
    findings["qnn_graph_differences_b6_b7"] = {
        "model": SCAN_MODEL, "sub_model": "gemm_only",
        "differences": qnn_graph_difference(work, SCAN_MODEL)}
    findings["duration_s"] = round(time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(findings, indent=1) + "\n", encoding="utf-8",
                           newline="\n")
    log(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
