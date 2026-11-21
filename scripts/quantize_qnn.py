"""Statically quantize the FP32 MobileNetV2 to INT8 QDQ for the QNN HTP (NPU) backend.

Uses ONNX Runtime's QNN-specific quantization helpers:

1. ``qnn_preprocess_model`` applies QNN-friendly graph fusions to the FP32 model,
2. ``get_qnn_qdq_config`` builds a static QDQ configuration (calibrated on the
   ``calibration`` split from ``scripts/get_data.py``, preprocessed with
   ``scripts/common/preprocess.py``),
3. ``onnxruntime.quantization.quantize`` writes the QDQ model.

Two variants are produced in ``models/mobilenetv2/``:

- ``mobilenetv2_int8qdq_b1.onnx``: helper defaults (QUInt8 activations and
  weights, per-tensor, MinMax calibration),
- ``mobilenetv2_int8qdq_pc_b1.onnx``: the same with ``per_channel=True``.

Each model gets a JSON sidecar (same name, ``.json``) recording the source
model, calibration data, the configuration that was applied, what the output
graph actually contains, the ONNX Runtime version, and the model's SHA256.

Usage:
    python scripts/quantize_qnn.py [--fp32 PATH] [--calibration-count N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402
from common.preprocess import (  # noqa: E402
    IMAGENETTE_ROOT,
    PREPROCESS_DESCRIPTION,
    REPO_ROOT,
    SPLITS_DIR,
    load_image,
    read_split,
)

MODEL_DIR = REPO_ROOT / "models" / "mobilenetv2"
DEFAULT_FP32 = MODEL_DIR / "mobilenetv2_fp32_b1.onnx"
CALIBRATION_SPLIT = "calibration"
VARIANTS = {
    "mobilenetv2_int8qdq_b1.onnx": {"per_channel": False},
    "mobilenetv2_int8qdq_pc_b1.onnx": {"per_channel": True},
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_reader(input_name: str, images: list[str]):
    from onnxruntime.quantization import CalibrationDataReader

    class ImageListReader(CalibrationDataReader):
        """Feeds preprocessed images one at a time (batch 1, matching the model)."""

        def __init__(self) -> None:
            self._it = iter(images)

        def get_next(self):
            path = next(self._it, None)
            return None if path is None else {input_name: load_image(IMAGENETTE_ROOT / path)}

    return ImageListReader()


def describe_qdq_graph(path: Path) -> dict:
    """Summarize what the quantizer actually produced."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    inits = {i.name: i for i in model.graph.initializer}
    ops = Counter(n.op_type for n in model.graph.node)
    compute = sorted(op for op in ops if op not in {"QuantizeLinear", "DequantizeLinear", "Constant"})

    # Constant tensors are DequantizeLinear nodes whose quantized input is an
    # initializer: INT32 ones are biases, everything else is a weight.
    def empty() -> dict:
        return {"count": 0, "by_type": Counter(), "per_channel": 0, "per_tensor": 0,
                "symmetric_zero_point": 0}

    groups = {"weights": empty(), "biases": empty()}
    activation_types: Counter = Counter()
    for n in model.graph.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in inits:
            q = inits[n.input[0]]
            g = groups["biases" if q.data_type == onnx.TensorProto.INT32 else "weights"]
            g["count"] += 1
            g["by_type"][onnx.TensorProto.DataType.Name(q.data_type)] += 1
            scale = numpy_helper.to_array(inits[n.input[1]])
            g["per_channel" if scale.size > 1 else "per_tensor"] += 1
            zp = (numpy_helper.to_array(inits[n.input[2]]).astype(int)
                  if len(n.input) > 2 and n.input[2] in inits else 0)
            mid = 128 if q.data_type == onnx.TensorProto.UINT8 else 0
            g["symmetric_zero_point"] += int(np.all(zp == mid))
        elif n.op_type == "QuantizeLinear" and len(n.input) > 2 and n.input[2] in inits:
            zp = inits[n.input[2]]
            activation_types[onnx.TensorProto.DataType.Name(zp.data_type)] += 1
    for g in groups.values():
        g["by_type"] = dict(g["by_type"])

    # Nodes whose outputs nothing consumes, e.g. the Q/DQ chains the quantizer
    # leaves on the min/max constants of Clip (ReLU6) nodes it folded away.
    consumed = {i for n in model.graph.node for i in n.input} | {o.name for o in model.graph.output}
    unused = Counter(n.op_type for n in model.graph.node if not any(o in consumed for o in n.output))

    return {
        "node_counts": dict(sorted(ops.items())),
        "unused_output_nodes": dict(unused),
        "compute_op_types": compute,
        "weights": groups["weights"],
        "biases": groups["biases"],
        "activation_quantize_nodes_by_type": dict(activation_types),
    }


def quantize_variant(fp32: Path, preprocessed: Path, out: Path, per_channel: bool,
                     images: list[str]) -> dict:
    """Quantize ``preprocessed`` (``fp32`` after QNN graph preprocessing, or ``fp32`` itself)."""
    import onnxruntime as ort
    from onnxruntime.quantization import quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

    session = ort.InferenceSession(str(preprocessed), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    del session

    config = get_qnn_qdq_config(
        str(preprocessed),
        make_reader(input_name, images),
        per_channel=per_channel,
    )
    quantize(str(preprocessed), str(out), config)

    applied = {
        "helper": "onnxruntime.quantization.execution_providers.qnn.get_qnn_qdq_config",
        "calibration_method": config.calibrate_method.name,
        "activation_type": config.activation_type.name,
        "weight_type": config.weight_type.name,
        "per_channel": config.per_channel,
        "quant_format": config.quant_format.name,
        "extra_options": {
            k: v for k, v in config.extra_options.items() if k != "TensorQuantOverrides"
        },
        "tensor_quant_overrides_count": len(config.extra_options.get("TensorQuantOverrides", {})),
    }
    return {
        "model": out.name,
        "source_model": {"file": fp32.name, "sha256": sha256_of(fp32)},
        "qnn_preprocess_applied": preprocessed != fp32,
        "calibration": {
            "split": CALIBRATION_SPLIT,
            "split_file": f"data/imagenette/splits/{CALIBRATION_SPLIT}.txt",
            "split_file_sha256": sha256_of(SPLITS_DIR / f"{CALIBRATION_SPLIT}.txt"),
            "count": len(images),
            "preprocessing": PREPROCESS_DESCRIPTION,
        },
        "quantization_config": applied,
        "graph": describe_qdq_graph(out),
        "onnxruntime_version": ort.__version__,
        "sha256": sha256_of(out),
        "size_bytes": out.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fp32", type=Path, default=DEFAULT_FP32, help="FP32 source model")
    parser.add_argument("--calibration-count", type=int, default=None,
                        help="use only the first N calibration images (default: all)")
    args = parser.parse_args()

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    from onnxruntime.quantization.execution_providers.qnn import qnn_preprocess_model

    fp32 = args.fp32.resolve()
    if not fp32.is_file():
        print(f"{fp32} not found; run scripts/export_mobilenetv2.py first", file=sys.stderr)
        return 1
    images = read_split(CALIBRATION_SPLIT)
    if args.calibration_count is not None:
        images = images[: args.calibration_count]

    with tempfile.TemporaryDirectory() as tmp:
        preprocessed = Path(tmp) / "preprocessed.onnx"
        changed = qnn_preprocess_model(str(fp32), str(preprocessed))
        print(f"qnn_preprocess_model: {'graph modified' if changed else 'no changes'}")
        source = preprocessed if changed else fp32

        for name, opts in VARIANTS.items():
            out = MODEL_DIR / name
            print(f"quantizing {name} (per_channel={opts['per_channel']}, "
                  f"{len(images)} calibration images)")
            sidecar = quantize_variant(fp32, source, out, opts["per_channel"], images)
            sidecar_path = out.with_suffix(".json")
            sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
            g = sidecar["graph"]
            print(f"  wrote {out} ({out.stat().st_size} bytes), sidecar {sidecar_path.name}")
            print(f"  weights: {g['weights']}")
            print(f"  biases: {g['biases']}")
            print(f"  activations: {g['activation_quantize_nodes_by_type']}")
            print(f"  compute ops: {g['compute_op_types']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
