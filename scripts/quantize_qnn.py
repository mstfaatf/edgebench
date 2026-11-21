"""Statically quantize the FP32 MobileNetV2 to INT8 QDQ for the QNN HTP (NPU) backend.

Uses ONNX Runtime's QNN-specific quantization helpers:

1. ``qnn_preprocess_model`` applies QNN-friendly graph fusions to the FP32 model,
2. ``get_qnn_qdq_config`` builds a static QDQ configuration (calibrated on the
   ``calibration`` split from ``scripts/get_data.py``, preprocessed with
   ``scripts/common/preprocess.py``),
3. ``onnxruntime.quantization.quantize`` writes the QDQ model,
4. dead nodes and unused initializers are pruned (``scripts/common/onnx_prune.py``)
   and the pruned model is checked to give bit-identical CPU outputs to the
   unpruned one on the ``eval`` split.

Three variants are produced in ``models/mobilenetv2/`` (all with uint8
activations and MinMax calibration):

- ``mobilenetv2_int8qdq_b1.onnx``: helper defaults (uint8 weights, per-tensor),
- ``mobilenetv2_int8qdq_pc_b1.onnx``: uint8 weights, per-channel,
- ``mobilenetv2_int8qdq_pc_s8w_b1.onnx``: signed int8 symmetric weights, per-channel.

The unpruned quantizer output is kept in ``models/mobilenetv2/unpruned/``.
Each model gets a JSON sidecar (same name, ``.json``) recording the source
model, calibration data, the configuration that was applied, the pruning
result, what the output graph contains, the ONNX Runtime version, and the
model's SHA256.

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

from common.onnx_prune import prune  # noqa: E402
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
UNPRUNED_DIR = MODEL_DIR / "unpruned"
CALIBRATION_SPLIT = "calibration"
VERIFY_SPLIT = "eval"
# name -> (per_channel, weight type name in onnxruntime.quantization.QuantType)
VARIANTS = {
    "mobilenetv2_int8qdq_b1.onnx": (False, "QUInt8"),
    "mobilenetv2_int8qdq_pc_b1.onnx": (True, "QUInt8"),
    "mobilenetv2_int8qdq_pc_s8w_b1.onnx": (True, "QInt8"),
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


def cpu_logits(model: Path, images: list[str]):
    """Logits of ``model`` on the CPU EP for each image, as one [N, classes] array."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return np.stack([session.run(None, {name: load_image(IMAGENETTE_ROOT / p)})[0][0] for p in images])


def quantize_variant(fp32: Path, preprocessed: Path, out: Path, per_channel: bool,
                     weight_type: str, calibration: list[str], verify: list[str]) -> dict:
    """Quantize ``preprocessed`` (``fp32`` after QNN graph preprocessing, or ``fp32`` itself)."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

    session = ort.InferenceSession(str(preprocessed), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    del session

    config = get_qnn_qdq_config(
        str(preprocessed),
        make_reader(input_name, calibration),
        per_channel=per_channel,
        weight_type=QuantType[weight_type],
    )
    unpruned = UNPRUNED_DIR / out.name
    unpruned.parent.mkdir(parents=True, exist_ok=True)
    quantize(str(preprocessed), str(unpruned), config)

    model = onnx.load(str(unpruned))
    report = prune(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(out))

    print(f"  verifying pruned vs unpruned on {len(verify)} {VERIFY_SPLIT} images (CPU EP)")
    before = cpu_logits(unpruned, verify)
    after = cpu_logits(out, verify)
    identical = bool(np.array_equal(before, after))
    if not identical:
        raise RuntimeError(f"pruned {out.name} differs from the unpruned model: "
                           f"max abs diff {float(np.max(np.abs(before - after)))}")

    applied = {
        "helper": "onnxruntime.quantization.execution_providers.qnn.get_qnn_qdq_config",
        "helper_arguments": {"per_channel": per_channel, "weight_type": weight_type},
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
            "count": len(calibration),
            "preprocessing": PREPROCESS_DESCRIPTION,
        },
        "quantization_config": applied,
        "pruning": {
            **report.as_dict(),
            "unpruned_model": {"file": f"unpruned/{unpruned.name}", "sha256": sha256_of(unpruned),
                               "graph": describe_qdq_graph(unpruned)},
            "verification": {"split": VERIFY_SPLIT, "count": len(verify),
                             "execution_provider": "CPUExecutionProvider",
                             "bit_identical_logits": identical},
        },
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
    calibration = read_split(CALIBRATION_SPLIT)
    if args.calibration_count is not None:
        calibration = calibration[: args.calibration_count]
    verify = read_split(VERIFY_SPLIT)

    with tempfile.TemporaryDirectory() as tmp:
        preprocessed = Path(tmp) / "preprocessed.onnx"
        changed = qnn_preprocess_model(str(fp32), str(preprocessed))
        print(f"qnn_preprocess_model: {'graph modified' if changed else 'no changes'}")
        source = preprocessed if changed else fp32

        for name, (per_channel, weight_type) in VARIANTS.items():
            out = MODEL_DIR / name
            print(f"quantizing {name} (per_channel={per_channel}, weights {weight_type}, "
                  f"{len(calibration)} calibration images)")
            sidecar = quantize_variant(fp32, source, out, per_channel, weight_type,
                                       calibration, verify)
            sidecar_path = out.with_suffix(".json")
            sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
            g, pr = sidecar["graph"], sidecar["pruning"]
            print(f"  pruned: nodes {pr['nodes_before']} -> {pr['nodes_after']} "
                  f"(removed {pr['removed_nodes_by_op']}, {pr['iterations']} passes), "
                  f"initializers {pr['initializers_before']} -> {pr['initializers_after']}; "
                  f"bit-identical: {pr['verification']['bit_identical_logits']}")
            print(f"  wrote {out} ({out.stat().st_size} bytes), sidecar {sidecar_path.name}")
            print(f"  weights: {g['weights']}")
            print(f"  biases: {g['biases']}")
            print(f"  activations: {g['activation_quantize_nodes_by_type']}")
            print(f"  compute ops: {g['compute_op_types']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
