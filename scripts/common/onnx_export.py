"""Export torchvision classifiers to fixed-shape ONNX, and check the result.

Used by ``scripts/export_models.py`` and by ``scripts/tests/test_export_models.py``.
Imports torch, torchvision, onnx and onnxruntime lazily, inside the functions,
so importing this module is cheap.

What an exported file guarantees:

- every graph input and output has concrete dimensions; the batch dimension is
  fixed per file;
- input ``input`` float32 [batch, 3, 224, 224] is the output of
  ``common.preprocess.load_image`` (ImageNet mean/std already applied:
  ``edgebench.normalization`` = ``external`` in the ONNX metadata);
- output ``logits`` float32 [batch, 1000], raw logits in torchvision class order;
- exporting again with the same tool versions gives a byte-identical file.
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from pathlib import Path
from typing import Any

from common.model_registry import (
    EXPORTER,
    INPUT_NAME,
    META_NORMALIZATION,
    META_PREFIX,
    NUM_CLASSES,
    OPSET,
    OUTPUT_NAME,
    ModelSpec,
)
from common.preprocess import PREPROCESS_DESCRIPTION

# Largest accepted absolute difference between two FP32 logit vectors for the
# same image (ONNX Runtime CPU vs PyTorch, or batch N vs batch 1).
#
# Both sides compute the same float32 graph, but with different kernels (MLAS vs
# PyTorch's CPU backend), fused vs separate operators and batch-dependent GEMM
# blocking, so additions happen in a different order. Each reordering moves a
# result by about one rounding step: float32 has 24 significant bits, so a logit
# of magnitude ~20 moves in steps of ~2e-6, and the error grows with the depth of
# the network and the length of the dot products (up to 4608 terms in ResNet-18).
# Measured differences are 1e-5 to 2.3e-5 (see docs/models.md); 1e-4 leaves a 4-9x
# margin while staying 4-5 orders of magnitude below a real export error (wrong
# weights, layout or layer give differences of order 1-10).
LOGIT_ATOL = 1e-4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_versions() -> dict[str, str]:
    import onnx
    import torch
    import torchvision

    return {"onnx": onnx.__version__, "torch": torch.__version__,
            "torchvision": torchvision.__version__}


# --- weights -----------------------------------------------------------------


def weights_path(spec: ModelSpec) -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / Path(spec.weights_url).name


def ensure_weights(spec: ModelSpec, download: bool = True) -> Path:
    """The verified weights file in torch's cache, downloading it if needed.

    Raises FileNotFoundError if it is missing and ``download`` is false, and
    RuntimeError if its SHA256 is not the pinned one.
    """
    import torch

    path = weights_path(spec)
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"weights for {spec.name} not downloaded: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(spec.weights_url, str(path), progress=False)
    digest = sha256_file(path)
    if digest != spec.weights_sha256:
        raise RuntimeError(f"SHA256 mismatch for {path}\n  expected {spec.weights_sha256}\n"
                           f"  got      {digest}\nDelete the file and run again to download it.")
    return path


def load_torch_model(spec: ModelSpec, download: bool = True) -> Any:
    """The torchvision model with its pinned, verified weights, in eval mode."""
    import torchvision

    ensure_weights(spec, download)
    weights = getattr(getattr(torchvision.models, spec.weights_enum), spec.weights)
    if Path(weights.url).name != Path(spec.weights_url).name:
        raise RuntimeError(f"torchvision {spec.weights_id} now points at {weights.url}, "
                           f"not the pinned {spec.weights_url}")
    return getattr(torchvision.models, spec.constructor)(weights=weights).eval()


# --- export ------------------------------------------------------------------


def input_shape(spec: ModelSpec, batch: int) -> tuple[int, int, int, int]:
    return (batch, 3, spec.input_size, spec.input_size)


def model_metadata(spec: ModelSpec, variant: str, batch: int,
                   versions: dict[str, str]) -> dict[str, str]:
    """Key/value pairs stored in the ONNX file's metadata_props."""
    return {
        META_PREFIX + "model": spec.name,
        META_PREFIX + "variant": variant,
        META_PREFIX + "batch_size": str(batch),
        META_PREFIX + "weights": spec.weights_id,
        META_PREFIX + "weights_sha256": spec.weights_sha256,
        META_NORMALIZATION: "external",
        META_PREFIX + "input_preprocessing": json.dumps(PREPROCESS_DESCRIPTION, sort_keys=True),
        META_PREFIX + "exporter": EXPORTER,
        META_PREFIX + "opset": str(OPSET),
        META_PREFIX + "torch_version": versions["torch"],
        META_PREFIX + "torchvision_version": versions["torchvision"],
        META_PREFIX + "onnx_version": versions["onnx"],
    }


def export_fp32(spec: ModelSpec, batch: int, model: Any = None) -> bytes:
    """Serialized FP32 ONNX model with a fixed batch size and EdgeBench metadata."""
    import onnx
    import torch

    model = load_torch_model(spec) if model is None else model
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            (torch.zeros(input_shape(spec, batch)),),
            buffer,
            opset_version=OPSET,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamo=False,
        )
    proto = onnx.load_from_string(buffer.getvalue())
    metadata = model_metadata(spec, "fp32", batch, tool_versions())
    metadata[META_PREFIX + "ir_version"] = str(proto.ir_version)
    onnx.helper.set_model_props(proto, dict(sorted(metadata.items())))
    return proto.SerializeToString()


# --- checks ------------------------------------------------------------------


def dims(value_info: Any) -> list[int | str | None]:
    """Dimensions of a graph input/output: int if concrete, str if symbolic, None if unset."""
    out: list[int | str | None] = []
    for d in value_info.type.tensor_type.shape.dim:
        which = d.WhichOneof("value")
        out.append(d.dim_value if which == "dim_value" else d.dim_param if which else None)
    return out


def static_shape_problems(proto: Any) -> list[str]:
    """Graph inputs/outputs (initializers excluded) whose shape is not fully concrete."""
    initializers = {i.name for i in proto.graph.initializer}
    problems = []
    for kind, values in (("input", proto.graph.input), ("output", proto.graph.output)):
        for v in values:
            if v.name in initializers:
                continue
            shape = dims(v)
            if not shape or not all(isinstance(d, int) and d > 0 for d in shape):
                problems.append(f"{kind} '{v.name}' has shape {shape}")
    return problems


def check_exported(data: bytes, spec: ModelSpec, batch: int) -> dict[str, Any]:
    """onnx.checker, strict shape inference and shape checks; raises on any problem."""
    import onnx

    proto = onnx.load_from_string(data)
    onnx.checker.check_model(proto, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(proto, check_type=True, strict_mode=True)
    problems = static_shape_problems(proto)
    expected = {INPUT_NAME: list(input_shape(spec, batch)), OUTPUT_NAME: [batch, NUM_CLASSES]}
    declared = {v.name: dims(v) for v in [*proto.graph.input, *proto.graph.output]
                if v.name in expected}
    if declared != expected:
        problems.append(f"expected shapes {expected}, file declares {declared}")
    inferred_out = {v.name: dims(v) for v in inferred.graph.output}
    if inferred_out.get(OUTPUT_NAME) != expected[OUTPUT_NAME]:
        problems.append(f"shape inference gives output {inferred_out}")
    unknown = [v.name for v in inferred.graph.value_info
               if not all(isinstance(d, int) and d > 0 for d in dims(v))]
    if unknown:
        problems.append(f"{len(unknown)} intermediate tensors without concrete shapes, "
                        f"e.g. {unknown[:3]}")
    if problems:
        raise RuntimeError(f"{spec.name} batch {batch}: " + "; ".join(problems))
    return {"ir_version": proto.ir_version,
            "opset": next(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx")),
            "shapes": declared, "nodes": len(proto.graph.node),
            "metadata": {p.key: p.value for p in proto.metadata_props}}


# --- running -----------------------------------------------------------------


def ort_session(path: Path, threads: int = 4) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_ort(session: Any, x: Any) -> Any:
    """Logits for all rows of ``x`` [n, 3, H, W], fed in the session's fixed batch size."""
    import numpy as np

    batch = session.get_inputs()[0].shape[0]
    if len(x) % batch:
        raise ValueError(f"{len(x)} images do not fill batches of {batch}")
    return np.concatenate([session.run(None, {INPUT_NAME: x[i:i + batch]})[0]
                           for i in range(0, len(x), batch)])


def run_torch(model: Any, x: Any, batch: int = 8) -> Any:
    import numpy as np
    import torch

    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i + batch])).numpy()
                               for i in range(0, len(x), batch)])


def compare_logits(actual: Any, expected: Any) -> dict[str, Any]:
    """Max absolute difference and top-1 agreement between two [n, classes] arrays."""
    import numpy as np

    agree = int(np.sum(actual.argmax(axis=1) == expected.argmax(axis=1)))
    return {"images": int(len(actual)),
            "max_abs_diff": float(np.max(np.abs(actual.astype(np.float64) - expected))),
            "top1_agreement": agree / len(actual)}


def within_tolerance(comparison: dict[str, Any]) -> bool:
    return comparison["max_abs_diff"] <= LOGIT_ATOL and comparison["top1_agreement"] == 1.0


# --- check images --------------------------------------------------------------

CHECK_IMAGES = 64  # spread evenly over eval-quick, so over 64 different classes


def check_images(count: int = CHECK_IMAGES) -> tuple[list[str], Any] | None:
    """Preprocessed ImageNetV2 eval-quick images [count, 3, 224, 224], or None if absent."""
    import numpy as np

    from common.manifests import EVAL_QUICK_MANIFEST, IMAGENETV2_ROOT, read_manifest
    from common.preprocess import load_image

    if not (IMAGENETV2_ROOT / ".extracted").is_file():
        return None
    entries = read_manifest(EVAL_QUICK_MANIFEST)
    chosen = [entries[i * len(entries) // count] for i in range(count)]
    x = np.concatenate([load_image(IMAGENETV2_ROOT / e.path) for e in chosen])
    return [e.path for e in chosen], x
