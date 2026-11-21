"""Export torchvision MobileNetV2 to a fixed-shape FP32 ONNX model.

Writes ``models/mobilenetv2/mobilenetv2_fp32_b1.onnx``:

- weights: torchvision ``MobileNet_V2_Weights.IMAGENET1K_V2``,
- input ``input``: float32 [1, 3, 224, 224] (no dynamic axes),
- output ``logits``: float32 [1, 1000] (raw logits, no softmax),
- opset 17, TorchScript-based exporter.

The model is checked with ``onnx.checker``, compared against PyTorch on a
preprocessed image (if the data has been downloaded), and run on a tensor
filled with 0.5, the same input ``tools/ort_probe`` uses, where its top-1
class must be 677. A JSON sidecar with the export settings and the model's
SHA256 is written next to the model.

Usage:
    python scripts/export_mobilenetv2.py [--output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402
from common.preprocess import (  # noqa: E402
    IMAGENETTE_ROOT,
    PREPROCESS_DESCRIPTION,
    REPO_ROOT,
    load_image,
    read_split,
)

DEFAULT_OUTPUT = REPO_ROOT / "models" / "mobilenetv2" / "mobilenetv2_fp32_b1.onnx"
WEIGHTS = "IMAGENET1K_V2"
OPSET = 17
INPUT_SHAPE = (1, 3, 224, 224)
PROBE_INPUT_VALUE = 0.5
PROBE_EXPECTED_TOP1 = 677


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import torchvision

    weights = torchvision.models.MobileNet_V2_Weights[WEIGHTS]
    model = torchvision.models.mobilenet_v2(weights=weights).eval()

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            (torch.zeros(INPUT_SHAPE),),
            str(out),
            opset_version=OPSET,
            input_names=["input"],
            output_names=["logits"],
            dynamo=False,
        )
    onnx.checker.check_model(str(out), full_check=True)
    proto = onnx.load(str(out))
    print(f"exported {out} (opset {OPSET}, IR {proto.ir_version}, weights {WEIGHTS})")

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    if list(inp.shape) != list(INPUT_SHAPE):
        raise RuntimeError(f"expected static input shape {INPUT_SHAPE}, got {inp.shape}")

    failures = 0
    probe = session.run(None, {inp.name: np.full(INPUT_SHAPE, PROBE_INPUT_VALUE, np.float32)})[0][0]
    top1 = int(probe.argmax())
    ok = top1 == PROBE_EXPECTED_TOP1
    failures += 0 if ok else 1
    print(f"probe input (all {PROBE_INPUT_VALUE}): top-1 {top1} ({probe[top1]:.6f}), "
          f"expected {PROBE_EXPECTED_TOP1}: {'OK' if ok else 'MISMATCH'}")

    torch_check = None
    if IMAGENETTE_ROOT.is_dir():
        image = read_split("eval")[0]
        x = load_image(IMAGENETTE_ROOT / image)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()[0]
        got = session.run(None, {inp.name: x})[0][0]
        diff = float(np.max(np.abs(ref - got)))
        same = int(ref.argmax()) == int(got.argmax())
        failures += 0 if same and diff < 1e-3 else 1
        torch_check = {"image": image, "max_abs_diff": diff, "top1_match": same}
        print(f"PyTorch vs ORT CPU on {image}: max abs diff {diff:.3g}, top-1 match {same}")
    else:
        print("PyTorch comparison skipped: run scripts/get_data.py first")

    sidecar = {
        "model": out.name,
        "source": f"torchvision.models.mobilenet_v2 (weights={WEIGHTS})",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "onnx_version": onnx.__version__,
        "exporter": "torch.onnx.export (TorchScript, dynamo=False)",
        "opset": OPSET,
        "ir_version": proto.ir_version,
        "input": {"name": inp.name, "shape": list(INPUT_SHAPE), "dtype": "float32"},
        "output": {"name": session.get_outputs()[0].name, "shape": [1, 1000], "dtype": "float32",
                   "semantics": "logits"},
        "expected_preprocessing": PREPROCESS_DESCRIPTION,
        "probe_check": {"input_value": PROBE_INPUT_VALUE, "top1": top1, "top1_value": float(probe[top1]),
                        "expected_top1": PROBE_EXPECTED_TOP1},
        "torch_check": torch_check,
        "sha256": sha256_of(out),
        "size_bytes": out.stat().st_size,
    }
    sidecar_path = out.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {sidecar_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
