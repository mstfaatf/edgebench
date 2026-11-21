"""Measure MobileNetV2 FP32 and INT8 QDQ accuracy on the ONNX Runtime CPU EP.

Runs every model on the ``eval`` split from ``scripts/get_data.py`` (500
Imagenette val images, 50 per class), preprocessed with
``scripts/common/preprocess.py``, and reports:

- top-1 accuracy over all 1000 ImageNet classes,
- top-1 accuracy restricted to the 10 Imagenette classes (argmax over just
  those 10 logits),
- for each INT8 model, top-1 agreement with the FP32 model (fraction of images
  with the same predicted class, for both definitions above).

Note on the numbers: most Imagenette val images are taken from the ImageNet-1k
*train* set, which the pretrained weights were trained on, so absolute accuracy
is optimistic. Comparisons between FP32 and INT8 are unaffected. The summary
records how many eval images come from ImageNet val.

A JSON summary is written to ``benchmarks/results/local/feasibility/``.

Usage:
    python scripts/eval_cpu.py [--output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402
from common.preprocess import (  # noqa: E402
    IMAGENETTE_ROOT,
    PREPROCESS_DESCRIPTION,
    REPO_ROOT,
    SPLITS_DIR,
    load_image,
    load_imagenette_classes,
    read_split,
    wnid_of,
)

MODEL_DIR = REPO_ROOT / "models" / "mobilenetv2"
REFERENCE = "fp32"
MODELS = {
    "fp32": "mobilenetv2_fp32_b1.onnx",
    "int8_qdq_per_tensor": "mobilenetv2_int8qdq_b1.onnx",
    "int8_qdq_per_channel": "mobilenetv2_int8qdq_pc_b1.onnx",
}
EVAL_SPLIT = "eval"
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "results" / "local" / "feasibility" / "cpu_accuracy_mobilenetv2.json"
)


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
    import onnxruntime as ort

    missing = [name for name in MODELS.values() if not (MODEL_DIR / name).is_file()]
    if missing:
        print(f"missing models in {MODEL_DIR}: {missing}\n"
              "run scripts/export_mobilenetv2.py and scripts/quantize_qnn.py first", file=sys.stderr)
        return 1

    classes = load_imagenette_classes()
    class_indices = np.array([c.imagenet_index for c in classes])
    wnid_to_index = {c.wnid: c.imagenet_index for c in classes}
    images = read_split(EVAL_SPLIT)
    labels = np.array([wnid_to_index[wnid_of(p)] for p in images])
    from_imagenet_val = sum(Path(p).name.startswith("ILSVRC2012_val_") for p in images)

    print(f"preprocessing {len(images)} eval images")
    inputs = [load_image(IMAGENETTE_ROOT / p) for p in images]

    preds_1000: dict[str, np.ndarray] = {}
    preds_10: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    for key, filename in MODELS.items():
        path = MODEL_DIR / filename
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        logits = np.stack([session.run(None, {input_name: x})[0][0] for x in inputs])
        preds_1000[key] = logits.argmax(axis=1)
        preds_10[key] = class_indices[logits[:, class_indices].argmax(axis=1)]
        results[key] = {
            "model": filename,
            "sha256": sha256_of(path),
            "top1_1000": float(np.mean(preds_1000[key] == labels)),
            "top1_imagenette10": float(np.mean(preds_10[key] == labels)),
            "correct_1000": int(np.sum(preds_1000[key] == labels)),
            "correct_imagenette10": int(np.sum(preds_10[key] == labels)),
        }

    for key, r in results.items():
        if key == REFERENCE:
            r["agreement_with_fp32"] = None
            continue
        r["agreement_with_fp32"] = {
            "top1_1000": float(np.mean(preds_1000[key] == preds_1000[REFERENCE])),
            "top1_imagenette10": float(np.mean(preds_10[key] == preds_10[REFERENCE])),
        }

    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "execution_provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dataset": {
            "name": "imagenette2-320",
            "split": EVAL_SPLIT,
            "split_file": f"data/imagenette/splits/{EVAL_SPLIT}.txt",
            "split_file_sha256": sha256_of(SPLITS_DIR / f"{EVAL_SPLIT}.txt"),
            "count": len(images),
            "images_from_imagenet_val": from_imagenet_val,
            "note": "Most Imagenette val images come from the ImageNet-1k train set, so absolute "
                    "accuracy is optimistic for ImageNet-pretrained weights; FP32 vs INT8 "
                    "comparisons are unaffected.",
        },
        "preprocessing": PREPROCESS_DESCRIPTION,
        "metrics": {
            "top1_1000": "argmax over all 1000 ImageNet logits equals the label",
            "top1_imagenette10": "argmax over the 10 Imagenette class logits equals the label",
            "agreement_with_fp32": "fraction of images whose predicted class matches the FP32 model",
        },
        "results": results,
    }

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'model':<22} {'top1/1000':>10} {'top1/10':>9} {'agree/1000':>11} {'agree/10':>9}")
    for key, r in results.items():
        agree = r["agreement_with_fp32"]
        a1000 = f"{agree['top1_1000']:.4f}" if agree else "-"
        a10 = f"{agree['top1_imagenette10']:.4f}" if agree else "-"
        print(f"{key:<22} {r['top1_1000']:>10.4f} {r['top1_imagenette10']:>9.4f} {a1000:>11} {a10:>9}")
    print(f"\n{from_imagenet_val} of {len(images)} eval images are from ImageNet val "
          "(the rest are ImageNet train images)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
