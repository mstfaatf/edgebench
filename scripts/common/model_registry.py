"""Models EdgeBench exports, and the manifest of exported files (standard library only).

Each ``ModelSpec`` says which torchvision constructor or timm architecture and
which pretrained weights a model comes from, pins the weights file by SHA256
(and, for timm, by Hugging Face revision), and holds the mean/std the weights
were trained with. torchvision's file names only carry the first 8 hex digits
of the hash, and its download check only compares that prefix, so the full
hash is checked here after download.

Exported files are named ``<model>_<variant>_b<batch>.onnx`` and live in
``models/<model>/``. ``models/manifest.json`` (committed) lists the SHA256 and
size of every exported file; ``scripts/export_models.py --verify`` re-exports
and compares against it.

To add a model: add a ``ModelSpec`` to ``MODELS``, run
``scripts/export_models.py --model <name>``, and commit the updated manifest.

This module does not import torch or onnx, so its tests run anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

BATCH_SIZES = (1, 4, 8)

# Export settings shared by every model. TorchScript-based exporter at opset 17:
# the settings the feasibility study validated on the CPU EP and the QNN EP.
OPSET = 17
EXPORTER = "torch.onnx.export (TorchScript, dynamo=False)"
INPUT_NAME = "input"
OUTPUT_NAME = "logits"
NUM_CLASSES = 1000

# Keys written to the ONNX model's metadata_props.
META_PREFIX = "edgebench."
META_NORMALIZATION = META_PREFIX + "normalization"
# Where ImageNet mean/std normalization happens, as in the results schema's
# model.normalization: "external" (the caller passes normalized input),
# "in_graph" (Sub/Div nodes in the model) or "folded" (merged into the first conv).
NORMALIZATIONS = ("external", "in_graph", "folded")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSpec:
    name: str                 # file and directory name, e.g. "mobilenetv2"
    display_name: str
    # torchvision: function in torchvision.models; timm: architecture name.
    constructor: str
    # torchvision: weights enum class in torchvision.models; timm: "" (unused).
    weights_enum: str
    # torchvision: member of weights_enum; timm: pretrained tag (model "<constructor>.<tag>").
    weights: str
    weights_url: str          # where the weights file is downloaded from
    weights_sha256: str       # full SHA256 of that file
    weights_size_bytes: int
    source: str = "torchvision"  # "torchvision" or "timm"
    # timm only: Hugging Face repository, pinned revision and file of the weights.
    hf_repo: str = ""
    hf_revision: str = ""
    weights_file: str = ""
    # Normalization the weights were trained with; the exported model expects
    # input normalized with exactly these (see common/preprocess.py).
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    input_size: int = 224     # square input, NCHW [batch, 3, size, size]
    # Optional self-check on a tensor filled with probe_value: expected top-1 class.
    probe_value: float | None = None
    probe_top1: int | None = None
    notes: str = ""

    @property
    def weights_id(self) -> str:
        if self.source == "timm":
            return f"timm {self.constructor}.{self.weights} ({self.hf_repo}@{self.hf_revision})"
        return f"torchvision {self.weights_enum}.{self.weights}"


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            name="mobilenetv2",
            display_name="MobileNetV2",
            constructor="mobilenet_v2",
            weights_enum="MobileNet_V2_Weights",
            weights="IMAGENET1K_V2",
            weights_url="https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
            weights_sha256="7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f",
            weights_size_bytes=14_258_573,
            # tools/ort_probe uses the same input (--fill 0.5).
            probe_value=0.5,
            probe_top1=677,
            notes="torchvision's improved training recipe; ImageNet-1k top-1 72.154 "
                  "as reported by torchvision (evaluated with resize 232).",
        ),
        ModelSpec(
            name="resnet18",
            display_name="ResNet-18",
            constructor="resnet18",
            weights_enum="ResNet18_Weights",
            weights="IMAGENET1K_V1",
            weights_url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
            weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            weights_size_bytes=46_830_571,
            notes="The only ImageNet weights torchvision provides for ResNet-18 (original "
                  "recipe); ImageNet-1k top-1 69.758 as reported by torchvision.",
        ),
        ModelSpec(
            name="efficientnet_lite0",
            display_name="EfficientNet-Lite0",
            source="timm",
            constructor="efficientnet_lite0",
            weights_enum="",
            weights="ra_in1k",
            hf_repo="timm/efficientnet_lite0.ra_in1k",
            hf_revision="29046ba6cc39eacfd5c7e6ffc1ae6188cfaa7609",
            weights_file="model.safetensors",
            weights_url="https://huggingface.co/timm/efficientnet_lite0.ra_in1k/resolve/"
                        "29046ba6cc39eacfd5c7e6ffc1ae6188cfaa7609/model.safetensors",
            weights_sha256="96c33bc4ce15e6f834b0f7da41d4bae33547fff1e88e0ba1e812a148693d5dd4",
            weights_size_bytes=18_802_812,
            # ImageNet mean/std (the defaults); timm evaluates these weights with a bicubic
            # resize and crop_pct 0.875 (= resize 256, crop 224).
            notes="Trained in timm (RandAugment recipe). Chosen over the ported TensorFlow "
                  "weights (tf_efficientnet_lite0.in1k), whose TF-style 'same' padding "
                  "exports as runtime shape arithmetic feeding Pad nodes, which does not "
                  "fold to constants; see docs/models.md.",
        ),
    )
}


def file_name(model: str, variant: str, batch: int) -> str:
    return f"{model}_{variant}_b{batch}.onnx"


def relative_path(model: str, variant: str, batch: int) -> str:
    """Path of an exported file relative to ``models/``, with forward slashes."""
    return f"{model}/{file_name(model, variant, batch)}"


def expected_files(models: list[str] | None = None, variant: str = "fp32",
                   batches: tuple[int, ...] = BATCH_SIZES) -> list[str]:
    names = sorted(MODELS) if models is None else models
    return sorted(relative_path(m, variant, b) for m in names for b in batches)


# Variants listed in the manifest: FP32 exports, pruned INT8 QDQ models, and the
# unpruned quantizer output kept as a reference; each also with the input
# normalization folded into the first convolution (input in [0, 1]).
VARIANTS = ("fp32", "int8", "int8_unpruned",
            "fp32_folded", "int8_folded", "int8_folded_unpruned")


def all_expected_files() -> list[str]:
    return sorted(f for v in VARIANTS for f in expected_files(variant=v))


# --- manifest ------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestFile:
    file: str          # relative to models/
    sha256: str
    size_bytes: int


def format_manifest(files: list[ManifestFile], tools: dict[str, str]) -> str:
    doc = {
        "description": "SHA256 and size of every model file produced by "
                       "scripts/export_models.py with the tool versions below. "
                       "Paths are relative to models/.",
        "tools": dict(sorted(tools.items())),
        "files": [{"file": f.file, "sha256": f.sha256, "size_bytes": f.size_bytes}
                  for f in sorted(files, key=lambda f: f.file)],
    }
    return json.dumps(doc, indent=2) + "\n"


def parse_manifest(text: str) -> tuple[list[ManifestFile], dict[str, str]]:
    doc = json.loads(text)
    files = [ManifestFile(f["file"], f["sha256"], int(f["size_bytes"])) for f in doc["files"]]
    names = [f.file for f in files]
    if names != sorted(names) or len(set(names)) != len(names):
        raise ValueError("manifest files must be unique and sorted by path")
    return files, dict(doc["tools"])


def read_manifest(path: Path = MANIFEST_PATH) -> tuple[list[ManifestFile], dict[str, str]]:
    if not path.is_file():
        return [], {}
    return parse_manifest(path.read_text(encoding="utf-8"))


def write_manifest(files: list[ManifestFile], tools: dict[str, str],
                   path: Path = MANIFEST_PATH) -> None:
    path.write_text(format_manifest(files, tools), encoding="utf-8", newline="\n")
