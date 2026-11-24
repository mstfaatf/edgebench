"""Models EdgeBench exports, and the manifest of exported files (standard library only).

Each ``ModelSpec`` says which torchvision constructor and pretrained weights a
model comes from, and pins the weights file by SHA256. torchvision's file
names only carry the first 8 hex digits of the hash, and its download check
only compares that prefix, so the full hash is checked here after download.

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


@dataclass(frozen=True)
class ModelSpec:
    name: str                 # file and directory name, e.g. "mobilenetv2"
    display_name: str
    constructor: str          # function in torchvision.models
    weights_enum: str         # class in torchvision.models
    weights: str              # member of weights_enum
    weights_url: str          # what torchvision downloads for these weights
    weights_sha256: str       # full SHA256 of that file
    weights_size_bytes: int
    input_size: int = 224     # square input, NCHW [batch, 3, size, size]
    # Optional self-check on a tensor filled with probe_value: expected top-1 class.
    probe_value: float | None = None
    probe_top1: int | None = None
    notes: str = ""

    @property
    def weights_id(self) -> str:
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
