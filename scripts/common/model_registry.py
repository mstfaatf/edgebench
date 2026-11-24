"""Models EdgeBench exports, and the manifest of exported files (standard library only).

Each ``ModelSpec`` says which torchvision constructor or timm architecture and
which pretrained weights a model comes from, pins the weights file by SHA256
(and, for timm, by Hugging Face revision), and holds the mean/std the weights
were trained with. torchvision's file names only carry the first 8 hex digits
of the hash, and its download check only compares that prefix, so the full
hash is checked here after download.

Exported files are named ``<model>_<variant>_b<batch>.onnx`` and live in
``models/<model>/``. ``models/manifest.json`` (committed) lists the SHA256 and
size of every model file, what it is and which file it was made from (see the
manifest section below); ``scripts/export_models.py --verify`` re-exports
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
# edgebench.input_range: what the input tensor holds.
INPUT_RANGE_EXTERNAL = "(x - mean) / std for x in [0, 1]"
INPUT_RANGE_FOLDED = "[0, 1]"


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
#
# models/manifest.json is the committed index of every model file: its hash and
# size, what it is (from the file's edgebench.* metadata, which stays the source
# of truth) and the SHA256 of the file it was derived from. Following parents
# always ends at an FP32 export with external normalization (a "root").

MANIFEST_VERSION = "1.0.0"
MANIFEST_SCHEMA = REPO_ROOT / "schemas" / f"model-manifest.v{MANIFEST_VERSION}.schema.json"
ROOT_VARIANT = "fp32"
# Variant -> the variant of the file it is made from (roots have none).
PARENT_VARIANT = {"fp32_folded": "fp32", "int8": "fp32", "int8_unpruned": "fp32",
                  "int8_folded": "fp32_folded", "int8_folded_unpruned": "fp32_folded"}


class MetadataError(ValueError):
    """A model file's edgebench.* metadata lacks a key, or holds an unusable value."""


def meta_value(meta: dict[str, str], key: str, where: str = "") -> str:
    """``meta["edgebench.<key>"]``; MetadataError naming the key if it is missing or empty."""
    value = meta.get(META_PREFIX + key)
    if value is None or value == "":
        raise MetadataError(f"{where + ': ' if where else ''}metadata key "
                            f"'{META_PREFIX + key}' is missing")
    return value


@dataclass(frozen=True)
class ManifestFile:
    file: str          # relative to models/
    sha256: str
    size_bytes: int
    model: str
    variant: str
    precision: str     # "fp32" or "int8"
    normalization: str  # "external" or "folded"
    batch_size: int
    pruned: bool | None          # INT8 only: dead Q/DQ chains removed; None for FP32
    parent_sha256: str | None    # the file this one was made from; None for roots


ENTRY_KEYS = tuple(ManifestFile.__dataclass_fields__)


def manifest_entry(rel: str, sha256: str, size_bytes: int, meta: dict[str, str]) -> ManifestFile:
    """The manifest entry of a model file, from its edgebench.* metadata."""
    variant = meta_value(meta, "variant", rel)
    precision = meta_value(meta, "precision", rel)
    pruned = None
    if precision == "int8":
        flag = meta_value(meta, "quant.pruned", rel)
        if flag not in ("true", "false"):
            raise MetadataError(f"{rel}: '{META_PREFIX}quant.pruned' is {flag!r}")
        pruned = flag == "true"
    parent = None if variant == ROOT_VARIANT else meta_value(meta, "source_sha256", rel)
    return ManifestFile(rel, sha256, size_bytes, meta_value(meta, "model", rel), variant,
                        precision, meta_value(meta, "normalization", rel),
                        int(meta_value(meta, "batch_size", rel)), pruned, parent)


def format_manifest(files: list[ManifestFile], tools: dict[str, str]) -> str:
    doc = {
        "manifest_version": MANIFEST_VERSION,
        "description": "Every model file produced by scripts/export_models.py, fold_models.py "
                       "and quantize_models.py with the export tool versions below: SHA256, "
                       "size, what it is (from the file's edgebench.* metadata) and the SHA256 "
                       "of the file it was made from. Paths are relative to models/.",
        "tools": dict(sorted(tools.items())),
        "files": [{k: getattr(f, k) for k in ENTRY_KEYS}
                  for f in sorted(files, key=lambda f: f.file)],
    }
    return json.dumps(doc, indent=2) + "\n"


def parse_manifest(text: str) -> tuple[list[ManifestFile], dict[str, str]]:
    doc = json.loads(text)
    if doc.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"manifest_version is {doc.get('manifest_version')!r}, "
                         f"expected {MANIFEST_VERSION!r}")
    files = []
    for f in doc["files"]:
        missing = [k for k in ENTRY_KEYS if k not in f]
        if missing:
            raise ValueError(f"manifest entry {f.get('file')!r} lacks {', '.join(missing)}")
        files.append(ManifestFile(**{k: f[k] for k in ENTRY_KEYS}))
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


def lineage(files: list[ManifestFile], entry: ManifestFile) -> list[ManifestFile]:
    """``entry`` and its ancestors, nearest first, ending at a root; ValueError if broken."""
    by_sha = {f.sha256: f for f in files}
    chain = [entry]
    while chain[-1].parent_sha256 is not None:
        parent = by_sha.get(chain[-1].parent_sha256)
        if parent is None:
            raise ValueError(f"{chain[-1].file}: parent {chain[-1].parent_sha256} "
                             "is not in the manifest")
        if parent in chain:
            raise ValueError(f"{entry.file}: lineage has a cycle at {parent.file}")
        chain.append(parent)
    return chain


def lineage_problems(files: list[ManifestFile]) -> list[str]:
    """Everything wrong with the manifest's parent links (empty list: all fine).

    Every file must lead, through parents present in the manifest and without
    cycles, to a root: an FP32 export with external normalization. Each step
    must go to the expected parent variant of the same model and batch size.
    """
    problems = []
    shas = [f.sha256 for f in files]
    if len(set(shas)) != len(shas):
        problems.append("two manifest entries have the same SHA256")
    for f in files:
        if f.variant == ROOT_VARIANT:
            if f.parent_sha256 is not None:
                problems.append(f"{f.file}: a root (variant {ROOT_VARIANT}) has a parent")
            if f.normalization != "external" or f.precision != "fp32":
                problems.append(f"{f.file}: a root must be FP32 with external normalization")
            continue
        if f.variant not in PARENT_VARIANT:
            problems.append(f"{f.file}: unknown variant {f.variant!r}")
            continue
        try:
            chain = lineage(files, f)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        parent = chain[1]
        if parent.variant != PARENT_VARIANT[f.variant]:
            problems.append(f"{f.file}: parent {parent.file} is {parent.variant}, "
                            f"expected {PARENT_VARIANT[f.variant]}")
        if (parent.model, parent.batch_size) != (f.model, f.batch_size):
            problems.append(f"{f.file}: parent {parent.file} is another model or batch size")
    return problems
