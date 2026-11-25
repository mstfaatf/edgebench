"""ImageNet-1k (1000-class) evaluation on ImageNetV2: labels, image cache, inference.

``common/accuracy.py`` keeps the 10-class Imagenette metrics of the feasibility
study; this module is the 1000-class path used by ``scripts/eval_accuracy.py``.

Images: the eval-full or eval-quick manifest (``common/manifests.py``), in
manifest order. Each label is the manifest's, which must equal the integer
folder name (``imagenetv2_label``; a string sort of the folder names would
mislabel 988 of 1000 classes) and index ``common/imagenet_classes.json``.

Image cache: decoding and resizing 10,000 JPEGs takes minutes, so the resized
and center-cropped uint8 images (``preprocess_image_uint8``, steps 1-3 of the
shared transform) are stored once in ``data/cache/imagenetv2/`` (not
committed). Converting a cached image with ``unit_from_uint8`` is exactly what
``load_image(..., normalized=False)`` does after cropping, so the tensors are
bit-identical; ``verify_cache`` checks that on a sample of images before the
cache is used. Normalization (``normalize`` with the model's own mean/std) is
applied afterwards for external-normalization models; folded models take the
[0, 1] tensor as it is.

Needs NumPy and Pillow.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common.manifests import (
    DATA_DIR,
    IMAGENETV2_ROOT,
    ManifestEntry,
    imagenetv2_label,
    load_imagenet_classes,
    read_manifest,
    sha256_of,
)
from common.preprocess import (
    PREPROCESS_DESCRIPTION,
    load_image,
    normalize,
    preprocess_image_uint8,
    unit_from_uint8,
)

CACHE_DIR = DATA_DIR / "cache" / "imagenetv2"
TOP_K = 5
CHUNK = 200  # images converted to float32 at a time (multiple of every batch size)


class LabelError(ValueError):
    """A manifest label does not match the image's folder or the class list."""


@dataclass(frozen=True)
class EvalSet:
    name: str                     # "eval-full" or "eval-quick"
    manifest: Path
    manifest_sha256: str
    entries: list[ManifestEntry]
    labels: np.ndarray            # [N] int64, ImageNet-1k index


def load_eval_set(name: str, manifest: Path) -> EvalSet:
    """The split with every label checked against its folder name and the class list."""
    entries = read_manifest(manifest)
    classes = load_imagenet_classes()
    if [c.index for c in classes] != list(range(1000)):
        raise LabelError("imagenet_classes.json must list indices 0..999 in order")
    for e in entries:
        if imagenetv2_label(e.path) != e.label:
            raise LabelError(f"{e.path}: manifest label {e.label} is not its folder's")
    return EvalSet(name, manifest, sha256_of(manifest), entries,
                   np.array([e.label for e in entries], dtype=np.int64))


# --- image cache ----------------------------------------------------------------


def cache_paths(eval_set: EvalSet) -> tuple[Path, Path]:
    stem = f"{eval_set.name}-{eval_set.manifest_sha256[:12]}-u8"
    return CACHE_DIR / f"{stem}.npy", CACHE_DIR / f"{stem}.json"


def build_cache(eval_set: EvalSet, progress: Callable[[str], None] = print) -> np.ndarray:
    """The [N, 224, 224, 3] uint8 images of the split, from the cache or built now."""
    from PIL import Image

    array_path, meta_path = cache_paths(eval_set)
    meta = {"manifest": eval_set.manifest.name, "manifest_sha256": eval_set.manifest_sha256,
            "count": len(eval_set.entries), "shape": [224, 224, 3], "dtype": "uint8",
            "preprocessing": {k: PREPROCESS_DESCRIPTION[k] for k in
                              ("resize_shorter_side", "interpolation", "center_crop")}}
    if array_path.is_file() and meta_path.is_file() and \
            json.loads(meta_path.read_text(encoding="utf-8")) == meta:
        return np.load(array_path, mmap_mode="r")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = array_path.with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                    shape=(len(eval_set.entries), 224, 224, 3))
    for i, e in enumerate(eval_set.entries):
        with Image.open(IMAGENETV2_ROOT / e.path) as img:
            out[i] = preprocess_image_uint8(img)
        if (i + 1) % 1000 == 0:
            progress(f"  cached {i + 1}/{len(eval_set.entries)} images")
    out.flush()
    del out
    tmp.replace(array_path)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return np.load(array_path, mmap_mode="r")


def verify_cache(eval_set: EvalSet, cache: np.ndarray, sample: int = 200,
                 seed: int = 20261125) -> dict[str, Any]:
    """Compare cached tensors with ``load_image`` on a seeded sample; raises if any differs."""
    indices = sorted(random.Random(seed).sample(range(len(eval_set.entries)), sample))
    for i in indices:
        expected = load_image(IMAGENETV2_ROOT / eval_set.entries[i].path, normalized=False)
        if not np.array_equal(unit_from_uint8(cache[i]), expected):
            raise RuntimeError(f"cached image {eval_set.entries[i].path} differs from load_image")
    return {"sample": sample, "seed": seed, "bit_identical": True,
            "compared_with": "common.preprocess.load_image(normalized=False)"}


def model_input(cache: np.ndarray, start: int, stop: int, normalization: str,
                mean, std) -> np.ndarray:
    """The input tensor for images [start, stop) as a file of this normalization expects it."""
    unit = unit_from_uint8(np.asarray(cache[start:stop]))
    if normalization == "folded":
        return unit
    return normalize(unit, tuple(mean), tuple(std))


# --- inference and metrics ------------------------------------------------------


def top_k(logits: np.ndarray, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the k largest logits per row; ties broken by lower index."""
    order = np.argsort(-logits.astype(np.float64), axis=1, kind="stable")[:, :k]
    return order.astype(np.int16), np.take_along_axis(logits, order, axis=1).astype(np.float32)


def predict(session: Any, cache: np.ndarray, batch: int, normalization: str, mean, std,
            count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Top-5 indices [N, 5] and logits [N, 5] for the first ``count`` images (default all)."""
    n = len(cache) if count is None else count
    if n % batch:
        raise ValueError(f"{n} images do not split into batches of {batch}")
    name = session.get_inputs()[0].name
    indices, scores = [], []
    for start in range(0, n, CHUNK):
        x = model_input(cache, start, min(start + CHUNK, n), normalization, mean, std)
        for b in range(0, len(x), batch):
            idx, val = top_k(session.run(None, {name: x[b:b + batch]})[0])
            indices.append(idx)
            scores.append(val)
    return np.concatenate(indices), np.concatenate(scores)


def correct(top5: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-image top-1 and top-5 correctness."""
    return top5[:, 0] == labels, np.any(top5 == labels[:, None], axis=1)
