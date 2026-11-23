"""Dataset split manifests and ImageNet-1k labels (standard library only).

A manifest pins the exact images of a split: one row per image with its path
(relative to the dataset root, forward slashes), its ImageNet-1k class index and
the SHA256 of the file bytes. Manifests are committed under ``datasets/``; the
images themselves are downloaded by ``scripts/get_data.py`` into ``data/`` and
are never committed.

File format: UTF-8, LF line endings, tab-separated, header ``path label sha256``,
rows sorted by (label, path) with the label compared as an integer.

This module has no third-party dependencies so that the manifest, label and
sampling checks in ``scripts/tests/`` run anywhere, including CI machines that
do not have the images.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATASETS_DIR = REPO_ROOT / "datasets"
IMAGENET_CLASSES_JSON = Path(__file__).resolve().parent / "imagenet_classes.json"

NUM_CLASSES = 1000

# ImageNetV2 matched-frequency, as extracted by scripts/get_data.py.
IMAGENETV2_DIR = DATA_DIR / "imagenetv2"
IMAGENETV2_ROOT = IMAGENETV2_DIR / "imagenetv2-matched-frequency-format-val"
IMAGENETV2_MANIFEST_DIR = DATASETS_DIR / "imagenetv2-matched-frequency"
EVAL_FULL_MANIFEST = IMAGENETV2_MANIFEST_DIR / "eval-full.tsv"
EVAL_QUICK_MANIFEST = IMAGENETV2_MANIFEST_DIR / "eval-quick.tsv"
EVAL_QUICK_SEED = 20261123
EVAL_QUICK_PER_CLASS = 1

# Imagenette calibration images (paths relative to data/imagenette/imagenette2-320/).
CALIBRATION_MANIFEST = DATASETS_DIR / "imagenette" / "calibration.tsv"

HEADER = ("path", "label", "sha256")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CLASS_DIR_RE = re.compile(r"0|[1-9][0-9]{0,2}")


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    label: int
    sha256: str


@dataclass(frozen=True)
class ImageNetClass:
    index: int
    wnid: str
    name: str


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_imagenet_classes(path: Path = IMAGENET_CLASSES_JSON) -> list[ImageNetClass]:
    """The 1000 ImageNet-1k classes in class-index order (the torchvision order)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ImageNetClass(**c) for c in data["classes"]]


def imagenetv2_label(relpath: str) -> int:
    """Class index of an ImageNetV2 image: its class directory name read as an integer.

    ImageNetV2 names its class directories ``0`` .. ``999`` after the ImageNet-1k
    class index. They must be parsed as integers; sorting them as strings
    (``0, 1, 10, 100, ...``), as a generic image-folder loader does, assigns wrong
    labels to almost every class.
    """
    parts = relpath.split("/")
    if len(parts) != 2 or not _CLASS_DIR_RE.fullmatch(parts[0]):
        raise ValueError(f"not an ImageNetV2 image path '<class index>/<file>': {relpath!r}")
    label = int(parts[0])
    if label >= NUM_CLASSES:
        raise ValueError(f"class index {label} out of range in {relpath!r}")
    return label


def sort_key(entry: ManifestEntry) -> tuple[int, str]:
    return entry.label, entry.path


def sort_entries(entries: Iterable[ManifestEntry]) -> list[ManifestEntry]:
    return sorted(entries, key=sort_key)


def format_manifest(entries: Iterable[ManifestEntry]) -> str:
    rows = ["\t".join(HEADER)]
    rows += [f"{e.path}\t{e.label}\t{e.sha256}" for e in sort_entries(entries)]
    return "\n".join(rows) + "\n"


def write_manifest(path: Path, entries: Iterable[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_manifest(entries), encoding="utf-8", newline="\n")


def parse_manifest(text: str, source: str = "<manifest>") -> list[ManifestEntry]:
    """Parse and check a manifest: header, field formats, unique paths, sort order."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or tuple(lines[0].split("\t")) != HEADER:
        raise ValueError(f"{source}: header must be {chr(9).join(HEADER)!r}")
    entries: list[ManifestEntry] = []
    for n, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"{source}:{n}: expected 3 tab-separated fields, got {len(fields)}")
        path, label_text, digest = fields
        if not path or "\\" in path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"{source}:{n}: path must be relative with forward slashes: {path!r}")
        if not re.fullmatch(r"0|[1-9][0-9]*", label_text) or int(label_text) >= NUM_CLASSES:
            raise ValueError(f"{source}:{n}: label must be an integer in [0, {NUM_CLASSES}): "
                             f"{label_text!r}")
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"{source}:{n}: sha256 must be 64 lowercase hex digits")
        entries.append(ManifestEntry(path, int(label_text), digest))
    paths = [e.path for e in entries]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{source}: duplicate paths")
    if entries != sort_entries(entries):
        raise ValueError(f"{source}: rows must be sorted by (label as integer, path)")
    return entries


def read_manifest(path: Path) -> list[ManifestEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest {path} not found")
    return parse_manifest(path.read_text(encoding="utf-8"), str(path))


def stratified_sample(entries: Iterable[ManifestEntry], per_class: int,
                      seed: int) -> list[ManifestEntry]:
    """Draw ``per_class`` images from every class with ``random.Random(seed)``.

    Classes are visited in increasing class index, and each class's candidates
    in path order, so the result depends only on the set of entries and the seed.
    """
    by_label: dict[int, list[ManifestEntry]] = {}
    for e in entries:
        by_label.setdefault(e.label, []).append(e)
    rng = random.Random(seed)
    chosen: list[ManifestEntry] = []
    for label in sorted(by_label):
        candidates = sorted(by_label[label], key=sort_key)
        if len(candidates) < per_class:
            raise ValueError(f"class {label} has {len(candidates)} images, need {per_class}")
        chosen += rng.sample(candidates, per_class)
    return sort_entries(chosen)


def content_overlap(a: Iterable[ManifestEntry],
                    b: Iterable[ManifestEntry]) -> list[tuple[ManifestEntry, ManifestEntry]]:
    """Pairs of entries from ``a`` and ``b`` whose files have identical bytes."""
    by_hash: dict[str, list[ManifestEntry]] = {}
    for e in a:
        by_hash.setdefault(e.sha256, []).append(e)
    return [(x, y) for y in b for x in by_hash.get(y.sha256, [])]


def duplicate_contents(entries: Iterable[ManifestEntry]) -> dict[str, list[ManifestEntry]]:
    """Groups of entries within one manifest that share the same file bytes."""
    by_hash: dict[str, list[ManifestEntry]] = {}
    for e in entries:
        by_hash.setdefault(e.sha256, []).append(e)
    return {h: group for h, group in by_hash.items() if len(group) > 1}


def verify_files(entries: Iterable[ManifestEntry], root: Path) -> list[str]:
    """Hash every listed file under ``root``; return one message per problem."""
    problems = []
    for e in entries:
        path = root / e.path
        if not path.is_file():
            problems.append(f"missing: {e.path}")
        elif sha256_of(path) != e.sha256:
            problems.append(f"content differs: {e.path}")
    return problems
