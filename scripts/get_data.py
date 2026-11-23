"""Download the evaluation and calibration datasets and check them against their manifests.

Two datasets, chosen with ``--dataset``:

``imagenette`` (default)
    Imagenette (https://github.com/fastai/imagenette), a 10-class subset of
    ImageNet. Downloads ``imagenette2-320.tgz`` into ``data/downloads/``
    (skipped if it is already there), verifies its SHA256 on every run, extracts
    it to ``data/imagenette/imagenette2-320/`` and writes two deterministic,
    seeded image lists to ``data/imagenette/splits/``:

    - ``calibration.txt``: 200 images from the train split (20 per class), used
      to calibrate static quantization,
    - ``eval.txt``: 500 images from the val split (50 per class), used for the
      variant comparisons in the feasibility study.

    Paths are relative to ``data/imagenette/imagenette2-320/``, one per line,
    sorted. The calibration images are also checked against the committed
    manifest ``datasets/imagenette/calibration.tsv``.

``imagenetv2``
    ImageNetV2 matched-frequency (https://github.com/modestyachts/ImageNetV2),
    10,000 images, 10 per ImageNet-1k class, collected after the original
    ImageNet and disjoint from its training set. This is the accuracy
    evaluation set. Downloads the archive from the dataset's Hugging Face
    repository at a pinned revision, verifies its SHA256, extracts it to
    ``data/imagenetv2/`` and checks the images against the committed manifests
    ``datasets/imagenetv2-matched-frequency/eval-full.tsv`` (all images) and
    ``eval-quick.tsv`` (1 image per class, seeded). Once extracted, later runs
    skip the download and only compare the file list with the manifest;
    ``--verify`` hashes the archive (if present) and every image again.

``all`` does both. ``data/`` is not committed; the manifests are.

Usage:
    python scripts/get_data.py [--dataset imagenette|imagenetv2|all]
                               [--verify] [--force-extract] [--write-manifests]
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.manifests import (  # noqa: E402
    CALIBRATION_MANIFEST,
    EVAL_FULL_MANIFEST,
    EVAL_QUICK_MANIFEST,
    EVAL_QUICK_PER_CLASS,
    EVAL_QUICK_SEED,
    IMAGENETV2_DIR,
    IMAGENETV2_ROOT,
    ManifestEntry,
    content_overlap,
    imagenetv2_label,
    read_manifest,
    sha256_of,
    stratified_sample,
    verify_files,
    write_manifest,
)
from common.preprocess import DATA_DIR, IMAGENETTE_ROOT, SPLITS_DIR, load_imagenette_classes  # noqa: E402

DOWNLOADS_DIR = DATA_DIR / "downloads"

ARCHIVE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
ARCHIVE_SHA256 = "569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c"
ARCHIVE_PATH = DOWNLOADS_DIR / "imagenette2-320.tgz"

# Hugging Face dataset repository linked from the ImageNetV2 README, pinned to a
# revision. The SHA256 is the one Hugging Face publishes for the file (its LFS
# object id). Despite the .tar.gz name, the file is an uncompressed tar.
IMAGENETV2_REVISION = "d626240be2538720e83103a0e1178d24aca8b12c"
IMAGENETV2_URL = ("https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/"
                  f"{IMAGENETV2_REVISION}/imagenetv2-matched-frequency.tar.gz")
IMAGENETV2_SHA256 = "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c"
IMAGENETV2_SIZE = 1_264_079_360
IMAGENETV2_ARCHIVE = DOWNLOADS_DIR / "imagenetv2-matched-frequency.tar.gz"
IMAGENETV2_IMAGES = 10_000

SEED = 20261120
CALIBRATION_PER_CLASS = 20  # 200 images
EVAL_PER_CLASS = 50  # 500 images
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, next_report = 0, 0.1
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                print(f"  {done / total:.0%} of {total / 1e6:.0f} MB", flush=True)
                next_report += 0.1
    tmp.replace(dest)


def ensure_verified_archive(url: str, path: Path, expected: str) -> None:
    if not path.is_file():
        download(url, path)
    digest = sha256_of(path)
    if digest != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {path}\n  expected {expected}\n  got      {digest}\n"
            "Delete the file and re-run to download it again."
        )
    print(f"archive OK: {path} (sha256 {digest})")


def extracted_marker_ok(root: Path, archive_sha256: str) -> bool:
    marker = root / ".extracted"
    return marker.is_file() and marker.read_text().strip() == archive_sha256


def extract(archive: Path, root: Path, archive_sha256: str) -> None:
    """Extract ``archive`` (gzip or plain tar) whose single top-level directory is ``root``."""
    if root.exists():
        shutil.rmtree(root)
    print(f"extracting to {root.parent}")
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(root.parent, filter="data")
    if not root.is_dir():
        raise RuntimeError(f"{archive} did not contain {root.name}/")
    (root / ".extracted").write_text(archive_sha256 + "\n")


# --- Imagenette ---------------------------------------------------------------


def list_images(split: str, wnid: str) -> list[str]:
    class_dir = IMAGENETTE_ROOT / split / wnid
    if not class_dir.is_dir():
        raise FileNotFoundError(f"missing class directory {class_dir}")
    return sorted(
        f"{split}/{wnid}/{p.name}" for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )


def sample_split(split: str, per_class: int, rng: random.Random) -> list[str]:
    chosen: list[str] = []
    for cls in load_imagenette_classes():
        images = list_images(split, cls.wnid)
        if len(images) < per_class:
            raise RuntimeError(f"{split}/{cls.wnid} has {len(images)} images, need {per_class}")
        chosen += rng.sample(images, per_class)
    return sorted(chosen)


def write_split(name: str, paths: list[str]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out = SPLITS_DIR / f"{name}.txt"
    out.write_text("".join(p + "\n" for p in paths), encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(paths)} images, sha256 {sha256_of(out)})")


def calibration_entries(paths: list[str]) -> list[ManifestEntry]:
    index_of = {c.wnid: c.imagenet_index for c in load_imagenette_classes()}
    return [ManifestEntry(p, index_of[Path(p).parent.name], sha256_of(IMAGENETTE_ROOT / p))
            for p in paths]


def get_imagenette(force_extract: bool, write_manifests: bool) -> None:
    ensure_verified_archive(ARCHIVE_URL, ARCHIVE_PATH, ARCHIVE_SHA256)
    if extracted_marker_ok(IMAGENETTE_ROOT, ARCHIVE_SHA256) and not force_extract:
        print(f"already extracted: {IMAGENETTE_ROOT}")
    else:
        extract(ARCHIVE_PATH, IMAGENETTE_ROOT, ARCHIVE_SHA256)

    rng = random.Random(SEED)
    calibration = sample_split("train", CALIBRATION_PER_CLASS, rng)
    evaluation = sample_split("val", EVAL_PER_CLASS, rng)
    overlap = set(calibration) & set(evaluation)
    if overlap:
        raise RuntimeError(f"calibration and eval splits overlap: {sorted(overlap)[:5]}")

    write_split("calibration", calibration)
    write_split("eval", evaluation)

    entries = calibration_entries(calibration)
    if write_manifests:
        write_manifest(CALIBRATION_MANIFEST, entries)
        print(f"wrote {CALIBRATION_MANIFEST} ({len(entries)} images)")
    elif entries != read_manifest(CALIBRATION_MANIFEST):
        raise RuntimeError(f"the calibration split differs from {CALIBRATION_MANIFEST}")
    else:
        print(f"calibration split matches {CALIBRATION_MANIFEST.relative_to(DATA_DIR.parent)}")


# --- ImageNetV2 ---------------------------------------------------------------


def list_imagenetv2_files() -> list[str]:
    """All image paths under the extracted root, relative, with forward slashes."""
    paths = []
    for p in IMAGENETV2_ROOT.rglob("*"):
        if p.is_dir() or p.name == ".extracted":
            continue
        rel = p.relative_to(IMAGENETV2_ROOT).as_posix()
        imagenetv2_label(rel)  # raises on anything but <class index>/<file>
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            raise RuntimeError(f"unexpected file in {IMAGENETV2_ROOT}: {rel}")
        paths.append(rel)
    return paths


def imagenetv2_entries() -> list[ManifestEntry]:
    return [ManifestEntry(p, imagenetv2_label(p), sha256_of(IMAGENETV2_ROOT / p))
            for p in list_imagenetv2_files()]


def get_imagenetv2(verify: bool, force_extract: bool, write_manifests: bool) -> None:
    fresh = force_extract or not extracted_marker_ok(IMAGENETV2_ROOT, IMAGENETV2_SHA256)
    if fresh or verify:
        if not fresh and not IMAGENETV2_ARCHIVE.is_file():
            print(f"archive not present ({IMAGENETV2_ARCHIVE}); checking the extracted images only")
        else:
            ensure_verified_archive(IMAGENETV2_URL, IMAGENETV2_ARCHIVE, IMAGENETV2_SHA256)
    if fresh:
        IMAGENETV2_DIR.mkdir(parents=True, exist_ok=True)
        extract(IMAGENETV2_ARCHIVE, IMAGENETV2_ROOT, IMAGENETV2_SHA256)
    else:
        print(f"already extracted: {IMAGENETV2_ROOT}")

    if write_manifests:
        full = imagenetv2_entries()
        if len(full) != IMAGENETV2_IMAGES:
            raise RuntimeError(f"expected {IMAGENETV2_IMAGES} images, found {len(full)}")
        quick = stratified_sample(full, EVAL_QUICK_PER_CLASS, EVAL_QUICK_SEED)
        write_manifest(EVAL_FULL_MANIFEST, full)
        write_manifest(EVAL_QUICK_MANIFEST, quick)
        print(f"wrote {EVAL_FULL_MANIFEST} ({len(full)} images) and "
              f"{EVAL_QUICK_MANIFEST} ({len(quick)} images)")

    full = read_manifest(EVAL_FULL_MANIFEST)
    quick = read_manifest(EVAL_QUICK_MANIFEST)
    if quick != stratified_sample(full, EVAL_QUICK_PER_CLASS, EVAL_QUICK_SEED):
        raise RuntimeError(f"{EVAL_QUICK_MANIFEST} is not the seeded sample of {EVAL_FULL_MANIFEST}")

    on_disk = sorted(list_imagenetv2_files())
    listed = sorted(e.path for e in full)
    if on_disk != listed:
        missing, extra = set(listed) - set(on_disk), set(on_disk) - set(listed)
        raise RuntimeError(f"extracted images differ from {EVAL_FULL_MANIFEST}: "
                           f"{len(missing)} missing, {len(extra)} not listed; "
                           "re-run with --force-extract")
    if fresh or verify:
        print(f"hashing {len(full)} images")
        problems = verify_files(full, IMAGENETV2_ROOT)
        if problems:
            raise RuntimeError(f"{len(problems)} images do not match the manifest, e.g. "
                               f"{problems[:3]}; re-run with --force-extract")
        print(f"all {len(full)} images match {EVAL_FULL_MANIFEST.name}")
    else:
        print(f"{len(full)} images present (file list matches; --verify hashes them)")

    overlap = content_overlap(read_manifest(CALIBRATION_MANIFEST), full)
    if overlap:
        raise RuntimeError(f"calibration and evaluation images overlap: {overlap[:3]}")
    print(f"eval-full: {len(full)} images, eval-quick: {len(quick)} images "
          f"(seed {EVAL_QUICK_SEED}); no content overlap with the calibration images")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["imagenette", "imagenetv2", "all"],
                        default="imagenette", help="which dataset to fetch (default: imagenette)")
    parser.add_argument("--verify", action="store_true",
                        help="ImageNetV2: hash the archive (if present) and every image again")
    parser.add_argument("--force-extract", action="store_true", help="re-extract the archive")
    parser.add_argument("--write-manifests", action="store_true",
                        help="regenerate the committed manifests under datasets/ from the "
                             "extracted images (for maintainers)")
    args = parser.parse_args()

    if args.dataset in ("imagenette", "all"):
        get_imagenette(args.force_extract, args.write_manifests)
    if args.dataset in ("imagenetv2", "all"):
        get_imagenetv2(args.verify, args.force_extract, args.write_manifests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
