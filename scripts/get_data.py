"""Download Imagenette and build the calibration and evaluation splits.

Imagenette (https://github.com/fastai/imagenette) is a 10-class subset of
ImageNet. This script:

1. downloads ``imagenette2-320.tgz`` into ``data/downloads/`` (skipped if it is
   already there) and verifies its SHA256,
2. extracts it to ``data/imagenette/imagenette2-320/``,
3. writes two deterministic, seeded image lists to ``data/imagenette/splits/``:

   - ``calibration.txt``: 200 images from the train split (20 per class), used
     to calibrate static quantization,
   - ``eval.txt``: 500 images from the val split (50 per class), used to
     measure accuracy.

   Paths are relative to ``data/imagenette/imagenette2-320/``, one per line,
   sorted. The two lists come from different upstream splits, so they never
   overlap.

``data/`` is not committed. Re-running the script verifies the archive again
and rewrites identical split files.

Usage:
    python scripts/get_data.py [--force-extract]
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.preprocess import DATA_DIR, IMAGENETTE_ROOT, SPLITS_DIR, load_imagenette_classes  # noqa: E402

ARCHIVE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
ARCHIVE_SHA256 = "569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c"
ARCHIVE_PATH = DATA_DIR / "downloads" / "imagenette2-320.tgz"

SEED = 20261120
CALIBRATION_PER_CLASS = 20  # 200 images
EVAL_PER_CLASS = 50  # 500 images
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)


def ensure_archive() -> None:
    if not ARCHIVE_PATH.is_file():
        download(ARCHIVE_URL, ARCHIVE_PATH)
    digest = sha256_of(ARCHIVE_PATH)
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(
            f"SHA256 mismatch for {ARCHIVE_PATH}\n  expected {ARCHIVE_SHA256}\n  got      {digest}\n"
            "Delete the file and re-run to download it again."
        )
    print(f"archive OK: {ARCHIVE_PATH} (sha256 {digest})")


def ensure_extracted(force: bool) -> None:
    marker = IMAGENETTE_ROOT / ".extracted"
    if marker.is_file() and marker.read_text().strip() == ARCHIVE_SHA256 and not force:
        print(f"already extracted: {IMAGENETTE_ROOT}")
        return
    if IMAGENETTE_ROOT.exists():
        shutil.rmtree(IMAGENETTE_ROOT)
    print(f"extracting to {IMAGENETTE_ROOT.parent}")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(IMAGENETTE_ROOT.parent, filter="data")
    marker.write_text(ARCHIVE_SHA256 + "\n")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force-extract", action="store_true", help="re-extract the archive")
    args = parser.parse_args()

    ensure_archive()
    ensure_extracted(args.force_extract)

    rng = random.Random(SEED)
    calibration = sample_split("train", CALIBRATION_PER_CLASS, rng)
    evaluation = sample_split("val", EVAL_PER_CLASS, rng)
    overlap = set(calibration) & set(evaluation)
    if overlap:
        raise RuntimeError(f"calibration and eval splits overlap: {sorted(overlap)[:5]}")

    write_split("calibration", calibration)
    write_split("eval", evaluation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
