"""Shared image preprocessing and dataset helpers for ImageNet-style classifiers.

Every script that feeds images to a model (calibration, accuracy evaluation,
reference-output generation for the C++ engine) must use this module, so that
all of them see bit-identical input tensors.

The transform is the standard torchvision ImageNet evaluation pipeline:

1. decode and convert to RGB,
2. resize so the shorter side is 256 (bilinear, PIL),
3. center crop 224x224,
4. scale to [0, 1],
5. normalize with a per-channel mean/std (ImageNet by default; per model in
   ``common.model_registry``),
6. lay out as NCHW float32 with a leading batch dimension of 1.

It is implemented with PIL and NumPy only (no torch import), and matches
``torchvision.transforms`` ``Resize(256)``, ``CenterCrop(224)``, ``ToTensor()``,
``Normalize(mean, std)`` applied to a PIL image; ``scripts/tests/test_preprocess.py``
checks that equivalence when torchvision is installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

RESIZE_SIZE = 256
CROP_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
IMAGENETTE_ROOT = DATA_DIR / "imagenette" / "imagenette2-320"
SPLITS_DIR = DATA_DIR / "imagenette" / "splits"
CLASSES_JSON = Path(__file__).resolve().parent / "imagenette_classes.json"



def preprocess_description(mean: tuple[float, ...] = MEAN,
                           std: tuple[float, ...] = STD) -> dict:
    """The transform as a JSON-ready dict, for sidecars and model metadata."""
    return {
        "resize_shorter_side": RESIZE_SIZE,
        "interpolation": "bilinear (PIL)",
        "center_crop": CROP_SIZE,
        "scale": "1/255",
        "mean": list(mean),
        "std": list(std),
        "layout": "NCHW",
        "dtype": "float32",
    }


# With the ImageNet mean/std, used by most torchvision models.
PREPROCESS_DESCRIPTION = preprocess_description()


def _resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    # Same output size rule as torchvision.transforms.functional.resize with an int size.
    w, h = img.size
    short, long = (w, h) if w <= h else (h, w)
    if short == size:
        return img
    new_short, new_long = size, int(size * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return img.resize((new_w, new_h), Image.BILINEAR)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    # Same offset rounding as torchvision.transforms.functional.center_crop.
    w, h = img.size
    if w < size or h < size:
        raise ValueError(f"image {w}x{h} is smaller than the {size}x{size} crop")
    top = int(round((h - size) / 2.0))
    left = int(round((w - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def preprocess_image_unit(img: Image.Image) -> np.ndarray:
    """Steps 1-4 and 6: a [1, 3, 224, 224] float32 RGB array in [0, 1], not normalized.

    For models that normalize internally (e.g. Qualcomm AI Hub exports, which take a
    [0, 1] image and quantize it to uint8).
    """
    img = _center_crop(_resize_shorter_side(img.convert("RGB"), RESIZE_SIZE), CROP_SIZE)
    x = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    return np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)


def normalize(x_unit: np.ndarray, mean: tuple[float, ...] = MEAN,
              std: tuple[float, ...] = STD) -> np.ndarray:
    """Step 5: per-channel mean/std normalization of an [N, 3, H, W] array in [0, 1].

    The default is the ImageNet mean/std. Models trained with other constants
    (e.g. 0.5/0.5) must pass their own: ``common.model_registry`` holds them per model.
    """
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    return np.ascontiguousarray((x_unit - mean_arr) / std_arr, dtype=np.float32)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Return a normalized [1, 3, 224, 224] float32 array for a PIL image."""
    return normalize(preprocess_image_unit(img))


def load_image(path: str | Path, *, normalized: bool = True, mean: tuple[float, ...] = MEAN,
               std: tuple[float, ...] = STD) -> np.ndarray:
    """Load an image file and preprocess it to a [1, 3, 224, 224] float32 array.

    With ``normalized=False`` the array is in [0, 1] (no mean/std normalization);
    otherwise ``mean``/``std`` (default ImageNet) are applied.
    """
    with Image.open(path) as img:
        unit = preprocess_image_unit(img)
    return normalize(unit, mean, std) if normalized else unit


@dataclass(frozen=True)
class ImagenetteClass:
    wnid: str
    imagenet_index: int
    name: str


def load_imagenette_classes() -> list[ImagenetteClass]:
    """The 10 Imagenette classes, ordered by ImageNet-1k index."""
    data = json.loads(CLASSES_JSON.read_text(encoding="utf-8"))
    return [ImagenetteClass(**c) for c in data["classes"]]


def read_split(name: str, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Read a split file: one image path per line, relative to the Imagenette root."""
    path = splits_dir / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/get_data.py first")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wnid_of(relpath: str) -> str:
    """WordNet ID of an Imagenette image, taken from its class directory."""
    return Path(relpath).parent.name
