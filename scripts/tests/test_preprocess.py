"""Tests for scripts/common/preprocess.py.

Skipped when NumPy/Pillow (or, for the equivalence test, torchvision) are not
installed, so the suite still runs where only the validator's dependencies are.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from common import preprocess  # noqa: E402


def make_image(width: int, height: int, mode: str = "RGB", seed: int = 0):
    rng = np.random.default_rng(seed)
    channels = {"RGB": 3, "L": 1, "RGBA": 4}[mode]
    data = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    return Image.fromarray(data.squeeze(-1) if channels == 1 else data, mode)


@pytest.mark.parametrize("size", [(320, 240), (240, 320), (256, 256), (500, 333), (224, 300)])
def test_shape_and_dtype(size):
    x = preprocess.preprocess_image(make_image(*size))
    assert x.shape == (1, 3, 224, 224)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_non_rgb_images_are_converted(mode):
    assert preprocess.preprocess_image(make_image(300, 280, mode)).shape == (1, 3, 224, 224)


def test_too_small_image_is_rejected():
    # Resize makes the shorter side 256, so only a degenerate aspect ratio can
    # end up smaller than the crop; check the guard directly.
    with pytest.raises(ValueError):
        preprocess._center_crop(make_image(200, 300), 224)


@pytest.mark.parametrize("size,mode", [((320, 240), "RGB"), ((240, 427), "RGB"),
                                       ((256, 256), "RGB"), ((333, 500), "L")])
def test_matches_torchvision(size, mode):
    transforms = pytest.importorskip("torchvision.transforms")
    reference = transforms.Compose([
        transforms.Resize(preprocess.RESIZE_SIZE),
        transforms.CenterCrop(preprocess.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(preprocess.MEAN, preprocess.STD),
    ])
    img = make_image(*size, mode=mode, seed=1)
    expected = reference(img.convert("RGB")).numpy()[np.newaxis]
    np.testing.assert_allclose(preprocess.preprocess_image(img), expected, rtol=0, atol=1e-6)


def test_imagenette_classes():
    classes = preprocess.load_imagenette_classes()
    assert len(classes) == 10
    assert len({c.wnid for c in classes}) == 10
    assert len({c.imagenet_index for c in classes}) == 10
    assert all(0 <= c.imagenet_index < 1000 for c in classes)
    assert [c.imagenet_index for c in classes] == sorted(c.imagenet_index for c in classes)
    json.loads(preprocess.CLASSES_JSON.read_text(encoding="utf-8"))  # well-formed


def test_wnid_of():
    assert preprocess.wnid_of("val/n01440764/ILSVRC2012_val_00046252.JPEG") == "n01440764"
