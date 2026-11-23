"""Tests for the split manifests, the ImageNet-1k label mapping and the seeded sampling.

Everything except the last section runs on the committed files only (standard
library, any OS), so CI checks the manifests without the images. Tests that
need the downloaded datasets skip with a reason when ``data/`` is absent.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from common import manifests as mf
from common.manifests import ManifestEntry

SHA_A = "a" * 64
SHA_B = "b" * 64
IMAGENETTE_CLASSES_JSON = Path(mf.__file__).resolve().parent / "imagenette_classes.json"


def entry(path: str, label: int, sha: str = SHA_A) -> ManifestEntry:
    return ManifestEntry(path, label, sha)


def manifest_text(*rows: str) -> str:
    return "path\tlabel\tsha256\n" + "".join(r + "\n" for r in rows)


# --- labels from ImageNetV2 paths ------------------------------------------------


@pytest.mark.parametrize("path,label", [("0/x.jpeg", 0), ("2/x.jpeg", 2), ("10/x.jpeg", 10),
                                        ("100/x.jpeg", 100), ("999/x.jpeg", 999)])
def test_imagenetv2_label_is_the_class_directory_as_an_integer(path, label):
    assert mf.imagenetv2_label(path) == label


@pytest.mark.parametrize("path", ["1000/x.jpeg", "010/x.jpeg", "-1/x.jpeg", "n01440764/x.jpeg",
                                  "x.jpeg", "1/2/x.jpeg", "1\\x.jpeg", "/1/x.jpeg"])
def test_imagenetv2_label_rejects_other_paths(path):
    with pytest.raises(ValueError):
        mf.imagenetv2_label(path)


def test_class_10_sorts_after_class_2():
    entries = [entry(f"{c}/a.jpeg", c) for c in (10, 2, 100, 1, 0)]
    assert [e.label for e in mf.sort_entries(entries)] == [0, 1, 2, 10, 100]


def test_string_sorted_folders_would_give_wrong_labels():
    # What a generic image-folder loader does: enumerate the sorted directory names.
    names = sorted(str(i) for i in range(mf.NUM_CLASSES))
    string_sort_label = {name: i for i, name in enumerate(names)}
    assert string_sort_label["10"] == 2
    assert mf.imagenetv2_label("10/a.jpeg") == 10
    wrong = sum(string_sort_label[str(i)] != i for i in range(mf.NUM_CLASSES))
    assert wrong == 988  # only 12 directory names keep their position


# --- manifest format -----------------------------------------------------------


def test_manifest_round_trip():
    entries = [entry("10/b.jpeg", 10, SHA_B), entry("2/a.jpeg", 2)]
    text = mf.format_manifest(entries)
    assert text == manifest_text(f"2/a.jpeg\t2\t{SHA_A}", f"10/b.jpeg\t10\t{SHA_B}")
    assert mf.parse_manifest(text) == mf.sort_entries(entries)


def test_write_manifest_uses_lf(tmp_path):
    path = tmp_path / "m.tsv"
    mf.write_manifest(path, [entry("0/a.jpeg", 0)])
    assert b"\r" not in path.read_bytes()
    assert mf.read_manifest(path) == [entry("0/a.jpeg", 0)]


def test_string_sorted_manifest_is_rejected():
    text = manifest_text(f"10/a.jpeg\t10\t{SHA_A}", f"2/a.jpeg\t2\t{SHA_B}")
    with pytest.raises(ValueError, match="sorted"):
        mf.parse_manifest(text)


@pytest.mark.parametrize("text,match", [
    ("path,label,sha256\n", "header"),
    ("", "header"),
    (manifest_text("0/a.jpeg\t0"), "3 tab-separated"),
    (manifest_text(f"0/a.jpeg\t1000\t{SHA_A}"), "label"),
    (manifest_text(f"0/a.jpeg\t01\t{SHA_A}"), "label"),
    (manifest_text(f"0/a.jpeg\tx\t{SHA_A}"), "label"),
    (manifest_text("0/a.jpeg\t0\t" + "A" * 64), "sha256"),
    (manifest_text("0/a.jpeg\t0\tabc"), "sha256"),
    (manifest_text(f"0\\a.jpeg\t0\t{SHA_A}"), "path"),
    (manifest_text(f"../a.jpeg\t0\t{SHA_A}"), "path"),
    (manifest_text(f"0/a.jpeg\t0\t{SHA_A}", f"0/a.jpeg\t0\t{SHA_B}"), "duplicate"),
])
def test_malformed_manifests_are_rejected(text, match):
    with pytest.raises(ValueError, match=match):
        mf.parse_manifest(text)


def test_content_overlap_and_duplicates_compare_hashes_not_names():
    a = [entry("train/x.JPEG", 0, SHA_A), entry("train/y.JPEG", 0, SHA_B)]
    b = [entry("0/other-name.jpeg", 0, SHA_A), entry("0/train/y.JPEG", 0, "c" * 64)]
    assert mf.content_overlap(a, b) == [(a[0], b[0])]
    assert mf.duplicate_contents(a + b) == {SHA_A: [a[0], b[0]]}
    assert mf.content_overlap(a, [entry("train/x.JPEG", 0, "d" * 64)]) == []


# --- seeded stratified sampling --------------------------------------------------


def synthetic_entries(classes: int = 50, per_class: int = 10) -> list[ManifestEntry]:
    return [entry(f"{c}/{i:02d}.jpeg", c, f"{c * per_class + i:064x}")
            for c in range(classes) for i in range(per_class)]


def test_stratified_sample_takes_n_per_class():
    sample = mf.stratified_sample(synthetic_entries(), per_class=3, seed=1)
    assert Counter(e.label for e in sample) == {c: 3 for c in range(50)}
    assert sample == mf.sort_entries(sample)


def test_stratified_sample_is_deterministic_and_order_independent():
    entries = synthetic_entries()
    shuffled = list(entries)
    random.Random(0).shuffle(shuffled)
    first = mf.stratified_sample(entries, 1, mf.EVAL_QUICK_SEED)
    assert mf.stratified_sample(entries, 1, mf.EVAL_QUICK_SEED) == first
    assert mf.stratified_sample(shuffled, 1, mf.EVAL_QUICK_SEED) == first
    assert mf.stratified_sample(entries, 1, mf.EVAL_QUICK_SEED + 1) != first


def test_stratified_sample_known_values():
    # Pins the sampling algorithm (class order, candidate order, random.sample).
    sample = mf.stratified_sample(synthetic_entries(classes=3), 1, mf.EVAL_QUICK_SEED)
    rng = random.Random(mf.EVAL_QUICK_SEED)
    expected = [rng.sample([f"{c}/{i:02d}.jpeg" for i in range(10)], 1)[0] for c in range(3)]
    assert [e.path for e in sample] == expected


def test_stratified_sample_needs_enough_images():
    with pytest.raises(ValueError, match="class 0"):
        mf.stratified_sample(synthetic_entries(classes=1, per_class=2), 3, 0)


# --- ImageNet-1k class list ----------------------------------------------------------


def test_imagenet_classes_are_in_sorted_wnid_order():
    classes = mf.load_imagenet_classes()
    assert [c.index for c in classes] == list(range(mf.NUM_CLASSES))
    wnids = [c.wnid for c in classes]
    assert len(set(wnids)) == mf.NUM_CLASSES
    # ImageNet-1k class index i is the i-th WordNet ID in sorted order.
    assert wnids == sorted(wnids)
    assert all(c.name for c in classes)


def test_imagenet_classes_agree_with_the_imagenette_mapping():
    classes = mf.load_imagenet_classes()
    imagenette = json.loads(IMAGENETTE_CLASSES_JSON.read_text(encoding="utf-8"))["classes"]
    for c in imagenette:
        assert classes[c["imagenet_index"]].wnid == c["wnid"]
        assert classes[c["imagenet_index"]].name == c["name"]


@pytest.mark.parametrize("weights", ["MobileNet_V2_Weights.IMAGENET1K_V2",
                                     "ResNet18_Weights.IMAGENET1K_V1"])
def test_imagenet_classes_match_torchvision_weights(weights):
    models = pytest.importorskip("torchvision.models")
    enum_name, member = weights.split(".")
    categories = getattr(getattr(models, enum_name), member).meta["categories"]
    assert [c.name for c in mf.load_imagenet_classes()] == list(categories)


# --- committed manifests ---------------------------------------------------------------


@pytest.fixture(scope="module")
def eval_full():
    return mf.read_manifest(mf.EVAL_FULL_MANIFEST)


@pytest.fixture(scope="module")
def eval_quick():
    return mf.read_manifest(mf.EVAL_QUICK_MANIFEST)


@pytest.fixture(scope="module")
def calibration():
    return mf.read_manifest(mf.CALIBRATION_MANIFEST)


def test_eval_full_has_10_images_per_class(eval_full):
    assert len(eval_full) == 10_000
    assert Counter(e.label for e in eval_full) == {c: 10 for c in range(mf.NUM_CLASSES)}


def test_eval_full_labels_come_from_the_class_directories(eval_full):
    assert all(e.label == mf.imagenetv2_label(e.path) for e in eval_full)


def test_eval_full_has_no_duplicate_images(eval_full):
    assert mf.duplicate_contents(eval_full) == {}


def test_eval_quick_is_the_seeded_sample_of_eval_full(eval_full, eval_quick):
    assert len(eval_quick) == 1000
    assert Counter(e.label for e in eval_quick) == {c: 1 for c in range(mf.NUM_CLASSES)}
    assert set(eval_quick) <= set(eval_full)
    assert eval_quick == mf.stratified_sample(eval_full, mf.EVAL_QUICK_PER_CLASS,
                                              mf.EVAL_QUICK_SEED)


def test_calibration_manifest_is_20_imagenette_train_images_per_class(calibration):
    imagenette = json.loads(IMAGENETTE_CLASSES_JSON.read_text(encoding="utf-8"))["classes"]
    index_of = {c["wnid"]: c["imagenet_index"] for c in imagenette}
    assert len(calibration) == 200
    assert Counter(e.label for e in calibration) == {i: 20 for i in index_of.values()}
    for e in calibration:
        split, wnid, _ = e.path.split("/")
        assert split == "train"
        assert e.label == index_of[wnid]
    assert mf.duplicate_contents(calibration) == {}


def test_calibration_and_evaluation_images_do_not_overlap(calibration, eval_full):
    assert mf.content_overlap(calibration, eval_full) == []
    assert not {e.path for e in calibration} & {e.path for e in eval_full}


@pytest.mark.parametrize("path", [mf.EVAL_FULL_MANIFEST, mf.EVAL_QUICK_MANIFEST,
                                  mf.CALIBRATION_MANIFEST])
def test_committed_manifests_are_canonical(path):
    # Byte-for-byte what write_manifest produces (LF, sorted, no trailing spaces).
    assert path.read_bytes().decode("utf-8") == mf.format_manifest(mf.read_manifest(path))


# --- downloaded data (skipped without data/) -----------------------------------------------


@pytest.fixture(scope="module")
def imagenetv2_root():
    if not (mf.IMAGENETV2_ROOT / ".extracted").is_file():
        pytest.skip(f"ImageNetV2 not downloaded ({mf.IMAGENETV2_ROOT}); "
                    "run scripts/get_data.py --dataset imagenetv2")
    return mf.IMAGENETV2_ROOT


def test_downloaded_class_directories_are_0_to_999(imagenetv2_root):
    names = {p.name for p in imagenetv2_root.iterdir() if p.is_dir()}
    assert names == {str(i) for i in range(mf.NUM_CLASSES)}


def test_downloaded_files_match_the_manifest(imagenetv2_root, eval_full, eval_quick):
    on_disk = sorted(p.relative_to(imagenetv2_root).as_posix()
                     for p in imagenetv2_root.glob("*/*") if p.is_file())
    assert on_disk == sorted(e.path for e in eval_full)
    # Hashing all 10,000 images takes a while; get_data.py --verify does that.
    assert mf.verify_files(eval_quick[::10], imagenetv2_root) == []


def test_downloaded_calibration_images_match_the_manifest(calibration):
    root = mf.DATA_DIR / "imagenette" / "imagenette2-320"
    if not (root / ".extracted").is_file():
        pytest.skip(f"Imagenette not downloaded ({root}); run scripts/get_data.py")
    assert mf.verify_files(calibration, root) == []
