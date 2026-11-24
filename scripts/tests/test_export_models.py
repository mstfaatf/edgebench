"""Tests for the FP32 export pipeline (scripts/export_models.py).

The registry and manifest tests use only the standard library and run in CI.
The rest skip, with the reason shown, when torch/onnx/onnxruntime, the pinned
weights, the exported files or the ImageNetV2 images are not available.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from common import model_registry as reg

SHA_RE = re.compile(r"[0-9a-f]{64}")


# --- registry and manifest (no third-party packages) ----------------------------------


def test_registry_entries_are_consistent():
    assert set(reg.MODELS) == {"mobilenetv2", "resnet18", "efficientnet_lite0"}
    for name, spec in reg.MODELS.items():
        assert spec.name == name and re.fullmatch(r"[a-z0-9_]+", name)
        assert SHA_RE.fullmatch(spec.weights_sha256)
        assert spec.weights_size_bytes > 0 and spec.input_size == 224
        assert (spec.probe_value is None) == (spec.probe_top1 is None)
        assert len(spec.mean) == 3 and len(spec.std) == 3 and all(s > 0 for s in spec.std)
        if spec.source == "torchvision":
            assert spec.weights_url.startswith("https://download.pytorch.org/models/")
            # torchvision names weight files <name>-<first 8 hex digits of the SHA256>.pth.
            assert Path(spec.weights_url).stem.endswith("-" + spec.weights_sha256[:8])
            assert not spec.hf_repo
        else:
            assert spec.source == "timm"
            assert re.fullmatch(r"[0-9a-f]{40}", spec.hf_revision)
            assert spec.weights_url == (f"https://huggingface.co/{spec.hf_repo}/resolve/"
                                        f"{spec.hf_revision}/{spec.weights_file}")
            assert spec.hf_repo == f"timm/{spec.constructor}.{spec.weights}"


def test_pinned_weights():
    assert reg.MODELS["mobilenetv2"].weights == "IMAGENET1K_V2"
    assert reg.MODELS["resnet18"].weights == "IMAGENET1K_V1"  # torchvision's only ResNet-18 weights
    lite = reg.MODELS["efficientnet_lite0"]
    assert (lite.source, lite.hf_repo) == ("timm", "timm/efficientnet_lite0.ra_in1k")
    assert lite.weights_id == ("timm efficientnet_lite0.ra_in1k (timm/efficientnet_lite0.ra_in1k"
                               "@29046ba6cc39eacfd5c7e6ffc1ae6188cfaa7609)")


def test_normalization_constants_per_model():
    imagenet = (reg.IMAGENET_MEAN, reg.IMAGENET_STD)
    for name in ("mobilenetv2", "resnet18", "efficientnet_lite0"):
        assert (reg.MODELS[name].mean, reg.MODELS[name].std) == imagenet


def test_file_naming():
    assert reg.file_name("mobilenetv2", "fp32", 1) == "mobilenetv2_fp32_b1.onnx"
    assert reg.relative_path("resnet18", "fp32", 8) == "resnet18/resnet18_fp32_b8.onnx"
    assert reg.expected_files(["resnet18"]) == [
        "resnet18/resnet18_fp32_b1.onnx", "resnet18/resnet18_fp32_b4.onnx",
        "resnet18/resnet18_fp32_b8.onnx"]


def test_manifest_round_trip():
    files = [reg.ManifestFile("b/x.onnx", "b" * 64, 2), reg.ManifestFile("a/x.onnx", "a" * 64, 1)]
    text = reg.format_manifest(files, {"torch": "1", "onnx": "2"})
    parsed, tools = reg.parse_manifest(text)
    assert [f.file for f in parsed] == ["a/x.onnx", "b/x.onnx"]
    assert tools == {"onnx": "2", "torch": "1"}


def test_manifest_rejects_unsorted_or_duplicate_files():
    good = reg.format_manifest([reg.ManifestFile("a.onnx", "a" * 64, 1)], {})
    with pytest.raises(ValueError):
        reg.parse_manifest(good.replace('"files": [', '"files": [{"file": "z.onnx", '
                                        '"sha256": "' + "c" * 64 + '", "size_bytes": 1}, '))


def test_committed_manifest_lists_every_fp32_export():
    text = reg.MANIFEST_PATH.read_text(encoding="utf-8")
    files, tools = reg.parse_manifest(text)
    assert text == reg.format_manifest(files, tools)  # canonical: LF, sorted, stable
    assert [f.file for f in files] == reg.expected_files()
    assert set(tools) == {"onnx", "timm", "torch", "torchvision"}
    for f in files:
        assert SHA_RE.fullmatch(f.sha256) and f.size_bytes > 0
    # A model's files differ only in the batch dimension, so they have the same size.
    for name in reg.MODELS:
        assert len({f.size_bytes for f in files if f.file.startswith(name + "/")}) == 1


# --- needs torch, onnx, onnxruntime -----------------------------------------------------


@pytest.fixture(scope="module")
def ox():
    for module in ("numpy", "torch", "torchvision", "timm", "onnx", "onnxruntime"):
        pytest.importorskip(module)
    from common import onnx_export

    return onnx_export


def test_registry_matches_torchvision_and_timm(ox):
    import timm
    import torchvision

    for spec in reg.MODELS.values():
        if spec.source == "torchvision":
            weights = getattr(getattr(torchvision.models, spec.weights_enum), spec.weights)
            assert weights.url == spec.weights_url
            assert callable(getattr(torchvision.models, spec.constructor))
            declared = weights.transforms()
            mean, std = declared.mean, declared.std
        else:
            cfg = timm.create_model(f"{spec.constructor}.{spec.weights}",
                                    pretrained=False).pretrained_cfg
            assert cfg["hf_hub_id"] == spec.hf_repo
            assert cfg["num_classes"] == reg.NUM_CLASSES
            assert cfg["input_size"] == (3, spec.input_size, spec.input_size)
            mean, std = cfg["mean"], cfg["std"]
        # Wrong constants only cost accuracy, silently: they must match the publisher's.
        assert tuple(round(float(v), 6) for v in mean) == spec.mean, spec.name
        assert tuple(round(float(v), 6) for v in std) == spec.std, spec.name


def test_normalization_mismatch_is_refused(ox):
    import dataclasses

    spec = dataclasses.replace(reg.MODELS["resnet18"], mean=(0.5, 0.5, 0.5))
    with pytest.raises(RuntimeError, match="mean/std"):
        ox._check_normalization(spec, reg.IMAGENET_MEAN, reg.IMAGENET_STD)


@pytest.fixture(scope="module")
def torch_models(ox):
    models = {}
    for name, spec in reg.MODELS.items():
        try:
            models[name] = ox.load_torch_model(spec, download=False)
        except FileNotFoundError as exc:
            pytest.skip(f"{exc}; run scripts/export_models.py")
    return models


@pytest.fixture(scope="module")
def exported(ox):
    paths = {rel: reg.MODELS_DIR / rel for rel in reg.expected_files()}
    missing = [rel for rel, p in paths.items() if not p.is_file()]
    if missing:
        pytest.skip(f"exported models missing ({missing[0]}, ...); run scripts/export_models.py")
    return paths


def test_export_is_deterministic(ox, torch_models):
    spec = reg.MODELS["mobilenetv2"]
    first = ox.export_fp32(spec, 1, torch_models["mobilenetv2"])
    assert ox.export_fp32(spec, 1, torch_models["mobilenetv2"]) == first
    files, tools = reg.read_manifest()
    if tools == ox.tool_versions():
        entry = next(f for f in files if f.file == reg.relative_path("mobilenetv2", "fp32", 1))
        assert ox.sha256_bytes(first) == entry.sha256


def test_exported_files_match_the_manifest(ox, exported):
    files, tools = reg.read_manifest()
    if tools != ox.tool_versions():
        pytest.skip(f"manifest written with {tools}, this environment has {ox.tool_versions()}")
    for f in files:
        assert ox.sha256_file(exported[f.file]) == f.sha256, f.file


@pytest.mark.parametrize("rel", reg.expected_files())
def test_exported_file_is_static_fp32_with_external_normalization(ox, exported, rel):
    import onnx

    data = exported[rel].read_bytes()
    spec = reg.MODELS[rel.split("/")[0]]
    batch = int(rel.rsplit("_b", 1)[1].removesuffix(".onnx"))
    info = ox.check_exported(data, spec, batch)  # checker, shape inference, shapes
    assert info["opset"] == reg.OPSET
    assert info["shapes"] == {"input": [batch, 3, 224, 224], "logits": [batch, 1000]}
    assert ox.static_shape_problems(onnx.load_from_string(data)) == []
    meta = info["metadata"]
    assert meta[reg.META_NORMALIZATION] == "external"
    assert meta["edgebench.batch_size"] == str(batch)
    assert meta["edgebench.weights_sha256"] == spec.weights_sha256
    assert meta["edgebench.exporter"] == reg.EXPORTER
    assert meta["edgebench.ir_version"] == str(info["ir_version"])
    preprocessing = json.loads(meta["edgebench.input_preprocessing"])
    assert (preprocessing["mean"], preprocessing["std"]) == (list(spec.mean), list(spec.std))
    assert ("edgebench.timm_version" in meta) == (spec.source == "timm")


def test_static_shape_problems_finds_symbolic_and_missing_dims(ox):
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
    problems = ox.static_shape_problems(helper.make_model(graph))
    assert len(problems) == 2 and "'x'" in problems[0] and "'y'" in problems[1]


# --- needs the ImageNetV2 images -------------------------------------------------------------


@pytest.fixture(scope="module")
def images(ox):
    """Check images in [0, 1]; each model normalizes them with its own mean/std."""
    loaded = ox.check_images()
    if loaded is None:
        pytest.skip("ImageNetV2 not downloaded; run scripts/get_data.py --dataset imagenetv2")
    return loaded[1]


def model_of(rel: str) -> reg.ModelSpec:
    return reg.MODELS[rel.split("/")[0]]


@pytest.fixture(scope="module")
def ort_logits(ox, exported, images):
    return {rel: ox.run_ort(ox.ort_session(path), ox.normalize_for(model_of(rel), images))
            for rel, path in exported.items()}


@pytest.mark.parametrize("rel", reg.expected_files())
def test_onnx_runtime_matches_pytorch(ox, torch_models, ort_logits, images, rel):
    spec = model_of(rel)
    reference = ox.run_torch(torch_models[spec.name], ox.normalize_for(spec, images))
    comparison = ox.compare_logits(ort_logits[rel], reference)
    assert comparison["max_abs_diff"] <= ox.LOGIT_ATOL, comparison
    assert comparison["top1_agreement"] == 1.0, comparison


@pytest.mark.parametrize("name", sorted(reg.MODELS))
@pytest.mark.parametrize("batch", [4, 8])
def test_batched_files_match_batch_1_row_by_row(ox, ort_logits, name, batch):
    batched = ort_logits[reg.relative_path(name, "fp32", batch)]
    single = ort_logits[reg.relative_path(name, "fp32", 1)]
    comparison = ox.compare_logits(batched, single)
    assert comparison["max_abs_diff"] <= ox.LOGIT_ATOL, comparison
    assert comparison["top1_agreement"] == 1.0, comparison


# Top-1 on ImageNetV2 eval-quick (1000 images, b8 files). A sanity check of
# the export, not a published number: a broken export or label mapping scores
# near 0.1%, and wrong normalization constants cost several points. Torchvision
# reports ImageNet-1k top-1 of 72.2% (MobileNetV2) and 69.8% (ResNet-18);
# ImageNetV2 matched-frequency is typically 10-13 points lower. EfficientNet-Lite0
# should land at or above MobileNetV2.
SANITY_TOP1 = {"mobilenetv2": (0.55, 0.65), "resnet18": (0.52, 0.62),
               "efficientnet_lite0": (0.587, 0.68)}


@pytest.fixture(scope="module")
def eval_quick(ox, images):
    import numpy as np

    from common.manifests import EVAL_QUICK_MANIFEST, IMAGENETV2_ROOT, read_manifest
    from common.preprocess import load_image

    entries = read_manifest(EVAL_QUICK_MANIFEST)
    x = np.concatenate([load_image(IMAGENETV2_ROOT / e.path, normalized=False) for e in entries])
    return x, np.array([e.label for e in entries])


@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_sanity_accuracy_on_eval_quick(ox, exported, eval_quick, name):
    import numpy as np

    x_unit, labels = eval_quick
    x = ox.normalize_for(reg.MODELS[name], x_unit)
    logits = ox.run_ort(ox.ort_session(exported[reg.relative_path(name, "fp32", 8)]), x)
    top1 = float(np.mean(logits.argmax(axis=1) == labels))
    top5 = float(np.mean([label in row for row, label in zip(np.argsort(-logits)[:, :5], labels)]))
    print(f"{name} eval-quick top-1 {top1:.3f}, top-5 {top5:.3f}")
    low, high = SANITY_TOP1[name]
    assert low <= top1 <= high, f"{name} eval-quick top-1 {top1:.3f} (top-5 {top5:.3f})"
