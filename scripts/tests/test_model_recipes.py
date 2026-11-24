"""Tests for the model manifest, its lineage and the recipe sidecars.

The manifest, lineage and recipe-schema tests use only the standard library and
jsonschema, and run in CI. The tests on real model files skip, with the reason
shown, when onnx or the model files are not available.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from common import model_recipe as mr
from common import model_registry as reg

jsonschema = pytest.importorskip("jsonschema")

SCRIPTS = Path(__file__).resolve().parents[1]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SHA = {k: k * 64 for k in "abcdef"}
TIMM = {"weights_source": "timm",
        "weights": "timm efficientnet_lite0.ra_in1k (timm/efficientnet_lite0.ra_in1k@" + "1" * 40 + ")",
        "weights_url": "https://huggingface.co/timm/efficientnet_lite0.ra_in1k/resolve/"
                       + "1" * 40 + "/model.safetensors",
        "weights_hf_repo": "timm/efficientnet_lite0.ra_in1k", "weights_hf_revision": "1" * 40,
        "weights_file": "model.safetensors", "timm_version": "1.0.31"}


def schema(path):
    return jsonschema.Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


# --- synthetic model files --------------------------------------------------------


def fp32_meta(model="mobilenetv2", batch=1, **extra):
    pre = {"center_crop": 224, "dtype": "float32", "interpolation": "bilinear (PIL)",
           "layout": "NCHW", "mean": MEAN, "resize_shorter_side": 256, "scale": "1/255",
           "std": STD}
    meta = {"model": model, "variant": "fp32", "precision": "fp32", "batch_size": str(batch),
            "weights_source": "torchvision",
            "weights": "torchvision MobileNet_V2_Weights.IMAGENET1K_V2",
            "weights_url": "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
            "weights_sha256": SHA["a"], "normalization": "external",
            "input_range": reg.INPUT_RANGE_EXTERNAL, "input_preprocessing": json.dumps(pre),
            "exporter": reg.EXPORTER, "opset": "17", "ir_version": "8", "onnx_version": "1.23.2",
            "torch_version": "2.13.0+cpu", "torchvision_version": "0.28.0+8fb8771"}
    meta.update(extra)
    return {reg.META_PREFIX + k: v for k, v in meta.items()}


def folded(meta, parent_sha, parent_file):
    m = dict(meta)
    pre = json.loads(m["edgebench.input_preprocessing"])
    pre.update(mean=None, std=None, normalization="folded into the first convolution")
    m.update({"edgebench.variant": "fp32_folded", "edgebench.normalization": "folded",
              "edgebench.input_preprocessing": json.dumps(pre),
              "edgebench.input_range": reg.INPUT_RANGE_FOLDED,
              "edgebench.folded_mean": json.dumps(MEAN), "edgebench.folded_std": json.dumps(STD),
              "edgebench.fold_padding": "mean_pad", "edgebench.fold_first_conv": "/conv/Conv",
              "edgebench.source_model": parent_file, "edgebench.source_sha256": parent_sha})
    return m


def int8(meta, parent_sha, parent_file, pruned=True):
    m = dict(meta)
    base = m["edgebench.variant"].replace("fp32", "int8")
    q = {"format": "qdq", "method": "static", "weight_type": "int8", "weight_symmetric": "true",
         "per_channel": "true", "activation_type": "uint8", "activation_symmetric": "false",
         "bias_type": "int32", "calibration_method": "MinMax",
         "calibration_dataset": "Imagenette calibration split", "calibration_manifest_sha256":
         SHA["f"], "calibration_count": "200", "calibration_batch": "1", "qnn_preprocess":
         "no changes", "tool": "onnxruntime.quantization 1.32.0", "pruned": str(pruned).lower(),
         "pruned_chains": "70" if pruned else "0", "pruned_nodes": "210" if pruned else "0"}
    m.update({f"edgebench.quant.{k}": v for k, v in q.items()})
    m.update({"edgebench.variant": base if pruned else base + "_unpruned",
              "edgebench.precision": "int8", "edgebench.onnxruntime_version": "1.32.0",
              "edgebench.source_model": parent_file, "edgebench.source_sha256": parent_sha})
    return m


def model_file(meta, sha, batch=1, name=None):
    model, variant = meta["edgebench.model"], meta["edgebench.variant"]
    return mr.ModelFile(name or reg.relative_path(model, variant, batch), sha, 1000, meta,
                        [mr.Tensor("input", "float32", [batch, 3, 224, 224])],
                        [mr.Tensor("logits", "float32", [batch, 1000])])


def family():
    """A complete lineage: FP32 -> FP32 folded -> INT8 folded (+ unpruned), FP32 -> INT8."""
    root = fp32_meta()
    root_file = reg.relative_path("mobilenetv2", "fp32", 1)
    fold = folded(root, SHA["a"], root_file)
    fold_file = reg.relative_path("mobilenetv2", "fp32_folded", 1)
    return [model_file(root, SHA["a"]), model_file(fold, SHA["b"]),
            model_file(int8(fold, SHA["b"], fold_file), SHA["c"]),
            model_file(int8(fold, SHA["b"], fold_file, pruned=False), SHA["d"]),
            model_file(int8(root, SHA["a"], root_file), SHA["e"])]


def entries(files):
    return [reg.manifest_entry(f.file, f.sha256, f.size_bytes, f.metadata) for f in files]


# --- recipe sidecar (CI) ----------------------------------------------------------


@pytest.mark.parametrize("index", range(5))
def test_recipes_of_every_variant_match_the_schema(index):
    recipe = mr.build_recipe(family()[index])
    assert mr.schema_problems(recipe) == []
    assert recipe["recipe_version"] == mr.RECIPE_VERSION
    assert (recipe["quantization"] is None) == (recipe["identity"]["precision"] == "fp32")
    assert (recipe["lineage"]["parent"] is None) == (recipe["identity"]["variant"] == "fp32")
    assert (recipe["preprocessing"]["fold"] is None) == \
        (recipe["identity"]["normalization"] == "external")
    assert recipe["preprocessing"]["mean"] == MEAN  # folded files keep the constants they fold


def test_timm_recipe_has_the_hugging_face_pin():
    recipe = mr.build_recipe(model_file(fp32_meta(model="efficientnet_lite0", **TIMM), SHA["a"]))
    assert mr.schema_problems(recipe) == []
    w = recipe["weights"]
    assert (w["library"], w["hf_revision"], w["file"]) == ("timm", "1" * 40, "model.safetensors")
    assert recipe["export"]["tool_versions"]["timm"] == "1.0.31"


def test_torchvision_recipe_has_no_hugging_face_fields():
    recipe = mr.build_recipe(family()[0])
    assert set(recipe["weights"]) == {"library", "id", "url", "sha256"}
    recipe["weights"]["hf_revision"] = "1" * 40
    assert mr.schema_problems(recipe)


REQUIRED_KEYS = sorted(set(family()[2].metadata) - {"edgebench.timm_version"})


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_a_missing_metadata_key_is_an_error_naming_it(key):
    mf = family()[2]
    meta = {k: v for k, v in mf.metadata.items() if k != key}
    with pytest.raises(reg.MetadataError, match=key.replace(".", r"\.")):
        mr.build_recipe(replace(mf, metadata=meta))


@pytest.mark.parametrize("key", ["weights_hf_repo", "weights_hf_revision", "weights_file",
                                 "timm_version"])
def test_timm_keys_are_required_for_timm_weights(key):
    meta = fp32_meta(model="efficientnet_lite0", **TIMM)
    del meta["edgebench." + key]
    with pytest.raises(reg.MetadataError, match=key):
        mr.build_recipe(model_file(meta, SHA["a"]))


@pytest.mark.parametrize("key,value", [("edgebench.quant.pruned", "yes"),
                                       ("edgebench.quant.calibration_count", "many"),
                                       ("edgebench.weights_source", "keras"),
                                       ("edgebench.input_preprocessing", "{")])
def test_unusable_metadata_values_are_errors(key, value):
    mf = family()[2]
    with pytest.raises(reg.MetadataError, match=key.replace(".", r"\.")):
        mr.build_recipe(replace(mf, metadata={**mf.metadata, key: value}))


def test_schema_rejects_nulls_and_inconsistent_documents():
    recipe = mr.build_recipe(family()[2])
    for path, value in ((("weights", "url"), None), (("quantization",), None),
                        (("lineage", "parent"), None), (("preprocessing", "fold"), None)):
        doc = json.loads(json.dumps(recipe))
        target = doc
        for p in path[:-1]:
            target = target[p]
        target[path[-1]] = value
        assert mr.schema_problems(doc), path
    unpruned = mr.build_recipe(family()[3])
    unpruned["quantization"]["pruned_chains"] = 5
    assert mr.schema_problems(unpruned)


def test_recipe_path_and_format():
    assert mr.recipe_path(Path("m/m_int8_b1.onnx")) == Path("m/m_int8_b1.recipe.json")
    text = mr.format_recipe(mr.build_recipe(family()[0]))
    assert text.endswith("}\n") and "\r" not in text


def test_consistency_problems_find_disagreements():
    files = family()
    mf, entry = files[2], entries(files)[2]
    recipe = mr.build_recipe(mf)
    assert mr.consistency_problems(recipe, entry, mf) == []
    stale = json.loads(json.dumps(recipe))
    stale["quantization"]["calibration"]["image_count"] = 100
    assert "quantization" in mr.consistency_problems(stale, entry, mf)[0]
    wrong = replace(entry, batch_size=4)
    assert "batch_size" in mr.consistency_problems(recipe, wrong, mf)[0]
    assert "not in the manifest" in mr.consistency_problems(recipe, None, mf)[0]


# --- manifest and lineage (CI) ----------------------------------------------------


def test_manifest_entries_come_from_metadata():
    e = entries(family())
    assert [(f.variant, f.precision, f.normalization, f.pruned) for f in e] == [
        ("fp32", "fp32", "external", None), ("fp32_folded", "fp32", "folded", None),
        ("int8_folded", "int8", "folded", True), ("int8_folded_unpruned", "int8", "folded", False),
        ("int8", "int8", "external", True)]
    assert [f.parent_sha256 for f in e] == [None, SHA["a"], SHA["b"], SHA["b"], SHA["a"]]


def test_manifest_round_trip_and_format_checks():
    files = entries(family())
    text = reg.format_manifest(list(reversed(files)), {"torch": "1", "onnx": "2"})
    parsed, tools = reg.parse_manifest(text)
    assert parsed == sorted(files, key=lambda f: f.file) and tools == {"onnx": "2", "torch": "1"}
    doc = json.loads(text)
    assert doc["manifest_version"] == reg.MANIFEST_VERSION
    assert list(schema(reg.MANIFEST_SCHEMA).iter_errors(doc)) == []
    with pytest.raises(ValueError, match="manifest_version"):
        reg.parse_manifest(text.replace(f'"{reg.MANIFEST_VERSION}"', '"0.9.0"', 1))
    del doc["files"][0]["parent_sha256"]
    with pytest.raises(ValueError, match="parent_sha256"):
        reg.parse_manifest(json.dumps(doc))
    doc = json.loads(text)
    doc["files"].reverse()
    with pytest.raises(ValueError, match="sorted"):
        reg.parse_manifest(json.dumps(doc))


def test_lineage_of_a_sound_family():
    files = entries(family())
    assert reg.lineage_problems(files) == []
    chain = reg.lineage(files, files[2])
    assert [f.variant for f in chain] == ["int8_folded", "fp32_folded", "fp32"]


@pytest.mark.parametrize("change,message", [
    (lambda f: [f[0], replace(f[1], parent_sha256=SHA["f"]), *f[2:]], "not in the manifest"),
    (lambda f: [replace(f[0], parent_sha256=SHA["b"]), *f[1:]], "has a parent"),
    (lambda f: [replace(f[0], variant="fp32_folded", parent_sha256=SHA["c"]), *f[1:]], "cycle"),
    (lambda f: [*f[:4], replace(f[4], parent_sha256=SHA["b"])], "expected fp32"),
    (lambda f: [f[0], replace(f[1], batch_size=4), *f[2:]], "another model or batch size"),
    (lambda f: [replace(f[0], normalization="folded"), *f[1:]], "external normalization"),
    (lambda f: [*f, replace(f[4], file="x/y.onnx")], "same SHA256"),
])
def test_lineage_problems_are_found(change, message):
    problems = reg.lineage_problems(change(entries(family())))
    assert any(message in p for p in problems), problems


def test_committed_manifest_matches_its_schema_and_has_sound_lineage():
    doc = json.loads(reg.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert list(schema(reg.MANIFEST_SCHEMA).iter_errors(doc)) == []
    files, _ = reg.read_manifest()
    assert reg.lineage_problems(files) == []
    depth = {"fp32": 1, "fp32_folded": 2, "int8": 2, "int8_unpruned": 2,
             "int8_folded": 3, "int8_folded_unpruned": 3}
    for f in files:
        chain = reg.lineage(files, f)
        assert len(chain) == depth[f.variant], f.file
        root = chain[-1]
        assert (root.variant, root.precision, root.normalization) == ("fp32", "fp32", "external")
        assert (root.model, root.batch_size) == (f.model, f.batch_size)
        assert f.file == reg.relative_path(f.model, f.variant, f.batch_size)
        assert f.precision == ("int8" if f.variant.startswith("int8") else "fp32")
        assert f.normalization == ("folded" if "folded" in f.variant else "external")
        assert f.pruned == (None if f.precision == "fp32" else not f.variant.endswith("unpruned"))


# --- real model files (need onnx and the files) -----------------------------------


@pytest.fixture(scope="module")
def on_disk():
    pytest.importorskip("onnx")
    files, _ = reg.read_manifest()
    present = [f for f in files if (reg.MODELS_DIR / f.file).is_file()]
    if not present:
        pytest.skip("no model files; run scripts/export_models.py, fold_models.py and "
                    "quantize_models.py")
    return present


def test_every_file_on_disk_agrees_with_its_sidecar_and_manifest_entry(on_disk):
    problems = []
    for entry in on_disk:
        path = reg.MODELS_DIR / entry.file
        sidecar = mr.recipe_path(path)
        if not sidecar.is_file():
            problems.append(f"{entry.file}: no sidecar")
            continue
        recipe = json.loads(sidecar.read_text(encoding="utf-8"))
        problems += mr.schema_problems(recipe)
        problems += mr.consistency_problems(recipe, entry, mr.read_model_file(path, entry.file))
    assert problems == []


def test_sidecars_regenerate_byte_identical(on_disk):
    for entry in on_disk[::7]:
        path = reg.MODELS_DIR / entry.file
        mf = mr.read_model_file(path, entry.file)
        assert mr.format_recipe(mr.build_recipe(mf)) == \
            mr.recipe_path(path).read_text(encoding="utf-8"), entry.file


def test_check_command_passes(on_disk):
    r = subprocess.run([sys.executable, str(SCRIPTS / "model_recipes.py"), "--check"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "lineage of" in r.stdout and "OK" in r.stdout


def test_model_info_prints_recipe_and_lineage(on_disk, tmp_path):
    folded_int8 = next(f for f in on_disk if f.variant == "int8_folded")
    r = subprocess.run([sys.executable, str(SCRIPTS / "model_info.py"),
                        str(reg.MODELS_DIR / folded_int8.file)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "(sidecar)" in r.stdout and "calibration" in r.stdout
    assert r.stdout.count("<- ") == 2  # INT8 folded <- FP32 folded <- FP32
    # Without a sidecar, the recipe comes from the file's metadata.
    copy = tmp_path / Path(folded_int8.file).name
    copy.write_bytes((reg.MODELS_DIR / folded_int8.file).read_bytes())
    r = subprocess.run([sys.executable, str(SCRIPTS / "model_info.py"), str(copy), "--json"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["identity"]["sha256"] == folded_int8.sha256
