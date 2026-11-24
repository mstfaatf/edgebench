"""Tests for the INT8 QDQ models from scripts/quantize_models.py.

Skip, with the reason shown, when onnx/onnxruntime, the quantized files, the
ImageNetV2 images, or (for the NPU test) the QNN execution provider and an NPU
are not available.
"""

from __future__ import annotations

import json

import pytest

from common import model_registry as reg

INT8_FILES = reg.expected_files(variant="int8")
CALIBRATION_IMAGES = 200


@pytest.fixture(scope="module")
def oq():
    for module in ("numpy", "onnx", "onnxruntime"):
        pytest.importorskip(module)
    from common import onnx_quantize

    return onnx_quantize


@pytest.fixture(scope="module")
def ox(oq):
    from common import onnx_export

    return onnx_export


@pytest.fixture(scope="module")
def quantized(oq):
    files = [*INT8_FILES, *reg.expected_files(variant="int8_unpruned")]
    missing = [f for f in files if not (reg.MODELS_DIR / f).is_file()]
    if missing:
        pytest.skip(f"INT8 models missing ({missing[0]}, ...); run scripts/quantize_models.py")
    return {f: reg.MODELS_DIR / f for f in files}


def load(path):
    import onnx

    return onnx.load(str(path))


# --- quantization parameters --------------------------------------------------------


def make_qdq_model(scale: float):
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("QuantizeLinear", ["x", "s", "z"], ["q"]),
         helper.make_node("DequantizeLinear", ["q", "s", "z"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor("s", TensorProto.FLOAT, [], [scale]),
         helper.make_tensor("z", TensorProto.UINT8, [], [128])])
    return helper.make_model(graph)


def test_quant_param_differences_detects_a_changed_scale(oq):
    same = {1: make_qdq_model(0.1), 4: make_qdq_model(0.1), 8: make_qdq_model(0.1)}
    assert oq.quant_param_differences(same) == []
    assert set(oq.quant_params(same[1])) == {"s", "z"}
    changed = {1: make_qdq_model(0.1), 4: make_qdq_model(0.1000001)}
    assert oq.quant_param_differences(changed) == ["b4 vs b1: s differs"]


@pytest.mark.parametrize("name", sorted(reg.MODELS))
@pytest.mark.parametrize("variant", ["int8", "int8_unpruned"])
def test_all_batch_sizes_share_scales_and_zero_points(oq, quantized, name, variant):
    protos = {b: load(quantized[reg.relative_path(name, variant, b)]) for b in reg.BATCH_SIZES}
    assert oq.quant_params(protos[1])
    assert oq.quant_param_differences(protos) == []


# --- metadata ------------------------------------------------------------------------


@pytest.mark.parametrize("rel", [*INT8_FILES, *reg.expected_files(variant="int8_unpruned")])
def test_metadata_describes_the_int8_recipe(oq, quantized, rel):
    from common.manifests import CALIBRATION_MANIFEST, sha256_of

    meta = {p.key: p.value for p in load(quantized[rel]).metadata_props}
    name = rel.split("/")[0]
    batch = int(rel.rsplit("_b", 1)[1].removesuffix(".onnx"))
    pruned = "_unpruned_" not in rel
    source_rel = reg.relative_path(name, "fp32", batch)
    source_meta = {p.key: p.value for p in load(reg.MODELS_DIR / source_rel).metadata_props}

    assert meta["edgebench.variant"] == ("int8" if pruned else "int8_unpruned")
    assert meta["edgebench.precision"] == "int8"
    assert meta["edgebench.quant.pruned"] == ("true" if pruned else "false")
    assert meta["edgebench.source_model"] == source_rel
    manifest = {f.file: f for f in reg.read_manifest()[0]}
    assert meta["edgebench.source_sha256"] == manifest[source_rel].sha256
    expected_recipe = {"format": "qdq", "method": "static", "weight_type": "int8",
                       "weight_symmetric": "true", "per_channel": "true",
                       "activation_type": "uint8", "activation_symmetric": "false",
                       "calibration_method": "MinMax",
                       "calibration_count": str(CALIBRATION_IMAGES)}
    assert {k: meta["edgebench.quant." + k] for k in expected_recipe} == expected_recipe
    assert meta["edgebench.quant.calibration_manifest_sha256"] == sha256_of(CALIBRATION_MANIFEST)
    # Everything that describes the model and its input carries over unchanged.
    for key in oq.CARRIED_KEYS:
        assert meta.get("edgebench." + key) == source_meta.get("edgebench." + key), key
    assert meta["edgebench.normalization"] == "external"
    assert not any(k.startswith("onnx.") for k in meta)  # nothing left from the quantizer


def test_mobilenetv2_matches_the_feasibility_int8_graph(oq, quantized):
    reference = reg.MODELS_DIR / "mobilenetv2" / "mobilenetv2_int8qdq_pc_s8w_b1.onnx"
    if not reference.is_file():
        pytest.skip(f"{reference} not present (scripts/quantize_qnn.py)")
    new = load(quantized["mobilenetv2/mobilenetv2_int8_b1.onnx"])
    assert oq.graph_without_metadata(new) == oq.graph_without_metadata(load(reference))


def test_quantized_files_match_the_manifest(ox, quantized):
    files, tools = reg.read_manifest()
    if tools != ox.tool_versions():
        pytest.skip(f"manifest written with {tools}, this environment has {ox.tool_versions()}")
    manifest = {f.file: f.sha256 for f in files}
    for rel, path in quantized.items():
        assert ox.sha256_file(path) == manifest[rel], rel


def test_quantization_is_deterministic(oq, ox, quantized):
    if not oq.calibration_images_present():
        pytest.skip("Imagenette not downloaded; run scripts/get_data.py")
    files, tools = reg.read_manifest()
    if tools != ox.tool_versions():
        pytest.skip(f"manifest written with {tools}, this environment has {ox.tool_versions()}")
    manifest = {f.file: f.sha256 for f in files}
    result = oq.quantize(reg.MODELS["mobilenetv2"], [1])[1]
    assert ox.sha256_bytes(result["pruned"]) == manifest["mobilenetv2/mobilenetv2_int8_b1.onnx"]
    assert ox.sha256_bytes(result["unpruned"]) == \
        manifest["mobilenetv2/mobilenetv2_int8_unpruned_b1.onnx"]


# --- outputs on real images -----------------------------------------------------------


@pytest.fixture(scope="module")
def images(ox):
    loaded = ox.check_images()
    if loaded is None:
        pytest.skip("ImageNetV2 not downloaded; run scripts/get_data.py --dataset imagenetv2")
    return loaded[1]


def logits(ox, rel, x_unit):
    spec = reg.MODELS[rel.split("/")[0]]
    return ox.run_ort(ox.ort_session(reg.MODELS_DIR / rel), ox.normalize_for(spec, x_unit))


@pytest.mark.parametrize("rel", INT8_FILES)
def test_pruned_and_unpruned_give_bit_identical_logits(ox, quantized, images, rel):
    import numpy as np

    unpruned = rel.replace("_int8_b", "_int8_unpruned_b")
    assert np.array_equal(logits(ox, rel, images), logits(ox, unpruned, images))


@pytest.mark.parametrize("name", sorted(reg.MODELS))
@pytest.mark.parametrize("batch", [4, 8])
def test_int8_batched_files_match_batch_1(ox, quantized, images, name, batch):
    comparison = ox.compare_logits(logits(ox, reg.relative_path(name, "int8", batch), images),
                                   logits(ox, reg.relative_path(name, "int8", 1), images))
    assert comparison["max_abs_diff"] <= ox.LOGIT_ATOL, comparison
    assert comparison["top1_agreement"] == 1.0, comparison


# INT8 vs FP32 on ImageNetV2 eval-quick (1000 images, b8 files): a sanity check of
# the quantization, not the published evaluation. Static INT8 typically costs a few
# points of top-1 and changes the top-1 class on 5-15% of images.
MAX_TOP1_DROP = 0.04
MIN_AGREEMENT = 0.85


@pytest.fixture(scope="module")
def eval_quick(ox, images):
    import numpy as np

    from common.manifests import EVAL_QUICK_MANIFEST, IMAGENETV2_ROOT, read_manifest
    from common.preprocess import load_image

    entries = read_manifest(EVAL_QUICK_MANIFEST)
    x = np.concatenate([load_image(IMAGENETV2_ROOT / e.path, normalized=False) for e in entries])
    return x, np.array([e.label for e in entries])


@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_int8_vs_fp32_sanity_accuracy_on_eval_quick(ox, quantized, eval_quick, name):
    import numpy as np

    x, labels = eval_quick
    fp32 = logits(ox, reg.relative_path(name, "fp32", 8), x).argmax(axis=1)
    int8 = logits(ox, reg.relative_path(name, "int8", 8), x).argmax(axis=1)
    top1_fp32, top1_int8 = float(np.mean(fp32 == labels)), float(np.mean(int8 == labels))
    agreement = float(np.mean(fp32 == int8))
    print(f"{name} eval-quick top-1 FP32 {top1_fp32:.3f}, INT8 {top1_int8:.3f}, "
          f"top-1 agreement {agreement:.3f}")
    assert top1_int8 >= top1_fp32 - MAX_TOP1_DROP
    assert agreement >= MIN_AGREEMENT


# --- NPU (QNN HTP), strict mode ------------------------------------------------------------

NPU_IMAGES = 16


@pytest.fixture(scope="module")
def npu_available(oq):
    pytest.importorskip("onnxruntime_qnn")
    from common.ort_qnn import NoNpuDeviceError, list_devices, register_qnn, select_qnn_npu

    register_qnn()
    try:
        select_qnn_npu(list_devices())
    except NoNpuDeviceError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_int8_runs_on_the_npu_in_strict_mode(oq, ox, quantized, images, npu_available, name):
    """Session creation fails if any node would fall back to the CPU; no timing here."""
    import numpy as np

    rel = reg.relative_path(name, "int8", 1)
    session = oq.npu_session(quantized[rel])
    x = ox.normalize_for(reg.MODELS[name], images[:NPU_IMAGES])
    npu = ox.run_ort(session, x).argmax(axis=1)
    cpu = ox.run_ort(ox.ort_session(quantized[rel]), x).argmax(axis=1)
    agreement = float(np.mean(npu == cpu))
    print(f"{name} INT8 b1 NPU (strict) vs CPU top-1 agreement {agreement:.3f} "
          f"on {NPU_IMAGES} images")
    assert agreement >= 0.75


def test_sidecars_record_the_checks(quantized):
    for rel in INT8_FILES:
        sidecar = json.loads((reg.MODELS_DIR / rel).with_suffix(".json").read_text(encoding="utf-8"))
        assert sidecar["variant"] == "int8"
        assert sidecar["unpruned_model"]["file"] == rel.replace("_int8_b", "_int8_unpruned_b")
        assert sidecar["checks"].get("pruned_vs_unpruned_bit_identical", True) is True
