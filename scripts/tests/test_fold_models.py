"""Tests for normalization folding (common/onnx_fold.py, scripts/fold_models.py).

The folding math is checked on tiny hand-built convolutions first; the
exported-model tests skip, with the reason shown, when onnx/onnxruntime, the
model files, the ImageNetV2 images or (for the NPU test) the QNN execution
provider are not available.
"""

from __future__ import annotations

import json

import pytest

from common import model_registry as reg

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from common import onnx_fold as of  # noqa: E402

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FOLDED_VARIANTS = ("fp32_folded", "int8_folded", "int8_folded_unpruned")


# --- the math, on tiny convolutions ---------------------------------------------------


def conv_model(*, pads=(1, 1, 1, 1), strides=(2, 2), kernel=3, bias=True, hw=(9, 8),
               group=1, seed=0, extra_consumer=False):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(4, 3 // group, kernel, kernel)).astype(np.float32)
    inits = [numpy_helper.from_array(w, "w")]
    inputs = ["x", "w"]
    if bias:
        inits.append(numpy_helper.from_array(rng.normal(size=4).astype(np.float32), "b"))
        inputs.append("b")
    nodes = [helper.make_node("Conv", inputs, ["c"], pads=list(pads), strides=list(strides),
                              group=group, kernel_shape=[kernel, kernel], name="first"),
             helper.make_node("Relu", ["c"], ["y"])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)]
    if extra_consumer:
        nodes.append(helper.make_node("Identity", ["x"], ["x2"]))
        outputs.append(helper.make_tensor_value_info("x2", TensorProto.FLOAT, None))
    graph = helper.make_graph(nodes, "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                                             [2, 3, *hw])], outputs, inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def run(model, x):
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {"x": x})[0]


def raw_and_normalized(hw=(9, 8), seed=1):
    raw = np.random.default_rng(seed).random((2, 3, *hw), dtype=np.float32)
    m = np.asarray(MEAN, np.float32).reshape(1, 3, 1, 1)
    s = np.asarray(STD, np.float32).reshape(1, 3, 1, 1)
    return raw, (raw - m) / s


@pytest.mark.parametrize("pads,strides,kernel,bias", [
    ((1, 1, 1, 1), (2, 2), 3, True),
    ((3, 3, 3, 3), (2, 2), 7, True),
    ((1, 1, 1, 1), (1, 1), 3, False),
    ((0, 1, 2, 1), (1, 2), 3, True),  # asymmetric padding
])
def test_exact_fold_matches_external_normalization(pads, strides, kernel, bias):
    hw = (13, 11)
    model = conv_model(pads=pads, strides=strides, kernel=kernel, bias=bias, hw=hw)
    raw, normalized = raw_and_normalized(hw)
    expected = run(model, normalized)
    exact, info = of.fold(model, MEAN, STD)
    naive, _ = of.fold(model, MEAN, STD, border_correction=False)
    assert info["border_correction"] and info["padded"]
    exact_err = np.abs(run(exact, raw) - expected).max()
    naive_err = np.abs(run(naive, raw) - expected).max()
    assert exact_err < 1e-4  # float32 rounding of up to 147-term sums
    assert naive_err > 1e-2  # the naive fold is wrong next to the border


def test_naive_fold_is_only_wrong_at_the_border():
    model = conv_model(pads=(1, 1, 1, 1), strides=(1, 1), hw=(10, 10))
    del model.graph.node[1]  # look at the Conv output itself, before Relu
    model.graph.output[0].name = "c"
    raw, normalized = raw_and_normalized((10, 10))
    diff = np.abs(run(of.fold(model, MEAN, STD, border_correction=False)[0], raw)
                  - run(model, normalized))
    interior = diff[:, :, 1:-1, 1:-1]
    assert interior.max() < 1e-5
    assert diff.max() > 1e-2


def test_unpadded_conv_needs_no_border_correction():
    model = conv_model(pads=(0, 0, 0, 0), strides=(1, 1), hw=(9, 9))
    raw, normalized = raw_and_normalized((9, 9))
    folded, info = of.fold(model, MEAN, STD)
    assert info == {"first_conv": "first", "padded": False, "border_correction": False}
    assert [n.op_type for n in folded.graph.node] == ["Conv", "Relu"]
    assert np.abs(run(folded, raw) - run(model, normalized)).max() < 1e-5


def test_border_correction_map_is_zero_in_the_interior():
    model = conv_model(pads=(1, 1, 1, 1), strides=(1, 1), hw=(10, 10))
    folded, info = of.fold(model, MEAN, STD)
    inits = {i.name: numpy_helper.to_array(i) for i in folded.graph.initializer}
    correction = inits["first_border_correction"]
    assert correction.shape == (1, 4, 10, 10) == tuple(info["map_shape"])
    assert np.all(correction[:, :, 1:-1, 1:-1] == 0)
    assert np.all(np.abs(correction[:, :, 0, :]) > 0)
    assert [n.op_type for n in folded.graph.node] == ["Conv", "Add", "Relu"]


def test_reference_conv2d_matches_onnx_runtime():
    model = conv_model(pads=(2, 1, 0, 1), strides=(2, 1), bias=False, hw=(9, 8))
    x = np.random.default_rng(3).random((2, 3, 9, 8), dtype=np.float32)
    w = numpy_helper.to_array(model.graph.initializer[0])
    del model.graph.node[1]
    model.graph.output[0].name = "c"
    expected = run(model, x)[0]
    np.testing.assert_allclose(of.conv2d(x[0], w, (2, 1), (2, 1, 0, 1), (1, 1)), expected,
                               rtol=0, atol=1e-5)


@pytest.mark.parametrize("kwargs,match", [
    ({"extra_consumer": True}, "consumers"),
    ({"group": 3}, "grouped"),
])
def test_unsupported_patterns_are_refused(kwargs, match):
    with pytest.raises(of.FoldError, match=match):
        of.fold(conv_model(**kwargs), MEAN, STD)


def test_input_feeding_something_else_is_refused():
    model = conv_model()
    model.graph.node[0].op_type = "ConvTranspose"
    with pytest.raises(of.FoldError, match="not Conv"):
        of.fold(model, MEAN, STD)


def test_folded_metadata():
    source = {"edgebench.variant": "fp32", "edgebench.normalization": "external",
              "edgebench.input_preprocessing": json.dumps({"mean": list(MEAN), "std": list(STD),
                                                           "center_crop": 224})}
    meta = of.folded_metadata(source, MEAN, STD, "fp32_folded",
                              {"first_conv": "c", "border_correction": True}, "m/m_fp32_b1.onnx",
                              "a" * 64)
    assert meta["edgebench.normalization"] == "folded"
    assert meta["edgebench.variant"] == "fp32_folded"
    assert meta["edgebench.input_range"] == "[0, 1]"
    assert json.loads(meta["edgebench.folded_mean"]) == list(MEAN)
    assert json.loads(meta["edgebench.folded_std"]) == list(STD)
    assert meta["edgebench.fold_border_correction"] == "true"
    pre = json.loads(meta["edgebench.input_preprocessing"])
    assert pre["mean"] is None and pre["std"] is None and pre["center_crop"] == 224
    assert meta["edgebench.source_sha256"] == "a" * 64


# --- exported models ---------------------------------------------------------------


@pytest.fixture(scope="module")
def ox():
    from common import onnx_export

    return onnx_export


@pytest.fixture(scope="module")
def oq():
    from common import onnx_quantize

    return onnx_quantize


@pytest.fixture(scope="module")
def folded_files():
    files = {v: reg.expected_files(variant=v) for v in FOLDED_VARIANTS}
    missing = [f for fs in files.values() for f in fs if not (reg.MODELS_DIR / f).is_file()]
    if missing:
        pytest.skip(f"folded models missing ({missing[0]}, ...); run scripts/fold_models.py and "
                    "scripts/quantize_models.py --normalization folded")
    return files


def load(rel):
    return onnx.load(str(reg.MODELS_DIR / rel))


def meta_of(rel):
    return {p.key: p.value for p in load(rel).metadata_props}


@pytest.mark.parametrize("variant", FOLDED_VARIANTS)
@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_folded_metadata_in_files(folded_files, variant, name):
    manifest = {f.file: f.sha256 for f in reg.read_manifest()[0]}
    spec = reg.MODELS[name]
    for batch in reg.BATCH_SIZES:
        meta = meta_of(reg.relative_path(name, variant, batch))
        assert meta["edgebench.variant"] == variant
        assert meta["edgebench.normalization"] == "folded"
        assert meta["edgebench.input_range"] == "[0, 1]"
        assert json.loads(meta["edgebench.folded_mean"]) == list(spec.mean)
        assert json.loads(meta["edgebench.folded_std"]) == list(spec.std)
        assert meta["edgebench.fold_border_correction"] == "true"  # all three first convs pad
        source = reg.relative_path(name, "fp32" if variant == "fp32_folded" else "fp32_folded",
                                   batch)
        assert meta["edgebench.source_model"] == source
        assert meta["edgebench.source_sha256"] == manifest[source]


def test_folded_files_match_the_manifest(ox, folded_files):
    files, tools = reg.read_manifest()
    if tools != ox.tool_versions():
        pytest.skip(f"manifest written with {tools}, this environment has {ox.tool_versions()}")
    manifest = {f.file: f.sha256 for f in files}
    for rel in (f for fs in folded_files.values() for f in fs):
        assert ox.sha256_file(reg.MODELS_DIR / rel) == manifest[rel], rel


@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_folding_is_deterministic(ox, folded_files, name):
    data = (reg.MODELS_DIR / reg.relative_path(name, "fp32", 1)).read_bytes()
    spec = reg.MODELS[name]
    args = (spec.mean, spec.std, "fp32_folded", reg.relative_path(name, "fp32", 1),
            ox.sha256_bytes(data))
    first, _ = of.fold_file_bytes(data, *args)
    assert of.fold_file_bytes(data, *args)[0] == first
    assert first == (reg.MODELS_DIR / reg.relative_path(name, "fp32_folded", 1)).read_bytes()


@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_first_conv_of_every_model_is_foldable_and_padded(folded_files, name):
    conv = of.find_first_conv(load(reg.relative_path(name, "fp32", 1)))
    assert any(conv.pads)  # so every model needs the border correction
    folded = load(reg.relative_path(name, "fp32_folded", 1))
    adds = [n for n in folded.graph.node if n.name.endswith("_border_correction/Add")]
    assert len(adds) == 1


@pytest.fixture(scope="module")
def images(ox):
    loaded = ox.check_images()
    if loaded is None:
        pytest.skip("ImageNetV2 not downloaded; run scripts/get_data.py --dataset imagenetv2")
    return loaded[1]  # in [0, 1]


def logits(ox, rel, x):
    return ox.run_ort(ox.ort_session(reg.MODELS_DIR / rel), x)


@pytest.mark.parametrize("batch", reg.BATCH_SIZES)
@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_exact_fold_matches_external_on_real_images(ox, folded_files, images, name, batch):
    spec = reg.MODELS[name]
    external = logits(ox, reg.relative_path(name, "fp32", batch), ox.normalize_for(spec, images))
    folded = logits(ox, reg.relative_path(name, "fp32_folded", batch), images)
    comparison = ox.compare_logits(folded, external)
    print(f"{name} b{batch} exact fold vs external: {comparison}")
    assert comparison["max_abs_diff"] <= ox.LOGIT_ATOL, comparison
    assert comparison["top1_agreement"] == 1.0


@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_naive_fold_is_measurably_wrong_on_real_images(ox, folded_files, images, name):
    """Comparison only: the naive fold (no border correction) is never written to disk."""
    spec = reg.MODELS[name]
    naive, _ = of.fold(load(reg.relative_path(name, "fp32", 1)), spec.mean, spec.std,
                       border_correction=False)
    session = ort.InferenceSession(naive.SerializeToString(), providers=["CPUExecutionProvider"])
    external = logits(ox, reg.relative_path(name, "fp32", 1), ox.normalize_for(spec, images))
    comparison = ox.compare_logits(ox.run_ort(session, images), external)
    print(f"{name} naive fold vs external: {comparison}")
    assert comparison["max_abs_diff"] > 100 * ox.LOGIT_ATOL


@pytest.mark.parametrize("variant", ["fp32_folded", "int8_folded"])
@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_folded_batches_match_batch_1(ox, folded_files, images, variant, name):
    b1 = logits(ox, reg.relative_path(name, variant, 1), images)
    for batch in (4, 8):
        comparison = ox.compare_logits(logits(ox, reg.relative_path(name, variant, batch), images),
                                       b1)
        assert comparison["max_abs_diff"] <= ox.LOGIT_ATOL, (batch, comparison)
        assert comparison["top1_agreement"] == 1.0


@pytest.mark.parametrize("variant", ["int8_folded", "int8_folded_unpruned"])
@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_folded_int8_batch_sizes_share_scales(oq, folded_files, variant, name):
    protos = {b: load(reg.relative_path(name, variant, b)) for b in reg.BATCH_SIZES}
    assert oq.quant_param_differences(protos) == []


@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_folded_int8_pruned_matches_unpruned(ox, folded_files, images, name):
    assert np.array_equal(logits(ox, reg.relative_path(name, "int8_folded", 1), images),
                          logits(ox, reg.relative_path(name, "int8_folded_unpruned", 1), images))


@pytest.fixture(scope="module")
def eval_quick(images):
    from common.manifests import EVAL_QUICK_MANIFEST, IMAGENETV2_ROOT, read_manifest
    from common.preprocess import load_image

    entries = read_manifest(EVAL_QUICK_MANIFEST)
    x = np.concatenate([load_image(IMAGENETV2_ROOT / e.path, normalized=False) for e in entries])
    return x, np.array([e.label for e in entries])


@pytest.mark.parametrize("name", sorted(reg.MODELS))
def test_folded_int8_sanity_accuracy_on_eval_quick(ox, folded_files, eval_quick, name):
    """Not the published evaluation: only that the folded INT8 model is sane."""
    x, labels = eval_quick
    spec = reg.MODELS[name]
    int8_ext = logits(ox, reg.relative_path(name, "int8", 8), ox.normalize_for(spec, x)).argmax(1)
    int8_fold = logits(ox, reg.relative_path(name, "int8_folded", 8), x).argmax(1)
    fp32_fold = logits(ox, reg.relative_path(name, "fp32_folded", 8), x).argmax(1)
    top1_ext, top1_fold = float(np.mean(int8_ext == labels)), float(np.mean(int8_fold == labels))
    print(f"{name} eval-quick top-1 INT8 external {top1_ext:.3f}, INT8 folded {top1_fold:.3f}, "
          f"FP32 folded {float(np.mean(fp32_fold == labels)):.3f}; INT8 folded vs INT8 external "
          f"agreement {float(np.mean(int8_fold == int8_ext)):.3f}, vs FP32 folded "
          f"{float(np.mean(int8_fold == fp32_fold)):.3f}")
    assert top1_fold > 0.4  # far above chance (0.1%): the model works


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
def test_folded_int8_runs_on_the_npu_in_strict_mode(ox, oq, folded_files, images, npu_available,
                                                    name):
    rel = reg.relative_path(name, "int8_folded", 1)
    x = images[:16]
    npu = ox.run_ort(oq.npu_session(reg.MODELS_DIR / rel), x).argmax(1)
    cpu = logits(ox, rel, x).argmax(1)
    nodes_folded = len(load(rel).graph.node)
    nodes_external = len(load(reg.relative_path(name, "int8", 1)).graph.node)
    agreement = float(np.mean(npu == cpu))
    print(f"{name} INT8 folded b1 on NPU (strict): NPU vs CPU top-1 {agreement:.3f} on 16 "
          f"images; nodes {nodes_folded} (external INT8 {nodes_external})")
    assert agreement >= 0.75
