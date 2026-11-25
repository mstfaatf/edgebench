"""The imported autoencoder's pipeline: source bundle, exact graph clean-up, model files.

Source-bundle checks use only the standard library and run in CI. Clean-up tests
need onnx and NumPy (and onnxruntime for output comparisons); tests on the
model files need them on disk (``scripts/prepare_autoencoder.py``), the NAB data
(``scripts/get_data.py --dataset nab``) and, for the NPU, onnxruntime-qnn and
the HTP device. They skip visibly otherwise.
"""

from __future__ import annotations

import json

import pytest

from common import autoencoder_pipeline as ap
from common import model_registry as reg
from common import timeseries as ts

SPEC = reg.IMPORTED_MODELS["conv1d_autoencoder"]
SOURCE = reg.REPO_ROOT / SPEC.source_dir


# --- the committed source (CI) ------------------------------------------------------


def test_source_bundle_matches_the_registry_pins():
    assert ap.source_problems(SPEC) == []


def test_training_record_describes_the_committed_files():
    record = ap.training_record(SPEC)
    assert record["model"] == SPEC.name and record["design"] == "conv1d"
    assert record["edgebench_commit"] == SPEC.code_commit
    assert record["data"]["split_manifest_sha256"] == ts.sha256_file(ts.SPLIT_MANIFEST)
    assert record["data"]["normalization"]["mean"] == \
        ts.read_split_manifest()["normalization"]["mean"]
    assert {f["file"]: f["sha256"] for f in record["export"]["files"].values()} == \
        {f"{SPEC.name}_fp32_b{b}.onnx": sha for b, sha in SPEC.source_sha256}


def test_a_changed_source_file_is_reported(tmp_path, monkeypatch):
    import shutil
    from dataclasses import replace

    copy = tmp_path / "src"
    shutil.copytree(SOURCE, copy)
    (copy / "reference_keras.npy").write_bytes(b"changed")
    spec = replace(SPEC, source_dir=copy.relative_to(tmp_path).as_posix())
    monkeypatch.setattr(ap, "REPO_ROOT", tmp_path)
    problems = ap.source_problems(spec)
    assert any("reference_keras.npy: SHA256 differs from SHA256SUMS" in p for p in problems)


def test_the_source_is_committed_byte_exact():
    attributes = (reg.REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "models/sources/** binary" in attributes  # no line-ending conversion


# --- clean-up (onnx, NumPy) ---------------------------------------------------------


@pytest.fixture(scope="module")
def onnx_mod():
    pytest.importorskip("numpy")
    return pytest.importorskip("onnx")


def source_proto(onnx_mod, batch=1):
    return onnx_mod.load_from_string(ap.source_bytes(SPEC, batch))


@pytest.mark.parametrize("batch", reg.BATCH_SIZES)
def test_cleanup_folds_biases_and_removes_layer_wrapping(onnx_mod, batch):
    from common.onnx_cleanup import clean_conv1d_graph

    cleaned, report = clean_conv1d_graph(source_proto(onnx_mod, batch))
    assert (report.nodes_before, report.nodes_after) == (26, 11)
    assert report.ops_after == {"Conv": 3, "ConvTranspose": 2, "Relu": 4, "Reshape": 2}
    assert (report.bias_adds_folded, report.squeeze_unsqueeze_pairs_removed,
            report.reshape_chains_collapsed) == (5, 4, 2)
    ops = [n.op_type for n in cleaned.graph.node]
    assert ops == ["Reshape", "Conv", "Relu", "Conv", "Relu", "ConvTranspose", "Relu",
                   "ConvTranspose", "Relu", "Conv", "Reshape"]
    assert all(len(n.input) == 3 for n in cleaned.graph.node if n.op_type.startswith("Conv"))
    # Fixed names: tf2onnx numbers its helpers per batch size, the calibration cache needs names.
    assert cleaned.graph.node[0].output[0] == "input_nchw"


def test_cleanup_gives_the_same_tensor_names_for_every_batch(onnx_mod):
    from common.onnx_cleanup import clean_conv1d_graph

    names = []
    for batch in reg.BATCH_SIZES:
        cleaned, _ = clean_conv1d_graph(source_proto(onnx_mod, batch))
        names.append({o for n in cleaned.graph.node for o in n.output}
                     | {i.name for i in cleaned.graph.initializer})
    assert names[0] == names[1] == names[2]


def mutations(onnx_mod):
    """(description, change) pairs; each change breaks the expected pattern."""
    from onnx import helper, numpy_helper

    def node(proto, op, k=0):
        return [n for n in proto.graph.node if n.op_type == op][k]

    def init(proto, name):
        return next(i for i in proto.graph.initializer if i.name == name)

    def bias_shape(p):
        add = node(p, "Add")
        b = init(p, add.input[1])
        arr = numpy_helper.to_array(b).reshape(1, 1, -1)
        b.CopyFrom(numpy_helper.from_array(arr, b.name))

    def conv_has_bias(p):
        conv = node(p, "Conv")
        add = node(p, "Add")
        conv.input.append(add.input[1])

    def squeeze_axis(p):
        sq = node(p, "Squeeze")
        a = init(p, sq.input[1])
        replacement = numpy_helper.from_array(numpy_helper.to_array(a) * 0 + 3, "other_axes")
        p.graph.initializer.append(replacement)
        sq.input[1] = "other_axes"

    def other_activation(p):
        node(p, "Relu").op_type = "Sigmoid"

    def second_consumer(p):
        relu = node(p, "Relu")
        p.graph.node.append(helper.make_node("Identity", [relu.output[0]], ["extra"]))

    def grouped(p):
        conv = node(p, "Conv", 1)
        for a in conv.attribute:
            if a.name == "group":
                a.i = 2

    def conv_transpose_output_padding(p):
        node(p, "ConvTranspose").attribute.append(helper.make_attribute("output_padding", [0, 1]))

    return [("bias of another shape", bias_shape), ("conv already has a bias", conv_has_bias),
            ("squeeze on another axis", squeeze_axis), ("another activation", other_activation),
            ("tensor used twice", second_consumer), ("grouped convolution", grouped),
            ("output_padding", conv_transpose_output_padding)]


def test_cleanup_refuses_anything_but_the_exact_pattern(onnx_mod):
    from common.onnx_cleanup import CleanupError, clean_conv1d_graph

    accepted = []
    for what, change in mutations(onnx_mod):
        proto = source_proto(onnx_mod)
        change(proto)
        try:
            clean_conv1d_graph(proto)
            accepted.append(what)
        except CleanupError:
            pass
    assert accepted == []


def test_cleaned_graph_computes_the_same_outputs(onnx_mod):
    np = pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")
    from common.onnx_cleanup import clean_conv1d_graph

    raw = ap.source_bytes(SPEC, 4)
    cleaned, _ = clean_conv1d_graph(onnx_mod.load_from_string(raw))
    x = np.random.default_rng(0).standard_normal((4, 128, 1)).astype(np.float32)

    def run(model: bytes):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        return session.run(None, {"input": x})[0]

    a, b = run(raw), run(cleaned.SerializeToString())
    assert np.abs(a - b).max() <= 1e-6


def test_fp32_build_is_deterministic(onnx_mod):
    manifest = ts.read_split_manifest()
    sha = ts.sha256_file(ts.SPLIT_MANIFEST)
    first = ap.build_fp32(SPEC, 1, manifest, sha)
    second = ap.build_fp32(SPEC, 1, manifest, sha)
    assert first["imported"] == second["imported"] and first["fp32"] == second["fp32"]
    proto = onnx_mod.load_from_string(first["imported"])
    meta = {p.key: p.value for p in proto.metadata_props}
    assert meta["edgebench.weights_sha256"] == SPEC.source_sha256_of(1)
    assert json.loads(meta["edgebench.input_preprocessing"])["mean"] == \
        manifest["normalization"]["mean"]
    del proto.metadata_props[:]
    assert proto.SerializeToString() == first["raw"]  # the imported copy's graph is the source


# --- model files on disk ----------------------------------------------------------------


@pytest.fixture(scope="module")
def files(onnx_mod):
    present = [f for f in reg.read_manifest()[0] if f.model == SPEC.name
               and (reg.MODELS_DIR / f.file).is_file()]
    if not present:
        pytest.skip("autoencoder files missing; run scripts/prepare_autoencoder.py")
    return present


def test_manifest_lists_every_variant_and_batch(files):
    assert sorted(f.file for f in files) == reg.imported_files()
    for f in files:
        assert ap.sha256_bytes((reg.MODELS_DIR / f.file).read_bytes()) == f.sha256


def test_int8_files_share_scales_across_batches(files, onnx_mod):
    from common import onnx_quantize as oq

    for variant in ("int8", "int8_unpruned"):
        protos = {b: onnx_mod.load(str(reg.MODELS_DIR / reg.relative_path(SPEC.name, variant, b)))
                  for b in reg.BATCH_SIZES}
        assert oq.quant_param_differences(protos) == []


def test_int8_has_no_relu_or_bias_add_left(files, onnx_mod):
    """With the bias inside the convolution, every ReLU folds into its output range."""
    proto = onnx_mod.load(str(reg.MODELS_DIR / reg.relative_path(SPEC.name, "int8", 1)))
    ops = [n.op_type for n in proto.graph.node]
    assert "Relu" not in ops and "Add" not in ops
    assert ops.count("QuantizeLinear") == 7


def test_int8_runs_strict_on_the_npu(files):
    pytest.importorskip("onnxruntime_qnn")
    np = pytest.importorskip("numpy")
    if not ts.nab_present():
        pytest.skip("NAB data missing; run scripts/get_data.py --dataset nab")
    from common import onnx_quantize as oq

    path = reg.MODELS_DIR / reg.relative_path(SPEC.name, "int8", 1)
    try:
        session = oq.npu_session(path)
    except RuntimeError as exc:
        if "NPU" in str(exc):
            pytest.skip(str(exc))
        raise
    x = ts.load_split_windows().reference[:, :, None]
    out = np.concatenate([session.run(None, {"input": x[i:i + 1]})[0] for i in range(len(x))])
    again = np.concatenate([session.run(None, {"input": x[i:i + 1]})[0] for i in range(len(x))])
    assert np.array_equal(out, again)
    assert np.isfinite(out).all()
