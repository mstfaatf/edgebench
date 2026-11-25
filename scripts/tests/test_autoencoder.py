"""The autoencoder architecture, its ONNX stand-ins and the training notebook.

Architecture and notebook checks use only the standard library (and git) and
run in CI; stand-in tests need onnx, onnxruntime and NumPy, and the NPU test
needs onnxruntime-qnn and the HTP device. They skip visibly otherwise.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from common import autoencoder as ae
from common import timeseries as ts

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks" / "train_autoencoder.ipynb"
NOTEBOOK_PINS = {"tensorflow": "2.21.0", "keras": "3.15.1", "tf2onnx": "1.17.0",
                 "onnx": "1.23.2", "onnxruntime": "1.32.0"}


# --- architecture ----------------------------------------------------------------


@pytest.mark.parametrize("length, kernel, stride, expected", [
    (128, 7, 2, (2, 3)), (64, 7, 2, (2, 3)), (128, 7, 1, (3, 3)), (5, 1, 1, (0, 0)),
    (7, 3, 2, (1, 1)), (8, 3, 2, (0, 1)),
])
def test_same_padding_matches_tensorflow(length, kernel, stride, expected):
    assert ae.same_padding(length, kernel, stride) == expected


def test_conv1d_design_halves_and_doubles_back_to_the_window():
    assert ae.layer_lengths("conv1d") == [64, 32, 64, 128, 128]
    assert ae.DESIGNS["conv1d"][-1].units == 1 and ae.DESIGNS["conv1d"][-1].activation is None
    assert ae.input_shape("conv1d", 4) == [4, ts.WINDOW_LENGTH, 1]
    assert ae.input_shape("dense", 8) == [8, ts.WINDOW_LENGTH]


def test_parameter_counts_are_small():
    # Keras counts: Conv1D k*in*out + out, Conv1DTranspose k*out*in + out, Dense in*out + out.
    assert ae.parameter_count("conv1d") == (7 * 1 * 32 + 32) + (7 * 32 * 16 + 16) + \
        (7 * 16 * 16 + 16) + (7 * 16 * 32 + 32) + (7 * 32 * 1 + 1) == 9505
    assert ae.parameter_count("dense") == 18704
    assert ae.parameter_count("conv1d") * 4 < (1 << 20)


def test_every_standin_names_a_design():
    assert set(ae.STANDINS.values()) == set(ae.DESIGNS)


# --- ONNX stand-ins (onnx, onnxruntime, NumPy) -----------------------------------


@pytest.fixture(scope="module")
def onnx_stack():
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    return np, onnx, ort


@pytest.mark.parametrize("layout", sorted(ae.STANDINS))
@pytest.mark.parametrize("batch", [1, 4])
def test_standin_matches_the_numpy_keras_reference(onnx_stack, layout, batch):
    np, onnx, ort = onnx_stack
    from common.onnx_export import static_shape_problems

    design = ae.STANDINS[layout]
    proto = ae.build_standin(layout, batch, seed=3)
    assert static_shape_problems(proto) == []
    assert len(proto.SerializeToString()) < (1 << 20)
    x = ae.model_inputs(design, np.random.default_rng(batch).normal(
        size=(batch, ts.WINDOW_LENGTH)).astype(np.float32))
    session = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    y = session.run(None, {ae.INPUT_NAME: x})[0]
    assert [o.name for o in session.get_outputs()] == [ae.OUTPUT_NAME]
    assert y.shape == x.shape
    np.testing.assert_allclose(y, ae.reference_forward(design, ae.random_weights(design, 3), x),
                               atol=1e-5, rtol=0)


def test_standin_layouts_of_one_design_compute_the_same_function(onnx_stack):
    np, _, ort = onnx_stack
    x = np.random.default_rng(0).normal(size=(4, ts.WINDOW_LENGTH, 1)).astype(np.float32)
    outs = [ort.InferenceSession(ae.build_standin(layout, 4).SerializeToString(),
                                 providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
            for layout in ("conv1d", "conv1d_per_layer", "conv1d_native")]
    for other in outs[1:]:
        np.testing.assert_allclose(other, outs[0], atol=1e-5, rtol=0)


def test_rename_io(onnx_stack):
    _, onnx, _ = onnx_stack
    proto = ae.build_standin("dense", 1)
    ae.rename_io(proto, "x", "y")
    assert [i.name for i in proto.graph.input] == ["x"]
    assert [o.name for o in proto.graph.output] == ["y"]
    onnx.checker.check_model(proto, full_check=True)
    ae.rename_io(proto)  # and back
    assert proto.graph.input[0].name == ae.INPUT_NAME
    with pytest.raises(ValueError, match="already used"):
        ae.rename_io(proto, input_name=proto.graph.node[0].output[0])


def test_standin_quantizes_with_the_int8_recipe(onnx_stack, tmp_path):
    np, onnx, _ = onnx_stack
    pytest.importorskip("onnxruntime.quantization")
    from common import onnx_quantize as oq

    fp32 = tmp_path / "fp32.onnx"
    onnx.save(ae.build_standin("conv1d", 1), str(fp32))
    rng = np.random.default_rng(0)
    samples = [rng.normal(size=(1, ts.WINDOW_LENGTH, 1)).astype(np.float32) for _ in range(8)]
    out = tmp_path / "int8.onnx"
    oq.quantize_qdq(fp32, out, oq.array_reader(ae.INPUT_NAME, samples),
                    tmp_path / "ranges.json", tmp_path)
    ops = [n.op_type for n in onnx.load(str(out)).graph.node]
    assert ops.count("Conv") == 3 and ops.count("ConvTranspose") == 2
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
    int8 = [i for i in onnx.load(str(out)).graph.initializer
            if i.data_type == onnx.TensorProto.INT8]
    kernels = [i for i in int8 if len(i.dims) == 4]
    zero_points = [i for i in int8 if len(i.dims) == 1]
    assert len(kernels) == 5 and len(zero_points) == 5  # per-channel int8, one per layer


def test_conv1d_int8_runs_strict_on_the_npu(onnx_stack, tmp_path):
    np, onnx, ort = onnx_stack
    pytest.importorskip("onnxruntime_qnn")
    from common import onnx_quantize as oq
    from common.ort_qnn import NoNpuDeviceError

    fp32 = tmp_path / "fp32.onnx"
    onnx.save(ae.build_standin("conv1d", 1), str(fp32))
    rng = np.random.default_rng(0)
    samples = [rng.normal(size=(1, ts.WINDOW_LENGTH, 1)).astype(np.float32) for _ in range(16)]
    int8 = tmp_path / "int8.onnx"
    oq.quantize_qdq(fp32, int8, oq.array_reader(ae.INPUT_NAME, samples),
                    tmp_path / "ranges.json", tmp_path)
    try:
        session = oq.npu_session(int8)
    except NoNpuDeviceError as exc:
        pytest.skip(str(exc))
    x = samples[0]
    npu = session.run(None, {ae.INPUT_NAME: x})[0]
    cpu = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"]).run(
        None, {ae.INPUT_NAME: x})[0]
    assert npu.shape == x.shape
    assert float(np.abs(npu - cpu).max()) < 0.5


# --- the notebook ----------------------------------------------------------------


def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells() -> list[str]:
    return ["".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == "code"]


def test_notebook_is_committed_without_outputs():
    nb = notebook()
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_notebook_cells_are_plain_python():
    for source in code_cells():
        ast.parse(source)  # no shell or IPython magics


def test_notebook_installs_the_pinned_versions_first():
    first = ast.parse(code_cells()[0])
    pins = next(ast.literal_eval(node.value) for node in ast.walk(first)
                if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PINS")
    assert pins == NOTEBOOK_PINS
    assert "pip" in code_cells()[0] and "install" in code_cells()[0]


def pinned_commit() -> str:
    for source in code_cells():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == \
                    "EDGEBENCH_COMMIT":
                return ast.literal_eval(node.value)
    raise AssertionError("EDGEBENCH_COMMIT not set in the notebook")


def test_notebook_uses_the_shared_code_and_no_copy_of_it():
    sources = "\n".join(code_cells())
    for needed in ("ts.load_split_windows()", "ae.DESIGNS[", "get_data.py", "--dataset",
                   "enable_op_determinism", "set_random_seed", "from_function"):
        assert needed in sources, needed
    for duplicated in ("def window", "def normaliz", "def plan_splits", ".mean()", ".std()"):
        assert duplicated not in sources, duplicated


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def test_notebook_commit_has_the_current_split_manifest():
    commit = pinned_commit()
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    if shutil.which("git") is None or git("rev-parse", "HEAD").returncode != 0:
        pytest.skip("not a git checkout")
    if git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        pytest.skip(f"commit {commit[:12]} not in this clone (shallow or not fetched)")
    assert git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0, \
        "the notebook's commit is not an ancestor of HEAD"
    manifest_at_commit = git("show", f"{commit}:datasets/nab-machine-temperature/splits.json")
    assert manifest_at_commit.returncode == 0, manifest_at_commit.stderr
    assert json.loads(manifest_at_commit.stdout) == ts.read_split_manifest(), \
        "the split manifest changed since the notebook's commit; update EDGEBENCH_COMMIT"
    for path in ("scripts/common/timeseries.py", "scripts/common/autoencoder.py",
                 "scripts/get_data.py"):
        assert git("cat-file", "-e", f"{commit}:{path}").returncode == 0, path
