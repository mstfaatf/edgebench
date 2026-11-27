"""Tests for the reference outputs (common/reference_outputs.py, scripts/make_references.py).

Selection, index format and the staleness check use only the standard library
and run in CI. The SplitMix64 reproduction needs NumPy; the tests on the
generated references skip, with the reason shown, when the model files or
models/references/ are not present.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from common import model_registry as reg
from common import reference_outputs as ro
from common.model_registry import IMPORTED_MODELS
from common.manifests import EVAL_QUICK_MANIFEST, read_manifest

SCRIPTS = Path(__file__).resolve().parents[1]


# --- input set (CI) -------------------------------------------------------------


def test_image_selection_is_deterministic_and_has_distinct_classes():
    entries = read_manifest(EVAL_QUICK_MANIFEST)
    picked = ro.select_images(entries)
    assert len(picked) == ro.IMAGE_COUNT == 8
    assert len({e.label for e in picked}) == 8
    assert ro.select_images(list(reversed(entries))) == picked  # independent of input order
    # Pinned: changing the rule or the manifest changes the reference set.
    assert [e.label for e in picked] == [843, 177, 691, 975, 3, 45, 688, 815]
    assert ro.select_images(entries, seed=ro.IMAGE_SEED + 1) != picked


def test_selection_refuses_too_few_classes():
    entries = read_manifest(EVAL_QUICK_MANIFEST)[:3]
    with pytest.raises(ValueError, match="distinct classes"):
        ro.select_images(entries)


def test_input_set_ids():
    assert ro.image_input_set("folded", None, None) == "images-unit"
    a = ro.image_input_set("external", reg.IMAGENET_MEAN, reg.IMAGENET_STD)
    assert a.startswith("images-normalized-") and len(a) == len("images-normalized-") + 8
    assert ro.image_input_set("external", (0.5,) * 3, (0.5,) * 3) != a
    assert ro.splitmix_input_set([4, 3, 224, 224]) == "splitmix64-seed0-4x3x224x224"
    assert ro.splitmix_input_set((1, 128, 1)) == "splitmix64-seed0-1x128x1"


def test_index_round_trip():
    rows = [ro.IndexRow("output", "images-unit", "m/m_fp32_b4.onnx", "cpu", 1,
                        ("inputs/images-unit/04.f32", "inputs/images-unit/05.f32"),
                        "outputs/m/m_fp32_b4/cpu/images-unit-1.f32", "float32", (4, 1000), "a" * 64),
            ro.IndexRow("input", "images-unit", "-", "-", 0, (), "inputs/images-unit/04.f32",
                        "float32", (1, 3, 224, 224), "b" * 64)]
    text = ro.format_index(rows)
    assert text.splitlines()[1].startswith("input\t")  # inputs first
    assert sorted(ro.parse_index(text), key=lambda r: r.kind) == sorted(rows, key=lambda r: r.kind)
    with pytest.raises(ValueError, match="header"):
        ro.parse_index("kind\tfile\n")


# --- staleness check on a synthetic tree (CI) -----------------------------------


def write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def tree(tmp_path):
    """A one-file model directory with a complete, current reference set."""
    models, root = tmp_path / "models", tmp_path / "models" / "references"
    rel = "m/m_fp32_b1.onnx"
    model_sha = write(models / rel, b"model")
    entry = reg.ManifestFile(rel, model_sha, 5, "m", "fp32", "fp32", "external", 1, None, None)
    rows = []
    for input_set in ("images-normalized-00000000", "splitmix64-seed0-b1"):
        inp = f"inputs/{input_set}/00.f32"
        rows.append(ro.IndexRow("input", input_set, "-", "-", 0, (), inp, "float32", (1,),
                                write(root / inp, b"x" + input_set.encode())))
    for device in ro.expected_devices(entry):  # cpu and npu_fp16
        outputs = []
        for input_set in ("images-normalized-00000000", "splitmix64-seed0-b1"):
            inp = f"inputs/{input_set}/00.f32"
            out = f"outputs/m/m_fp32_b1/{device}/{input_set}-0.f32"
            sha = write(root / out, b"y" + device.encode() + input_set.encode())
            rows.append(ro.IndexRow("output", input_set, rel, device, 0, (inp,), out, "float32",
                                    (1,), sha))
            outputs.append({"input_set": input_set, "run": 0, "sha256": sha})
        ro.write_json(ro.record_path(rel, device, root), {"model_sha256": model_sha,
                                                          "outputs": outputs})
    (root / "index.tsv").write_text(ro.format_index(rows), encoding="utf-8")
    selection = [{"path": e.path} for e in ro.select_images(read_manifest(EVAL_QUICK_MANIFEST))]
    ro.write_json(root / "inputs.json", {"images": {"selection": selection}})
    return models, root, entry


def test_current_references_pass(tree):
    models, root, entry = tree
    assert ro.check_problems([entry], root, models) == []


def test_stale_reference_is_found(tree):
    models, root, entry = tree
    (models / entry.file).write_bytes(b"model, regenerated")
    problems = ro.check_problems([entry], root, models)
    assert any("stale" in p for p in problems), problems


def test_damaged_or_missing_tensors_are_found(tree):
    models, root, entry = tree
    (root / "outputs/m/m_fp32_b1/cpu/splitmix64-seed0-b1-0.f32").write_bytes(b"changed")
    (root / "inputs/images-normalized-00000000/00.f32").unlink()
    problems = ro.check_problems([entry], root, models)
    assert any("SHA256 differs" in p for p in problems), problems
    assert any("00.f32: missing" in p for p in problems), problems


def test_missing_device_reference_is_found(tree):
    models, root, entry = tree
    int8 = reg.ManifestFile(entry.file, entry.sha256, 5, "m", "int8", "int8", "external", 1,
                            True, "c" * 64)
    assert any("no npu reference" in p for p in ro.check_problems([int8], root, models))
    ro.record_path(entry.file, "npu_fp16", root).unlink()
    assert any("no npu_fp16 reference" in p for p in ro.check_problems([entry], root, models))


def test_expected_devices_per_precision():
    def entry(precision):
        return reg.ManifestFile("m/x.onnx", "0" * 64, 1, "m", precision, precision, "external", 1,
                                None, None)
    assert ro.expected_devices(entry("int8")) == ["cpu", "npu"]
    assert ro.expected_devices(entry("fp32")) == ["cpu", "npu_fp16"]
    # INT8 files of the image models also get the finalization-mode-1 device.
    image = reg.ManifestFile("mobilenetv2/x.onnx", "0" * 64, 1, "mobilenetv2", "int8", "int8",
                             "external", 1, True, "c" * 64)
    assert ro.expected_devices(image) == ["cpu", "npu", "npu_finalization1"]
    assert ro.NPU_DEVICE_OPTIONS["npu_finalization1"] == {
        "htp_graph_finalization_optimization_mode": "1"}


def test_changed_selection_is_found(tree):
    models, root, entry = tree
    ro.write_json(root / "inputs.json", {"images": {"selection": [{"path": "0/x.jpeg"}]}})
    assert any("selection" in p for p in ro.check_problems([entry], root, models))


def test_changed_reference_windows_are_found(tree):
    from common import timeseries as ts

    models, root, entry = tree
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    starts = ts.read_split_manifest()["reference_windows"]["window_starts"]
    inputs["windows"] = {"window_starts": starts,
                         "split_manifest_sha256": ts.sha256_file(ts.SPLIT_MANIFEST)}
    ro.write_json(root / "inputs.json", inputs)
    assert ro.check_problems([entry], root, models) == []
    inputs["windows"]["window_starts"] = starts[::-1]
    ro.write_json(root / "inputs.json", inputs)
    assert any("reference windows" in p for p in ro.check_problems([entry], root, models))


def test_window_input_set_names_the_split_manifest():
    assert ro.window_input_set("b97ee8e2" + "0" * 56) == "windows-reference-b97ee8e2"


# --- SplitMix64 reproduction (needs NumPy) --------------------------------------


def test_splitmix64_matches_the_documented_generator():
    np = pytest.importorskip("numpy")
    x = ro.splitmix64_floats(0, 3)
    expected = [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]  # docs/cli.md
    assert x.dtype == np.float32
    assert [float(v) for v in x] == [(v >> 40) / 2 ** 24 for v in expected]
    assert float(x[0]) == 14819496 / 16777216

    # The scalar reference implementation from docs/cli.md, for several seeds.
    mask = (1 << 64) - 1
    for seed in (0, 1, 99, mask):
        state, scalar = seed, []
        for _ in range(500):
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            z ^= z >> 31
            scalar.append(np.float32(z >> 40) * np.float32(2.0 ** -24))
        assert np.array_equal(ro.splitmix64_floats(seed, 500), np.array(scalar, np.float32))


# --- generated references (need the files) --------------------------------------


@pytest.fixture(scope="module")
def references():
    if not ro.INDEX_TSV.is_file():
        pytest.skip("no reference outputs; run scripts/make_references.py")
    files = [f for f in reg.read_manifest()[0] if (reg.MODELS_DIR / f.file).is_file()]
    if not files:
        pytest.skip("no model files")
    return files


def test_check_command_passes(references):
    r = subprocess.run([sys.executable, str(SCRIPTS / "make_references.py"), "--check"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_file_has_cpu_and_npu_references(references):
    for f in references:
        for device in ro.expected_devices(f):
            rec = json.loads(ro.record_path(f.file, device).read_text(encoding="utf-8"))
            assert rec["model_sha256"] == f.sha256
            data = "windows" if f.model in reg.IMPORTED_MODELS else "images"
            assert {o["input_set"].split("-")[0] for o in rec["outputs"]} == {data, "splitmix64"}
            if device.startswith("npu"):
                assert rec["strict"] is True and rec["repeat_identical"] is True
                fp16 = rec["provider_options"].get("enable_htp_fp16_precision")
                assert (fp16 == "1") == (device == "npu_fp16"), f.file
            else:
                assert rec["intra_op_threads"] == ro.CPU_THREADS


def test_cpu_references_are_reproducible(references):
    """Running the CPU reference session again gives byte-identical outputs."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    sys.path.insert(0, str(SCRIPTS))
    import make_references as mk

    rows = ro.parse_index(ro.INDEX_TSV.read_text(encoding="utf-8"))
    for f in references[::9]:
        session = mk.cpu_session(reg.MODELS_DIR / f.file)
        name = session.get_inputs()[0].name
        for r in (r for r in rows if r.model == f.file and r.device == "cpu"):
            x = np.concatenate([np.fromfile(ro.REFERENCE_DIR / i, dtype="<f4")
                                for i in r.inputs]).reshape(session.get_inputs()[0].shape)
            y = np.ascontiguousarray(session.run(None, {name: x})[0], dtype="<f4")
            assert hashlib.sha256(y.tobytes()).hexdigest() == r.sha256, r.file


# --- batch equivalence ------------------------------------------------------------


def _floats(*values: float) -> bytes:
    from array import array

    return array("f", values).tobytes()


def batch_tree(tmp_path, b4_items: list[bytes]):
    """Records for a b1 and a b4 file (cpu device) with 4 one-float image items."""
    root = tmp_path / "references"
    b1 = reg.ManifestFile("m/m_int8_b1.onnx", "1" * 64, 5, "m", "int8", "int8", "external", 1,
                          True, "f" * 64)
    b4 = reg.ManifestFile("m/m_int8_b4.onnx", "4" * 64, 5, "m", "int8", "int8", "external", 4,
                          True, "e" * 64)
    items = [_floats(float(i)) for i in range(4)]
    outputs = []
    for run, item in enumerate(items):
        out = f"outputs/m/m_int8_b1/cpu/images-x-{run}.f32"
        write(root / out, item)
        outputs.append({"input_set": "images-x", "run": run, "file": out, "shape": [1, 1]})
    ro.write_json(ro.record_path(b1.file, "cpu", root), {"model_sha256": b1.sha256, "outputs": outputs})
    out = "outputs/m/m_int8_b4/cpu/images-x-0.f32"
    write(root / out, b"".join(b4_items))
    ro.write_json(ro.record_path(b4.file, "cpu", root), {
        "model_sha256": b4.sha256,
        "outputs": [{"input_set": "images-x", "run": 0, "file": out, "shape": [4, 1]},
                    {"input_set": "splitmix64-seed0-4x1", "run": 0, "file": out, "shape": [4, 1]}]})
    return root, [b1, b4], items


def test_batch_equivalence_of_identical_items(tmp_path):
    root, manifest, items = batch_tree(tmp_path, [_floats(float(i)) for i in range(4)])
    entries = {e["file"]: e for e in ro.batch_equivalence(manifest, root)["entries"]}
    assert entries["m/m_int8_b1.onnx"]["bitwise_equal_to_b1"] is True
    b4 = entries["m/m_int8_b4.onnx"]
    assert (b4["bitwise_equal_to_b1"], b4["items_compared"], b4["items_differing"]) == (True, 4, 0)
    assert (b4["sha256"], b4["provider"], b4["compute_precision"]) == ("4" * 64, "cpu", "int8")
    assert b4["ep_compile_options"] == {}  # the EP defaults
    assert ro.entry_key(b4) == ("4" * 64, "cpu", "int8", "{}")
    # Bit-identical outputs make identical predictions: nothing to run.
    assert {k: b4[k] for k in ro.PREDICTION_KEYS} == ro.BITWISE_PREDICTIONS


def test_batch_equivalence_finds_one_differing_item(tmp_path):
    root, manifest, _ = batch_tree(tmp_path, [_floats(0.0), _floats(1.0), _floats(2.5), _floats(3.0)])
    b4 = next(e for e in ro.batch_equivalence(manifest, root)["entries"] if e["batch_size"] == 4)
    assert (b4["bitwise_equal_to_b1"], b4["items_differing"], b4["max_abs_diff"]) == (False, 1, 0.5)
    # Without a prediction comparison the answer is not available.
    assert {k: b4[k] for k in ro.PREDICTION_KEYS} == ro.NO_PREDICTIONS


def test_batch_equivalence_takes_prediction_comparisons_by_key(tmp_path):
    root, manifest, _ = batch_tree(tmp_path, [_floats(0.0), _floats(1.0), _floats(2.5), _floats(3.0)])
    found = {"predictions_equal_to_b1": False, "predictions_basis": "eval_quick_top1",
             "predictions_compared": 1000, "predictions_differing": 6}
    doc = ro.batch_equivalence(manifest, root, {("4" * 64, "cpu", "int8", "{}"): found}, {"x": 1})
    b4 = next(e for e in doc["entries"] if e["batch_size"] == 4)
    assert {k: b4[k] for k in ro.PREDICTION_KEYS} == found
    assert doc["prediction_inputs"] == {"x": 1}
    assert ro.committed_predictions(doc) == {("4" * 64, "cpu", "int8", "{}"): found}


def test_batch_equivalence_problems_name_changed_entries(tmp_path):
    root, manifest, _ = batch_tree(tmp_path, [_floats(float(i)) for i in range(4)])
    path = tmp_path / "batch_equivalence.json"
    doc = ro.batch_equivalence(manifest, root)
    path.write_text(ro.format_batch_equivalence(doc), encoding="utf-8")
    assert ro.batch_equivalence_problems(manifest, root, path) == []
    doc["entries"][1]["bitwise_equal_to_b1"] = False
    path.write_text(ro.format_batch_equivalence(doc), encoding="utf-8")
    assert any("m_int8_b4" in p and "differs" in p
               for p in ro.batch_equivalence_problems(manifest, root, path))


def test_batch_equivalence_problems_need_a_prediction_comparison(tmp_path):
    root, manifest, _ = batch_tree(tmp_path, [_floats(0.0), _floats(1.0), _floats(2.5), _floats(3.0)])
    path = tmp_path / "batch_equivalence.json"
    path.write_text(ro.format_batch_equivalence(ro.batch_equivalence(manifest, root)),
                    encoding="utf-8")
    assert any("no prediction comparison" in p
               for p in ro.batch_equivalence_problems(manifest, root, path))
    found = {"predictions_equal_to_b1": True, "predictions_basis": "eval_quick_top1",
             "predictions_compared": 1000, "predictions_differing": 0}
    doc = ro.batch_equivalence(manifest, root, {("4" * 64, "cpu", "int8", "{}"): found}, {"x": 1})
    path.write_text(ro.format_batch_equivalence(doc), encoding="utf-8")
    assert ro.batch_equivalence_problems(manifest, root, path, {"x": 1}) == []
    # A split manifest changed since the comparison was made.
    assert any("other inputs" in p
               for p in ro.batch_equivalence_problems(manifest, root, path, {"x": 2}))


def finalization_tree(tmp_path, b4_default: list[bytes], b4_mode1: list[bytes]):
    """batch_tree plus npu and npu_finalization1 records of a b1 and b4 image INT8 file."""
    root = tmp_path / "references"
    b1 = reg.ManifestFile("mobilenetv2/mobilenetv2_int8_b1.onnx", "1" * 64, 5, "mobilenetv2",
                          "int8", "int8", "external", 1, True, "f" * 64)
    b4 = reg.ManifestFile("mobilenetv2/mobilenetv2_int8_b4.onnx", "4" * 64, 5, "mobilenetv2",
                          "int8", "int8", "external", 4, True, "e" * 64)
    items = [_floats(float(i)) for i in range(4)]
    for device, b4_items in (("cpu", items), ("npu", b4_default),
                             ("npu_finalization1", b4_mode1)):
        outputs = []
        for run, item in enumerate(items):
            out = f"outputs/mobilenetv2/mobilenetv2_int8_b1/{device}/images-x-{run}.f32"
            write(root / out, item)
            outputs.append({"input_set": "images-x", "run": run, "file": out, "shape": [1, 1]})
        ro.write_json(ro.record_path(b1.file, device, root),
                      {"model_sha256": b1.sha256, "outputs": outputs})
        out = f"outputs/mobilenetv2/mobilenetv2_int8_b4/{device}/images-x-0.f32"
        write(root / out, b"".join(b4_items))
        ro.write_json(ro.record_path(b4.file, device, root), {
            "model_sha256": b4.sha256,
            "outputs": [{"input_set": "images-x", "run": 0, "file": out, "shape": [4, 1]}]})
    return root, [b1, b4], items


def test_finalization_entries_have_their_own_options_and_compare_with_the_defaults(tmp_path):
    differ = [_floats(0.0), _floats(1.0), _floats(2.5), _floats(3.0)]
    same = [_floats(float(i)) for i in range(4)]
    root, manifest, _ = finalization_tree(tmp_path, differ, same)
    doc = ro.batch_equivalence(manifest, root)
    by = {(e["batch_size"], e["reference_device"]): e for e in doc["entries"]}
    default, mode1 = by[(4, "npu")], by[(4, "npu_finalization1")]
    assert default["ep_compile_options"] == {} and default["bitwise_equal_to_b1"] is False
    assert mode1["ep_compile_options"] == {"htp_graph_finalization_optimization_mode": "1"}
    assert mode1["bitwise_equal_to_b1"] is True
    assert ro.entry_key(default) != ro.entry_key(mode1)
    # Compared with the default-options outputs of the same file: item 2 differs.
    assert mode1["vs_default_options"] == {"reference_device": "npu", "items_compared": 4,
                                           "items_differing": 1, "max_abs_diff": 0.5}
    assert "vs_default_options" not in default
    # Batch-N entries of the confirmed device need an eval-quick confirmation.
    assert mode1["eval_quick_confirmation"] is None
    assert "eval_quick_confirmation" not in by[(1, "npu_finalization1")]
    path = tmp_path / "batch_equivalence.json"
    found = {"predictions_equal_to_b1": False, "predictions_basis": "eval_quick_top1",
             "predictions_compared": 1000, "predictions_differing": 6}
    preds = {ro.entry_key(default): found}
    path.write_text(ro.format_batch_equivalence(ro.batch_equivalence(manifest, root, preds)),
                    encoding="utf-8")
    assert any("no eval-quick confirmation" in p
               for p in ro.batch_equivalence_problems(manifest, root, path))
    confirmation = {"basis": "eval_quick_top1", "compared": 1000, "outputs_differing": 0,
                    "top1_differing": 0, "outputs_bitwise_equal_to_b1": True}
    doc = ro.batch_equivalence(manifest, root, preds, confirmations={ro.entry_key(mode1): confirmation})
    path.write_text(ro.format_batch_equivalence(doc), encoding="utf-8")
    assert ro.batch_equivalence_problems(manifest, root, path) == []
    assert ro.committed_confirmations(doc) == {ro.entry_key(mode1): confirmation}


def test_npu_compute_precision_per_reference_device():
    fp32 = reg.ManifestFile("m/m_fp32_b1.onnx", "1" * 64, 5, "m", "fp32", "fp32", "external", 1, None, None)
    assert [ro.compute_precision(fp32, d) for d in ro.expected_devices(fp32)] == ["fp32", "fp16"]


def test_committed_batch_equivalence_matches_the_manifest():
    """The committed summary (runs in CI): well formed, one entry per file, provider and
    compute precision, keyed by the manifest's SHA256, batch 1 always equivalent."""
    doc = json.loads(ro.BATCH_EQUIVALENCE_JSON.read_text(encoding="utf-8"))
    assert doc["batch_equivalence_version"] == ro.BATCH_EQUIVALENCE_VERSION
    manifest = {f.file: f for f in reg.read_manifest()[0]}
    keys = set()
    for e in doc["entries"]:
        f = manifest[e["file"]]
        assert e["sha256"] == f.sha256 and e["batch_size"] == f.batch_size, e["file"]
        assert e["b1_sha256"] == manifest[e["b1_file"]].sha256
        key = ro.entry_key(e)
        assert e["ep_compile_options"] == ro.DEVICE_COMPILE_OPTIONS.get(e["reference_device"], {})
        assert key not in keys
        keys.add(key)
        if e["batch_size"] == 1:
            assert e["bitwise_equal_to_b1"] is True and e["items_differing"] == 0
        assert e["bitwise_equal_to_b1"] == (e["items_differing"] == 0)
        if e["bitwise_equal_to_b1"]:
            assert {k: e[k] for k in ro.PREDICTION_KEYS} == ro.BITWISE_PREDICTIONS
        else:  # every file whose bits differ has its predictions compared
            assert e["predictions_basis"] == ("test_split_flags" if e["model"] in IMPORTED_MODELS
                                              else "eval_quick_top1"), e["file"]
            assert e["predictions_equal_to_b1"] == (e["predictions_differing"] == 0)
            assert 0 <= e["predictions_differing"] <= e["predictions_compared"]
        if e["reference_device"] in ro.CONFIRMED_DEVICES and e["batch_size"] > 1:
            c = e["eval_quick_confirmation"]
            assert c["compared"] == 1000 and c["outputs_bitwise_equal_to_b1"] == (
                c["outputs_differing"] == 0), e["file"]
    assert len(keys) == sum(len(ro.expected_devices(f)) for f in manifest.values())
    # Finalization mode 1 makes every image INT8 file at batch 4 and 8 compute exactly
    # batch 1's outputs, on the references and on all of eval-quick.
    mode1 = [e for e in doc["entries"] if e["reference_device"] == "npu_finalization1"]
    assert len(mode1) == 36
    assert all(e["bitwise_equal_to_b1"] for e in mode1)
    assert all(e["eval_quick_confirmation"]["outputs_differing"] == 0
               for e in mode1 if e["batch_size"] > 1)
    assert ro.BATCH_EQUIVALENCE_JSON.stat().st_size < 1 << 20


def test_committed_batch_equivalence_matches_the_references(references):
    from common import batch_predictions as bp

    assert ro.batch_equivalence_problems(reg.read_manifest()[0],
                                         prediction_inputs=bp.basis_inputs()) == []
