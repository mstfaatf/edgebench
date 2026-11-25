"""Tests for the ImageNetV2 accuracy evaluation and its published results.

Statistics, the result validator and the checks on the committed results in
benchmarks/results/published/accuracy/ use only the standard library and
jsonschema, and run in CI. The data-path tests need NumPy and Pillow; the
recomputation from the per-image predictions skips, with the reason shown,
when the prediction files are not present.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from common import accuracy_stats as st
from common import model_registry as reg

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import validate_accuracy as va  # noqa: E402
import validate_results as vr  # noqa: E402

PUBLISHED = reg.REPO_ROOT / "benchmarks" / "results" / "published" / "accuracy"
EVAL_FULL = reg.REPO_ROOT / "datasets" / "imagenetv2-matched-frequency" / "eval-full.tsv"


# --- statistics (CI) ---------------------------------------------------------------


def test_wilson_interval_known_values():
    low, high = st.wilson_interval(50, 100)
    assert low == pytest.approx(0.403831, abs=1e-6) and high == pytest.approx(0.596169, abs=1e-6)
    assert st.wilson_interval(0, 10)[0] == 0.0 and st.wilson_interval(10, 10)[1] == 1.0
    low, high = st.wilson_interval(0, 10)
    assert high == pytest.approx(0.277533, abs=1e-6)  # not degenerate at 0, unlike Wald
    with pytest.raises(ValueError):
        st.wilson_interval(11, 10)


def test_mcnemar_exact():
    assert st.mcnemar_exact(0, 0) == 1.0
    assert st.mcnemar_exact(5, 5) == 1.0
    assert st.mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)
    assert st.mcnemar_exact(9, 1) == st.mcnemar_exact(1, 9)
    assert st.mcnemar_exact(0, 3) == pytest.approx(0.25)
    # Large discordant counts: close to the chi-square approximation with continuity correction.
    import math

    b, c = 480, 560
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    assert st.mcnemar_exact(b, c) == pytest.approx(math.erfc(math.sqrt(chi2 / 2)), rel=0.05)


def test_paired_counts():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    assert st.paired_counts(a, b) == {"both_correct": 1, "only_a_correct": 2, "only_b_correct": 1,
                                      "both_wrong": 1}
    with pytest.raises(ValueError):
        st.paired_counts([True], [True, False])


# --- committed results (CI) ----------------------------------------------------------


@pytest.fixture(scope="module")
def published():
    files = sorted(PUBLISHED.glob("*.json"))
    if not files:
        pytest.skip("no published accuracy results")
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in files}


def test_published_results_validate(published):
    results = va.validate_directory(sorted(PUBLISHED.glob("*.json")))
    assert len(results) == 22
    assert {str(p): i for p, i in results.items() if i} == {}


def test_published_runs_cover_the_plan(published):
    runs = {k: v for k, v in published.items() if v["kind"] == "run"}
    assert len(runs) == 21
    manifest = {f.file: f for f in reg.read_manifest()[0]}
    eval_full_sha = hashlib.sha256(EVAL_FULL.read_bytes()).hexdigest()
    for run_id, r in runs.items():
        assert r["dataset"]["split"] == "eval-full"
        assert r["dataset"]["manifest_sha256"] == eval_full_sha
        assert r["dataset"]["image_count"] == 10000
        entry = manifest[r["model"]["file"]]
        assert r["model"]["sha256"] == entry.sha256, run_id  # tied to the manifest
        assert entry.pruned in (None, True)  # pruned files only
        ex = r["execution"]
        if ex["device"] == "npu":
            assert ex["strict"] and ex["session_config"] == {"session.disable_cpu_ep_fallback": "1"}
        if run_id.endswith("_fp16"):
            assert ex["compute_precision"] == "fp16" and r["model"]["stored_precision"] == "fp32"
            assert ex["provider_options"]["enable_htp_fp16_precision"] == "1"
        elif ex["device"] == "npu":
            assert ex["compute_precision"] == "int8"
        if r["status"] == "success":
            assert ex["batch_equivalence"]["identical_top5"] is True
            batch = r["model"]["batch_size"]
            assert ex["batch_equivalence"]["candidates"][f"b{batch}"] == "identical"
        pre = r["preprocessing"]
        assert pre["image_cache"]["bit_identical"] is True
        assert (pre["mean"] is None) == (r["model"]["normalization"] == "folded")


def test_published_comparisons_include_the_npu_cpu_question(published):
    comparisons = published["comparisons"]["comparisons"]
    names = {c["name"] for c in comparisons}
    for m in reg.MODELS:
        assert f"{m}: NPU INT8 vs CPU INT8" in names
        assert f"{m}: NPU INT8 folded vs CPU INT8 folded" in names


@pytest.mark.parametrize("change,message", [
    (lambda d: d["mobilenetv2_int8_cpu"]["metrics"]["top1"].update(accuracy=0.9), "accuracy"),
    (lambda d: d["mobilenetv2_int8_cpu"]["metrics"]["top1"]["ci95"].update(low=0.1), "Wilson"),
    (lambda d: d["comparisons"]["comparisons"][0].update(p_value=0.5), "McNemar"),
    (lambda d: d["comparisons"]["comparisons"][0]["counts"].update(both_wrong=0), "add up"),
    (lambda d: d["comparisons"]["runs"]["mobilenetv2_int8_cpu"].update(
        top5_indices_sha256="0" * 64), "does not match its result"),
])
def test_validator_finds_tampering(published, tmp_path, change, message):
    docs = json.loads(json.dumps(published))
    change(docs)
    for name, doc in docs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
    issues = [i for v in va.validate_directory(sorted(tmp_path.glob("*.json"))).values() for i in v]
    assert any(message in i for i in issues), issues


def test_validate_results_skips_accuracy_documents(published, tmp_path, capsys):
    shutil.copy(PUBLISHED / "comparisons.json", tmp_path / "comparisons.json")
    assert vr.main([str(tmp_path), "--strict"]) == 0
    out = capsys.readouterr().out
    assert "SKIP" in out and "1 skipped" in out


# --- data path (needs NumPy and Pillow) --------------------------------------------


def test_uint8_path_equals_the_shared_helper():
    np = pytest.importorskip("numpy")
    image = pytest.importorskip("PIL.Image")
    from common import preprocess as pp

    rng = np.random.default_rng(7)
    img = image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))
    assert np.array_equal(pp.unit_from_uint8(pp.preprocess_image_uint8(img)),
                          pp.preprocess_image_unit(img))
    batch = np.stack([pp.preprocess_image_uint8(img)] * 2)
    assert pp.unit_from_uint8(batch).shape == (2, 3, 224, 224)


def test_top_k_breaks_ties_by_lower_index():
    np = pytest.importorskip("numpy")
    from common import imagenet_eval as ie

    logits = np.array([[0.0, 2.0, 2.0, 1.0, 3.0, -1.0]], dtype=np.float32)
    idx, val = ie.top_k(logits, 3)
    assert idx.tolist() == [[4, 1, 2]] and val.tolist() == [[3.0, 2.0, 2.0]]


def test_labels_are_checked_against_folders(tmp_path):
    pytest.importorskip("numpy")
    from common import imagenet_eval as ie
    from common.manifests import EVAL_QUICK_MANIFEST

    good = ie.load_eval_set("eval-quick", EVAL_QUICK_MANIFEST)
    assert len(good.entries) == 1000 and good.labels.tolist() == list(range(1000))
    lines = EVAL_QUICK_MANIFEST.read_text(encoding="utf-8").splitlines()
    path, _, sha = lines[1].split("\t")
    lines[1] = "\t".join([path, "1", sha])  # folder 0, label 1
    bad = tmp_path / "bad.tsv"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        ie.load_eval_set("eval-quick", bad)


def test_metrics_recompute_from_the_predictions(published):
    np = pytest.importorskip("numpy")
    from common.manifests import read_manifest

    labels = np.array([e.label for e in read_manifest(EVAL_FULL)])
    checked = 0
    for r in (r for r in published.values() if r["kind"] == "run" and r["predictions"]):
        ref = r["predictions"]["top5_indices"]
        path = reg.REPO_ROOT / ref["file"]
        if not path.is_file():
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]
        top5 = np.load(path)
        assert int((top5[:, 0] == labels).sum()) == r["metrics"]["top1"]["correct"]
        assert int(np.any(top5 == labels[:, None], axis=1).sum()) == r["metrics"]["top5"]["correct"]
        checked += 1
    if not checked:
        pytest.skip("prediction files not present; run scripts/eval_accuracy.py")
