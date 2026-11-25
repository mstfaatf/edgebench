"""Tests for the anomaly-detection metrics and the published autoencoder results.

The metric tests and the checks of the committed results in
benchmarks/results/published/accuracy/anomaly/ use only the standard library
and jsonschema, and run in CI. Recomputing the metrics from the per-window
score files needs NumPy, the score files (``scripts/eval_anomaly.py``, not
committed) and the NAB data; those tests skip visibly otherwise.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

import validate_accuracy as va
from common import anomaly_stats as st
from common import model_registry as reg

pytest.importorskip("jsonschema")

SCRIPTS = Path(__file__).resolve().parents[1]
PUBLISHED = reg.REPO_ROOT / "benchmarks" / "results" / "published" / "accuracy" / "anomaly"
RUNS = ("conv1d_autoencoder_fp32_cpu", "conv1d_autoencoder_int8_cpu",
        "conv1d_autoencoder_int8_npu", "conv1d_autoencoder_fp32_npu_fp16")


# --- metrics ------------------------------------------------------------------------


def brute_force_auc(scores, labels):
    pos = [s for s, v in zip(scores, labels) if v]
    neg = [s for s, v in zip(scores, labels) if not v]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


@pytest.mark.parametrize("seed", range(5))
def test_roc_auc_is_the_pairwise_probability_with_ties_as_half(seed):
    rng = random.Random(seed)
    scores = [rng.choice([0.1, 0.2, 0.3, 0.4, 0.5]) for _ in range(40)]
    labels = [rng.random() < 0.3 for _ in range(40)]
    labels[0], labels[1] = True, False
    assert st.roc_auc(scores, labels) == pytest.approx(brute_force_auc(scores, labels), abs=1e-12)


def test_roc_auc_extremes_and_errors():
    assert st.roc_auc([0.1, 0.2, 0.9, 0.8], [False, False, True, True]) == 1.0
    assert st.roc_auc([0.9, 0.8, 0.1, 0.2], [False, False, True, True]) == 0.0
    assert st.roc_auc([0.5] * 4, [False, True, False, True]) == 0.5
    with pytest.raises(ValueError):
        st.roc_auc([0.1, 0.2], [True, True])


def test_average_precision_known_values():
    # The example of scikit-learn's average_precision_score documentation: 0.8333...
    assert st.average_precision([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == \
        pytest.approx(5 / 6)
    assert st.average_precision([0.9, 0.1], [True, False]) == 1.0
    # A tie between a positive and a negative counts as one threshold: precision 1/2.
    assert st.average_precision([0.5, 0.5], [True, False]) == 0.5


def test_confusion_flags_strictly_above_the_threshold():
    c = st.confusion([0.1, 0.5, 0.6, 0.5], [False, True, True, False], 0.5)
    assert c == {"tp": 1, "fp": 0, "fn": 1, "tn": 2}
    prf = st.precision_recall_f1(c)
    assert (prf["precision"], prf["recall"]) == (1.0, 0.5)
    assert prf["f1"] == pytest.approx(2 / 3)
    assert st.precision_recall_f1({"tp": 0, "fp": 0, "fn": 3, "tn": 1}) == \
        {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_spearman_uses_average_ranks():
    assert st.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert st.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    a, b = [1, 2, 2, 3], [1, 3, 2, 4]
    ra, rb = st.average_ranks(a), st.average_ranks(b)
    assert ra == [1, 2.5, 2.5, 4]
    mean = 2.5
    pearson = sum((x - mean) * (y - mean) for x, y in zip(ra, rb)) / (
        sum((x - mean) ** 2 for x in ra) * sum((y - mean) ** 2 for y in rb)) ** 0.5
    assert st.spearman(a, b) == pytest.approx(pearson)


def test_events_report_the_first_flag_and_its_delay():
    starts = list(range(20))
    flags = [False] * 20
    flags[8] = flags[12] = True
    ev = st.events(starts, flags, [(10, 14), (17, 30)], window=4)
    assert ev[0] == {"start": 10, "stop": 14, "windows": 7, "flagged_windows": 2,
                     "detected": True, "first_flagged_window_start": 8, "detection_point": 11,
                     "delay_points": 1, "delay_fraction_of_range": 0.25}
    assert ev[1]["detected"] is False and ev[1]["windows"] == 6
    assert ev[1]["delay_points"] is None


# --- published results (CI) -----------------------------------------------------------


@pytest.fixture(scope="module")
def published():
    files = sorted(PUBLISHED.glob("*.json"))
    if not files:
        pytest.skip("no published anomaly-detection results")
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in files}


def test_published_results_validate(published):
    results = va.validate_directory(sorted(PUBLISHED.glob("*.json")))
    assert len(results) == len(published)
    assert {str(p): i for p, i in results.items() if i} == {}


def test_published_runs_cover_the_plan(published):
    runs = {k: v for k, v in published.items() if v["kind"] == "run"}
    assert sorted(runs) == sorted(RUNS)
    files = {f.file: f for f in reg.read_manifest()[0]}
    for run_id, r in runs.items():
        assert r["status"] == "success", run_id
        entry = files[r["model"]["file"]]
        assert entry.sha256 == r["model"]["sha256"] and entry.variant == r["model"]["variant"]
        assert r["threshold"]["rule"] == "max_validation_score"
        assert r["threshold"]["split"] == "validation"
        candidates = r["execution"]["batch_equivalence"]["candidates"]
        # The largest batch size whose reconstructions equal batch 1's.
        assert r["model"]["batch_size"] == max(int(b[1:]) for b, v in candidates.items()
                                               if v == "identical")
    assert sorted(c["name"] for c in published["comparisons"]["comparisons"]) == \
        ["CPU INT8 vs CPU FP32", "NPU FP16 vs CPU FP32", "NPU INT8 vs CPU INT8"]


TAMPERING = [
    (lambda d: d["conv1d_autoencoder_int8_cpu"]["metrics"]["at_threshold"].update(tp=1),
     "do not add up"),
    (lambda d: d["conv1d_autoencoder_int8_cpu"]["metrics"]["at_threshold"].update(f1=0.9),
     "f1"),
    (lambda d: d["conv1d_autoencoder_fp32_cpu"]["metrics"]["events"].update(detected=4),
     "events detected"),
    (lambda d: d["conv1d_autoencoder_int8_npu"]["metrics"]["events"]["ranges"][0].update(
        delay_points=0), "delay"),
    (lambda d: d["comparisons"]["comparisons"][0].update(p_value=0.5), "McNemar"),
    (lambda d: d["comparisons"]["comparisons"][0].update(flag_agreement=0.5), "flag_agreement"),
    (lambda d: d["comparisons"]["comparisons"][0]["counts"].update(both_correct=1),
     "add up"),
    (lambda d: d["conv1d_autoencoder_int8_cpu"]["dataset"].update(name="another series"),
     "different datasets"),
]


@pytest.mark.parametrize("change,message", TAMPERING)
def test_validator_finds_tampering(published, tmp_path, change, message):
    docs = json.loads(json.dumps(published))
    change(docs)
    for name, doc in docs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
    issues = [i for v in va.validate_directory(sorted(tmp_path.glob("*.json"))).values()
              for i in v]
    assert any(message in i for i in issues), issues


def test_validate_results_skips_anomaly_documents(published, tmp_path):
    (tmp_path / "x.json").write_text(json.dumps(published[RUNS[0]]), encoding="utf-8")
    out = subprocess.run([sys.executable, str(SCRIPTS / "validate_results.py"), "--strict",
                          str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "1 skipped" in out.stdout


# --- recomputed from the score files (NumPy, local files) -----------------------------


@pytest.fixture(scope="module")
def scores(published):
    np = pytest.importorskip("numpy")
    from common import timeseries as ts

    if not ts.nab_present():
        pytest.skip("NAB data missing; run scripts/get_data.py --dataset nab")
    out = {}
    for run_id in RUNS:
        r = published[run_id]
        arrays = {}
        for split, ref in r["scores"].items():
            path = reg.REPO_ROOT / ref["file"]
            if not path.is_file():
                pytest.skip(f"{path} missing; run scripts/eval_anomaly.py")
            arrays[split] = np.load(path)
        out[run_id] = arrays
    return out, ts.load_split_windows()


def test_metrics_recompute_from_the_scores(published, scores):
    import eval_anomaly as ea

    arrays, data = scores
    for run_id, a in arrays.items():
        r = published[run_id]
        assert ea.sha256_file(reg.REPO_ROOT / r["scores"]["test"]["file"]) == \
            r["scores"]["test"]["sha256"]
        threshold, metrics = ea.detection_metrics(a["validation"], a["test"], data)
        assert threshold == r["threshold"], run_id
        assert threshold["value"] == float(a["validation"].max())
        assert metrics == r["metrics"], run_id


def test_threshold_never_looks_at_the_test_split(published, scores):
    import eval_anomaly as ea

    arrays, data = scores
    a = arrays["conv1d_autoencoder_fp32_cpu"]
    shuffled = a["test"][::-1].copy()  # any change to the test scores leaves the threshold alone
    assert ea.detection_metrics(a["validation"], shuffled, data)[0] == \
        published["conv1d_autoencoder_fp32_cpu"]["threshold"]


def test_scores_have_the_split_lengths(published, scores):
    arrays, data = scores
    for a in arrays.values():
        assert len(a["validation"]) == len(data.starts["validation"])
        assert len(a["test"]) == len(data.starts["test"])
        assert (a["validation"] >= 0).all() and (a["test"] >= 0).all()
