"""The CPU/NPU matrix definition (common/matrix.py) and its check (check_matrix.py).

The definition and the batch-equivalence expectations run anywhere (CI). The
checks of a real matrix run only where one exists under
benchmarks/results/local/matrix/ (skipped otherwise); they also show that the
check notices a tampered result.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import check_matrix as cm
from common import matrix as mx
from common import model_registry as reg

MATRIX_ROOT = reg.REPO_ROOT / "benchmarks" / "results" / "local" / "matrix"


def test_the_matrix_has_every_model_batch_and_configuration():
    runs = mx.planned_runs()
    assert len(runs) == 57
    assert len({r.run_id for r in runs}) == 57
    core = [r for r in runs if r.config in ("cpu_fp32", "cpu_int8", "npu_int8", "npu_fp16")]
    assert len(core) == 48
    assert {(r.model, r.batch) for r in core} == {
        (m, b) for m in mx.IMAGE_MODELS + mx.TIME_SERIES_MODELS for b in (1, 4, 8)}
    # The image models use external normalization, the pruned INT8 files and the
    # autoencoder's cleaned graph.
    assert {r.variant for r in core} == {"fp32", "int8"}
    mode1 = [r for r in runs if r.config == "npu_int8_finalization1"]
    assert [(r.model, r.batch) for r in mode1] == [(m, 8) for m in mx.IMAGE_MODELS]
    assert all(dict(r.ep_compile_options) == {mx.FINALIZATION_OPTION: "1"} for r in mode1)
    folded = [r for r in runs if "folded" in r.config]
    assert sorted((r.model, r.provider) for r in folded) == sorted(
        (m, p) for m in mx.IMAGE_MODELS for p in ("cpu", "npu"))
    assert all(r.batch == 1 and r.variant == "int8_folded" for r in folded)
    for r in runs:
        assert (r.compute_precision == "fp16") == (r.config == "npu_fp16")
        assert (reg.MODELS_DIR / r.file).name == Path(r.file).name


def test_run_arguments():
    runs = {r.run_id: r for r in mx.planned_runs()}
    settings = dict(trials=3, iterations=100, warmup=20, cooldown_ms=1000, cache_dir="C:/c")
    cpu = mx.run_arguments(runs["resnet18_b4_cpu_int8"], **settings)
    assert cpu[:2] == ["--provider", "cpu"] and "--ep-cache" not in cpu
    assert "--init-breakdown" not in cpu  # an NPU-only option
    npu = mx.run_arguments(runs["mobilenetv2_b8_npu_int8_finalization1"], **settings)
    for flag in ("--init-breakdown", "--profile-accelerator", "--ep-cache"):
        assert flag in npu
    assert npu[npu.index("--ep-cache") + 1] == "use"
    assert npu[npu.index("--htp-graph-finalization-mode") + 1] == "1"
    fp16 = mx.run_arguments(runs["efficientnet_lite0_b1_npu_fp16"], **settings)
    assert "--npu-fp16" in fp16
    build = mx.cache_build_arguments(runs["mobilenetv2_b8_npu_int8_finalization1"], "C:/c")
    assert build[build.index("--ep-cache") + 1] == "build"
    assert "--htp-graph-finalization-mode" in build  # the same key as the timed run


def test_expected_batch_equivalence_follows_the_compile_options():
    summary = json.loads(cm.BATCH_EQUIVALENCE_JSON.read_text(encoding="utf-8"))
    manifest = {f.file: f for f in reg.read_manifest()[0]}
    sha = manifest["mobilenetv2/mobilenetv2_int8_b8.onnx"].sha256
    default = cm.expected_equivalence(summary, sha, "npu", "int8", {}, 8)
    assert default["predictions_equal_to_b1"] is False
    mode1 = cm.expected_equivalence(summary, sha, "npu", "int8", {mx.FINALIZATION_OPTION: "1"}, 8)
    assert mode1["bitwise_equal_to_b1"] is True and mode1["predictions_basis"] == "bitwise"
    assert cm.expected_equivalence(summary, sha, "npu", "int8", {mx.FINALIZATION_OPTION: "2"},
                                   8) is None
    assert cm.expected_equivalence(summary, sha, "npu", "int8", {}, 1)["bitwise_basis"] == \
        "batch_size_1"
    assert cm.compile_options({"backend_path": "x", "htp_performance_mode": "burst",
                               "enable_htp_fp16_precision": "1"}) == {}


TAMPERED_RUN = "mobilenetv2_b8_npu_int8"


def latest_matrix(with_run: str | None = None) -> Path | None:
    """The newest local matrix directory (with `with_run` among its runs, if given)."""
    dirs = sorted(d for d in MATRIX_ROOT.glob("*") if (d / "plan.json").is_file()) \
        if MATRIX_ROOT.is_dir() else []
    for d in reversed(dirs):
        runs = json.loads((d / "plan.json").read_text(encoding="utf-8"))["runs"]
        if with_run is None or any(r["run_id"] == with_run for r in runs):
            return d
    return None


@pytest.fixture(scope="module")
def matrix_dir():
    d = latest_matrix()
    if d is None:
        pytest.skip("no local matrix run (scripts/run_matrix.py)")
    return d


# Warnings about the machine, not the engine: the check reports them (a run measured
# that way is not clean), but they say nothing about whether the engine is right.
SYSTEM_STATE = ("unexpected warning [background_cpu_load_high]",
                "unexpected warning [trial_spread_high]")


def test_a_real_matrix_has_no_engine_problems(matrix_dir):
    plan = json.loads((matrix_dir / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads(cm.BATCH_EQUIVALENCE_JSON.read_text(encoding="utf-8"))
    manifest = {f.file: f for f in reg.read_manifest()[0]}
    problems = {}
    for run in plan["runs"]:
        found, _ = cm.check_run(run, matrix_dir, plan, summary, manifest)
        engine = [p for p in found if not p.startswith(SYSTEM_STATE)]
        if engine:
            problems[run["run_id"]] = engine
    assert problems == {}


@pytest.mark.parametrize("change,message", [
    (lambda d: d["config"]["ep_context_cache"].update(hit=False), "hit False"),
    (lambda d: d["runtime"]["qnn"].update(partition_count=2), "2 QNN partitions"),
    (lambda d: d["warnings"].append({"code": "trial_spread_high", "message": "m",
                                     "path": "/timing/aggregate/trial_spread"}),
     "unexpected warning [trial_spread_high]"),
    (lambda d: d["execution"]["effective_precision"].update(compute="fp16"), "effective precision"),
    (lambda d: d["execution"]["verification_details"]["module_checks"][0].update(
        modules_outside_bundle=["C:/x/QnnHtp.dll"]), "modules outside the bundle"),
])
def test_the_check_notices_a_changed_npu_result(tmp_path, change, message):
    matrix_dir = latest_matrix(with_run=TAMPERED_RUN)
    if matrix_dir is None:
        pytest.skip(f"no local matrix run with {TAMPERED_RUN} (scripts/run_matrix.py)")
    plan = json.loads((matrix_dir / "plan.json").read_text(encoding="utf-8"))
    run = next(r for r in plan["runs"] if r["run_id"] == TAMPERED_RUN)
    copy = tmp_path / "m"
    for rel in (run["result"], run["cache_build"]["result"]):
        (copy / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(matrix_dir / rel, copy / rel)
    doc = json.loads((copy / run["result"]).read_text(encoding="utf-8"))
    change(doc)
    (copy / run["result"]).write_text(json.dumps(doc), encoding="utf-8")
    summary = json.loads(cm.BATCH_EQUIVALENCE_JSON.read_text(encoding="utf-8"))
    manifest = {f.file: f for f in reg.read_manifest()[0]}
    problems, _ = cm.check_run(run, copy, plan, summary, manifest)
    assert any(message in p for p in problems), problems


def test_a_suite_manifest_reads_as_a_plan():
    manifest = json.loads((reg.REPO_ROOT / "docs" / "examples" / "suite" / "example-suite-manifest.json")
                          .read_text(encoding="utf-8"))
    plan = cm.plan_from_suite(manifest)
    assert len(plan["runs"]) == len(manifest["runs"])
    for run, record in zip(plan["runs"], manifest["runs"]):
        c = record["configuration"]
        assert run["settings"] == {"trials": c["trials"], "iterations": c["iterations"],
                                   "warmup": c["warmup"]}
        assert run["config"] == f"{c['provider']}_{c['compute_precision']}"
        assert run["result"] == record["result_file"]
        assert run["cached"] is (c["init"] == "cached")
    assert plan["doctor"]["checks"]  # from the last pre-flight attempt
