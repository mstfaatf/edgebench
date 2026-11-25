"""Evaluate the autoencoder as an anomaly detector on the NAB test split, on the CPU and the NPU.

Score: the mean squared reconstruction error of each window (float64, from the
model's float32 output), on windows of the normalized series exactly as the
split manifest defines them (``common/timeseries.py``, ``load_split_windows``).

Threshold: chosen on the validation split only, never on the test split. Rule
``max_validation_score``: the largest score of any validation window. The
validation split is anomaly-free by construction, so this is the lowest
threshold that flags none of the held-out normal data; it needs no percentile
choice, and with stride-1 windows (2,676 heavily overlapping validation
windows from 2,803 points) a high percentile would sit next to the maximum
anyway. A test window is flagged when its score is strictly greater.

Test-split metrics (``common/anomaly_stats.py``): window-level ROC-AUC and
PR-AUC (average precision), precision / recall / F1 at the threshold (a window
is positive when any of its points lies in a labelled anomaly window), and
event-level detection: for each of the 4 labelled anomaly ranges, whether any
overlapping window is flagged and how far into the range the first flag comes.

Runs (pruned files only), all through the same scoring code:

- ``conv1d_autoencoder_fp32_cpu``, ``conv1d_autoencoder_int8_cpu``: CPU EP, 4 intra-op threads;
- ``conv1d_autoencoder_int8_npu``: QNN HTP, strict (CPU fallback disabled);
- ``conv1d_autoencoder_fp32_npu_fp16``: QNN HTP, strict, FP16 compute
  (``enable_htp_fp16_precision=1``) on the FP32 file.

Batch size: per run, batch 1, 4 and 8 are run on every validation window on
that device; the largest whose reconstructions equal batch 1's bit for bit is
used (the last batch of a split is padded with copies of its last window and
the padding outputs dropped).

Paired comparisons (``comparisons.json``): NPU INT8 vs CPU INT8, CPU INT8 vs
CPU FP32, NPU FP16 vs CPU FP32: Spearman correlation of the test scores, flag
agreement, and McNemar's exact test on whether each run's flag is correct.
Stride-1 windows overlap and are not independent, so the McNemar p-values are
optimistic; they are reported as a screening statistic.

Output: one result per run (``<run_id>.json``, schema
``schemas/anomaly-result.v1.0.0.schema.json``) and ``comparisons.json`` in
``--output-dir`` (default ``benchmarks/results/published/accuracy/anomaly/``);
per-window scores as ``.npy`` (float64) in ``--scores-dir`` (default
``benchmarks/results/local/accuracy/anomaly_scores/``, not committed), with
their SHA256 in the results. Reruns give the same numbers: every output used is
bit-identical between runs on both devices.

Usage:
    python scripts/eval_anomaly.py [--only RUN_ID ...] [--output-dir DIR] [--scores-dir DIR]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import anomaly_stats as st  # noqa: E402
from common.accuracy_stats import TEST_METHOD, mcnemar_exact, paired_counts  # noqa: E402
from common.model_registry import MODELS_DIR, REPO_ROOT, read_manifest, relative_path  # noqa: E402

ANOMALY_SCHEMA_VERSION = "1.0.0"
MODEL = "conv1d_autoencoder"
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "published" / "accuracy" / "anomaly"
SCORES_DIR = REPO_ROOT / "benchmarks" / "results" / "local" / "accuracy" / "anomaly_scores"
CPU_THREADS = 4
BATCHES = (1, 4, 8)
SAMPLING_MINUTES = 5
QNN_BASE = {"htp_performance_mode": "burst"}  # plus backend_path, set at run time
FP16 = {"enable_htp_fp16_precision": "1"}
SCORE_DEFINITION = ("mean squared reconstruction error over the window's points, in float64 "
                    "from the model's float32 output")


@dataclass(frozen=True)
class Run:
    run_id: str
    variant: str
    device: str
    compute_precision: str
    provider_options: dict = field(default_factory=dict)


def planned_runs() -> list[Run]:
    return [
        Run(f"{MODEL}_fp32_cpu", "fp32", "cpu", "fp32"),
        Run(f"{MODEL}_int8_cpu", "int8", "cpu", "int8"),
        Run(f"{MODEL}_int8_npu", "int8", "npu", "int8", dict(QNN_BASE)),
        Run(f"{MODEL}_fp32_npu_fp16", "fp32", "npu", "fp16", {**QNN_BASE, **FP16}),
    ]


def comparison_plan() -> list[tuple[str, str, str, str]]:
    return [
        ("NPU INT8 vs CPU INT8", "Does the NPU detect the same anomalies as the CPU on the same "
         "INT8 file?", f"{MODEL}_int8_npu", f"{MODEL}_int8_cpu"),
        ("CPU INT8 vs CPU FP32", "What does INT8 quantization cost the detector?",
         f"{MODEL}_int8_cpu", f"{MODEL}_fp32_cpu"),
        ("NPU FP16 vs CPU FP32", "What does FP16 compute on the NPU cost against FP32 on the "
         "CPU (same FP32 file)?", f"{MODEL}_fp32_npu_fp16", f"{MODEL}_fp32_cpu"),
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sessions:
    """CPU sessions, and strict NPU sessions (the QNN plugin is registered on first use)."""

    def __init__(self) -> None:
        self.npu = None
        self.backend = None

    def create(self, run: Run, path: Path):
        import onnxruntime as ort

        if run.device == "cpu":
            options = ort.SessionOptions()
            options.intra_op_num_threads = CPU_THREADS
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        from common import ort_qnn

        if self.npu is None:
            ort_qnn.register_qnn()
            self.npu = ort_qnn.select_qnn_npu(ort_qnn.list_devices())
            self.backend = str(ort_qnn.htp_backend_path())
        options = ort_qnn.qnn_npu_session_options(
            self.npu, {"backend_path": self.backend, **run.provider_options}, strict=True)
        return ort.InferenceSession(str(path), sess_options=options)


def reconstruct(session, windows, batch: int):
    """Reconstructions ``[n, length]``; the last batch is padded with its last window."""
    import numpy as np

    n = len(windows)
    pad = (-n) % batch
    x = windows[:, :, None]
    if pad:
        x = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)])
    name = session.get_inputs()[0].name
    return np.concatenate([session.run(None, {name: x[i:i + batch]})[0]
                           for i in range(0, len(x), batch)])[:n, :, 0]


def window_scores(windows, reconstructions):
    import numpy as np

    return ((reconstructions.astype(np.float64) - windows.astype(np.float64)) ** 2).mean(axis=1)


def environment() -> dict:
    import numpy as np
    import onnxruntime as ort

    try:
        from common.ort_qnn import qnn_versions

        qnn = qnn_versions()
    except ImportError:
        qnn = {"onnxruntime_qnn": None, "qnn_backend": None}
    return {"python": platform.python_version(), "numpy": np.__version__,
            "onnxruntime": ort.__version__, **qnn, "os": platform.platform(),
            "machine": platform.machine()}


def test_ranges(data) -> list[tuple[int, int]]:
    """Labelled anomaly ranges inside the test split, in series order."""
    segments = [tuple(s) for s in data.manifest["splits"]["test"]["segments"]]
    return [(w["start"], w["stop"]) for w in data.manifest["anomaly_windows"]
            if any(a <= w["start"] and w["stop"] <= b for a, b in segments)]


def dataset_block(data, manifest_sha: str) -> dict:
    m = data.manifest
    return {
        "name": "NAB machine_temperature_system_failure",
        "split_manifest": "datasets/nab-machine-temperature/splits.json",
        "split_manifest_sha256": manifest_sha,
        "window_length": m["windowing"]["length"],
        "window_stride": m["windowing"]["stride"],
        "window_label": m["windowing"]["window_is_anomalous"],
        "validation_windows": len(data.starts["validation"]),
        "test_windows": len(data.starts["test"]),
        "test_anomalous_windows": int(data.labels["test"].sum()),
        "anomaly_ranges": [list(r) for r in test_ranges(data)],
        "sampling_interval_minutes": SAMPLING_MINUTES,
    }


def detection_metrics(val_scores, test_scores, data) -> tuple[dict, dict]:
    """(threshold block, metrics block) from the score arrays; standard library math."""
    val, test = val_scores.tolist(), test_scores.tolist()
    labels = [bool(v) for v in data.labels["test"].tolist()]
    threshold = max(val)
    c = st.confusion(test, labels, threshold)
    prf = st.precision_recall_f1(c)
    flags = [s > threshold for s in test]
    window = data.manifest["windowing"]["length"]
    ev = st.events(data.starts["test"], flags, test_ranges(data), window)
    for e in ev:
        e["delay_minutes"] = None if e["delay_points"] is None else \
            e["delay_points"] * SAMPLING_MINUTES
    threshold_block = {"rule": st.THRESHOLD_RULE, "flag_rule": st.FLAG_RULE,
                       "split": "validation", "value": threshold,
                       "validation_windows": len(val),
                       "validation_flagged": sum(1 for s in val if s > threshold)}
    metrics = {
        "test_windows": len(test), "positives": sum(labels),
        "roc_auc": st.roc_auc(test, labels), "roc_auc_method": st.ROC_METHOD,
        "pr_auc": st.average_precision(test, labels), "pr_auc_method": st.PR_METHOD,
        "at_threshold": {**c, **prf,
                         "false_positive_rate": c["fp"] / (c["fp"] + c["tn"])},
        "events": {"total": len(ev), "detected": sum(e["detected"] for e in ev), "ranges": ev},
    }
    return threshold_block, metrics


def evaluate(run: Run, sessions: Sessions, data, manifest_sha: str, scores_dir: Path,
             env: dict) -> dict:
    import numpy as np

    from common import model_recipe as mr

    rel1 = relative_path(MODEL, run.variant, 1)
    recipe = mr.build_recipe(mr.read_model_file(MODELS_DIR / rel1, rel1))
    result = {
        "anomaly_schema_version": ANOMALY_SCHEMA_VERSION, "kind": "run", "run_id": run.run_id,
        "timestamp_utc": utc_now(), "status": "success", "error": None, "model": None,
        "execution": {
            "device": run.device,
            "execution_provider": "CPUExecutionProvider" if run.device == "cpu"
            else "QNNExecutionProvider",
            "strict": run.device == "npu", "compute_precision": run.compute_precision,
            "intra_op_threads": CPU_THREADS if run.device == "cpu" else None,
            "provider_options": {} if run.device == "cpu" else
            {"backend_path": "<onnxruntime_qnn>/QnnHtp.dll", **run.provider_options},
            "session_config": {"session.disable_cpu_ep_fallback": "1"} if run.device == "npu"
            else {},
            "batch_equivalence": None},
        "dataset": dataset_block(data, manifest_sha),
        "input": {"normalization": recipe["preprocessing"]["normalization"],
                  "mean": recipe["preprocessing"]["mean"], "std": recipe["preprocessing"]["std"],
                  "input_range": recipe["preprocessing"]["input_range"]},
        "score": {"definition": SCORE_DEFINITION, "higher_is_more_anomalous": True},
        "threshold": None, "metrics": None, "scores": None, "environment": env}

    def model_block(batch: int) -> dict:
        rel = relative_path(MODEL, run.variant, batch)
        entry = next(f for f in read_manifest()[0] if f.file == rel)
        if sha256_file(MODELS_DIR / rel) != entry.sha256:
            raise RuntimeError(f"{rel} does not match models/manifest.json")
        return {"file": rel, "sha256": entry.sha256, "model": MODEL, "variant": run.variant,
                "stored_precision": entry.precision, "normalization": entry.normalization,
                "batch_size": batch}

    val = data.windows["validation"]
    reference, candidates, outputs = None, {}, {}
    try:
        for batch in BATCHES:
            path = MODELS_DIR / relative_path(MODEL, run.variant, batch)
            try:
                session = sessions.create(run, path)
            except Exception:  # noqa: BLE001 - recorded
                if batch == 1:
                    raise
                candidates[f"b{batch}"] = "session_refused"
                continue
            outputs[batch] = reconstruct(session, val, batch)
            del session
            gc.collect()
            if batch == 1:
                reference = outputs[1]
            same = bool(np.array_equal(outputs[batch], reference))
            candidates[f"b{batch}"] = "identical" if same else "differs"
            print(f"  b{batch}: validation reconstructions vs b1 "
                  f"{'identical' if same else 'DIFFERENT'}")
    except Exception as exc:  # noqa: BLE001 - a refused strict session is a result
        result.update(status="refused", model=model_block(1),
                      error={"stage": "session_creation", "message": str(exc).strip()[:2000]})
        print(f"  REFUSED: {str(exc).splitlines()[0][:200]}")
        return result
    batch = max(b for b in outputs if candidates[f"b{b}"] == "identical")
    result["model"] = model_block(batch)
    result["execution"]["batch_equivalence"] = {
        "split": "validation", "reference_batch": 1, "criterion": "reconstructions bit-identical",
        "identical": True, "windows": int(len(val)), "candidates": candidates}

    session = sessions.create(run, MODELS_DIR / result["model"]["file"])
    test = data.windows["test"]
    val_recon = reconstruct(session, val, batch)
    test_recon = reconstruct(session, test, batch)
    repeat = reconstruct(session, test, batch)
    del session
    gc.collect()
    if not np.array_equal(test_recon, repeat) or not np.array_equal(val_recon, outputs[batch]):
        raise RuntimeError(f"{run.run_id}: outputs differ between identical runs")
    val_scores, test_scores = window_scores(val, val_recon), window_scores(test, test_recon)
    result["threshold"], result["metrics"] = detection_metrics(val_scores, test_scores, data)

    scores_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, array in (("validation", val_scores), ("test", test_scores)):
        path = scores_dir / f"{run.run_id}.{split}_scores.npy"
        np.save(path, array, allow_pickle=False)
        files[split] = {"file": repo_relative(path), "sha256": sha256_file(path)}
    result["scores"] = files
    m = result["metrics"]
    print(f"  batch {batch}: threshold {result['threshold']['value']:.4g}, ROC-AUC "
          f"{m['roc_auc']:.4f}, PR-AUC {m['pr_auc']:.4f}, F1 {m['at_threshold']['f1']:.4f}, "
          f"events {m['events']['detected']}/{m['events']['total']}")
    return result


def write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_scores(ref: dict):
    import numpy as np

    path = REPO_ROOT / ref["file"] if not Path(ref["file"]).is_absolute() else Path(ref["file"])
    if sha256_file(path) != ref["sha256"]:
        raise RuntimeError(f"{path} does not match its result's SHA256")
    return np.load(path)


def comparisons(output_dir: Path, data, manifest_sha: str) -> dict | None:
    """Paired comparisons of the run results in ``output_dir``; None if any run is missing."""
    results = {}
    for run in planned_runs():
        path = output_dir / f"{run.run_id}.json"
        if not path.is_file():
            return None
        results[run.run_id] = json.loads(path.read_text(encoding="utf-8"))
    labels = [bool(v) for v in data.labels["test"].tolist()]
    out = []
    for name, question, a, b in comparison_plan():
        ra, rb = results[a], results[b]
        if ra["status"] != "success" or rb["status"] != "success":
            continue
        sa = load_scores(ra["scores"]["test"]).tolist()
        sb = load_scores(rb["scores"]["test"]).tolist()
        fa = [s > ra["threshold"]["value"] for s in sa]
        fb = [s > rb["threshold"]["value"] for s in sb]
        counts = paired_counts([f == v for f, v in zip(fa, labels)],
                               [f == v for f, v in zip(fb, labels)])
        ma, mb = ra["metrics"], rb["metrics"]
        out.append({
            "name": name, "question": question, "a": a, "b": b,
            "score_spearman": st.spearman(sa, sb),
            "flag_agreement": sum(x == y for x, y in zip(fa, fb)) / len(fa),
            "flagged_a": sum(fa), "flagged_b": sum(fb),
            "flagged_only_a": sum(x and not y for x, y in zip(fa, fb)),
            "flagged_only_b": sum(y and not x for x, y in zip(fa, fb)),
            "counts": counts,
            "p_value": mcnemar_exact(counts["only_a_correct"], counts["only_b_correct"]),
            "f1_a": ma["at_threshold"]["f1"], "f1_b": mb["at_threshold"]["f1"],
            "f1_delta": ma["at_threshold"]["f1"] - mb["at_threshold"]["f1"],
            "roc_auc_a": ma["roc_auc"], "roc_auc_b": mb["roc_auc"],
            "events_detected_a": ma["events"]["detected"],
            "events_detected_b": mb["events"]["detected"],
        })
    return {"anomaly_schema_version": ANOMALY_SCHEMA_VERSION, "kind": "comparisons",
            "timestamp_utc": utc_now(), "dataset": dataset_block(data, manifest_sha),
            "test": TEST_METHOD, "correlation": st.CORRELATION_METHOD,
            "independence_note": "Test windows overlap (stride 1), so neighbouring windows are "
                                 "not independent and the p-values are optimistic.",
            "runs": {k: {"status": r["status"],
                         "test_scores_sha256": r["scores"]["test"]["sha256"]
                         if r["scores"] else None} for k, r in sorted(results.items())},
            "comparisons": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    runs = planned_runs()
    parser.add_argument("--only", nargs="+", choices=[r.run_id for r in runs], metavar="RUN_ID",
                        help="only these runs (default: all)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR)
    args = parser.parse_args(argv)

    from common.platform_guard import NonNativeInterpreterError, require_native_arm64

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    from common.timeseries import SPLIT_MANIFEST, load_split_windows

    data = load_split_windows()
    manifest_sha = sha256_file(SPLIT_MANIFEST)
    env = environment()
    sessions = Sessions()
    for run in (r for r in runs if args.only is None or r.run_id in args.only):
        print(f"[{run.run_id}]")
        write_json(args.output_dir / f"{run.run_id}.json",
                   evaluate(run, sessions, data, manifest_sha, args.scores_dir, env))
    summary = comparisons(args.output_dir, data, manifest_sha)
    if summary is not None:
        write_json(args.output_dir / "comparisons.json", summary)
        for c in summary["comparisons"]:
            print(f"{c['name']}: Spearman {c['score_spearman']:.4f}, flag agreement "
                  f"{c['flag_agreement']:.4f}, McNemar p {c['p_value']:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
