"""Measure the accuracy of every model variant on ImageNetV2 eval-full, on the CPU and the NPU.

Runs (pruned files only; the unpruned references give bit-identical outputs):

- CPU EP, 4 intra-op threads: FP32 external, FP32 folded, INT8 external and
  INT8 folded for every model;
- QNN HTP (NPU), strict mode (CPU fallback disabled): INT8 external and INT8
  folded;
- QNN HTP, strict mode, FP16 compute (``enable_htp_fp16_precision=1``) on the
  FP32 external file. A strict session the EP refuses is recorded as a
  ``refused`` result;
- QNN HTP, strict mode, INT8 external and INT8 folded at batch 8 (``*_npu_b8``):
  the batch size the default graph finalization computes differently from
  batch 1 (docs/npu.md, "Batch size and INT8 outputs"), measured as it is.

Batch size: for each run, batch 1, 4 and 8 are tried on eval-quick (1,000
images) on that device; the largest one whose top-5 predictions equal batch 1's
on every image is used for eval-full (batch_rule ``largest_identical``: larger
batches have less per-call overhead, and a fixed rule keeps reruns identical;
the eval-quick times are printed, not recorded). On the NPU, INT8 files at
batch 8 compute different outputs from batch 1, so INT8 NPU runs use batch 4.
The ``*_npu_b8`` runs are asked for batch 8 (batch_rule ``fixed``) and record how
many eval-quick images differ from batch 1 there.

Images: eval-full (10,000 images) in manifest order, labels checked against
their folders and the class list, preprocessed with the shared helper through
a uint8 image cache verified bit-identical to it (``common/imagenet_eval.py``);
each file gets the input its recipe expects (normalized with the model's own
mean/std, or [0, 1] for folded files).

Output: one result per run (``<run_id>.json``, schema
``schemas/accuracy-result.v1.1.0.schema.json``) and ``comparisons.json`` (paired
comparisons with McNemar's exact test) in ``--output-dir`` (default
``benchmarks/results/published/accuracy/``); per-image top-5 predictions as
``.npy`` files in ``--predictions-dir`` (default
``benchmarks/results/local/accuracy/predictions/``, not committed), with their
SHA256 in the results.

Usage:
    python scripts/eval_accuracy.py [--only RUN_ID ...] [--output-dir DIR]
                                    [--predictions-dir DIR]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.accuracy_stats import (  # noqa: E402
    CI_METHOD,
    TEST_METHOD,
    mcnemar_exact,
    paired_counts,
    wilson_interval,
)
from common.model_registry import MODELS, MODELS_DIR, REPO_ROOT, read_manifest, relative_path  # noqa: E402

ACCURACY_SCHEMA_VERSION = "1.1.0"
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "published" / "accuracy"
PREDICTIONS_DIR = REPO_ROOT / "benchmarks" / "results" / "local" / "accuracy" / "predictions"
CPU_THREADS = 4
BATCHES = (1, 4, 8)
QNN_BASE = {"htp_performance_mode": "burst"}  # plus backend_path, set at run time
FP16 = {"enable_htp_fp16_precision": "1"}


@dataclass(frozen=True)
class Run:
    run_id: str
    model: str
    variant: str
    device: str                 # "cpu" or "npu"
    compute_precision: str      # "fp32", "int8" or "fp16"
    provider_options: dict = field(default_factory=dict)
    batch: int | None = None    # a fixed batch size; None: the largest identical to batch 1


def planned_runs() -> list[Run]:
    runs = []
    for m in sorted(MODELS):
        for v in ("fp32", "fp32_folded", "int8", "int8_folded"):
            runs.append(Run(f"{m}_{v}_cpu", m, v, "cpu", "int8" if v.startswith("int8") else "fp32"))
        for v in ("int8", "int8_folded"):
            runs.append(Run(f"{m}_{v}_npu", m, v, "npu", "int8", dict(QNN_BASE)))
        runs.append(Run(f"{m}_fp32_npu_fp16", m, "fp32", "npu", "fp16", {**QNN_BASE, **FP16}))
        for v in ("int8", "int8_folded"):
            runs.append(Run(f"{m}_{v}_npu_b8", m, v, "npu", "int8", dict(QNN_BASE), batch=8))
    return runs


def comparison_plan() -> list[tuple[str, str, str, str]]:
    """(name, question, a, b) for every paired comparison."""
    plan = []
    for m in sorted(MODELS):
        plan += [
            (f"{m}: NPU INT8 vs CPU INT8", "Does the NPU get the same accuracy as the CPU on the "
             "same INT8 file (external normalization)?", f"{m}_int8_npu", f"{m}_int8_cpu"),
            (f"{m}: NPU INT8 folded vs CPU INT8 folded", "Does the NPU get the same accuracy as the "
             "CPU on the same INT8 file (normalization folded)?", f"{m}_int8_folded_npu",
             f"{m}_int8_folded_cpu"),
            (f"{m}: CPU INT8 vs CPU FP32", "What does INT8 quantization cost (external "
             "normalization)?", f"{m}_int8_cpu", f"{m}_fp32_cpu"),
            (f"{m}: CPU INT8 folded vs CPU FP32 folded", "What does INT8 quantization cost "
             "(normalization folded)?", f"{m}_int8_folded_cpu", f"{m}_fp32_folded_cpu"),
            (f"{m}: CPU FP32 folded vs CPU FP32", "Does folding the normalization change FP32 "
             "accuracy?", f"{m}_fp32_folded_cpu", f"{m}_fp32_cpu"),
            (f"{m}: CPU INT8 folded vs CPU INT8", "Does folding the normalization change INT8 "
             "accuracy?", f"{m}_int8_folded_cpu", f"{m}_int8_cpu"),
            (f"{m}: NPU FP16 vs CPU FP32", "What does FP16 compute on the NPU cost against FP32 "
             "on the CPU (same FP32 file)?", f"{m}_fp32_npu_fp16", f"{m}_fp32_cpu"),
            (f"{m}: NPU INT8 b8 vs NPU INT8 b4", "Does batch 8, which the NPU computes differently "
             "from batch 1, change accuracy against batch 4, which it computes exactly like batch "
             "1 (same INT8 variant, external normalization)?", f"{m}_int8_npu_b8",
             f"{m}_int8_npu"),
            (f"{m}: NPU INT8 folded b8 vs NPU INT8 folded b4", "Does batch 8 change accuracy "
             "against batch 4 on the NPU (same INT8 variant, normalization folded)?",
             f"{m}_int8_folded_npu_b8", f"{m}_int8_folded_npu"),
        ]
    return plan


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proportion(correct: int, total: int) -> dict:
    low, high = wilson_interval(correct, total)
    return {"correct": correct, "total": total, "accuracy": correct / total,
            "ci95": {"low": low, "high": high, "method": CI_METHOD}}


class Sessions:
    """Creates CPU and strict NPU sessions; the QNN plugin is registered on first use."""

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


def choose_batch(run: Run, sessions: Sessions, quick, quick_cache, recipe) -> tuple[int, dict]:
    """The largest batch size whose eval-quick top-5 predictions equal batch 1's, or the
    run's fixed batch size (compared with batch 1 all the same)."""
    import numpy as np

    from common import imagenet_eval as ie

    pre = recipe["preprocessing"]
    reference, timings, candidates, differing = None, {}, {}, {}
    for batch in BATCHES if run.batch is None else sorted({1, run.batch}):
        path = MODELS_DIR / relative_path(run.model, run.variant, batch)
        try:
            session = sessions.create(run, path)
        except Exception as exc:  # noqa: BLE001 - recorded
            if batch == 1:
                raise
            candidates[f"b{batch}"] = "session_refused"
            print(f"  b{batch}: session refused ({str(exc).splitlines()[0][:120]})")
            continue
        start = time.perf_counter()
        top5, _ = ie.predict(session, quick_cache, batch, pre["normalization"], pre["mean"],
                             pre["std"])
        timings[batch] = time.perf_counter() - start
        del session
        gc.collect()
        if batch == 1:
            reference = top5
        differing[batch] = int(np.count_nonzero(np.any(top5 != reference, axis=1)))
        identical = differing[batch] == 0
        candidates[f"b{batch}"] = "identical" if identical else "differs"
        print(f"  b{batch}: eval-quick {timings[batch]:.1f} s, top-5 vs b1 "
              f"{'identical' if identical else 'DIFFERENT'}")
    # The largest equivalent batch size: the least per-call overhead, and (unlike the
    # measured times, which differ by noise) the same choice on every run.
    if run.batch is not None:
        if run.batch not in timings:
            raise RuntimeError(f"batch {run.batch}: the strict session was refused")
        best, rule = run.batch, "fixed"
    else:
        best, rule = max(b for b in timings if candidates[f"b{b}"] == "identical"), \
            "largest_identical"
    return best, {"split": "eval-quick", "reference_batch": 1, "batch_rule": rule,
                  "identical_top5": differing[best] == 0, "top5_differing": differing[best],
                  "images": len(quick.entries), "candidates": candidates}


def evaluate(run: Run, sessions: Sessions, full, cache, quick, quick_cache, cache_check: dict,
             predictions_dir: Path, env: dict) -> dict:
    import numpy as np

    from common import imagenet_eval as ie
    from common import model_recipe as mr

    rel1 = relative_path(run.model, run.variant, 1)
    recipe = mr.build_recipe(mr.read_model_file(MODELS_DIR / rel1, rel1))
    pre = recipe["preprocessing"]
    result = {
        "accuracy_schema_version": ACCURACY_SCHEMA_VERSION, "kind": "run", "run_id": run.run_id,
        "timestamp_utc": utc_now(), "status": "success", "error": None,
        "model": None, "execution": {
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
        "dataset": dataset_block(full),
        "preprocessing": {
            "resize_shorter_side": pre["resize_shorter_side"], "interpolation": pre["interpolation"],
            "center_crop": pre["center_crop"], "scale": pre["scale"],
            "mean": pre["mean"] if pre["normalization"] == "external" else None,
            "std": pre["std"] if pre["normalization"] == "external" else None,
            "input": pre["input_range"],
            "image_cache": {"used": True, "verified_sample": cache_check["sample"],
                            "bit_identical": cache_check["bit_identical"]}},
        "metrics": None, "predictions": None, "environment": env}

    def model_block(batch: int) -> dict:
        rel = relative_path(run.model, run.variant, batch)
        entry = next(f for f in read_manifest()[0] if f.file == rel)
        if sha256_file(MODELS_DIR / rel) != entry.sha256:
            raise RuntimeError(f"{rel} does not match models/manifest.json")
        return {"file": rel, "sha256": entry.sha256, "model": run.model, "variant": run.variant,
                "stored_precision": entry.precision, "normalization": entry.normalization,
                "batch_size": batch}

    try:
        batch, equivalence = choose_batch(run, sessions, quick, quick_cache, recipe)
    except Exception as exc:  # noqa: BLE001 - a refused strict session is a result
        result.update(status="refused", model=model_block(1),
                      error={"stage": "session_creation", "message": str(exc).strip()[:2000]})
        print(f"  REFUSED: {str(exc).splitlines()[0][:200]}")
        return result
    result["model"] = model_block(batch)
    result["execution"]["batch_equivalence"] = equivalence
    session = sessions.create(run, MODELS_DIR / result["model"]["file"])
    top5, logits = ie.predict(session, cache, batch, pre["normalization"], pre["mean"], pre["std"])
    del session
    gc.collect()
    top1_ok, top5_ok = ie.correct(top5, full.labels)
    result["metrics"] = {"top1": proportion(int(top1_ok.sum()), len(full.labels)),
                         "top5": proportion(int(top5_ok.sum()), len(full.labels))}
    predictions_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, array in (("top5_indices", top5), ("top5_logits", logits)):
        path = predictions_dir / f"{run.run_id}.{key}.npy"
        np.save(path, array, allow_pickle=False)
        files[key] = {"file": repo_relative(path), "sha256": sha256_file(path)}
    result["predictions"] = files
    print(f"  batch {batch}: top-1 {result['metrics']['top1']['accuracy']:.4f}, "
          f"top-5 {result['metrics']['top5']['accuracy']:.4f}")
    return result


def dataset_block(eval_set) -> dict:
    return {"name": "ImageNetV2 matched-frequency", "split": eval_set.name,
            "manifest": repo_relative(eval_set.manifest), "manifest_sha256": eval_set.manifest_sha256,
            "image_count": len(eval_set.entries),
            "label_source": "manifest label = integer folder name; class order of "
                            "scripts/common/imagenet_classes.json"}


def write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")


def comparisons(output_dir: Path, full) -> dict | None:
    """Paired comparisons of the run results in ``output_dir``; None if any run is missing."""
    import numpy as np

    results = {}
    for run in planned_runs():
        path = output_dir / f"{run.run_id}.json"
        if not path.is_file():
            return None
        results[run.run_id] = json.loads(path.read_text(encoding="utf-8"))

    def top1_correct(run_id: str):
        ref = results[run_id]["predictions"]["top5_indices"]
        path = REPO_ROOT / ref["file"] if not Path(ref["file"]).is_absolute() else Path(ref["file"])
        if sha256_file(path) != ref["sha256"]:
            raise RuntimeError(f"{path} does not match its result's SHA256")
        top1 = np.load(path)[:, 0]
        return top1, top1 == full.labels

    out = []
    for name, question, a, b in comparison_plan():
        if results[a]["status"] != "success" or results[b]["status"] != "success":
            continue
        pa, ca = top1_correct(a)
        pb, cb = top1_correct(b)
        counts = paired_counts(ca.tolist(), cb.tolist())
        out.append({"name": name, "question": question, "a": a, "b": b,
                    "top1_a": results[a]["metrics"]["top1"]["accuracy"],
                    "top1_b": results[b]["metrics"]["top1"]["accuracy"],
                    "top1_delta": results[a]["metrics"]["top1"]["accuracy"]
                    - results[b]["metrics"]["top1"]["accuracy"],
                    "top1_agreement": float(np.mean(pa == pb)), "counts": counts,
                    "p_value": mcnemar_exact(counts["only_a_correct"], counts["only_b_correct"])})
    return {"accuracy_schema_version": ACCURACY_SCHEMA_VERSION, "kind": "comparisons",
            "timestamp_utc": utc_now(), "dataset": dataset_block(full), "test": TEST_METHOD,
            "runs": {k: {"status": r["status"],
                         "top5_indices_sha256": r["predictions"]["top5_indices"]["sha256"]
                         if r["predictions"] else None} for k, r in sorted(results.items())},
            "comparisons": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    runs = planned_runs()
    parser.add_argument("--only", nargs="+", choices=[r.run_id for r in runs], metavar="RUN_ID",
                        help="only these runs (default: all)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args(argv)

    from common.platform_guard import NonNativeInterpreterError, require_native_arm64

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    import numpy as np

    from common import imagenet_eval as ie
    from common.manifests import EVAL_FULL_MANIFEST, EVAL_QUICK_MANIFEST

    full = ie.load_eval_set("eval-full", EVAL_FULL_MANIFEST)
    quick = ie.load_eval_set("eval-quick", EVAL_QUICK_MANIFEST)
    print(f"eval-full: {len(full.entries)} images, labels checked; building/loading the image cache")
    cache = ie.build_cache(full)
    cache_check = ie.verify_cache(full, cache)
    print(f"  cache verified bit-identical to load_image on {cache_check['sample']} images")
    position = {e.path: i for i, e in enumerate(full.entries)}
    missing = [e.path for e in quick.entries if e.path not in position]
    if missing:
        print(f"eval-quick images not in eval-full: {missing[:3]}", file=sys.stderr)
        return 1
    quick_cache = np.asarray(cache[[position[e.path] for e in quick.entries]])

    env = environment()
    sessions = Sessions()
    selected = [r for r in runs if args.only is None or r.run_id in args.only]
    for run in selected:
        print(f"[{run.run_id}]")
        write_json(args.output_dir / f"{run.run_id}.json",
                   evaluate(run, sessions, full, cache, quick, quick_cache, cache_check,
                            args.predictions_dir, env))
    summary = comparisons(args.output_dir, full)
    if summary is not None:
        write_json(args.output_dir / "comparisons.json", summary)
        print(f"wrote {len(summary['comparisons'])} comparisons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
