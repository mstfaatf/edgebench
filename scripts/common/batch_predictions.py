"""Whether a batch-N file makes the same predictions as its batch-1 file.

The batch-equivalence summary (``reference_outputs.batch_equivalence``) says
whether a file at batch 4 or 8 computes, per item, the exact bytes its batch-1
file computes. Where it does not, this module answers the weaker question that
matters for a benchmark: does the difference change any prediction?

- Image models: the top-1 class of every image of ImageNetV2 eval-quick (1,000
  images), batch N against batch 1, same provider and compute precision.
- The time-series autoencoder: the anomaly flag of every window of the test
  split at one fixed threshold (the batch-1 file's ``max_validation_score`` on
  the same provider, the rule of ``scripts/eval_anomaly.py``), batch N against
  batch 1.

Needs NumPy, ONNX Runtime (and the QNN EP for NPU entries) and the data; the
caller provides the sessions (CPU sessions with the reference settings, strict
NPU sessions), so the computation is the one the references were made with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from common.manifests import EVAL_QUICK_MANIFEST, REPO_ROOT
from common.model_registry import IMPORTED_MODELS, MODELS, MODELS_DIR
from common.reference_outputs import entry_key, sha256_file

EVAL_QUICK_TOP1 = "eval_quick_top1"
TEST_SPLIT_FLAGS = "test_split_flags"
THRESHOLD_RULE = ("max_validation_score of the batch-1 file on the same provider and compute "
                  "precision; a window is flagged if its score is above it")

SessionFactory = Callable[[Path, str], Any]  # (model path, reference device) -> session


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def basis_inputs() -> dict[str, Any]:
    """What each basis is computed on, with the manifest hashes (checked by --check)."""
    from common.timeseries import SPLIT_MANIFEST

    return {
        EVAL_QUICK_TOP1: {"split": "eval-quick", "manifest": _repo_relative(EVAL_QUICK_MANIFEST),
                          "manifest_sha256": sha256_file(EVAL_QUICK_MANIFEST),
                          "compared": "top-1 class of every image"},
        TEST_SPLIT_FLAGS: {"split": "test", "manifest": _repo_relative(SPLIT_MANIFEST),
                           "manifest_sha256": sha256_file(SPLIT_MANIFEST),
                           "compared": "anomaly flag of every window",
                           "threshold_rule": THRESHOLD_RULE},
    }


class _Data:
    """The eval-quick images and the NAB windows, loaded on first use."""

    def __init__(self) -> None:
        self._quick = None
        self._windows = None

    def quick(self):
        if self._quick is None:
            from common import imagenet_eval as ie

            eval_set = ie.load_eval_set("eval-quick", EVAL_QUICK_MANIFEST)
            self._quick = ie.build_cache(eval_set)
        return self._quick

    def windows(self):
        if self._windows is None:
            from common.timeseries import load_split_windows

            self._windows = load_split_windows()
        return self._windows


def _image_top1(session, model: str, variant: str, batch: int, data: _Data):
    from common import imagenet_eval as ie

    spec = MODELS[model]
    normalization = "folded" if "folded" in variant else "external"
    top5, _ = ie.predict(session, data.quick(), batch, normalization, spec.mean, spec.std)
    return top5[:, 0].copy()


def _window_scores(session, windows, batch: int):
    import numpy as np

    n = len(windows)
    pad = (-n) % batch
    x = windows[:, :, None]
    if pad:  # the last batch is padded with its last window
        x = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)])
    name = session.get_inputs()[0].name
    out = np.concatenate([session.run(None, {name: x[i:i + batch]})[0]
                          for i in range(0, len(x), batch)])[:n, :, 0]
    return ((out.astype(np.float64) - windows.astype(np.float64)) ** 2).mean(axis=1)


def compute(entries: list[dict[str, Any]], session: SessionFactory,
            progress: Callable[[str], None] = print) -> dict[tuple, dict[str, Any]]:
    """Prediction comparisons for the summary entries that are not bitwise equal to b1.

    Returns {entry_key: {predictions_equal_to_b1, predictions_basis,
    predictions_compared, predictions_differing}}.
    """
    import gc

    import numpy as np

    data = _Data()
    cache: dict[tuple[str, str, int], Any] = {}  # (file, device, batch) -> predictions

    def predictions(file: str, model: str, variant: str, batch: int, device: str):
        key = (file, device, batch)
        if key in cache:
            return cache[key]
        s = session(MODELS_DIR / file, device)
        if model in IMPORTED_MODELS:
            w = data.windows()
            value = (_window_scores(s, w.windows["validation"], batch),
                     _window_scores(s, w.windows["test"], batch))
        else:
            value = _image_top1(s, model, variant, batch, data)
        del s
        gc.collect()
        cache[key] = value
        return value

    out = {}
    for e in entries:
        if e["bitwise_equal_to_b1"]:
            continue
        device = e["reference_device"]
        b1 = predictions(e["b1_file"], e["model"], e["variant"], 1, device)
        bn = predictions(e["file"], e["model"], e["variant"], e["batch_size"], device)
        if e["model"] in IMPORTED_MODELS:
            threshold = float(np.max(b1[0]))
            flags_b1, flags_bn = b1[1] > threshold, bn[1] > threshold
            differing = int(np.count_nonzero(flags_b1 != flags_bn))
            compared, basis = len(flags_b1), TEST_SPLIT_FLAGS
        else:
            differing = int(np.count_nonzero(b1 != bn))
            compared, basis = len(b1), EVAL_QUICK_TOP1
        progress(f"  {e['file']} ({e['provider']}, {e['compute_precision']}): "
                 f"{differing} of {compared} predictions differ from batch 1")
        out[entry_key(e)] = {
            "predictions_equal_to_b1": differing == 0,
            "predictions_basis": basis,
            "predictions_compared": compared,
            "predictions_differing": differing,
        }
    return out


def _image_logits(session, model: str, variant: str, batch: int, data: _Data):
    """The full logits [N, classes] of every eval-quick image."""
    import numpy as np

    from common import imagenet_eval as ie

    spec = MODELS[model]
    normalization = "folded" if "folded" in variant else "external"
    cache = data.quick()
    name = session.get_inputs()[0].name
    out = []
    for start in range(0, len(cache), ie.CHUNK):
        x = ie.model_input(cache, start, min(start + ie.CHUNK, len(cache)), normalization,
                           spec.mean, spec.std)
        out += [session.run(None, {name: x[b:b + batch]})[0] for b in range(0, len(x), batch)]
    return np.concatenate(out)


def confirm(entries: list[dict[str, Any]], session: SessionFactory,
            progress: Callable[[str], None] = print) -> dict[tuple, dict[str, Any]]:
    """Whole-split confirmation of image entries: batch N against batch 1 on every
    eval-quick image, same reference device.

    The 8 reference images decide bitwise_equal_to_b1; this checks the same claim on
    all 1,000 eval-quick images: how many images' full logits differ in any bit, and
    how many get a different top-1 class. Returns {entry_key: {basis, compared,
    outputs_differing, top1_differing, outputs_bitwise_equal_to_b1}}.
    """
    import gc

    import numpy as np

    data = _Data()
    cache: dict[tuple[str, str], Any] = {}

    def logits(file: str, model: str, variant: str, batch: int, device: str):
        key = (file, device)
        if key not in cache:
            s = session(MODELS_DIR / file, device)
            cache[key] = _image_logits(s, model, variant, batch, data)
            del s
            gc.collect()
        return cache[key]

    out = {}
    for e in entries:
        if e["model"] in IMPORTED_MODELS:
            raise ValueError(f"{e['file']}: eval-quick confirmation is for image models only")
        device = e["reference_device"]
        b1 = logits(e["b1_file"], e["model"], e["variant"], 1, device)
        bn = logits(e["file"], e["model"], e["variant"], e["batch_size"], device)
        differing = int(np.count_nonzero(np.any(b1 != bn, axis=1)))
        top1 = int(np.count_nonzero(b1.argmax(axis=1) != bn.argmax(axis=1)))
        progress(f"  {e['file']} ({device}): {differing} of {len(b1)} eval-quick images differ "
                 f"from batch 1 in any output bit, {top1} in top-1")
        out[entry_key(e)] = {"basis": EVAL_QUICK_TOP1, "compared": len(b1),
                             "outputs_differing": differing, "top1_differing": top1,
                             "outputs_bitwise_equal_to_b1": differing == 0}
    return out
