"""Write reference inputs and Python outputs for comparing ort_probe with Python.

For a fixed set of 10 eval images (the first image of each Imagenette class in
``data/imagenette/splits/eval.txt``) this writes

- the preprocessed input tensors from ``scripts/common/preprocess.py``
  (normalized float32, [1, 3, 224, 224]), and
- the Python ONNX Runtime output logits for

  ============  ===========================================  ==========================
  config        model                                        session
  ============  ===========================================  ==========================
  cpu_fp32      mobilenetv2_fp32_b1.onnx                     CPU EP, 4 intra-op threads
  cpu_s8w       mobilenetv2_int8qdq_pc_s8w_b1.onnx           CPU EP, 4 intra-op threads
  npu_s8w       mobilenetv2_int8qdq_pc_s8w_b1.onnx           QNN NPU, strict
  npu_fp16      mobilenetv2_fp32_b1.onnx                     QNN NPU, strict, FP16
  ============  ===========================================  ==========================

as raw little-endian float32 files, plus ``index.tsv`` (read by ort_probe) and
``manifest.json`` (shapes, dtypes, image ids, session settings, model hashes).
NPU sessions use the same provider options as ``scripts/common/npu_harness.py``.

Every config is also run in a second, independent session; the largest
difference between the two (``repeat_max_abs_diff`` in the manifest) is the
run-to-run variation that bounds a fair C++-vs-Python tolerance.

Each run writes a new directory ``data/ort_probe_reference/<UTC timestamp>/``
(never overwritten) and refreshes ``data/ort_probe_reference/latest/`` as a copy
of it; ``ort_probe --reference-dir`` and ``scripts/npu_overhead.py`` use
``latest/``. ``data/`` is not committed.

Usage:
    python tools/ort_probe/make_reference.py [--out-root DIR]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "mobilenetv2"
FP32 = MODEL_DIR / "mobilenetv2_fp32_b1.onnx"
S8W = MODEL_DIR / "mobilenetv2_int8qdq_pc_s8w_b1.onnx"
CPU_THREADS = 4
# config -> (model, target, extra QNN provider options)
CONFIGS = {
    "cpu_fp32": (FP32, "cpu", {}),
    "cpu_s8w": (S8W, "cpu", {}),
    "npu_s8w": (S8W, "npu", {}),
    "npu_fp16": (FP32, "npu", {"enable_htp_fp16_precision": "1"}),
}
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "ort_probe_reference"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    import numpy as np
    import onnxruntime as ort
    import onnxruntime_qnn

    from common import ort_qnn
    from common.preprocess import IMAGENETTE_ROOT, load_image, read_split, wnid_of

    eval_images = read_split("eval")
    images, seen = [], set()
    for p in eval_images:  # first image of each class, in split order (sorted)
        if wnid_of(p) not in seen:
            seen.add(wnid_of(p))
            images.append(p)
    inputs = [load_image(IMAGENETTE_ROOT / p) for p in images]

    from common.results_io import run_stamp

    root = args.out_root.resolve()
    out = root / run_stamp()
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier reference set
    rows = []
    for i, (image, x) in enumerate(zip(images, inputs)):
        rel = f"inputs/normalized/{i:02d}.f32"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        x.astype("<f4").tofile(out / rel)
        rows.append(("input", "normalized", i, image, rel, "float32", x.shape))

    ort_qnn.register_qnn()
    npu = ort_qnn.select_qnn_npu(ort_qnn.list_devices())
    base_qnn = {"backend_path": onnxruntime_qnn.get_qnn_htp_path(), "htp_performance_mode": "burst"}
    log_dir = out / "logs"

    def session(model: Path, target: str, extra: dict, tag: str):
        if target == "cpu":
            so = ort.SessionOptions()
            so.intra_op_num_threads = CPU_THREADS
            return ort.InferenceSession(str(model), sess_options=so,
                                        providers=["CPUExecutionProvider"])
        so = ort_qnn.qnn_npu_session_options(npu, {**base_qnn, **extra}, strict=True)
        with ort_qnn.capture_native_stderr(log_dir / f"{tag}.log"):
            return ort.InferenceSession(str(model), sess_options=so)

    def run(s) -> list:
        name = s.get_inputs()[0].name
        return [np.asarray(s.run(None, {name: x})[0], dtype=np.float32) for x in inputs]

    configs = {}
    for config, (model, target, extra) in CONFIGS.items():
        first = session(model, target, extra, f"{config}_a")
        outs = run(first)
        repeat_in_session = max(float(np.max(np.abs(a - b))) for a, b in zip(outs, run(first)))
        first = None
        gc.collect()
        second = session(model, target, extra, f"{config}_b")
        repeat = max(float(np.max(np.abs(a - b))) for a, b in zip(outs, run(second)))
        second = None
        gc.collect()
        for i, (image, y) in enumerate(zip(images, outs)):
            rel = f"outputs/{config}/{i:02d}.f32"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            y.astype("<f4").tofile(out / rel)
            rows.append(("output", config, i, image, rel, "float32", y.shape))
        configs[config] = {
            "model": model.relative_to(REPO_ROOT).as_posix(),
            "model_sha256": sha256_of(model),
            "target": target,
            "session": ({"intra_op_num_threads": CPU_THREADS} if target == "cpu" else
                        {"provider_options": {**base_qnn, **extra},
                         "session_config": dict(ort_qnn.STRICT_CONFIG)}),
            "repeat_max_abs_diff_same_session": repeat_in_session,
            "repeat_max_abs_diff_new_session": repeat,
            "top1": [int(np.argmax(y)) for y in outs],
        }
        print(f"{config:<9} top-1 {configs[config]['top1']}  repeat diff: same session "
              f"{repeat_in_session:.3g}, new session {repeat:.3g}")

    with (out / "index.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("kind\tconfig\timage\timage_id\tfile\tdtype\tshape\n")
        for kind, config, i, image, rel, dtype, shape in rows:
            f.write(f"{kind}\t{config}\t{i}\t{image}\t{rel}\t{dtype}\t"
                    f"{','.join(str(d) for d in shape)}\n")
    manifest = {
        "description": "ort_probe reference tensors: preprocessed inputs and Python outputs",
        "onnxruntime_version": ort.__version__,
        "images": images,
        "tensors": [{"kind": k, "config": c, "image": i, "image_id": im, "file": rel,
                     "dtype": dt, "shape": list(sh)} for k, c, i, im, rel, dt, sh in rows],
        "configs": configs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    latest = root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(out, latest)
    print(f"wrote {len(rows)} tensors, index.tsv and manifest.json to {out} (copied to {latest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
