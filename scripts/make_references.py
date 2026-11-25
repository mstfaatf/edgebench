"""Write, or check, the reference outputs of every model file.

For every file in ``models/manifest.json`` this runs Python's ONNX Runtime on
the two fixed input sets of ``common/reference_outputs.py`` (8 ImageNetV2
eval-quick images, or for the time-series autoencoder the 8 reference windows
of its split manifest; and the SplitMix64 input ``edgebench run`` generates) and
stores inputs, outputs, an index and one JSON record per file and device in
``models/references/``:

- ``cpu``: every file, CPU EP with 4 intra-op threads and the engine's CPU
  provider session settings (sequential execution, all graph optimizations,
  ``session.intra_op.allow_spinning`` 1);
- ``npu``: every INT8 file, QNN HTP in strict mode (CPU fallback disabled),
  burst performance mode;
- ``npu_fp16``: every FP32 file, the same session with
  ``enable_htp_fp16_precision`` 1 (the NPU computes FP32 models in FP16).

Each NPU input is run twice in the same session; whether the two outputs are
identical is recorded.

The directory is replaced as a whole; the previous one is kept until the new one
is complete. ``--check`` writes nothing and fails (exit 1) if a reference is
missing, stale (recorded model SHA256 differs from the current file) or
damaged (a tensor's SHA256 differs from the index).

Usage:
    python scripts/make_references.py [--no-npu]
    python scripts/make_references.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import reference_outputs as ro  # noqa: E402
from common.model_registry import IMPORTED_MODELS, MODELS_DIR, read_manifest  # noqa: E402


def cpu_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = ro.CPU_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class Writer:
    """Collects tensors under a new reference directory and their index rows."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.rows: list[ro.IndexRow] = []

    def tensor(self, rel: str, array, kind: str, input_set: str, model: str = "-",
               device: str = "-", run: int = 0, inputs: tuple[str, ...] = ()) -> str:
        import numpy as np

        data = np.ascontiguousarray(array, dtype="<f4").tobytes()
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.rows.append(ro.IndexRow(kind, input_set, model, device, run, inputs, rel, "float32",
                                     tuple(int(d) for d in array.shape),
                                     hashlib.sha256(data).hexdigest()))
        return rel


def generate(npu: bool) -> int:
    import numpy as np
    import onnxruntime as ort

    from common import model_recipe as mr
    from common.manifests import EVAL_QUICK_MANIFEST, IMAGENETV2_ROOT
    from common.manifests import read_manifest as read_split
    from common.preprocess import load_image, normalize, preprocess_description

    files, _ = read_manifest()
    present = [f for f in files if (MODELS_DIR / f.file).is_file()]
    if not present:
        print("no model files; run scripts/export_models.py first", file=sys.stderr)
        return 1
    images = ro.select_images(read_split(EVAL_QUICK_MANIFEST))
    for e in images:
        path = IMAGENETV2_ROOT / e.path
        if not path.is_file() or ro.sha256_file(path) != e.sha256:
            print(f"{path} missing or changed; run scripts/get_data.py --dataset imagenetv2",
                  file=sys.stderr)
            return 1

    split_windows = None
    split_sha = ""
    if any(f.model in IMPORTED_MODELS for f in present):
        from common.timeseries import SPLIT_MANIFEST, load_split_windows

        split_windows = load_split_windows()
        split_sha = ro.sha256_file(SPLIT_MANIFEST)

    tmp = ro.REFERENCE_DIR.with_name(ro.REFERENCE_DIR.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    w = Writer(tmp)
    unit = np.concatenate([load_image(IMAGENETV2_ROOT / e.path, normalized=False) for e in images])
    input_sets: dict[str, dict] = {}

    def window_files() -> tuple[str, list[str]]:
        set_id = ro.window_input_set(split_sha)
        if set_id not in input_sets:
            x = split_windows.reference[:, :, None]
            names = [w.tensor(f"inputs/{set_id}/{i:02d}.f32", x[i:i + 1], "input", set_id)
                     for i in range(len(x))]
            input_sets[set_id] = {
                "kind": "windows", "split_manifest_sha256": split_sha,
                "window_starts": split_windows.manifest["reference_windows"]["window_starts"],
                "files": [{"file": n, "sha256": w.rows[-len(names) + k].sha256}
                          for k, n in enumerate(names)]}
        return set_id, [f["file"] for f in input_sets[set_id]["files"]]

    def image_files(recipe: dict) -> tuple[str, list[str]]:
        p = recipe["preprocessing"]
        set_id = ro.image_input_set(p["normalization"], p["mean"], p["std"])
        if set_id not in input_sets:
            x = unit if p["normalization"] == "folded" else normalize(unit, tuple(p["mean"]),
                                                                       tuple(p["std"]))
            names = [w.tensor(f"inputs/{set_id}/{i:02d}.f32", x[i:i + 1], "input", set_id)
                     for i in range(len(images))]
            input_sets[set_id] = {
                "kind": "images", "normalization": p["normalization"],
                "mean": None if p["normalization"] == "folded" else p["mean"],
                "std": None if p["normalization"] == "folded" else p["std"],
                "files": [{"file": n, "sha256": w.rows[-len(names) + k].sha256}
                          for k, n in enumerate(names)]}
        return set_id, [f["file"] for f in input_sets[set_id]["files"]]

    def splitmix_file(shape: list[int]) -> tuple[str, str]:
        set_id = ro.splitmix_input_set(shape)
        if set_id not in input_sets:
            x = ro.splitmix64_floats(ro.SPLITMIX_SEED, int(np.prod(shape))).reshape(shape)
            name = w.tensor(f"inputs/{set_id}/00.f32", x, "input", set_id)
            input_sets[set_id] = {"kind": "splitmix64", "seed": ro.SPLITMIX_SEED,
                                  "shape": shape, "files": [{"file": name,
                                                             "sha256": w.rows[-1].sha256}]}
        return set_id, input_sets[set_id]["files"][0]["file"]

    def run_all(session, entry, recipe, device: str, repeat: bool) -> tuple[list[dict], bool]:
        shape = recipe["export"]["inputs"][0]["shape"]
        name = recipe["export"]["inputs"][0]["name"]
        batch = shape[0]
        series = recipe["preprocessing"]["kind"] == "time_series"
        set_id, files_ = window_files() if series else image_files(recipe)
        runs = [(set_id, k, files_[k * batch:(k + 1) * batch]) for k in range(len(files_) // batch)]
        splitmix_set, splitmix_input = splitmix_file(shape)
        runs.append((splitmix_set, 0, [splitmix_input]))
        outputs, identical = [], True
        for input_set, k, inputs in runs:
            x = np.concatenate([np.fromfile(tmp / f, dtype="<f4") for f in inputs]).reshape(shape)
            y = session.run(None, {name: x})[0]
            if repeat:
                identical &= bool(np.array_equal(y, session.run(None, {name: x})[0]))
            rel = (ro.output_dir(entry.file, Path("")) / device / f"{input_set}-{k}.f32").as_posix()
            w.tensor(rel, y, "output", input_set, entry.file, device, k, tuple(inputs))
            summary = ({"window_mse": [float(v) for v in
                                       ((y.astype(np.float64) - x) ** 2).reshape(batch, -1).mean(1)]}
                       if series else {"top1": [int(v) for v in y.argmax(axis=1)]})
            outputs.append({"input_set": input_set, "run": k, "inputs": inputs, "file": rel,
                            "shape": list(y.shape), "sha256": w.rows[-1].sha256, **summary})
        return outputs, identical

    qnn = {}
    if npu:
        from common import onnx_quantize as oq
        from common.ort_qnn import qnn_versions

        qnn = qnn_versions()

    failures = 0
    for entry in present:
        path = MODELS_DIR / entry.file
        recipe = mr.build_recipe(mr.read_model_file(path, entry.file))
        common = {"reference_version": ro.REFERENCE_VERSION, "model_file": entry.file,
                  "model_sha256": entry.sha256, "variant": entry.variant,
                  "input_expected": recipe["preprocessing"]["input_range"],
                  "onnxruntime": ort.__version__}
        if ro.sha256_file(path) != entry.sha256:
            print(f"{entry.file}: file does not match models/manifest.json", file=sys.stderr)
            return 1
        outputs, _ = run_all(cpu_session(path), entry, recipe, "cpu", repeat=False)
        ro.write_json(ro.record_path(entry.file, "cpu", tmp), {
            **common, "device": "cpu", "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": ro.CPU_THREADS,
            "session": {"execution_mode": "sequential", "graph_optimization": "all",
                        "session.intra_op.allow_spinning": "1"},
            "outputs": outputs})
        line = f"{entry.file}: cpu {len(outputs)} runs"
        device = ro.NPU_DEVICES.get(entry.precision)
        if npu and device:
            fp16 = device == "npu_fp16"
            try:
                session = oq.npu_session(path, fp16=fp16)
            except Exception as exc:  # noqa: BLE001 - reported, the run fails at the end
                failures += 1
                print(f"{entry.file}: NPU strict session failed: {str(exc)[:300]}",
                      file=sys.stderr)
                continue
            outputs_npu, identical = run_all(session, entry, recipe, device, repeat=True)
            del session
            if recipe["preprocessing"]["kind"] == "time_series":
                # Reconstructions: the largest difference from the CPU on the windows.
                worst = max(float(np.abs(np.fromfile(tmp / a["file"], dtype="<f4")
                                         - np.fromfile(tmp / b["file"], dtype="<f4")).max())
                            for a, b in zip(outputs, outputs_npu)
                            if a["input_set"].startswith(ro.WINDOWS))
                agreement = {"windows_max_abs_diff_vs_cpu": worst}
                said = f"window outputs vs cpu max |diff| {worst:.3g}"
            else:
                # Top-1 agreement on the images only: on the random input it says little.
                cpu_top1 = [t for o in outputs if o["input_set"].startswith(ro.IMAGES)
                            for t in o["top1"]]
                npu_top1 = [t for o in outputs_npu if o["input_set"].startswith(ro.IMAGES)
                            for t in o["top1"]]
                agree = sum(a == b for a, b in zip(cpu_top1, npu_top1))
                agreement = {"images_top1_agreement_with_cpu": f"{agree}/{len(cpu_top1)}"}
                said = f"image top-1 vs cpu {agree}/{len(cpu_top1)}"
            provider_options = {"backend_path": "<onnxruntime_qnn>/QnnHtp.dll",
                                "htp_performance_mode": "burst"}
            if fp16:
                provider_options["enable_htp_fp16_precision"] = "1"
            ro.write_json(ro.record_path(entry.file, device, tmp), {
                **common, "device": device, "execution_provider": "QNNExecutionProvider",
                "strict": True, "session": {"session.disable_cpu_ep_fallback": "1"},
                "provider_options": provider_options,
                **({"compute_precision": "fp16"} if fp16 else {}),
                **qnn, "repeat_identical": identical, **agreement,
                "outputs": outputs_npu})
            line += f"; {device} strict ok, {said}, repeat identical {identical}"
        print(line)

    selection = [{"path": e.path, "label": e.label, "sha256": e.sha256} for e in images]
    ro.write_json(tmp / "inputs.json", {
        "reference_version": ro.REFERENCE_VERSION,
        "images": {"source": "ImageNetV2 eval-quick "
                             "(datasets/imagenetv2-matched-frequency/eval-quick.tsv)",
                   "rule": f"random.Random({ro.IMAGE_SEED}).sample of the manifest in "
                           f"(label, path) order, first {ro.IMAGE_COUNT} distinct classes",
                   "seed": ro.IMAGE_SEED, "selection": selection,
                   "preprocessing": preprocess_description()},
        **({"windows": {
            "source": "reference_windows of datasets/nab-machine-temperature/splits.json",
            "rule": "one window centred in each labelled anomaly of the test split, plus "
                    "evenly spaced normal test windows (common/timeseries.py)",
            "split_manifest_sha256": split_sha,
            "window_starts": split_windows.manifest["reference_windows"]["window_starts"],
            "normalization": "(x - mean) / std with the split manifest's mean and std"}}
           if split_windows is not None else {}),
        "splitmix64": {"seed": ro.SPLITMIX_SEED, "algorithm": "docs/cli.md, Input data"},
        "input_sets": dict(sorted(input_sets.items()))})
    (tmp / "index.tsv").write_text(ro.format_index(w.rows), encoding="utf-8", newline="\n")
    shutil.rmtree(ro.REFERENCE_DIR, ignore_errors=True)
    tmp.rename(ro.REFERENCE_DIR)
    print(f"wrote {sum(r.kind == 'output' for r in w.rows)} output and "
          f"{sum(r.kind == 'input' for r in w.rows)} input tensors to {ro.REFERENCE_DIR}")
    problems = ro.check_problems(read_manifest()[0]) if not failures else []
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    return 1 if failures or problems else 0


def check() -> int:
    files, _ = read_manifest()
    problems = ro.check_problems(files)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    present = sum((MODELS_DIR / f.file).is_file() for f in files)
    print(f"checked references of {present} model file(s): "
          + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="check the references against the model files; write nothing")
    parser.add_argument("--no-npu", action="store_true", help="CPU references only")
    args = parser.parse_args(argv)
    if args.check:
        return check()

    from common.platform_guard import NonNativeInterpreterError, require_native_arm64

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2
    return generate(npu=not args.no_npu)


if __name__ == "__main__":
    sys.exit(main())
