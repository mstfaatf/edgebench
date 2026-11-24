"""Write, or check, the reference outputs of every model file.

For every file in ``models/manifest.json`` this runs Python's ONNX Runtime on
the two fixed input sets of ``common/reference_outputs.py`` (8 ImageNetV2
eval-quick images, and the SplitMix64 input ``edgebench run`` generates) and
stores inputs, outputs, an index and one JSON record per file and device in
``models/references/``:

- ``cpu``: every file, CPU EP with 4 intra-op threads and the engine's CPU
  provider session settings (sequential execution, all graph optimizations,
  ``session.intra_op.allow_spinning`` 1);
- ``npu``: every INT8 file, QNN HTP in strict mode (CPU fallback disabled),
  burst performance mode. Each NPU input is run twice in the same session;
  whether the two outputs are identical is recorded.

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
import ctypes
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import reference_outputs as ro  # noqa: E402
from common.model_registry import MODELS_DIR, read_manifest  # noqa: E402


def file_product_version(path: Path) -> str | None:
    """The ProductVersion string of a Windows DLL's version resource."""
    version = ctypes.WinDLL("version")
    size = version.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(str(path), 0, size, buf):
        return None
    value, length = ctypes.c_wchar_p(), ctypes.c_uint()
    # Language 0409 (US English), code page 04B0 (Unicode): what the QNN DLLs use.
    if not version.VerQueryValueW(buf, r"\StringFileInfo\040904B0\ProductVersion",
                                  ctypes.byref(value), ctypes.byref(length)):
        return None
    return value.value


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

    tmp = ro.REFERENCE_DIR.with_name(ro.REFERENCE_DIR.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    w = Writer(tmp)
    unit = np.concatenate([load_image(IMAGENETV2_ROOT / e.path, normalized=False) for e in images])
    input_sets: dict[str, dict] = {}

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
        set_id = ro.splitmix_input_set(shape[0])
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
        set_id, files_ = image_files(recipe)
        runs = [(set_id, k, files_[k * batch:(k + 1) * batch]) for k in range(len(images) // batch)]
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
            outputs.append({"input_set": input_set, "run": k, "inputs": inputs, "file": rel,
                            "shape": list(y.shape), "sha256": w.rows[-1].sha256,
                            "top1": [int(v) for v in y.argmax(axis=1)]})
        return outputs, identical

    qnn = {}
    if npu:
        import onnxruntime_qnn

        from common import onnx_quantize as oq

        backend = Path(onnxruntime_qnn.get_library_path()).parent / "QnnHtp.dll"
        qnn = {"onnxruntime_qnn": onnxruntime_qnn.__version__,
               "qnn_backend": file_product_version(backend)}

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
        if npu and entry.precision == "int8":
            try:
                session = oq.npu_session(path)
            except Exception as exc:  # noqa: BLE001 - reported, the run fails at the end
                failures += 1
                print(f"{entry.file}: NPU strict session failed: {str(exc)[:300]}",
                      file=sys.stderr)
                continue
            outputs_npu, identical = run_all(session, entry, recipe, "npu", repeat=True)
            del session
            # Top-1 agreement on the images only: on the random input it says little.
            cpu_top1 = [t for o in outputs if o["input_set"].startswith(ro.IMAGES)
                        for t in o["top1"]]
            npu_top1 = [t for o in outputs_npu if o["input_set"].startswith(ro.IMAGES)
                        for t in o["top1"]]
            agree = sum(a == b for a, b in zip(cpu_top1, npu_top1))
            ro.write_json(ro.record_path(entry.file, "npu", tmp), {
                **common, "device": "npu", "execution_provider": "QNNExecutionProvider",
                "strict": True, "session": {"session.disable_cpu_ep_fallback": "1"},
                "provider_options": {"backend_path": "<onnxruntime_qnn>/QnnHtp.dll",
                                     "htp_performance_mode": "burst"},
                **qnn, "repeat_identical": identical,
                "images_top1_agreement_with_cpu": f"{agree}/{len(cpu_top1)}",
                "outputs": outputs_npu})
            line += f"; npu strict ok, image top-1 vs cpu {agree}/{len(cpu_top1)}, " \
                    f"repeat identical {identical}"
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
