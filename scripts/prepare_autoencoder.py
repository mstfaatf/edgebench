"""Make the autoencoder's model files from its committed source, check them, or verify them.

Reads the SHA256-pinned Colab bundle in ``models/sources/conv1d_autoencoder/``
(see ``common/autoencoder_pipeline.py``) and writes, for batch sizes 1, 4 and 8,
into ``models/conv1d_autoencoder/``:

- ``conv1d_autoencoder_fp32_imported_b<n>.onnx``: the source graph with
  ``edgebench.*`` metadata (reference copy);
- ``conv1d_autoencoder_fp32_b<n>.onnx``: the exactly cleaned graph
  (``common/onnx_cleanup.py``);
- ``conv1d_autoencoder_int8_b<n>.onnx`` and ``..._int8_unpruned_b<n>.onnx``: the
  default INT8 recipe on the cleaned graph, calibrated on the 200 calibration
  windows of ``datasets/nab-machine-temperature/splits.json``;

each with a check sidecar (``<name>.json``) and a recipe sidecar
(``<name>.recipe.json``), and their entries in ``models/manifest.json``.

Checks, each of which fails the run (NAB data needed: ``get_data.py --dataset nab``):

- the source matches its SHA256SUMS and the registry pins;
- the imported file's graph is the source graph byte for byte;
- the cleaned graph gives the imported graph's outputs on every validation
  window (CPU EP; the maximum absolute difference is recorded) and matches the
  bundle's reference outputs (Keras, and ONNX Runtime on x64 in Colab) to 1e-4;
- batch 4 and 8 give batch 1's outputs (FP32 and INT8, CPU EP);
- every batch size of the INT8 model has identical scales and zero points, and
  the pruned file gives the unpruned file's outputs bit for bit.

The INT8 recipe also runs once on the uncleaned graph, for comparison only (not
written): the check sidecar of the batch-1 INT8 file records whether the
clean-up changed the INT8 reconstruction error.

``--verify`` builds everything again in memory and compares it with the
manifest and the files on disk, without writing anything.

Usage:
    python scripts/prepare_autoencoder.py [--verify]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402
from common.model_registry import (  # noqa: E402
    BATCH_SIZES,
    IMPORTED_MODELS,
    MANIFEST_PATH,
    MODELS_DIR,
    read_manifest,
    relative_path,
    write_manifest,
)

MODEL = "conv1d_autoencoder"
REFERENCE_TOLERANCE = 1e-4  # ARM64 vs x64 ONNX Runtime, and vs Keras (float rounding)
CPU_THREADS = 4


def write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def cpu_session(data: bytes):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    return ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])


def run_windows(session, windows, batch: int):
    """Outputs ``[n, length]`` for windows ``[n, length]``; the last batch is padded."""
    import numpy as np

    n = len(windows)
    pad = (-n) % batch
    x = windows[:, :, None]
    if pad:
        x = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)])
    name = session.get_inputs()[0].name
    out = np.concatenate([session.run(None, {name: x[i:i + batch]})[0]
                          for i in range(0, len(x), batch)])
    return out[:n, :, 0]


def window_mse(x, y):
    import numpy as np

    return ((y.astype(np.float64) - x.astype(np.float64)) ** 2).mean(axis=1)


def load_inputs():
    from common import autoencoder_pipeline as ap
    from common.timeseries import SPLIT_MANIFEST, load_split_windows

    spec = IMPORTED_MODELS[MODEL]
    problems = ap.source_problems(spec)
    if problems:
        raise SystemExit("error: the source bundle is not the pinned one: " + "; ".join(problems))
    data = load_split_windows()
    return spec, data, ap.sha256_bytes(SPLIT_MANIFEST.read_bytes())


def build(raw_int8: bool):
    from common import autoencoder_pipeline as ap

    spec, data, manifest_sha = load_inputs()
    built = ap.build_all(spec, ap.calibration_windows(data.calibration), data.manifest,
                         manifest_sha, BATCH_SIZES, raw_int8=raw_int8)
    return spec, data, built


def verify() -> int:
    from common import onnx_export as ox

    _, _, built = build(raw_int8=False)
    by_file = {f.file: f for f in read_manifest()[0]}
    failures = 0
    for rel, data in sorted(built["files"].items()):
        digest = ox.sha256_bytes(data)
        entry = by_file.get(rel)
        on_disk = MODELS_DIR / rel
        disk = ox.sha256_file(on_disk) if on_disk.is_file() else "missing"
        ok = entry is not None and entry.sha256 == digest and entry.size_bytes == len(data)
        failures += not ok
        state = ("identical" if ok and disk == digest else "MISMATCH" if not ok
                 else "manifest matches, file on disk differs")
        print(f"{rel}: rebuilt {digest[:12]} ({len(data)} bytes); manifest "
              f"{entry.sha256[:12] if entry else 'missing'}; file {disk[:12]}: {state}")
    print("verify: " + ("OK" if not failures else f"{failures} file(s) differ from the manifest"))
    return 1 if failures else 0


def prepare() -> int:
    import numpy as np
    import onnx

    from common import onnx_quantize as oq
    from common.model_recipe import record_model_file
    from common.onnx_quantize import graph_without_metadata

    spec, data, built = build(raw_int8=True)
    files, reports = built["files"], built["reports"]
    val = data.windows["validation"]
    failures = []

    def fail(message: str) -> None:
        failures.append(message)
        print(f"  FAIL {message}", file=sys.stderr)

    outputs: dict[str, np.ndarray] = {}
    for rel, blob in sorted(files.items()):
        batch = int(rel.rsplit("_b", 1)[1].removesuffix(".onnx"))
        outputs[rel] = run_windows(cpu_session(blob), val, batch)

    # Bundle reference arrays (Keras, and ONNX Runtime on x64 in Colab).
    src = Path(spec.source_dir)
    from common.model_registry import REPO_ROOT

    ref_in = np.load(REPO_ROOT / src / "reference_inputs.npy")
    ref_keras = np.load(REPO_ROOT / src / "reference_keras.npy")
    ref_onnx = np.load(REPO_ROOT / src / "reference_onnx_b1.npy")
    if not np.array_equal(ref_in[:, :, 0], data.reference):
        fail("the bundle's reference inputs are not the split manifest's reference windows")

    manifest, tools = read_manifest()
    by_file = {f.file: f for f in manifest}
    sidecars: dict[str, dict] = {}
    for batch in BATCH_SIZES:
        rel_imp = relative_path(MODEL, "fp32_imported", batch)
        rel_fp32 = relative_path(MODEL, "fp32", batch)
        rel_int8 = relative_path(MODEL, "int8", batch)
        rel_unp = relative_path(MODEL, "int8_unpruned", batch)
        raw = (REPO_ROOT / spec.source_file(batch)).read_bytes()
        same_graph = graph_without_metadata(onnx.load_from_string(files[rel_imp])) == raw
        if not same_graph:
            fail(f"{rel_imp}: graph differs from the source file")
        diff = float(np.abs(outputs[rel_fp32] - outputs[rel_imp]).max())
        identical = bool(np.array_equal(outputs[rel_fp32], outputs[rel_imp]))
        cleaned_ref = run_windows(cpu_session(files[rel_fp32]), data.reference, batch)
        vs_onnx = float(np.abs(cleaned_ref - ref_onnx[:, :, 0]).max())
        vs_keras = float(np.abs(cleaned_ref - ref_keras[:, :, 0]).max())
        if diff > REFERENCE_TOLERANCE or vs_onnx > REFERENCE_TOLERANCE or \
                vs_keras > REFERENCE_TOLERANCE:
            fail(f"{rel_fp32}: differs from the imported graph or the bundle references")
        cleanup = reports["cleanup"][batch]
        sidecars[rel_imp] = {"source_file": spec.source_file(batch),
                             "graph_identical_to_source": same_graph,
                             "nodes": cleanup["nodes_before"], "ops": cleanup["ops_before"]}
        sidecars[rel_fp32] = {
            "cleanup": cleanup,
            "vs_imported_on_validation": {"windows": int(len(val)), "max_abs_diff": diff,
                                          "bit_identical": identical},
            "vs_bundle_reference": {"windows": int(len(ref_in)),
                                    "max_abs_diff_vs_colab_onnxruntime_x64": vs_onnx,
                                    "max_abs_diff_vs_keras": vs_keras},
        }
        pruned_same = bool(np.array_equal(outputs[rel_int8], outputs[rel_unp]))
        if not pruned_same:
            fail(f"{rel_int8}: pruned and unpruned outputs differ")
        mse_int8 = window_mse(val, outputs[rel_int8])
        mse_fp32 = window_mse(val, outputs[rel_fp32])
        sidecars[rel_int8] = {
            "pruning": reports["prune"][batch],
            "qnn_preprocess_changed": reports["qnn_preprocess_changed"][batch],
            "pruned_vs_unpruned_bit_identical": pruned_same,
            "vs_fp32_on_validation": {
                "max_abs_diff": float(np.abs(outputs[rel_int8] - outputs[rel_fp32]).max()),
                "mean_window_mse_int8": float(mse_int8.mean()),
                "mean_window_mse_fp32": float(mse_fp32.mean())},
        }
        sidecars[rel_unp] = {"pruned_file": rel_int8}
        if batch != 1:
            for variant in ("fp32_imported", "fp32", "int8", "int8_unpruned"):
                rel = relative_path(MODEL, variant, batch)
                b1 = outputs[relative_path(MODEL, variant, 1)]
                same = bool(np.array_equal(outputs[rel], b1))
                sidecars[rel]["vs_b1_on_validation_bit_identical"] = same
                sidecars[rel]["vs_b1_max_abs_diff"] = float(np.abs(outputs[rel] - b1).max())
                if float(np.abs(outputs[rel] - b1).max()) > REFERENCE_TOLERANCE:
                    fail(f"{rel}: differs from batch 1")

    protos = {b: onnx.load_from_string(files[relative_path(MODEL, "int8", b)])
              for b in BATCH_SIZES}
    params = oq.quant_param_differences(protos)
    if params:
        fail(f"INT8 quantization parameters differ between batch sizes: {params[:3]}")

    # The clean-up's effect on INT8: same recipe on the uncleaned graph (not shipped).
    raw_int8 = run_windows(cpu_session(built["raw_int8"][1]["pruned"]), val, 1)
    rel_b1 = relative_path(MODEL, "int8", 1)
    fp32_b1 = outputs[relative_path(MODEL, "fp32", 1)]
    raw_nodes = len(onnx.load_from_string(built["raw_int8"][1]["pruned"]).graph.node)
    clean_nodes = len(protos[1].graph.node)
    effect = {
        "description": "The INT8 recipe on the uncleaned (imported) graph versus on the "
                       "cleaned graph, batch 1, CPU EP, all validation windows. The uncleaned "
                       "INT8 file is not shipped.",
        "int8_nodes": {"uncleaned": raw_nodes, "cleaned": clean_nodes},
        "mean_window_mse": {"fp32": float(window_mse(val, fp32_b1).mean()),
                            "int8_uncleaned": float(window_mse(val, raw_int8).mean()),
                            "int8_cleaned": float(window_mse(val, outputs[rel_b1]).mean())},
        "max_abs_diff_vs_fp32": {
            "int8_uncleaned": float(np.abs(raw_int8 - fp32_b1).max()),
            "int8_cleaned": float(np.abs(outputs[rel_b1] - fp32_b1).max())},
    }
    sidecars[rel_b1]["cleanup_effect_on_int8"] = effect

    for rel, blob in sorted(files.items()):
        path = MODELS_DIR / rel
        write_atomically(path, blob)
        entry = record_model_file(path, rel, blob)
        by_file[rel] = entry
        sidecar = {"model": Path(rel).name, "model_name": MODEL, "variant": entry.variant,
                   "batch_size": entry.batch_size, "parent_sha256": entry.parent_sha256,
                   "onnx_metadata": {p.key: p.value
                                     for p in onnx.load_from_string(blob).metadata_props},
                   "checks": sidecars[rel], "sha256": entry.sha256,
                   "size_bytes": entry.size_bytes}
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n",
                                             encoding="utf-8", newline="\n")
        print(f"{rel}: {len(blob)} bytes, sha256 {entry.sha256[:12]}")
    write_manifest(list(by_file.values()), tools)
    print(f"updated {MANIFEST_PATH}")

    c = reports["cleanup"][1]
    print(f"\nclean-up: {c['nodes_before']} -> {c['nodes_after']} nodes "
          f"({c['ops_before']} -> {c['ops_after']})")
    for batch in BATCH_SIZES:
        s = sidecars[relative_path(MODEL, "fp32", batch)]
        print(f"  b{batch}: cleaned vs imported on {s['vs_imported_on_validation']['windows']} "
              f"validation windows: max |diff| {s['vs_imported_on_validation']['max_abs_diff']:.3g}"
              f" (bit-identical {s['vs_imported_on_validation']['bit_identical']}); vs Colab "
              f"ORT x64 {s['vs_bundle_reference']['max_abs_diff_vs_colab_onnxruntime_x64']:.3g},"
              f" vs Keras {s['vs_bundle_reference']['max_abs_diff_vs_keras']:.3g}")
    print(f"INT8 nodes uncleaned/cleaned: {raw_nodes}/{clean_nodes}; mean window MSE "
          + ", ".join(f"{k} {v:.4g}" for k, v in effect["mean_window_mse"].items())
          + "; max |diff| vs FP32 "
          + ", ".join(f"{k} {v:.4g}" for k, v in effect["max_abs_diff_vs_fp32"].items()))
    if failures:
        print(f"{len(failures)} check(s) failed", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="build again in memory and compare with models/manifest.json")
    args = parser.parse_args(argv)
    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2
    return verify() if args.verify else prepare()


if __name__ == "__main__":
    sys.exit(main())
