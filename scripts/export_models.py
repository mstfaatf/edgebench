"""Export torchvision and timm classifiers to fixed-shape FP32 ONNX files, one per batch size.

For every model in ``common/model_registry.py`` (``--model``, default all) and
batch size (``--batch``, default 1 4 8) this writes
``models/<model>/<model>_fp32_b<batch>.onnx`` and a JSON sidecar next to it,
then updates ``models/manifest.json`` with the file's SHA256 and size.

Each file is checked before it is written: ``onnx.checker`` (full check),
strict shape inference, concrete dimensions on every graph input and output,
and the expected input [batch, 3, 224, 224] / output [batch, 1000]. After
writing, ONNX Runtime (CPU EP) runs it:

- on 64 preprocessed ImageNetV2 images (if downloaded with
  ``scripts/get_data.py --dataset imagenetv2``), compared with PyTorch, and
  for batch sizes above 1 also with the batch-1 file, row by row;
- for models with a probe check, on a tensor filled with a constant.

The weights are verified against the SHA256 pinned in the registry before use.

``--verify`` exports again in memory and compares with the manifest and with
the files on disk, without writing anything. Byte-identical output needs the
same torch, torchvision and onnx versions as recorded in the manifest.

The models expect normalized input (each model's own mean/std, applied by
``common/preprocess.py``); see docs/models.md.

Usage:
    python scripts/export_models.py [--model NAME ...] [--batch N ...] [--verify]
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
    EXPORTER,
    MANIFEST_PATH,
    MODELS,
    MODELS_DIR,
    ManifestFile,
    read_manifest,
    relative_path,
    write_manifest,
)

VARIANT = "fp32"


def export_and_check(spec, batch, model):
    from common import onnx_export as ox

    data = ox.export_fp32(spec, batch, model)
    return data, ox.check_exported(data, spec, batch)


def write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def verify(names: list[str], batches: list[int]) -> int:
    from common import onnx_export as ox

    manifest, tools = read_manifest()
    by_file = {f.file: f for f in manifest}
    current = ox.tool_versions()
    if tools != current:
        print(f"note: manifest was written with {tools}, this environment has {current}; "
              "the files are not expected to match")
    failures = 0
    for name in names:
        spec = MODELS[name]
        model = ox.load_torch_model(spec)
        for batch in batches:
            rel = relative_path(name, VARIANT, batch)
            data, _ = export_and_check(spec, batch, model)
            digest = ox.sha256_bytes(data)
            entry = by_file.get(rel)
            on_disk = MODELS_DIR / rel
            disk_digest = ox.sha256_file(on_disk) if on_disk.is_file() else None
            ok = entry is not None and entry.sha256 == digest and entry.size_bytes == len(data)
            failures += 0 if ok else 1
            print(f"{rel}: re-export {digest[:12]} ({len(data)} bytes); manifest "
                  f"{entry.sha256[:12] if entry else 'missing'}; file "
                  f"{(disk_digest or 'missing')[:12]}: "
                  f"{'identical' if ok and disk_digest == digest else 'MISMATCH' if not ok else 'manifest matches, file on disk differs'}")
    print("verify: " + ("OK" if not failures else f"{failures} file(s) differ from the manifest"))
    return 1 if failures else 0


def export(names: list[str], batches: list[int]) -> int:
    from common import onnx_export as ox

    versions = ox.tool_versions()
    manifest, tools = read_manifest()
    by_file = {f.file: f for f in manifest}
    if tools and tools != versions and by_file:
        print(f"note: manifest entries from {tools} are kept; this run uses {versions}")
    images = ox.check_images()
    if images is None:
        print("PyTorch comparison skipped: run scripts/get_data.py --dataset imagenetv2 first")
    failures = 0
    for name in names:
        spec = MODELS[name]
        model = ox.load_torch_model(spec)
        x = ox.normalize_for(spec, images[1]) if images else None
        reference = ox.run_torch(model, x) if images else None
        b1_logits = None
        for batch in sorted(batches):
            rel = relative_path(name, VARIANT, batch)
            path = MODELS_DIR / rel
            data, info = export_and_check(spec, batch, model)
            write_atomically(path, data)
            session = ox.ort_session(path)

            probe = None
            if spec.probe_value is not None and batch == 1:
                import numpy as np

                logits = session.run(None, {"input": np.full(ox.input_shape(spec, 1),
                                                             spec.probe_value, np.float32)})[0][0]
                top1 = int(logits.argmax())
                probe = {"input_value": spec.probe_value, "top1": top1,
                         "top1_value": float(logits[top1]), "expected_top1": spec.probe_top1}
                failures += top1 != spec.probe_top1

            torch_check = batch_check = None
            if images:
                logits = ox.run_ort(session, x)
                torch_check = ox.compare_logits(logits, reference)
                failures += not ox.within_tolerance(torch_check)
                if batch == 1:
                    b1_logits = logits
                elif b1_logits is None and (MODELS_DIR / relative_path(name, VARIANT, 1)).is_file():
                    b1_logits = ox.run_ort(
                        ox.ort_session(MODELS_DIR / relative_path(name, VARIANT, 1)), x)
                if batch != 1 and b1_logits is not None:
                    batch_check = ox.compare_logits(logits, b1_logits)
                    failures += not ox.within_tolerance(batch_check)

            sidecar = {
                "model": path.name,
                "model_name": spec.name,
                "variant": VARIANT,
                "batch_size": batch,
                "source": {"weights": spec.weights_id, "url": spec.weights_url,
                           "sha256": spec.weights_sha256},
                "exporter": EXPORTER,
                "opset": info["opset"],
                "ir_version": info["ir_version"],
                "versions": versions,
                "input": {"name": "input", "shape": info["shapes"]["input"], "dtype": "float32",
                          "normalization": "external", "mean": list(spec.mean),
                          "std": list(spec.std)},
                "output": {"name": "logits", "shape": info["shapes"]["logits"],
                           "dtype": "float32", "semantics": "logits"},
                "onnx_metadata": info["metadata"],
                "nodes": info["nodes"],
                "probe_check": probe,
                "torch_check": None if torch_check is None else {
                    **torch_check, "source": "ImageNetV2 eval-quick", "atol": ox.LOGIT_ATOL},
                "batch_check": None if batch_check is None else {
                    **batch_check, "against": relative_path(name, VARIANT, 1),
                    "atol": ox.LOGIT_ATOL},
                "sha256": ox.sha256_bytes(data),
                "size_bytes": len(data),
            }
            path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
            by_file[rel] = ManifestFile(rel, sidecar["sha256"], len(data))

            parts = [f"{len(data)} bytes", f"sha256 {sidecar['sha256'][:12]}"]
            if probe:
                parts.append(f"probe top-1 {probe['top1']} (expected {probe['expected_top1']})")
            if torch_check:
                parts.append(f"vs PyTorch max |diff| {torch_check['max_abs_diff']:.3g}, "
                             f"top-1 {torch_check['top1_agreement']:.1%}")
            if batch_check:
                parts.append(f"vs b1 max |diff| {batch_check['max_abs_diff']:.3g}, "
                             f"top-1 {batch_check['top1_agreement']:.1%}")
            print(f"{rel}: " + "; ".join(parts))

    write_manifest(list(by_file.values()), versions)
    print(f"updated {MANIFEST_PATH}")
    if failures:
        print(f"{failures} check(s) failed (tolerance {ox.LOGIT_ATOL})", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", choices=sorted(MODELS), default=sorted(MODELS),
                        help="models to export (default: all)")
    parser.add_argument("--batch", nargs="+", type=int, choices=BATCH_SIZES,
                        default=list(BATCH_SIZES), help="batch sizes (default: 1 4 8)")
    parser.add_argument("--verify", action="store_true",
                        help="re-export in memory and compare with models/manifest.json")
    args = parser.parse_args(argv)

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    names = list(dict.fromkeys(args.model))
    batches = sorted(set(args.batch))
    return verify(names, batches) if args.verify else export(names, batches)


if __name__ == "__main__":
    sys.exit(main())
