"""Quantize the exported FP32 models to static INT8 QDQ, prune them, and check the result.

For every model in ``common/model_registry.py`` (``--model``, default all) and
batch size (``--batch``, default 1 4 8) this reads
``models/<model>/<model>_fp32_b<batch>.onnx`` (from ``scripts/export_models.py``)
and writes, next to it:

- ``<model>_int8_b<batch>.onnx``: the INT8 QDQ model with dead Q/DQ chains
  pruned, and a JSON sidecar;
- ``<model>_int8_unpruned_b<batch>.onnx``: the quantizer's output before
  pruning, kept as a reference;
- the same from the normalization-folded FP32 files (``scripts/fold_models.py``):
  ``<model>_int8_folded_b<batch>.onnx`` and ``<model>_int8_folded_unpruned_b<batch>.onnx``,
  calibrated with [0, 1] images (no mean/std), since those models take raw input.

Every file gets a recipe sidecar (``<name>.recipe.json``, see
``common/model_recipe.py``) and an entry in ``models/manifest.json``. The recipe (per-channel symmetric
int8 weights, uint8 activations, MinMax calibration on the 200 Imagenette
calibration images, batch-1 ranges shared by all batch sizes) is described in
``common/onnx_quantize.py`` and docs/models.md.

Checks, each of which fails the run:

- every batch size of a model has identical scales and zero points;
- ``onnx.checker`` (full check) on every file;
- with ImageNetV2 downloaded: pruned and unpruned files give bit-identical
  logits on 64 images, and batch 4 and 8 give the same logits as batch 1.
  INT8 vs FP32 top-1 agreement on the same images is printed and recorded.

``--verify`` quantizes again in memory and compares with the manifest and the
files on disk, without writing anything.

Usage:
    python scripts/quantize_models.py [--model NAME ...] [--batch N ...]
                                      [--normalization external|folded|all] [--verify]
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
    MANIFEST_PATH,
    MODELS,
    MODELS_DIR,
    read_manifest,
    relative_path,
    write_manifest,
)


def write_atomically(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def check_batches_share_params(name: str, produced: dict, variant: str) -> list[str]:
    """Compare quantization parameters of all batch sizes (produced now or on disk)."""
    import onnx

    from common import onnx_quantize as oq

    protos = {}
    for batch in BATCH_SIZES:
        if batch in produced:
            protos[batch] = onnx.load_from_string(produced[batch])
        elif (MODELS_DIR / relative_path(name, variant, batch)).is_file():
            protos[batch] = onnx.load(str(MODELS_DIR / relative_path(name, variant, batch)))
    return oq.quant_param_differences(protos)


SOURCES = {"external": ["fp32"], "folded": ["fp32_folded"], "all": ["fp32", "fp32_folded"]}


def verify(names: list[str], batches: list[int], sources: list[str]) -> int:
    from common import onnx_export as ox
    from common import onnx_quantize as oq

    by_file = {f.file: f for f in read_manifest()[0]}
    failures = 0
    for name, source in ((n, s) for n in names for s in sources):
        pruned_variant, unpruned_variant = oq.INT8_VARIANTS[source]
        results = oq.quantize(MODELS[name], batches, source_variant=source)
        for batch, r in results.items():
            for variant, data in ((pruned_variant, r["pruned"]),
                                  (unpruned_variant, r["unpruned"])):
                rel = relative_path(name, variant, batch)
                digest = ox.sha256_bytes(data)
                entry = by_file.get(rel)
                on_disk = MODELS_DIR / rel
                disk = ox.sha256_file(on_disk) if on_disk.is_file() else "missing"
                ok = entry is not None and entry.sha256 == digest and entry.size_bytes == len(data)
                failures += not ok
                print(f"{rel}: re-quantized {digest[:12]} ({len(data)} bytes); manifest "
                      f"{entry.sha256[:12] if entry else 'missing'}; file {disk[:12]}: "
                      f"{'identical' if ok and disk == digest else 'MISMATCH' if not ok else 'manifest matches, file on disk differs'}")
    print("verify: " + ("OK" if not failures else f"{failures} file(s) differ from the manifest"))
    return 1 if failures else 0


def quantize(names: list[str], batches: list[int], sources: list[str]) -> int:
    import numpy as np
    import onnx

    from common import onnx_export as ox
    from common import onnx_quantize as oq
    from common.model_recipe import record_model_file

    if not oq.calibration_images_present():
        print("Imagenette calibration images missing: run scripts/get_data.py first",
              file=sys.stderr)
        return 1
    # Check every source first, so a missing one stops the run before anything is written.
    missing = [relative_path(n, s, b) for n in names for s in sources for b in sorted({1, *batches})
               if not (MODELS_DIR / relative_path(n, s, b)).is_file()]
    if missing:
        print(f"source FP32 file(s) missing: {', '.join(missing)}; run scripts/export_models.py "
              "and scripts/fold_models.py first (or choose --normalization)", file=sys.stderr)
        return 1
    manifest, tools = read_manifest()
    by_file = {f.file: f for f in manifest}
    images = ox.check_images()
    if images is None:
        print("output checks skipped: run scripts/get_data.py --dataset imagenetv2 first")
    failures = 0
    for name, source in ((n, s) for n in names for s in sources):
        spec = MODELS[name]
        pruned_variant, unpruned_variant = oq.INT8_VARIANTS[source]
        print(f"quantizing {name} from {source} (calibrating on batch 1)")
        results = oq.quantize(spec, batches, source_variant=source)
        for variant, key in ((pruned_variant, "pruned"), (unpruned_variant, "unpruned")):
            problems = check_batches_share_params(
                name, {b: r[key] for b, r in results.items()}, variant)
            if problems:
                failures += 1
                print(f"  {variant}: quantization parameters differ between batch sizes: "
                      f"{problems[:3]}", file=sys.stderr)
        if not failures:
            print("  scales and zero points identical across batch sizes")

        # Folded models take the [0, 1] images as they are.
        x = None if images is None else (
            ox.normalize_for(spec, images[1]) if source == "fp32" else images[1])
        b1_logits = None
        for batch in sorted(results):
            r = results[batch]
            rel = relative_path(name, pruned_variant, batch)
            rel_unpruned = relative_path(name, unpruned_variant, batch)
            write_atomically(MODELS_DIR / rel, r["pruned"])
            write_atomically(MODELS_DIR / rel_unpruned, r["unpruned"])
            by_file[rel] = record_model_file(MODELS_DIR / rel, rel, r["pruned"])
            by_file[rel_unpruned] = record_model_file(MODELS_DIR / rel_unpruned, rel_unpruned,
                                                      r["unpruned"])

            checks: dict = {}
            if x is not None:
                logits = ox.run_ort(ox.ort_session(MODELS_DIR / rel), x)
                unpruned = ox.run_ort(ox.ort_session(MODELS_DIR / rel_unpruned), x)
                fp32 = ox.run_ort(ox.ort_session(MODELS_DIR / r["source"]), x)
                checks["pruned_vs_unpruned_bit_identical"] = bool(np.array_equal(logits, unpruned))
                failures += not checks["pruned_vs_unpruned_bit_identical"]
                checks["vs_fp32"] = ox.compare_logits(logits, fp32)
                if batch == 1:
                    b1_logits = logits
                elif b1_logits is None and \
                        (MODELS_DIR / relative_path(name, pruned_variant, 1)).is_file():
                    b1_logits = ox.run_ort(
                        ox.ort_session(MODELS_DIR / relative_path(name, pruned_variant, 1)), x)
                if batch != 1 and b1_logits is not None:
                    checks["vs_b1"] = ox.compare_logits(logits, b1_logits)
                    failures += not ox.within_tolerance(checks["vs_b1"])
                checks["images"] = f"{len(x)} ImageNetV2 eval-quick images"

            proto_meta = onnx.load_from_string(r["pruned"]).metadata_props
            sidecar = {
                "model": Path(rel).name,
                "model_name": name,
                "variant": pruned_variant,
                "batch_size": batch,
                "source_model": {"file": r["source"], "sha256": r["source_sha256"]},
                "unpruned_model": {"file": rel_unpruned,
                                   "sha256": by_file[rel_unpruned].sha256},
                "qnn_preprocess_changed": r["qnn_preprocess_changed"],
                "tensor_quant_overrides": r["tensor_quant_overrides"],
                "pruning": r["prune_report"],
                "onnx_metadata": {p.key: p.value for p in proto_meta},
                "checks": checks,
                "sha256": by_file[rel].sha256,
                "size_bytes": by_file[rel].size_bytes,
            }
            (MODELS_DIR / rel).with_suffix(".json").write_text(
                json.dumps(sidecar, indent=2) + "\n", encoding="utf-8", newline="\n")

            pr = r["prune_report"]
            line = (f"{rel}: {len(r['pruned'])} bytes, sha256 {by_file[rel].sha256[:12]}; pruned "
                    f"{pr['nodes_before']} -> {pr['nodes_after']} nodes "
                    f"{pr['removed_nodes_by_op']}")
            if checks:
                line += (f"; pruned == unpruned: {checks['pruned_vs_unpruned_bit_identical']}; "
                         f"vs FP32 top-1 {checks['vs_fp32']['top1_agreement']:.1%}")
                if "vs_b1" in checks:
                    line += (f"; vs b1 max |diff| {checks['vs_b1']['max_abs_diff']:.3g}, "
                             f"top-1 {checks['vs_b1']['top1_agreement']:.1%}")
            print(line)

    write_manifest(list(by_file.values()), tools or ox.tool_versions())
    print(f"updated {MANIFEST_PATH}")
    if failures:
        print(f"{failures} check(s) failed", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", choices=sorted(MODELS), default=sorted(MODELS),
                        help="models to quantize (default: all)")
    parser.add_argument("--batch", nargs="+", type=int, choices=BATCH_SIZES,
                        default=list(BATCH_SIZES), help="batch sizes (default: 1 4 8)")
    parser.add_argument("--normalization", choices=sorted(SOURCES), default="all",
                        help="quantize the external-normalization FP32 files, the "
                             "normalization-folded ones (scripts/fold_models.py), or both "
                             "(default: all)")
    parser.add_argument("--verify", action="store_true",
                        help="quantize again in memory and compare with models/manifest.json")
    args = parser.parse_args(argv)

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    names = list(dict.fromkeys(args.model))
    batches = sorted(set(args.batch))
    sources = SOURCES[args.normalization]
    return (verify(names, batches, sources) if args.verify
            else quantize(names, batches, sources))


if __name__ == "__main__":
    sys.exit(main())
