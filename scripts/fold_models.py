"""Make normalization-folded FP32 variants of the exported models.

For every model (``--model``, default all) and batch size (``--batch``, default
1 4 8) this reads ``models/<model>/<model>_fp32_b<batch>.onnx`` (from
``scripts/export_models.py``; it must match ``models/manifest.json``) and writes
``models/<model>/<model>_fp32_folded_b<batch>.onnx`` plus a JSON sidecar with
the checks and a recipe sidecar (``<name>.recipe.json``): the
same model with its input normalization (the registry's mean/std) folded into
the first convolution. The convolution's zero padding becomes padding with the
per-channel mean, which keeps the fold exact (see ``common/onnx_fold.py``). The
folded file takes the [0, 1] image tensor directly, without mean/std.

Checks, each of which fails the run: ``onnx.checker`` (full check); with
ImageNetV2 downloaded, on 64 images, the folded file fed raw images gives the
same logits as the FP32 file fed normalized images (1e-4, top-1 identical),
and batch 4 and 8 match batch 1.

INT8 versions of the folded files come from
``scripts/quantize_models.py --normalization folded``.

``--verify`` folds again in memory and compares with the manifest and the files
on disk, without writing anything.

Usage:
    python scripts/fold_models.py [--model NAME ...] [--batch N ...] [--verify]
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

VARIANT = "fp32_folded"


def fold_one(name: str, batch: int, by_file: dict) -> tuple[bytes, dict, str]:
    """Folded bytes, fold info and source path for one FP32 file."""
    from common import onnx_export as ox
    from common import onnx_fold as of

    spec = MODELS[name]
    source_rel = relative_path(name, "fp32", batch)
    source = MODELS_DIR / source_rel
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found; run scripts/export_models.py first")
    data = source.read_bytes()
    digest = ox.sha256_bytes(data)
    entry = by_file.get(source_rel)
    if entry is None or entry.sha256 != digest:
        raise RuntimeError(f"{source_rel} does not match models/manifest.json; "
                           "run scripts/export_models.py --verify")
    folded, info = of.fold_file_bytes(data, spec.mean, spec.std, VARIANT, source_rel, digest)
    return folded, info, source_rel


def verify(names: list[str], batches: list[int]) -> int:
    from common import onnx_export as ox

    by_file = {f.file: f for f in read_manifest()[0]}
    failures = 0
    for name in names:
        for batch in batches:
            data, _, _ = fold_one(name, batch, by_file)
            rel = relative_path(name, VARIANT, batch)
            digest = ox.sha256_bytes(data)
            entry = by_file.get(rel)
            on_disk = MODELS_DIR / rel
            disk = ox.sha256_file(on_disk) if on_disk.is_file() else "missing"
            ok = entry is not None and entry.sha256 == digest and entry.size_bytes == len(data)
            failures += not ok
            print(f"{rel}: re-folded {digest[:12]} ({len(data)} bytes); manifest "
                  f"{entry.sha256[:12] if entry else 'missing'}; file {disk[:12]}: "
                  f"{'identical' if ok and disk == digest else 'MISMATCH' if not ok else 'manifest matches, file on disk differs'}")
    print("verify: " + ("OK" if not failures else f"{failures} file(s) differ from the manifest"))
    return 1 if failures else 0


def fold(names: list[str], batches: list[int]) -> int:
    from common import onnx_export as ox
    from common.model_recipe import record_model_file

    manifest, tools = read_manifest()
    by_file = {f.file: f for f in manifest}
    images = ox.check_images()
    if images is None:
        print("output checks skipped: run scripts/get_data.py --dataset imagenetv2 first")
    failures = 0
    for name in names:
        spec = MODELS[name]
        b1_logits = None
        for batch in batches:
            data, info, source_rel = fold_one(name, batch, by_file)
            rel = relative_path(name, VARIANT, batch)
            path = MODELS_DIR / rel
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
            by_file[rel] = record_model_file(path, rel, data)

            checks: dict = {}
            if images is not None:
                raw = images[1]
                logits = ox.run_ort(ox.ort_session(path), raw)
                external = ox.run_ort(ox.ort_session(MODELS_DIR / source_rel),
                                      ox.normalize_for(spec, raw))
                checks["vs_external"] = ox.compare_logits(logits, external)
                failures += not ox.within_tolerance(checks["vs_external"])
                if batch == 1:
                    b1_logits = logits
                elif b1_logits is None and (MODELS_DIR / relative_path(name, VARIANT, 1)).is_file():
                    b1_logits = ox.run_ort(
                        ox.ort_session(MODELS_DIR / relative_path(name, VARIANT, 1)), raw)
                if batch != 1 and b1_logits is not None:
                    checks["vs_b1"] = ox.compare_logits(logits, b1_logits)
                    failures += not ox.within_tolerance(checks["vs_b1"])
                checks["images"] = f"{len(raw)} ImageNetV2 eval-quick images"
                checks["atol"] = ox.LOGIT_ATOL

            sidecar = {
                "model": path.name,
                "model_name": name,
                "variant": VARIANT,
                "batch_size": batch,
                "source_model": {"file": source_rel, "sha256": by_file[source_rel].sha256},
                "input": {"name": "input", "range": "[0, 1]", "normalization": "folded",
                          "folded_mean": list(spec.mean), "folded_std": list(spec.std)},
                "fold": info,
                "checks": checks,
                "sha256": by_file[rel].sha256,
                "size_bytes": len(data),
            }
            path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
            line = (f"{rel}: {len(data)} bytes, sha256 {by_file[rel].sha256[:12]}; padding "
                    f"{info['padding']}"
                    + (f" {info['mean_pad']}" if info["padding"] == "mean_pad" else ""))
            if checks:
                c = checks["vs_external"]
                line += (f"; vs external max |diff| {c['max_abs_diff']:.3g}, "
                         f"top-1 {c['top1_agreement']:.1%}")
                if "vs_b1" in checks:
                    line += f"; vs b1 max |diff| {checks['vs_b1']['max_abs_diff']:.3g}"
            print(line)

    write_manifest(list(by_file.values()), tools or ox.tool_versions())
    print(f"updated {MANIFEST_PATH}")
    if failures:
        print(f"{failures} check(s) failed (tolerance {ox.LOGIT_ATOL})", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", choices=sorted(MODELS), default=sorted(MODELS),
                        help="models to fold (default: all)")
    parser.add_argument("--batch", nargs="+", type=int, choices=BATCH_SIZES,
                        default=list(BATCH_SIZES), help="batch sizes (default: 1 4 8)")
    parser.add_argument("--verify", action="store_true",
                        help="fold again in memory and compare with models/manifest.json")
    args = parser.parse_args(argv)

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    names = list(dict.fromkeys(args.model))
    batches = sorted(set(args.batch))
    return verify(names, batches) if args.verify else fold(names, batches)


if __name__ == "__main__":
    sys.exit(main())
