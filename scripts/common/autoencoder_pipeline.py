"""Model files of the imported Conv1D autoencoder, built in memory from its committed source.

The autoencoder is trained in Colab (``notebooks/train_autoencoder.ipynb``);
training there is repeatable but not bit-reproducible across machines, so the
exported bundle itself is the source of truth. It is committed in
``models/sources/conv1d_autoencoder/`` (three FP32 ONNX files, the training
record, reference arrays and ``SHA256SUMS``) and pinned by SHA256 in
``common/model_registry.py`` (``IMPORTED_MODELS``). Everything below is derived
from it deterministically, so ``scripts/prepare_autoencoder.py --verify`` can
rebuild every file and compare it byte for byte:

- ``fp32_imported``: the source graph unchanged, plus ``edgebench.*`` metadata
  (kept as a reference, like the unpruned INT8 files);
- ``fp32``: the exactly cleaned graph (``common/onnx_cleanup.py``: bias Adds
  folded into the convolutions, per-layer Squeeze/Unsqueeze removed);
- ``int8`` / ``int8_unpruned``: the default INT8 recipe
  (``common/onnx_quantize.py``) on the cleaned graph, calibrated on the 200
  calibration windows of the split manifest at batch 1; batch 4 and 8 reuse
  the batch-1 ranges.

Normalization is external (the caller passes ``(x - mean) / std`` with the
training split's mean and std from the split manifest). There is no
normalization-folded variant: see ``FOLD_NOT_APPLICABLE``.

Imports onnx, onnxruntime and NumPy lazily, inside the functions.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from common.model_registry import (
    IMPORTED_ROOT_VARIANT,
    META_PREFIX,
    REPO_ROOT,
    ImportedModelSpec,
    relative_path,
)
from common.timeseries import SPLIT_MANIFEST

INPUT_RANGE_SERIES = ("(x - mean) / std for x in the raw series, with the mean and std "
                      "of the training split")
FOLD_NOT_APPLICABLE = (
    "not applicable: the normalization constants are statistics of this one series' "
    "training split, not a property of the input format (as ImageNet mean/std are for "
    "[0, 1] pixels), so callers normalize anyway; there is no fixed raw input range to "
    "fold to")
CLEANUP_TOOL = "scripts/common/onnx_cleanup.py (clean_conv1d_graph)"
LAYOUT = "NWC: [batch, time step, channel], one channel"
CALIBRATION_DATASET = ("NAB machine_temperature training windows, calibration selection "
                       "(datasets/nab-machine-temperature/splits.json)")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repo_path(rel: str) -> Path:
    return REPO_ROOT / rel


# --- the committed source --------------------------------------------------------


def source_problems(spec: ImportedModelSpec) -> list[str]:
    """Everything that differs from the pinned source (empty list: the source is intact).

    Checks SHA256SUMS itself, every file it lists, the training record and the
    record's own export hashes against the registry pins.
    """
    src = repo_path(spec.source_dir)
    sums = src / "SHA256SUMS"
    if not sums.is_file():
        return [f"{sums} is missing"]
    problems = []
    if sha256_bytes(sums.read_bytes()) != spec.sha256sums_sha256:
        problems.append(f"{sums.name}: SHA256 differs from the registry pin")
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        listed[name.lstrip("*")] = digest
    for name, digest in sorted(listed.items()):
        path = src / name
        if not path.is_file():
            problems.append(f"{name}: listed in SHA256SUMS but missing")
        elif sha256_bytes(path.read_bytes()) != digest:
            problems.append(f"{name}: SHA256 differs from SHA256SUMS")
    on_disk = {p.name for p in src.iterdir() if p.is_file()} - {"SHA256SUMS"}
    if on_disk != set(listed):
        problems.append(f"files not listed in SHA256SUMS: {sorted(on_disk - set(listed))}")
    if listed.get("training_record.json") != spec.training_record_sha256:
        problems.append("training_record.json: SHA256 differs from the registry pin")
    for batch, pinned in spec.source_sha256:
        name = Path(spec.source_file(batch)).name
        if listed.get(name) != pinned:
            problems.append(f"{name}: SHA256 differs from the registry pin")
    if not problems:
        record = training_record(spec)
        exported = {f["file"]: f["sha256"] for f in record["export"]["files"].values()}
        for batch, pinned in spec.source_sha256:
            if exported.get(Path(spec.source_file(batch)).name) != pinned:
                problems.append(f"training record: export hash for batch {batch} differs")
        if record["edgebench_commit"] != spec.code_commit:
            problems.append("training record: edgebench_commit differs from the registry")
    return problems


def training_record(spec: ImportedModelSpec) -> dict:
    return json.loads((repo_path(spec.source_dir) / "training_record.json").read_text(
        encoding="utf-8"))


def source_bytes(spec: ImportedModelSpec, batch: int) -> bytes:
    data = repo_path(spec.source_file(batch)).read_bytes()
    if sha256_bytes(data) != spec.source_sha256_of(batch):
        raise ValueError(f"{spec.source_file(batch)}: SHA256 differs from the registry pin")
    return data


# --- metadata --------------------------------------------------------------------


def series_preprocessing(split_manifest: dict, split_manifest_sha256: str) -> dict:
    norm = split_manifest["normalization"]
    windowing = split_manifest["windowing"]
    return {
        "kind": "time_series",
        "dataset": split_manifest["dataset"],
        "split_manifest": SPLIT_MANIFEST.relative_to(REPO_ROOT).as_posix(),
        "split_manifest_sha256": split_manifest_sha256,
        "window_length": windowing["length"],
        "window_stride": windowing["stride"],
        "layout": LAYOUT,
        "dtype": "float32",
        "formula": norm["formula"],
        "mean": norm["mean"],
        "std": norm["std"],
        "std_ddof": norm["std_ddof"],
        "computed_on": norm["computed_on"],
    }


def imported_metadata(spec: ImportedModelSpec, batch: int, proto: Any, record: dict,
                      split_manifest: dict, split_manifest_sha256: str) -> dict[str, str]:
    """``edgebench.*`` metadata of the imported source graph (variant fp32_imported)."""
    import onnx

    versions = record["versions"]
    shape = [d.dim_value for d in proto.graph.input[0].type.tensor_type.shape.dim]
    meta = {
        "model": spec.name,
        "variant": IMPORTED_ROOT_VARIANT,
        "precision": "fp32",
        "batch_size": str(batch),
        "weights_source": "colab_bundle",
        "weights": f"{spec.name} trained by {spec.notebook} at {spec.notebook_commit[:7]} "
                   f"(Colab, seed {record['reproducibility']['seed']})",
        "weights_sha256": spec.source_sha256_of(batch),
        "weights_path": spec.source_file(batch),
        "weights_notebook": spec.notebook,
        "weights_notebook_commit": spec.notebook_commit,
        "weights_code_commit": spec.code_commit,
        "training_record_sha256": spec.training_record_sha256,
        "normalization": "external",
        "input_range": INPUT_RANGE_SERIES,
        "input_preprocessing": json.dumps(series_preprocessing(split_manifest,
                                                               split_manifest_sha256),
                                          sort_keys=True),
        "input_shape": json.dumps(shape),
        "fold_not_applicable": FOLD_NOT_APPLICABLE,
        "exporter": spec.exporter,
        "opset": str(next(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx"))),
        "ir_version": str(proto.ir_version),
        "tensorflow_version": versions["tensorflow"],
        "keras_version": versions["keras"],
        "tf2onnx_version": versions["tf2onnx"],
        "onnx_version": versions["onnx"],
        "local_onnx_version": onnx.__version__,
    }
    return {META_PREFIX + k: v for k, v in sorted(meta.items())}


def set_metadata(proto: Any, metadata: dict[str, str]) -> None:
    import onnx

    del proto.metadata_props[:]
    onnx.helper.set_model_props(proto, dict(sorted(metadata.items())))


# --- building the files ----------------------------------------------------------


def build_fp32(spec: ImportedModelSpec, batch: int, split_manifest: dict,
               split_manifest_sha256: str) -> dict[str, Any]:
    """The imported and the cleaned FP32 files of one batch size, as bytes, and the report."""
    import onnx

    from common.onnx_cleanup import clean_conv1d_graph

    raw = source_bytes(spec, batch)
    record = training_record(spec)
    proto = onnx.load_from_string(raw)
    if proto.metadata_props:
        raise ValueError(f"{spec.source_file(batch)}: the source already has metadata_props")
    meta = imported_metadata(spec, batch, proto, record, split_manifest, split_manifest_sha256)

    imported = onnx.ModelProto()
    imported.CopyFrom(proto)
    set_metadata(imported, meta)
    onnx.checker.check_model(imported, full_check=True)
    imported_data = imported.SerializeToString()

    cleaned, report = clean_conv1d_graph(proto)
    cleaned_meta = dict(meta)
    cleaned_meta.update({
        META_PREFIX + "variant": "fp32",
        META_PREFIX + "source_model": relative_path(spec.name, IMPORTED_ROOT_VARIANT, batch),
        META_PREFIX + "source_sha256": sha256_bytes(imported_data),
        META_PREFIX + "cleanup": json.dumps({
            "tool": CLEANUP_TOOL,
            "nodes_before": report.nodes_before,
            "nodes_after": report.nodes_after,
            "bias_adds_folded": report.bias_adds_folded,
            "squeeze_unsqueeze_pairs_removed": report.squeeze_unsqueeze_pairs_removed,
            "reshape_chains_collapsed": report.reshape_chains_collapsed,
        }, sort_keys=True),
    })
    set_metadata(cleaned, cleaned_meta)
    onnx.checker.check_model(cleaned, full_check=True)
    return {"raw": raw, "imported": imported_data, "fp32": cleaned.SerializeToString(),
            "cleanup": report.as_dict()}


def calibration_windows(windows: Any) -> list[Any]:
    """The calibration windows ``[n, length]`` as batch-1 model inputs ``[1, length, 1]``."""
    return [windows[i:i + 1, :, None] for i in range(len(windows))]


def quantize_files(fp32_by_batch: dict[int, bytes], calibration: list[Any],
                   split_manifest_sha256: str, source_variant: str = "fp32",
                   spec_name: str = "") -> dict[int, dict[str, Any]]:
    """The INT8 recipe on FP32 files (bytes by batch size; batch 1 required).

    Batch 1 is calibrated on ``calibration``; batch 4 and 8 reuse its ranges.
    Returns per batch the pruned and unpruned INT8 bytes and the prune report.
    ``source_variant`` names the variant the files are made from (``fp32``; the
    raw-graph comparison passes ``fp32_imported``, whose result is not shipped).
    """
    import onnx

    from common import onnx_quantize as oq
    from common.onnx_prune import prune

    if 1 not in fp32_by_batch:
        raise ValueError("the batch-1 file is needed: it provides the calibration ranges")
    results: dict[int, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="edgebench-ae-quant-") as tmp_dir:
        tmp = Path(tmp_dir)
        cache = tmp / "calibration_ranges.json"
        for batch in sorted(fp32_by_batch):
            source = onnx.load_from_string(fp32_by_batch[batch])
            source_meta = {p.key: p.value for p in source.metadata_props}
            stripped = tmp / f"fp32_b{batch}.onnx"
            bare = onnx.ModelProto()
            bare.CopyFrom(source)
            del bare.metadata_props[:]
            onnx.save(bare, str(stripped))
            out = tmp / f"int8_b{batch}.onnx"
            reader = oq.array_reader("input", calibration) if batch == 1 else None
            changed, _ = oq.quantize_qdq(stripped, out, reader, cache, tmp)
            unpruned, pruned = onnx.load(str(out)), onnx.load(str(out))
            report = prune(pruned)
            source_rel = relative_path(spec_name or source_meta[META_PREFIX + "model"],
                                       source_variant, batch)
            for proto, variant, pruning in ((unpruned, oq.INT8_UNPRUNED, None),
                                            (pruned, oq.INT8, report)):
                meta = oq.quant_metadata(
                    None, source_meta, variant, proto, source_rel,
                    sha256_bytes(fp32_by_batch[batch]), changed, len(calibration), pruning,
                    calibration_dataset=CALIBRATION_DATASET,
                    calibration_manifest_sha256=split_manifest_sha256)
                set_metadata(proto, meta)
                onnx.checker.check_model(proto, full_check=True)
            results[batch] = {"pruned": pruned.SerializeToString(),
                              "unpruned": unpruned.SerializeToString(),
                              "prune_report": report.as_dict(),
                              "qnn_preprocess_changed": changed}
    return results


def build_all(spec: ImportedModelSpec, calibration: list[Any], split_manifest: dict,
              split_manifest_sha256: str, batches: tuple[int, ...] = (1, 4, 8),
              raw_int8: bool = False) -> dict[str, Any]:
    """Every file of the model: {"files": {relative path: bytes}, "reports": ...}.

    With ``raw_int8`` the INT8 recipe also runs on the imported (uncleaned) graph,
    for comparison only; those bytes are returned under ``"raw_int8"``, not in files.
    """
    files: dict[str, bytes] = {}
    reports: dict[str, Any] = {"cleanup": {}, "prune": {}, "qnn_preprocess_changed": {}}
    fp32, imported = {}, {}
    for batch in batches:
        built = build_fp32(spec, batch, split_manifest, split_manifest_sha256)
        files[relative_path(spec.name, IMPORTED_ROOT_VARIANT, batch)] = built["imported"]
        files[relative_path(spec.name, "fp32", batch)] = built["fp32"]
        reports["cleanup"][batch] = built["cleanup"]
        fp32[batch], imported[batch] = built["fp32"], built["imported"]
    for batch, r in quantize_files(fp32, calibration, split_manifest_sha256,
                                   spec_name=spec.name).items():
        files[relative_path(spec.name, "int8", batch)] = r["pruned"]
        files[relative_path(spec.name, "int8_unpruned", batch)] = r["unpruned"]
        reports["prune"][batch] = r["prune_report"]
        reports["qnn_preprocess_changed"][batch] = r["qnn_preprocess_changed"]
    out: dict[str, Any] = {"files": files, "reports": reports}
    if raw_int8:
        # Batch 1 only: tf2onnx's per-batch tensor names keep the uncleaned graphs of
        # other batch sizes from reusing the batch-1 calibration ranges.
        out["raw_int8"] = quantize_files({1: imported[1]}, calibration, split_manifest_sha256,
                                         source_variant=IMPORTED_ROOT_VARIANT,
                                         spec_name=spec.name)
    return out
