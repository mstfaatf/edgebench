"""Static INT8 QDQ quantization of the exported FP32 models, and checks on the result.

Used by ``scripts/quantize_models.py`` and ``scripts/tests/test_quantize_models.py``.
Imports onnx, onnxruntime and NumPy lazily, inside the functions.

Recipe (the default INT8 recipe; see docs/models.md):

- ONNX Runtime's QNN quantization helpers: ``qnn_preprocess_model``, then
  ``get_qnn_qdq_config(per_channel=True, weight_type=QInt8)`` and
  ``quantize_static`` with the configuration it returns;
- weights: signed int8, symmetric, per output channel; activations: uint8,
  asymmetric, per tensor; biases: int32;
- calibration: MinMax over the 200 Imagenette calibration images
  (``datasets/imagenette/calibration.tsv``, in manifest order), preprocessed
  with the model's own mean/std, one image at a time on the batch-1 model;
- batch sizes 4 and 8 reuse the batch-1 calibration ranges (ONNX Runtime's
  calibration cache), so every batch size of a model has the same scales and
  zero points; ``quant_params`` / ``quant_param_differences`` check that;
- the quantizer leaves dead QuantizeLinear/DequantizeLinear chains on
  constants (e.g. the min/max inputs of Clip nodes folded into activation
  ranges); they are pruned (``common/onnx_prune.py``) and the unpruned file is
  kept as a reference with bit-identical outputs.

Metadata: the FP32 source's ``edgebench.*`` keys that describe the model and its
input carry over (model, batch_size, weights and their source, normalization,
input range and preprocessing, export versions); variant, precision, opset and
IR version are rewritten, and the recipe (including what pruning removed) is
added as flat ``edgebench.quant.*`` keys plus the source model's path and
SHA256.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from common.manifests import CALIBRATION_MANIFEST, DATA_DIR, read_manifest, sha256_of
from common.model_registry import (
    INPUT_NAME,
    META_PREFIX,
    MODELS_DIR,
    ModelSpec,
    relative_path,
)

INT8 = "int8"
INT8_UNPRUNED = "int8_unpruned"
INT8_FOLDED = "int8_folded"
INT8_FOLDED_UNPRUNED = "int8_folded_unpruned"
# FP32 source variant -> (pruned, unpruned) INT8 variants.
INT8_VARIANTS = {"fp32": (INT8, INT8_UNPRUNED),
                 "fp32_folded": (INT8_FOLDED, INT8_FOLDED_UNPRUNED)}
IMAGENETTE_ROOT = DATA_DIR / "imagenette" / "imagenette2-320"
QUANT_PREFIX = META_PREFIX + "quant."

# Source keys that describe the model, its input or how the FP32 file was made,
# and so stay valid for the quantized file.
CARRIED_KEYS = ("model", "batch_size", "weights", "weights_source", "weights_url",
                "weights_sha256", "normalization", "input_range", "input_preprocessing",
                "exporter", "torch_version", "torchvision_version", "onnx_version",
                # timm models only
                "timm_version", "weights_hf_repo", "weights_hf_revision", "weights_file",
                # normalization-folded sources only (see common/onnx_fold.py)
                "folded_mean", "folded_std", "fold_padding", "fold_first_conv",
                # imported models only (see common/autoencoder_pipeline.py)
                "weights_path", "weights_notebook", "weights_notebook_commit",
                "weights_code_commit", "training_record_sha256", "tensorflow_version",
                "keras_version", "tf2onnx_version", "local_onnx_version", "input_shape",
                "fold_not_applicable", "cleanup")
IMAGENETTE_CALIBRATION = "Imagenette calibration split (datasets/imagenette/calibration.tsv)"


def calibration_entries() -> list:
    return read_manifest(CALIBRATION_MANIFEST)


def calibration_images_present() -> bool:
    return (IMAGENETTE_ROOT / ".extracted").is_file()


def _reader(spec: ModelSpec, entries: list, normalized: bool) -> Any:
    from onnxruntime.quantization import CalibrationDataReader

    from common.preprocess import load_image

    class ManifestReader(CalibrationDataReader):
        """Calibration images in manifest order, one per call (batch 1)."""

        def __init__(self) -> None:
            self._it = iter(entries)

        def get_next(self):
            e = next(self._it, None)
            if e is None:
                return None
            return {INPUT_NAME: load_image(IMAGENETTE_ROOT / e.path, normalized=normalized,
                                           mean=spec.mean, std=spec.std)}

    return ManifestReader()


def quant_metadata(spec: ModelSpec | None, source_meta: dict[str, str], variant: str,
                   proto: Any, source_rel: str, source_sha256: str, preprocess_changed: bool,
                   calibration_count: int, prune_report: Any = None, *,
                   calibration_dataset: str = IMAGENETTE_CALIBRATION,
                   calibration_manifest_sha256: str | None = None) -> dict[str, str]:
    """INT8 metadata; ``prune_report`` is the pruning of this file (None: not pruned).

    The calibration set defaults to the Imagenette calibration split (its manifest
    SHA256 is read from datasets/imagenette/calibration.tsv).
    """
    import onnxruntime as ort

    if calibration_manifest_sha256 is None:
        calibration_manifest_sha256 = sha256_of(CALIBRATION_MANIFEST)
    meta = {META_PREFIX + k: source_meta[META_PREFIX + k] for k in CARRIED_KEYS
            if META_PREFIX + k in source_meta}
    opset = next(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx"))
    meta.update({
        META_PREFIX + "variant": variant,
        META_PREFIX + "precision": "int8",
        META_PREFIX + "opset": str(opset),
        META_PREFIX + "ir_version": str(proto.ir_version),
        META_PREFIX + "source_model": source_rel,
        META_PREFIX + "source_sha256": source_sha256,
        META_PREFIX + "onnxruntime_version": ort.__version__,
        QUANT_PREFIX + "format": "qdq",
        QUANT_PREFIX + "method": "static",
        QUANT_PREFIX + "weight_type": "int8",
        QUANT_PREFIX + "weight_symmetric": "true",
        QUANT_PREFIX + "per_channel": "true",
        QUANT_PREFIX + "activation_type": "uint8",
        QUANT_PREFIX + "activation_symmetric": "false",
        QUANT_PREFIX + "bias_type": "int32",
        QUANT_PREFIX + "calibration_method": "MinMax",
        QUANT_PREFIX + "calibration_dataset": calibration_dataset,
        QUANT_PREFIX + "calibration_manifest_sha256": calibration_manifest_sha256,
        QUANT_PREFIX + "calibration_count": str(calibration_count),
        QUANT_PREFIX + "calibration_batch": "1 (ranges shared by all batch sizes)",
        QUANT_PREFIX + "tool": f"onnxruntime.quantization {ort.__version__} "
                               "(qnn_preprocess_model, get_qnn_qdq_config, quantize_static)",
        QUANT_PREFIX + "qnn_preprocess": "applied" if preprocess_changed else "no changes",
        QUANT_PREFIX + "pruned": "true" if variant in (INT8, INT8_FOLDED) else "false",
        # Dead QuantizeLinear/DequantizeLinear chains on constants, and all nodes,
        # removed by pruning (0 for the unpruned reference file).
        QUANT_PREFIX + "pruned_chains": str(
            prune_report.removed_nodes.get("QuantizeLinear", 0) if prune_report else 0),
        QUANT_PREFIX + "pruned_nodes": str(
            sum(prune_report.removed_nodes.values()) if prune_report else 0),
    })
    return dict(sorted(meta.items()))


def _set_metadata(proto: Any, metadata: dict[str, str]) -> None:
    import onnx

    del proto.metadata_props[:]
    onnx.helper.set_model_props(proto, metadata)


def array_reader(input_name: str, samples: Any) -> Any:
    """A CalibrationDataReader over ``samples`` (arrays), one per call, under one input name."""
    from onnxruntime.quantization import CalibrationDataReader

    class SampleReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._it = iter(samples)

        def get_next(self):
            sample = next(self._it, None)
            return None if sample is None else {input_name: sample}

    return SampleReader()


def strip_metadata(source: Path, dest: Path) -> None:
    """Copy of ``source`` without metadata_props (qnn_preprocess_model in ONNX Runtime 1.32
    fails on a model that has them: it passes the repeated proto field to dict.update)."""
    import onnx

    proto = onnx.load(str(source))
    del proto.metadata_props[:]
    onnx.save(proto, str(dest))


def quantize_qdq(source: Path, out: Path, reader: Any, calibration_cache: Path,
                 work_dir: Path) -> tuple[bool, Any]:
    """The INT8 recipe on one FP32 file (without metadata_props) -> QDQ model at ``out``.

    ``reader`` is a CalibrationDataReader, or None to reuse the ranges an
    earlier call wrote to ``calibration_cache`` (same tensor names, e.g. the
    same model at another batch size). Returns (whether qnn_preprocess_model
    changed the model, the QNN quantization config).
    """
    from onnxruntime.quantization import QuantType, quantize_static
    from onnxruntime.quantization.execution_providers.qnn import (
        get_qnn_qdq_config,
        qnn_preprocess_model,
    )

    preprocessed = work_dir / f"{out.stem}.preprocessed.onnx"
    changed = bool(qnn_preprocess_model(str(source), str(preprocessed)))
    model = preprocessed if changed else source
    config = get_qnn_qdq_config(str(model), reader, per_channel=True,
                                weight_type=QuantType.QInt8)
    # The arguments onnxruntime.quantization.quantize() passes for a
    # StaticQuantConfig, plus the calibration cache: written by the first
    # (calibrating) call, read without calibrating again by the others.
    quantize_static(
        str(model), str(out), config.calibration_data_reader,
        calibrate_method=config.calibrate_method,
        quant_format=config.quant_format,
        activation_type=config.activation_type,
        weight_type=config.weight_type,
        op_types_to_quantize=config.op_types_to_quantize,
        nodes_to_quantize=config.nodes_to_quantize,
        nodes_to_exclude=config.nodes_to_exclude,
        per_channel=config.per_channel,
        reduce_range=config.reduce_range,
        use_external_data_format=config.use_external_data_format,
        calibration_providers=config.calibration_providers,
        extra_options=config.extra_options,
        calibration_cache_path=str(calibration_cache),
    )
    return changed, config


def quantize(spec: ModelSpec, batches: list[int], calibration_count: int | None = None,
             source_variant: str = "fp32") -> dict[int, dict[str, Any]]:
    """Quantize the model's FP32 files; returns per batch the unpruned and pruned bytes.

    ``source_variant`` is ``fp32`` (external normalization: calibration images are
    normalized with the model's mean/std) or ``fp32_folded`` (normalization folded
    into the model: calibration images stay in [0, 1]). The batch-1 FP32 file is
    always calibrated (it provides the ranges for every batch size), even when
    batch 1 is not requested.
    """
    pruned_variant, unpruned_variant = INT8_VARIANTS[source_variant]
    normalized_input = source_variant == "fp32"
    import onnx

    from common.onnx_prune import prune

    entries = calibration_entries()
    if calibration_count is not None:
        entries = entries[:calibration_count]
    results: dict[int, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="edgebench-quant-") as tmp_dir:
        tmp = Path(tmp_dir)
        cache = tmp / "calibration_ranges.json"
        for batch in [1] + sorted(b for b in set(batches) if b != 1):
            source_rel = relative_path(spec.name, source_variant, batch)
            fp32 = MODELS_DIR / source_rel
            if not fp32.is_file():
                raise FileNotFoundError(f"{fp32} not found; run scripts/export_models.py "
                                        "(and scripts/fold_models.py for folded models) first")
            # The quantizer gets the FP32 graph without metadata_props; the INT8
            # metadata is written from the source file afterwards.
            stripped = tmp / f"fp32_b{batch}.onnx"
            strip_metadata(fp32, stripped)
            out = tmp / f"quantized_b{batch}.onnx"
            changed, config = quantize_qdq(
                stripped, out, _reader(spec, entries, normalized_input) if batch == 1 else None,
                cache, tmp)
            if batch not in batches:
                continue
            source_proto = onnx.load(str(fp32))
            source_meta = {p.key: p.value for p in source_proto.metadata_props}
            source_sha = sha256_of(fp32)
            unpruned = onnx.load(str(out))
            pruned = onnx.load(str(out))
            report = prune(pruned)
            for proto, variant, pruning in ((unpruned, unpruned_variant, None),
                                            (pruned, pruned_variant, report)):
                _set_metadata(proto, quant_metadata(spec, source_meta, variant, proto, source_rel,
                                                    source_sha, changed, len(entries), pruning))
                onnx.checker.check_model(proto, full_check=True)
            results[batch] = {
                "unpruned": unpruned.SerializeToString(),
                "pruned": pruned.SerializeToString(),
                "prune_report": report.as_dict(),
                "qnn_preprocess_changed": changed,
                "source": source_rel,
                "source_sha256": source_sha,
                "tensor_quant_overrides": len(config.extra_options.get("TensorQuantOverrides", {})),
            }
    return results


# --- checks ----------------------------------------------------------------------


def quant_params(proto: Any) -> dict[str, Any]:
    """Every scale and zero-point initializer used by a Q/DQ node, by name."""
    from onnx import numpy_helper

    inits = {i.name: i for i in proto.graph.initializer}
    params = {}
    for n in proto.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            for name in n.input[1:]:
                if name in inits:
                    params[name] = numpy_helper.to_array(inits[name])
    return params


def quant_param_differences(protos: dict[int, Any]) -> list[str]:
    """Scales/zero points that differ (in name, dtype or value) between batch sizes."""
    import numpy as np

    params = {b: quant_params(p) for b, p in protos.items()}
    reference_batch = min(params)
    reference = params[reference_batch]
    problems = []
    for batch, current in params.items():
        if set(current) != set(reference):
            problems.append(f"b{batch} vs b{reference_batch}: different Q/DQ parameter sets "
                            f"({len(set(current) ^ set(reference))} names differ)")
            continue
        for name, value in reference.items():
            other = current[name]
            if other.dtype != value.dtype or other.shape != value.shape or \
                    not np.array_equal(other, value):
                problems.append(f"b{batch} vs b{reference_batch}: {name} differs")
    return problems


def graph_without_metadata(proto: Any) -> bytes:
    """Serialized model with metadata_props removed, for graph comparisons."""
    import onnx

    copy = onnx.ModelProto()
    copy.CopyFrom(proto)
    del copy.metadata_props[:]
    return copy.SerializeToString()


def npu_session(path: Path, fp16: bool = False) -> Any:
    """A QNN HTP (NPU) session in strict mode: creation fails if any node would fall back to CPU.

    ``fp16`` adds ``enable_htp_fp16_precision`` = 1, for FP32 files (FP16 compute on the NPU).
    """
    import onnxruntime as ort
    import onnxruntime_qnn

    from common.ort_qnn import list_devices, qnn_npu_session_options, register_qnn, select_qnn_npu

    register_qnn()
    npu = select_qnn_npu(list_devices())
    backend = str(Path(onnxruntime_qnn.get_library_path()).parent / "QnnHtp.dll")
    provider_options = {"backend_path": backend, "htp_performance_mode": "burst"}
    if fp16:
        provider_options["enable_htp_fp16_precision"] = "1"
    options = qnn_npu_session_options(npu, provider_options, strict=True)
    return ort.InferenceSession(str(path), sess_options=options)
