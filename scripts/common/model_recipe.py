"""Recipe sidecars: a readable account of how a model file was made.

Every model file listed in ``models/manifest.json`` carries its recipe in its
own ONNX ``metadata_props`` (the ``edgebench.*`` keys written by export, fold
and quantize). That metadata is the source of truth. The recipe sidecar
``<model file stem>.recipe.json``, written next to the file, is a structured,
readable copy of it, built only from the file itself (its metadata, its graph
inputs and outputs, its SHA256 and size), so it can always be regenerated from
the model file alone. ``schemas/model-recipe.v<version>.schema.json`` describes
it.

A key the recipe needs that is missing from the metadata is an error naming the
key (``MetadataError``); no field is ever filled with a guess or null.

Two kinds of model share the format: exported image classifiers (weights from
torchvision or timm, image preprocessing) and imported models (weights = a
committed source file from ``models/sources/``, time-series preprocessing,
possibly an exact graph clean-up). Image files written before the
preprocessing ``kind`` key existed have no ``kind`` in their
``edgebench.input_preprocessing``; their kind is ``image``.

The builder and the checks use the standard library only; reading a model file
(``read_model_file``) imports onnx.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.model_registry import (
    IMPORTED_MODELS,
    META_PREFIX,
    REPO_ROOT,
    ManifestFile,
    MetadataError,
    manifest_entry,
    meta_value,
    root_variant,
)

RECIPE_VERSION = "1.1.0"
RECIPE_SCHEMA = REPO_ROOT / "schemas" / f"model-recipe.v{RECIPE_VERSION}.schema.json"
RECIPE_SUFFIX = ".recipe.json"

# Where the normalization of each placement is applied.
NORMALIZATION_APPLIED_BY = {
    "external": "the caller, before the model",
    "folded": "the model (folded into the first convolution)",
}


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: list[int]


@dataclass(frozen=True)
class ModelFile:
    """What a recipe is built from: everything comes from the model file."""
    file: str                  # relative to models/, forward slashes
    sha256: str
    size_bytes: int
    metadata: dict[str, str]   # all metadata_props
    inputs: list[Tensor]
    outputs: list[Tensor]


def recipe_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name.removesuffix(".onnx") + RECIPE_SUFFIX)


def _bool(meta: dict[str, str], key: str, where: str) -> bool:
    value = meta_value(meta, key, where)
    if value not in ("true", "false"):
        raise MetadataError(f"{where}: '{META_PREFIX + key}' is {value!r}, expected true or false")
    return value == "true"


def _int(meta: dict[str, str], key: str, where: str) -> int:
    value = meta_value(meta, key, where)
    try:
        return int(value)
    except ValueError:
        raise MetadataError(f"{where}: '{META_PREFIX + key}' is {value!r}, "
                            "expected an integer") from None


def _json(meta: dict[str, str], key: str, where: str) -> Any:
    value = meta_value(meta, key, where)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise MetadataError(f"{where}: '{META_PREFIX + key}' is not valid JSON") from None


def _required(d: dict, key: str, what: str, where: str) -> Any:
    if d.get(key) is None:
        raise MetadataError(f"{where}: {what} has no '{key}'")
    return d[key]


# edgebench.cleanup (JSON): what the exact graph clean-up of an imported model rewrote.
CLEANUP_KEYS = ("tool", "nodes_before", "nodes_after", "bias_adds_folded",
                "squeeze_unsqueeze_pairs_removed", "reshape_chains_collapsed")


def _series_preprocessing(meta: dict[str, str], pre: dict, entry: ManifestFile, what: str,
                          where: str) -> dict[str, Any]:
    if entry.normalization != "external":
        raise MetadataError(f"{where}: time-series input supports external normalization only")
    keys = ("dataset", "split_manifest", "split_manifest_sha256", "window_length",
            "window_stride", "layout", "dtype", "mean", "std", "std_ddof", "computed_on")
    out: dict[str, Any] = {"kind": "time_series"}
    out.update({k: _required(pre, k, what, where) for k in keys})
    out.update({
        "normalization": entry.normalization,
        "normalization_applied_by": NORMALIZATION_APPLIED_BY[entry.normalization],
        "input_range": meta_value(meta, "input_range", where),
        "fold": None,
        "fold_not_applicable": meta_value(meta, "fold_not_applicable", where),
    })
    return out


def build_recipe(mf: ModelFile) -> dict[str, Any]:
    """The recipe sidecar document of a model file."""
    meta, where = mf.metadata, mf.file
    entry = manifest_entry(mf.file, mf.sha256, mf.size_bytes, meta)

    library = meta_value(meta, "weights_source", where)
    weights: dict[str, Any] = {"library": library, "id": meta_value(meta, "weights", where)}
    local_tool_versions = None
    if library in ("torchvision", "timm"):
        weights.update(url=meta_value(meta, "weights_url", where),
                       sha256=meta_value(meta, "weights_sha256", where))
        tool_versions = {"onnx": meta_value(meta, "onnx_version", where),
                         "torch": meta_value(meta, "torch_version", where),
                         "torchvision": meta_value(meta, "torchvision_version", where)}
    elif library == "colab_bundle":
        weights.update(sha256=meta_value(meta, "weights_sha256", where),
                       path=meta_value(meta, "weights_path", where),
                       notebook=meta_value(meta, "weights_notebook", where),
                       notebook_commit=meta_value(meta, "weights_notebook_commit", where),
                       code_commit=meta_value(meta, "weights_code_commit", where),
                       training_record_sha256=meta_value(meta, "training_record_sha256", where))
        tool_versions = {"tensorflow": meta_value(meta, "tensorflow_version", where),
                         "keras": meta_value(meta, "keras_version", where),
                         "tf2onnx": meta_value(meta, "tf2onnx_version", where),
                         "onnx": meta_value(meta, "onnx_version", where)}
        local_tool_versions = {"onnx": meta_value(meta, "local_onnx_version", where)}
    else:
        raise MetadataError(f"{where}: '{META_PREFIX}weights_source' is {library!r}")
    if library == "timm":
        weights.update(hf_repo=meta_value(meta, "weights_hf_repo", where),
                       hf_revision=meta_value(meta, "weights_hf_revision", where),
                       file=meta_value(meta, "weights_file", where))
        tool_versions["timm"] = meta_value(meta, "timm_version", where)

    pre = _json(meta, "input_preprocessing", where)
    what = f"'{META_PREFIX}input_preprocessing'"
    kind = pre.get("kind", "image")
    if kind == "time_series":
        preprocessing = _series_preprocessing(meta, pre, entry, what, where)
        shape = _json(meta, "input_shape", where)
        if len(mf.inputs) != 1 or shape != mf.inputs[0].shape:
            raise MetadataError(f"{where}: '{META_PREFIX}input_shape' {shape} is not the "
                                "graph input's shape")
    elif kind != "image":
        raise MetadataError(f"{where}: {what} has kind {kind!r}")
    elif entry.normalization == "external":
        mean, std = _required(pre, "mean", what, where), _required(pre, "std", what, where)
        fold = None
    elif entry.normalization == "folded":
        mean, std = _json(meta, "folded_mean", where), _json(meta, "folded_std", where)
        fold = {"padding": meta_value(meta, "fold_padding", where),
                "first_conv": meta_value(meta, "fold_first_conv", where)}
    else:
        raise MetadataError(f"{where}: normalization {entry.normalization!r} is not supported")
    if kind == "image":
        preprocessing = {
            "kind": "image",
            "resize_shorter_side": _required(pre, "resize_shorter_side", what, where),
            "interpolation": _required(pre, "interpolation", what, where),
            "center_crop": _required(pre, "center_crop", what, where),
            "scale": _required(pre, "scale", what, where),
            "layout": _required(pre, "layout", what, where),
            "dtype": _required(pre, "dtype", what, where),
            "mean": mean,
            "std": std,
            "normalization": entry.normalization,
            "normalization_applied_by": NORMALIZATION_APPLIED_BY[entry.normalization],
            "input_range": meta_value(meta, "input_range", where),
            "fold": fold,
        }

    cleanup = None
    if META_PREFIX + "cleanup" in meta:
        c = _json(meta, "cleanup", where)
        cleanup = {k: _required(c, k, f"'{META_PREFIX}cleanup'", where) for k in CLEANUP_KEYS}

    quantization = None
    if entry.precision == "int8":
        q = "quant."
        quantization = {
            "format": meta_value(meta, q + "format", where),
            "method": meta_value(meta, q + "method", where),
            "weight_type": meta_value(meta, q + "weight_type", where),
            "weight_symmetric": _bool(meta, q + "weight_symmetric", where),
            "per_channel": _bool(meta, q + "per_channel", where),
            "activation_type": meta_value(meta, q + "activation_type", where),
            "activation_symmetric": _bool(meta, q + "activation_symmetric", where),
            "bias_type": meta_value(meta, q + "bias_type", where),
            "calibration": {
                "method": meta_value(meta, q + "calibration_method", where),
                "dataset": meta_value(meta, q + "calibration_dataset", where),
                "manifest_sha256": meta_value(meta, q + "calibration_manifest_sha256", where),
                "sample_count": _int(meta, q + "calibration_count", where),
                "batch": meta_value(meta, q + "calibration_batch", where),
            },
            "qnn_preprocess": meta_value(meta, q + "qnn_preprocess", where),
            "pruned": _bool(meta, q + "pruned", where),
            "pruned_chains": _int(meta, q + "pruned_chains", where),
            "pruned_nodes": _int(meta, q + "pruned_nodes", where),
            "quantizer": meta_value(meta, q + "tool", where),
            "onnxruntime_version": meta_value(meta, "onnxruntime_version", where),
        }
    elif entry.precision != "fp32":
        raise MetadataError(f"{where}: precision {entry.precision!r} is not supported")

    parent = None
    if entry.variant != root_variant(entry.model):
        parent = {"file": meta_value(meta, "source_model", where),
                  "sha256": meta_value(meta, "source_sha256", where)}

    return {
        "recipe_version": RECIPE_VERSION,
        "identity": {
            "file": mf.file,
            "model": entry.model,
            "variant": entry.variant,
            "precision": entry.precision,
            "normalization": entry.normalization,
            "batch_size": entry.batch_size,
            "sha256": mf.sha256,
            "size_bytes": mf.size_bytes,
        },
        "weights": weights,
        "export": {
            "exporter": meta_value(meta, "exporter", where),
            "opset": _int(meta, "opset", where),
            "ir_version": _int(meta, "ir_version", where),
            "inputs": [vars(t) for t in mf.inputs],
            "outputs": [vars(t) for t in mf.outputs],
            "tool_versions": tool_versions,
            "local_tool_versions": local_tool_versions,
        },
        "preprocessing": preprocessing,
        "graph_cleanup": cleanup,
        "quantization": quantization,
        "lineage": {"parent": parent},
    }


def format_recipe(recipe: dict[str, Any]) -> str:
    return json.dumps(recipe, indent=2) + "\n"


# --- checks (standard library) -------------------------------------------------


def consistency_problems(recipe: dict[str, Any], entry: ManifestFile | None,
                         mf: ModelFile) -> list[str]:
    """Where a sidecar, the manifest entry and the model file disagree (empty: they agree).

    ``recipe`` is the sidecar as read from disk, ``entry`` the file's manifest
    entry, ``mf`` the model file itself. The sidecar must equal the recipe
    rebuilt from the file, and the manifest entry must equal the entry derived
    from the file's metadata, hash and size.
    """
    problems = []
    try:
        expected = build_recipe(mf)
        from_meta = manifest_entry(mf.file, mf.sha256, mf.size_bytes, mf.metadata)
    except MetadataError as exc:
        return [str(exc)]
    if recipe != expected:
        diff = [k for k in expected if recipe.get(k) != expected[k]]
        diff += [k for k in recipe if k not in expected]
        problems.append(f"{mf.file}: sidecar differs from the model file in {', '.join(diff)}")
    problems += source_pin_problems(expected)
    if entry is None:
        problems.append(f"{mf.file}: not in the manifest")
    elif entry != from_meta:
        fields = [k for k in vars(from_meta) if getattr(entry, k) != getattr(from_meta, k)]
        problems.append(f"{mf.file}: manifest entry differs from the model file in "
                        f"{', '.join(fields)}")
    return problems


def source_pin_problems(recipe: dict[str, Any]) -> list[str]:
    """An imported model's weights must be the source file the registry pins, by SHA256."""
    identity, weights = recipe["identity"], recipe["weights"]
    if weights["library"] != "colab_bundle":
        return []
    where = identity["file"]
    spec = IMPORTED_MODELS.get(identity["model"])
    if spec is None:
        return [f"{where}: weights from a source file, but {identity['model']} is not an "
                "imported model"]
    batch = identity["batch_size"]
    problems = []
    if weights["path"] != spec.source_file(batch):
        problems.append(f"{where}: weights path {weights['path']}, expected {spec.source_file(batch)}")
    if weights["sha256"] != spec.source_sha256_of(batch):
        problems.append(f"{where}: weights SHA256 {weights['sha256'][:12]} is not the pinned "
                        f"source {spec.source_sha256_of(batch)[:12]}")
    if (weights["notebook_commit"], weights["code_commit"],
            weights["training_record_sha256"]) != (spec.notebook_commit, spec.code_commit,
                                                   spec.training_record_sha256):
        problems.append(f"{where}: notebook commit, code commit or training record SHA256 "
                        "differ from the registry")
    return problems


# --- model files (needs onnx) ---------------------------------------------------


def _tensors(values: Any, inits: set[str]) -> list[Tensor]:
    from onnx import helper

    out = []
    for v in values:
        if v.name in inits:
            continue
        t = v.type.tensor_type
        out.append(Tensor(v.name, helper.tensor_dtype_to_np_dtype(t.elem_type).name,
                          [d.dim_value if d.HasField("dim_value") else d.dim_param
                           for d in t.shape.dim]))
    return out


def model_file_from_bytes(rel: str, data: bytes) -> ModelFile:
    import onnx

    proto = onnx.load_from_string(data)
    inits = {i.name for i in proto.graph.initializer}
    return ModelFile(rel, hashlib.sha256(data).hexdigest(), len(data),
                     {p.key: p.value for p in proto.metadata_props},
                     _tensors(proto.graph.input, inits), _tensors(proto.graph.output, set()))


def read_model_file(path: Path, rel: str) -> ModelFile:
    return model_file_from_bytes(rel, path.read_bytes())


def schema_problems(recipe: dict[str, Any]) -> list[str]:
    """Schema violations of a recipe document (needs jsonschema)."""
    import jsonschema

    schema = json.loads(RECIPE_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in sorted(validator.iter_errors(recipe), key=lambda e: list(e.absolute_path))]


def record_model_file(path: Path, rel: str, data: bytes | None = None) -> ManifestFile:
    """Write the recipe sidecar of the model file at ``path`` and return its manifest entry.

    Used at the end of export, fold and quantize. Raises MetadataError if the
    file's metadata lacks a key, ValueError if the recipe does not match its
    schema; nothing is written in either case.
    """
    mf = model_file_from_bytes(rel, path.read_bytes() if data is None else data)
    recipe = build_recipe(mf)
    problems = schema_problems(recipe)
    if problems:
        raise ValueError(f"{rel}: recipe does not match {RECIPE_SCHEMA.name}: {problems[0]}")
    recipe_path(path).write_text(format_recipe(recipe), encoding="utf-8", newline="\n")
    return manifest_entry(rel, mf.sha256, mf.size_bytes, mf.metadata)
