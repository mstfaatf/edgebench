"""Reference outputs ("answer keys") for every model file, and their checks.

For every file in ``models/manifest.json`` this stores what Python's ONNX
Runtime computes on two fixed input sets, so that other implementations (the
C++ engine's CPU provider now, its NPU provider later) can be checked against
the same file and the same input:

- ``images`` (image classifiers): 8 ImageNetV2 eval-quick images
  (``select_images``: a seeded sample of distinct classes), preprocessed with
  ``common/preprocess.py``. A file gets the input its recipe expects:
  normalized with the model's own mean/std for external normalization, the
  [0, 1] tensor for folded files.
- ``windows`` (time-series models): the 8 reference windows of the split
  manifest (``reference_windows``: one centred in each labelled anomaly of the
  test split, four normal test windows), normalized with the manifest's mean/std.

  Batch-1 files run the 8 images or windows one at a time, batch-4 files in 2
  runs, batch-8 files in one run.
- ``splitmix64``: the input ``edgebench run`` generates and times by default
  (``--input random --seed 0``): one SplitMix64 stream filling the input
  tensor in row-major order (docs/cli.md, "Input data"), one run per file.

Layout (under ``models/references/``, not committed, like the models)::

    inputs.json                      the input sets: rule, images, files, hashes
    index.tsv                        every input and output tensor (read by the C++ tests)
    inputs/<input set>/<nn>.f32      raw little-endian float32 tensors
    outputs/<model>/<file stem>/<device>/<input set>-<run>.f32
    outputs/<model>/<file stem>/<device>.json   one record per file and device

Devices: ``cpu`` (CPU EP, ``CPU_THREADS`` intra-op threads, the same session
settings as the engine's CPU provider) for every file; ``npu`` (QNN HTP, strict:
CPU fallback disabled) for every INT8 file; ``npu_fp16`` (the same, with
``enable_htp_fp16_precision`` 1) for every FP32 file; ``npu_finalization1`` (the
``npu`` settings plus ``htp_graph_finalization_optimization_mode`` 1) for the
INT8 files of the image models, whose batch-8 outputs that option changes.

The format is the one ``tools/ort_probe/make_reference.py`` uses: raw float32
files plus a tab-separated index. This module's index, record and check code
uses the standard library only; generating tensors needs NumPy (and onnx /
ONNX Runtime in ``scripts/make_references.py``).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.manifests import EVAL_QUICK_MANIFEST, ManifestEntry, read_manifest
from common.model_registry import MODELS, MODELS_DIR, ManifestFile

REFERENCE_DIR = MODELS_DIR / "references"
INPUTS_JSON = REFERENCE_DIR / "inputs.json"
INDEX_TSV = REFERENCE_DIR / "index.tsv"
REFERENCE_VERSION = "1.0.0"

IMAGE_COUNT = 8
IMAGE_SEED = 20261124
SPLITMIX_SEED = 0
CPU_THREADS = 4  # the engine's default; the C++ match tests use the same

INDEX_HEADER = ("kind", "input_set", "model", "device", "run", "inputs", "file", "dtype",
                "shape", "sha256")
IMAGES = "images"
WINDOWS = "windows"
SPLITMIX = "splitmix64"


# --- input sets (standard library) ---------------------------------------------


def select_images(entries: list[ManifestEntry], count: int = IMAGE_COUNT,
                  seed: int = IMAGE_SEED) -> list[ManifestEntry]:
    """``count`` eval-quick images of distinct classes: a seeded sample, in sample order.

    ``random.Random(seed).sample`` over the manifest's (label, path) order; an
    image whose class is already taken is skipped (eval-quick has one image
    per class, so none is).
    """
    rng = random.Random(seed)
    ordered = sorted(entries, key=lambda e: (e.label, e.path))
    picked: list[ManifestEntry] = []
    for e in rng.sample(ordered, len(ordered)):
        if all(p.label != e.label for p in picked):
            picked.append(e)
        if len(picked) == count:
            return picked
    raise ValueError(f"fewer than {count} distinct classes in the manifest")


def image_input_set(normalization: str, mean, std) -> str:
    """Input set id of the 8 images as a file of this normalization expects them."""
    if normalization == "folded":
        return f"{IMAGES}-unit"
    digest = hashlib.sha256(json.dumps([list(mean), list(std)]).encode()).hexdigest()[:8]
    return f"{IMAGES}-normalized-{digest}"


def window_input_set(split_manifest_sha256: str) -> str:
    """Input set id of the split manifest's reference windows (normalized as it says)."""
    return f"{WINDOWS}-reference-{split_manifest_sha256[:8]}"


def splitmix_input_set(shape: list[int] | tuple[int, ...]) -> str:
    """Input set id of the engine's random input for an input tensor of ``shape``."""
    return f"{SPLITMIX}-seed{SPLITMIX_SEED}-{'x'.join(str(d) for d in shape)}"


def splitmix64_floats(seed: int, count: int):
    """The engine's random input (docs/cli.md): SplitMix64, top 24 bits scaled to [0, 1)."""
    import numpy as np

    golden = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        state = np.uint64(seed) + golden * np.arange(1, count + 1, dtype=np.uint64)
        z = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    return (z >> np.uint64(40)).astype(np.float32) * np.float32(2.0 ** -24)


# --- index and records (standard library) --------------------------------------


@dataclass(frozen=True)
class IndexRow:
    kind: str          # "input" or "output"
    input_set: str
    model: str         # model file relative to models/, "-" for inputs
    device: str        # "cpu", "npu", "-" for inputs
    run: int           # run number within the input set (0 for inputs)
    inputs: tuple[str, ...]  # output rows: the input files, concatenated in order
    file: str          # relative to models/references/
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def line(self) -> str:
        return "\t".join([self.kind, self.input_set, self.model, self.device, str(self.run),
                          ",".join(self.inputs) or "-", self.file, self.dtype,
                          ",".join(str(d) for d in self.shape), self.sha256])


def format_index(rows: list[IndexRow]) -> str:
    rows = sorted(rows, key=lambda r: (r.kind != "input", r.model, r.device, r.input_set, r.run,
                                       r.file))
    return "\n".join(["\t".join(INDEX_HEADER)] + [r.line() for r in rows]) + "\n"


def parse_index(text: str) -> list[IndexRow]:
    lines = text.splitlines()
    if not lines or tuple(lines[0].split("\t")) != INDEX_HEADER:
        raise ValueError("index.tsv: unexpected header")
    rows = []
    for n, line in enumerate(lines[1:], start=2):
        f = line.split("\t")
        if len(f) != len(INDEX_HEADER):
            raise ValueError(f"index.tsv line {n}: {len(f)} fields")
        rows.append(IndexRow(f[0], f[1], f[2], f[3], int(f[4]),
                             () if f[5] == "-" else tuple(f[5].split(",")), f[6], f[7],
                             tuple(int(d) for d in f[8].split(",")), f[9]))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def output_dir(model_rel: str, root: Path = REFERENCE_DIR) -> Path:
    """Where the outputs of one model file go: outputs/<model>/<file stem>/."""
    return root / "outputs" / model_rel.removesuffix(".onnx")


def record_path(model_rel: str, device: str, root: Path = REFERENCE_DIR) -> Path:
    return output_dir(model_rel, root) / f"{device}.json"


NPU_DEVICES = {"int8": "npu", "fp32": "npu_fp16"}  # NPU reference device per precision
FINALIZATION1 = "npu_finalization1"
# The QNN EP provider options of each NPU reference device beyond backend_path and
# htp_performance_mode (burst), exactly as passed.
NPU_DEVICE_OPTIONS: dict[str, dict[str, str]] = {
    "npu": {},
    "npu_fp16": {"enable_htp_fp16_precision": "1"},
    FINALIZATION1: {"htp_graph_finalization_optimization_mode": "1"},
}
# The provider options that can change the compiled graph (the engine's
# compile_relevant_ep_options(): all but backend_path, htp_performance_mode and
# enable_htp_fp16_precision, which the compute precision already says).
DEVICE_COMPILE_OPTIONS: dict[str, dict[str, str]] = {
    FINALIZATION1: {"htp_graph_finalization_optimization_mode": "1"},
}


def expected_devices(entry: ManifestFile) -> list[str]:
    """cpu for every file; npu (INT8 as stored) for INT8 files, npu_fp16 (FP16 compute,
    ``enable_htp_fp16_precision`` 1) for FP32 files; npu_finalization1 for the INT8
    files of the image models."""
    npu = NPU_DEVICES.get(entry.precision)
    devices = ["cpu", npu] if npu else ["cpu"]
    if entry.precision == "int8" and entry.model in MODELS:
        devices.append(FINALIZATION1)
    return devices


def check_problems(manifest: list[ManifestFile], root: Path = REFERENCE_DIR,
                   models_dir: Path = MODELS_DIR) -> list[str]:
    """Everything stale or missing in the references (empty list: all current).

    For every manifest file present on disk: a record per expected device whose
    model SHA256 equals the file's current SHA256 (otherwise the reference is
    stale), and every tensor the index lists for it exists with its recorded
    SHA256. The input set must still be the documented selection.
    """
    problems: list[str] = []
    index_path, inputs_path = root / "index.tsv", root / "inputs.json"
    if not index_path.is_file() or not inputs_path.is_file():
        return [f"{root} has no index.tsv / inputs.json; run scripts/make_references.py"]
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    expected = [e.path for e in select_images(read_manifest(EVAL_QUICK_MANIFEST))]
    if [i["path"] for i in inputs["images"]["selection"]] != expected:
        problems.append("inputs.json: the image selection is not the documented one")
    windows = inputs.get("windows")
    if windows is not None:
        from common.timeseries import SPLIT_MANIFEST, read_split_manifest

        split = read_split_manifest(SPLIT_MANIFEST)
        if windows["window_starts"] != split["reference_windows"]["window_starts"] or \
                windows["split_manifest_sha256"] != sha256_file(SPLIT_MANIFEST):
            problems.append("inputs.json: the reference windows are not the split manifest's")
    rows = parse_index(index_path.read_text(encoding="utf-8"))
    tensors = {r.file: r for r in rows}
    for r in rows:
        path = root / r.file
        if not path.is_file():
            problems.append(f"{r.file}: missing")
        elif sha256_file(path) != r.sha256:
            problems.append(f"{r.file}: SHA256 differs from index.tsv")
        for i in r.inputs:
            if i not in tensors or tensors[i].kind != "input":
                problems.append(f"{r.file}: input {i} is not an indexed input tensor")
    by_model: dict[tuple[str, str], list[IndexRow]] = {}
    for r in rows:
        if r.kind == "output":
            by_model.setdefault((r.model, r.device), []).append(r)
    for entry in manifest:
        model = models_dir / entry.file
        if not model.is_file():
            continue
        current = sha256_file(model)
        for device in expected_devices(entry):
            record = record_path(entry.file, device, root)
            if not record.is_file():
                problems.append(f"{entry.file}: no {device} reference")
                continue
            rec = json.loads(record.read_text(encoding="utf-8"))
            if rec.get("model_sha256") != current:
                problems.append(f"{entry.file}: {device} reference is stale (made from "
                                f"{str(rec.get('model_sha256'))[:12]}, file is {current[:12]})")
            outputs = {(o["input_set"], o["run"]): o["sha256"] for o in rec.get("outputs", [])}
            indexed = {(r.input_set, r.run): r.sha256 for r in by_model.get((entry.file, device), [])}
            if outputs != indexed:
                problems.append(f"{entry.file}: {device} record and index.tsv list different "
                                "outputs")
            if not any(k[0].startswith(SPLITMIX) for k in outputs) or \
                    not any(k[0].startswith((IMAGES, WINDOWS)) for k in outputs):
                problems.append(f"{entry.file}: {device} reference lacks an input set")
    return problems


# --- batch equivalence (standard library) ---------------------------------------
#
# Whether a model file at batch N computes, per item, exactly what the same
# variant computes at batch 1 on the same device (bitwise_equal_to_b1). Derived
# from the stored image / window outputs (8 items, the same inputs at every batch
# size); the SplitMix64 input does not count, because its batch items differ from
# each other. Where the bytes differ, whether any prediction changes
# (predictions_equal_to_b1) comes from common/batch_predictions.py, which runs the
# models on eval-quick or on the autoencoder's test split. The summary is
# committed (models/batch_equivalence.json) so that the engine can record, by
# model SHA256, whether a result at batch N may be compared with b1.

BATCH_EQUIVALENCE_JSON = MODELS_DIR / "batch_equivalence.json"
BATCH_EQUIVALENCE_VERSION = "1.2.0"
DEVICE_PROVIDER = {"cpu": "cpu", "npu": "npu", "npu_fp16": "npu", FINALIZATION1: "npu"}
NPU_COMPUTE_PRECISION = {"npu": "int8", "npu_fp16": "fp16", FINALIZATION1: "int8"}
# Devices whose batch-N entries also carry a whole-split confirmation on eval-quick
# (common/batch_predictions.confirm), because the 8 reference images alone are the
# only evidence that the option removes the batch-size difference.
CONFIRMED_DEVICES = (FINALIZATION1,)
PREDICTION_KEYS = ("predictions_equal_to_b1", "predictions_basis", "predictions_compared",
                   "predictions_differing")
BITWISE_PREDICTIONS = {"predictions_equal_to_b1": True, "predictions_basis": "bitwise",
                       "predictions_compared": None, "predictions_differing": None}
NO_PREDICTIONS = {"predictions_equal_to_b1": None, "predictions_basis": "not_available",
                  "predictions_compared": None, "predictions_differing": None}


def compute_precision(entry: ManifestFile, device: str) -> str:
    """What the device computes in: the stored precision on the CPU; INT8 or FP16 on the NPU."""
    return NPU_COMPUTE_PRECISION.get(device, entry.precision)


def _item_bytes(record: dict[str, Any], root: Path) -> list[bytes] | None:
    """Per-item output bytes of the image / window runs of a record, in input order."""
    runs = sorted((o for o in record.get("outputs", [])
                   if o["input_set"].startswith((IMAGES, WINDOWS))), key=lambda o: o["run"])
    if not runs:
        return None
    items: list[bytes] = []
    for o in runs:
        data = (root / o["file"]).read_bytes()
        count = o["shape"][0]
        size = len(data) // count
        items += [data[i * size:(i + 1) * size] for i in range(count)]
    return items


def _max_abs_diff(a: bytes, b: bytes) -> float:
    from array import array

    x, y = array("f"), array("f")
    x.frombytes(a)
    y.frombytes(b)
    return max((abs(p - q) for p, q in zip(x, y)), default=0.0)


def entry_key(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    """(sha256, provider, compute precision, compile options as sorted compact JSON)."""
    return (entry["sha256"], entry["provider"], entry["compute_precision"],
            json.dumps(entry["ep_compile_options"], sort_keys=True, separators=(",", ":")))


def _items(record_file: Path, root: Path) -> list[bytes] | None:
    if not record_file.is_file():
        return None
    return _item_bytes(json.loads(record_file.read_text(encoding="utf-8")), root)


def batch_equivalence(manifest: list[ManifestFile], root: Path = REFERENCE_DIR,
                      predictions: dict[tuple, dict[str, Any]] | None = None,
                      prediction_inputs: dict[str, Any] | None = None,
                      confirmations: dict[tuple, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The batch-equivalence summary of every file with references for it and its b1 sibling.

    ``predictions`` (common/batch_predictions.compute, keyed by entry_key) fills
    the prediction fields of the entries that are not bitwise equal; without one,
    an entry gets basis ``not_available``. Bitwise-equal entries are trivially
    prediction-equal (basis ``bitwise``). Each entry records the compile-relevant
    EP options of its reference device (``ep_compile_options``, {} for the EP
    defaults): the engine uses an entry only for runs with exactly those options.
    Entries of a non-default device also compare their items with the default
    device's outputs of the same file (``vs_default_options``), and batch-N entries
    of CONFIRMED_DEVICES carry ``eval_quick_confirmation`` (from
    ``confirmations``, common/batch_predictions.confirm; null if missing).
    """
    predictions = predictions or {}
    confirmations = confirmations or {}
    by_key = {(e.model, e.variant, e.batch_size): e for e in manifest}
    entries = []
    for entry in sorted(manifest, key=lambda e: (e.model, e.variant, e.batch_size)):
        b1 = by_key.get((entry.model, entry.variant, 1))
        if b1 is None:
            continue
        for device in expected_devices(entry):
            record_file, b1_file = record_path(entry.file, device, root), record_path(b1.file, device, root)
            if not record_file.is_file() or not b1_file.is_file():
                continue
            record = json.loads(record_file.read_text(encoding="utf-8"))
            b1_record = json.loads(b1_file.read_text(encoding="utf-8"))
            items, b1_items = _item_bytes(record, root), _item_bytes(b1_record, root)
            if items is None or b1_items is None or len(items) != len(b1_items):
                continue
            differing = [i for i, (a, b) in enumerate(zip(items, b1_items)) if a != b]
            e = {
                "file": entry.file,
                "sha256": record["model_sha256"],
                "model": entry.model,
                "variant": entry.variant,
                "batch_size": entry.batch_size,
                "provider": DEVICE_PROVIDER[device],
                "compute_precision": compute_precision(entry, device),
                "ep_compile_options": DEVICE_COMPILE_OPTIONS.get(device, {}),
                "reference_device": device,
                "bitwise_equal_to_b1": not differing,
                "b1_file": b1.file,
                "b1_sha256": b1_record["model_sha256"],
                "items_compared": len(items),
                "items_differing": len(differing),
                "max_abs_diff": (max(_max_abs_diff(items[i], b1_items[i]) for i in differing)
                                 if differing else 0.0),
            }
            if not differing:
                e.update(BITWISE_PREDICTIONS)
            else:
                found = predictions.get(entry_key(e))
                e.update({k: found[k] for k in PREDICTION_KEYS} if found else NO_PREDICTIONS)
            if device in DEVICE_COMPILE_OPTIONS:
                default = NPU_DEVICES[entry.precision]
                base = _items(record_path(entry.file, default, root), root)
                if base is not None and len(base) == len(items):
                    other = [i for i, (a, b) in enumerate(zip(items, base)) if a != b]
                    e["vs_default_options"] = {
                        "reference_device": default, "items_compared": len(items),
                        "items_differing": len(other),
                        "max_abs_diff": (max(_max_abs_diff(items[i], base[i]) for i in other)
                                         if other else 0.0)}
            if device in CONFIRMED_DEVICES and entry.batch_size > 1:
                e["eval_quick_confirmation"] = confirmations.get(entry_key(e))
            entries.append(e)
    return {
        "batch_equivalence_version": BATCH_EQUIVALENCE_VERSION,
        "description": "Whether each model file computes, per batch item, exactly the output its "
                       "batch-1 sibling computes on the same provider (bitwise_equal_to_b1: from "
                       "the stored reference outputs on the 8 reference images or windows, "
                       "models/references, written by scripts/make_references.py), and, where it "
                       "does not, whether any prediction changes (predictions_equal_to_b1: top-1 "
                       "class on ImageNetV2 eval-quick, or anomaly flags on the autoencoder's test "
                       "split). Keyed by model SHA256, provider, compute precision and the "
                       "compile-relevant EP options (ep_compile_options, {} for the EP defaults); "
                       "an entry applies only to runs with exactly those options.",
        "prediction_inputs": prediction_inputs,
        "entries": entries,
    }


def committed_predictions(doc: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    """The prediction fields of a summary's entries that are not bitwise equal."""
    return {entry_key(e): {k: e.get(k) for k in PREDICTION_KEYS}
            for e in doc.get("entries", [])
            if e.get("bitwise_equal_to_b1") is False
            and e.get("predictions_basis") not in (None, "not_available")}


def committed_confirmations(doc: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    """The eval-quick confirmations of a summary's entries."""
    return {entry_key(e): e["eval_quick_confirmation"] for e in doc.get("entries", [])
            if e.get("eval_quick_confirmation") is not None}


def format_batch_equivalence(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2) + "\n"


def batch_equivalence_problems(manifest: list[ManifestFile], root: Path = REFERENCE_DIR,
                               path: Path = BATCH_EQUIVALENCE_JSON,
                               prediction_inputs: dict[str, Any] | None = None) -> list[str]:
    """Differences between the committed summary and the one the references give.

    The prediction fields cannot be recomputed without running the models, so they
    are taken from the committed summary: every entry that is not bitwise equal
    must have one, and, if ``prediction_inputs`` is given, the committed summary's
    must equal it (the split manifests it was computed on are unchanged).
    """
    if not path.is_file():
        return [f"{path.name} is missing; run scripts/make_references.py --batch-equivalence"]
    committed = json.loads(path.read_text(encoding="utf-8"))
    if committed.get("batch_equivalence_version") != BATCH_EQUIVALENCE_VERSION:
        return [f"{path.name} has version {committed.get('batch_equivalence_version')!r}, not "
                f"{BATCH_EQUIVALENCE_VERSION}; run scripts/make_references.py --batch-equivalence"]
    current = batch_equivalence(manifest, root, committed_predictions(committed),
                                committed.get("prediction_inputs"),
                                committed_confirmations(committed))
    old = {entry_key(e): e for e in committed.get("entries", [])}
    new = {entry_key(e): e for e in current["entries"]}
    problems = []
    if prediction_inputs is not None and committed.get("prediction_inputs") != prediction_inputs:
        problems.append(f"{path.name}: the prediction comparisons were made on other inputs (a "
                        "split manifest changed); run scripts/make_references.py "
                        "--batch-equivalence")
    for key in sorted(set(old) | set(new)):
        e = new.get(key) or old[key]
        name = f"{e['file']} ({key[1]}, {key[2]}, {e['reference_device']})"
        if key not in new:
            problems.append(f"{path.name}: {name} has no current references")
        elif key not in old:
            problems.append(f"{path.name}: {name} is not listed")
        elif old[key] != new[key]:
            problems.append(f"{path.name}: {name} differs from the references")
        elif new[key]["predictions_basis"] == "not_available":
            problems.append(f"{path.name}: {name} is not bitwise equal to batch 1 and has no "
                            "prediction comparison")
        elif "eval_quick_confirmation" in new[key] and new[key]["eval_quick_confirmation"] is None:
            problems.append(f"{path.name}: {name} has no eval-quick confirmation")
    return problems


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
