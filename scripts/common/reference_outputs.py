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
CPU fallback disabled) for every INT8 file.

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
from common.model_registry import MODELS_DIR, ManifestFile

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


def expected_devices(entry: ManifestFile) -> list[str]:
    return ["cpu", "npu"] if entry.precision == "int8" else ["cpu"]


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


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
