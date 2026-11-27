"""Check that the numbers quoted in the documentation match the committed results.

The published accuracy and anomaly-detection results
(``benchmarks/results/published/accuracy/``), the model manifest
(``models/manifest.json``), the batch-equivalence summary
(``models/batch_equivalence.json``), the NAB split manifest, the autoencoder's
committed training record, the weights pinned in
``scripts/common/model_registry.py`` and the NPU batch-size findings
(``docs/findings/npu_int8_batch_size.json``)
are the sources of truth. This script finds the tables and sentences in
README.md, docs/accuracy.md, docs/autoencoder.md, docs/models.md and
docs/npu.md that quote them and checks every number:

* result tables are matched row by row (same rows, same order, every cell);
* sentences that summarise results (ranges, maxima, counts) are recomputed;
* every SHA256 quoted in full or abbreviated (``abcd1234…ef01``) must belong to
  a committed file or pin, and file sizes must match.

A number agrees if the source value, rounded to the number of digits the
document shows, gives the same text. Standard library only; reads JSON and
Markdown, runs on any OS (it is part of the CI python job):

    python scripts/check_doc_numbers.py

Exit codes: 0 consistent, 1 differences found.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import model_registry as reg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_DIR = REPO_ROOT / "benchmarks" / "results" / "published" / "accuracy"
ANOMALY_DIR = ACCURACY_DIR / "anomaly"
SPLITS = REPO_ROOT / "datasets" / "nab-machine-temperature" / "splits.json"
TRAINING_RECORD = REPO_ROOT / "models" / "sources" / "conv1d_autoencoder" / "training_record.json"
BATCH_EQUIVALENCE = REPO_ROOT / "models" / "batch_equivalence.json"
NPU_BATCH_FINDINGS = REPO_ROOT / "docs" / "findings" / "npu_int8_batch_size.json"
DOCS = ("README.md", "docs/accuracy.md", "docs/autoencoder.md", "docs/models.md",
        "docs/npu.md")

AE = "conv1d_autoencoder"
IMAGE_MODELS = ("mobilenetv2", "resnet18", "efficientnet_lite0")
# Row order of the accuracy table, per model: (run id suffix, File, Device, Compute).
ACCURACY_ROWS = (
    ("fp32_cpu", "FP32", "CPU", "FP32"),
    ("fp32_folded_cpu", "FP32 folded", "CPU", "FP32"),
    ("int8_cpu", "INT8", "CPU", "INT8"),
    ("int8_folded_cpu", "INT8 folded", "CPU", "INT8"),
    ("int8_npu", "INT8", "NPU (strict)", "INT8"),
    ("int8_folded_npu", "INT8 folded", "NPU (strict)", "INT8"),
    ("fp32_npu_fp16", "FP32 file", "NPU (strict)", "FP16"),
)
ANOMALY_RUNS = ("fp32_cpu", "int8_cpu", "int8_npu", "fp32_npu_fp16")


# ------------------------------------------------------------------- numbers
def _parse(text: str) -> Decimal:
    t = text.strip().replace(",", "").replace("−", "-").replace("+", "")
    t = t.rstrip("%")
    try:
        return Decimal(t)
    except InvalidOperation:
        raise ValueError(f"not a number: {text!r}") from None


def agrees(text: str, value: float, scale: float = 1.0) -> bool:
    """True if ``value * scale``, rounded to the digits shown in ``text``, equals it.

    Both round-half-even (Python's formatting of the binary value) and
    round-half-up of the shortest decimal representation are accepted.
    """
    shown = _parse(text)
    v = value * scale
    t = text.strip().replace(",", "").rstrip("%")
    if "e" in t.lower():
        mantissa = t.lower().split("e")[0]
        digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
        candidates = {Decimal(f"{v:.{digits}e}")}
        exp = Decimal(f"{v:e}").adjusted()
        quant = Decimal(1).scaleb(exp - digits)
        candidates.add(Decimal(repr(v)).quantize(quant, rounding=ROUND_HALF_UP))
    else:
        digits = len(t.split(".")[1]) if "." in t else 0
        quant = Decimal(1).scaleb(-digits)
        candidates = {Decimal(f"{v:.{digits}f}"),
                      Decimal(repr(v)).quantize(quant, rounding=ROUND_HALF_UP)}
    return any(c == shown for c in candidates)


# ------------------------------------------------------------------- sources
@dataclass
class Sources:
    """Everything the documentation may quote, loaded from committed files."""
    runs: dict[str, dict]
    comparisons: dict[str, dict]
    anomaly_runs: dict[str, dict]
    anomaly_comparisons: dict[str, dict]
    manifest: dict[str, dict]
    splits: dict
    training_record: dict
    batch_equivalence: list[dict] = field(default_factory=list)
    npu_batch: dict = field(default_factory=dict)
    known_sha256: set[str] = field(default_factory=set)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256s(obj: Any) -> set[str]:
    found: set[str] = set()
    for text in re.findall(r"\b[0-9a-f]{64}\b", json.dumps(obj)):
        found.add(text)
    return found


def load_sources(root: Path = REPO_ROOT) -> Sources:
    acc = root / ACCURACY_DIR.relative_to(REPO_ROOT)
    ano = root / ANOMALY_DIR.relative_to(REPO_ROOT)
    runs = {p.stem: _json(p) for p in sorted(acc.glob("*.json")) if p.name != "comparisons.json"}
    comparisons = {c["name"]: c for c in _json(acc / "comparisons.json")["comparisons"]}
    anomaly_runs = {p.stem: _json(p) for p in sorted(ano.glob("*.json"))
                    if p.name != "comparisons.json"}
    anomaly_comparisons = {c["name"]: c for c in _json(ano / "comparisons.json")["comparisons"]}
    manifest_doc = _json(root / "models" / "manifest.json")
    manifest = {f["file"]: f for f in manifest_doc["files"]}
    splits = _json(root / SPLITS.relative_to(REPO_ROOT))
    record = _json(root / TRAINING_RECORD.relative_to(REPO_ROOT))
    equivalence = _json(root / BATCH_EQUIVALENCE.relative_to(REPO_ROOT))["entries"]
    npu_batch = _json(root / NPU_BATCH_FINDINGS.relative_to(REPO_ROOT))
    known = set()
    for obj in (runs, anomaly_runs, manifest_doc, splits, record, equivalence):
        known |= _sha256s(obj)
    for spec in reg.MODELS.values():
        known.add(spec.weights_sha256)
    for imported in reg.IMPORTED_MODELS.values():
        known |= {sha for _, sha in imported.source_sha256}
        known |= {imported.training_record_sha256, imported.sha256sums_sha256}
    return Sources(runs, comparisons, anomaly_runs, anomaly_comparisons, manifest, splits,
                   record, equivalence, npu_batch, known)


# ------------------------------------------------------------------- markdown helpers
def table(doc: str, header_start: str) -> list[list[str]] | None:
    """Body rows (cells, stripped, ``**`` removed) of the table whose header line
    starts with ``header_start``; None if there is no such table."""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(header_start):
            rows = []
            for body in lines[i + 2:]:
                if not body.startswith("|"):
                    break
                cells = re.split(r"(?<!\\)\|", body.strip())[1:-1]
                rows.append([c.strip().replace("**", "") for c in cells])
            return rows
    return None


def prose(doc: str) -> str:
    """The document with all whitespace runs collapsed, for matching sentences."""
    return re.sub(r"\s+", " ", doc)


class Checker:
    def __init__(self, docs: dict[str, str], src: Sources) -> None:
        self.docs = docs
        self.src = src
        self.problems: list[str] = []
        self.checked = 0

    # -- primitives
    def number(self, where: str, text: str, value: float, scale: float = 1.0) -> None:
        self.checked += 1
        try:
            ok = agrees(text, value, scale)
        except ValueError as exc:
            self.problems.append(f"{where}: {exc}")
            return
        if not ok:
            self.problems.append(f"{where}: document says {text!r}, source gives {value * scale!r}")

    def equal(self, where: str, shown: Any, expected: Any) -> None:
        self.checked += 1
        if shown != expected:
            self.problems.append(f"{where}: document says {shown!r}, source gives {expected!r}")

    def rows(self, doc: str, header: str, count: int) -> list[list[str]]:
        rows = table(self.docs[doc], header)
        if rows is None:
            self.problems.append(f"{doc}: table '{header}' not found")
            return []
        if len(rows) != count:
            self.problems.append(f"{doc}: table '{header}' has {len(rows)} rows, expected {count}")
        return rows

    def sentence(self, doc: str, pattern: str,
                 *values: float | int | str | Callable[[str], bool]) -> None:
        """Find ``pattern`` (whitespace-collapsed) in ``doc``; compare each group with
        the matching value (number, exact int/str, or a predicate on the text)."""
        m = re.search(pattern, prose(self.docs[doc]))
        if m is None:
            self.problems.append(f"{doc}: sentence not found: /{pattern}/")
            return
        for i, value in enumerate(values, start=1):
            where = f"{doc}: /{pattern}/ group {i}"
            text = m.group(i).rstrip(".")  # a number may end the sentence
            if callable(value):
                self.checked += 1
                if not value(text):
                    self.problems.append(f"{where}: {text!r} does not match the source")
            elif isinstance(value, str):
                self.equal(where, text, value)
            elif isinstance(value, int) and not isinstance(value, bool) and "." not in text:
                self.equal(where, int(text.replace(",", "")), value)
            else:
                self.number(where, text, float(value))

    # -- convenience on sources
    def run(self, run_id: str) -> dict:
        return self.src.runs[run_id]

    def comparison(self, model: str, label: str) -> dict:
        return self.src.comparisons[f"{model}: {label}"]

    def comparisons_like(self, label: str) -> list[dict]:
        return [self.comparison(m, label) for m in IMAGE_MODELS]


# ------------------------------------------------------------------- checks
def check_accuracy_tables(c: Checker) -> None:
    doc = "docs/accuracy.md"
    rows = c.rows(doc, "| Model | File | Device | Compute | Batch | Top-1 % (95% CI)",
                  len(IMAGE_MODELS) * len(ACCURACY_ROWS))
    expected = [(m, r) for m in IMAGE_MODELS for r in ACCURACY_ROWS]
    for row, (model, (suffix, file, device, compute)) in zip(rows, expected):
        run = c.run(f"{model}_{suffix}")
        where = f"{doc}: accuracy row {model}_{suffix}"
        c.equal(where + " model", row[0], reg.MODELS[model].display_name)
        c.equal(where + " file/device/compute", row[1:4], [file, device, compute])
        c.equal(where + " compute (run)", compute.lower(), run["execution"]["compute_precision"])
        c.equal(where + " batch", int(row[4]), run["model"]["batch_size"])
        for cell, key in ((row[5], "top1"), (row[6], "top5")):
            m = re.fullmatch(r"([\d.]+) \(([\d.]+)-([\d.]+)\)", cell)
            if m is None:
                c.problems.append(f"{where}: cannot read {cell!r}")
                continue
            metric = run["metrics"][key]
            c.number(where + f" {key}", m.group(1), metric["accuracy"], 100)
            c.number(where + f" {key} CI low", m.group(2), metric["ci95"]["low"], 100)
            c.number(where + f" {key} CI high", m.group(3), metric["ci95"]["high"], 100)

    comps = list(c.src.comparisons.values())
    rows = c.rows(doc, "| Comparison (A vs B)", len(comps))
    for row, comp in zip(rows, comps):
        model, label = comp["name"].split(": ", 1)
        where = f"{doc}: comparison row {comp['name']}"
        c.equal(where + " name", row[0], f"{reg.MODELS[model].display_name}: {label}")
        c.number(where + " delta", row[1], comp["top1_delta"], 100)
        c.number(where + " agreement", row[2], comp["top1_agreement"], 100)
        c.equal(where + " A right B wrong", int(row[3]), comp["counts"]["only_a_correct"])
        c.equal(where + " B right A wrong", int(row[4]), comp["counts"]["only_b_correct"])
        c.number(where + " p", row[5], comp["p_value"])


def _anomaly(c: Checker, suffix: str) -> dict:
    return c.src.anomaly_runs[f"{AE}_{suffix}"]


def _events(run: dict) -> str:
    events = run["metrics"]["events"]
    return f"{events['detected']} of {events['total']}"


def check_anomaly_tables(c: Checker) -> None:
    doc = "docs/accuracy.md"
    rows = c.rows(doc, "| Run | ROC-AUC | PR-AUC | F1 | Events detected |", len(ANOMALY_RUNS))
    for row, suffix in zip(rows, ANOMALY_RUNS):
        run = _anomaly(c, suffix)
        m = run["metrics"]
        where = f"{doc}: anomaly row {suffix}"
        c.number(where + " ROC-AUC", row[1], m["roc_auc"])
        c.number(where + " PR-AUC", row[2], m["pr_auc"])
        c.number(where + " F1", row[3], m["at_threshold"]["f1"])
        c.equal(where + " events", row[4], _events(run))

    doc = "docs/autoencoder.md"
    rows = c.rows(doc, "| Run | Batch | Threshold |", len(ANOMALY_RUNS))
    for row, suffix in zip(rows, ANOMALY_RUNS):
        run = _anomaly(c, suffix)
        m, at = run["metrics"], run["metrics"]["at_threshold"]
        where = f"{doc}: metrics row {suffix}"
        device = run["execution"]["device"].upper()
        precision = run["execution"]["compute_precision"].upper()
        c.equal(where + " run", row[0], f"{device} {precision}")
        c.equal(where + " batch", int(row[1]), run["model"]["batch_size"])
        c.number(where + " threshold", row[2], run["threshold"]["value"])
        for cell, value in zip(row[3:8], (m["roc_auc"], m["pr_auc"], at["precision"],
                                           at["recall"], at["f1"])):
            c.number(where, cell, value)
        c.number(where + " FPR", row[8], at["false_positive_rate"], 100)
        c.equal(where + " events", row[9], _events(run))

    ranges = _anomaly(c, "fp32_cpu")["metrics"]["events"]["ranges"]
    rows = c.rows(doc, "| Labelled window (index range)", len(ranges))
    for i, row in enumerate(rows[:len(ranges)]):
        rng = ranges[i]
        where = f"{doc}: event row {i + 1}"
        m = re.match(r"\[(\d+), (\d+)\)", row[0])
        c.equal(where + " range", (int(m.group(1)), int(m.group(2))) if m else row[0],
                (rng["start"], rng["stop"]))
        for cell, suffix in zip(row[1:], ANOMALY_RUNS):
            event = _anomaly(c, suffix)["metrics"]["events"]["ranges"][i]
            if not event["detected"]:
                c.equal(f"{where} {suffix}", cell, "not detected")
                continue
            m = re.match(r"(\d+)(?: points)? \((\d+) h(?: (\d+) min)?", cell)
            if m is None:
                c.problems.append(f"{where} {suffix}: cannot read {cell!r}")
                continue
            minutes = int(m.group(2)) * 60 + int(m.group(3) or 0)
            c.equal(f"{where} {suffix} delay", (int(m.group(1)), minutes),
                    (event["delay_points"], event["delay_minutes"]))

    comps = c.src.anomaly_comparisons
    rows = c.rows(doc, "| Comparison | Spearman", len(comps))
    for row in rows:
        comp = comps.get(row[0])
        where = f"{doc}: comparison row {row[0]}"
        if comp is None:
            c.problems.append(f"{where}: no such comparison in the results")
            continue
        c.number(where + " spearman", row[1], comp["score_spearman"])
        c.number(where + " agreement", row[2], comp["flag_agreement"], 100)
        flagged = [int(x.replace(",", "")) for x in row[3].split(" / ")]
        c.equal(where + " flagged", flagged, [comp["flagged_a"], comp["flagged_b"]])
        only = [int(x.replace(",", "")) for x in row[4].split(" / ")]
        c.equal(where + " only correct", only,
                [comp["counts"]["only_a_correct"], comp["counts"]["only_b_correct"]])
        c.number(where + " p", row[5], comp["p_value"])


def check_accuracy_sentences(c: Checker) -> None:
    same_file = c.comparisons_like("NPU INT8 vs CPU INT8") + \
        c.comparisons_like("NPU INT8 folded vs CPU INT8 folded")
    deltas = [x["top1_delta"] * 100 for x in same_file]
    c.sentence("README.md", r"the NPU matches the CPU within ([\d.]+) top-1 points",
               max(abs(d) for d in deltas))
    doc = "docs/accuracy.md"
    c.sentence(doc, r"NPU INT8 top-1 is within ([\d.]+) points of CPU INT8 "
               r"\(from ([-+\d.]+) to ([-+\d.]+)\)",
               max(abs(d) for d in deltas), min(deltas), max(deltas))
    p = [x["p_value"] for x in same_file]
    c.sentence(doc, r"no difference is statistically significant \(McNemar p = ([\d.]+) to "
               r"([\d.]+)\)", min(p), max(p))
    agree = [x["top1_agreement"] * 100 for x in same_file]
    c.sentence(doc, r"same top-1 class on ([\d.]+)-([\d.]+)% of images: their outputs",
               min(agree), max(agree))

    quant = c.comparisons_like("CPU INT8 vs CPU FP32") + \
        c.comparisons_like("CPU INT8 folded vs CPU FP32 folded")
    loss = [-x["top1_delta"] * 100 for x in quant]
    c.sentence(doc, r"CPU INT8 is ([\d.]+)-([\d.]+) points below CPU FP32", min(loss), max(loss))
    significant = [x for x in quant if x["p_value"] < 0.05]
    c.equal(f"{doc}: comparisons with p < 0.05", [x["name"] for x in significant],
            ["mobilenetv2: CPU INT8 vs CPU FP32"])
    ext = c.comparison("mobilenetv2", "CPU INT8 vs CPU FP32")
    fold = c.comparison("mobilenetv2", "CPU INT8 folded vs CPU FP32 folded")
    c.sentence(doc, r"\(([-\d.]+) points, p = ([\d.]+)\); its folded counterpart is borderline "
               r"\(([-\d.]+), p = ([\d.]+)\)",
               ext["top1_delta"] * 100, ext["p_value"], fold["top1_delta"] * 100,
               fold["p_value"])
    agree = [x["top1_agreement"] * 100 for x in quant]
    c.sentence(doc, r"INT8 and FP32 agree on (\d+)-(\d+)% of top-1", min(agree), max(agree))

    folded_fp32 = c.comparisons_like("CPU FP32 folded vs CPU FP32")
    c.equal(f"{doc}: FP32 folded vs external agreement", [x["top1_agreement"] for x in folded_fp32],
            [1.0] * len(folded_fp32))
    c.sentence(doc, r"same top-1 class on all ([\d,]+) images",
               c.run("mobilenetv2_fp32_cpu")["dataset"]["image_count"])
    folded = c.comparisons_like("CPU INT8 folded vs CPU INT8")
    c.sentence(doc, r"INT8 folded vs INT8 external differ by at most ([\d.]+) points",
               max(abs(x["top1_delta"]) * 100 for x in folded))
    differ = [(1 - x["top1_agreement"]) * 100 for x in folded]
    c.sentence(doc, r"their predictions differ on (\d+)-(\d+)% of images",
               min(differ), max(differ))
    c.equal(f"{doc}: INT8 folded vs external not significant",
            all(x["p_value"] >= 0.05 for x in folded), True)

    fp16 = c.comparisons_like("NPU FP16 vs CPU FP32")
    agree = [x["top1_agreement"] * 100 for x in fp16]
    c.sentence(doc, r"FP16 on the NPU is as accurate as FP32 on the CPU\*\*: within ([\d.]+) "
               r"points, same top-1 class on ([\d.]+)-([\d.]+)% of images",
               max(abs(x["top1_delta"]) * 100 for x in fp16), min(agree), max(agree))

    npu = c.src.anomaly_comparisons["NPU INT8 vs CPU INT8"]
    c.sentence(doc, r"flag the same test windows except (\d+) \(([\d.]+)% agreement\)",
               npu["flagged_only_a"] + npu["flagged_only_b"], npu["flag_agreement"] * 100)
    cpu_ev = _anomaly(c, "int8_cpu")["metrics"]["events"]["ranges"]
    npu_ev = _anomaly(c, "int8_npu")["metrics"]["events"]["ranges"]
    gaps = [abs(a["delay_points"] - b["delay_points"]) for a, b in zip(cpu_ev, npu_ev)]
    c.sentence(doc, r"detect the same events within (\d+) to (\d+) points", min(gaps), max(gaps))
    fp16 = c.src.anomaly_comparisons["NPU FP16 vs CPU FP32"]
    c.sentence(doc, r"NPU FP16 matches CPU FP32 on all but (\d+) windows",
               fp16["flagged_only_a"] + fp16["flagged_only_b"])


def check_autoencoder_sentences(c: Checker) -> None:
    doc = "docs/autoencoder.md"
    s = c.src.splits
    rows = c.rows(doc, "| Split | Index ranges |", 3)
    for row, name in zip(rows, ("train", "validation", "test")):
        split = s["splits"][name]
        where = f"{doc}: split row {name}"
        ranges = [[int(a), int(b)] for a, b in re.findall(r"\[(\d+), (\d+)\)", row[1])]
        c.equal(where + " ranges", ranges, split["segments"])
        c.equal(where + " points/windows/anomalous",
                [int(x.replace(",", "")) for x in row[2:5]],
                [split["points"], split["windows"], split["anomalous_windows"]])
    norm = s["normalization"]
    c.sentence(doc, r"mean ([\d.]+), std ([\d.]+)\. This is external",
               repr(norm["mean"]), repr(norm["std"]))
    c.sentence(doc, r"2013-12-02 to 2014-02-19 \(([\d,]+) points\)",
               lambda t: int(t.replace(",", "")) == s["series"]["points"])
    c.sentence(doc, r"Only ([\d,]+) points come before",
               lambda t: int(t.replace(",", "")) == s["anomaly_windows"][0]["start"])
    c.sentence(doc, r"Series SHA256 \| `([0-9a-f]{64})`", s["source"]["series"]["sha256"])
    c.sentence(doc, r"Labels SHA256 \| `([0-9a-f]{64})`", s["source"]["labels"]["sha256"])
    c.sentence(doc, r"NAB commit \| `([0-9a-f]{40})`", s["source"]["commit"])

    t = c.src.training_record["training"]
    c.sentence(doc, r"(\d+) run, best epoch (\d+) restored \((\d+) s\)",
               t["epochs_run"], t["best_epoch"], t["seconds"])
    pct = t["validation_window_mse_percentiles"]
    c.sentence(doc, r"train ([\d.e-]+), validation ([\d.e-]+); validation window MSE "
               r"p50 ([\d.e-]+), p99 ([\d.e-]+), max ([\d.e-]+)",
               t["final_train_mse"], t["final_validation_mse"], pct["p50"], pct["p99"], pct["max"])
    files = c.src.training_record["export"]["files"]
    c.sentence(doc, r"b1 `[0-9a-f]+…[0-9a-f]+` \(([\d,]+) bytes\), b4 `[0-9a-f]+…[0-9a-f]+`, "
               r"b8 `[0-9a-f]+…[0-9a-f]+` \(([\d,]+) bytes\)",
               files["b1"]["size_bytes"], files["b8"]["size_bytes"])
    c.equal(f"{doc}: b4 size equals b8 size", files["b4"]["size_bytes"], files["b8"]["size_bytes"])
    c.sentence(doc, r"max abs diff ([\d.e-]+) on the validation windows \(all ([\d,]+) at b1 and b4, "
               r"the ([\d,]+) that fill whole batches at b8\), ([\d.e-]+) on the reference windows",
               max(f["keras_vs_onnx_max_abs_validation"] for f in files.values()),
               files["b1"]["compared_windows"], files["b8"]["compared_windows"],
               max(f["keras_vs_onnx_max_abs_reference"] for f in files.values()))
    c.equal(f"{doc}: b4 compared windows", files["b4"]["compared_windows"],
            files["b1"]["compared_windows"])

    m = c.src.manifest
    sizes = {v: m[f"{AE}/{AE}_{v}_b1.onnx"]["size_bytes"]
             for v in reg.IMPORTED_VARIANTS}
    c.sentence(doc, r"FP32 file, b1 \| ([\d,]+) bytes \(with metadata\) \| ([\d,]+) bytes",
               sizes["fp32_imported"], sizes["fp32"])
    rows = c.rows(doc, "| Variant | What | Parent | b1 size |", len(reg.IMPORTED_VARIANTS))
    for row in rows:
        variant = row[0].strip("`")
        if variant in sizes:
            c.equal(f"{doc}: variant {variant} b1 size", int(row[3].replace(",", "")),
                    sizes[variant])
        else:
            c.problems.append(f"{doc}: variant table lists unknown variant {variant!r}")

    fp32 = _anomaly(c, "fp32_cpu")["metrics"]
    c.sentence(doc, r"Positives are ([\d.]+)% of the test windows, so a random score has "
               r"PR-AUC ([\d.]+)", fp32["positives"] / fp32["test_windows"] * 100,
               fp32["positives"] / fp32["test_windows"])
    c.sentence(doc, r"Its precision is high \(([\d.]+)\) but it flags (\d+)% of the labelled "
               r"windows; ROC-AUC ([\d.]+)", fp32["at_threshold"]["precision"],
               fp32["at_threshold"]["recall"] * 100, fp32["roc_auc"])
    npu = c.src.anomaly_comparisons["NPU INT8 vs CPU INT8"]
    c.sentence(doc, r"correlate only ([\d.]+) because", npu["score_spearman"])
    c.sentence(doc, r"The McNemar p of ([\d.]+) is not", npu["p_value"])
    fp16 = c.src.anomaly_comparisons["NPU FP16 vs CPU FP32"]
    c.sentence(doc, r"\(Spearman ([\d.]+), (\d+) windows", fp16["score_spearman"],
               fp16["flagged_only_a"] + fp16["flagged_only_b"])


def check_models(c: Checker) -> None:
    doc = "docs/models.md"
    m = c.src.manifest
    c.sentence(doc, r"\| `cpu` \| all (\d+) \|", len(m))
    c.sentence(doc, r"\| `npu` \| the (\d+) INT8 files",
               sum(1 for f in m.values() if f["precision"] == "int8"))
    c.sentence(doc, r"\| `npu_fp16` \| the (\d+) FP32 files",
               sum(1 for f in m.values() if f["precision"] == "fp32"))
    for spec in reg.MODELS.values():
        c.sentence(doc, rf"`{spec.weights_sha256}`, ([\d,]+) bytes",
                   lambda t, s=spec: int(t.replace(",", "")) == s.weights_size_bytes)
    # model_info.py example output: "<file> (<variant>, sha256 <12 hex>)"
    for file, variant, prefix in re.findall(r"(\S+\.onnx) \((\w+), sha256 ([0-9a-f]{12})\)",
                                            c.docs[doc]):
        entry = m.get(file)
        c.checked += 1
        if entry is None or entry["variant"] != variant or not entry["sha256"].startswith(prefix):
            c.problems.append(f"{doc}: {file} ({variant}, sha256 {prefix}) does not match "
                              f"the manifest")


# Rows of the batch-equivalence table in models.md: (model, variants, label shown).
BATCH_GROUPS = (
    [(m, ("fp32", "fp32_folded"), "FP32, FP32 folded") for m in IMAGE_MODELS]
    + [(m, ("int8", "int8_unpruned", "int8_folded", "int8_folded_unpruned"), "INT8 (all four)")
       for m in IMAGE_MODELS]
    + [(AE, ("fp32_imported",), "FP32 imported"), (AE, ("fp32",), "FP32"),
       (AE, ("int8", "int8_unpruned"), "INT8, INT8 unpruned")]
)
PROVIDER_LABEL = {("cpu", "fp32"): "CPU, FP32", ("cpu", "int8"): "CPU, INT8",
                  ("npu", "fp16"): "NPU, FP16", ("npu", "int8"): "NPU, INT8"}


def batch_equivalence_rows(entries: list[dict]) -> list[tuple[str, str, str, str, str, float]]:
    """(model, variants, provider, b4, b8, largest difference) per group and provider, in
    table order. A cell is 'same' if every file of the group at that batch size computes
    the batch-1 outputs exactly, 'differs' if none does, 'mixed' otherwise."""
    rows = []
    for model, variants, label in BATCH_GROUPS:
        group = [e for e in entries if e["model"] == model and e["variant"] in variants
                 and e["batch_size"] > 1]
        for key in sorted({(e["provider"], e["compute_precision"]) for e in group}):
            sel = [e for e in group if (e["provider"], e["compute_precision"]) == key]
            cells = []
            for batch in (4, 8):
                same = [e["bitwise_equal_to_b1"] for e in sel if e["batch_size"] == batch]
                cells.append("same" if all(same) else "differs" if not any(same) else "mixed")
            rows.append((model, label, PROVIDER_LABEL[key], cells[0], cells[1],
                         max(e["max_abs_diff"] for e in sel)))
    return rows


VARIANT_LABEL = {"fp32": "FP32", "fp32_folded": "FP32 folded", "fp32_imported": "FP32 imported",
                 "int8": "INT8", "int8_folded": "INT8 folded"}
MODEL_ORDER = (*IMAGE_MODELS, AE)


def prediction_rows(entries: list[dict]) -> list[tuple[str, str, str, int, str]]:
    """(model, variant, provider, batch, "N of M") for every entry whose bits differ
    from b1; a pruned file and its unpruned file share a row (their counts must agree)."""
    groups: dict[tuple, list[dict]] = {}
    for e in entries:
        if e["bitwise_equal_to_b1"]:
            continue
        variant = e["variant"].replace("_unpruned", "")
        key = (MODEL_ORDER.index(e["model"]), e["model"], variant,
               PROVIDER_LABEL[(e["provider"], e["compute_precision"])], e["batch_size"])
        groups.setdefault(key, []).append(e)
    rows = []
    for key in sorted(groups, key=lambda k: (k[0], k[2], k[3], k[4])):
        counts = {(e["predictions_differing"], e["predictions_compared"]) for e in groups[key]}
        differing, compared = counts.pop() if len(counts) == 1 else (None, None)
        text = "mixed" if differing is None else f"{differing:,} of {compared:,}"
        rows.append((key[1], VARIANT_LABEL[key[2]], key[3], key[4], text))
    return rows


def check_batch_equivalence(c: Checker) -> None:
    doc = "docs/models.md"
    expected = batch_equivalence_rows(c.src.batch_equivalence)
    rows = c.rows(doc, "| Model | Variants | Provider, compute | b4 vs b1 | b8 vs b1 |", len(expected))
    for shown, want in zip(rows, expected):
        where = f"{doc}: batch equivalence row {shown[:3]}"
        c.equal(where, [s.strip("`") for s in shown[:5]], list(want[:5]))
        if want[5] == 0:
            c.equal(where, shown[5], "0")
        else:
            c.number(where, shown[5], want[5])
    predicted = prediction_rows(c.src.batch_equivalence)
    rows = c.rows(doc, "| Model | Variant | Provider, compute | Batch | Predictions differing from b1 |",
                  len(predicted))
    for shown, want in zip(rows, predicted):
        c.equal(f"{doc}: batch predictions row {shown[:4]}",
                [shown[0].strip("`"), shown[1], shown[2], int(shown[3]), shown[4]], list(want))
    entries = c.src.batch_equivalence
    c.sentence(doc, r"summary lists (\d+) entries \(every model file on each of its reference "
                    r"devices\), (\d+) of them not bitwise equal, (\d+) with different predictions",
               len(entries), sum(1 for e in entries if not e["bitwise_equal_to_b1"]),
               sum(1 for e in entries if e["predictions_equal_to_b1"] is False))


MIB = 1024 * 1024
DISPLAY = {"mobilenetv2": "MobileNetV2", "resnet18": "ResNet-18",
           "efficientnet_lite0": "EfficientNet-Lite0"}
TAIL_LABEL = {"global_average_pool": "GlobalAveragePool",
              "global_average_pool_flatten": "GlobalAveragePool + Flatten",
              "tail_to_logits": "GlobalAveragePool to logits", "gemm_only": "Gemm only"}


def _differing(hist: dict[str, int]) -> str:
    """'0', or 'N of M' (elements not 0 LSB apart, of all) for an LSB histogram."""
    total = sum(hist.values())
    n = total - hist.get("0", 0)
    return "0" if n == 0 else f"{n:,} of {total:,}"


def _same_or_differs(counts: list[int], items: int) -> str:
    return ("same" if all(n == items for n in counts) else
            "differs" if not any(counts) else "mixed")


def check_npu_batch_findings(c: Checker) -> None:
    """docs/npu.md, "Batch size and INT8 outputs", against docs/findings/."""
    doc = "docs/npu.md"
    f = c.src.npu_batch
    models = f["models"]
    items = f["reference_images"]
    batches = list(models[IMAGE_MODELS[0]]["threshold"]["batches"])
    rows = c.rows(doc, "| Batch | MobileNetV2 | ResNet-18 | EfficientNet-Lite0 | Largest activation",
                  len(batches))
    for shown, batch in zip(rows, batches):
        where = f"{doc}: threshold row b{batch}"
        c.equal(where, shown[0], batch)
        for i, m in enumerate(IMAGE_MODELS):
            row = models[m]["threshold"]["batches"][batch]
            c.equal(where, shown[1 + i], f"{row['items_equal_b1_reference']} of {items}")
            c.number(where, shown[4 + i].replace(" MiB", ""),
                     row["largest_activation"]["bytes"], 1 / MIB)
    for m in IMAGE_MODELS:
        t = models[m]["threshold"]
        c.equal(f"{doc}: {m} diverging batches identical", t["diverging_batches_identical"], True)
        c.equal(f"{doc}: {m} position independent", t["b8_independent_of_position_in_batch"], True)
    c.sentence(doc, r"smallest batch size that differs is (\d+) for all three models",
               lambda t: all(models[m]["threshold"]["first_diverging_batch"] == int(t)
                             for m in IMAGE_MODELS))
    c.sentence(doc, r"VTCM is (\d+) MiB", f["vtcm_bytes"] // MIB)

    scan = f["layer_scan"]
    c.sentence(doc, r"Of the (\d+) activations of ResNet-18 \(the output of every activation "
                    r"DequantizeLinear\), compared at b1 and b(\d+), only the last one, `(\w+)`, "
                    r"differs",
               scan["cut_points"], scan["batch"], scan["first_diverging"])
    c.equal(f"{doc}: layer scan differing tensors",
            [r["tensor"] for r in scan["rows"] if r["differing"]], [scan["first_diverging"]])

    rows = c.rows(doc, "| Model | Sub-model | b2 | b4 | b6 | b7 | b8 |",
                  len(IMAGE_MODELS) * len(TAIL_LABEL))
    for shown, (m, v) in zip(rows, [(m, v) for m in IMAGE_MODELS for v in TAIL_LABEL]):
        tail = models[m]["tail"][v]
        c.equal(f"{doc}: tail row {m} {v}", shown,
                [DISPLAY[m], TAIL_LABEL[v], *(_differing(tail[b]) for b in ("2", "4", "6", "7", "8"))])
    histograms = [h for m in IMAGE_MODELS
                  for h in [models[m]["threshold"]["output_lsb_histogram_b8_minus_b1"],
                            *models[m]["tail"]["tail_to_logits"].values(),
                            *models[m]["tail"]["gemm_only"].values()]]
    c.sentence(doc, r"every differing element is (\+1) LSB",
               lambda t: t == "+1" and all(set(h) <= {"0", "1"} for h in histograms))

    rows = c.rows(doc, "| Model | Output scale | Outputs | +1 LSB | Top-1 changes |",
                  len(IMAGE_MODELS))
    for shown, m in zip(rows, IMAGE_MODELS):
        t = models[m]["threshold"]
        hist = t["output_lsb_histogram_b8_minus_b1"]
        total = sum(hist.values())
        where = f"{doc}: output LSB row {m}"
        c.equal(where, shown[0], DISPLAY[m])
        c.number(where, shown[1], t["output_scale"])
        c.equal(where, int(shown[2].replace(",", "")), total)
        count, share = shown[3].split(" (")
        c.equal(where, int(count.replace(",", "")), hist.get("1", 0))
        c.number(where, share.rstrip(")"), hist.get("1", 0) / total, 100)
        c.equal(where, int(shown[4]), t["top1_changes_b8_reference_images"])

    rows = c.rows(doc, "| Model | Batch | Mean error (LSB) | Standard deviation (LSB) |",
                  2 * len(IMAGE_MODELS))
    for shown, (m, b) in zip(rows, [(m, b) for m in IMAGE_MODELS for b in ("b1", "b8")]):
        e = models[m]["gemm_emulation"]
        r = e[b]
        where = f"{doc}: Gemm emulation row {m} {b}"
        c.equal(where, shown[:2], [DISPLAY[m], b[1:]])
        c.number(where, shown[2], r["error_vs_exact_mean_lsb"])
        c.number(where, shown[3], r["error_vs_exact_std_lsb"])
        low, high = shown[4].split(" to ")
        c.number(where, low, r["error_vs_exact_min_lsb"])
        c.number(where, high, r["error_vs_exact_max_lsb"])
        c.number(where, shown[5], r["minus_exactly_rounded_histogram"].get("0", 0)
                 / e["outputs_compared"], 100)

    configs = list(models[IMAGE_MODELS[0]]["options_items_equal_b1_reference"])
    rows = c.rows(doc, "| Option | b1 | b6 | b7 | b8 |", len(configs))
    for shown, cfg in zip(rows, configs):
        c.equal(f"{doc}: option row {cfg}", [shown[0].strip("`"), *shown[1:]], [cfg, *(
            _same_or_differs([models[m]["options_items_equal_b1_reference"][cfg][b]
                              for m in IMAGE_MODELS], items) for b in ("1", "6", "7", "8"))])

    rows = c.rows(doc, "| Model | Default b8: top-1 changes | Mode 1 b8: bitwise equal to b1 |",
                  len(IMAGE_MODELS))
    for shown, m in zip(rows, IMAGE_MODELS):
        q = models[m]["eval_quick"]
        lat = {k: v["median_ms"] for k, v in models[m]["latency"].items()}
        where = f"{doc}: mode 1 row {m}"
        c.equal(where, shown[:3], [
            DISPLAY[m],
            f"{q['default_b8']['top1_differing']} of {q['default_b8']['images']:,}",
            f"{q['finalization_1_b8']['bitwise_equal']:,} of {q['finalization_1_b8']['images']:,}"])
        for i, key in enumerate(("default_b1", "finalization_1_b1", "default_b8",
                                 "finalization_1_b8")):
            c.number(where, shown[3 + i], lat[key])
        c.number(where, shown[7].lstrip("+").rstrip("%"),
                 lat["finalization_1_b8"] / lat["default_b8"] - 1, 100)
        c.equal(f"{doc}: {m} mode 1 leaves b1 unchanged",
                q["finalization_1_b1"]["bitwise_equal"], q["finalization_1_b1"]["images"])

    diffs = f["qnn_graph_differences_b6_b7"]["differences"]
    c.sentence(doc, r"differ only in the batch dimension of (\d+) tensors",
               lambda t: int(t) == len(diffs) and all(d.endswith("/dims[0]: 6 -> 7")
                                                      for d in diffs))


def check_hashes(c: Checker) -> None:
    """Every SHA256 a document quotes, in full or abbreviated, is a known one."""
    for doc, text in c.docs.items():
        for full in re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", text):
            c.checked += 1
            if full not in c.src.known_sha256:
                c.problems.append(f"{doc}: SHA256 {full} is not in any committed file or pin")
        for head, tail in re.findall(r"(?<![0-9a-f])([0-9a-f]{8,})(?:…|\.\.\.)([0-9a-f]*)", text):
            c.checked += 1
            if not any(k.startswith(head) and k.endswith(tail) for k in c.src.known_sha256):
                c.problems.append(f"{doc}: abbreviated SHA256 {head}…{tail} matches no "
                                  f"committed file or pin")


CHECKS = (check_accuracy_tables, check_anomaly_tables, check_accuracy_sentences,
          check_autoencoder_sentences, check_models, check_batch_equivalence,
          check_npu_batch_findings, check_hashes)


def read_docs(root: Path = REPO_ROOT) -> dict[str, str]:
    return {d: (root / d).read_text(encoding="utf-8") for d in DOCS}


def check(docs: dict[str, str], src: Sources) -> Checker:
    c = Checker(docs, src)
    for fn in CHECKS:
        try:
            fn(c)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            c.problems.append(f"{fn.__name__}: cannot evaluate ({type(exc).__name__}: {exc})")
    return c


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    c = check(read_docs(), load_sources())
    for problem in c.problems:
        print(problem)
    print(f"{c.checked} numbers checked in {len(DOCS)} documents: "
          f"{'OK' if not c.problems else f'{len(c.problems)} difference(s)'}")
    return 1 if c.problems else 0


if __name__ == "__main__":
    sys.exit(main())
