"""Time-series data for the anomaly-detection autoencoder: NAB series, windows and splits.

The series is ``realKnownCause/machine_temperature_system_failure.csv`` from the
Numenta Anomaly Benchmark (NAB, https://github.com/numenta/NAB, MIT license),
with its labelled anomaly windows from ``labels/combined_windows.json``, both
pinned to one NAB commit and by SHA256. ``scripts/get_data.py --dataset nab``
downloads them; nothing from NAB is committed.

Everything that decides *which* values a model sees is plain Python (no NumPy),
so it runs and is tested anywhere, including the CI job without NumPy, and
gives the same bytes on every platform:

- the series is used in file order (one point every 5 minutes; the file repeats
  one hour of timestamps, a clock correction, which is kept as recorded);
- a point is anomalous if its timestamp lies inside a labelled window (both
  ends inclusive); every window must map to one contiguous index range;
- splits are contiguous index ranges in time order (``plan_splits``): training
  and validation come from the longest anomaly-free stretch (training first),
  the test split is everything else and holds every labelled anomaly;
- normalization is ``(x - mean) / std`` with mean and (population) std of the
  training points only, computed with ``math.fsum`` (correctly rounded);
- windows are ``WINDOW_LENGTH`` consecutive points taken every ``WINDOW_STRIDE``
  points and never cross a segment boundary;
- the committed split manifest (``SPLIT_MANIFEST``) records all of it, with the
  SHA256 of the source files, of the raw values and of the normalized float32
  series that every window is cut from.

NumPy is imported only by ``normalized_array`` and ``windows_array``, which
build the arrays a model is fed; they produce exactly the bytes hashed here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
NAB_DIR = REPO_ROOT / "data" / "nab"
SPLIT_MANIFEST = REPO_ROOT / "datasets" / "nab-machine-temperature" / "splits.json"
MANIFEST_VERSION = "1.0.0"
DATASET = "nab-machine-temperature"

NAB_REPOSITORY = "https://github.com/numenta/NAB"
NAB_COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"
NAB_LICENSE = "MIT"
SERIES_PATH = "data/realKnownCause/machine_temperature_system_failure.csv"
SERIES_SHA256 = "92bf5b87fc7f9bba8ca0b7ec63ccaac8cb4a1371a258e8c29a10ae9c018d82a4"
SERIES_SIZE = 732_223
LABELS_PATH = "labels/combined_windows.json"
LABELS_SHA256 = "1e1fbc4601321aad8d0f8b3784c8134299379f68f6c1f7777565f8ffd57ab6b1"
LABELS_SIZE = 15_359
LABELS_KEY = "realKnownCause/machine_temperature_system_failure.csv"
SAMPLING_INTERVAL_S = 300

# Windowing and split parameters (reasons in docs/autoencoder.md).
WINDOW_LENGTH = 128  # 10 h 40 min at 5-minute sampling
WINDOW_STRIDE = 1  # a score for every time step; neighbouring windows overlap
LEAD_IN = 576  # 2 days of normal data kept in the test split before its first anomaly
TRAIN_FRACTION = 0.75  # of the anomaly-free stretch (minus the lead-in); the rest validates
CALIBRATION_COUNT = 200  # INT8 calibration windows, evenly spaced over the training windows
REFERENCE_NORMAL = 4  # fixed test windows with no labelled point...
# ...plus one window centred inside every labelled anomaly of the test split.


def nab_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/numenta/NAB/{NAB_COMMIT}/{path}"


def local_path(path: str) -> Path:
    """Where ``get_data.py`` keeps a NAB file (``data/nab/<path in the NAB repo>``)."""
    return NAB_DIR / path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- reading ---------------------------------------------------------------------


@dataclass(frozen=True)
class Series:
    timestamps: tuple[str, ...]
    values: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.values)


def parse_series(text: str) -> Series:
    """A NAB data file: header ``timestamp,value``, one point per line."""
    rows = list(csv.reader(text.splitlines()))
    if not rows or rows[0] != ["timestamp", "value"]:
        raise ValueError(f"expected header 'timestamp,value', got {rows[0] if rows else None}")
    timestamps, values = [], []
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            raise ValueError(f"line {line}: expected 2 fields, got {len(row)}")
        datetime.fromisoformat(row[0])  # raises on a malformed timestamp
        value = float(row[1])
        if not math.isfinite(value):
            raise ValueError(f"line {line}: value {row[1]!r} is not finite")
        timestamps.append(row[0])
        values.append(value)
    if not values:
        raise ValueError("the series has no points")
    return Series(tuple(timestamps), tuple(values))


def parse_label_windows(text: str, key: str = LABELS_KEY) -> list[tuple[str, str]]:
    """The labelled anomaly windows of one data file from NAB's ``combined_windows.json``."""
    windows = json.loads(text)
    if key not in windows:
        raise ValueError(f"no label windows for {key}")
    out = []
    for w in windows[key]:
        if len(w) != 2:
            raise ValueError(f"label window {w!r} is not a [start, end] pair")
        start, end = w
        if datetime.fromisoformat(start) > datetime.fromisoformat(end):
            raise ValueError(f"label window {w!r} ends before it starts")
        out.append((start, end))
    return out


def irregular_steps(series: Series, interval_s: int = SAMPLING_INTERVAL_S) -> list[dict]:
    """Consecutive points whose timestamps are not ``interval_s`` apart."""
    ts = [datetime.fromisoformat(t) for t in series.timestamps]
    return [{"index": i + 1, "from": series.timestamps[i], "to": series.timestamps[i + 1],
             "step_s": int((ts[i + 1] - ts[i]).total_seconds())}
            for i in range(len(ts) - 1) if (ts[i + 1] - ts[i]).total_seconds() != interval_s]


# --- labels and splits -----------------------------------------------------------

Segment = tuple[int, int]  # half-open index range [start, stop)


def label_ranges(timestamps: Sequence[str], windows: Iterable[tuple[str, str]]) -> list[Segment]:
    """Index range of every label window (points with start <= timestamp <= end), sorted.

    Raises if a window covers no point, covers a non-contiguous set of points
    (possible where timestamps repeat) or overlaps another window.
    """
    ts = [datetime.fromisoformat(t) for t in timestamps]
    ranges = []
    for start, end in windows:
        lo, hi = datetime.fromisoformat(start), datetime.fromisoformat(end)
        inside = [i for i, t in enumerate(ts) if lo <= t <= hi]
        if not inside:
            raise ValueError(f"label window {start} .. {end} covers no point")
        if inside[-1] - inside[0] + 1 != len(inside):
            raise ValueError(f"label window {start} .. {end} covers non-contiguous points")
        ranges.append((inside[0], inside[-1] + 1))
    ranges.sort()
    for (_, stop), (start, _) in zip(ranges, ranges[1:]):
        if start < stop:
            raise ValueError("label windows overlap")
    return ranges


def anomaly_mask(n: int, ranges: Iterable[Segment]) -> list[bool]:
    mask = [False] * n
    for start, stop in ranges:
        mask[start:stop] = [True] * (stop - start)
    return mask


@dataclass(frozen=True)
class SplitPlan:
    train: tuple[Segment, ...]
    validation: tuple[Segment, ...]
    test: tuple[Segment, ...]

    def segments(self, split: str) -> tuple[Segment, ...]:
        return {"train": self.train, "validation": self.validation, "test": self.test}[split]


SPLITS = ("train", "validation", "test")


def plan_splits(n: int, anomalies: Sequence[Segment], *, window: int = WINDOW_LENGTH,
                lead_in: int = LEAD_IN, train_fraction: float = TRAIN_FRACTION) -> SplitPlan:
    """Time-ordered splits of a series of ``n`` points with the given anomaly ranges.

    Training and validation are the longest anomaly-free stretch (the earliest
    one on a tie), training first, minus ``lead_in`` points at its end when an
    anomaly follows (those stay with the test split, so it has normal windows
    before that anomaly). The test split is every other point: up to two
    segments, before and after the stretch. Raises if a split would be shorter
    than one window or the test split would hold no anomaly.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not anomalies:
        raise ValueError("the series has no labelled anomaly, so there is nothing to test")
    bounds = [0] + [b for r in sorted(anomalies) for b in r] + [n]
    runs = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds), 2)
            if bounds[i + 1] > bounds[i]]
    start, stop = max(runs, key=lambda r: (r[1] - r[0], -r[0]))
    usable_stop = stop - lead_in if stop < n else stop
    split_at = start + int((usable_stop - start) * train_fraction)
    train, validation = (start, split_at), (split_at, usable_stop)
    test = tuple(s for s in ((0, start), (usable_stop, n)) if s[1] > s[0])
    for name, segs in (("train", (train,)), ("validation", (validation,))):
        if segs[0][1] - segs[0][0] < window:
            raise ValueError(f"the {name} split is shorter than one window ({window} points)")
    for seg in test:
        if seg[1] - seg[0] < window:
            raise ValueError(f"test segment {seg} is shorter than one window ({window} points)")
    if not any(s0 <= a and b <= s1 for a, b in anomalies for s0, s1 in test):
        raise ValueError("no anomaly lies inside the test split")
    return SplitPlan((train,), (validation,), test)


def window_starts(segments: Iterable[Segment], window: int = WINDOW_LENGTH,
                  stride: int = WINDOW_STRIDE) -> list[int]:
    """Start index of every window that fits inside one of the segments."""
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be positive")
    return [s for start, stop in segments for s in range(start, stop - window + 1, stride)]


def window_labels(starts: Iterable[int], mask: Sequence[bool],
                  window: int = WINDOW_LENGTH) -> list[bool]:
    """A window is anomalous if any of its points is labelled."""
    prefix = [0]
    for m in mask:
        prefix.append(prefix[-1] + int(m))
    return [prefix[s + window] - prefix[s] > 0 for s in starts]


def evenly_spaced(items: Sequence[int], count: int) -> list[int]:
    """``count`` items spread evenly from the first to the last (all if fewer)."""
    n = len(items)
    if count >= n:
        return list(items)
    if count == 1:
        return [items[0]]
    return [items[i * (n - 1) // (count - 1)] for i in range(count)]


def reference_starts(plan: SplitPlan, anomalies: Sequence[Segment], mask: Sequence[bool],
                     window: int = WINDOW_LENGTH, normal: int = REFERENCE_NORMAL) -> list[int]:
    """Fixed test windows for reference outputs: one centred in each test anomaly plus
    ``normal`` evenly spaced windows without a labelled point."""
    starts = window_starts(plan.test, window)
    labels = window_labels(starts, mask, window)
    clean = [s for s, bad in zip(starts, labels) if not bad]
    inside = [a + (b - a - window) // 2 for a, b in anomalies
              if b - a >= window and any(s0 <= a and b <= s1 for s0, s1 in plan.test)]
    return sorted(set(evenly_spaced(clean, normal)) | set(inside))


# --- normalization and bytes -----------------------------------------------------


def normalization(values: Sequence[float], segments: Iterable[Segment]) -> tuple[float, float]:
    """Mean and population standard deviation of the points in ``segments``."""
    points = [v for start, stop in segments for v in values[start:stop]]
    if len(points) < 2:
        raise ValueError("need at least two points")
    mean = math.fsum(points) / len(points)
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in points) / len(points))
    if std == 0.0:
        raise ValueError("the training points are constant")
    return mean, std


def normalized_float32_bytes(values: Sequence[float], mean: float, std: float) -> bytes:
    """``(x - mean) / std`` in float64, rounded to float32, little-endian."""
    return struct.pack(f"<{len(values)}f", *((v - mean) / std for v in values))


def float64_bytes(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}d", *values)


def normalized_array(values: Sequence[float], mean: float, std: float) -> Any:
    """The normalized series as a NumPy float32 array (same bytes as the function above)."""
    import numpy as np

    return ((np.asarray(values, dtype=np.float64) - mean) / std).astype(np.float32)


def windows_array(series32: Any, starts: Sequence[int], window: int = WINDOW_LENGTH) -> Any:
    """Windows ``[len(starts), window]`` (float32 copies) cut from a normalized series."""
    import numpy as np

    series32 = np.asarray(series32, dtype=np.float32)
    idx = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(window, dtype=np.int64)
    return np.ascontiguousarray(series32[idx])


# --- the split manifest ----------------------------------------------------------


def build_manifest(series: Series, label_windows: Sequence[tuple[str, str]],
                   series_sha256: str, labels_sha256: str, *, window: int = WINDOW_LENGTH,
                   stride: int = WINDOW_STRIDE, lead_in: int = LEAD_IN,
                   train_fraction: float = TRAIN_FRACTION,
                   calibration_count: int = CALIBRATION_COUNT) -> dict:
    """Everything needed to rebuild (and check) the windows of every split."""
    n = len(series)
    anomalies = label_ranges(series.timestamps, label_windows)
    mask = anomaly_mask(n, anomalies)
    plan = plan_splits(n, anomalies, window=window, lead_in=lead_in,
                       train_fraction=train_fraction)
    mean, std = normalization(series.values, plan.train)
    splits = {}
    for name in SPLITS:
        segs = plan.segments(name)
        starts = window_starts(segs, window, stride)
        labels = window_labels(starts, mask, window)
        splits[name] = {
            "segments": [list(s) for s in segs],
            "points": sum(b - a for a, b in segs),
            "anomalous_points": sum(sum(mask[a:b]) for a, b in segs),
            "windows": len(starts),
            "anomalous_windows": sum(labels),
        }
    for name in ("train", "validation"):
        if splits[name]["anomalous_points"]:
            raise ValueError(f"the {name} split contains labelled points")
    train_starts = window_starts(plan.train, window, stride)
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": DATASET,
        "description": "Windows and time-ordered splits of NAB machine_temperature_system_failure "
                       "for the anomaly-detection autoencoder (see docs/autoencoder.md).",
        "source": {
            "repository": NAB_REPOSITORY,
            "commit": NAB_COMMIT,
            "license": NAB_LICENSE,
            "series": {"path": SERIES_PATH, "sha256": series_sha256},
            "labels": {"path": LABELS_PATH, "key": LABELS_KEY, "sha256": labels_sha256},
        },
        "series": {
            "points": n,
            "first_timestamp": series.timestamps[0],
            "last_timestamp": series.timestamps[-1],
            "sampling_interval_s": SAMPLING_INTERVAL_S,
            "order": "file order",
            "irregular_steps": irregular_steps(series),
            "values_float64_sha256": hashlib.sha256(float64_bytes(series.values)).hexdigest(),
        },
        "anomaly_windows": [
            {"start_timestamp": w[0], "end_timestamp": w[1], "start": r[0], "stop": r[1]}
            for w, r in zip(sorted(label_windows, key=lambda w: datetime.fromisoformat(w[0])),
                            anomalies)
        ],
        "windowing": {"length": window, "stride": stride,
                      "window_is_anomalous": "any point inside a labelled window"},
        "split_rule": {"lead_in": lead_in, "train_fraction": train_fraction},
        "normalization": {
            "computed_on": "train",
            "formula": "(x - mean) / std",
            "mean": mean,
            "std": std,
            "std_ddof": 0,
            "normalized_float32_sha256":
                hashlib.sha256(normalized_float32_bytes(series.values, mean, std)).hexdigest(),
        },
        "splits": splits,
        "calibration": {
            "split": "train",
            "count": min(calibration_count, len(train_starts)),
            "selection": "evenly spaced over the training windows",
            "window_starts": evenly_spaced(train_starts, calibration_count),
        },
        "reference_windows": {
            "split": "test",
            "window_starts": reference_starts(plan, anomalies, mask, window),
        },
    }


def format_manifest(manifest: dict) -> str:
    return json.dumps(manifest, indent=2) + "\n"


def read_split_manifest(path: Path = SPLIT_MANIFEST) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"{path}: unsupported manifest_version "
                         f"{manifest.get('manifest_version')!r} (expected {MANIFEST_VERSION})")
    return manifest


def write_split_manifest(manifest: dict, path: Path = SPLIT_MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_manifest(manifest), encoding="utf-8", newline="\n")


def manifest_differences(expected: dict, actual: dict, where: str = "") -> list[str]:
    """Paths (``a/b/c``) where two manifests differ."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                out.append(f"{where}{key} (only in one)")
            else:
                out += manifest_differences(expected[key], actual[key], f"{where}{key}/")
        return out
    return [] if expected == actual else [where.rstrip("/") or "(root)"]


def load_nab(directory: Path = NAB_DIR) -> tuple[Series, list[tuple[str, str]], str, str]:
    """The pinned series and label windows from ``directory``, after checking their SHA256."""
    series_path, labels_path = directory / SERIES_PATH, directory / LABELS_PATH
    for path, expected in ((series_path, SERIES_SHA256), (labels_path, LABELS_SHA256)):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run: python scripts/get_data.py "
                                    "--dataset nab")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"SHA256 mismatch for {path}: expected {expected}, got {digest}")
    series = parse_series(series_path.read_text(encoding="utf-8"))
    labels = parse_label_windows(labels_path.read_text(encoding="utf-8"))
    return series, labels, SERIES_SHA256, LABELS_SHA256


def nab_present(directory: Path = NAB_DIR) -> bool:
    return (directory / SERIES_PATH).is_file() and (directory / LABELS_PATH).is_file()


@dataclass(frozen=True)
class SplitWindows:
    """The windows of every split, as NumPy arrays ``[N, length]`` (float32)."""
    manifest: dict
    normalized: Any  # float32 [points]
    windows: dict  # split -> float32 [N, length]
    labels: dict  # split -> bool [N]
    starts: dict  # split -> list[int]
    calibration: Any  # float32 [count, length]
    reference: Any  # float32 [count, length]


def load_split_windows(directory: Path = NAB_DIR, manifest_path: Path = SPLIT_MANIFEST) -> SplitWindows:
    """Rebuild every split from the NAB files and check it against the committed manifest.

    Raises if anything differs (files, normalization constants, split ranges,
    hashes), so a caller on another machine gets exactly these windows or none.
    """
    import numpy as np

    series, labels, series_sha, labels_sha = load_nab(directory)
    manifest = read_split_manifest(manifest_path)
    window, stride = manifest["windowing"]["length"], manifest["windowing"]["stride"]
    rule = manifest["split_rule"]
    rebuilt = build_manifest(series, labels, series_sha, labels_sha, window=window,
                             stride=stride, lead_in=rule["lead_in"],
                             train_fraction=rule["train_fraction"],
                             calibration_count=manifest["calibration"]["count"])
    problems = manifest_differences(manifest, rebuilt)
    if problems:
        raise ValueError(f"the data differs from {manifest_path}: {', '.join(problems[:8])}")
    norm = manifest["normalization"]
    series32 = normalized_array(series.values, norm["mean"], norm["std"])
    if hashlib.sha256(series32.astype("<f4").tobytes()).hexdigest() != \
            norm["normalized_float32_sha256"]:
        raise ValueError("the NumPy-normalized series differs from the manifest hash")
    anomalies = [(w["start"], w["stop"]) for w in manifest["anomaly_windows"]]
    mask = anomaly_mask(len(series), anomalies)
    windows, wlabels, starts = {}, {}, {}
    for name in SPLITS:
        s = window_starts([tuple(x) for x in manifest["splits"][name]["segments"]], window, stride)
        starts[name] = s
        windows[name] = windows_array(series32, s, window)
        wlabels[name] = np.asarray(window_labels(s, mask, window), dtype=bool)
    return SplitWindows(manifest, series32, windows, wlabels, starts,
                        windows_array(series32, manifest["calibration"]["window_starts"], window),
                        windows_array(series32, manifest["reference_windows"]["window_starts"],
                                      window))
