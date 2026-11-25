"""Windowing, labels, splits and the split manifest of common/timeseries.py.

The synthetic tests use only the standard library and run in CI without the
dataset; tests on the real NAB files skip visibly when they are not downloaded
(``python scripts/get_data.py --dataset nab``), and the NumPy array tests skip
without NumPy.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from datetime import datetime, timedelta

import pytest

from common import timeseries as ts


def synthetic(n: int = 2000, anomalies=((300, 400), (1700, 1800)), seed: int = 7):
    """A series with timestamps every 5 minutes and label windows around ``anomalies``."""
    rng = random.Random(seed)
    t0 = datetime(2020, 1, 1)
    stamps = [(t0 + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n)]
    values = [50 + 10 * math.sin(i / 40) + rng.gauss(0, 1) for i in range(n)]
    windows = [((t0 + timedelta(minutes=5 * a)).strftime("%Y-%m-%d %H:%M:%S.000000"),
                (t0 + timedelta(minutes=5 * (b - 1))).strftime("%Y-%m-%d %H:%M:%S.000000"))
               for a, b in anomalies]
    return ts.Series(tuple(stamps), tuple(values)), windows


def csv_text(series: ts.Series) -> str:
    return "timestamp,value\n" + "".join(f"{t},{v!r}\n" for t, v in zip(series.timestamps,
                                                                        series.values))


# --- parsing ----------------------------------------------------------------------


def test_parse_series_round_trip():
    series, _ = synthetic(50)
    assert ts.parse_series(csv_text(series)) == series


@pytest.mark.parametrize("text, message", [
    ("time,value\n2020-01-01 00:00:00,1\n", "header"),
    ("timestamp,value\n", "no points"),
    ("timestamp,value\n2020-01-01 00:00:00,1,2\n", "2 fields"),
    ("timestamp,value\n2020-13-01 00:00:00,1\n", "month"),
    ("timestamp,value\n2020-01-01 00:00:00,nan\n", "not finite"),
])
def test_parse_series_rejects_malformed_files(text, message):
    with pytest.raises(ValueError, match=message):
        ts.parse_series(text)


def test_parse_label_windows():
    text = json.dumps({ts.LABELS_KEY: [["2020-01-01 00:00:00.000000", "2020-01-02 00:00:00.000000"]],
                       "other.csv": []})
    assert ts.parse_label_windows(text) == [("2020-01-01 00:00:00.000000",
                                             "2020-01-02 00:00:00.000000")]
    with pytest.raises(ValueError, match="no label windows"):
        ts.parse_label_windows(json.dumps({}))
    with pytest.raises(ValueError, match="ends before"):
        ts.parse_label_windows(json.dumps({ts.LABELS_KEY: [["2020-01-02", "2020-01-01"]]}))


def test_irregular_steps_reports_repeated_timestamps():
    series, _ = synthetic(10)
    stamps = list(series.timestamps)
    stamps[6] = stamps[3]  # clock set back
    steps = ts.irregular_steps(ts.Series(tuple(stamps), series.values))
    assert [s["index"] for s in steps] == [6, 7]
    assert steps[0]["step_s"] == -600


# --- labels -----------------------------------------------------------------------


def test_label_ranges_are_inclusive_half_open_index_ranges():
    series, windows = synthetic()
    assert ts.label_ranges(series.timestamps, windows) == [(300, 400), (1700, 1800)]


def test_label_ranges_refuse_ambiguous_windows():
    series, windows = synthetic(100, anomalies=((10, 20),))
    with pytest.raises(ValueError, match="covers no point"):
        ts.label_ranges(series.timestamps, [("1999-01-01", "1999-01-02")])
    stamps = list(series.timestamps)
    stamps[50] = stamps[15]  # a timestamp inside the window repeats later
    with pytest.raises(ValueError, match="non-contiguous"):
        ts.label_ranges(stamps, windows)
    with pytest.raises(ValueError, match="overlap"):
        ts.label_ranges(series.timestamps, [(series.timestamps[10], series.timestamps[20]),
                                            (series.timestamps[15], series.timestamps[30])])


def test_anomaly_mask_and_window_labels():
    mask = ts.anomaly_mask(20, [(5, 8)])
    assert [i for i, m in enumerate(mask) if m] == [5, 6, 7]
    starts = list(range(0, 17))
    labels = ts.window_labels(starts, mask, window=4)
    # windows [s, s+4) touch [5, 8) for s = 2..7
    assert [s for s, bad in zip(starts, labels) if bad] == [2, 3, 4, 5, 6, 7]


# --- splits and windows -----------------------------------------------------------


def test_plan_splits_uses_the_longest_clean_stretch_in_time_order():
    plan = ts.plan_splits(2000, [(300, 400), (1700, 1800)], window=32, lead_in=100,
                          train_fraction=0.75)
    # clean stretch [400, 1700), minus 100 lead-in points -> [400, 1600)
    assert plan.train == ((400, 1300),)
    assert plan.validation == ((1300, 1600),)
    assert plan.test == ((0, 400), (1600, 2000))
    covered = sorted(plan.train + plan.validation + plan.test)
    assert covered[0][0] == 0 and covered[-1][1] == 2000
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))  # no gap, no overlap


def test_plan_splits_without_a_following_anomaly_keeps_no_lead_in():
    plan = ts.plan_splits(1000, [(100, 150)], window=32, lead_in=100, train_fraction=0.5)
    assert plan.train == ((150, 575),) and plan.validation == ((575, 1000),)
    assert plan.test == ((0, 150),)


def test_plan_splits_tie_picks_the_earliest_stretch():
    # Clean stretches [0, 200), [300, 500) and [600, 800) are equally long.
    plan = ts.plan_splits(1000, [(200, 300), (500, 600), (800, 1000)], window=16, lead_in=0,
                          train_fraction=0.5)
    assert plan.train == ((0, 100),) and plan.validation == ((100, 200),)
    assert plan.test == ((200, 1000),)


@pytest.mark.parametrize("n, anomalies, kwargs, message", [
    (1000, [], {}, "no labelled anomaly"),
    (1000, [(100, 200)], {"train_fraction": 1.0}, "train_fraction"),
    (300, [(100, 200)], {"window": 64, "lead_in": 0}, "shorter than one window"),
    (1000, [(10, 20)], {"window": 64, "lead_in": 0}, "shorter than one window"),
])
def test_plan_splits_refuses_unusable_series(n, anomalies, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ts.plan_splits(n, anomalies, **{"window": 32, "lead_in": 0, **kwargs})


def test_window_starts_never_cross_segment_boundaries():
    starts = ts.window_starts([(0, 10), (20, 26)], window=4, stride=3)
    assert starts == [0, 3, 6, 20]
    assert ts.window_starts([(0, 3)], window=4) == []
    with pytest.raises(ValueError):
        ts.window_starts([(0, 10)], window=0)


def test_evenly_spaced_includes_both_ends():
    items = list(range(100, 200))
    picked = ts.evenly_spaced(items, 5)
    assert picked == [100, 124, 149, 174, 199]
    assert ts.evenly_spaced(items, 1000) == items
    assert ts.evenly_spaced(items, 1) == [100]


def test_reference_starts_cover_every_test_anomaly_and_normal_windows():
    series, windows = synthetic()
    anomalies = ts.label_ranges(series.timestamps, windows)
    plan = ts.plan_splits(len(series), anomalies, window=32, lead_in=100, train_fraction=0.75)
    mask = ts.anomaly_mask(len(series), anomalies)
    starts = ts.reference_starts(plan, anomalies, mask, window=32, normal=3)
    labels = ts.window_labels(starts, mask, window=32)
    assert sum(labels) == 2 and len(starts) == 5
    for a, b in anomalies:  # one window fully inside each anomaly
        assert any(a <= s and s + 32 <= b for s in starts)
    assert all(any(s0 <= s and s + 32 <= s1 for s0, s1 in plan.test) for s in starts)


# --- normalization and bytes ------------------------------------------------------


def test_normalization_uses_only_the_given_segments():
    values = [1.0, 2.0, 3.0, 100.0, 200.0]
    mean, std = ts.normalization(values, [(0, 3)])
    assert mean == 2.0 and std == math.sqrt(2.0 / 3.0)
    with pytest.raises(ValueError, match="constant"):
        ts.normalization([5.0, 5.0, 5.0], [(0, 3)])


def test_normalized_float32_bytes_round_each_value_once():
    values = [0.1, 1.0 / 3.0, 1e6]
    data = ts.normalized_float32_bytes(values, 0.5, 3.0)
    expected = [struct.unpack("<f", struct.pack("<f", (v - 0.5) / 3.0))[0] for v in values]
    assert list(struct.unpack("<3f", data)) == expected


# --- the manifest -----------------------------------------------------------------


def test_manifest_on_synthetic_data():
    series, windows = synthetic()
    m = ts.build_manifest(series, windows, "a" * 64, "b" * 64, window=32, stride=2, lead_in=100,
                          train_fraction=0.75, calibration_count=10)
    assert m["splits"]["train"]["anomalous_points"] == 0
    assert m["splits"]["validation"]["anomalous_points"] == 0
    assert m["splits"]["test"]["anomalous_points"] == 200
    assert m["splits"]["train"]["windows"] == len(ts.window_starts([(400, 1300)], 32, 2))
    assert len(m["calibration"]["window_starts"]) == m["calibration"]["count"] == 10
    train_starts = set(ts.window_starts([(400, 1300)], 32, 2))
    assert set(m["calibration"]["window_starts"]) <= train_starts
    mean, std = m["normalization"]["mean"], m["normalization"]["std"]
    assert (mean, std) == ts.normalization(series.values, [(400, 1300)])
    assert m["normalization"]["normalized_float32_sha256"] == hashlib.sha256(
        ts.normalized_float32_bytes(series.values, mean, std)).hexdigest()
    # Deterministic and JSON round-trippable (floats included).
    again = ts.build_manifest(series, windows, "a" * 64, "b" * 64, window=32, stride=2,
                              lead_in=100, train_fraction=0.75, calibration_count=10)
    assert json.loads(ts.format_manifest(m)) == m == again


def test_manifest_differences_name_the_changed_paths():
    series, windows = synthetic()
    m = ts.build_manifest(series, windows, "a" * 64, "b" * 64, window=32, lead_in=100)
    changed = json.loads(json.dumps(m))
    changed["normalization"]["mean"] += 1e-12
    changed["splits"]["test"]["segments"][0][1] += 1
    del changed["dataset"]
    assert ts.manifest_differences(m, changed) == [
        "dataset (only in one)", "normalization/mean", "splits/test/segments"]
    assert ts.manifest_differences(m, m) == []


def test_read_split_manifest_rejects_other_versions(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"manifest_version": "9.0.0"}))
    with pytest.raises(ValueError, match="manifest_version"):
        ts.read_split_manifest(path)


def test_load_nab_checks_hashes(tmp_path):
    for rel in (ts.SERIES_PATH, ts.LABELS_PATH):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("not the file")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ts.load_nab(tmp_path)
    with pytest.raises(FileNotFoundError, match="get_data.py --dataset nab"):
        ts.load_nab(tmp_path / "missing")


def test_committed_manifest_matches_the_pinned_parameters():
    m = ts.read_split_manifest()
    assert m["source"]["commit"] == ts.NAB_COMMIT
    assert m["source"]["series"]["sha256"] == ts.SERIES_SHA256
    assert m["source"]["labels"]["sha256"] == ts.LABELS_SHA256
    assert m["windowing"] == {"length": ts.WINDOW_LENGTH, "stride": ts.WINDOW_STRIDE,
                              "window_is_anomalous": "any point inside a labelled window"}
    assert m["split_rule"] == {"lead_in": ts.LEAD_IN, "train_fraction": ts.TRAIN_FRACTION}
    segments = sorted(tuple(s) for name in ts.SPLITS for s in m["splits"][name]["segments"])
    assert segments[0][0] == 0 and segments[-1][1] == m["series"]["points"]
    assert all(a[1] == b[0] for a, b in zip(segments, segments[1:]))
    for name in ("train", "validation"):
        assert m["splits"][name]["anomalous_windows"] == 0
    train = m["splits"]["train"]["segments"][0]
    assert all(train[0] <= s and s + ts.WINDOW_LENGTH <= train[1]
               for s in m["calibration"]["window_starts"])
    test_anomalies = [w for w in m["anomaly_windows"]
                      if any(a <= w["start"] and w["stop"] <= b
                             for a, b in m["splits"]["test"]["segments"])]
    assert len(test_anomalies) == len(m["anomaly_windows"]) == 4


# --- real data (skips without the download) ---------------------------------------

needs_nab = pytest.mark.skipif(not ts.nab_present(),
                               reason="NAB data not downloaded (python scripts/get_data.py "
                                      "--dataset nab)")


@needs_nab
def test_real_data_rebuilds_the_committed_manifest():
    series, labels, s_sha, l_sha = ts.load_nab()
    assert ts.build_manifest(series, labels, s_sha, l_sha) == ts.read_split_manifest()


@needs_nab
def test_real_data_repeated_hour_is_outside_every_label_window():
    m = ts.read_split_manifest()
    (step,) = m["series"]["irregular_steps"]
    assert step["step_s"] == -3300
    assert not any(w["start"] <= step["index"] < w["stop"] for w in m["anomaly_windows"])


@needs_nab
def test_numpy_windows_are_the_manifest_bytes():
    np = pytest.importorskip("numpy")
    data = ts.load_split_windows()
    m = data.manifest
    assert hashlib.sha256(data.normalized.astype("<f4").tobytes()).hexdigest() == \
        m["normalization"]["normalized_float32_sha256"]
    for name in ts.SPLITS:
        w = data.windows[name]
        assert w.dtype == np.float32 and w.shape == (m["splits"][name]["windows"], ts.WINDOW_LENGTH)
        assert int(data.labels[name].sum()) == m["splits"][name]["anomalous_windows"]
        s = data.starts[name][-1]
        assert np.array_equal(w[-1], data.normalized[s:s + ts.WINDOW_LENGTH])
    train = data.windows["train"].astype(np.float64)
    # Training windows overlap, so their mean is close to (not exactly) 0.
    assert abs(train.mean()) < 0.05 and abs(train.std() - 1.0) < 0.05
    assert data.calibration.shape == (m["calibration"]["count"], ts.WINDOW_LENGTH)
    assert data.reference.shape == (len(m["reference_windows"]["window_starts"]), ts.WINDOW_LENGTH)


@needs_nab
def test_load_split_windows_refuses_a_different_manifest(tmp_path):
    pytest.importorskip("numpy")
    m = ts.read_split_manifest()
    m["normalization"]["std"] *= 1.0000001
    path = tmp_path / "splits.json"
    ts.write_split_manifest(m, path)
    with pytest.raises(ValueError, match="normalization/std"):
        ts.load_split_windows(manifest_path=path)


def test_numpy_normalization_matches_stdlib_bytes():
    np = pytest.importorskip("numpy")
    series, _ = synthetic(500)
    mean, std = ts.normalization(series.values, [(0, 300)])
    arr = ts.normalized_array(series.values, mean, std)
    assert arr.astype("<f4").tobytes() == ts.normalized_float32_bytes(series.values, mean, std)
    w = ts.windows_array(arr, [0, 10, 372], window=128)
    assert w.shape == (3, 128) and np.array_equal(w[1], arr[10:138])
