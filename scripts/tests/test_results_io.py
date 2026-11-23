"""Tests for scripts/common/results_io.py (per-run, never-overwritten summaries)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from common.results_io import new_run, run_stamp

T1 = datetime(2026, 11, 21, 1, 30, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 11, 21, 1, 30, 5, tzinfo=timezone.utc)


def test_run_stamp_format():
    assert run_stamp(T1) == "20261121T013000.000Z"
    assert run_stamp(datetime(2026, 11, 21, 23, 52, 10, 123456, tzinfo=timezone.utc)) ==         "20261121T235210.123Z"


def test_run_stamp_truncates_milliseconds():
    # .999999 s must not round up into the next second (or day).
    t = datetime(2026, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
    assert run_stamp(t) == "20261231T235959.999Z"


def test_run_stamp_converts_to_utc():
    cet = timezone(timedelta(hours=1))
    assert run_stamp(datetime(2026, 11, 21, 2, 30, 0, 5000, tzinfo=cet)) == "20261121T013000.005Z"


def test_same_second_runs_get_distinct_stamps(tmp_path):
    a = new_run(tmp_path, "eval", now=datetime(2026, 11, 21, 1, 30, 0, 100000, tzinfo=timezone.utc))
    a.write({"run": "a"})
    b = new_run(tmp_path, "eval", now=datetime(2026, 11, 21, 1, 30, 0, 101000, tzinfo=timezone.utc))
    b.write({"run": "b"})
    assert a.summary != b.summary
    assert json.loads(a.summary.read_text()) == {"run": "a"}


def test_old_stamp_files_are_left_alone(tmp_path):
    old = tmp_path / "eval_20261120-013000.json"
    old.write_text('{"old": true}', encoding="utf-8")
    new_run(tmp_path, "eval", now=T1).write({"new": True})
    assert sorted(p.name for p in tmp_path.glob("eval_*.json")) == [
        "eval_20261120-013000.json", "eval_20261121T013000.000Z.json", "eval_latest.json"]


def test_paths_and_latest_copy(tmp_path):
    out = new_run(tmp_path, "npu_check", now=T1)
    assert out.summary.name == "npu_check_20261121T013000.000Z.json"
    assert out.run_dir.name == "npu_check_20261121T013000.000Z"
    out.make_run_dir()
    out.write({"run": 1})
    assert json.loads(out.latest.read_text()) == {"run": 1}

    second = new_run(tmp_path, "npu_check", now=T2)
    second.write({"run": 2})
    assert json.loads(out.summary.read_text()) == {"run": 1}  # first run untouched
    assert json.loads(second.latest.read_text()) == {"run": 2}


def test_never_overwrites(tmp_path):
    out = new_run(tmp_path, "eval", now=T1)
    out.write({"a": 1})
    with pytest.raises(FileExistsError):
        new_run(tmp_path, "eval", now=T1)
    with pytest.raises(FileExistsError):
        out.write({"a": 2})
    assert json.loads(out.summary.read_text()) == {"a": 1}


def test_run_dir_is_never_reused(tmp_path):
    out = new_run(tmp_path, "overhead", now=T1)
    out.make_run_dir()
    with pytest.raises(FileExistsError):
        out.make_run_dir()
