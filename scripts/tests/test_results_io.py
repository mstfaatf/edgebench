"""Tests for scripts/common/results_io.py (per-run, never-overwritten summaries)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from common.results_io import new_run, run_stamp

T1 = datetime(2026, 11, 21, 1, 30, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 11, 21, 1, 30, 5, tzinfo=timezone.utc)


def test_run_stamp_format():
    assert run_stamp(T1) == "20261121-013000"


def test_paths_and_latest_copy(tmp_path):
    out = new_run(tmp_path, "npu_check", now=T1)
    assert out.summary.name == "npu_check_20261121-013000.json"
    assert out.run_dir.name == "npu_check_20261121-013000"
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
