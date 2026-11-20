"""Tests for scripts/validate_results.py and the benchmark result schema."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

import validate_results as vr

EXAMPLES = vr.REPO_ROOT / "docs" / "examples"
SUCCESS = EXAMPLES / "example-result.json"
FAILED = EXAMPLES / "example-result-failed.json"


@pytest.fixture(scope="module")
def validator():
    return vr.make_validator()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# --- Valid documents ---------------------------------------------------------

@pytest.mark.parametrize("path", [SUCCESS, FAILED], ids=lambda p: p.name)
def test_examples_are_valid(validator, path):
    assert vr.validate_file(path, validator) == []


def test_examples_are_marked_illustrative():
    for path in (SUCCESS, FAILED):
        doc = load(path)
        assert doc["run_id"] == vr.NIL_UUID
        assert any(w["code"] == "illustrative_example" for w in doc["warnings"])


def test_main_validates_directory_recursively(tmp_path, capsys):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    shutil.copy(FAILED, nested / "run.json")
    assert vr.main([str(tmp_path)]) == 0
    assert "1 passed, 0 failed" in capsys.readouterr().out


# --- Invalid documents -------------------------------------------------------

def _set(path: str, value: Any) -> Callable[[dict], None]:
    *parents, last = path.split("/")

    def mutate(doc: dict) -> None:
        node = doc
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        node[last] = value

    return mutate


def _delete(key: str) -> Callable[[dict], None]:
    return lambda doc: doc.pop(key)


BROKEN = [
    # (id, base document, mutation, expected JSON Pointer of the reported issue)
    ("missing_schema_version", SUCCESS, _delete("schema_version"), "/"),
    ("unknown_schema_major", SUCCESS, _set("schema_version", "2.0.0"), "/schema_version"),
    ("wrong_type_batch_size", SUCCESS, _set("config/batch_size", "1"), "/config/batch_size"),
    ("wrong_type_mean", SUCCESS, _set("timing/aggregate/stats/mean_ms", "fast"), "/timing/aggregate/stats/mean_ms"),
    ("unknown_top_level_field", SUCCESS, _set("speedup", 10), "/"),
    ("unknown_nested_field", SUCCESS, _set("timing/trials/0/stats/median_ms", 1.0), "/timing/trials/0/stats"),
    ("bad_uuid", SUCCESS, _set("run_id", "not-a-uuid"), "/run_id"),
    ("timestamp_not_utc", SUCCESS, _set("timestamp_utc", "2000-01-01T00:00:00+01:00"), "/timestamp_utc"),
    ("success_with_error", SUCCESS, _set("error", {"stage": "warmup", "message": "x", "code": None}), "/error"),
    ("success_without_timing", SUCCESS, _set("timing", None), "/timing"),
    ("failed_without_error", FAILED, _set("error", None), "/error"),
    ("no_quantization_but_format", SUCCESS, _set("model/quantization/format", "qdq"), "/model/quantization/format"),
    ("verified_without_method", SUCCESS, _set("execution/verification_method", None),
     "/execution/verification_method"),
    # Consistency rules beyond JSON Schema
    ("nil_uuid_without_example_warning", SUCCESS, _set("warnings", []), "/run_id"),
    ("p99_with_too_few_samples", SUCCESS, _set("config/measured_iterations", 50), "/timing/trials/0/stats/count"),
    ("percentiles_out_of_order", SUCCESS, _set("timing/aggregate/stats/p95_ms", 3.0), "/timing/aggregate/stats/p99_ms"),
    ("throughput_mismatch", SUCCESS, _set("timing/aggregate/stats/inferences_per_sec", 1.0),
     "/timing/aggregate/stats/inferences_per_sec"),
    ("trial_count_mismatch", SUCCESS, _set("config/trials", 5), "/timing/trials"),
]


@pytest.mark.parametrize("case", BROKEN, ids=[c[0] for c in BROKEN])
def test_broken_documents_fail(validator, case):
    _, base, mutate, expected_path = case
    doc = copy.deepcopy(load(base))
    mutate(doc)
    issues = vr.validate_document(doc, validator, base)
    assert issues, "expected the document to be rejected"
    assert expected_path in {i.path or "/" for i in issues}, [str(i) for i in issues]


def test_missing_raw_file_fails(validator, tmp_path):
    target = tmp_path / "result.json"
    shutil.copy(SUCCESS, target)  # without the latencies CSV next to it
    issues = vr.validate_file(target, validator)
    assert [i.path for i in issues] == ["/raw/latencies_file"]


def test_invalid_json_fails(validator, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    issues = vr.validate_file(bad, validator)
    assert len(issues) == 1 and "invalid JSON" in issues[0].message


def test_main_reports_failure(tmp_path, capsys):
    doc = load(FAILED)
    del doc["schema_version"]
    (tmp_path / "bad.json").write_text(json.dumps(doc), encoding="utf-8")
    shutil.copy(FAILED, tmp_path / "good.json")
    assert vr.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "PASS" in out and "1 passed, 1 failed" in out
