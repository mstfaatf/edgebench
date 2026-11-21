"""Tests for scripts/validate_results.py and the benchmark result schemas."""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

import validate_results as vr

EXAMPLES = vr.REPO_ROOT / "docs" / "examples"
V100 = EXAMPLES / "v1.0.0"
V110 = EXAMPLES / "v1.1.0"
SUCCESS = V100 / "example-result.json"
FAILED = V100 / "example-result-failed.json"
NPU_110 = V110 / "example-npu-int8-s8w-strict.json"
CPU_110 = V110 / "example-cpu-fp32.json"
PUBLISHED = vr.REPO_ROOT / "benchmarks" / "results" / "published"
INVALID_FIXTURES = Path(__file__).parent / "fixtures" / "results" / "v1.1.0" / "invalid"

ALL_EXAMPLES = sorted(EXAMPLES.rglob("*.json"))
V100_EXAMPLES = sorted(V100.glob("*.json"))
FIXTURES = sorted(INVALID_FIXTURES.glob("*.json"))


@pytest.fixture(scope="module")
def validators():
    return vr.Validators()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(vr.REPO_ROOT).as_posix()


def set_pointer(doc: Any, pointer: str, value: Any) -> None:
    *parents, last = [p.replace("~1", "/").replace("~0", "~") for p in pointer.lstrip("/").split("/")]
    node = doc
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


# --- Schema files ------------------------------------------------------------

def test_supported_versions():
    assert vr.supported_versions() == ["1.0.0", "1.1.0"]


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0"])
def test_schema_files_name_their_version(version):
    schema = vr.load_schema(vr.schema_path(version))
    assert schema["$id"].endswith(f"/schemas/benchmark-result.v{version}.schema.json")
    declared = schema["properties"]["schema_version"]
    assert declared.get("enum", [declared.get("const")]) == [version]


def test_no_unversioned_schema_file():
    assert not (vr.SCHEMA_DIR / "benchmark-result.schema.json").exists()


def test_cpp_schema_version_constant_is_newest():
    cmake = (vr.REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"set\(EDGEBENCH_RESULTS_SCHEMA_VERSION (\d+\.\d+\.\d+)\)", cmake)
    assert match, "EDGEBENCH_RESULTS_SCHEMA_VERSION not set in CMakeLists.txt"
    assert match.group(1) == vr.supported_versions()[-1]


def test_load_check_values_match_cpp_loader():
    schema = vr.load_schema(vr.schema_path("1.1.0"))
    runtime = next(b for b in schema["properties"]["runtime"]["oneOf"] if b.get("type") == "object")
    values = runtime["properties"]["onnxruntime"]["properties"]["load_check"]["enum"]
    loader = (vr.REPO_ROOT / "engine" / "src" / "ort_loader.cpp").read_text(encoding="utf-8")
    # The C++ loader's OrtLoadResult::load_check() success value; failures leave runtime null.
    assert '"full_path_verified"' in loader
    assert "full_path_verified" in values


# --- Valid documents -----------------------------------------------------------

@pytest.mark.parametrize("path", ALL_EXAMPLES, ids=rel)
def test_examples_are_valid(validators, path):
    assert vr.validate_file(path, validators) == []


def test_examples_cover_both_versions():
    versions = {load(p)["schema_version"] for p in ALL_EXAMPLES}
    assert versions == {"1.0.0", "1.1.0"}
    names = {p.name for p in V110.glob("*.json")}
    assert {"example-cpu-fp32.json", "example-cpu-int8-s8w.json", "example-npu-int8-s8w-strict.json"} <= names


@pytest.mark.parametrize("path", ALL_EXAMPLES, ids=rel)
def test_examples_are_marked_illustrative(path):
    doc = load(path)
    assert doc["run_id"] == vr.NIL_UUID
    assert any(w["code"] == "illustrative_example" for w in doc["warnings"])


@pytest.mark.parametrize("path", sorted(PUBLISHED.rglob("*.json")), ids=rel)
def test_published_results_are_valid(validators, path):
    assert vr.errors(vr.validate_file(path, validators)) == []


def test_published_directory_passes(capsys):
    assert vr.main([str(PUBLISHED)]) == 0


@pytest.mark.parametrize("path", V100_EXAMPLES, ids=rel)
def test_v100_examples_are_valid_v110(validators, path):
    """The bump is additive: a 1.0.0 document with only schema_version changed is valid 1.1.0."""
    doc = load(path)
    doc["schema_version"] = "1.1.0"
    assert vr.validate_document(doc, validators, path) == []


def test_v110_document_fails_against_v100_schema():
    forced = vr.Validators(forced_schema=vr.load_schema(vr.schema_path("1.0.0")))
    paths = {i.path for i in vr.validate_file(NPU_110, forced)}
    assert "/schema_version" in paths and "/config" in paths


def test_npu_example_records_profiled_pass_separately():
    doc = load(NPU_110)
    assert doc["config"]["profiling"] == {"ort": False, "qnn_level": "off"}
    assert doc["timing"]["accelerator"]["separate_run"] is True
    assert doc["config"]["ep_context_cache"]["state"] == "loaded"
    assert doc["execution"]["qnn_validation"]["rejected_nodes"] == []


def test_main_validates_directory_recursively(tmp_path, capsys):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    shutil.copy(FAILED, nested / "run.json")
    assert vr.main([str(tmp_path)]) == 0
    assert "1 passed, 0 failed" in capsys.readouterr().out


# --- Schema version selection ------------------------------------------------

def test_missing_schema_version_names_supported_versions(validators):
    doc = load(NPU_110)
    del doc["schema_version"]
    issues = vr.validate_document(doc, validators)
    assert [i.path for i in issues] == ["/schema_version"]
    assert "missing" in issues[0].message and "1.0.0, 1.1.0" in issues[0].message


@pytest.mark.parametrize("version", ["2.0.0", "1.2.0", "0.9.0", "1.1"])
def test_unknown_schema_version_names_supported_versions(validators, version):
    doc = load(NPU_110)
    doc["schema_version"] = version
    issues = vr.validate_document(doc, validators)
    assert [i.path for i in issues] == ["/schema_version"]
    assert "unsupported" in issues[0].message and "1.0.0, 1.1.0" in issues[0].message


def test_v100_document_is_checked_with_v100_schema(validators):
    doc = load(SUCCESS)
    doc["config"]["run_binding"] = "run"  # a 1.1.0 field
    issues = vr.validate_document(doc, validators, SUCCESS)
    assert [i.path for i in issues] == ["/config"]


# --- Invalid documents (both versions) ---------------------------------------

def _set(path: str, value: Any) -> Callable[[dict], None]:
    return lambda doc: set_pointer(doc, "/" + path, value)


def _delete(key: str) -> Callable[[dict], None]:
    return lambda doc: doc.pop(key)


BROKEN = [
    # (id, base document, mutation, expected JSON Pointer of the reported issue)
    ("missing_schema_version", SUCCESS, _delete("schema_version"), "/schema_version"),
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
    # The same 1.0.0 rules on a 1.1.0 document
    ("v110_throughput_mismatch", NPU_110, _set("timing/aggregate/stats/inferences_per_sec", 1.0),
     "/timing/aggregate/stats/inferences_per_sec"),
    ("v110_trial_count_mismatch", CPU_110, _set("config/trials", 5), "/timing/trials"),
    ("v110_percentiles_out_of_order", NPU_110, _set("timing/trials/1/stats/p99_ms", 0.1),
     "/timing/trials/1/stats/p99_ms"),
]


@pytest.mark.parametrize("case", BROKEN, ids=[c[0] for c in BROKEN])
def test_broken_documents_fail(validators, case):
    _, base, mutate, expected_path = case
    doc = copy.deepcopy(load(base))
    mutate(doc)
    issues = vr.errors(vr.validate_document(doc, validators, base))
    assert issues, "expected the document to be rejected"
    assert expected_path in {i.path or "/" for i in issues}, [str(i) for i in issues]


def test_missing_raw_file_fails(validators, tmp_path):
    target = tmp_path / "result.json"
    shutil.copy(SUCCESS, target)  # without the latencies CSV next to it
    issues = vr.validate_file(target, validators)
    assert [i.path for i in issues] == ["/raw/latencies_file"]


def test_invalid_json_fails(validators, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    issues = vr.validate_file(bad, validators)
    assert len(issues) == 1 and "invalid JSON" in issues[0].message


def test_main_reports_failure(tmp_path, capsys):
    doc = load(FAILED)
    del doc["schema_version"]
    (tmp_path / "bad.json").write_text(json.dumps(doc), encoding="utf-8")
    shutil.copy(FAILED, tmp_path / "good.json")
    assert vr.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "PASS" in out and "1 passed, 1 failed" in out


# --- 1.1.0 rule fixtures -----------------------------------------------------

def build_fixture(path: Path) -> tuple[dict[str, Any], Path, dict[str, str]]:
    spec = load(path)
    base = vr.REPO_ROOT / spec["base"]
    doc = load(base)
    for pointer, value in spec["set"].items():
        set_pointer(doc, pointer, copy.deepcopy(value))
    return doc, base, spec["expect"]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_invalid_fixture_fails_for_the_intended_reason(validators, path):
    doc, base, expect = build_fixture(path)
    issues = vr.validate_document(doc, validators, base)
    assert len(issues) == 1, [str(i) for i in issues]
    issue = issues[0]
    assert (issue.severity, issue.code, issue.path) == (expect["severity"], expect["code"], expect["path"]), str(issue)


def test_every_v1_1_rule_has_a_fixture():
    codes = {load(p)["expect"]["code"] for p in FIXTURES}
    assert set(vr.V1_1_RULES) <= codes, set(vr.V1_1_RULES) - codes


@pytest.mark.parametrize("path", [p for p in FIXTURES if load(p)["expect"]["code"]], ids=lambda p: p.stem)
def test_rule_does_not_apply_to_v100(validators, path):
    """Rules for 1.1.0 fields are not applied to documents declaring 1.0.0."""
    doc, base, _ = build_fixture(path)
    issues = vr.v1_1_issues(doc)
    assert issues  # the rule fires on the 1.1.0 document ...
    doc["schema_version"] = "1.0.0"
    assert not any(i.code for i in vr.consistency_issues(doc, base))  # ... but is not run for 1.0.0


@pytest.mark.parametrize("severity", [vr.ERROR, vr.WARNING])
def test_cli_exit_codes_by_severity(tmp_path, capsys, severity):
    fixture = next(p for p in FIXTURES if load(p)["expect"]["severity"] == severity and load(p)["expect"]["code"])
    doc, _, _ = build_fixture(fixture)
    target = tmp_path / "result.json"
    target.write_text(json.dumps(doc), encoding="utf-8")
    plain = vr.main([str(target)])
    out = capsys.readouterr().out
    strict = vr.main(["--strict", str(target)])
    strict_out = capsys.readouterr().out
    if severity == vr.ERROR:
        assert (plain, strict) == (1, 1)
        assert out.startswith("FAIL") and "error:" in out
    else:
        assert (plain, strict) == (0, 1)  # warnings pass unless --strict
        assert out.startswith("WARN") and "warning:" in out and "0 failed, 1 with warnings" in out
        assert strict_out.startswith("FAIL") and "1 failed" in strict_out


# --- Helpers -------------------------------------------------------------------

@pytest.mark.parametrize("path, directory, expected", [
    (r"C:\dev\bin\onnxruntime.dll", r"C:\dev\bin", True),
    (r"c:/DEV/Bin/QnnHtp.dll", "C:\\dev\\bin\\", True),
    ("\\\\?\\C:\\dev\\bin\\x.dll", r"C:\dev\bin", True),
    (r"C:\dev\bin2\x.dll", r"C:\dev\bin", False),
    (r"C:\Windows\System32\onnxruntime.dll", r"C:\dev\bin", False),
])
def test_windows_path_within(path, directory, expected):
    assert vr._windows_path_within(path, directory) is expected
