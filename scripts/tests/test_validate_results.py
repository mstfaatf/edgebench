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
V120 = EXAMPLES / "v1.2.0"
SUCCESS = V100 / "example-result.json"
FAILED = V100 / "example-result-failed.json"
NPU_110 = V110 / "example-npu-int8-s8w-strict.json"
CPU_110 = V110 / "example-cpu-fp32.json"
NPU_120 = V120 / "example-npu-int8-s8w-strict.json"
CPU_120 = V120 / "example-cpu-fp32.json"
B8_120 = V120 / "example-npu-int8-b8-not-equivalent.json"
REFUSAL_120 = V120 / "example-npu-strict-refusal.json"
PUBLISHED = vr.REPO_ROOT / "benchmarks" / "results" / "published"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "results"
INVALID_FIXTURES = FIXTURE_ROOT / "v1.1.0" / "invalid"
INVALID_FIXTURES_120 = FIXTURE_ROOT / "v1.2.0" / "invalid"

ALL_EXAMPLES = sorted(EXAMPLES.rglob("*.json"))
V100_EXAMPLES = sorted(V100.glob("*.json"))
V110_EXAMPLES = sorted(V110.glob("*.json"))
FIXTURES_110 = sorted(INVALID_FIXTURES.glob("*.json"))
FIXTURES_120 = sorted(INVALID_FIXTURES_120.glob("*.json"))
FIXTURES = FIXTURES_110 + FIXTURES_120


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
    assert vr.supported_versions() == ["1.0.0", "1.1.0", "1.2.0"]


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "1.2.0"])
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


def test_examples_cover_every_version():
    versions = {load(p)["schema_version"] for p in ALL_EXAMPLES}
    assert versions == {"1.0.0", "1.1.0", "1.2.0"}
    for directory in (V110, V120):
        names = {p.name for p in directory.glob("*.json")}
        assert {"example-cpu-fp32.json", "example-cpu-int8-s8w.json",
                "example-npu-int8-s8w-strict.json"} <= names
    assert {p.name for p in V120.glob("*.json")} >= {"example-npu-strict-refusal.json",
                                                     "example-npu-int8-b8-not-equivalent.json"}


@pytest.mark.parametrize("path", ALL_EXAMPLES, ids=rel)
def test_examples_are_marked_illustrative(path):
    doc = load(path)
    assert doc["run_id"] == vr.NIL_UUID
    assert any(w["code"] == "illustrative_example" for w in doc["warnings"])


# Accuracy results have their own schema and validator (scripts/validate_accuracy.py).
@pytest.mark.parametrize("path", sorted(p for p in PUBLISHED.rglob("*.json")
                                        if not vr.is_accuracy_document(p)), ids=rel)
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


@pytest.mark.parametrize("path", V100_EXAMPLES + V110_EXAMPLES, ids=rel)
def test_older_examples_are_valid_v120(validators, path):
    """The 1.2.0 bump is additive too: 1.0.0 and 1.1.0 documents with only schema_version
    changed are valid 1.2.0 (they are not C++ results, or omit only keys added later)."""
    doc = load(path)
    doc["schema_version"] = "1.2.0"
    doc["tool"]["implementation"] = "python"  # the C++ engine would write every key
    assert vr.validate_document(doc, validators, path) == []


def test_v120_document_fails_against_v110_schema():
    forced = vr.Validators(forced_schema=vr.load_schema(vr.schema_path("1.1.0")))
    paths = {i.path for i in vr.validate_file(NPU_120, forced)}
    assert "/schema_version" in paths and "/runtime/qnn" in paths


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
    assert "missing" in issues[0].message and "1.0.0, 1.1.0, 1.2.0" in issues[0].message


@pytest.mark.parametrize("version", ["2.0.0", "1.3.0", "0.9.0", "1.1"])
def test_unknown_schema_version_names_supported_versions(validators, version):
    doc = load(NPU_110)
    doc["schema_version"] = version
    issues = vr.validate_document(doc, validators)
    assert [i.path for i in issues] == ["/schema_version"]
    assert "unsupported" in issues[0].message and "1.0.0, 1.1.0, 1.2.0" in issues[0].message


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
    for pointer in spec.get("remove", []):
        remove_pointer(doc, pointer)
    return doc, base, spec["expect"]


def remove_pointer(doc: Any, pointer: str) -> None:
    *parents, last = pointer.lstrip("/").split("/")
    node = doc
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    del node[last]


def test_missing_keys_only_reported_for_cpp_results(validators):
    doc = load(CPU_110)
    del doc["config"]["run_binding"]
    del doc["timing"]["scope"]
    paths = {i.path for i in vr.validate_document(doc, validators, CPU_110) if i.code}
    assert paths == {"/config/run_binding", "/timing/scope"}
    doc["tool"]["implementation"] = "python"  # other writers may omit 1.1.0 keys
    assert vr.validate_document(doc, validators, CPU_110) == []


def test_missing_keys_descend_into_nullable_blocks(validators):
    # Keys 1.1.0 added to blocks that are oneOf [null, object] in the schema.
    doc = load(NPU_110)
    del doc["runtime"]["ep_device"]["selection"]
    del doc["timing"]["session_init"]["cached_ms"]
    del doc["config"]["ep_context_cache"]["files"]
    paths = {i.path for i in vr.validate_document(doc, validators, NPU_110) if i.code}
    assert {"/runtime/ep_device/selection", "/timing/session_init/cached_ms",
            "/config/ep_context_cache/files"} <= paths


def test_missing_keys_skip_null_blocks(validators):
    doc = load(CPU_110)
    assert doc["runtime"]["ep_device"] is None and doc["memory"] is None
    assert vr.validate_document(doc, validators, CPU_110) == []  # nothing below null is required


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_invalid_fixture_fails_for_the_intended_reason(validators, path):
    doc, base, expect = build_fixture(path)
    issues = vr.validate_document(doc, validators, base)
    assert len(issues) == 1, [str(i) for i in issues]
    issue = issues[0]
    assert (issue.severity, issue.code, issue.path) == (expect["severity"], expect["code"], expect["path"]), str(issue)


def test_run_stopped_for_a_foreign_module_may_record_it(validators):
    """A failed run whose error is module_outside_bundle records the module on purpose."""
    doc, base, _ = build_fixture(INVALID_FIXTURES / "module_outside_bundle.json")
    schema = validators.for_version("1.1.0").schema
    assert any(i.code == "module_outside_bundle" for i in vr.consistency_issues(doc, base, schema))
    doc["status"] = "failed"
    doc["error"] = {"stage": "session_creation", "code": "module_outside_bundle",
                    "message": "onnxruntime.dll loaded from outside the bundle directory"}
    assert not any(i.code == "module_outside_bundle" for i in vr.consistency_issues(doc, base, schema))
    doc["error"]["code"] = "ORT_FAIL"  # any other failure: the rule applies again
    assert any(i.code == "module_outside_bundle" for i in vr.consistency_issues(doc, base, schema))


def test_every_v1_1_rule_has_a_fixture():
    codes = {load(p)["expect"]["code"] for p in FIXTURES_110}
    assert set(vr.V1_1_RULES) <= codes, set(vr.V1_1_RULES) - codes


def test_every_v1_2_rule_has_a_fixture():
    codes = {load(p)["expect"]["code"] for p in FIXTURES_120}
    assert set(vr.V1_2_RULES) == codes, set(vr.V1_2_RULES) ^ codes


def test_v1_2_fixtures_use_v1_2_documents():
    assert all(load(vr.REPO_ROOT / load(p)["base"])["schema_version"] == "1.2.0" for p in FIXTURES_120)


@pytest.mark.parametrize("path", [p for p in FIXTURES if load(p)["expect"]["code"]], ids=lambda p: p.stem)
def test_rule_does_not_apply_to_v100(validators, path):
    """Rules for 1.1.0 fields are not applied to documents declaring 1.0.0."""
    doc, base, _ = build_fixture(path)
    schema = validators.for_version(doc["schema_version"]).schema
    issues = [i for i in vr.consistency_issues(doc, base, schema) if i.code]
    assert issues  # the rule fires on the 1.1.0 / 1.2.0 document ...
    doc["schema_version"] = "1.0.0"
    assert not any(i.code for i in vr.consistency_issues(doc, base, schema))  # ... not for 1.0.0


@pytest.mark.parametrize("path", FIXTURES_120, ids=lambda p: p.stem)
def test_v1_2_rule_does_not_apply_to_v110(validators, path):
    doc, base, expect = build_fixture(path)
    doc["schema_version"] = "1.1.0"
    schema = validators.for_version("1.1.0").schema
    assert not any(i.code in vr.V1_2_RULES for i in vr.consistency_issues(doc, base, schema))


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


# --- 1.1.0 amendment: unknown values in failed results are null ----------------

def test_failed_result_may_have_null_batch_size_and_provider(validators):
    doc = load(V110 / "example-failed-ort-load.json")
    assert doc["config"]["batch_size"] is None
    assert vr.validate_document(doc, validators, V110 / "example-failed-ort-load.json") == []


def test_failed_result_with_runtime_but_no_provider_is_valid(validators):
    doc = load(CPU_110)
    doc["status"] = "failed"
    doc["error"] = {"stage": "ep_registration", "message": "no CPU EP device", "code": "device_selection_failed"}
    doc["runtime"]["execution_provider"] = None
    doc["runtime"]["ep_devices"] = None
    doc["config"]["batch_size"] = None
    doc["timing"] = None
    doc["execution"] = None
    assert vr.validate_document(doc, validators, CPU_110) == []


def test_successful_result_still_requires_device_list(validators):
    doc = load(CPU_110)
    doc["runtime"]["ep_devices"] = None
    paths = {i.path for i in vr.validate_document(doc, validators, CPU_110)}
    assert "/runtime/ep_devices" in paths


def test_cli_prints_non_ascii_paths_on_a_legacy_code_page(tmp_path):
    # Windows runners give a child Python a cp1252 stdout, which cannot encode
    # every path; the validator must still report and exit 0.
    import os
    import subprocess
    import sys

    target = tmp_path / "测试 dir"
    target.mkdir()
    shutil.copy(V110 / "example-cpu-fp32.json", target)
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    result = subprocess.run(
        [sys.executable, str(vr.REPO_ROOT / "scripts" / "validate_results.py"), "--strict", str(target)],
        capture_output=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert b"PASS " in result.stdout


# --- 1.2.0 ---------------------------------------------------------------------

def test_missing_keys_name_the_version_that_added_them(validators):
    doc = load(CPU_120)
    del doc["config"]["run_binding"]          # 1.1.0
    del doc["timing"]["profiled_pass"]        # 1.2.0
    del doc["execution"]["batch_equivalence"]  # 1.2.0
    codes = {(i.path, i.code) for i in vr.validate_document(doc, validators, CPU_120)}
    assert codes == {("/config/run_binding", "missing_1_1_0_key"),
                     ("/timing/profiled_pass", "missing_1_2_0_key"),
                     ("/execution/batch_equivalence", "missing_1_2_0_key")}


def test_unsupported_nodes_only_on_strict_refusals(validators):
    doc = load(CPU_120)
    doc["status"] = "failed"
    doc["error"] = {"stage": "session_creation", "message": "x", "code": "ORT_FAIL",
                    "kind": "session_creation_failed", "unsupported_nodes": []}
    issues = vr.validate_document(doc, validators, CPU_120)
    assert ("/error/unsupported_nodes", "unsupported_nodes_mismatch") in {(i.path, i.code) for i in issues}
    doc["error"]["unsupported_nodes"] = None
    assert not any(i.code == "unsupported_nodes_mismatch" for i in vr.validate_document(doc, validators, CPU_120))


def test_strict_refusal_may_name_no_nodes(validators):
    doc = load(REFUSAL_120)
    doc["error"]["unsupported_nodes"] = []
    assert vr.validate_document(doc, validators, REFUSAL_120) == []


@pytest.mark.parametrize("equivalence", [
    {"equivalent_to_b1": True, "basis": "not_available", "reference_batch": 1},
    {"equivalent_to_b1": None, "basis": "reference_outputs", "reference_batch": 1},
    {"equivalent_to_b1": False, "basis": "batch_size_1", "reference_batch": 1},
    {"equivalent_to_b1": True, "basis": "reference_outputs", "reference_batch": 4},
])
def test_batch_equivalence_schema_rules(validators, equivalence):
    doc = load(NPU_120)
    doc["execution"]["batch_equivalence"] = equivalence
    paths = {i.path for i in vr.validate_document(doc, validators, NPU_120)}
    assert any(p.startswith("/execution/batch_equivalence") for p in paths), paths


UNKNOWN_BATCH = {"bitwise_equal_to_b1": None, "bitwise_basis": "not_available",
                 "predictions_equal_to_b1": None, "predictions_basis": "not_available",
                 "predictions_compared": None, "predictions_differing": None, "reference_batch": 1}


def test_batch_equivalence_unknown_is_valid(validators):
    doc = load(NPU_120)
    doc["execution"]["batch_equivalence"] = dict(UNKNOWN_BATCH)
    assert vr.validate_document(doc, validators, NPU_120) == []


def test_bits_differ_with_predictions_unknown_is_valid(validators):
    doc = load(NPU_120)
    doc["execution"]["batch_equivalence"] = {**UNKNOWN_BATCH, "bitwise_equal_to_b1": False,
                                             "bitwise_basis": "reference_outputs"}
    assert vr.validate_document(doc, validators, NPU_120) == []


@pytest.mark.parametrize("change", [
    {"bitwise_equal_to_b1": True, "bitwise_basis": "reference_outputs"},  # bitwise needs basis bitwise
    {"predictions_basis": "eval_quick_top1"},                            # counts required
    {"predictions_equal_to_b1": True},                                   # not_available needs null
    {"bitwise_basis": "batch_size_1"},                                   # batch 1 is bitwise equal
])
def test_batch_equivalence_shape_rules_are_in_the_schema(validators, change):
    doc = load(NPU_120)
    doc["execution"]["batch_equivalence"] = {**UNKNOWN_BATCH, "bitwise_equal_to_b1": False,
                                             "bitwise_basis": "reference_outputs", **change}
    issues = vr.validate_document(doc, validators, NPU_120)
    assert any(i.path.startswith("/execution/batch_equivalence") for i in vr.errors(issues)), issues


def test_cache_mode_off_requires_cache_disabled(validators):
    doc = load(B8_120)
    doc["config"]["ep_context_cache"]["enabled"] = True
    paths = {i.path for i in vr.validate_document(doc, validators, B8_120)}
    assert "/config/ep_context_cache/enabled" in paths


def test_verification_details_may_still_be_a_string(validators):
    doc = load(NPU_120)
    doc["execution"]["verification_details"] = "placement from the verbose session log"
    assert vr.validate_document(doc, validators, NPU_120) == []


def test_b8_example_is_flagged_not_equivalent():
    doc = load(B8_120)
    be = doc["execution"]["batch_equivalence"]
    assert (be["bitwise_equal_to_b1"], be["bitwise_basis"]) == (False, "reference_outputs")
    # docs/accuracy.md: 6 of the 1,000 eval-quick top-1 classes change at batch 8.
    assert (be["predictions_equal_to_b1"], be["predictions_basis"], be["predictions_differing"],
            be["predictions_compared"]) == (False, "eval_quick_top1", 6, 1000)
    assert any(w["code"] == "batch_not_equivalent" for w in doc["warnings"])
    assert doc["notes"] == []


def test_cache_example_key_id_is_recomputed(validators):
    path = V120 / "example-npu-int8-ep-cache-init-breakdown.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    cache = doc["config"]["ep_context_cache"]
    assert (cache["mode"], cache["state"], cache["hit"], cache["verified_equal_outputs"]) == \
        ("use", "generated", False, True)
    assert all(t["session_init"]["cold_ms"] and t["session_init"]["context_build_ms"]
               and t["session_init"]["cached_init_ms"] for t in doc["timing"]["trials"])
    assert vr.validate_document(doc, validators, path) == []
    doc["config"]["ep_context_cache"]["key"]["ep_options"]["htp_performance_mode"] = "balanced"
    codes = {i.code for i in vr.validate_document(doc, validators, path)}
    assert "cache_key_id_mismatch" in codes


def test_init_breakdown_allows_cached_values_with_the_cache_off(validators):
    path = V120 / "example-npu-int8-ep-cache-init-breakdown.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    cache = doc["config"]["ep_context_cache"]
    cache.update({"enabled": False, "mode": "off", "state": "disabled", "files": [], "key": None,
                  "key_id": None, "hit": None, "verified_equal_outputs": None,
                  "context_path": None, "cache_file": None})
    doc["timing"]["session_init"]["context_build_ms"] = None
    doc["timing"]["session_init"]["ep_context_state"] = "cold"
    assert [i for i in vr.validate_document(doc, validators, path)
            if i.code == "session_init_not_applicable"] == []
    doc["config"]["init_breakdown"] = False
    assert any(i.code == "session_init_not_applicable"
               for i in vr.validate_document(doc, validators, path))


def test_cpp_engine_examples_record_every_module_check():
    doc = load(B8_120)
    checks = doc["execution"]["verification_details"]["module_checks"]
    stages = [c["stage"] for c in checks]
    assert stages[0] == "after_provider_creation" and stages[-1] == "placement_session"
    assert [c["trial_index"] for c in checks if c["stage"] == "after_session_creation"] == \
        list(range(doc["config"]["trials"]))
    assert all(c["modules_outside_bundle"] == [] for c in checks)
