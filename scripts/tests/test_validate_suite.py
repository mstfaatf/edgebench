"""Suite configuration and manifest schemas, and scripts/validate_suite.py.

The configuration fixtures in fixtures/suite/ are shared with the engine's tests
(engine/tests/suite_test.cpp): valid/ holds configurations both accept; each
invalid/ fixture says whether the schema alone can reject it ("expect.schema")
and which error kind the engine reports ("expect.engine_kind").
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import validate_results as vr
import validate_suite as vs

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "suite"
VALID = sorted((FIXTURES / "valid").glob("*.json"))
INVALID = sorted((FIXTURES / "invalid").glob("*.json"))
EXAMPLE_MANIFEST = REPO / "docs" / "examples" / "suite" / "example-suite-manifest.json"
CONFIGS = sorted((REPO / "benchmarks" / "configs").glob("*.json"))

# The canonical hash of valid/explicit_cpu.json; the engine's test pins the same value.
EXPLICIT_CPU_HASH = "3234ed5a4fe45e164cf6e0e42aced9fc3b0eb382a1321afe8918062cbc04f571"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", VALID + CONFIGS, ids=lambda p: p.name)
def test_valid_configurations_pass_the_schema(path):
    assert vs.validate_file(path) == []


@pytest.mark.parametrize("path", INVALID, ids=lambda p: p.stem)
def test_invalid_fixture_schema_verdict(path):
    fixture = load(path)
    issues = vs.schema_issues(fixture["config"], "suite-config", "1.0.0")
    if fixture["expect"]["schema"] == "invalid":
        assert issues, f"{path.name}: the schema should reject it ({fixture['description']})"
    else:
        assert issues == [], f"{path.name}: only the engine can reject it, the schema says {issues}"


def test_every_invalid_fixture_names_an_engine_kind():
    kinds = {load(p)["expect"]["engine_kind"] for p in INVALID}
    assert {"invalid_combination", "unknown_model", "model_file_mismatch", "unmatched_exclude",
            "duplicate_configuration", "no_model_file"} <= kinds
    assert len(INVALID) >= 15


def test_config_hash_is_canonical_and_pinned():
    doc = load(FIXTURES / "valid" / "explicit_cpu.json")
    assert vs.config_hash(doc) == EXPLICIT_CPU_HASH
    reordered = json.loads(json.dumps(doc, sort_keys=True, indent=7))
    assert vs.config_hash(reordered) == EXPLICIT_CPU_HASH
    doc["repetitions"] = 3
    assert vs.config_hash(doc) != EXPLICIT_CPU_HASH


def manifest():
    return load(EXAMPLE_MANIFEST)


def manifest_issues(doc):
    issues = vs.schema_issues(doc, "suite-manifest", "1.0.0")
    return issues or vs.manifest_issues(doc, None)


def test_example_manifest_passes():
    assert manifest_issues(manifest()) == []


def tampered(change):
    doc = copy.deepcopy(manifest())
    change(doc)
    return manifest_issues(doc)


def test_counts_must_match_the_runs():
    def change(m):
        m["counts"]["done"] += 1
    assert any("/counts/done" in i for i in tampered(change))


def test_order_index_must_follow_the_list():
    def change(m):
        m["runs"][0]["order_index"], m["runs"][1]["order_index"] = 1, 0
    assert any("order_index" in i for i in tampered(change))


def test_config_hash_must_match_the_config():
    def change(m):
        m["suite"]["config"]["repetitions"] = 99
    assert any("config_sha256" in i for i in tampered(change))


def test_rotated_rounds_run_every_configuration_once():
    def change(m):
        m["runs"][0], m["runs"][-1] = m["runs"][-1], m["runs"][0]
        for i, r in enumerate(m["runs"]):
            r["order_index"] = i
    assert any("round" in i for i in tampered(change))


def test_skipped_preflight_is_never_publishable():
    def change(m):
        m["invocations"][0]["skip_preflight"] = True
        m["preflight_ever_skipped"] = True
        m["publishable"] = True
        m["not_publishable_reasons"] = []
    assert any("/publishable" in i for i in tampered(change))


def test_a_done_run_needs_a_validated_result():
    def change(m):
        m["runs"][0]["validation"] = None
    assert tampered(change)


def test_repetitions_must_be_complete():
    def change(m):
        m["runs"] = m["runs"][:-1]
        m["counts"]["done"] -= 1
    assert any("repetitions" in i for i in tampered(change))


def test_results_validator_skips_suite_documents():
    assert vr.is_accuracy_document(EXAMPLE_MANIFEST)
    assert vr.is_accuracy_document(FIXTURES / "valid" / "explicit_cpu.json")


def test_result_files_are_checked_when_present(tmp_path):
    doc = manifest()
    issues = vs.manifest_issues(doc, tmp_path)  # no result files next to the copy
    assert any("does not exist" in i for i in issues)
