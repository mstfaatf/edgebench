"""Tests for scripts/validate_doctor.py: the example reports pass, and every rule
fires on a fixture made for it (scripts/tests/fixtures/doctor/invalid/: a base
example, JSON-pointer edits, and the one issue expected)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import validate_doctor as vd

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "docs" / "examples" / "doctor"
NPU_EXAMPLE = EXAMPLES / "example-doctor-npu.json"
RUNNER_EXAMPLE = EXAMPLES / "example-doctor-x64.json"
FIXTURES = sorted((Path(__file__).parent / "fixtures" / "doctor" / "invalid").glob("*.json"))


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_pointer(doc: Any, pointer: str, value: Any) -> None:
    *parents, last = pointer.lstrip("/").split("/")
    node = doc
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def build_fixture(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = load(path)
    doc = load(REPO / spec["base"])
    for pointer, value in spec["set"].items():
        set_pointer(doc, pointer, copy.deepcopy(value))
    return doc, spec


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.json")), ids=lambda p: p.stem)
def test_examples_are_valid(path):
    assert vd.validate_document(load(path)) == []


def test_runner_example_passes_the_runner_rules():
    assert vd.validate_document(load(RUNNER_EXAMPLE), runner=True) == []


def test_npu_example_is_not_a_runner_report():
    # On the NPU machine the NPU checks pass, which a runner must not report.
    issues = vd.validate_document(load(NPU_EXAMPLE), runner=True)
    assert {i.code for i in issues} == {"runner_unexpected_status"}
    assert {i.path for i in issues} == {"/checks/2/status", "/checks/4/status", "/checks/5/status"}


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_invalid_fixture_fails_for_the_intended_reason(path):
    doc, spec = build_fixture(path)
    issues = vd.validate_document(doc, runner=spec.get("runner", False))
    assert len(issues) == 1, [str(i) for i in issues]
    expect = spec["expect"]
    assert issues[0].code == expect["code"]
    assert issues[0].path == expect["path"]


def test_every_rule_has_a_fixture():
    covered = {load(p)["expect"]["code"] for p in FIXTURES}
    assert covered == set(vd.RULES)


def test_schema_errors_are_reported():
    doc = load(NPU_EXAMPLE)
    doc["checks"][0]["status"] = "ok"
    issues = vd.validate_document(doc)
    assert issues and not issues[0].code and issues[0].path == "/checks/0/status"
    assert vd.validate_document({"checks": []})[0].message.startswith("doctor_schema_version")


def test_cli_exit_codes(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(RUNNER_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    assert vd.main([str(good)]) == 0
    assert vd.main(["--runner", str(good)]) == 0
    bad = tmp_path / "bad.json"
    doc = load(RUNNER_EXAMPLE)
    doc["overall"]["exit_code"] = 4
    bad.write_text(json.dumps(doc), encoding="utf-8")
    assert vd.main([str(bad)]) == 1
    assert "overall_exit_code_mismatch" in capsys.readouterr().out
    # A byte-order mark (PowerShell 5.1 redirection) is accepted.
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + RUNNER_EXAMPLE.read_bytes())
    assert vd.main([str(bom)]) == 0
    assert vd.main([str(tmp_path / "missing.json")]) == 2


def test_results_validator_skips_doctor_reports():
    import validate_results as vr

    assert vr.is_accuracy_document(NPU_EXAMPLE)
