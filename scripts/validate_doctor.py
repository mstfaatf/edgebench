"""Validate `edgebench doctor --json` reports against their schema and check their verdicts.

For every given file (or ``*.json`` with ``doctor_schema_version`` in a given
directory, searched recursively): the schema
(``schemas/doctor-report.v<version>.schema.json``), then

- the nine checks appear once each, in report order (``check_order``);
- a warning or failure names what is wrong (``check_code_missing``);
- ``overall.status`` is fail if a check failed, else warn if one warned, else
  pass (``overall_status_mismatch``);
- ``overall.exit_code`` is 0 without a failure, else the first failing check's
  exit code (``overall_exit_code_mismatch``).

With ``--runner`` (CI machines, which have no Qualcomm NPU) also: no check
fails, the NPU checks (backend_versions, npu_probe, npu_context) are
not_applicable or warn, and the exit code is 0 (``runner_unexpected_status``).

Standard library and jsonschema only, so it runs anywhere (CI included).
Exit status: 0 all valid, 1 a check failed, 2 usage error.

Usage:
    python scripts/validate_doctor.py [--runner] FILE_OR_DIR [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
VERSION_KEY = "doctor_schema_version"
CHECK_ORDER = ["ort_bundle", "qnn_bundle", "backend_versions", "devices", "npu_probe",
               "npu_context", "ep_cache_dir", "power", "background_load"]
NPU_CHECKS = {"backend_versions", "npu_probe", "npu_context"}
RULES = ("check_order", "check_code_missing", "overall_status_mismatch",
         "overall_exit_code_mismatch", "runner_unexpected_status")


@dataclass(frozen=True)
class Issue:
    path: str
    message: str
    code: str = ""  # rule; empty for schema errors

    def __str__(self) -> str:
        return f"{self.path or '/'}: {self.message}" + (f" [{self.code}]" if self.code else "")


def load_schema(version: str) -> dict[str, Any]:
    path = SCHEMA_DIR / f"doctor-report.v{version}.schema.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def schema_issues(doc: dict[str, Any]) -> list[Issue]:
    import jsonschema

    version = doc.get(VERSION_KEY)
    if not isinstance(version, str):
        return [Issue(f"/{VERSION_KEY}", f"{VERSION_KEY} is missing")]
    try:
        schema = load_schema(version)
    except FileNotFoundError:
        return [Issue(f"/{VERSION_KEY}", f"unknown {VERSION_KEY} {version!r}")]
    validator = jsonschema.Draft202012Validator(schema)
    issues = []
    for error in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        pointer = "/" + "/".join(str(p) for p in error.absolute_path)
        issues.append(Issue(pointer if pointer != "/" else "", error.message))
    return issues


def rule_issues(doc: dict[str, Any], runner: bool = False) -> list[Issue]:
    issues: list[Issue] = []
    checks = doc["checks"]
    names = [c["name"] for c in checks]
    if names != CHECK_ORDER:
        issues.append(Issue("/checks", f"checks are {names}, expected {CHECK_ORDER}",
                            "check_order"))
    for i, c in enumerate(checks):
        if c["status"] in ("warn", "fail") and not c.get("code"):
            issues.append(Issue(f"/checks/{i}/code", f"{c['name']} is {c['status']} without a code",
                                "check_code_missing"))
    statuses = [c["status"] for c in checks]
    expected = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")
    if doc["overall"]["status"] != expected:
        issues.append(Issue("/overall/status", f"{doc['overall']['status']!r}, but the checks give "
                                               f"{expected!r}", "overall_status_mismatch"))
    first_fail = next((c for c in checks if c["status"] == "fail"), None)
    expected_code = first_fail["exit_code"] if first_fail else 0
    if doc["overall"]["exit_code"] != expected_code:
        issues.append(Issue("/overall/exit_code",
                            f"{doc['overall']['exit_code']}, but "
                            + (f"the first failing check ({first_fail['name']}) has exit code "
                               f"{expected_code}" if first_fail else "no check failed (0)"),
                            "overall_exit_code_mismatch"))
    if runner:
        for i, c in enumerate(checks):
            bad = c["status"] == "fail" or (c["name"] in NPU_CHECKS
                                             and c["status"] not in ("not_applicable", "warn"))
            if bad:
                issues.append(Issue(f"/checks/{i}/status",
                                    f"{c['name']} is {c['status']} on a machine without an NPU "
                                    f"({c['summary']})", "runner_unexpected_status"))
        if doc["overall"]["exit_code"] != 0:
            issues.append(Issue("/overall/exit_code", f"exit code {doc['overall']['exit_code']} on "
                                                      "a runner (expected 0)",
                                "runner_unexpected_status"))
    return issues


def validate_document(doc: Any, runner: bool = False) -> list[Issue]:
    if not isinstance(doc, dict):
        return [Issue("", "not a JSON object")]
    issues = schema_issues(doc)
    if issues:
        return issues
    return rule_issues(doc, runner)


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*.json")):
                try:
                    doc = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(doc, dict) and VERSION_KEY in doc:
                    files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--runner", action="store_true",
                        help="also require what a machine without an NPU must report")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(errors="backslashreplace")
    try:
        files = collect(args.paths)
    except FileNotFoundError as exc:
        print(f"error: no such file or directory: {exc}", file=sys.stderr)
        return 2
    if not files:
        print("No doctor reports found.")
        return 0
    failed = 0
    for f in files:
        try:
            # utf-8-sig: PowerShell 5.1 redirection writes a byte-order mark.
            doc = json.loads(f.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            print(f"FAIL {f}\n  error: invalid JSON: {exc}")
            failed += 1
            continue
        issues = validate_document(doc, args.runner)
        print(("FAIL " if issues else "PASS ") + str(f))
        for issue in issues:
            print(f"  error: {issue}")
        failed += bool(issues)
    print(f"\n{len(files) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
