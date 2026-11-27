"""Validate suite configuration files and suite manifests (docs/suite.md).

A document with ``suite_config_version`` is checked against
schemas/suite-config.v<version>.schema.json; one with ``suite_manifest_version``
against schemas/suite-manifest.v<version>.schema.json, plus these rules:

- suite.config_sha256 is the SHA256 of suite.config serialized canonically;
- counts match the runs' statuses; run IDs are unique, short_id is the first 12
  hex digits; order_index runs 0..n-1 in the order of the list;
- every configuration has the suite's number of repetitions, 0..R-1;
- rotation "rotated": round r (repetition r) runs every configuration once,
  in one block; "blocked": the repetitions of a configuration are one block;
- preflight_ever_skipped is true exactly if an invocation had skip_preflight,
  and then publishable is false;
- a done run's result file exists, has the recorded SHA256, and its suite block
  names this suite, run, repetition and order (skipped with --no-files).

The engine's own checks of a configuration (model files, matrix combinations,
the rules of ``edgebench run``) need the engine: ``edgebench suite CONFIG --dry-run``.

Standard library and jsonschema only; runs anywhere.

Usage:
    python scripts/validate_suite.py FILE_OR_DIR [...] [--no-files]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
MANIFEST_NAME = "suite-manifest.json"
KINDS = {"suite_config_version": "suite-config", "suite_manifest_version": "suite-manifest"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: Any) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def kind_of(doc: Any) -> tuple[str, str] | None:
    """(schema name, version) of a suite document, or None."""
    if isinstance(doc, dict):
        for key, kind in KINDS.items():
            if key in doc:
                return kind, str(doc[key])
    return None


def schema_issues(doc: Any, kind: str, version: str) -> list[str]:
    path = SCHEMA_DIR / f"{kind}.v{version}.schema.json"
    if not path.is_file():
        return [f"/: unsupported {kind} version {version!r}"]
    schema = json.loads(path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    out = []
    for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        where = "/" + "/".join(str(p) for p in e.absolute_path)
        out.append(f"{where}: {e.message[:200]}")
    return out


def manifest_issues(m: dict[str, Any], directory: Path | None) -> list[str]:
    out: list[str] = []
    add = out.append
    if config_hash(m["suite"]["config"]) != m["suite"]["config_sha256"]:
        add("/suite/config_sha256: is not the SHA256 of suite.config")
    runs = m["runs"]
    counts = Counter(r["status"] for r in runs)
    for status, n in m["counts"].items():
        if counts.get(status, 0) != n:
            add(f"/counts/{status}: {n}, the runs say {counts.get(status, 0)}")
    ids = [r["run_id"] for r in runs]
    if len(set(ids)) != len(ids):
        add("/runs: run IDs are not unique")
    for i, r in enumerate(runs):
        if r["order_index"] != i:
            add(f"/runs/{i}/order_index: {r['order_index']}, expected {i}")
        if r["short_id"] != r["run_id"][:12]:
            add(f"/runs/{i}/short_id: is not the first 12 digits of run_id")
    reps: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        reps[r["configuration_id"]].append(r["repetition"])
    want = list(range(m["repetitions"]))
    for cid, got in reps.items():
        if sorted(got) != want:
            add(f"/runs: configuration {cid[:12]} has repetitions {sorted(got)}, expected {want}")
    n = len(reps)
    if m["rotation"] == "rotated":
        for start in range(0, len(runs), max(n, 1)):
            block = runs[start:start + n]
            r0 = start // max(n, 1)
            if {r["configuration_id"] for r in block} != set(reps) or any(
                    r["repetition"] != r0 for r in block):
                add(f"/runs/{start}: round {r0} does not run every configuration once")
    else:
        seen: list[str] = []
        for r in runs:
            if not seen or seen[-1] != r["configuration_id"]:
                if r["configuration_id"] in seen:
                    add(f"/runs: configuration {r['configuration_id'][:12]} is not one block")
                seen.append(r["configuration_id"])
    skipped = any(i["skip_preflight"] for i in m["invocations"])
    if m["preflight_ever_skipped"] != skipped:
        add(f"/preflight_ever_skipped: {m['preflight_ever_skipped']}, the invocations say {skipped}")
    if skipped and m["publishable"]:
        add("/publishable: a suite that skipped its pre-flight check is not publishable")
    if m["publishable"] != (not m["not_publishable_reasons"]):
        add("/publishable: disagrees with not_publishable_reasons")
    if directory is None:
        return out
    for i, r in enumerate(runs):
        if r["status"] != "done":
            continue
        path = directory / r["result_file"]
        if not path.is_file():
            add(f"/runs/{i}/result_file: {r['result_file']} does not exist")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["result_sha256"]:
            add(f"/runs/{i}/result_sha256: does not match {r['result_file']}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        link = doc.get("suite")
        want_link = {"suite_name": m["suite"]["name"],
                     "suite_config_hash": m["suite"]["config_sha256"], "run_id": r["run_id"],
                     "repetition": r["repetition"], "order_index": r["order_index"],
                     "preflight_skipped": m["preflight_ever_skipped"]}
        if link is None or any(link.get(k) != v for k, v in want_link.items()):
            add(f"/runs/{i}: the result's suite block {link} does not name this run")
    return out


def validate_file(path: Path, check_files: bool = True) -> list[str]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return [f"cannot read: {exc}"]
    kind = kind_of(doc)
    if kind is None:
        return ["not a suite configuration or manifest"]
    issues = schema_issues(doc, *kind)
    if not issues and kind[0] == "suite-manifest":
        issues = manifest_issues(doc, path.parent if check_files else None)
    return issues


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*.json")
                            if kind_of(_load(f)) is not None)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, metavar="FILE_OR_DIR")
    parser.add_argument("--no-files", action="store_true",
                        help="do not check the result files a manifest lists")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    try:
        files = collect(args.paths)
    except FileNotFoundError as exc:
        print(f"error: no such file or directory: {exc}", file=sys.stderr)
        return 2
    failed = 0
    for path in files:
        issues = validate_file(path, not args.no_files)
        failed += bool(issues)
        print(f"{'FAIL' if issues else 'PASS'} {path}")
        for i in issues:
            print(f"  {i}")
    print(f"\n{len(files) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
