"""Print per-file line coverage from a Cobertura XML report as a Markdown table.

    python scripts/coverage_summary.py coverage.xml [--prefix engine/src]

Files are listed relative to the repository with forward slashes; --prefix keeps
only files under that directory. Lines that appear in several classes of the
report (e.g. inline functions) are counted once, covered if any entry has hits.
Standard library only; reports only, never fails on low coverage.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def normalize(filename: str, sources: list[str]) -> str:
    """Repository-relative path with forward slashes (best effort)."""
    path = filename.replace("\\", "/")
    for source in sources:
        root = source.replace("\\", "/").rstrip("/") + "/"
        if path.lower().startswith(root.lower()):
            path = path[len(root):]
            break
    return path


def line_coverage(report: Path) -> dict[str, tuple[int, int]]:
    """Map file -> (covered lines, coverable lines)."""
    tree = ET.parse(report)
    sources = [s.text or "" for s in tree.getroot().iter("source")]
    lines: dict[str, dict[int, bool]] = {}
    for cls in tree.getroot().iter("class"):
        name = normalize(cls.get("filename", ""), sources)
        per_file = lines.setdefault(name, {})
        for line in cls.iter("line"):
            number = int(line.get("number", "0"))
            hit = int(line.get("hits", "0")) > 0
            per_file[number] = per_file.get(number, False) or hit
    return {name: (sum(v.values()), len(v)) for name, v in lines.items()}


def find_suffix(path: str, prefix: str) -> str | None:
    """The part of path starting at prefix (paths may be absolute), or None."""
    prefix = prefix.replace("\\", "/").strip("/") + "/"
    index = path.lower().find(prefix.lower())
    if index != 0 and (index < 0 or path[index - 1] != "/"):
        return None
    return path[index:]


def markdown(coverage: dict[str, tuple[int, int]], prefix: str | None) -> str:
    rows = []
    for name, (covered, total) in coverage.items():
        shown = find_suffix(name, prefix) if prefix else name
        if shown is None or total == 0:
            continue
        rows.append((shown, covered, total))
    rows.sort()
    covered_sum = sum(r[1] for r in rows)
    total_sum = sum(r[2] for r in rows)
    out = ["| File | Lines covered | Line coverage |", "|---|---:|---:|"]
    for shown, covered, total in rows:
        out.append(f"| `{shown}` | {covered} / {total} | {100.0 * covered / total:.1f}% |")
    if total_sum:
        out.append(f"| **Total ({len(rows)} files)** | **{covered_sum} / {total_sum}** | "
                   f"**{100.0 * covered_sum / total_sum:.1f}%** |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path, help="Cobertura XML file")
    parser.add_argument("--prefix", help="only files under this directory, e.g. engine/src")
    args = parser.parse_args(argv)
    if not args.report.is_file():
        print(f"error: {args.report} does not exist", file=sys.stderr)
        return 2
    print(markdown(line_coverage(args.report), args.prefix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
