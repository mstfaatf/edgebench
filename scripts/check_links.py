"""Check that every relative link and #anchor in the repository's Markdown files resolves.

For every Markdown file (the tracked ``*.md`` files, or the files given), every
inline link or image ``[text](target)`` and every reference definition
``[label]: target`` outside fenced code blocks and inline code:

- links with a scheme (``https:``, ``mailto:``, ...) are not checked;
- a path must exist, relative to the linking file (or to the repository root
  for a path starting with ``/``), and stay inside the repository;
- a ``#fragment`` on a Markdown target (or a bare ``#fragment``) must be the
  anchor GitHub generates for one of the target's headings: lower case, inline
  code, emphasis and link markup removed, punctuation other than ``-`` and
  ``_`` dropped, spaces turned into ``-``, and ``-1``, ``-2`` ... appended to
  repeated headings.

Standard library only; runs anywhere (CI included). Exit status: 0 every link
resolves, 1 broken links found, 2 usage error.

Usage:
    python scripts/check_links.py [--root DIR] [FILE ...]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]

FENCE = re.compile(r"^\s{0,3}(```|~~~)")
INLINE_CODE = re.compile(r"(`+)(.+?)\1")
# [text](target "title") and ![alt](target); the text may hold nested brackets once.
INLINE_LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


@dataclass(frozen=True)
class BrokenLink:
    file: str
    line: int
    target: str
    reason: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.target}: {self.reason}"


def _strip_markup(text: str) -> str:
    text = INLINE_CODE.sub(lambda m: m.group(2), text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links keep their text
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    return text.replace("*", "").replace("~", "")


def slug(heading: str) -> str:
    """The anchor GitHub generates for a heading's text (before de-duplication)."""
    text = _strip_markup(heading).strip().lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch in "-_ ")
    return text.replace(" ", "-")


def anchors(markdown: str) -> set[str]:
    """Every anchor of a Markdown document, with GitHub's -1, -2 suffixes for repeats."""
    seen: dict[str, int] = {}
    out: set[str] = set()
    in_fence = False
    for line in markdown.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING.match(line)
        if not m:
            continue
        base = slug(m.group(2))
        n = seen.get(base, 0)
        out.add(base if n == 0 else f"{base}-{n}")
        seen[base] = n + 1
    # Explicit HTML anchors.
    out.update(re.findall(r"<a\s+(?:name|id)=\"([^\"]+)\"", markdown))
    return out


def links(markdown: str) -> list[tuple[int, str]]:
    """(line number, target) of every link outside code."""
    out = []
    in_fence = False
    for number, line in enumerate(markdown.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        code_free = INLINE_CODE.sub(lambda m: " " * len(m.group(0)), line)
        out += [(number, m.group(1)) for m in INLINE_LINK.finditer(code_free)]
        m = REFERENCE.match(code_free)
        if m:
            out.append((number, m.group(1)))
    return out


def check_file(path: Path, root: Path = REPO_ROOT,
               cache: dict[Path, set[str]] | None = None) -> list[BrokenLink]:
    cache = {} if cache is None else cache
    text = path.read_text(encoding="utf-8")
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = path.resolve().as_posix()
    broken = []

    def doc_anchors(target: Path) -> set[str]:
        if target not in cache:
            cache[target] = anchors(target.read_text(encoding="utf-8"))
        return cache[target]

    for number, target in links(text):
        if SCHEME.match(target) or target.startswith("//"):
            continue
        location, _, fragment = target.partition("#")
        location = unquote(location)
        if location:
            base = root if location.startswith("/") else path.parent
            resolved = (base / location.lstrip("/")).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                broken.append(BrokenLink(rel, number, target, "points outside the repository"))
                continue
            if not resolved.exists():
                broken.append(BrokenLink(rel, number, target, "no such file or directory"))
                continue
        else:
            resolved = path.resolve()
        if fragment and resolved.is_file() and resolved.suffix.lower() == ".md":
            if unquote(fragment).lower() not in doc_anchors(resolved):
                broken.append(BrokenLink(rel, number, target,
                                         f"no heading with anchor #{fragment} in "
                                         f"{resolved.relative_to(root.resolve()).as_posix()}"))
    return broken


def tracked_markdown(root: Path = REPO_ROOT) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "*.md"], cwd=root, check=True,
                         capture_output=True).stdout.decode("utf-8")
    return [root / p for p in out.split("\0") if p and not p.startswith("third_party/")]


def check(paths: list[Path], root: Path = REPO_ROOT) -> list[BrokenLink]:
    cache: dict[Path, set[str]] = {}
    broken = []
    for p in paths:
        broken += check_file(p, root, cache)
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path, help="Markdown files (default: all tracked)")
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="the root links must stay inside (default: the repository)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        files = [f.resolve() for f in args.files] if args.files else tracked_markdown()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: cannot list the Markdown files: {exc}", file=sys.stderr)
        return 2
    missing = [f for f in files if not f.is_file()]
    if missing:
        print(f"error: no such file: {missing[0]}", file=sys.stderr)
        return 2
    broken = check(files, args.root.resolve())
    for b in broken:
        print(f"error: {b}")
    total = sum(len(links(f.read_text(encoding="utf-8"))) for f in files)
    print(f"Checked {total} links in {len(files)} Markdown files: "
          + ("OK" if not broken else f"{len(broken)} broken"))
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
