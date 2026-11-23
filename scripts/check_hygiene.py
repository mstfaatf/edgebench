"""Repository hygiene check: tracked files and commit messages.

Fails (exit 1) when:

* a local-only notes file (``CLAUDE.md``, see LOCAL_ONLY_FILES) is tracked by git;
* a tracked text file contains internal development-process wording (see
  FORBIDDEN below): phase or prompt numbers, names of AI tools or assistants, or
  commit attribution trailers;
* a tracked file is larger than MAX_FILE_BYTES and not in LARGE_FILE_ALLOWLIST;
* the message of a commit that is not yet on the base branch (default
  ``origin/main``) contains forbidden wording.

Commit messages already on the base branch are scanned and reported, but never
fail the check: published history is not rewritten.

A single line can be exempted by putting ``hygiene: allow`` on it. Use this only
for a genuine false positive, never to keep process wording.

Standard library only; runs on any OS from anywhere inside the repository:

    python scripts/check_hygiene.py [--base origin/main]

Exit codes: 0 clean, 1 findings, 2 usage error or git unavailable.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- rules
# The forbidden patterns, in one place. All are matched case-insensitively.
FORBIDDEN: dict[str, re.Pattern[str]] = {
    "phase_number": re.compile(r"\bphase[\s_-]*\d+", re.IGNORECASE),
    "prompt_number": re.compile(r"\bprompt[\s_-]*\d+(?:\.\d+)*\b", re.IGNORECASE),
    "ai_tool": re.compile(
        r"\b(?:claude|anthropic|chatgpt|openai|copilot|gemini|codex|gpt-?[345o]\w*"
        r"|ai[\s-]+assistant|ai[\s-]+generated|generated\s+(?:by|with)\s+(?:an?\s+)?ai)\b",
        re.IGNORECASE,
    ),
    "attribution": re.compile(r"co-authored-by", re.IGNORECASE),
}

EXEMPT_MARKER = "hygiene: allow"

# Files that must never be tracked (local working notes).
LOCAL_ONLY_FILES = {"CLAUDE.md"}

# Tracked files that are not scanned for wording: this check's own word list.
WORDING_EXCLUDED = {"scripts/check_hygiene.py"}
EXCLUDED_PREFIXES = ("third_party/",)

MAX_FILE_BYTES = 1024 * 1024
# Text files that are large by nature. Each entry is reviewed: the full ImageNetV2
# manifest has one row (path, label, SHA256) per image, 10,000 rows.
LARGE_FILE_ALLOWLIST: set[str] = {"datasets/imagenetv2-matched-frequency/eval-full.tsv"}

# Bytes read to decide whether a file is binary (a NUL byte means binary).
BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class Finding:
    where: str  # "path:line", "path" or "commit <sha>"
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.where}: [{self.rule}] {self.detail}"


# --------------------------------------------------------------------------- scanning


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """Return (line number, rule, matched text) for every forbidden match in text."""
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        if EXEMPT_MARKER in line:
            continue
        for rule, pattern in FORBIDDEN.items():
            for match in pattern.finditer(line):
                hits.append((number, rule, match.group(0)))
    return hits


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:BINARY_SNIFF_BYTES]


def check_files(root: Path, paths: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for rel in paths:
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        if Path(rel).name.lower() in {name.lower() for name in LOCAL_ONLY_FILES}:
            findings.append(Finding(rel, "local_only_file",
                                    "local-only file is tracked; git rm --cached it"))
            continue
        path = root / rel
        if not path.is_file():  # deleted in the working tree, or a submodule
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES and rel not in LARGE_FILE_ALLOWLIST:
            findings.append(Finding(rel, "large_file",
                                    f"{size} bytes (limit {MAX_FILE_BYTES})"))
        for _, rule, text in scan_text(rel):
            findings.append(Finding(rel, rule, f"in the file name: {text!r}"))
        if rel in WORDING_EXCLUDED:
            continue
        data = path.read_bytes()
        if is_binary(data):
            continue
        for number, rule, text in scan_text(data.decode("utf-8", errors="replace")):
            findings.append(Finding(f"{rel}:{number}", rule, repr(text)))
    return findings


def check_messages(commits: list[tuple[str, str]]) -> list[Finding]:
    findings = []
    for sha, message in commits:
        for number, rule, text in scan_text(message):
            findings.append(Finding(f"commit {sha[:12]} line {number}", rule, repr(text)))
    return findings


# --------------------------------------------------------------------------- git


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace")


def tracked_files(root: Path) -> list[str]:
    return [p for p in git(root, "ls-files", "-z").split("\0") if p]


def commits(root: Path, revision_range: str) -> list[tuple[str, str]]:
    out = git(root, "log", "--format=%H%x00%B%x1e", revision_range)
    result = []
    for record in out.split("\x1e"):
        record = record.strip("\n")
        if record:
            sha, _, message = record.partition("\0")
            result.append((sha, message))
    return result


def ref_exists(root: Path, ref: str) -> bool:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "--quiet",
                           ref + "^{commit}"], capture_output=True).returncode == 0


# --------------------------------------------------------------------------- main


def make_output_safe() -> None:
    """Never fail on printing: replace characters the console encoding cannot show.

    On Windows, stdout of a child process (or a console with a legacy code page)
    may use a code page such as cp1252, which cannot encode every path.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="origin/main",
                        help="branch whose commits only produce a report (default: origin/main)")
    make_output_safe()
    args = parser.parse_args(argv)

    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
        files = tracked_files(root)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"error: cannot run git: {e}", file=sys.stderr)
        return 2

    failures = check_files(root, files)
    print(f"Checked {len(files)} tracked files: {len(failures)} finding(s).")

    if ref_exists(root, args.base):
        new = commits(root, f"{args.base}..HEAD")
        history = commits(root, args.base)
        new_findings = check_messages(new)
        history_findings = check_messages(history)
        print(f"Checked {len(new)} commit message(s) not on {args.base}: "
              f"{len(new_findings)} finding(s).")
        print(f"Checked {len(history)} commit message(s) on {args.base} (reported only): "
              f"{len(history_findings)} finding(s).")
        failures += new_findings
        for finding in history_findings:
            print(f"note: {finding}")
    else:
        all_commits = commits(root, "HEAD")
        history_findings = check_messages(all_commits)
        print(f"{args.base} not found: checked {len(all_commits)} commit message(s) "
              f"reachable from HEAD (reported only): {len(history_findings)} finding(s).")
        for finding in history_findings:
            print(f"note: {finding}")

    for finding in failures:
        print(f"error: {finding}")
    print("Result: " + ("FAILED" if failures else "OK"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
