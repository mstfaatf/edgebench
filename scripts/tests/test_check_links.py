"""The Markdown link check (check_links.py): anchors, paths and the repository itself."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import check_hygiene
import check_links as cl

SCRIPT = Path(__file__).resolve().parents[1] / "check_links.py"


@pytest.mark.parametrize("heading, anchor", [
    ("Run doctor first", "run-doctor-first"),
    ("Accelerator time (`--profile-accelerator`)", "accelerator-time---profile-accelerator"),
    ("`edgebench doctor`", "edgebench-doctor"),
    ("1. Download the datasets", "1-download-the-datasets"),
    ("What the C++ tests prove", "what-the-c-tests-prove"),
    ("Batch 8 on the NPU", "batch-8-on-the-npu"),
    ("x64 and Ninja builds (CPU only)", "x64-and-ninja-builds-cpu-only"),
    ("A [linked](x.md) **bold** heading", "a-linked-bold-heading"),
])
def test_slug_follows_github(heading, anchor):
    assert cl.slug(heading) == anchor


def test_repeated_headings_get_numbered_anchors():
    doc = "# Options\n\n## Options\n\n```\n# not a heading\n```\n\n### Options\n"
    assert cl.anchors(doc) == {"options", "options-1", "options-2"}


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_finds_broken_paths_and_anchors(tmp_path):
    write(tmp_path / "docs" / "b.md", "# Title\n\n## Real section\n")
    write(tmp_path / "data.txt", "x")
    a = write(tmp_path / "docs" / "a.md", "\n".join([
        "# A",
        "[ok](b.md#real-section) [ok too](#a) [file](../data.txt) [web](https://example.com/x#y)",
        "[missing file](c.md) [missing anchor](b.md#nope) [self](#nowhere)",
        "`[in code](missing.md)` and an image ![img](../nope.png)",
        "```",
        "[in a fence](missing.md)",
        "```",
        "[ref]: b.md#also-nope",
        "[outside](../../outside.md)",
    ]))
    broken = cl.check_file(a, tmp_path)
    found = {(b.line, b.target) for b in broken}
    assert found == {(3, "c.md"), (3, "b.md#nope"), (3, "#nowhere"), (4, "../nope.png"),
                     (8, "b.md#also-nope"), (9, "../../outside.md")}
    assert any("outside the repository" in b.reason for b in broken)


def test_the_repository_has_no_broken_links():
    assert cl.check(cl.tracked_markdown()) == []


def test_hygiene_reports_broken_links(tmp_path):
    write(tmp_path / "README.md", "[gone](docs/gone.md)\n")
    findings = check_hygiene.check_markdown_links(tmp_path, ["README.md"])
    assert [f.rule for f in findings] == ["broken_link"]


def test_cli_exit_codes(tmp_path):
    good = write(tmp_path / "good.md", "# G\n[x](#g)\n")
    bad = write(tmp_path / "bad.md", "[x](#missing)\n")
    run = lambda *a: subprocess.run([sys.executable, str(SCRIPT), "--root", str(tmp_path),  # noqa: E731
                                     *map(str, a)],
                                    capture_output=True, text=True)
    assert run(good).returncode == 0
    r = run(bad)
    assert r.returncode == 1 and "#missing" in r.stdout
    assert run(tmp_path / "none.md").returncode == 2
