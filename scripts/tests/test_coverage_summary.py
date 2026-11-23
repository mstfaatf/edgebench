"""Tests for scripts/coverage_summary.py (standard library only; runs on any OS)."""

import coverage_summary as cs

REPORT = """<?xml version="1.0" encoding="utf-8"?>
<coverage line-rate="0.5" version="1.9">
  <sources><source>D:</source></sources>
  <packages>
    <package name="edgebench_tests.exe">
      <classes>
        <class name="a" filename="a\\edgebench\\edgebench\\engine\\src\\stats.cpp">
          <lines><line number="1" hits="1"/><line number="2" hits="0"/></lines>
        </class>
        <class name="b" filename="a\\edgebench\\edgebench\\engine\\include\\edgebench\\x.hpp">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
    <package name="edgebench.exe">
      <classes>
        <class name="a" filename="a\\edgebench\\edgebench\\engine\\src\\stats.cpp">
          <lines><line number="2" hits="3"/><line number="3" hits="0"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""


def test_lines_merge_across_packages(tmp_path):
    report = tmp_path / "coverage.xml"
    report.write_text(REPORT)
    coverage = cs.line_coverage(report)
    assert coverage["a/edgebench/edgebench/engine/src/stats.cpp"] == (2, 3)


def test_markdown_filters_by_prefix(tmp_path):
    report = tmp_path / "coverage.xml"
    report.write_text(REPORT)
    table = cs.markdown(cs.line_coverage(report), "engine/src").splitlines()
    assert table[2] == "| `engine/src/stats.cpp` | 2 / 3 | 66.7% |"
    assert table[-1] == "| **Total (1 files)** | **2 / 3** | **66.7%** |"
    assert len(table) == 4


def test_prefix_must_start_a_path_component():
    assert cs.find_suffix("x/engine/src/a.cpp", "engine/src") == "engine/src/a.cpp"
    assert cs.find_suffix("x/myengine/src/a.cpp", "engine/src") is None
