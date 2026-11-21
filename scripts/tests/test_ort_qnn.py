"""Tests for the placement-log parser in scripts/common/ort_qnn.py.

Skipped unless onnxruntime and onnxruntime-qnn are installed (Windows on Arm).
"""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxruntime_qnn")

from common.ort_qnn import QNN_EP, parse_placement_log  # noqa: E402


def line(sev: str, src: str, msg: str) -> str:
    return f"2026-11-20 19:16:10.1 [{sev}:onnxruntime:, {src}:123 onnxruntime::Fn] {msg}"


def test_all_on_qnn():
    log = "\n".join([
        line("V", "qnn_execution_provider.cc", "GetCapability pass 1 (pre-Layout-Transform)"),
        line("V", "qnn_execution_provider.cc", "Validation PASSED for 5 nodes in OrtNodeUnit (Conv) :"),
        line("V", "qnn_execution_provider.cc", "Validation PASSED for 1 nodes in OrtNodeUnit (QuantizeLinear) :"),
        line("V", "qnn_execution_provider.cc", "GetCapability pass 2 (post-Layout-Transform)"),
        line("V", "qnn_execution_provider.cc", "Validation PASSED for 5 nodes in OrtNodeUnit (Conv) :"),
        line("V", "qnn_execution_provider.cc", "Validation PASSED for 3 nodes in OrtNodeUnit (Transpose) :"),
        line("W", "ep_context_io_dispatch.cc", "EPContext encryption config unavailable"),
        line("V", "session_state.cc", "Node placements"),
        line("V", "session_state.cc", " All nodes placed on [QNNExecutionProvider]. Number of nodes: 1"),
        "DSP_INFO UNSUPPORTED_KEY: 49",
    ])
    r = parse_placement_log(log)
    assert r.ort_placements == {QNN_EP: 1}
    assert [p["passed_nodes"] for p in r.qnn_passes] == [6, 8]
    assert r.qnn_passes[1]["passed_units"] == {"Conv": 1, "Transpose": 1}
    assert r.rejections == []
    assert r.warnings == ["EPContext encryption config unavailable"]
    assert r.all_on_qnn


def test_fallback_is_detected():
    log = "\n".join([
        line("V", "qnn_execution_provider.cc", "GetCapability pass 1"),
        line("V", "qnn_execution_provider.cc", "Validation PASSED for 5 nodes in OrtNodeUnit (Conv) :"),
        line("V", "qnn_execution_provider.cc",
             "Validation FAILED for 1 nodes in OrtNodeUnit (Range) : Range: all inputs must be constant"),
        line("V", "session_state.cc", " Node(s) placed on [QNNExecutionProvider]. Number of nodes: 1"),
        line("V", "session_state.cc", " Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1"),
    ])
    r = parse_placement_log(log)
    assert r.ort_placements == {QNN_EP: 1, "CPUExecutionProvider": 1}
    assert r.qnn_passes[0]["failed_units"] == {"Range": 1}
    assert len(r.rejections) == 1 and "Range" in r.rejections[0]
    assert not r.all_on_qnn
