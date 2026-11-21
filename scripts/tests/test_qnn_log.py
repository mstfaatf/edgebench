"""Tests for the QNN EP placement-log parser (scripts/common/qnn_log.py).

The fixtures in ``fixtures/qnn_logs/`` are trimmed copies of real ONNX Runtime
1.32.0 + QNN EP 2.8.0 verbose session logs (ANSI colour codes included):

- ``det_non_strict.log`` / ``det_strict.log``: ``tools/qnn_rejection_check.py``,
  a ``Relu -> Det -> Relu`` model whose ``Det`` node the QNN EP rejects, run
  without and with ``session.disable_cpu_ep_fallback=1``;
- ``s8w_all_on_qnn.log``: the per-channel INT8 MobileNetV2, fully on QNN.

Regenerate the first two with
``python tools/qnn_rejection_check.py --save-fixtures scripts/tests/fixtures/qnn_logs``.
"""

from __future__ import annotations

from pathlib import Path

from common.qnn_log import QNN_EP, parse_placement_log

FIXTURES = Path(__file__).parent / "fixtures" / "qnn_logs"
DET_REASON = "Operators of type `Det` are not supported by QNN EP."


def parse(name: str):
    return parse_placement_log((FIXTURES / name).read_text(encoding="utf-8"))


def test_rejected_node_non_strict():
    r = parse("det_non_strict.log")
    assert [p["pass"] for p in r.qnn_passes] == [1, 2]
    for p in r.qnn_passes:
        assert p["passed_units"] == {"Relu": 2} and p["failed_units"] == {"Det": 1}
        assert p["failed_nodes"] == 1
    assert len(r.rejected_nodes) == 1
    node = r.rejected_nodes[0]
    assert (node["op_type"], node["node_name"], node["node_index"]) == ("Det", "unsupported_det", 1)
    assert node["reason"].startswith(DET_REASON)
    assert node["passes"] == [1, 2]
    assert r.rejections == [f"Det node 'unsupported_det' (index 1): {node['reason']}"]
    # The rejected node falls back to the CPU EP; the Relus become two QNN partitions.
    assert r.ort_placements == {"CPUExecutionProvider": 1, QNN_EP: 2}
    assert r.ort_placed_nodes["CPUExecutionProvider"] == ["Det (unsupported_det)"]
    assert len(r.ort_placed_nodes[QNN_EP]) == 2
    assert not r.strict_fallback_error
    assert not r.all_on_qnn


def test_rejected_node_strict():
    r = parse("det_strict.log")
    assert [(n["op_type"], n["node_name"]) for n in r.rejected_nodes] == [("Det", "unsupported_det")]
    assert r.unsupported_nodes == [{"op_type": "Det", "node_name": "unsupported_det"}]
    assert r.strict_fallback_error
    assert r.ort_placements == {}  # the session never got as far as placement
    assert not r.all_on_qnn
    # Warnings and errors have their ANSI colour codes stripped.
    assert any(w.startswith("Unsupported nodes in QNN EP") for w in r.warnings)
    assert not any("\x1b" in w for w in r.warnings)


def test_all_on_qnn():
    r = parse("s8w_all_on_qnn.log")
    assert r.ort_placements == {QNN_EP: 1}
    assert [(p["passed_nodes"], p["failed_nodes"]) for p in r.qnn_passes] == [(313, 0), (319, 0)]
    assert r.qnn_passes[1]["passed_units"]["Conv"] == 52
    assert r.qnn_passes[1]["passed_units"]["Transpose"] == 2
    assert r.rejected_nodes == [] and r.unsupported_nodes == []
    assert r.all_on_qnn


def test_lines_without_pass_markers_still_count():
    log = ("2026-01-01 00:00:00.0 [V:onnxruntime:, qnn_execution_provider.cc:1 f] "
           "Validation PASSED for 5 nodes in OrtNodeUnit (Conv) :\n"
           "2026-01-01 00:00:00.0 [V:onnxruntime:, session_state.cc:1 f]  "
           "All nodes placed on [QNNExecutionProvider]. Number of nodes: 1\n")
    r = parse_placement_log(log)
    assert r.qnn_passes[0]["passed_nodes"] == 5
    assert r.all_on_qnn


def test_parse_qnn_profile_csv():
    from common.qnn_log import parse_qnn_profile_csv

    header = "Msg Timestamp,Message,Time,Unit of Measurement,Timing Source,Event Level,Event Identifier\n"
    rows = []
    for accel, qnn_exec in [(270, 400), (280, 420), (275, 410)]:
        rows += [f"0,BACKEND,{accel},US,BACKEND,ROOT,Accelerator (execute) time\n",
                 f"0,BACKEND,{accel + 15},US,BACKEND,ROOT,QNN accelerator (execute) time\n",
                 f"0,BACKEND,{accel + 25},US,BACKEND,ROOT,RPC (execute) time\n",
                 f"0,EXECUTE,{qnn_exec},US,BACKEND,ROOT,QNN (execute) time\n",
                 "0,BACKEND,700000,CYCLES,BACKEND,ROOT,Accelerator (execute) time (cycles)\n",
                 "0,NODE,1000,CYCLES,BACKEND,SUB-EVENT,conv_1:OpId_3 (cycles)\n"]
    rows.append("0,FINALIZE,157485,US,BACKEND,ROOT,QNN (finalize) time\n")
    r = parse_qnn_profile_csv(header + "".join(rows))
    assert r["inferences"] == 3
    assert r["accelerator_execute_us"] == 275
    assert r["qnn_accelerator_execute_us"] == 290
    assert r["rpc_execute_us"] == 300
    assert r["qnn_execute_us"] == 410
    assert r["accelerator_execute_excluding_wait_us"] is None
    assert r["op_cycles_median"] == {"conv_1:OpId_3 (cycles)": 1000}
