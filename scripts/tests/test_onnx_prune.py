"""Tests for scripts/common/onnx_prune.py (skipped when onnx is not installed)."""

from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from common.onnx_prune import prune  # noqa: E402


def scalar(name: str, value: float):
    return helper.make_tensor(name, TensorProto.FLOAT, [], [value])


def make_model(nodes, initializers=(), outputs=("y",)):
    graph = helper.make_graph(
        list(nodes),
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info(o, TensorProto.FLOAT, [1]) for o in outputs],
        initializer=list(initializers),
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_removes_dead_chain_iteratively_and_its_initializers():
    model = make_model(
        [
            helper.make_node("Relu", ["x"], ["y"]),
            # Dead chain: Constant -> Add -> Neg, nothing consumes "d2".
            helper.make_node("Constant", [], ["c"], value=scalar("cv", 1.0)),
            helper.make_node("Add", ["c", "w"], ["d1"]),
            helper.make_node("Neg", ["d1"], ["d2"]),
        ],
        initializers=[scalar("w", 2.0), scalar("unused", 3.0)],
    )
    model.graph.value_info.append(helper.make_tensor_value_info("d1", TensorProto.FLOAT, [1]))
    report = prune(model)

    assert [n.op_type for n in model.graph.node] == ["Relu"]
    assert report.removed_nodes == {"Constant": 1, "Add": 1, "Neg": 1}
    assert report.iterations == 3
    assert report.nodes_before == 4 and report.nodes_after == 1
    assert report.removed_initializers == 2 and len(model.graph.initializer) == 0
    assert report.removed_value_info == 1
    onnx.checker.check_model(model)


def test_keeps_graph_outputs_and_live_nodes():
    model = make_model(
        [helper.make_node("Relu", ["x"], ["a"]), helper.make_node("Neg", ["a"], ["y"]),
         helper.make_node("Abs", ["a"], ["z"])],
        outputs=("y", "z"),
    )
    report = prune(model)
    assert len(model.graph.node) == 3
    assert report.iterations == 0 and not report.removed_nodes


def test_tensors_read_by_subgraphs_are_kept():
    then_branch = helper.make_graph(
        [helper.make_node("Identity", ["outer"], ["t"])], "then", [],
        [helper.make_tensor_value_info("t", TensorProto.FLOAT, [1])])
    else_branch = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["e"])], "else", [],
        [helper.make_tensor_value_info("e", TensorProto.FLOAT, [1])])
    model = make_model(
        [
            helper.make_node("Neg", ["x"], ["outer"]),  # only read inside the subgraph
            helper.make_node("Constant", [], ["cond"],
                             value=helper.make_tensor("cv", TensorProto.BOOL, [], [True])),
            helper.make_node("If", ["cond"], ["y"], then_branch=then_branch, else_branch=else_branch),
        ]
    )
    prune(model)
    assert [n.op_type for n in model.graph.node] == ["Neg", "Constant", "If"]
