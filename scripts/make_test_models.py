"""Generate the tiny ONNX models used by the C++ tests.

Writes the models to engine/tests/fixtures/models/ (committed, a few hundred
bytes each):

- fixed_float32.onnx: input "x" float32 [1, 4] -> MatMul (4x3) -> Add -> Relu ->
  output "y" float32 [1, 3]. Every dimension is fixed.
- dynamic_batch.onnx: the same graph with input/output dimension 0 named "batch".
- int64_input.onnx: input "ids" int64 [1, 4] -> Cast to float32 -> output "y" [1, 4].
- nan_output.onnx: input "x" float32 [1, 4] -> Sqrt(x - 2) -> output "y" [1, 4]. Every
  generated input lies in [0, 1), so every output value is NaN.
- runtime_error.onnx: input "x" float32 [1, 4] -> Gather(axis 1, index 7) -> output "y"
  [1, 1]. The session is created, but every Run() fails: the index is out of range.

Weights are fixed constants, so the output is byte-identical on every run.
Only needs the onnx package (no ONNX Runtime), so it runs on any platform:

    python scripts/make_test_models.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "engine" / "tests" / "fixtures" / "models"
OPSET = 17
IR_VERSION = 8

# 4x3 weights and a bias with simple, exactly representable values.
WEIGHTS = [0.5, -1.0, 0.25,
           1.0, 0.5, -0.5,
           -0.25, 1.0, 0.75,
           0.125, -0.5, 1.0]
BIAS = [0.1, -0.2, 0.3]


def _model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="edgebench-test-fixtures")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model


def dense_relu(batch: int | str) -> onnx.ModelProto:
    weights = helper.make_tensor("W", TensorProto.FLOAT, [4, 3], WEIGHTS)
    bias = helper.make_tensor("B", TensorProto.FLOAT, [3], BIAS)
    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["xw"], name="matmul"),
        helper.make_node("Add", ["xw", "B"], ["xwb"], name="add"),
        helper.make_node("Relu", ["xwb"], ["y"], name="relu"),
    ]
    graph = helper.make_graph(
        nodes, "dense_relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, 3])],
        initializer=[weights, bias])
    return _model(graph)


def int64_input() -> onnx.ModelProto:
    nodes = [helper.make_node("Cast", ["ids"], ["y"], name="cast", to=TensorProto.FLOAT)]
    graph = helper.make_graph(
        nodes, "int64_input",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
    return _model(graph)


def nan_output() -> onnx.ModelProto:
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    nodes = [
        helper.make_node("Sub", ["x", "two"], ["d"], name="sub"),
        helper.make_node("Sqrt", ["d"], ["y"], name="sqrt"),
    ]
    graph = helper.make_graph(
        nodes, "nan_output",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializer=[two])
    return _model(graph)


def runtime_error() -> onnx.ModelProto:
    index = helper.make_tensor("index", TensorProto.INT64, [1], [7])
    nodes = [helper.make_node("Gather", ["x", "index"], ["y"], name="gather", axis=1)]
    graph = helper.make_graph(
        nodes, "runtime_error",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
        initializer=[index])
    return _model(graph)


MODELS = {
    "fixed_float32.onnx": lambda: dense_relu(1),
    "dynamic_batch.onnx": lambda: dense_relu("batch"),
    "int64_input.onnx": int64_input,
    "nan_output.onnx": nan_output,
    "runtime_error.onnx": runtime_error,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, build in MODELS.items():
        path = args.output_dir / name
        path.write_bytes(build().SerializeToString())
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
