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
- npu_unsupported_op.onnx: input "x" float32 [1, 4, 4] -> Relu -> Det -> Relu -> output "y"
  float32 [1]. The QNN execution provider implements Relu but not Det, so a strict NPU
  session is refused, and without strict mode Det runs on the CPU between two QNN
  partitions.

Two more go to engine/resources/doctor/, where `edgebench doctor` embeds them
for its NPU probe:

- npu_probe_fp32.onnx: input "x" float32 [1, 3, 8, 8] -> Conv (4 filters 3x3, pads 1,
  bias) -> Relu -> output "y" float32 [1, 4, 8, 8].
- npu_probe_int8.onnx: the same convolution as a static INT8 QDQ model: uint8
  activations (input scale 1/255, output scale 0.05, zero point 0), symmetric int8
  weights (scale 1/64, every FP32 weight is an exact multiple of it) and an int32
  bias, as the QNN HTP backend runs them.

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


def npu_unsupported_op() -> onnx.ModelProto:
    nodes = [
        helper.make_node("Relu", ["x"], ["r"], name="accepted_relu_in"),
        helper.make_node("Det", ["r"], ["d"], name="unsupported_det"),
        helper.make_node("Relu", ["d"], ["y"], name="accepted_relu_out"),
    ]
    graph = helper.make_graph(
        nodes, "npu_unsupported_op",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    return _model(graph)


# The probe convolution: 4 filters of 3x3x3. Every weight is k/64 for an integer
# k in [-24, 24], so it is exact in float32 and in int8 with scale 1/64.
PROBE_SHAPE = [1, 3, 8, 8]
PROBE_OUT_SHAPE = [1, 4, 8, 8]
PROBE_WEIGHT_INT = [((i % 7) - 3) * 8 for i in range(4 * 3 * 3 * 3)]
PROBE_WEIGHT_SCALE = 1.0 / 64.0
PROBE_BIAS = [-0.125, 0.0, 0.125, 0.25]
PROBE_INPUT_SCALE = 1.0 / 255.0
PROBE_OUTPUT_SCALE = 0.05


def _probe_graph(name: str, nodes: list, initializers: list) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, PROBE_SHAPE)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, PROBE_OUT_SHAPE)],
        initializer=initializers)
    return _model(graph)


def npu_probe_fp32() -> onnx.ModelProto:
    weights = helper.make_tensor("W", TensorProto.FLOAT, [4, 3, 3, 3],
                                 [k * PROBE_WEIGHT_SCALE for k in PROBE_WEIGHT_INT])
    bias = helper.make_tensor("B", TensorProto.FLOAT, [4], PROBE_BIAS)
    nodes = [
        helper.make_node("Conv", ["x", "W", "B"], ["c"], name="conv", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"], name="relu"),
    ]
    return _probe_graph("npu_probe_fp32", nodes, [weights, bias])


def npu_probe_int8() -> onnx.ModelProto:
    bias_scale = PROBE_INPUT_SCALE * PROBE_WEIGHT_SCALE
    inits = [
        helper.make_tensor("x_scale", TensorProto.FLOAT, [], [PROBE_INPUT_SCALE]),
        helper.make_tensor("x_zero_point", TensorProto.UINT8, [], [0]),
        helper.make_tensor("W_quantized", TensorProto.INT8, [4, 3, 3, 3], PROBE_WEIGHT_INT),
        helper.make_tensor("W_scale", TensorProto.FLOAT, [], [PROBE_WEIGHT_SCALE]),
        helper.make_tensor("W_zero_point", TensorProto.INT8, [], [0]),
        helper.make_tensor("B_quantized", TensorProto.INT32, [4],
                           [round(b / bias_scale) for b in PROBE_BIAS]),
        helper.make_tensor("B_scale", TensorProto.FLOAT, [], [bias_scale]),
        helper.make_tensor("B_zero_point", TensorProto.INT32, [], [0]),
        helper.make_tensor("y_scale", TensorProto.FLOAT, [], [PROBE_OUTPUT_SCALE]),
        helper.make_tensor("y_zero_point", TensorProto.UINT8, [], [0]),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["x_q"],
                         name="x_quantize"),
        helper.make_node("DequantizeLinear", ["x_q", "x_scale", "x_zero_point"], ["x_dq"],
                         name="x_dequantize"),
        helper.make_node("DequantizeLinear", ["W_quantized", "W_scale", "W_zero_point"], ["W"],
                         name="W_dequantize"),
        helper.make_node("DequantizeLinear", ["B_quantized", "B_scale", "B_zero_point"], ["B"],
                         name="B_dequantize"),
        helper.make_node("Conv", ["x_dq", "W", "B"], ["c"], name="conv", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["r"], name="relu"),
        helper.make_node("QuantizeLinear", ["r", "y_scale", "y_zero_point"], ["y_q"],
                         name="y_quantize"),
        helper.make_node("DequantizeLinear", ["y_q", "y_scale", "y_zero_point"], ["y"],
                         name="y_dequantize"),
    ]
    return _probe_graph("npu_probe_int8", nodes, inits)


PROBE_DIR = REPO_ROOT / "engine" / "resources" / "doctor"
PROBE_MODELS = {
    "npu_probe_fp32.onnx": npu_probe_fp32,
    "npu_probe_int8.onnx": npu_probe_int8,
}

MODELS = {
    "fixed_float32.onnx": lambda: dense_relu(1),
    "dynamic_batch.onnx": lambda: dense_relu("batch"),
    "int64_input.onnx": int64_input,
    "nan_output.onnx": nan_output,
    "runtime_error.onnx": runtime_error,
    "npu_unsupported_op.onnx": npu_unsupported_op,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=PROBE_DIR)
    args = parser.parse_args(argv)
    for directory, models in ((args.output_dir, MODELS), (args.probe_dir, PROBE_MODELS)):
        directory.mkdir(parents=True, exist_ok=True)
        for name, build in models.items():
            path = directory / name
            path.write_bytes(build().SerializeToString())
            print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
