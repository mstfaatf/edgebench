"""The time-series autoencoder: architecture, ONNX stand-ins and a NumPy reference.

One layer list per candidate design is the single definition of the
architecture. The training notebook (``notebooks/train_autoencoder.ipynb``)
builds the Keras model from it; ``scripts/check_autoencoder_npu.py`` builds
untrained ONNX stand-ins from it to check QNN HTP op support before anything is
trained.

Designs (input: one window of ``WINDOW_LENGTH`` normalized points):

``conv1d``
    Keras ``Conv1D`` / ``Conv1DTranspose`` layers, channels-last input
    ``[batch, 128, 1]``: 128 -> 64 -> 32 -> 64 -> 128 time steps.
``dense``
    Keras ``Dense`` layers on a flat input ``[batch, 128]``: 128 -> 64 -> 16 -> 64 -> 128.

Stand-in layouts mirror what tf2onnx emits for these layers. TensorFlow runs a
1-D convolution as a 2-D one on ``[batch, 1, steps, channels]`` (NHWC), so
tf2onnx produces ``Conv`` / ``ConvTranspose`` with a ``1 x k`` kernel between
Transposes; its optimizer usually cancels the Transposes between layers. The
check covers both outcomes and a plain 1-D ``Conv`` as well:

- ``conv1d``: Unsqueeze + Transpose once at the input, 2-D convolutions, one
  Transpose + Squeeze at the output (the optimized tf2onnx layout);
- ``conv1d_per_layer``: Unsqueeze, Transpose, Conv, Transpose, Squeeze around
  every layer, activations in channels-last (the unoptimized layout);
- ``conv1d_native``: 1-D ``Conv`` on ``[batch, channels, steps]`` with a
  Transpose at each end;
- ``dense``: MatMul + Add per layer (tf2onnx's Dense); ``dense_gemm``: Gemm.

Padding follows TensorFlow's ``"same"``: total padding
``max((out - 1) * stride + kernel - in, 0)``, the smaller half first; a strided
``Conv1DTranspose`` with ``"same"`` doubles the length with the same split.
Weights use Keras layouts (Conv1D ``[k, in, out]``, Conv1DTranspose
``[k, out, in]``, Dense ``[in, out]``); ``reference_forward`` is a NumPy
implementation of the Keras layers that the stand-ins are tested against.

Imports onnx and NumPy lazily, inside the functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common.timeseries import WINDOW_LENGTH

INPUT_NAME = "input"
OUTPUT_NAME = "reconstruction"
OPSET = 17


@dataclass(frozen=True)
class Layer:
    kind: str  # "conv" | "conv_transpose" | "dense"
    units: int  # filters (conv) or units (dense)
    kernel: int = 0
    stride: int = 1
    activation: str | None = "relu"


DESIGNS: dict[str, tuple[Layer, ...]] = {
    "conv1d": (
        Layer("conv", 32, 7, 2),
        Layer("conv", 16, 7, 2),
        Layer("conv_transpose", 16, 7, 2),
        Layer("conv_transpose", 32, 7, 2),
        Layer("conv", 1, 7, 1, None),
    ),
    "dense": (
        Layer("dense", 64),
        Layer("dense", 16),
        Layer("dense", 64),
        Layer("dense", WINDOW_LENGTH, activation=None),
    ),
}

# Stand-in layout -> design it is built from.
STANDINS = {
    "conv1d": "conv1d",
    "conv1d_per_layer": "conv1d",
    "conv1d_native": "conv1d",
    "dense": "dense",
    "dense_gemm": "dense",
}


def input_shape(design: str, batch: int, window: int = WINDOW_LENGTH) -> list[int]:
    return [batch, window, 1] if design == "conv1d" else [batch, window]


def same_padding(length: int, kernel: int, stride: int) -> tuple[int, int]:
    """TensorFlow "same" padding (before, after) of a convolution over ``length`` points."""
    out = -(-length // stride)
    total = max((out - 1) * stride + kernel - length, 0)
    return total // 2, total - total // 2


def layer_lengths(design: str, window: int = WINDOW_LENGTH) -> list[int]:
    """Time steps after each layer of a convolutional design."""
    lengths, length = [], window
    for layer in DESIGNS[design]:
        length = -(-length // layer.stride) if layer.kind == "conv" else length * layer.stride
        lengths.append(length)
    return lengths


def parameter_count(design: str) -> int:
    count, channels = 0, 1 if design == "conv1d" else WINDOW_LENGTH
    for layer in DESIGNS[design]:
        k = layer.kernel if layer.kind != "dense" else 1
        count += k * channels * layer.units + layer.units
        channels = layer.units
    return count


def random_weights(design: str, seed: int = 0) -> list[tuple[Any, Any]]:
    """He-normal kernels and small biases in Keras layouts, one (kernel, bias) per layer."""
    import numpy as np

    rng = np.random.default_rng(seed)
    weights, channels = [], 1 if design == "conv1d" else WINDOW_LENGTH
    for layer in DESIGNS[design]:
        if layer.kind == "dense":
            shape, fan_in = (channels, layer.units), channels
        elif layer.kind == "conv":
            shape, fan_in = (layer.kernel, channels, layer.units), layer.kernel * channels
        else:
            shape, fan_in = (layer.kernel, layer.units, channels), layer.kernel * channels
        kernel = rng.normal(0.0, (2.0 / fan_in) ** 0.5, size=shape).astype(np.float32)
        bias = rng.normal(0.0, 0.05, size=layer.units).astype(np.float32)
        weights.append((kernel, bias))
        channels = layer.units
    return weights


# --- NumPy reference of the Keras layers ------------------------------------------


def _conv1d(x: Any, kernel: Any, bias: Any, stride: int) -> Any:
    """Keras Conv1D, padding "same": x [n, steps, in], kernel [k, in, out]."""
    import numpy as np

    k = kernel.shape[0]
    before, after = same_padding(x.shape[1], k, stride)
    xp = np.pad(x.astype(np.float64), ((0, 0), (before, after), (0, 0)))
    out_len = -(-x.shape[1] // stride)
    y = np.zeros((x.shape[0], out_len, kernel.shape[2]))
    for t in range(k):
        y += xp[:, t:t + stride * out_len:stride, :] @ kernel[t].astype(np.float64)
    return y + bias


def _conv1d_transpose(x: Any, kernel: Any, bias: Any, stride: int) -> Any:
    """Keras Conv1DTranspose, padding "same": x [n, steps, in], kernel [k, out, in]."""
    import numpy as np

    k, out_ch = kernel.shape[0], kernel.shape[1]
    out_len = x.shape[1] * stride
    before, _ = same_padding(out_len, k, stride)
    full = np.zeros((x.shape[0], (x.shape[1] - 1) * stride + k, out_ch))
    for t in range(k):
        full[:, t:t + stride * x.shape[1]:stride, :] += \
            x.astype(np.float64) @ kernel[t].T.astype(np.float64)
    return full[:, before:before + out_len, :] + bias


def reference_forward(design: str, weights: list, x: Any) -> Any:
    """Keras forward pass in float64 (inference: no dropout)."""
    import numpy as np

    y = np.asarray(x, dtype=np.float64)
    for layer, (kernel, bias) in zip(DESIGNS[design], weights):
        if layer.kind == "conv":
            y = _conv1d(y, kernel, bias, layer.stride)
        elif layer.kind == "conv_transpose":
            y = _conv1d_transpose(y, kernel, bias, layer.stride)
        else:
            y = y @ kernel.astype(np.float64) + bias
        if layer.activation == "relu":
            y = np.maximum(y, 0.0)
    return y


# --- ONNX stand-ins -------------------------------------------------------------


class _Graph:
    def __init__(self) -> None:
        self.nodes: list = []
        self.inits: list = []
        self.count = 0

    def name(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}_{self.count}"

    def const(self, prefix: str, array: Any) -> str:
        from onnx import numpy_helper

        name = self.name(prefix)
        self.inits.append(numpy_helper.from_array(array, name))
        return name

    def node(self, op: str, inputs: list[str], output: str | None = None, **attrs) -> str:
        from onnx import helper

        output = output or self.name(op.lower())
        self.nodes.append(helper.make_node(op, inputs, [output], name=self.name(op), **attrs))
        return output


def _axes(g: _Graph, *axes: int) -> str:
    import numpy as np

    return g.const("axes", np.array(axes, dtype=np.int64))


def _conv_node(g: _Graph, x: str, layer: Layer, kernel: Any, bias: Any, length: int,
               two_d: bool) -> str:
    """Conv or ConvTranspose on NCHW ``[n, c, 1, steps]`` (two_d) or NCW ``[n, c, steps]``."""
    if layer.kind == "conv":
        w = kernel.transpose(2, 1, 0)  # [out, in, k]
        before, after = same_padding(length, layer.kernel, layer.stride)
        op = "Conv"
    else:
        w = kernel.transpose(2, 1, 0)  # [in, out, k]
        before, after = same_padding(length * layer.stride, layer.kernel, layer.stride)
        op = "ConvTranspose"
    if two_d:
        w = w[:, :, None, :]
        attrs = dict(kernel_shape=[1, layer.kernel], strides=[1, layer.stride],
                     pads=[0, before, 0, after])
    else:
        attrs = dict(kernel_shape=[layer.kernel], strides=[layer.stride], pads=[before, after])
    return g.node(op, [x, g.const("W", w.copy()), g.const("B", bias)], **attrs)


def build_standin(layout: str, batch: int, seed: int = 0, window: int = WINDOW_LENGTH) -> Any:
    """An untrained FP32 ModelProto with fixed shapes, laid out like a tf2onnx export."""
    import onnx
    from onnx import TensorProto, helper

    design = STANDINS[layout]
    weights = random_weights(design, seed)
    layers = DESIGNS[design]
    g = _Graph()
    x = INPUT_NAME
    last = len(layers) - 1
    length = window
    if layout == "conv1d":
        x = g.node("Transpose", [g.node("Unsqueeze", [x, _axes(g, 1)])], perm=[0, 3, 1, 2])
    elif layout == "conv1d_native":
        x = g.node("Transpose", [x], perm=[0, 2, 1])
    for i, (layer, (kernel, bias)) in enumerate(zip(layers, weights)):
        out = OUTPUT_NAME if i == last and layout in ("dense", "dense_gemm",
                                                      "conv1d_per_layer") else None
        if layout in ("conv1d", "conv1d_native"):
            x = _conv_node(g, x, layer, kernel, bias, length, two_d=layout == "conv1d")
            if layer.activation == "relu":
                x = g.node("Relu", [x])
        elif layout == "conv1d_per_layer":
            nchw = g.node("Transpose", [g.node("Unsqueeze", [x, _axes(g, 1)])], perm=[0, 3, 1, 2])
            y = _conv_node(g, nchw, layer, kernel, bias, length, two_d=True)
            nwc = g.node("Transpose", [y], perm=[0, 2, 3, 1])
            if layer.activation == "relu":
                x = g.node("Relu", [g.node("Squeeze", [nwc, _axes(g, 1)])], out)
            else:
                x = g.node("Squeeze", [nwc, _axes(g, 1)], out)
        elif layout == "dense":
            y = g.node("MatMul", [x, g.const("W", kernel)])
            if layer.activation == "relu":
                x = g.node("Relu", [g.node("Add", [y, g.const("B", bias)])], out)
            else:
                x = g.node("Add", [y, g.const("B", bias)], out)
        else:  # dense_gemm
            y = g.node("Gemm", [x, g.const("W", kernel), g.const("B", bias)],
                       None if layer.activation else out)
            x = g.node("Relu", [y], out) if layer.activation == "relu" else y
        if layer.kind == "conv":
            length = -(-length // layer.stride)
        elif layer.kind == "conv_transpose":
            length *= layer.stride
    if layout == "conv1d":
        x = g.node("Squeeze", [g.node("Transpose", [x], perm=[0, 2, 3, 1]), _axes(g, 1)],
                   OUTPUT_NAME)
    elif layout == "conv1d_native":
        x = g.node("Transpose", [x], OUTPUT_NAME, perm=[0, 2, 1])
    shape = input_shape(design, batch, window)
    graph = helper.make_graph(
        g.nodes, f"{layout}_standin",
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, shape)],
        g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="edgebench-standin")
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def model_inputs(design: str, windows: Any) -> Any:
    """Windows ``[n, steps]`` reshaped to the design's input layout."""
    return windows[:, :, None] if design == "conv1d" else windows


def rename_io(proto: Any, input_name: str = INPUT_NAME, output_name: str = OUTPUT_NAME) -> None:
    """Rename a single-input, single-output graph's input and output in place."""
    graph = proto.graph
    inits = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in inits]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ValueError("expected exactly one graph input and one output")
    renames = {inputs[0].name: input_name, graph.output[0].name: output_name}
    taken = {o for n in graph.node for o in n.output} | inits | {i.name for i in graph.input}
    for old, new in renames.items():
        if old != new and new in taken:
            raise ValueError(f"name {new!r} is already used in the graph")
    for node in graph.node:
        node.input[:] = [renames.get(n, n) for n in node.input]
        node.output[:] = [renames.get(n, n) for n in node.output]
    inputs[0].name = input_name
    graph.output[0].name = output_name
    for vi in graph.value_info:
        vi.name = renames.get(vi.name, vi.name)
