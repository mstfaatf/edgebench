"""Exact clean-up of the tf2onnx graph of the Conv1D autoencoder.

tf2onnx exports each Keras ``Conv1D`` / ``Conv1DTranspose`` layer as a 2-D
convolution with a ``1 x k`` kernel on ``[batch, channels, 1, steps]``, wrapped
in its own ``Unsqueeze`` (axis 2) / ``Squeeze`` (axis 2), and the layer's bias
as a separate ``Add`` after the ``Squeeze``::

    input [N, L, 1] -> Unsqueeze(-3) -> Reshape [N, 1, 1, L]
      -> Conv(x, W) -> Squeeze(2) -> Add(b [1, C, 1]) -> Relu -> Unsqueeze(2)
      -> ...                                                         (per layer)
      -> Conv(x, W) -> Squeeze(2) -> Add(b [1, C, 1]) -> Reshape [N, L, 1] -> output

The ``Add`` between a convolution and its ``Relu`` keeps the INT8 quantizer from
giving the convolution's output the ReLU's range (it quantizes the convolution
output, then the sum, then the ReLU output), and the per-layer
``Squeeze``/``Unsqueeze`` pairs are layout work between every layer.

``clean_conv1d_graph`` rewrites exactly that pattern, and nothing else:

1. each bias ``Add`` is folded into its convolution's bias input (``B`` of
   ``Conv`` / ``ConvTranspose``, the same values reshaped from ``[1, C, 1]``
   to ``[C]``);
2. each ``Squeeze(2)`` -> ``Relu`` -> ``Unsqueeze(2)`` between two layers is
   removed, the ``Relu`` then runs on the 4-D tensor (it is elementwise);
3. the ``Unsqueeze`` + ``Reshape`` at the input and the ``Squeeze`` + ``Reshape``
   at the output become one ``Reshape`` each (a reshape of a reshape only
   depends on the final shape).

The result is ``Reshape -> (Conv|ConvTranspose -> Relu) x 4 -> Conv -> Reshape``.
Every rewrite is exact in real arithmetic; any graph that is not exactly the
expected chain (other ops, other axes, a tensor used twice, a convolution that
already has a bias, grouped or dilated convolutions, ...) raises
``CleanupError`` and nothing is produced.

Imports onnx and NumPy lazily, inside the functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CONV_OPS = ("Conv", "ConvTranspose")
# Names of the two reshapes in the cleaned graph (node name = output tensor name).
IN_RESHAPE, IN_SHAPE = "input_nchw", "input_nchw_shape"
OUT_RESHAPE, OUT_SHAPE = "reconstruction_nwc", "reconstruction_shape"


class CleanupError(ValueError):
    """The graph is not the exact pattern the clean-up handles."""


@dataclass
class CleanupReport:
    nodes_before: int
    nodes_after: int
    ops_before: dict[str, int]
    ops_after: dict[str, int]
    layers: int
    bias_adds_folded: int
    squeeze_unsqueeze_pairs_removed: int
    reshape_chains_collapsed: int
    layer_ops: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _counts(nodes) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in nodes:
        out[n.op_type] = out.get(n.op_type, 0) + 1
    return dict(sorted(out.items()))


def clean_conv1d_graph(proto: Any) -> tuple[Any, CleanupReport]:
    """The cleaned copy of ``proto`` and what was rewritten; CleanupError if not the pattern."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    g = proto.graph
    if len(proto.functions) or any(a.type in (onnx.AttributeProto.GRAPH,
                                              onnx.AttributeProto.GRAPHS)
                                   for n in g.node for a in n.attribute):
        raise CleanupError("the graph has functions or subgraphs")
    inits = {i.name: i for i in g.initializer}
    if len(inits) != len(g.initializer):
        raise CleanupError("duplicate initializer names")
    inputs = [i for i in g.input if i.name not in inits]
    if len(inputs) != 1 or len(g.output) != 1:
        raise CleanupError("expected exactly one graph input and one graph output")
    graph_input, graph_output = inputs[0], g.output[0]

    def dims(value_info) -> list[int]:
        t = value_info.type.tensor_type
        if t.elem_type != onnx.TensorProto.FLOAT or not all(d.HasField("dim_value")
                                                             for d in t.shape.dim):
            raise CleanupError(f"{value_info.name}: expected a float32 tensor with a fixed shape")
        return [d.dim_value for d in t.shape.dim]

    in_shape, out_shape = dims(graph_input), dims(graph_output)
    if len(in_shape) != 3 or in_shape[2] != 1 or out_shape != in_shape:
        raise CleanupError(f"expected input and output [N, L, 1], got {in_shape} and {out_shape}")
    batch, length = in_shape[0], in_shape[1]

    consumers: dict[str, list] = {}
    for n in g.node:
        for name in n.input:
            if name:
                consumers.setdefault(name, []).append(n)
    produced = {o for n in g.node for o in n.output}
    if graph_input.name not in consumers or graph_output.name not in produced:
        raise CleanupError("the graph input or output is not connected")

    def array(name: str):
        if name not in inits:
            raise CleanupError(f"{name}: expected an initializer")
        return numpy_helper.to_array(inits[name])

    def attrs(node) -> dict[str, Any]:
        return {a.name: helper.get_attribute_value(a) for a in node.attribute}

    def only_consumer(tensor: str, op_type: str):
        users = consumers.get(tensor, [])
        if len(users) != 1 or users[0].op_type != op_type:
            found = ", ".join(u.op_type for u in users) or "nothing"
            raise CleanupError(f"{tensor}: expected exactly one {op_type} consumer, found {found}")
        if tensor == graph_output.name:
            raise CleanupError(f"{tensor}: an intermediate tensor is a graph output")
        return users[0]

    def axes(node, expected: list[list[int]]) -> None:
        if len(node.input) != 2 or node.attribute:
            raise CleanupError(f"{node.name}: expected {node.op_type}(data, axes) without attributes")
        value = array(node.input[1]).tolist()
        if value not in expected:
            raise CleanupError(f"{node.name}: axes {value}, expected one of {expected}")

    def reshape_to(node, shape: list[int]) -> None:
        if len(node.input) != 2 or attrs(node).get("allowzero", 0) != 0:
            raise CleanupError(f"{node.name}: expected Reshape(data, shape)")
        value = array(node.input[1]).tolist()
        if value != shape:
            raise CleanupError(f"{node.name}: reshapes to {value}, expected {shape}")

    visited: list = []
    # Input adapter: Unsqueeze to [N, 1, L, 1], Reshape to [N, 1, 1, L].
    unsqueeze = only_consumer(graph_input.name, "Unsqueeze")
    axes(unsqueeze, [[-3], [1]])
    reshape_in = only_consumer(unsqueeze.output[0], "Reshape")
    in_target = [batch, 1, 1, length]
    reshape_to(reshape_in, in_target)
    visited += [unsqueeze, reshape_in]

    layers = []  # (conv node, weight name, bias array, relu node or None)
    x = reshape_in.output[0]
    channels = 1
    while True:
        users = consumers.get(x, [])
        if len(users) != 1 or users[0].op_type not in CONV_OPS:
            raise CleanupError(f"{x}: expected exactly one Conv or ConvTranspose consumer")
        conv = users[0]
        if len(conv.input) != 2 or conv.input[0] != x:
            raise CleanupError(f"{conv.name}: expected (data, weights) inputs and no bias")
        w = array(conv.input[1])
        a = attrs(conv)
        group = a.get("group", 1)
        if w.ndim != 4 or w.shape[2] != 1 or group != 1:
            raise CleanupError(f"{conv.name}: expected a 1 x k kernel, group 1; got {w.shape}")
        k = w.shape[3]
        if a.get("kernel_shape", [1, k]) != [1, k] or a.get("dilations", [1, 1]) != [1, 1] \
                or a.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise CleanupError(f"{conv.name}: unexpected kernel_shape, dilations or auto_pad")
        pads, strides = a.get("pads", [0, 0, 0, 0]), a.get("strides", [1, 1])
        if pads[0] != 0 or pads[2] != 0 or strides[0] != 1:
            raise CleanupError(f"{conv.name}: pads or strides along the unit axis")
        if conv.op_type == "ConvTranspose" and ("output_padding" in a or "output_shape" in a):
            raise CleanupError(f"{conv.name}: output_padding / output_shape are not handled")
        in_channels = w.shape[0] if conv.op_type == "ConvTranspose" else w.shape[1]
        out_channels = w.shape[1] if conv.op_type == "ConvTranspose" else w.shape[0]
        if in_channels != channels:
            raise CleanupError(f"{conv.name}: {in_channels} input channels, expected {channels}")
        if len(consumers.get(conv.input[1], [])) != 1:
            raise CleanupError(f"{conv.name}: its weights are shared with another node")

        squeeze = only_consumer(conv.output[0], "Squeeze")
        axes(squeeze, [[2]])
        add = only_consumer(squeeze.output[0], "Add")
        if len(add.input) != 2 or add.input[0] != squeeze.output[0] or add.attribute:
            raise CleanupError(f"{add.name}: expected Add(squeezed, bias)")
        bias = array(add.input[1])
        if bias.dtype != np.float32 or list(bias.shape) != [1, out_channels, 1]:
            raise CleanupError(f"{add.name}: bias of shape {list(bias.shape)}, expected "
                               f"[1, {out_channels}, 1] float32")
        if len(consumers.get(add.input[1], [])) != 1:
            raise CleanupError(f"{add.name}: its bias is shared with another node")
        visited += [conv, squeeze, add]
        channels = out_channels

        users = consumers.get(add.output[0], [])
        if len(users) == 1 and users[0].op_type == "Reshape":
            reshape_out = users[0]
            reshape_to(reshape_out, [batch, length, 1])
            if channels != 1 or reshape_out.output[0] != graph_output.name:
                raise CleanupError(f"{reshape_out.name}: expected the last layer (1 channel) "
                                   "to feed the graph output")
            layers.append((conv, bias, None))
            visited.append(reshape_out)
            break
        relu = only_consumer(add.output[0], "Relu")
        if len(relu.input) != 1 or relu.attribute:
            raise CleanupError(f"{relu.name}: unexpected inputs or attributes")
        unsq = only_consumer(relu.output[0], "Unsqueeze")
        axes(unsq, [[2]])
        visited += [relu, unsq]
        layers.append((conv, bias, relu))
        x = unsq.output[0]

    if len(visited) != len(g.node) or len({id(n) for n in visited}) != len(g.node):
        raise CleanupError(f"{len(g.node) - len(visited)} node(s) are outside the expected chain")

    # Build the cleaned graph. Layer nodes and tensors keep their names; the two
    # reshapes get fixed names, because tf2onnx numbers its helper nodes and
    # constants differently for every batch size, and INT8 calibration ranges
    # are shared across batch sizes by tensor name.
    for name in (IN_RESHAPE, IN_SHAPE, OUT_RESHAPE, OUT_SHAPE):
        if name in inits or name in produced or name in {n.name for n in g.node}:
            raise CleanupError(f"name {name} is taken")
    nodes, keep_inits = [], []
    nodes.append(helper.make_node("Reshape", [graph_input.name, IN_SHAPE], [IN_RESHAPE],
                                  name=IN_RESHAPE))
    keep_inits.append(numpy_helper.from_array(np.asarray(in_target, dtype=np.int64), IN_SHAPE))
    x = IN_RESHAPE
    for conv, bias, relu in layers:
        bias_name = f"{conv.name}/bias"
        if bias_name in inits:
            raise CleanupError(f"initializer name {bias_name} is taken")
        b = numpy_helper.from_array(np.ascontiguousarray(bias.reshape(-1)), bias_name)
        node = helper.make_node(conv.op_type, [x, conv.input[1], bias_name], list(conv.output),
                                name=conv.name)
        node.attribute.extend(conv.attribute)
        nodes.append(node)
        keep_inits += [inits[conv.input[1]], b]
        x = conv.output[0]
        if relu is not None:
            nodes.append(helper.make_node("Relu", [x], list(relu.output), name=relu.name))
            x = relu.output[0]
    nodes.append(helper.make_node("Reshape", [x, OUT_SHAPE], [graph_output.name],
                                  name=OUT_RESHAPE))
    keep_inits.append(numpy_helper.from_array(np.asarray([batch, length, 1], dtype=np.int64),
                                              OUT_SHAPE))

    graph = helper.make_graph(nodes, g.name, [graph_input], [graph_output], keep_inits,
                              doc_string=g.doc_string)
    cleaned = onnx.ModelProto()
    cleaned.CopyFrom(proto)
    cleaned.graph.CopyFrom(graph)
    del cleaned.metadata_props[:]
    inferred = onnx.shape_inference.infer_shapes(cleaned, check_type=True, strict_mode=True)
    cleaned.graph.value_info.extend(inferred.graph.value_info)
    onnx.checker.check_model(cleaned, full_check=True)

    report = CleanupReport(
        nodes_before=len(g.node), nodes_after=len(nodes),
        ops_before=_counts(g.node), ops_after=_counts(nodes), layers=len(layers),
        bias_adds_folded=len(layers),
        squeeze_unsqueeze_pairs_removed=len(layers) - 1,
        reshape_chains_collapsed=2,
        layer_ops=[c.op_type for c, _, _ in layers])
    return cleaned, report
