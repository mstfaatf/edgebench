"""Fold input normalization (mean/std) into a model's first convolution.

An external-normalization model computes ``y = conv(pad0((x - m) / s), W) + b``
for a raw input ``x`` in [0, 1], where ``pad0`` pads with zeros. A zero in
normalized space is the per-channel mean ``m`` in raw space, so

    pad0((x - m) / s) = (padm(x) - m) / s

where ``padm`` pads the raw image with ``m`` (per channel). Because
convolution is linear, the folded model computes

    y = conv(padm(x), W') + b',   W' = W / s (per input channel),
                                  b' = b - sum(W * m / s)

with the convolution's own padding set to 0. ``padm`` is built from two
``Concat`` nodes (rows, then columns) that put constant strips holding ``m``
around the input, each strip as wide as the original padding on that side, so
asymmetric padding works and stride and dilation are unchanged. The fold is
exact, and the first convolution still feeds its activation directly, which
lets a quantizer fold the activation (ReLU, ReLU6) into the convolution's
output range.

``mean_padding=False`` gives the naive fold (``W'``, ``b'`` and the original
zero padding). Its outputs next to the border are wrong; it exists only for
comparison. A convolution without padding needs no strips.

Only a plain first convolution is folded: the graph input must have exactly
one consumer, a ``Conv`` that reads it directly with constant weights (and
bias), ``group`` 1 and explicit padding, and the input must have a fixed NCHW
shape. Anything else is refused.

Needs onnx and NumPy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


class FoldError(ValueError):
    """The model does not have the first-convolution pattern folding requires."""


@dataclass
class FirstConv:
    node: Any
    weight: np.ndarray            # [C_out, C_in, kH, kW]
    bias: np.ndarray              # [C_out]
    strides: tuple[int, int]
    pads: tuple[int, int, int, int]  # top, left, bottom, right
    dilations: tuple[int, int]


def _attr(node: Any, name: str, default: Any) -> Any:
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def conv_input(graph: Any) -> Any:
    inits = {i.name for i in graph.initializer}
    return next(i for i in graph.input if i.name not in inits)


def find_first_conv(proto: Any) -> FirstConv:
    """The Conv that is the only consumer of the only graph input; FoldError otherwise."""
    graph = proto.graph
    inits = {i.name: i for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in inits]
    if len(inputs) != 1:
        raise FoldError(f"expected one graph input, found {len(inputs)}")
    name = inputs[0].name
    if name in {o.name for o in graph.output}:
        raise FoldError(f"graph input '{name}' is also a graph output")
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for n in graph.node for a in n.attribute):
        raise FoldError("models with subgraphs are not supported")
    consumers = [n for n in graph.node if name in n.input]
    if len(consumers) != 1:
        raise FoldError(f"graph input '{name}' has {len(consumers)} consumers; "
                        "folding needs exactly one Conv")
    node = consumers[0]
    if node.op_type != "Conv" or node.domain not in ("", "ai.onnx"):
        raise FoldError(f"graph input '{name}' feeds {node.op_type}, not Conv")
    if list(node.input[:1]) != [name] or name in node.input[1:]:
        raise FoldError(f"'{name}' must be the Conv's data input only")
    if len(node.input) < 2 or node.input[1] not in inits:
        raise FoldError("the first Conv's weights are not an initializer")
    if len(node.input) > 2 and node.input[2] and node.input[2] not in inits:
        raise FoldError("the first Conv's bias is not an initializer")
    if _attr(node, "group", 1) != 1:
        raise FoldError("the first Conv is grouped; only group 1 is supported")
    auto_pad = _attr(node, "auto_pad", b"NOTSET")
    if (auto_pad.decode() if isinstance(auto_pad, bytes) else auto_pad) != "NOTSET":
        raise FoldError("the first Conv uses auto_pad; explicit pads are required")
    weight = numpy_helper.to_array(inits[node.input[1]])
    if weight.ndim != 4:
        raise FoldError(f"expected a 2-D convolution, weights have shape {weight.shape}")
    bias = (numpy_helper.to_array(inits[node.input[2]])
            if len(node.input) > 2 and node.input[2] else np.zeros(weight.shape[0], weight.dtype))
    pads = tuple(_attr(node, "pads", [0, 0, 0, 0]))
    return FirstConv(node, weight, bias, tuple(_attr(node, "strides", [1, 1])),
                     (pads[0], pads[1], pads[2], pads[3]), tuple(_attr(node, "dilations", [1, 1])))


def _mean_pad(graph: Any, conv: FirstConv, mean: np.ndarray) -> list[int]:
    """Insert Concat nodes that pad the first Conv's input with ``mean``; pads -> 0.

    Returns the padding that was moved out of the Conv (top, left, bottom, right).
    """
    node = conv.node
    dims = [d.dim_value for d in conv_input(graph).type.tensor_type.shape.dim]
    if len(dims) != 4 or min(dims) <= 0:
        raise FoldError(f"the graph input needs a fixed NCHW shape, has {dims}")
    n, c, h, w = dims
    top, left, bottom, right = conv.pads
    prefix = (node.name or "first_conv") + "_mean_pad"
    value = mean.astype(np.float32).reshape(1, c, 1, 1)
    new_nodes = []
    current = node.input[0]

    def concat(axis: int, before: int, after: int, rows: int, label: str) -> None:
        nonlocal current
        inputs = []
        for side, size in (("before", before), ("after", after)):
            if size:
                shape = (n, c, size, w) if axis == 2 else (n, c, rows, size)
                name = f"{prefix}_{label}_{side}"
                graph.initializer.append(numpy_helper.from_array(
                    np.ascontiguousarray(np.broadcast_to(value, shape)), name))
                inputs.append(name)
        if not inputs:
            return
        if before:
            inputs.insert(1, current)
        else:
            inputs.insert(0, current)
        output = f"{prefix}_{label}"
        new_nodes.append(helper.make_node("Concat", inputs, [output], axis=axis,
                                          name=f"{prefix}/Concat_{label}"))
        current = output

    concat(2, top, bottom, h, "rows")
    concat(3, left, right, h + top + bottom, "cols")
    node.input[0] = current
    for a in node.attribute:
        if a.name == "pads":
            del a.ints[:]
            a.ints.extend([0, 0, 0, 0])
    index = list(graph.node).index(node)
    for offset, new in enumerate(new_nodes):
        graph.node.insert(index + offset, new)
    return [top, left, bottom, right]


def fold(proto: Any, mean, std, *, mean_padding: bool = True) -> tuple[Any, dict]:
    """A copy of ``proto`` with mean/std folded into the first Conv.

    ``mean_padding=False`` gives the naive fold, which keeps the Conv's zero
    padding and is therefore wrong next to the border; it exists only for
    comparison.
    """
    folded = onnx.ModelProto()
    folded.CopyFrom(proto)
    conv = find_first_conv(folded)
    graph = folded.graph
    mean64 = np.asarray(mean, np.float64)
    std64 = np.asarray(std, np.float64)
    if conv.weight.shape[1] != len(mean64):
        raise FoldError(f"first Conv has {conv.weight.shape[1]} input channels, "
                        f"normalization has {len(mean64)}")
    w64 = conv.weight.astype(np.float64)
    w_folded = (w64 / std64[None, :, None, None]).astype(np.float32)
    offset = np.einsum("ochw,c->o", w64, mean64 / std64)  # sum(W * m / s) per output channel
    b_folded = (conv.bias.astype(np.float64) - offset).astype(np.float32)

    node = conv.node
    inits = {i.name: i for i in graph.initializer}
    inits[node.input[1]].CopyFrom(numpy_helper.from_array(w_folded, node.input[1]))
    if len(node.input) > 2 and node.input[2]:
        inits[node.input[2]].CopyFrom(numpy_helper.from_array(b_folded, node.input[2]))
    else:
        bias_name = node.name + "_folded_bias"
        graph.initializer.append(numpy_helper.from_array(b_folded, bias_name))
        node.input.append(bias_name)

    padded = any(conv.pads)
    info = {"first_conv": node.name, "padded": padded, "padding": "none"}
    if padded:
        if mean_padding:
            info.update(padding="mean_pad", mean_pad=_mean_pad(graph, conv, mean64))
        else:
            info.update(padding="zero")
    return folded, info


def folded_metadata(source_meta: dict[str, str], mean, std, variant: str, info: dict,
                    source_rel: str, source_sha256: str) -> dict[str, str]:
    """Metadata of a folded file: the source's, with normalization and input contract changed."""
    meta = dict(source_meta)
    preprocessing = json.loads(meta["edgebench.input_preprocessing"])
    preprocessing.update(mean=None, std=None, normalization="folded into the first convolution")
    meta.update({
        "edgebench.variant": variant,
        "edgebench.normalization": "folded",
        "edgebench.input_preprocessing": json.dumps(preprocessing, sort_keys=True),
        "edgebench.input_range": "[0, 1]",
        "edgebench.folded_mean": json.dumps(list(mean)),
        "edgebench.folded_std": json.dumps(list(std)),
        "edgebench.fold_padding": info["padding"],
        "edgebench.fold_first_conv": info["first_conv"],
        "edgebench.source_model": source_rel,
        "edgebench.source_sha256": source_sha256,
    })
    return dict(sorted(meta.items()))


def fold_file_bytes(data: bytes, mean, std, variant: str, source_rel: str,
                    source_sha256: str) -> tuple[bytes, dict]:
    """Folded, metadata-updated and checked model bytes for a serialized FP32 model."""
    proto = onnx.load_from_string(data)
    folded, info = fold(proto, mean, std)
    source_meta = {p.key: p.value for p in proto.metadata_props}
    del folded.metadata_props[:]
    helper.set_model_props(folded, folded_metadata(source_meta, mean, std, variant, info,
                                                   source_rel, source_sha256))
    onnx.checker.check_model(folded, full_check=True)
    return folded.SerializeToString(), info
