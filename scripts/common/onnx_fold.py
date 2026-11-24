"""Fold input normalization (mean/std) into a model's first convolution.

An external-normalization model computes ``y = conv(pad0((x - m) / s), W) + b``
for a raw input ``x`` in [0, 1]. Because convolution is linear,

    conv(pad0((x - m) / s), W) = conv(pad0(x), W / s) - conv(pad0(m / s), W)

where ``pad0(m / s)`` is the constant image ``m / s`` (per channel) padded with
zeros. So the folded model computes

    y = conv(pad0(x), W') + b' + E,   W' = W / s (per input channel),
                                      b' = b - sum(W * m / s),
                                      E  = sum(W * m / s) - conv(pad0(m / s), W)

``W'`` and ``b'`` are the usual fold. ``E`` is the border correction: it does
not depend on the input, is zero wherever the kernel does not touch padding,
and is precomputed once as a constant ``[1, C_out, H_out, W_out]`` tensor added
after the folded convolution. Without it ("naive" folding) outputs next to the
border are wrong, because the original model pads with zeros in normalized
space (raw value = mean) while the folded one pads with zeros in raw space.
A convolution without padding needs no correction.

Only a plain first convolution is folded: the graph input must have exactly
one consumer, a ``Conv`` that reads it directly with constant weights (and
bias), ``group`` 1 and explicit padding. Anything else is refused.

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


def conv2d(x: np.ndarray, w: np.ndarray, strides, pads, dilations) -> np.ndarray:
    """Reference 2-D convolution of one image [C_in, H, W] (float64, zero padding)."""
    x = np.pad(x.astype(np.float64), ((0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    c_out, _, kh, kw = w.shape
    h_out = (x.shape[1] - dilations[0] * (kh - 1) - 1) // strides[0] + 1
    w_out = (x.shape[2] - dilations[1] * (kw - 1) - 1) // strides[1] + 1
    out = np.zeros((c_out, h_out, w_out))
    w64 = w.astype(np.float64)
    for i in range(kh):
        for j in range(kw):
            r, c = i * dilations[0], j * dilations[1]
            patch = x[:, r:r + strides[0] * h_out:strides[0], c:c + strides[1] * w_out:strides[1]]
            out += np.einsum("oc,chw->ohw", w64[:, :, i, j], patch)
    return out


def fold(proto: Any, mean, std, *, border_correction: bool = True) -> tuple[Any, dict]:
    """A copy of ``proto`` with mean/std folded into the first Conv.

    ``border_correction=False`` gives the naive fold, which is wrong next to the
    border of a padded convolution; it exists only for comparison.
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
    info = {"first_conv": node.name, "padded": padded, "border_correction": False}
    if padded and border_correction:
        dims = [d.dim_value for d in conv_input(graph).type.tensor_type.shape.dim]
        if len(dims) != 4 or min(dims) <= 0:
            raise FoldError(f"the graph input needs a fixed NCHW shape, has {dims}")
        h, w = dims[2], dims[3]
        # E = sum(W * m / s) - conv(pad0(m / s), W) is the convolution of an image
        # that holds m / s in the padding positions and 0 inside: the kernel taps
        # that fall on padding. Computed that way, E is exactly 0 in the interior.
        top, left, bottom, right = conv.pads
        border = np.empty((len(mean64), top + h + bottom, left + w + right))
        border[:] = (mean64 / std64)[:, None, None]
        border[:, top:top + h, left:left + w] = 0.0
        correction = conv2d(border, conv.weight, conv.strides, (0, 0, 0, 0), conv.dilations)
        correction = correction[np.newaxis].astype(np.float32)
        out_name = node.output[0]
        node.output[0] = out_name + "_before_border_correction"
        map_name = node.name + "_border_correction"
        graph.initializer.append(numpy_helper.from_array(correction, map_name))
        index = list(graph.node).index(node)
        graph.node.insert(index + 1, helper.make_node(
            "Add", [node.output[0], map_name], [out_name], name=map_name + "/Add"))
        nonzero = np.abs(correction) > 0
        info.update(border_correction=True, map_shape=list(correction.shape),
                    map_nonzero_fraction=float(nonzero.mean()),
                    map_max_abs=float(np.abs(correction).max()))
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
        "edgebench.fold_border_correction": "true" if info["border_correction"] else "false",
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
