"""Describe how an ONNX QDQ model is quantized, independent of the tool that produced it.

Constant tensors (weights, biases, other constants) can be quantized in two forms:

- **stored quantized**: an integer initializer feeding DequantizeLinear (what
  ONNX Runtime's quantizer writes), or
- **quantized in-graph**: a float initializer feeding QuantizeLinear →
  DequantizeLinear (what Qualcomm AI Hub exports contain; ONNX Runtime
  constant-folds the QuantizeLinear when it loads the model).

Both are reported the same way. A constant feeding input 1 of Conv/Gemm/MatMul
is a weight, input 2 a bias; anything else is "other" (e.g. Clip bounds,
normalization constants).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import numpy_helper

QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}
WEIGHTED_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}


def _array(inits: dict, name: str, base_dir: str | None) -> np.ndarray:
    return numpy_helper.to_array(inits[name], base_dir=base_dir or "")


def describe(model: onnx.ModelProto, base_dir: str | None = None) -> dict:
    """Summarize ``model`` (load it with ``load_external_data=False`` and pass ``base_dir``
    if it uses external data; only small scale/zero-point tensors are read)."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for n in g.node:
        for idx, i in enumerate(n.input):
            consumers.setdefault(i, []).append((n, idx))

    ops = Counter(n.op_type for n in g.node)
    compute = [n for n in g.node if n.op_type not in QDQ_OPS | {"Constant"}]

    groups = {k: {"count": 0, "form": Counter(), "type": Counter(), "per_channel": 0,
                  "per_tensor": 0, "symmetric": 0}
              for k in ("weights", "biases", "other_constants")}
    activation_types: Counter = Counter()

    for n in g.node:
        if n.op_type != "DequantizeLinear":
            continue
        src = n.input[0]
        if src in inits:
            form, qtype = "stored_quantized", inits[src].data_type
        elif src in producer and producer[src].op_type == "QuantizeLinear" \
                and producer[src].input[0] in inits:
            q = producer[src]
            zp_name = q.input[2] if len(q.input) > 2 else ""
            qtype = inits[zp_name].data_type if zp_name in inits else onnx.TensorProto.UINT8
            form = "quantized_in_graph"
        else:
            continue  # activation
        uses = consumers.get(n.output[0], [])
        role = "other_constants"
        if uses and all(c.op_type in WEIGHTED_OPS and idx == 1 for c, idx in uses):
            role = "weights"
        elif uses and all(c.op_type in WEIGHTED_OPS and idx == 2 for c, idx in uses):
            role = "biases"
        grp = groups[role]
        grp["count"] += 1
        grp["form"][form] += 1
        grp["type"][onnx.TensorProto.DataType.Name(qtype)] += 1
        scale = _array(inits, n.input[1], base_dir) if n.input[1] in inits else np.zeros(1)
        grp["per_channel" if scale.size > 1 else "per_tensor"] += 1
        zp = (_array(inits, n.input[2], base_dir).astype(np.int64)
              if len(n.input) > 2 and n.input[2] in inits else np.zeros(1, np.int64))
        mid = 128 if qtype == onnx.TensorProto.UINT8 else 0
        grp["symmetric"] += int(np.all(zp == mid))

    for n in g.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in inits:
            zp_name = n.input[2] if len(n.input) > 2 else ""
            qtype = inits[zp_name].data_type if zp_name in inits else onnx.TensorProto.UINT8
            activation_types[onnx.TensorProto.DataType.Name(qtype)] += 1

    # QDQ placement: is every compute node's tensor input from a DequantizeLinear, and
    # every output into a QuantizeLinear (i.e. the node runs as a quantized node unit)?
    def from_dq(name: str) -> bool:
        return name in producer and producer[name].op_type == "DequantizeLinear"

    def to_q(name: str) -> bool:
        return bool(consumers.get(name)) and all(c.op_type == "QuantizeLinear"
                                                 for c, _ in consumers[name])

    wrapped, unwrapped = 0, Counter()
    for n in compute:
        ins = [i for i in n.input if i and i not in inits]
        if all(from_dq(i) for i in ins) and all(to_q(o) for o in n.output):
            wrapped += 1
        else:
            unwrapped[n.op_type] += 1

    graph_in = g.input[0].type.tensor_type.elem_type if g.input else None
    graph_out = g.output[0].type.tensor_type.elem_type if g.output else None
    for grp in groups.values():
        grp["form"], grp["type"] = dict(grp["form"]), dict(grp["type"])
    return {
        "node_count": len(g.node),
        "node_counts": dict(sorted(ops.items())),
        "compute_node_count": len(compute),
        "compute_op_types": dict(sorted(Counter(n.op_type for n in compute).items())),
        "weights": groups["weights"],
        "biases": groups["biases"],
        "other_constants": groups["other_constants"],
        "activation_quantize_types": dict(activation_types),
        "compute_nodes_in_qdq_units": wrapped,
        "compute_nodes_not_in_qdq_units": dict(unwrapped),
        "graph_input_type": onnx.TensorProto.DataType.Name(graph_in) if graph_in else None,
        "graph_output_type": onnx.TensorProto.DataType.Name(graph_out) if graph_out else None,
        "opset": {o.domain or "ai.onnx": o.version for o in model.opset_import},
        "initializer_count": len(g.initializer),
    }
