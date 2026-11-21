"""Remove dead nodes and unused initializers from an ONNX graph.

A node is dead when none of its outputs is consumed by another node (including
nodes inside subgraphs) and none is a graph output. Removing a dead node can
make its producers dead too, so pruning repeats until nothing changes. After
the nodes, initializers that nothing reads (and that are not graph inputs) and
``value_info`` entries for tensors that no longer exist are dropped as well.

The static quantizer leaves such dead chains behind, e.g. the
QuantizeLinear/DequantizeLinear pairs on the min/max constants of ``Clip``
nodes that it folds into the activation quantization ranges.

Only the main graph is pruned; subgraphs are scanned for the tensors they read
but are otherwise left untouched.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import onnx


@dataclass
class PruneReport:
    iterations: int = 0  # passes that removed something
    removed_nodes: Counter = field(default_factory=Counter)  # by op type
    removed_initializers: int = 0
    removed_value_info: int = 0
    nodes_before: int = 0
    nodes_after: int = 0
    initializers_before: int = 0
    initializers_after: int = 0

    def as_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "removed_nodes_by_op": dict(sorted(self.removed_nodes.items())),
            "initializers_before": self.initializers_before,
            "initializers_after": self.initializers_after,
            "removed_initializers": self.removed_initializers,
            "removed_value_info": self.removed_value_info,
        }


def _subgraph_inputs(node: onnx.NodeProto) -> set[str]:
    """Names read by nodes in any subgraph attribute of ``node`` (recursively)."""
    names: set[str] = set()
    for attr in node.attribute:
        graphs = list(attr.graphs)
        if attr.HasField("g"):
            graphs.append(attr.g)
        for g in graphs:
            for n in g.node:
                names.update(i for i in n.input if i)
                names |= _subgraph_inputs(n)
    return names


def _consumed(graph: onnx.GraphProto) -> set[str]:
    names = {o.name for o in graph.output}
    for n in graph.node:
        names.update(i for i in n.input if i)
        names |= _subgraph_inputs(n)
    return names


def prune(model: onnx.ModelProto) -> PruneReport:
    """Prune ``model`` in place and return what was removed."""
    graph = model.graph
    report = PruneReport(nodes_before=len(graph.node), initializers_before=len(graph.initializer))

    while True:
        consumed = _consumed(graph)
        nodes = list(graph.node)
        dead = {i for i, n in enumerate(nodes) if not any(o in consumed for o in n.output if o)}
        if not dead:
            break
        report.iterations += 1
        report.removed_nodes.update(nodes[i].op_type for i in dead)
        keep = [n for i, n in enumerate(nodes) if i not in dead]
        del graph.node[:]
        graph.node.extend(keep)

    consumed = _consumed(graph)
    graph_inputs = {i.name for i in graph.input}
    unused = [i for i in graph.initializer if i.name not in consumed and i.name not in graph_inputs]
    if unused:
        names = {i.name for i in unused}
        keep_inits = [i for i in graph.initializer if i.name not in names]
        del graph.initializer[:]
        graph.initializer.extend(keep_inits)
        report.removed_initializers = len(unused)

    produced = {o for n in graph.node for o in n.output} | graph_inputs
    produced |= {i.name for i in graph.initializer}
    stale = [v for v in graph.value_info if v.name not in produced]
    if stale:
        keep_vi = [v for v in graph.value_info if v.name in produced]
        del graph.value_info[:]
        graph.value_info.extend(keep_vi)
        report.removed_value_info = len(stale)

    report.nodes_after = len(graph.node)
    report.initializers_after = len(graph.initializer)
    return report
