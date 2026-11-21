"""Parse ONNX Runtime / QNN EP verbose session logs for node placement.

Pure text processing with no ONNX Runtime import, so it can be tested anywhere.
``scripts/common/ort_qnn.py`` captures the logs; fixtures taken from real logs
are in ``scripts/tests/fixtures/qnn_logs/``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

QNN_EP = "QNNExecutionProvider"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LOG_LINE = re.compile(r"\[(?P<sev>[VIWEF]):onnxruntime:[^,]*, (?P<src>[\w.]+):\d+ [^\]]*\] (?P<msg>.*)")
_PASS = re.compile(r"GetCapability pass (?P<n>\d+)")
_VALID = re.compile(r"Validation (?P<res>PASSED|FAILED) for (?P<n>\d+) nodes in OrtNodeUnit \((?P<op>\w+)\)")
# Continuation lines of a "Validation FAILED" message (no log prefix, tab-indented).
_FAILED_NODE = re.compile(r"Operator type: (?P<op>\S+) Node name: (?P<name>.*?) Node index: (?P<idx>\d+)")
_REASON = re.compile(r"REASON\s*:\s*(?P<reason>.*)")
_UNSUPPORTED = re.compile(r"Unsupported nodes in QNN EP: (?P<list>.*)")
_UNSUPPORTED_ITEM = re.compile(r"\{ name: (?P<name>[^,]*), type: (?P<op>[^ }]*) \}")
_PLACED = re.compile(r"(?:All nodes|Node\(s\)) placed on \[(?P<ep>\w+)\]\. Number of nodes: (?P<n>\d+)")
_PLACED_NODE = re.compile(r"^(?P<op>\S+) \((?P<name>.*)\)$")
_STRICT_FAIL = "fallback to CPU EP has been explicitly disabled"


@dataclass
class PlacementReport:
    ort_placements: dict = field(default_factory=dict)  # EP -> nodes (after EP fusion)
    ort_placed_nodes: dict = field(default_factory=dict)  # EP -> ["OpType (name)", ...]
    qnn_passes: list = field(default_factory=list)  # per GetCapability pass
    rejected_nodes: list = field(default_factory=list)  # distinct nodes the QNN EP rejected
    unsupported_nodes: list = field(default_factory=list)  # from the strict-mode error summary
    strict_fallback_error: bool = False  # ORT refused a session that needed the CPU EP
    warnings: list = field(default_factory=list)  # distinct W/E/F messages

    @property
    def rejections(self) -> list[str]:
        """One human-readable line per rejected node."""
        return [f"{r['op_type']} node '{r['node_name']}' (index {r['node_index']}): {r['reason']}"
                for r in self.rejected_nodes]

    def as_dict(self) -> dict:
        return {
            "ort_node_placements": self.ort_placements,
            "ort_placed_nodes": self.ort_placed_nodes,
            "qnn_get_capability_passes": self.qnn_passes,
            "qnn_rejected_nodes": self.rejected_nodes,
            "qnn_rejections": self.rejections,
            "qnn_unsupported_nodes_error": self.unsupported_nodes,
            "strict_fallback_error_logged": self.strict_fallback_error,
            "log_warnings": self.warnings,
        }

    @property
    def all_on_qnn(self) -> bool:
        final = self.qnn_passes[-1] if self.qnn_passes else None
        return (set(self.ort_placements) == {QNN_EP} and not self.rejected_nodes
                and final is not None and final["failed_nodes"] == 0)


def parse_placement_log(text: str) -> PlacementReport:
    """Extract node placement facts from a verbose (severity 0) ORT session log.

    Understands the QNN EP's per-node-unit validation messages (including the
    continuation lines that carry the rejection reason), its strict-mode
    "Unsupported nodes" summary, and ONNX Runtime's "Node placements" block.
    Fixtures from real logs are in ``scripts/tests/fixtures/qnn_logs/``.
    """
    report = PlacementReport()
    current = None  # current GetCapability pass
    failed = None  # rejected node whose continuation lines are being read
    placing = None  # EP whose placed-node list is being read
    rejected: dict[tuple, dict] = {}
    for raw in text.splitlines():
        line = _ANSI.sub("", raw).rstrip()
        m = _LOG_LINE.search(line)
        if not m:
            if failed is not None and (f := _FAILED_NODE.search(line)):
                failed.update(op_type=f["op"], node_name=f["name"].strip(), node_index=int(f["idx"]))
            elif failed is not None and (r := _REASON.search(line)):
                failed["reason"] = r["reason"].strip()
                key = (failed["op_type"], failed["node_name"], failed["node_index"])
                entry = rejected.setdefault(key, {k: v for k, v in failed.items() if k != "pass"}
                                            | {"passes": []})
                if failed["pass"] not in entry["passes"]:
                    entry["passes"].append(failed["pass"])
                failed = None
            continue
        msg, sev = m["msg"].strip(), m["sev"]
        failed = None
        if p := _PASS.search(msg):
            current = {"pass": int(p["n"]), "passed_units": Counter(), "passed_nodes": 0,
                       "failed_units": Counter(), "failed_nodes": 0}
            report.qnn_passes.append(current)
        elif v := _VALID.search(msg):
            if current is None:  # logs without pass markers
                current = {"pass": 1, "passed_units": Counter(), "passed_nodes": 0,
                           "failed_units": Counter(), "failed_nodes": 0}
                report.qnn_passes.append(current)
            res = "passed" if v["res"] == "PASSED" else "failed"
            current[f"{res}_units"][v["op"]] += 1
            current[f"{res}_nodes"] += int(v["n"])
            if res == "failed":
                failed = {"op_type": v["op"], "node_name": "", "node_index": None,
                          "reason": None, "pass": current["pass"]}
        elif u := _UNSUPPORTED.search(msg):
            for i in _UNSUPPORTED_ITEM.finditer(u["list"]):
                item = {"op_type": i["op"], "node_name": i["name"].strip()}
                if item not in report.unsupported_nodes:  # logged once per capability pass
                    report.unsupported_nodes.append(item)
        if pl := _PLACED.search(msg):
            placing = pl["ep"]
            report.ort_placements[placing] = report.ort_placements.get(placing, 0) + int(pl["n"])
            report.ort_placed_nodes.setdefault(placing, [])
        elif placing and m["src"] == "session_state.cc" and (n := _PLACED_NODE.match(msg)):
            report.ort_placed_nodes[placing].append(f"{n['op']} ({n['name']})")
        elif not msg.startswith("Node placements"):
            placing = None
        if _STRICT_FAIL in msg:
            report.strict_fallback_error = True
        if sev in "WEF" and msg not in report.warnings:
            report.warnings.append(msg)
    for p in report.qnn_passes:
        p["passed_units"] = dict(sorted(p["passed_units"].items()))
        p["failed_units"] = dict(sorted(p["failed_units"].items()))
    report.rejected_nodes = list(rejected.values())
    return report


# --------------------------------------------------------------------------- QNN profile CSV

# Per-inference backend events written by the QNN EP profiler
# (provider options profiling_level=basic|detailed, profiling_file_path=...),
# from the accelerator outwards. Times are microseconds.
QNN_EXECUTE_EVENTS = {
    "accelerator_execute_us": ("BACKEND", "Accelerator (execute) time"),
    "accelerator_execute_excluding_wait_us": ("BACKEND", "Accelerator (execute excluding wait) time"),
    "qnn_accelerator_execute_us": ("BACKEND", "QNN accelerator (execute) time"),
    "rpc_execute_us": ("BACKEND", "RPC (execute) time"),
    "qnn_execute_us": ("EXECUTE", "QNN (execute) time"),
}


def parse_qnn_profile_csv(text: str) -> dict:
    """Summarize a QNN EP profiling CSV.

    Returns per-inference medians of the backend execute events
    (``QNN_EXECUTE_EVENTS``), the number of inferences seen, and per-op cycle
    medians (only present with ``profiling_level=detailed``).
    """
    import csv
    import io
    import statistics

    events: dict[str, list[float]] = {k: [] for k in QNN_EXECUTE_EVENTS}
    lookup = {v: k for k, v in QNN_EXECUTE_EVENTS.items()}
    ops: dict[str, list[float]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        msg, ident, level = row.get("Message"), row.get("Event Identifier"), row.get("Event Level")
        if msg == "NODE":
            ops.setdefault(ident, []).append(float(row["Time"]))
        elif level == "ROOT" and row.get("Unit of Measurement") == "US" \
                and (msg, ident) in lookup:
            events[lookup[(msg, ident)]].append(float(row["Time"]))
    out = {k: (statistics.median(v) if v else None) for k, v in events.items()}
    out["inferences"] = len(events["accelerator_execute_us"])
    out["op_cycles_median"] = {k: statistics.median(v) for k, v in ops.items()}
    return out
