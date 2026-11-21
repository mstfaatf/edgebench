"""Check that QNN EP node rejections are detected and reported correctly.

Builds a tiny ONNX model, ``Relu -> Det -> Relu``, in which the two Relu nodes
are supported by the QNN HTP backend and ``Det`` is not (the QNN EP does not
implement it). The model is then run on the NPU twice:

1. **non-strict**: the session must be created, the QNN EP must reject the
   ``Det`` node with a reason, and ONNX Runtime must place it on the CPU EP
   while the Relu nodes go to QNN. The NPU+CPU output is compared with a
   CPU-only run;
2. **strict** (``session.disable_cpu_ep_fallback=1``): session creation must
   fail, and the log must name the rejected node.

Placement is parsed with the same code the NPU checks use
(``scripts/common/ort_qnn.parse_placement_log``). Exit code 0 if every
expectation holds, 1 otherwise, 2 on a non-native interpreter.

``--save-fixtures DIR`` writes trimmed copies of both logs (only the lines the
parser reads, without file system paths) for ``scripts/tests/test_ort_qnn.py``.

``--write-model PATH`` only writes the model (e.g. for ``ort_probe``) and exits.

Usage:
    python tools/qnn_rejection_check.py [--save-fixtures scripts/tests/fixtures/qnn_logs]
    python tools/qnn_rejection_check.py --write-model tmp/relu_det_relu.onnx
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402

UNSUPPORTED_OP = "Det"
NODE_NAMES = ("accepted_relu_in", "unsupported_det", "accepted_relu_out")
# Sources and continuation lines kept in fixtures; everything else is noise for the parser.
FIXTURE_SOURCES = ("qnn_execution_provider.cc", "session_state.cc", "inference_session.cc")


def build_model(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("Relu", ["x"], ["r"], name=NODE_NAMES[0]),
        helper.make_node(UNSUPPORTED_OP, ["r"], ["d"], name=NODE_NAMES[1]),
        helper.make_node("Relu", ["d"], ["y"], name=NODE_NAMES[2]),
    ]
    graph = helper.make_graph(nodes, "qnn_rejection_check",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def trim_log(text: str) -> str:
    """Keep the log lines the placement parser reads; drop anything with a path."""
    kept, in_failed = [], False
    for line in text.splitlines():
        is_log = "onnxruntime:" in line and "] " in line
        if is_log:
            in_failed = "Validation FAILED" in line
            keep = any(src in line for src in FIXTURE_SOURCES) and (
                "LogNodeSupport" in line or "GetCapability" in line or "placed on" in line
                or "VerifyEachNodeIsAssignedToAnEp" in line or "[E:" in line)
        else:
            keep = in_failed and line.startswith("\t")
        if keep and not re.search(r"[A-Za-z]:\\", line):
            kept.append(line)
    return "\n".join(kept) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-fixtures", type=Path, default=None,
                        help="directory for trimmed log fixtures")
    parser.add_argument("--write-model", type=Path, default=None,
                        help="only write the Relu -> Det -> Relu model to this path and exit "
                             "(e.g. to run it through ort_probe)")
    args = parser.parse_args()

    if args.write_model:
        args.write_model.parent.mkdir(parents=True, exist_ok=True)
        build_model(args.write_model)
        print(f"wrote {args.write_model}")
        return 0

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    import numpy as np
    import onnxruntime as ort
    import onnxruntime_qnn

    from common import ort_qnn

    ort_qnn.register_qnn()
    npu = ort_qnn.select_qnn_npu(ort_qnn.list_devices())
    options = {"backend_path": onnxruntime_qnn.get_qnn_htp_path()}
    checks: list[tuple[str, bool]] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "relu_det_relu.onnx"
        build_model(model)
        x = np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32)
        cpu = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        expected = cpu.run(None, {"x": x})[0]
        del cpu

        logs, reports, errors, outputs = {}, {}, {}, {}
        for mode in ("non_strict", "strict"):
            so = ort_qnn.qnn_npu_session_options(npu, options, strict=mode == "strict",
                                                 log_severity=0)
            log = tmp / f"{mode}.log"
            with ort_qnn.capture_native_stderr(log):
                try:
                    session = ort.InferenceSession(str(model), sess_options=so)
                    errors[mode] = None
                except Exception as exc:  # noqa: BLE001 - the strict run is expected to fail
                    session, errors[mode] = None, f"{type(exc).__name__}: {exc}"
                if session is not None:
                    outputs[mode] = session.run(None, {"x": x})[0]
                session = None
                gc.collect()
            logs[mode] = log.read_text(encoding="utf-8", errors="replace")
            reports[mode] = ort_qnn.parse_placement_log(logs[mode])

    for mode in ("non_strict", "strict"):
        r = reports[mode]
        print(f"\n== {mode} ==")
        print(f"  session created: {errors[mode] is None}")
        if errors[mode]:
            print(f"  error: {errors[mode]}")
        print(f"  ORT placement: {r.ort_placements} {r.ort_placed_nodes}")
        print(f"  QNN validation: "
              f"{[(p['pass'], p['passed_units'], p['failed_units']) for p in r.qnn_passes]}")
        for line in r.rejections:
            print(f"  rejected: {line}")
        if r.unsupported_nodes:
            print(f"  unsupported (error summary): {r.unsupported_nodes}")
        print(f"  all nodes on QNN: {r.all_on_qnn}")

    ns, st = reports["non_strict"], reports["strict"]
    rejected_ops = {(n["op_type"], n["node_name"]) for n in ns.rejected_nodes}
    checks += [
        ("non-strict session created", errors["non_strict"] is None),
        ("non-strict: QNN rejects exactly the Det node",
         rejected_ops == {(UNSUPPORTED_OP, NODE_NAMES[1])}),
        ("non-strict: rejection has a reason",
         all(n["reason"] for n in ns.rejected_nodes) and bool(ns.rejected_nodes)),
        ("non-strict: Det placed on the CPU EP",
         ns.ort_placed_nodes.get("CPUExecutionProvider") == [f"{UNSUPPORTED_OP} ({NODE_NAMES[1]})"]),
        ("non-strict: Relu nodes placed on QNN", ns.ort_placements.get(ort_qnn.QNN_EP, 0) >= 1
         and sum(p["passed_units"].get("Relu", 0) for p in ns.qnn_passes[-1:]) == 2),
        ("non-strict: not reported as all on QNN", not ns.all_on_qnn),
        ("non-strict: output matches CPU-only run",
         "non_strict" in outputs and bool(np.allclose(outputs["non_strict"], expected,
                                                      rtol=1e-2, atol=1e-3))),
        ("strict session creation fails", errors["strict"] is not None),
        ("strict: error says CPU fallback is disabled",
         errors["strict"] is not None and "fallback to CPU EP has been explicitly disabled"
         in errors["strict"]),
        ("strict: log names the rejected node",
         {(n["op_type"], n["node_name"]) for n in st.rejected_nodes} == rejected_ops
         and {(n["op_type"], n["node_name"]) for n in st.unsupported_nodes} == rejected_ops),
        ("strict: fallback error logged", st.strict_fallback_error),
        ("strict: not reported as all on QNN", not st.all_on_qnn),
    ]
    print("\n== checks ==")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    if args.save_fixtures:
        args.save_fixtures.mkdir(parents=True, exist_ok=True)
        for mode, text in logs.items():
            out = args.save_fixtures / f"det_{mode}.log"
            out.write_text(trim_log(text), encoding="utf-8", newline="\n")
            print(f"wrote {out}")

    failed = [n for n, ok in checks if not ok]
    print(f"\n{'OK' if not failed else 'FAILED'} ({len(checks) - len(failed)}/{len(checks)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
