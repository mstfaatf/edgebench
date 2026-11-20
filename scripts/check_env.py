"""Check whether this machine's Python environment can run the EdgeBench toolchain.

Prints system information, then imports each library EdgeBench depends on,
reports its version, and runs a small functional check. Each check runs on its
own, so a failure in one does not stop the others.

The torch check exports a pretrained MobileNetV2 to ONNX. Later checks reuse
that model: the onnx checker, the ONNX Runtime CPU session, the PyTorch vs ONNX
Runtime output comparison, and (if available) a QNN execution provider session.

Everything runs in a single environment: onnxruntime-qnn 2.x is a plugin
execution provider that installs alongside the stock onnxruntime package.
TensorFlow and tf2onnx are optional; they have no native win-arm64 wheels and
are only reported if present.

Unlike other EdgeBench scripts, this one does not exit on a non-native
(emulated x64) interpreter. It reports the platform status so the problem is
visible in the output.

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --json results.json

Status values:
    PASS           imported and the functional check succeeded
    FAIL           installed but import or functional check raised an error
    NOT INSTALLED  the module cannot be found in this environment
    SKIP           check not applicable (e.g. a prerequisite is unavailable)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from common.platform_guard import explain, platform_status

PASS = "PASS"
FAIL = "FAIL"
NOT_INSTALLED = "NOT INSTALLED"
SKIP = "SKIP"

INPUT_SHAPE = (1, 3, 224, 224)
OPSET = 17
ATOL = 1e-4


@dataclass
class CheckResult:
    name: str
    status: str
    version: str | None = None
    detail: str = ""
    error: str | None = None
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class Skip(Exception):
    """Raised by a check when it does not apply in this environment."""


# Values produced by one check and consumed by later ones.
shared: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------

def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def windows_version() -> str:
    if sys.platform != "win32":
        return f"{platform.system()} {platform.release()} (not Windows)"
    release, version, _, _ = platform.win32_ver()
    build = sys.getwindowsversion().build
    # Older Pythons report release "10" on Windows 11; builds >= 22000 are Windows 11.
    if release == "10" and build >= 22000:
        release = "11"
    name = " ".join(p for p in ("Windows", release, platform.win32_edition()) if p)
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
        )
        display, _ = winreg.QueryValueEx(key, "DisplayVersion")
        ubr, _ = winreg.QueryValueEx(key, "UBR")
        return f"{name} {display} (build {build}.{ubr})"
    except OSError:
        return f"{name} (version {version}, build {build})"


def dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def system_info() -> dict[str, Any]:
    status = platform_status()
    return {
        "os": windows_version(),
        "cpu": cpu_name(),
        "python_version": platform.python_version(),
        "python_build": sys.version.replace("\n", " "),
        # platform.machine() reports the host OS architecture, even for an
        # emulated x64 interpreter. interpreter_platform (sysconfig) reports
        # what the interpreter itself was built for.
        "platform_machine": status.host_machine,
        "platform_architecture": list(platform.architecture()),
        "interpreter_platform": status.interpreter_platform,
        "native_arm64": status.is_native_arm64,
        "executable": status.executable,
        "pip_version": dist_version("pip"),
    }


# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------

def short_error(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}".strip()
    first_lines = msg.splitlines()[:6]
    return "\n".join(first_lines)


def run_check(
    name: str,
    module: str | None,
    func: Callable[[Any], tuple[str, dict[str, Any]] | str],
    dist: str | None = None,
) -> CheckResult:
    """Import `module` (if given), then run `func(mod)`.

    `func` returns either a detail string or (detail, extra dict).
    """
    start = time.perf_counter()
    result = CheckResult(name=name, status=FAIL)
    try:
        mod = None
        if module is not None:
            if importlib.util.find_spec(module) is None:
                result.status = NOT_INSTALLED
                result.detail = f"module '{module}' not found"
                return result
            # Read the installed version first so it is reported even if the import fails.
            result.version = dist_version(dist or module)
            mod = importlib.import_module(module)
            result.version = result.version or getattr(mod, "__version__", None)
        out = func(mod)
        if isinstance(out, tuple):
            result.detail, result.extra = out
        else:
            result.detail = out
        result.status = PASS
    except Skip as exc:
        result.status = SKIP
        result.detail = str(exc)
    except BaseException as exc:  # noqa: BLE001 - report every failure, including SystemExit from C extensions
        if isinstance(exc, KeyboardInterrupt):
            raise
        result.status = FAIL
        result.error = short_error(exc)
        result.extra["traceback"] = traceback.format_exc(limit=8)
    finally:
        result.seconds = round(time.perf_counter() - start, 2)
    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_numpy(np):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    total = float((a @ a.T).sum())
    assert total == 1134.0, total
    return f"3x4 matmul ok (sum={total:g})"


def check_pandas(pd):
    df = pd.DataFrame({"device": ["cpu", "npu", "cpu"], "ms": [10.0, 2.0, 12.0]})
    means = df.groupby("device")["ms"].mean().to_dict()
    assert means == {"cpu": 11.0, "npu": 2.0}, means
    return "DataFrame groupby ok"


def check_torch(torch):
    import numpy as np
    import torchvision

    tmp_dir: Path = shared["tmp_dir"]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = tmp_dir / "mobilenetv2.onnx"

    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    model = torchvision.models.mobilenet_v2(weights=weights).eval()

    torch.manual_seed(0)
    x = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 1000), tuple(y.shape)

    # Prefer the TorchScript-based exporter for a fixed-shape graph; fall back to
    # the dynamo exporter on torch builds where the legacy one is unavailable.
    exporter = "torchscript"
    common = dict(
        opset_version=OPSET,
        input_names=["input"],
        output_names=["logits"],
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(model, (x,), str(onnx_path), dynamo=False, **common)
    except TypeError:
        torch.onnx.export(model, (x,), str(onnx_path), **common)
    except Exception as legacy_exc:  # noqa: BLE001
        exporter = f"dynamo (torchscript exporter failed: {type(legacy_exc).__name__})"
        torch.onnx.export(model, (x,), str(onnx_path), dynamo=True, **common)

    shared["onnx_path"] = onnx_path
    shared["input"] = x.numpy()
    shared["torch_output"] = y.numpy()
    size_mb = onnx_path.stat().st_size / 1e6
    top1 = int(np.argmax(y.numpy()))
    return (
        f"mobilenet_v2 forward ok (top1={top1}); exported to {onnx_path.name} "
        f"(opset {OPSET}, {size_mb:.1f} MB, {exporter} exporter)",
        {"onnx_path": str(onnx_path), "exporter": exporter},
    )


def check_torchvision(tv):
    # Functional coverage comes from the torch check, which builds mobilenet_v2.
    names = tv.models.list_models()
    assert "mobilenet_v2" in names
    return f"{len(names)} models registered (mobilenet_v2 exercised in torch check)"


def tiny_onnx_model(onnx):
    """Build a one-node Add model so the onnx/onnxruntime checks can run without torch."""
    from onnx import TensorProto, helper

    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 2])
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 2])
    graph = helper.make_graph([helper.make_node("Add", ["a", "b"], ["c"])], "add", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    return model


def check_onnx(onnx):
    path = shared.get("onnx_path")
    if path is not None:
        onnx.checker.check_model(str(path), full_check=True)
        model = onnx.load(str(path))
        return f"checker ok on {path.name} ({len(model.graph.node)} nodes)"
    model = tiny_onnx_model(onnx)
    onnx.checker.check_model(model, full_check=True)
    return "checker ok on built-in Add model (no exported model available)"


def register_qnn_plugin(ort) -> str | None:
    """Register the QNN plugin execution provider, if one is installed.

    onnxruntime-qnn 2.x ships the QNN EP as a plugin library on top of the stock
    onnxruntime package. It does not appear in get_available_providers() until
    it is registered. Older onnxruntime-qnn releases bundled the EP directly, in
    which case there is nothing to register. Returns a short note, or None.
    """
    if "qnn_plugin" in shared:
        return shared["qnn_plugin"]
    note = None
    if importlib.util.find_spec("onnxruntime_qnn") is not None and hasattr(
        ort, "register_execution_provider_library"
    ):
        import onnxruntime_qnn

        ort.register_execution_provider_library(
            onnxruntime_qnn.get_ep_name(), onnxruntime_qnn.get_library_path()
        )
        note = f"registered QNN plugin EP (onnxruntime_qnn {onnxruntime_qnn.__version__})"
    shared["qnn_plugin"] = note
    return note


def check_onnxruntime(ort):
    import numpy as np

    try:
        plugin_note = register_qnn_plugin(ort)
    except Exception as exc:  # noqa: BLE001
        plugin_note = f"QNN plugin registration failed: {short_error(exc).splitlines()[0]}"
    providers = ort.get_available_providers()
    shared["ort_providers"] = providers
    path = shared.get("onnx_path")
    if path is not None:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        out = sess.run(None, {name: shared["input"]})[0]
        shared["ort_output"] = out
        what = f"CPU session on {path.name} ok, output {tuple(out.shape)}"
    else:
        import onnx

        sess = ort.InferenceSession(
            tiny_onnx_model(onnx).SerializeToString(), providers=["CPUExecutionProvider"]
        )
        a = np.ones((2, 2), dtype=np.float32)
        out = sess.run(None, {"a": a, "b": a})[0]
        assert np.all(out == 2.0)
        what = "CPU session on built-in Add model ok"
    prefix = f"{plugin_note}; " if plugin_note else ""
    return f"{prefix}providers={providers}; {what}", {"providers": providers, "qnn_plugin": plugin_note}


def check_cross(_):
    import numpy as np

    if "torch_output" not in shared or "ort_output" not in shared:
        raise Skip("needs both the torch export and the onnxruntime CPU run to succeed")
    a, b = shared["torch_output"], shared["ort_output"]
    max_diff = float(np.max(np.abs(a - b)))
    ok = bool(np.allclose(a, b, atol=ATOL))
    detail = f"max abs diff {max_diff:.3g} (atol {ATOL:g})"
    if not ok:
        raise AssertionError(f"outputs differ: {detail}")
    return detail, {"max_abs_diff": max_diff, "atol": ATOL}


def check_timm(timm):
    import torch

    model = timm.create_model("tf_efficientnet_lite0", pretrained=False).eval()
    with torch.no_grad():
        y = model(torch.randn(*INPUT_SHAPE))
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"tf_efficientnet_lite0 created ({params:.1f}M params), forward ok {tuple(y.shape)}"


def check_tensorflow(tf):
    total = float(tf.reduce_sum(tf.constant([[1.0, 2.0], [3.0, 4.0]])).numpy())
    assert total == 10.0
    devices = [d.device_type for d in tf.config.list_physical_devices()]
    return f"reduce_sum ok; devices={devices}"


def check_tf2onnx(tf2onnx):
    return "import ok"


def check_qnn(_):
    if importlib.util.find_spec("onnxruntime") is None:
        raise Skip("onnxruntime not installed")
    import onnxruntime as ort

    plugin = register_qnn_plugin(ort)
    providers = ort.get_available_providers()
    if "QNNExecutionProvider" not in providers:
        raise Skip("QNNExecutionProvider not in available providers")
    path = shared.get("onnx_path")
    if path is None:
        raise Skip("needs the exported ONNX model from the torch check")

    options = {"backend_path": "QnnHtp.dll"}

    def make_session(strict: bool):
        so = ort.SessionOptions()
        if strict:
            # Only succeeds if QNN takes the whole graph.
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if plugin:
            # Plugin EPs are attached to a discovered device rather than by name.
            npu = [
                d for d in ort.get_ep_devices()
                if d.ep_name == "QNNExecutionProvider"
                and d.device.type == ort.OrtHardwareDeviceType.NPU
            ]
            if not npu:
                raise RuntimeError("QNN EP registered but no NPU device was discovered")
            so.add_provider_for_devices(npu, options)
            return ort.InferenceSession(str(path), sess_options=so)
        return ort.InferenceSession(
            str(path), sess_options=so, providers=[("QNNExecutionProvider", options)]
        )

    feed = {"input": shared["input"]}

    # Default options: nodes QNN cannot take fall back to the CPU EP.
    sess = make_session(strict=False)
    active = sess.get_providers()
    out = sess.run(None, feed)[0]
    notes = [f"HTP session created and ran; active providers={active}"]
    extra: dict[str, Any] = {"active_providers": active}
    if "torch_output" in shared:
        # Informational only: the HTP backend may compute FP32 graphs at reduced precision.
        import numpy as np

        diff = float(np.max(np.abs(out - shared["torch_output"])))
        same_top1 = int(np.argmax(out)) == int(np.argmax(shared["torch_output"]))
        notes.append(f"max abs diff vs torch {diff:.3g}, top1 match={same_top1}")
        extra.update(max_abs_diff_vs_torch=diff, top1_match=same_top1)

    strict = "ok"
    try:
        make_session(strict=True).run(None, feed)
    except Exception as exc:  # noqa: BLE001
        strict = f"failed ({short_error(exc).splitlines()[0][:200]})"
    notes.append(f"without CPU fallback: {strict}")
    extra["strict_no_cpu_fallback"] = strict
    return "; ".join(notes), extra


def check_onnxruntime_qnn(qnn):
    lib = Path(qnn.get_library_path())
    htp = Path(qnn.get_qnn_htp_path())
    missing = [p.name for p in (lib, htp) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing from package: {', '.join(missing)}")
    sdk = getattr(qnn.build_and_package_info, "qnn_version", "unknown")
    return f"plugin EP library present ({lib.name}, {htp.name}); QNN SDK {sdk}", {"qnn_sdk": sdk}


# Modules that are not expected to be installable on win-arm64. Their absence
# is reported but is not a problem.
OPTIONAL = {
    "tensorflow": "optional: no native win-arm64 wheel; Keras models are trained and converted off-device",
    "tf2onnx": "optional: requires tensorflow",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(tmp_dir: Path) -> list[CheckResult]:
    shared.clear()
    shared["tmp_dir"] = tmp_dir

    # Order matters: torch produces the model that onnx/onnxruntime/QNN consume.
    plan = [
        ("numpy", "numpy", check_numpy, None),
        ("pandas", "pandas", check_pandas, None),
        ("torch", "torch", check_torch, None),
        ("torchvision", "torchvision", check_torchvision, None),
        ("onnx", "onnx", check_onnx, None),
        ("onnxruntime", "onnxruntime", check_onnxruntime, "onnxruntime"),
        ("onnxruntime-qnn", "onnxruntime_qnn", check_onnxruntime_qnn, "onnxruntime-qnn"),
        ("torch vs onnxruntime CPU", None, check_cross, None),
        ("timm", "timm", check_timm, None),
        ("tensorflow", "tensorflow", check_tensorflow, None),
        ("tf2onnx", "tf2onnx", check_tf2onnx, None),
        ("QNN execution provider", None, check_qnn, None),
    ]
    results = []
    for name, module, func, dist in plan:
        print(f"  running: {name} ...", flush=True)
        result = run_check(name, module, func, dist)
        if result.status == NOT_INSTALLED and name in OPTIONAL:
            result.detail = OPTIONAL[name]
        results.append(result)
    return results


def print_report(info: dict[str, Any], results: list[CheckResult]) -> None:
    print("\nSystem")
    print("-" * 60)
    for key, value in info.items():
        print(f"  {key:<22} {value}")
    if not info["native_arm64"]:
        print("\n  WARNING: results below do not reflect native ARM64 behaviour.")
        for line in explain(platform_status()).splitlines():
            print(f"  {line}")

    print("\nChecks")
    print("-" * 60)
    width = max(len(r.name) for r in results)
    for r in results:
        version = r.version or "-"
        print(f"  {r.name:<{width}}  {r.status:<13}  {version}")
        if r.detail:
            print(f"  {'':<{width}}    {r.detail}")
        if r.error:
            for line in r.error.splitlines():
                print(f"  {'':<{width}}    ! {line}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("\nSummary: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, metavar="PATH", help="also write results as JSON")
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tmp",
        help="directory for exported models (default: <repo>/tmp)",
    )
    args = parser.parse_args(argv)

    print("EdgeBench environment check")
    info = system_info()
    results = run_all(args.tmp_dir)
    print_report(info, results)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "system": info,
            "results": [asdict(r) for r in results],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
