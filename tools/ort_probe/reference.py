"""Python reference for ort_probe: same model, same input, same printout.

Runs the model on the ONNX Runtime CPU EP and, if available, on the QNN EP
attached to the NPU with CPU fallback disabled. Every input value is 0.5, so
the printed top-1 index and first five outputs can be compared directly with
the C++ ort_probe output.

If the model file does not exist, a pretrained torchvision MobileNetV2 is
exported to that path first (fixed 1x3x224x224 input, opset 17).

Usage:
    python tools/ort_probe/reference.py [model.onnx]     # default: tmp/mobilenetv2.onnx
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common.platform_guard import NonNativeInterpreterError, require_native_arm64  # noqa: E402

INPUT_VALUE = 0.5
QNN_EP = "QNNExecutionProvider"


def export_mobilenet_v2(path: Path) -> None:
    import torch
    import torchvision

    model = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.DEFAULT
    ).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            (torch.randn(1, 3, 224, 224),),
            str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["logits"],
            dynamo=False,
        )
    print(f"exported MobileNetV2 to {path}")


def run(session) -> "np.ndarray":
    import numpy as np

    inp = session.get_inputs()[0]
    if any(not isinstance(d, int) or d <= 0 for d in inp.shape):
        raise ValueError(f"input shape must be fully static, got {inp.shape}")
    x = np.full(inp.shape, INPUT_VALUE, dtype=np.float32)
    return session.run(None, {inp.name: x})[0].reshape(-1)


def print_output(label: str, out) -> None:
    top = int(out.argmax())
    print(f"  {label} top-1: {top} ({out[top]:.6f})")
    print(f"  {label} first 5: " + " ".join(f"{v:.6f}" for v in out[:5]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", type=Path, default=REPO_ROOT / "tmp" / "mobilenetv2.onnx")
    args = parser.parse_args()

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2

    import numpy as np
    import onnxruntime as ort

    model = args.model.resolve()
    if not model.is_file():
        export_mobilenet_v2(model)

    print(f"onnxruntime {ort.__version__}")
    print(f"model: {model}\ninput: every value = {INPUT_VALUE}")
    failures = 0

    print("\n== CPU EP ==")
    cpu = None
    try:
        cpu = run(ort.InferenceSession(str(model), providers=["CPUExecutionProvider"]))
        print_output("cpu", cpu)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")
        failures += 1

    print("\n== QNN EP on NPU (CPU fallback disabled) ==")
    npu = None
    try:
        import onnxruntime_qnn

        ort.register_execution_provider_library(QNN_EP, onnxruntime_qnn.get_library_path())
        devices = [
            d for d in ort.get_ep_devices()
            if d.ep_name == QNN_EP and d.device.type == ort.OrtHardwareDeviceType.NPU
        ]
        if not devices:
            print("  skipped: no QNN NPU device found")
        else:
            so = ort.SessionOptions()
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            so.add_provider_for_devices(devices, {"backend_path": onnxruntime_qnn.get_qnn_htp_path()})
            npu = run(ort.InferenceSession(str(model), sess_options=so))
            print_output("npu", npu)
    except ImportError:
        print("  skipped: onnxruntime-qnn not installed")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")
        failures += 1

    if cpu is not None and npu is not None:
        print("\n== CPU vs NPU ==")
        same = int(cpu.argmax()) == int(npu.argmax())
        print(f"  max abs diff: {float(np.max(np.abs(cpu - npu))):.6g}")
        print(f"  top-1 match: {'yes' if same else 'NO'}")
        failures += 0 if same else 1

    print(f"\n{'FAILED' if failures else 'OK'} ({failures} failure{'' if failures == 1 else 's'})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
