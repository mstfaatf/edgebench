# Environment Feasibility

This document records which parts of the EdgeBench Python toolchain run natively
on Windows on Snapdragon (ARM64), based on `scripts/check_env.py`. It covers
model export (PyTorch → ONNX), ONNX Runtime CPU inference, and the Qualcomm QNN
execution provider on the Hexagon NPU.

Results below were collected on 2026-11-20.

## System

| | |
|---|---|
| Device | Surface Laptop 7 |
| CPU | Snapdragon(R) X 10-core X1P64100 @ 3.40 GHz (Snapdragon X Plus) |
| OS | Windows 11 Home 25H2, build 26200.9974 (ARM64) |
| Python | 3.12.8, official python.org ARM64 build (`MSC v.1942 64 bit (ARM64)`) |
| Interpreter platform | `win-arm64` (`sysconfig.get_platform()`) |
| `platform.machine()` | `ARM64` |
| `platform.architecture()` | `('64bit', 'WindowsPE')` |
| pip | 26.3 |

> **Note on detecting emulation:** `platform.machine()` reports the host OS
> architecture, not the interpreter's. An x64 Python running under emulation on
> the same machine also reports `ARM64`. `sysconfig.get_platform()` (`win-arm64`
> vs `win-amd64`) is the reliable check, and `check_env.py` reports both.

Two virtual environments are used, because `onnxruntime` and `onnxruntime-qnn`
are traditionally installed as alternatives to each other:

- `.venv`: general model-prep stack with stock `onnxruntime`
- `.venv-qnn`: `onnxruntime-qnn` with the minimum needed to export and run a model

All packages were installed with `pip install --only-binary=:all:`, so nothing
was built from source.

## Results

### `.venv` (stock onnxruntime)

| Library | Version | Install | Functional check | Notes |
|---|---|---|---|---|
| numpy | 2.5.5 | OK | PASS | matmul |
| pandas | 3.1.1 | OK | PASS | DataFrame groupby |
| torch | 2.13.0+cpu | OK¹ | PASS | Pretrained MobileNetV2 forward pass; ONNX export (opset 17, fixed 1×3×224×224, TorchScript exporter) |
| torchvision | 0.28.0 | OK¹ | PASS | Model registry; MobileNetV2 weights |
| onnx | 1.23.2 | OK | PASS | `onnx.checker` (full check) on exported MobileNetV2 |
| onnxruntime | 1.32.0 | OK | PASS | Providers: `AzureExecutionProvider`, `CPUExecutionProvider`. CPU inference on exported model |
| PyTorch vs ORT CPU | – | – | PASS | `np.allclose(atol=1e-4)`, max abs diff 1.12e-05 |
| timm | 1.0.31 | OK | PASS | `tf_efficientnet_lite0` created (no weights) + forward pass |
| tensorflow | – | **FAIL** | NOT INSTALLED | No win-arm64 wheel on PyPI (`from versions: none`) |
| tf2onnx | 1.17.0 | OK | **FAIL** | Pure-Python wheel installs, but `import tf2onnx` requires `tensorflow` |
| QNN EP | – | – | SKIP | Not part of this environment |

### `.venv-qnn` (onnxruntime-qnn)

| Library | Version | Install | Functional check | Notes |
|---|---|---|---|---|
| onnxruntime-qnn | 2.8.0 (QNN SDK 2.52.0) | OK | PASS | Installs as a **plugin EP** on top of `onnxruntime` 1.32.0 (pulled in as a dependency) |
| numpy | 2.5.5 | OK | PASS | |
| onnx | 1.23.2 | OK | PASS | `onnx.checker` on exported MobileNetV2 |
| torch | 2.13.0+cpu | OK¹ | PASS | Same as `.venv` |
| torchvision | 0.28.0 | OK¹ | PASS | Same as `.venv` |
| onnxruntime (CPU) | 1.32.0 | OK | PASS | After plugin registration, providers: `AzureExecutionProvider`, `CPUExecutionProvider`, `QNNExecutionProvider` |
| PyTorch vs ORT CPU | – | – | PASS | max abs diff 1.12e-05 |
| **QNN EP (HTP / NPU)** | 2.8.0 | – | **PASS** | FP32 MobileNetV2 session created on the NPU device with `backend_path=QnnHtp.dll` and ran. Also succeeds with CPU fallback disabled (whole graph on QNN). Output vs PyTorch: max abs diff 0.0129, top-1 matches |

¹ torch and torchvision have **no win-arm64 wheels on PyPI**. Official native
win-arm64 CPU wheels are published on the PyTorch package index
(`https://download.pytorch.org/whl/cpu`), and that is where these came from. The
newest win-arm64 torchvision (0.28.0) pins torch 2.13.0, so torch is held at
2.13.0 even though 2.15.0 is available for win-arm64. Without the PyTorch index,
`pip install torchvision` falls back to an unrelated, ancient pure-Python release
(0.1.6), which should be avoided.

## Implications

**Works natively on ARM64:**

- PyTorch + torchvision + timm, including pretrained-model inference and ONNX
  export. The PyTorch-side model-prep pipeline can run on-device.
- ONNX and ONNX Runtime CPU inference. The CPU export path is numerically
  faithful to PyTorch (1e-5 max abs diff).
- `onnxruntime-qnn` installs and exposes `QNNExecutionProvider` with NPU and GPU
  devices. An FP32 model runs end-to-end on the HTP backend.

**Things to account for:**

- **onnxruntime-qnn is now a plugin EP.** As of 2.x it depends on the stock
  `onnxruntime` package and does not appear in `get_available_providers()` until
  registered:
  ```python
  import onnxruntime as ort, onnxruntime_qnn
  ort.register_execution_provider_library(
      onnxruntime_qnn.get_ep_name(), onnxruntime_qnn.get_library_path())
  ```
  Sessions are then attached to a discovered NPU device with
  `SessionOptions.add_provider_for_devices(...)`. The C++ benchmark will need the
  equivalent C API calls (`RegisterExecutionProviderLibrary`,
  `GetEpDevices`, `SessionOptionsAppendExecutionProvider_V2`). Because the QNN
  package now sits on top of stock ORT, a single environment may be enough,
  which is worth confirming later.
- **FP32 on the NPU is not bit-exact.** The HTP output differs from PyTorch by up
  to ~1e-2 (likely reduced-precision compute). Accuracy comparisons between CPU
  and NPU need a task-level metric (e.g. top-1 agreement) rather than a tight
  `allclose`. Quantized (INT8/QDQ) models are the expected production path for
  the NPU and remain to be tested.

**Does not work natively, with proposed workarounds (not yet done):**

| Failure | Proposed workaround |
|---|---|
| TensorFlow: no native win-arm64 wheel | Train/export the Keras model off-device (e.g. Google Colab or an x64 machine), convert with `tf2onnx` there, and bring only the `.onnx` file to the Snapdragon machine |
| tf2onnx: unusable without TensorFlow | Same as above: run the conversion in the off-device TensorFlow environment |
| torch/torchvision not on PyPI for win-arm64 | Install from the PyTorch index (`--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple`) and pin matching torch/torchvision versions |

## How to reproduce

Requires a **native ARM64** Python (3.12 here) from python.org. Check with:

```powershell
python -c "import sysconfig; print(sysconfig.get_platform())"   # must print win-arm64
```

General environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
foreach ($p in "numpy","pandas","onnx","onnxruntime","tensorflow","tf2onnx") {
    .\.venv\Scripts\python.exe -m pip install --only-binary=:all: $p
}
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: `
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple `
    torch torchvision
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: timm
.\.venv\Scripts\python.exe scripts\check_env.py --json tmp\check_main.json
```

QNN environment:

```powershell
python -m venv .venv-qnn
.\.venv-qnn\Scripts\python.exe -m pip install --upgrade pip
foreach ($p in "onnxruntime-qnn","numpy","onnx") {
    .\.venv-qnn\Scripts\python.exe -m pip install --only-binary=:all: $p
}
.\.venv-qnn\Scripts\python.exe -m pip install --only-binary=:all: `
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple `
    torch torchvision
.\.venv-qnn\Scripts\python.exe scripts\check_env.py --json tmp\check_qnn.json
```

`check_env.py` prints system information and a PASS / FAIL / NOT INSTALLED /
SKIP line per library, with error messages for failures. `--json PATH` also
writes the full results (including tracebacks) as JSON. The exported model is
written to `tmp/` (override with `--tmp-dir`). The first run downloads the
MobileNetV2 weights (~14 MB) into the PyTorch hub cache. The script exits with
status 1 if any check FAILs.
