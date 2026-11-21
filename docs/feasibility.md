# Environment Feasibility

This document records which parts of the EdgeBench toolchain run natively on
Windows on Snapdragon (ARM64). It covers the Python model-prep stack (PyTorch →
ONNX, ONNX Runtime CPU, the Qualcomm QNN execution provider on the Hexagon
NPU) and the native C++ toolchain linking ONNX Runtime.

Results below were collected on 2026-11-20. For setup instructions see
[setup.md](setup.md).

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

## Python

All packages are installed into a **single virtual environment** from
[`requirements.txt`](../requirements.txt) with `pip install --only-binary=:all:`,
so nothing is built from source. `onnxruntime` and `onnxruntime-qnn` coexist in
that environment (see [Implications](#implications)).

| Library | Version | Install | Functional check | Notes |
|---|---|---|---|---|
| numpy | 2.5.5 | OK | PASS | matmul |
| pandas | 3.1.1 | OK | PASS | DataFrame groupby |
| torch | 2.13.0+cpu | OK¹ | PASS | Pretrained MobileNetV2 forward pass; ONNX export (opset 17, fixed 1×3×224×224, TorchScript exporter) |
| torchvision | 0.28.0+8fb8771 | OK¹ | PASS | Model registry; MobileNetV2 weights |
| onnx | 1.23.2 | OK | PASS | `onnx.checker` (full check) on exported MobileNetV2 |
| onnxruntime | 1.32.0 | OK | PASS | CPU inference on exported model. Providers after QNN plugin registration: `AzureExecutionProvider`, `CPUExecutionProvider`, `QNNExecutionProvider` |
| onnxruntime-qnn | 2.8.0 (QNN SDK 2.52.0) | OK | PASS | Plugin EP library and `QnnHtp.dll` present |
| PyTorch vs ORT CPU | – | – | PASS | `np.allclose(atol=1e-4)`, max abs diff 1.12e-05 |
| timm | 1.0.31 | OK | PASS | `tf_efficientnet_lite0` created (no weights) + forward pass |
| **QNN EP (HTP / NPU)** | 2.8.0 | – | **PASS** | FP32 MobileNetV2 session on the NPU device; also succeeds with CPU fallback disabled (whole graph on QNN). Output vs PyTorch: max abs diff 0.0129, top-1 matches |
| tensorflow | – | **FAIL** | NOT INSTALLED | No win-arm64 wheel on PyPI (`from versions: none`) |
| tf2onnx | 1.17.0 | OK | **FAIL** | Pure-Python wheel installs, but `import tf2onnx` requires `tensorflow`. Not part of `requirements.txt` |

¹ torch and torchvision have **no win-arm64 wheels on PyPI**. Official native
win-arm64 CPU wheels are published on the PyTorch package index
(`https://download.pytorch.org/whl/cpu`), which `requirements.txt` adds as an
extra index. The newest win-arm64 torchvision (0.28.0) pins torch 2.13.0, so
torch is held at 2.13.0 even though 2.15.0 is available for win-arm64. Without
the PyTorch index, `pip install torchvision` falls back to an unrelated, ancient
pure-Python release (0.1.6). `requirements.txt` pins the exact local version
(`0.28.0+8fb8771`) to rule this out.

## C++ toolchain

| Component | Version | Notes |
|---|---|---|
| Visual Studio Community 2022 | 17.12.3 | Used for builds |
| MSVC toolset | 14.42.34433 (cl 19.42.34435) | ARM64-hosted ARM64 compiler (`Hostarm64\arm64`) present; native ARM64 target, not ARM64EC |
| Windows SDK | 10.0.26100.0 (10.0.22621.0 also installed) | ARM64 import libraries present |
| CMake | 3.29.5-msvc4 | Bundled with Visual Studio; not on `PATH` outside a developer shell |
| Ninja | 1.12.1 | Bundled with Visual Studio |
| Visual Studio Build Tools 2026 | 18.10.1 (MSVC 14.51.36231) | Also installed, but **without** the ARM64 compiler (only x64-targeting `cl.exe`); not used |

A hello-world built with `vcvarsall arm64` produces an `AA64 machine (ARM64)`
PE image.

### Where the native binaries come from

| Binary | Official source | Notes |
|---|---|---|
| ONNX Runtime 1.32.0: headers, `onnxruntime.dll`, `onnxruntime.lib`, `onnxruntime_providers_shared.dll` | NuGet [`Microsoft.ML.OnnxRuntime` 1.32.0](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/1.32.0), `runtimes/win-arm64/native/` and `build/native/include/` | Same version as the Python wheel. The GitHub release page for the main ONNX Runtime repository was unreachable when checked, so NuGet is used. The package hash matched the NuGet catalog's SHA512 |
| QNN EP plugin `onnxruntime_providers_qnn.dll` + QNN backends (`QnnHtp.dll`, `QnnHtpPrepare.dll`, `QnnHtpV*Stub.dll`, `libQnnHtpV*Skel.so`, `QnnSystem.dll`, `QnnGpu.dll`, …) | GitHub release [`onnxruntime/onnxruntime-qnn` v2.8.0](https://github.com/onnxruntime/onnxruntime-qnn/releases/tag/v2.8.0), `onnxruntime-qnn-2.8.0-win-arm64.zip` | Maintained by Qualcomm. Same file set as the Python wheel; bundles the QNN SDK 2.52.0 runtime, so no separate QAIRT SDK install is needed. SHA256 matched the release's published digest. Also available as NuGet `Qualcomm.ML.OnnxRuntime.QNN` |

The QNN EP 2.8.0 release notes state compatibility with ONNX Runtime
**>= 1.24.1** (built and tested against 1.29.0). 1.32.0 is within that range
and works in both Python and C++. The old bundled-EP NuGet package
`Microsoft.ML.OnnxRuntime.QNN` stops at 1.24.4 and is not used.

`scripts/fetch_deps.ps1` downloads both archives with pinned SHA256 checksums.

### ort_probe results

`tools/ort_probe` (C++17, ONNX Runtime C++ API) registers the QNN plugin with
`Ort::Env::RegisterExecutionProviderLibrary`, enumerates devices with
`Ort::Env::GetEpDevices`, and attaches the NPU device with
`Ort::SessionOptions::AppendExecutionProvider_V2` and
`session.disable_cpu_ep_fallback=1`. The C++ wrappers cover everything needed,
so no raw C API calls are required. The probe loads the `onnxruntime.dll`
copied next to it, not the copy in `System32`.

EP devices found:

| EP | EP vendor | Device type | Device vendor |
|---|---|---|---|
| CPUExecutionProvider | Microsoft | CPU | Qualcomm |
| QNNExecutionProvider | Qualcomm | NPU | (unset) |
| QNNExecutionProvider | Qualcomm | GPU | (unset) |

MobileNetV2 (`tmp/mobilenetv2.onnx`, FP32, opset 17), every input value 0.5:

| Run | Top-1 | Top-1 logit | First 5 outputs |
|---|---|---|---|
| C++ CPU EP | 677 | 1.860237 | -0.512211 0.156862 0.270116 0.342880 0.658771 |
| Python CPU EP | 677 | 1.860237 | -0.512211 0.156862 0.270116 0.342880 0.658771 |
| C++ QNN EP (NPU) | 677 | 1.832031 | -0.512695 0.136597 0.247070 0.323242 0.643066 |
| Python QNN EP (NPU) | 677 | 1.832031 | -0.512695 0.136597 0.247070 0.323242 0.643066 |

- C++ and Python produce identical values at the printed precision (6 decimal
  places), on both CPU and NPU.
- CPU vs NPU: top-1 matches. Max abs diff over all 1000 logits is 0.0596.

## INT8 QDQ quantization (CPU check)

MobileNetV2 (`IMAGENET1K_V2`, opset 17, fixed [1, 3, 224, 224]) quantized with
ONNX Runtime 1.32.0's QNN helpers (`get_qnn_qdq_config` + `quantize`), MinMax
calibration on 200 Imagenette train images. See
[setup.md](setup.md#preparing-data-and-models) for the pipeline. Accuracy on the
500-image Imagenette eval split, CPU execution provider:

| Model | Size | Top-1 (1000 classes) | Top-1 (10 Imagenette classes) | Agreement with FP32 (1000 / 10) |
|---|---|---|---|---|
| FP32 | 13.99 MB | 71.0% | 98.4% | – |
| INT8 QDQ, per-tensor (uint8 / uint8) | 3.73 MB | 70.4% | 98.6% | 85.6% / 98.2% |
| INT8 QDQ, per-channel (uint8 / uint8) | 3.99 MB | 69.4% | 98.4% | 92.6% / 99.2% |

- Top-1 over 1000 classes is modest because several Imagenette classes have
  close ImageNet neighbours. The most common FP32 errors are cassette player →
  tape player (14 images) and church → monastery (12). Most eval images
  come from the ImageNet train set, so these numbers are not ImageNet accuracy.
- Per-channel weights track FP32 more closely (92.6% vs 85.6% agreement over
  1000 classes) even though their top-1 is slightly lower on this small set.
- `qnn_preprocess_model` makes no changes to this model. The quantizer folds the
  ReLU6 `Clip` nodes into the activation quantization ranges but leaves their
  70 min/max constants with QuantizeLinear/DequantizeLinear chains whose
  output nothing consumes. ONNX Runtime does not remove them: with all graph
  optimizations enabled it constant-folds the QuantizeLinear nodes, but the 70
  dead DequantizeLinear nodes remain. The sidecar reports them as
  `unused_output_nodes`.
- With `per_channel=True` the uint8 weights get zero point 128 on every channel
  (symmetric), although the helper's `WeightSymmetric` setting is `false`.
- Running these models on the NPU is not yet tested.

## Implications

**Works natively on ARM64:**

- PyTorch + torchvision + timm, including pretrained-model inference and ONNX
  export. The PyTorch-side model-prep pipeline can run on-device.
- ONNX and ONNX Runtime CPU inference, from Python and C++. The CPU export path
  is numerically faithful to PyTorch (1e-5 max abs diff).
- The QNN execution provider on the NPU, from Python and C++, with the whole
  FP32 MobileNetV2 graph running on HTP.
- MSVC native ARM64 builds linking the official ONNX Runtime binaries.

**Things to account for:**

- **onnxruntime-qnn is a plugin EP.** As of 2.x it depends on the stock
  `onnxruntime` package, so one environment holds both. The EP does not appear
  in `get_available_providers()` until registered, and sessions attach to a
  discovered NPU device rather than naming the provider:
  ```python
  import onnxruntime as ort, onnxruntime_qnn
  ort.register_execution_provider_library(
      onnxruntime_qnn.get_ep_name(), onnxruntime_qnn.get_library_path())
  ```
  C++ uses the equivalent `Ort::Env` / `Ort::SessionOptions` calls (see above).
- **FP32 on the NPU is not bit-exact.** NPU logits differ from CPU by up to
  ~6e-2 on a constant input and ~1e-2 on a random one (likely reduced-precision
  compute on HTP). CPU vs NPU accuracy comparisons need a task-level metric
  such as top-1 agreement, not a tight `allclose`. Quantized (INT8/QDQ) models
  are the expected production path for the NPU and remain to be tested.
- **Runtime DLL placement matters.** `QnnHtp.dll` loads its skel/stub files from
  its own directory, and Windows ships an unrelated `onnxruntime.dll` in
  `System32`. The C++ build copies all ONNX Runtime and QNN runtime files next
  to the executable.

**Does not work natively, with proposed workarounds (not yet done):**

| Failure | Proposed workaround |
|---|---|
| TensorFlow: no native win-arm64 wheel | Train/export the Keras model off-device (e.g. Google Colab or an x64 machine), convert with `tf2onnx` there, and bring only the `.onnx` file to the Snapdragon machine |
| tf2onnx: unusable without TensorFlow | Same as above: run the conversion in the off-device TensorFlow environment |

## How to reproduce

Follow [setup.md](setup.md) to create the environment and build the C++ tools,
then:

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py --json tmp\check_env.json
.\.venv\Scripts\python.exe tools\ort_probe\reference.py
.\build\ort_probe\arm64-release\Release\ort_probe.exe tmp\mobilenetv2.onnx
```

`check_env.py` prints system information and a PASS / FAIL / NOT INSTALLED /
SKIP line per library, with error messages for failures. `--json PATH` also
writes the full results (including tracebacks) as JSON. The exported model is
written to `tmp/` (override with `--tmp-dir`). The first run downloads the
MobileNetV2 weights (~14 MB) into the PyTorch hub cache. The script exits with
status 1 if any check FAILs. It reports a non-native interpreter rather than
exiting, so it can diagnose a misconfigured environment.
