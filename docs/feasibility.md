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
calibration on 200 Imagenette train images, uint8 activations. See
[setup.md](setup.md#preparing-data-and-models) for the pipeline. Accuracy on the
500-image Imagenette eval split, CPU execution provider:

| Model | Weights | Size | Top-1 (1000 classes) | Top-1 (10 Imagenette classes) | Agreement with FP32 (1000 / 10) |
|---|---|---|---|---|---|
| FP32 | – | 13.99 MB | 71.0% | 98.4% | – |
| INT8 QDQ per-tensor | uint8, asymmetric | 3.65 MB | 70.4% | 98.6% | 85.6% / 98.2% |
| INT8 QDQ per-channel u8w | uint8, zero point 128 | 3.90 MB | 69.4% | 98.4% | 92.6% / 99.2% |
| INT8 QDQ per-channel s8w | int8, symmetric | 3.89 MB | 69.8% | 98.6% | 93.6% / 99.6% |

- Top-1 over 1000 classes is modest because several Imagenette classes have
  close ImageNet neighbours. The most common FP32 errors are cassette player →
  tape player (14 images) and church → monastery (12). Most eval images
  come from the ImageNet train set, so these numbers are not ImageNet accuracy.
- Per-channel weights track FP32 more closely (92.6–93.6% vs 85.6% agreement
  over 1000 classes); the differences in top-1 are within noise for 500 images.
- `qnn_preprocess_model` makes no changes to this model.
- **Dead nodes.** The quantizer folds the ReLU6 `Clip` nodes into the
  activation quantization ranges but leaves their 70 min/max constants behind,
  each with a QuantizeLinear → DequantizeLinear chain whose output nothing
  consumes. ONNX Runtime does not remove them (with all graph optimizations it
  constant-folds the QuantizeLinear nodes, but the 70 dead DequantizeLinear
  nodes remain), so `quantize_qnn.py` prunes them: 513 → 303 nodes (70 each
  of Constant, QuantizeLinear, DequantizeLinear, removed in 3 passes) and
  588 → 448 initializers, for all three variants. The pruned models give
  bit-identical CPU logits to the unpruned ones on all 500 eval images.
- With `per_channel=True` and uint8 weights, every channel gets zero point 128
  (symmetric), although the helper's `WeightSymmetric` setting is `false`.

## INT8 and FP16 on the NPU (strict mode)

`scripts/npu_check.py` runs the pruned INT8 models and the FP32 model on the
Hexagon NPU through the QNN execution provider, with CPU fallback disabled,
and compares them with the CPU execution provider on the same 500 eval images.
Results below were collected on 2026-11-20 with ONNX Runtime 1.32.0 and
onnxruntime-qnn 2.8.0, plugged in, Windows power mode *Balanced*.

### Session setup

- **Device selection.** After registering the plugin EP, ONNX Runtime lists
  three EP devices: CPU (`CPUExecutionProvider`), and two
  `QNNExecutionProvider` devices, one of type NPU (metadata `LUID`) and one of
  type GPU (Adreno X1-85). The script selects the QNN device whose hardware type
  is NPU and fails if there is none, rather than taking the first QNN device.
- **Strict mode.** Every NPU session sets `session.disable_cpu_ep_fallback=1`.
- **Provider options.** `backend_path` = the `QnnHtp.dll` shipped with
  `onnxruntime-qnn`, `htp_performance_mode=burst` (the EP log confirms
  "Htp performance mode: burst"). The FP16 configuration adds
  `enable_htp_fp16_precision=1`.
- **Loaded binaries.** In Python, the ONNX Runtime core lives in
  `onnxruntime/capi/onnxruntime_pybind11_state.pyd`; no `onnxruntime.dll` is
  loaded, so the copy in `System32` cannot interfere. With an NPU session alive
  the process has `onnxruntime_providers_qnn.dll`, `QnnHtp.dll`,
  `QnnHtpPrepare.dll`, `QnnHtpV73Stub.dll` and `QnnSystem.dll` loaded, all from
  the `onnxruntime_qnn` package directory.

### Placement

A strict session that is created proves that ONNX Runtime assigned nothing to
the CPU, but the script also checks placement directly, three ways:

1. the QNN EP's verbose log, which validates every ONNX node unit in its two
   capability passes (before and after the NCHW → NHWC layout transform);
2. ONNX Runtime's own "Node placements" summary;
3. a one-inference ORT profile, which records the EP that executed each node.

| Model | Strict session | QNN validation, pass 1 / pass 2 (nodes passed, failed) | ORT placement | Executed (profile) |
|---|---|---|---|---|
| INT8 per-tensor | OK | 313, 0 / 319, 0 | 1 fused node on QNN | 1 node, QNN |
| INT8 per-channel u8w | OK | 313, 0 / 319, 0 | 1 fused node on QNN | 1 node, QNN |
| INT8 per-channel s8w | OK | 313, 0 / 319, 0 | 1 fused node on QNN | 1 node, QNN |
| FP32 + `enable_htp_fp16_precision` | OK | 100, 0 / 102, 0 | 1 fused node on QNN | 1 node, QNN |
| *Control:* unpruned INT8 per-tensor | OK | 383, 0 / 389, 0 | 1 fused node on QNN | 1 node, QNN |

- The whole graph becomes a single QNN partition, and no node is rejected.
- The pruned INT8 models have 303 nodes. ONNX Runtime's
  `EnsureUniqueDQForNodeUnit` gives each of the 10 DequantizeLinear nodes that
  feed two consumers (the residual connections) its own copy, giving 313. The
  layout transform then adds two Transpose node units (3 nodes each), giving 319.
  For FP32, ONNX Runtime turns the 70 `Constant` nodes into initializers,
  leaving 100 nodes.
- **Dead-node control.** The unpruned per-tensor model also runs in strict mode:
  the QNN EP accepts the 70 dead DequantizeLinear nodes as standalone node units
  (71 DequantizeLinear units instead of 1), and its NPU outputs are
  bit-identical to the pruned model's. The dead nodes alone do not break strict
  mode; pruning is still worth keeping because it makes node counts meaningful.
- Session creation logs one warning: "EPContext encryption config unavailable
  (Experimental function OrtEpApi_ReleaseEpContextConfig_SinceV28 is not
  available in this build)". It is harmless; the QNN backend also prints
  `DSP_INFO UNSUPPORTED_KEY` lines.

### Accuracy

500 eval images, shared preprocessing. "Same model on CPU" means the same ONNX
file on the CPU execution provider (FP32 for the FP16 row).

| Configuration | Top-1 (1000) | Top-1 (10) | Agreement with CPU FP32 (1000 / 10) | Agreement with same model on CPU (1000 / 10) | Max abs logit diff vs same model on CPU |
|---|---|---|---|---|---|
| CPU FP32 | 71.0% | 98.4% | – | – | – |
| CPU INT8 per-tensor | 70.4% | 98.6% | 85.6% / 98.2% | – | – |
| CPU INT8 per-channel u8w | 69.4% | 98.4% | 92.6% / 99.2% | – | – |
| CPU INT8 per-channel s8w | 69.8% | 98.6% | 93.6% / 99.6% | – | – |
| NPU INT8 per-tensor | 70.6% | 98.6% | 86.2% / 98.2% | 97.2% / 99.6% | 0.684 |
| NPU INT8 per-channel u8w | 69.8% | 98.8% | 92.8% / 99.2% | 96.6% / 99.6% | 0.570 |
| NPU INT8 per-channel s8w | 70.4% | 98.4% | 93.6% / 99.6% | 97.6% / 99.6% | 0.627 |
| NPU FP16 (FP32 model) | 71.0% | 98.4% | 100.0% / 100.0% | 100.0% / 100.0% | 0.052 |

- INT8 on the NPU is as accurate as INT8 on the CPU, but not numerically
  identical to it: QNN's integer kernels round differently from ONNX Runtime's
  CPU QDQ kernels, so 2–3% of images change their 1000-class prediction and
  logits differ by up to ~0.7.
- FP16 on the NPU matches CPU FP32 on every image.
- Accuracy runs are deterministic: repeated runs give identical numbers.

### Initialization and latency

> **Sanity check, not a benchmark.** Python, single-image `Run()` on one fixed
> eval image, 20 warmup + 200 timed runs, nearest-rank percentiles, one run of
> the script. Configurations were interleaved CPU/NPU with a 5 s cooldown. Use
> these numbers only to see orders of magnitude.

Session creation, median of 3 (ms):

| Configuration | Cold (no EP context) | EP context generation | From cached EP context |
|---|---|---|---|
| NPU INT8 per-tensor | 2554 | 2512 | 474 |
| NPU INT8 per-channel u8w | 2288 | 2403 | 364 |
| NPU INT8 per-channel s8w | 2437 | 2472 | 377 |
| NPU FP16 | 2546 | 2464 | 388 |
| CPU FP32 / INT8 | 24 / 44–61 | – | – |

- "Cold" means no EP context cache: QNN composes and finalizes the HTP graph on
  every session creation. With `ep.context_enable=1` and an explicit
  `ep.context_file_path` (under `cache/qnn/`, not committed), ONNX Runtime writes
  a small `*_ctx.onnx` model containing a single `EPContext` node plus a
  `*_ctx_qnn.bin` context binary (3.9 MB for INT8, 8.1 MB for FP16). Sessions
  created from the context model skip graph preparation, are 5–7× faster to
  create, also pass strict mode with everything on QNN, and give bit-identical
  outputs to the uncached session on all 500 eval images.
- The first NPU session in a configuration grows the process working set by
  230–250 MB (INT8) / ~180 MB (FP16), against ~10–17 MB for a CPU session. Later
  deltas are dominated by memory being released and reused, so they are not
  reported.

Steady-state latency of one `Run()` (ms):

| Configuration | Mean | p50 | p95 | Min |
|---|---|---|---|---|
| CPU FP32 | 9.86 | 7.89 | 16.32 | 5.88 |
| CPU INT8 per-tensor | 3.51 | 2.85 | 7.12 | 2.23 |
| CPU INT8 per-channel u8w | 3.33 | 2.85 | 5.43 | 2.40 |
| CPU INT8 per-channel s8w | 3.16 | 2.51 | 6.25 | 1.56 |
| NPU INT8 per-tensor | 0.477 | 0.433 | 0.667 | 0.350 |
| NPU INT8 per-channel u8w | 0.357 | 0.345 | 0.414 | 0.316 |
| NPU INT8 per-channel s8w | 0.352 | 0.343 | 0.393 | 0.318 |
| NPU FP16 | 0.661 | 0.630 | 0.727 | 0.593 |

- CPU timing sessions use `intra_op_num_threads=4`. With ONNX Runtime's default
  (one thread per core, 10 here), single-image latency on this machine was
  bimodal and unstable under light background load. In back-to-back sessions,
  FP32 ran at p50 3.7 ms in one and 49 ms in another, and the INT8 s8w model
  at p50 8 ms with p95 52 ms. With 4 threads the same models ran at a steady
  p50 of 7.6–7.8 ms (FP32) and 2.4 ms (INT8 s8w). The thread count has to be a
  controlled, recorded benchmark parameter.
- NPU latency is tight (p95 within ~0.1–0.25 ms of p50) and about 7–9× lower
  than INT8 on 4 CPU threads; INT8 and FP16 on the NPU are 15–28× faster than
  FP32 on the CPU.
- **Thermals and power.** The power mode was *Balanced* (on AC) throughout.
  Each timed run lasts well under a second; behaviour under sustained load
  (thermal throttling) has not been measured.

## Implications

**Works natively on ARM64:**

- PyTorch + torchvision + timm, including pretrained-model inference and ONNX
  export. The PyTorch-side model-prep pipeline can run on-device.
- ONNX and ONNX Runtime CPU inference, from Python and C++. The CPU export path
  is numerically faithful to PyTorch (1e-5 max abs diff).
- The QNN execution provider on the NPU, from Python and C++, with the whole
  FP32 MobileNetV2 graph running on HTP, and the statically quantized INT8 QDQ
  models running entirely on HTP in strict mode (no CPU fallback).
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
  such as top-1 agreement, not a tight `allclose`. The same holds for INT8:
  NPU and CPU INT8 kernels round differently (see above).
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

The quantization and NPU results come from the data and model pipeline in
[setup.md](setup.md#preparing-data-and-models), ending with
`scripts\eval_cpu.py` and `scripts\npu_check.py`.

`check_env.py` prints system information and a PASS / FAIL / NOT INSTALLED /
SKIP line per library, with error messages for failures. `--json PATH` also
writes the full results (including tracebacks) as JSON. The exported model is
written to `tmp/` (override with `--tmp-dir`). The first run downloads the
MobileNetV2 weights (~14 MB) into the PyTorch hub cache. The script exits with
status 1 if any check FAILs. It reports a non-native interpreter rather than
exiting, so it can diagnose a misconfigured environment.
