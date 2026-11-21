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

`tools/ort_probe` (C++, ONNX Runtime C++ API; see also [C++ path](#c-path)) registers the QNN plugin with
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
- **Rejection detection is validated against a real rejection.**
  `tools/qnn_rejection_check.py` builds a `Relu → Det → Relu` model (the QNN EP
  does not implement `Det`). Without strict mode, the QNN EP logs
  `Validation FAILED for 1 nodes in OrtNodeUnit (Det)` followed by the node
  and the reason ("Operators of type `Det` are not supported by QNN EP"), and
  ONNX Runtime places `Det` on the CPU EP and the two Relus in two QNN
  partitions. With strict mode, session creation fails with "This session
  contains graph nodes that are assigned to the default CPU EP, but fallback to
  CPU EP has been explicitly disabled by the user", and the EP logs
  `Unsupported nodes in QNN EP: { name: unsupported_det, type: Det }`. The
  parser reports the rejected node, its reason, and the CPU placement in both
  cases; trimmed copies of these logs are the parser's test fixtures. Two other
  candidates behaved differently than one might expect: `NonZero` is rejected,
  but for its dynamic output shape, and a float64 `Cast → Add → Cast` chain is
  accepted by the QNN EP, so "double precision" is not a reliable way to provoke
  a rejection.
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

Session creation, median of 3 (ms), run A (run B below the table):

| Configuration | Cold (no EP context) | EP context generation | From cached EP context |
|---|---|---|---|
| NPU INT8 per-tensor | 2554 | 2512 | 474 |
| NPU INT8 per-channel u8w | 2288 | 2403 | 364 |
| NPU INT8 per-channel s8w | 2437 | 2472 | 377 |
| NPU FP16 | 2546 | 2464 | 388 |
| CPU FP32 / INT8 | 24 / 44–61 | – | – |

- Run B: cold 1.86–2.19 s, cached 0.25–0.27 s for the NPU configurations.
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

Steady-state latency of one `Run()` (ms), from two runs of the script on the
same day with the same settings (run B was later):

| Configuration | Run A mean | Run A p50 / p95 / min | Run B mean | Run B p50 / p95 / min |
|---|---|---|---|---|
| CPU FP32 | 9.86 | 7.89 / 16.32 / 5.88 | 4.07 | 3.95 / 4.87 / 3.62 |
| CPU INT8 per-tensor | 3.51 | 2.85 / 7.12 / 2.23 | 1.63 | 1.59 / 1.89 / 1.46 |
| CPU INT8 per-channel u8w | 3.33 | 2.85 / 5.43 / 2.40 | 1.64 | 1.60 / 1.93 / 1.43 |
| CPU INT8 per-channel s8w | 3.16 | 2.51 / 6.25 / 1.56 | 1.44 | 1.40 / 1.74 / 1.23 |
| NPU INT8 per-tensor | 0.477 | 0.433 / 0.667 / 0.350 | 0.356 | 0.343 / 0.409 / 0.321 |
| NPU INT8 per-channel u8w | 0.357 | 0.345 / 0.414 / 0.316 | 0.338 | 0.323 / 0.403 / 0.308 |
| NPU INT8 per-channel s8w | 0.352 | 0.343 / 0.393 / 0.318 | 0.399 | 0.371 / 0.500 / 0.353 |
| NPU FP16 | 0.661 | 0.630 / 0.727 / 0.593 | 0.614 | 0.601 / 0.682 / 0.582 |

- CPU timing sessions use `intra_op_num_threads=4`. With ONNX Runtime's default
  (one thread per core, 10 here), single-image latency on this machine was
  bimodal and unstable under light background load. In back-to-back sessions,
  FP32 ran at p50 3.7 ms in one and 49 ms in another, and the INT8 s8w model
  at p50 8 ms with p95 52 ms. With 4 threads the same models ran at a steady
  p50 of 7.6–7.8 ms (FP32) and 2.4 ms (INT8 s8w). The thread count has to be a
  controlled, recorded benchmark parameter.
- **Even with a fixed thread count, CPU latency moved by ~2× between the two
  runs**, while NPU latency moved by at most ~25% (per configuration, within
  0.02–0.12 ms). Nothing in the configuration changed; the machine's background
  state did. CPU numbers need repeated runs, and a recorded system state, before
  they mean anything.
- NPU latency is tight within a run (p95 within ~0.05–0.25 ms of p50). INT8 on
  the NPU is 3.6–9× faster than INT8 on 4 CPU threads, depending on the run, and
  INT8 or FP16 on the NPU is 7–28× faster than FP32 on the CPU.
- Per-channel u8w and s8w perform the same on the NPU; the per-run differences
  between the INT8 variants are within run-to-run noise.
- **Thermals and power.** The power mode was *Balanced* (on AC) throughout.
  Each timed run lasts well under a second; behaviour under sustained load
  (thermal throttling) has not been measured.

## AI Hub baseline

To see whether our quantization leaves speed or accuracy on the table,
`scripts/compare_aihub.py` runs Qualcomm AI Hub's pre-quantized MobileNetV2
through the same strict NPU setup (same harness, `scripts/common/npu_harness.py`)
next to our per-tensor and per-channel s8w INT8 models. Results below were
collected on 2026-11-20, plugged in, Windows power mode *Balanced*.

### The asset

| | |
|---|---|
| Source | Hugging Face [`qualcomm/MobileNet-v2`](https://huggingface.co/qualcomm/MobileNet-v2) at revision `31a0cdd09d996f7d71847ce14a83d92ccb747caa` (AI Hub Models release 0.67.0). The repository holds the model card and `release_assets.json`; the assets themselves are on Qualcomm's public S3 bucket. No account or token is needed. The older `qualcomm/MobileNet-v2-Quantized` repository no longer exists |
| Asset | `mobilenet_v2-onnx-w8a8.zip` ("ONNX, w8a8, Universal"), S3 object version `sZ4TesoRKcf.fYEFLxRCWMdgKSihVzMh`, SHA256 `4e0ab0c7ec6136a7cccbbe96e6f2213e6cc259e7a08bcc2b699c44e6d7c8341e`. Its float twin `mobilenet_v2-onnx-float.zip` (SHA256 `4a479de6925b49e60317ab2e52827e5e8dd94e997d8994afba62855907eb8837`) is also downloaded |
| Built with | QAIRT 2.51.0, ONNX Runtime 1.30.0 (we run QNN SDK 2.52.0 via onnxruntime-qnn 2.8.0 and ONNX Runtime 1.32.0; it loads and runs without issues) |
| License | Model card: Apache-2.0. Original implementation ([tonylins/pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2)): Apache-2.0. `qai_hub_models` source: BSD-3-Clause |
| Contents | `mobilenet_v2_onnx/mobilenet_v2.onnx` (164 KB graph) + `mobilenet_v2.data` (14.1 MB external data: the weights, stored as float32), `labels.txt`, `metadata.json`. No pre-compiled QNN context binary |
| Weights | Not torchvision's: AI Hub loads the tonylins checkpoint (reported 71.8% ImageNet top-1; torchvision's `IMAGENET1K_V2` reports 72.15%) |

### Inputs, outputs and preprocessing

| | AI Hub w8a8 | Ours |
|---|---|---|
| Input | `image_tensor`, **uint8** [1, 3, 224, 224], NCHW RGB, scale 0.0039182 (≈ 1/255), zero point 0 | `input`, float32 [1, 3, 224, 224], NCHW RGB |
| Normalization | Inside the graph (`Sub` mean, `Div` std, both quantized), on a [0, 1] image | On the CPU before the model (shared preprocessing) |
| Output | `class_logits`, **uint8** [1, 1000], scale 0.131066, zero point 76 | `logits`, float32 [1, 1000] |
| Classes | ImageNet-1k in torchvision order (`labels.txt` differs only in two names: 134 "crane", 639 "maillot") | ImageNet-1k |

AI Hub's own ImageNet transform (`qai_hub_models`, `IMAGENET_TRANSFORM`) is
`Resize(256)`, `CenterCrop(224)`, `ToTensor()`, the same as our shared
pipeline up to the normalization step. The AI Hub model therefore gets the
shared pipeline's [0, 1] tensor before normalization, quantized with the
graph's input scale and zero point. This reproduces the original 8-bit pixel
values exactly on all 500 eval images. Its uint8 output is dequantized with the
graph's output parameters (`uint8_io_adapter` in `npu_harness.py`). AI Hub's
float model takes the same [0, 1] tensor as float32.

### Graph comparison

| | Ours per-tensor | Ours per-channel s8w | AI Hub w8a8 |
|---|---|---|---|
| Opset | 17 | 17 | 21 |
| Nodes (compute) | 303 (65) | 303 (65) | 470 (101) |
| Compute ops | 52 Conv, 10 Add, GlobalAveragePool, Flatten, Gemm | same | 52 Conv, 35 Clip, 10 Add, ReduceMean, Gemm, Sub, Div |
| Weights | uint8, per-tensor, asymmetric | int8, per-channel, symmetric | int8, per-channel, symmetric (float initializers quantized by in-graph QuantizeLinear) |
| Biases | int32 | int32, per-channel | int32, per-channel |
| Activations | uint8 | uint8 | uint8 |
| Every compute node in a QDQ unit | yes | yes | yes |
| Graph I/O | float32 | float32 | uint8 |
| QNN validation, pass 1 / 2 | 313 / 319 nodes, 0 rejected | 313 / 319, 0 rejected | 491 / 494, 0 rejected |
| HTP ops (QNN profile) | 71 | 71 | 105 |

AI Hub's quantization recipe is the same as our s8w variant (per-channel
symmetric int8 weights, uint8 activations). The structural differences are
that it keeps the ReLU6 `Clip` nodes as separate quantized units (our quantizer
folds them into the Conv output ranges), normalizes the input inside the
graph, and uses uint8 graph I/O.

### Placement

The AI Hub model passes strict mode with all nodes on QNN, checked the same
three ways as our models: QNN validation passes every node (0 rejected),
ONNX Runtime places a single fused node on the QNN EP, and the ORT profile shows
only the QNN EP executing. Its cached EP-context session also passes strict
mode and gives bit-identical outputs. It runs on the CPU EP as well.

### Accuracy

500 eval images. "Own float" is each model's float source model: our FP32
export, or AI Hub's float ONNX model.

| Configuration | Top-1 (1000) | Top-1 (10) | Agreement with own float (1000 / 10) | Agreement with same model on CPU (1000) | Max abs logit diff vs CPU |
|---|---|---|---|---|---|
| CPU ours FP32 | 71.0% | 98.4% | – | – | – |
| CPU ours INT8 per-tensor | 70.4% | 98.6% | 85.6% / 98.2% | – | – |
| CPU ours INT8 s8w | 69.8% | 98.6% | 93.6% / 99.6% | – | – |
| CPU AI Hub float | 77.4% | 99.2% | – | – | – |
| CPU AI Hub w8a8 | 75.2% | 99.0% | 84.2% / 99.8% | – | – |
| **NPU ours INT8 per-tensor** | 70.6% | 98.6% | 86.2% / 99.0% | 97.2% | 0.68 |
| **NPU ours INT8 s8w** | 70.4% | 98.4% | 93.6% / 99.4% | 97.6% | 0.63 |
| **NPU AI Hub w8a8** | 75.4% | 99.0% | 85.6% / 99.6% | 93.6% | 3.93 |

Held-out check: only 134 Imagenette val images come from ImageNet val, so
neither set of weights was trained on them (CPU, top-1 over 1000 classes):
ours FP32 71.6% (96/134), ours s8w 70.1% (94), AI Hub float 74.6% (100),
AI Hub w8a8 75.4% (101). With 134 images, one image is 0.75 points and the
standard error is about 4 points.

### Latency and initialization

NPU only, 3 rounds with the order rotated each round (every model timed once
in each position), 20 warmup + 200 timed runs, 3 session creations per init
measurement, 5 s cooldown. Medians over rounds. **Sanity check, not a benchmark.**

| NPU, strict | Cold init (ms) | Cached init (ms) | Mean | p50 | p95 | Min | p50 per round |
|---|---|---|---|---|---|---|---|
| Ours INT8 per-tensor | 2102 | 310 | 0.345 | 0.328 | 0.413 | 0.309 | 0.348 / 0.320 / 0.328 |
| Ours INT8 s8w | 2160 | 299 | 0.367 | 0.339 | 0.411 | 0.322 | 0.339 / 0.336 / 0.356 |
| AI Hub w8a8 | 2230 | 304 | 0.518 | 0.496 | 0.583 | 0.468 | 0.496 / 0.492 / 0.499 |

The AI Hub model card lists 0.456 ms for this asset on a Snapdragon X Elite,
in line with the 0.50 ms measured here on an X Plus.

### Findings

- **Speed: ours is faster.** AI Hub's model has a 46–51% higher p50 than either
  of ours, consistently in every round. Init times are on par.
- **Why.** QNN's detailed profiler (`compare_aihub.py --qnn-profile`) puts the
  AI Hub model at 479 µs of accelerator execute time against 298 µs for ours.
  Its in-graph normalization accounts for most of the difference: the `Div`
  takes 33.5% and the `Sub` 16.0% of its HTP cycles (element-wise quantized
  arithmetic on the full 3×224×224 input). The 52 convolutions cost about the
  same in both models. The kept `Clip` nodes are not the cause: re-quantizing
  our model with the Clips kept (`keep_removable_activations=True`) produced
  the same 494-node QNN graph shape and the same p50 (0.327 vs 0.330 ms over 4
  rounds). Our models do the same normalization on the CPU before `Run()`, at
  0.027 ms in NumPy. Folding it into the first convolution would remove it
  entirely.
- **Accuracy: the gap is in the weights, not the quantization.** AI Hub's model
  scores 5 points higher on the eval split, but its *float* model is 6.4
  points higher than ours, and its quantization loses more: 2.0–2.2 points and
  84–86% agreement with its float model, against 0.6–1.2 points and 93.6%
  agreement for our s8w model. On the 134 held-out images the float gap shrinks
  to 3 points (4 images), within noise. Published ImageNet val accuracy is
  71.8% (tonylins) vs 72.15% (torchvision V2). The eval split is mostly
  ImageNet train images, which favours a checkpoint that fits its training data
  more closely, so most of the eval-split gap is probably a property of the
  checkpoints on this data rather than a real accuracy difference.
- **Conclusion.** Our per-channel s8w recipe is at parity with AI Hub's on
  placement and quantization scheme. It is faster on the NPU and loses less
  accuracy to quantization. Per-channel s8w is the default INT8 recipe from here on.

## C++ path

`tools/ort_probe` runs the same models from native C++ with the ONNX Runtime
1.32.0 NuGet package and the QNN plugin EP 2.8.0 (`scripts/fetch_deps.ps1`), using
the same session settings as the Python harness. Results below were collected on
2026-11-20, plugged in, Windows power mode *Balanced*.

### Device selection

The probe registers `onnxruntime_providers_qnn.dll` (full path next to the
executable) with `Ort::Env::RegisterExecutionProviderLibrary`, lists
`Ort::Env::GetEpDevices()`, and attaches the session with
`Ort::SessionOptions::AppendExecutionProvider_V2` to the single QNN EP device
whose `Device().Type()` is `OrtHardwareDeviceType_NPU`. It fails if there is
not exactly one. Devices seen from C++, identical to Python:

| EP | EP vendor | Type | Vendor ID | Device ID | Metadata |
|---|---|---|---|---|---|
| CPUExecutionProvider | Microsoft | CPU | 0x4d4f4351 ("QCOM") | 0x7 | Description = Snapdragon(R) X 10-core X1P64100 @ 3.42 GHz |
| QNNExecutionProvider | Qualcomm | NPU | 0x4d4f4351 | 0x41304430 | LUID = 70553 |
| QNNExecutionProvider | Qualcomm | GPU | 0x4d4f4351 | 0x36334330 | Adreno(TM) X1-85, DXGI adapter 0, LUID = 68962 |

The hardware vendor string is empty for both QNN devices; only the numeric
vendor ID is set.

### Which onnxruntime.dll is loaded

Windows 11 ships an unrelated `onnxruntime.dll` in `C:\Windows\System32` (a
forwarder to the OS's own ONNX Runtime, version 1.17.1 here). It is not a
KnownDLL, so the loader's standard search order, which checks the
application directory first, finds the copy the build places next to the
executable. That alone is not a guarantee one can rely on, and a missing copy
fails badly. With only `ort_probe.exe` in a directory, the loader picks up
the System32 copy, and the ONNX Runtime C++ API (which by default initializes
itself during static initialization) asks it for C API version 32. ORT 1.17
answers "The requested API version [32] is not available, only API versions
[1, 17] are supported", and the process crashes with an access violation
before `main()` runs.

The probe therefore:

1. **delay-loads** `onnxruntime.dll` (`/DELAYLOAD:onnxruntime.dll`) and builds with
   `ORT_API_MANUAL_INIT`, so nothing touches ONNX Runtime before `main()`;
2. loads `<exe dir>\onnxruntime.dll` **by full path** with
   `LoadLibraryExW(..., LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`.
   Later delay-load resolution finds this already-loaded module by name;
3. checks that `OrtGetApiBase()->GetApi(ORT_API_VERSION)` succeeds before calling
   `Ort::InitApi(api)`, and otherwise exits with a clear message;
4. after creating the session, asserts that the loaded `onnxruntime.dll` is the
   one next to the executable and lists every loaded ONNX Runtime / QNN module.

Verified outcomes:

| Situation | Result |
|---|---|
| Normal build output, run from the repository root, from `%TEMP%`, from `C:\`, and with `C:\Windows\System32` as working directory (PowerShell and cmd) | Bundled `onnxruntime.dll` loaded, check OK |
| `ort_probe.exe` alone in a directory | "onnxruntime.dll not found (the build copies it next to the executable)", exit 1 |
| System32's `onnxruntime.dll` copied next to the executable | "...is ONNX Runtime 1.17.1, which does not provide C API version 32", exit 1 (no crash) |

With an NPU session alive (built from an EP context model), the loaded modules
are, all from the executable's directory: `onnxruntime.dll`,
`onnxruntime_providers_shared.dll`, `onnxruntime_providers_qnn.dll`,
`QnnHtp.dll`, `QnnHtpV73Stub.dll`, `QnnSystem.dll`. `QnnHtpPrepare.dll` is only
needed when a session compiles the graph; sessions created from a context binary
do not load it.

### C++ vs Python outputs

`tools/ort_probe/make_reference.py` writes 10 preprocessed eval images (the
first of each class) and the Python outputs for four configurations; the probe
reads the same input tensors (no image decoding in C++) and compares. CPU
sessions use 4 intra-op threads in both languages. Each Python configuration was
also run in a second, independent session to measure run-to-run variation.

| Config | Python repeat variation | C++ vs Python max abs diff (Run / IoBinding) | Tolerance | Top-1 match |
|---|---|---|---|---|
| CPU FP32 | 0 | 0 / 0 | 1e-5 | 10/10 |
| CPU INT8 s8w | 0 | 0 / 0 | 1e-5 | 10/10 |
| NPU INT8 s8w, strict | 0 | 0 / 0 | 0 | 10/10 |
| NPU FP16, strict | 0 | 0 / 0 | 0 | 10/10 |

The outputs are bit-identical. NPU execution is deterministic run to run and
across sessions, so the NPU tolerance is 0. C++ IoBinding with preallocated
buffers gives the same outputs as `Session::Run`.

### EP context cache

Same model, strict mode, median of 3 session creations:

| Config | Cold (compile) | EP context generation | From cached context | Cached = uncached outputs |
|---|---|---|---|---|
| NPU INT8 s8w | 2139 ms | 2005 ms | 277 ms | yes (bit-identical) |
| NPU FP16 | 2073 ms | 2277 ms | 323 ms | yes (bit-identical) |

Context models are interchangeable between the two bindings: Python loads the
one generated by the C++ probe (all Python NPU timings in the overhead breakdown
below use it), and the probe loads one generated by the Python harness and still
matches Python's outputs exactly. A context binary is tied to the device and the
QNN SDK version, not to the language binding.

### Strict-mode failure

The `Relu → Det → Relu` model from `tools/qnn_rejection_check.py --write-model`:

- **strict on**: session creation throws `Ort::Exception`; the probe prints
  `ONNX Runtime error 1 (ORT_FAIL): This session contains graph nodes that are
  assigned to the default CPU EP, but fallback to CPU EP has been explicitly
  disabled by the user.`, writes the code and message to its JSON summary, and
  exits with 3. No crash.
- **strict off**: the session is created (Det on the CPU EP, the Relus on QNN),
  runs, and matches a CPU-only run (max abs diff 4.7e-10).

### Overhead breakdown on the NPU

`scripts/npu_overhead.py`: per-channel s8w INT8, strict, cached EP context (the
same context model for every mode), `htp_performance_mode=burst`, one fixed
input. 20 warmup + 200 timed runs, 3 rounds with the order rotated each round,
5 s cooldown, CPU and NPU, C++ and Python interleaved. **Sanity check, not a
benchmark.** Latency is model execution only (the `Run` call); preprocessing is
not timed.

Accelerator time per inference comes from separate runs with the QNN profiler
at `profiling_level=basic`. Profiling inflates host-side time, so wall times
from those runs are not used. Host overhead is the wall-clock p50 minus the
accelerator execute time.

| Mode | Wall p50 (µs) | p50 per round | Wall mean | Accelerator execute | Host overhead | Share |
|---|---|---|---|---|---|---|
| C++ `Session::Run` | 327 | 321 / 327 / 329 | 338 | 275 | 52 | 16% |
| C++ IoBinding, preallocated | 316 | 318 / 316 / 316 | 340 | 273 | 43 | 14% |
| Python `InferenceSession.run` | 339 | 353 / 339 / 334 | 379 | 277 | 63 | 19% |
| Python IoBinding, preallocated | 351 | 351 / 354 / 348 | 384 | 276 | 75 | 21% |

The QNN profiler nests the accelerator time inside "QNN accelerator (execute)"
(~290 µs) and "RPC (execute)" (~300 µs), the call across to the DSP.

- The accelerator time is the same in every mode (273–277 µs): the binding
  only changes host-side cost.
- **About 80–86% of single-image latency is the accelerator.** Host-side
  overhead is 43–75 µs. C++ IoBinding is the leanest (43 µs); plain C++ `Run`
  adds ~9 µs (creating the input `Ort::Value` and allocating the output);
  Python adds another ~10 µs.
- Python IoBinding is *slower* than plain Python `run` (75 vs 63 µs), so in
  Python the binding API is not a win at this size.
- Python's mean is well above its p50 (379 vs 339 µs) and its p95 is higher
  (525 vs 389 µs for C++ Run), so C++ is also more consistent.

CPU sanity check, same run, 4 intra-op threads (ms, median over rounds):

| Model | C++ p50 | Python p50 | C++ mean | Python mean |
|---|---|---|---|---|
| INT8 s8w | 1.353 | 1.350 | 1.417 | 1.422 |
| FP32 | 3.958 | 3.967 | 4.207 | 4.114 |

On the CPU, C++ and Python are within 1%, as expected when compute dominates.

### Plugin EP API notes (C++)

Things that matter for the engine's QNN provider:

- **Load ONNX Runtime explicitly.** Delay-load `onnxruntime.dll`, define
  `ORT_API_MANUAL_INIT`, load the bundled DLL by full path, check
  `GetApi(ORT_API_VERSION)` and call `Ort::InitApi(api)`. Otherwise a wrong or
  missing DLL crashes during static initialization with no chance to report
  it. The engine executable needs the same treatment.
- **Registration.** `Env::RegisterExecutionProviderLibrary(name, full_path)`
  takes a wide-string path on Windows and must use an absolute path. Register
  once per `Env`, and release all sessions before
  `UnregisterExecutionProviderLibrary`.
- **Device enumeration.** `Env::GetEpDevices()` returns `ConstEpDevice` views
  owned by the environment. Select by `EpName() == "QNNExecutionProvider"` and
  `Device().Type() == OrtHardwareDeviceType_NPU`, not by position: the QNN EP
  also exposes the Adreno GPU. `Device().Vendor()` is empty for QNN devices;
  use `VendorId()` / `Metadata()` instead.
- **Attaching to a device.** `SessionOptions::AppendExecutionProvider_V2(env,
  {device}, options)` takes the EP options as a `std::unordered_map<std::string,
  std::string>`. It replaces the provider name. The plugin EP never appears in
  `GetAvailableProviders()`.
- **Provider options vs session config.** QNN options (`backend_path`,
  `htp_performance_mode`, `enable_htp_fp16_precision`, `profiling_level`,
  `profiling_file_path`) go in the provider-options map. Strict mode
  (`session.disable_cpu_ep_fallback`) and EP context (`ep.context_enable`,
  `ep.context_file_path`, `ep.context_embed_mode`) are **session config
  entries** (`AddConfigEntry`). `backend_path` should be an absolute path to the
  `QnnHtp.dll` shipped with the plugin, because QnnHtp loads its stub/skel files
  from its own directory.
- **EP context.** Generating is a side effect of creating a session with
  `ep.context_enable=1`. The default embed mode 0 writes `<name>_ctx.onnx` (~1 KB,
  one `EPContext` node) plus `<name>_ctx_qnn.bin`. Delete old files before
  regenerating. Load the `_ctx.onnx` like any model, with the same
  device/options, and strict mode still applies. `Ort::CompileModel` with
  `ModelCompilationOptions` is the newer explicit API for the same thing.
- **Errors.** Strict-mode refusal arrives as `Ort::Exception` with
  `GetOrtErrorCode() == ORT_FAIL` at session creation, not at run time.
- **Type info lifetime.** `Session::GetInputTypeInfo(i).GetTensorTypeAndShapeInfo()`
  on a temporary returns a dangling view (garbage element types). Keep the
  `TypeInfo` alive while using its tensor info.
- **IoBinding.** Bind `Ort::Value`s created over preallocated CPU buffers
  (`MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)`) once, then
  call `Session::Run(RunOptions, IoBinding)` repeatedly. This works with the QNN
  plugin EP and is the lowest-overhead mode measured.
- **QNN profiling** is configured per session through provider options and
  written when the session is released. It adds roughly 12 ms of host time per
  inference, so never time a profiled session.

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
.\build\arm64-release\bin\Release\ort_probe.exe --model tmp\mobilenetv2.onnx --target npu --fill 0.5
```

The quantization and NPU results come from the data and model pipeline in
[setup.md](setup.md#preparing-data-and-models), ending with
`scripts\eval_cpu.py`, `scripts\npu_check.py` and `scripts\compare_aihub.py`
(`--qnn-profile` for the per-op profile). The rejection check is
`tools\qnn_rejection_check.py`. The C++ results come from
`tools\ort_probe\make_reference.py`, `ort_probe.exe` (see
[setup.md](setup.md#3-run-ort_probe)) and `scripts\npu_overhead.py`.

`check_env.py` prints system information and a PASS / FAIL / NOT INSTALLED /
SKIP line per library, with error messages for failures. `--json PATH` also
writes the full results (including tracebacks) as JSON. The exported model is
written to `tmp/` (override with `--tmp-dir`). The first run downloads the
MobileNetV2 weights (~14 MB) into the PyTorch hub cache. The script exits with
status 1 if any check FAILs. It reports a non-native interpreter rather than
exiting, so it can diagnose a misconfigured environment.
