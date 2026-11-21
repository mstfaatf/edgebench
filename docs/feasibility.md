# Feasibility: CPU vs NPU inference on Windows on Snapdragon

Technical note on whether ONNX Runtime can run, verify and time image
classification models on the CPU and on the Hexagon NPU of a Snapdragon X
laptop, from Python and from native C++. All measurements were taken on
2026-11-20 on one device. Latency figures in this note are **sanity checks for
methodology, not published benchmark results**.

## 1. Summary

MobileNetV2 was exported from torchvision (FP32), statically quantized to INT8
QDQ in three variants with ONNX Runtime's QNN helpers, and run on the ONNX
Runtime CPU execution provider and on the NPU through the Qualcomm QNN plugin
execution provider, from Python and from C++. A pre-quantized MobileNetV2 from
Qualcomm AI Hub served as an external baseline.

- Every INT8 model and the FP32 model with FP16 precision run **entirely on
  the NPU** with CPU fallback disabled. Three independent checks confirm this,
  and the detection of rejected nodes is validated on a real rejection.
- INT8 on the NPU loses 0.6 points of top-1 against FP32 on the CPU for the
  default recipe (per-channel int8 symmetric weights), and FP16 on the NPU
  matches FP32 on every evaluated image.
- Single-image NPU latency (p50) is about 0.32–0.37 ms for INT8 and 0.60 ms for
  FP16, against 1.4 ms (INT8) and 4.0 ms (FP32) on 4 CPU threads in the same run.
  About 80–86% of NPU wall time is accelerator time.
- C++ and Python produce **bit-identical** outputs for every configuration
  tested. EP context caches cut NPU session creation from about 2 s to about
  0.3 s and work across both bindings.
- Our quantization recipe matches Qualcomm AI Hub's model on placement. It is
  faster on the NPU and loses less accuracy to quantization.
- Several conditions must hold for trustworthy numbers: the bundled ONNX Runtime
  DLL must be loaded explicitly, the NPU must be selected by device type, and
  CPU timing needs an explicit thread count and repeated trials.
  [Section 10](#10-requirements-for-the-engine) lists the requirements for the
  benchmark engine.

## 2. Test environment

| | |
|---|---|
| Device | Microsoft Surface Laptop 7 |
| CPU | Snapdragon X Plus X1P64100, 10 cores. The OS reports "Snapdragon(R) X 10-core X1P64100 @ 3.40 GHz"; the ORT CPU EP device reports "… @ 3.42 GHz" |
| NPU | Hexagon NPU, used through the QNN HTP backend (`QnnHtpV73Stub.dll` is the stub loaded) |
| GPU | Adreno X1-85. Enumerated as a QNN EP device but not used |
| OS | Windows 11 Home 25H2, build 26200.9974, ARM64 |
| Power | On AC, battery 80%, battery saver off. Windows power mode *Balanced* (no power-mode overlay), power plan *Balanced*. Same before and after every measurement script |
| ONNX Runtime | 1.32.0. Python wheel `onnxruntime`, and for C++ the NuGet package `Microsoft.ML.OnnxRuntime` 1.32.0 (`runtimes/win-arm64`) |
| QNN execution provider | Plugin EP 2.8.0 (`onnxruntime-qnn` wheel; GitHub release `onnxruntime/onnxruntime-qnn` v2.8.0, maintained by Qualcomm, for C++) |
| QNN SDK / QAIRT | 2.52.0 runtime bundled with the plugin; HTP backend build `v2.52.0.261104115025`, backend API interface 2.41.0 |
| Python | 3.12.8, python.org ARM64 build, interpreter platform `win-arm64`; numpy 2.5.5, onnx 1.23.2, torch 2.13.0+cpu, torchvision 0.28.0+8fb8771 (exact pins in [`requirements.txt`](../requirements.txt)) |
| C++ | Visual Studio 2022 17.12.3, MSVC 14.42.34433 (cl 19.42.34435, ARM64-hosted, native ARM64 target), Windows SDK 10.0.26100.0, CMake 3.29.5, C++20, GoogleTest 1.18.0 |

Toolchain notes:

- Every component runs natively on ARM64. `platform.machine()` reports `ARM64`
  even inside an emulated x64 Python; `sysconfig.get_platform()` (`win-arm64`
  vs `win-amd64`) is the reliable check, and scripts that run models refuse to
  run on a non-native interpreter.
- torch and torchvision have no win-arm64 wheels on PyPI; the official builds
  come from the PyTorch package index. TensorFlow has no win-arm64 wheel, and
  tf2onnx cannot be imported without it, so Keras models have to be converted
  off-device.
- ORT and the QNN EP share one Python environment: `onnxruntime-qnn` 2.x is a
  plugin EP for the stock `onnxruntime` package. QNN EP 2.8.0 states
  compatibility with ORT ≥ 1.24.1 (tested against 1.29.0). 1.32.0 worked
  throughout; no other ORT version was tried.
- `scripts/fetch_deps.ps1` downloads the C++ ORT and QNN EP packages pinned by
  SHA256. The GitHub release page of the main ONNX Runtime repository was
  unreachable when checked, so the C++ ORT package comes from NuGet.

## 3. Models and data

### MobileNetV2 FP32

torchvision `mobilenet_v2` with `IMAGENET1K_V2` weights, exported with the
TorchScript exporter (`dynamo=False`) at opset 17, IR version 8. Fixed input
`input` float32 [1, 3, 224, 224] (no dynamic axes), output `logits` float32
[1, 1000]. The export passes `onnx.checker` (full check). Max abs difference
from PyTorch on an eval image is 7.4e-6, with the same top-1. With every input
value 0.5 the top-1 class is 677 (logit 1.860237). File: 13,992,263 bytes.

### INT8 QDQ variants

Static quantization with ONNX Runtime 1.32.0's QNN helpers
(`qnn_preprocess_model`, `get_qnn_qdq_config`, `quantize`), QDQ format, MinMax
calibration on 200 images, uint8 asymmetric activations. `qnn_preprocess_model`
makes no changes to this model.

| Variant | File | Weights | Size | Role |
|---|---|---|---|---|
| Per-channel s8w | `mobilenetv2_int8qdq_pc_s8w_b1.onnx` | int8, symmetric, per-channel (`weight_type=QInt8`, `per_channel=True`) | 3,886,138 B | **Default INT8 recipe** |
| Per-tensor | `mobilenetv2_int8qdq_b1.onnx` | uint8, asymmetric, per-tensor (helper defaults) | 3,649,744 B | Reference |
| Per-channel u8w | `mobilenetv2_int8qdq_pc_b1.onnx` | uint8, per-channel (`per_channel=True`); every channel gets zero point 128 although `WeightSymmetric` is `false` | 3,904,207 B | Measured here, not carried forward |

Biases are int32 in all variants. The quantizer folds the ReLU6 `Clip` nodes
into the Conv output ranges.

**Pruning of unused Q/DQ chains.** The quantizer leaves the 70 min/max
constants of the folded `Clip` nodes behind, each with a QuantizeLinear →
DequantizeLinear chain that nothing consumes. ONNX Runtime does not remove
them. With all graph optimizations it constant-folds the QuantizeLinear nodes,
but the 70 dead DequantizeLinear nodes remain. `scripts/quantize_qnn.py`
therefore prunes dead nodes and unused initializers until nothing changes:
513 → 303 nodes (70 each of Constant, QuantizeLinear and DequantizeLinear, in 3
passes) and 588 → 448 initializers, for every variant. Pruned and unpruned
models give bit-identical CPU logits on all 500 eval images. The unpruned model
also runs fully on the NPU (section 4), so pruning is for clean graphs and
meaningful node counts; it is not needed for correctness.

### Qualcomm AI Hub baseline

| | |
|---|---|
| Source | Hugging Face [`qualcomm/MobileNet-v2`](https://huggingface.co/qualcomm/MobileNet-v2) at revision `31a0cdd09d996f7d71847ce14a83d92ccb747caa` (AI Hub Models 0.67.0). The repository holds the model card and `release_assets.json`; the assets are on Qualcomm's public S3 bucket and need no account. The older `qualcomm/MobileNet-v2-Quantized` repository no longer exists |
| Assets | `mobilenet_v2-onnx-w8a8.zip` (S3 version `sZ4TesoRKcf.fYEFLxRCWMdgKSihVzMh`, SHA256 `4e0ab0c7ec6136a7cccbbe96e6f2213e6cc259e7a08bcc2b699c44e6d7c8341e`) and its float twin `mobilenet_v2-onnx-float.zip` (S3 version `LtCxRMlzwHos5auC84iuUfZr8QBNTYbP`, SHA256 `4a479de6925b49e60317ab2e52827e5e8dd94e997d8994afba62855907eb8837`) |
| Built with | QAIRT 2.51.0, ONNX Runtime 1.30.0. Loads and runs unchanged on ORT 1.32.0 / QNN SDK 2.52.0 |
| License | Model card: Apache-2.0. Original implementation ([tonylins/pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2)): Apache-2.0. `qai_hub_models` source: BSD-3-Clause |
| Contents (w8a8) | `mobilenet_v2.onnx` (163,817 B graph, opset 21) + `mobilenet_v2.data` (14,078,056 B external data: float32 weights, quantized by in-graph QuantizeLinear nodes), `labels.txt`, `metadata.json`. No pre-compiled QNN context binary |
| Weights | The tonylins checkpoint, not torchvision's. Reported ImageNet top-1: 71.8% (tonylins) vs 72.15% (torchvision `IMAGENET1K_V2`) |

| | AI Hub w8a8 | Ours |
|---|---|---|
| Input | `image_tensor`, uint8 [1, 3, 224, 224] NCHW RGB, scale 0.0039182, zero point 0 (a [0, 1] image) | `input`, float32 [1, 3, 224, 224] NCHW RGB, normalized |
| Normalization | In the graph (`Sub` mean, `Div` std, both quantized) | Before the model, on the CPU |
| Output | `class_logits`, uint8 [1, 1000], scale 0.131066, zero point 76 | `logits`, float32 [1, 1000] |
| Classes | ImageNet-1k in torchvision order (two label names differ: 134 "crane", 639 "maillot") | ImageNet-1k |
| Graph | 470 nodes; 101 compute nodes: 52 Conv, 35 Clip, 10 Add, ReduceMean, Gemm, Sub, Div | 303 nodes; 65 compute nodes: 52 Conv, 10 Add, GlobalAveragePool, Flatten, Gemm |
| Weights / activations | int8 per-channel symmetric / uint8 | Same (s8w variant) |

AI Hub's own ImageNet transform is `Resize(256)`, `CenterCrop(224)`,
`ToTensor()`, the same as our pipeline up to normalization. Its model therefore
receives our shared [0, 1] tensor quantized with the graph's input parameters.
This reproduces the original 8-bit pixel values exactly on all 500 eval images.
Its uint8 output is dequantized with the graph's output parameters. AI Hub's
float model takes the same [0, 1] tensor as float32.

### Images

[Imagenette](https://github.com/fastai/imagenette) `imagenette2-320`
(10 ImageNet classes; archive SHA256 `569b4497…158c`, pinned in
`scripts/get_data.py`). Seeded, deterministic, class-stratified splits:

- **calibration**: 200 images from the Imagenette train split (20 per class);
- **eval**: 500 images from the Imagenette val split (50 per class).

Preprocessing, shared by every script (`scripts/common/preprocess.py`): resize
the shorter side to 256 (bilinear), center crop 224, scale to [0, 1],
ImageNet mean/std normalization, NCHW float32. It is checked against
torchvision's transforms in the tests.

**Train-set overlap.** 488 of the 500 eval images are ImageNet-1k *train*
images (only 134 of Imagenette's 3,925 val images come from ImageNet val).
ImageNet-pretrained weights were trained on them, so absolute top-1 is
optimistic and is not an ImageNet accuracy figure. Comparisons between
variants of the same weights are unaffected. A held-out check on the 134
ImageNet-val images is in [section 5](#5-accuracy).

## 4. Verifying genuine NPU execution

**Device selection.** After registering the plugin EP, ONNX Runtime lists
three EP devices (identical in Python and C++):

| EP | EP vendor | Device type | Vendor ID | Device ID | Metadata |
|---|---|---|---|---|---|
| CPUExecutionProvider | Microsoft | CPU | 0x4d4f4351 | 0x7 | Description = Snapdragon(R) X 10-core X1P64100 @ 3.42 GHz |
| QNNExecutionProvider | Qualcomm | NPU | 0x4d4f4351 | 0x41304430 | LUID = 70553 |
| QNNExecutionProvider | Qualcomm | GPU | 0x4d4f4351 | 0x36334330 | Adreno(TM) X1-85, DXGI adapter 0, LUID = 68962 |

The QNN EP also exposes the GPU, so "the first QNN device" is not
necessarily the NPU. Sessions attach to the single QNN device whose hardware
type is NPU, and fail if there is not exactly one. The hardware vendor string
is empty for both QNN devices; only the numeric vendor ID is set.

**Strict mode.** Every NPU session sets `session.disable_cpu_ep_fallback=1`, so
session creation fails if any node would run on the CPU EP. Provider options:
`backend_path` = absolute path of the `QnnHtp.dll` shipped with the plugin,
`htp_performance_mode=burst` (confirmed in the EP log), and for FP16
`enable_htp_fp16_precision=1`.

**Three placement checks.** A strict session that is created already implies
nothing was assigned to the CPU. Placement is also checked directly:

1. the QNN EP's verbose log, which validates every ONNX node unit in two
   capability passes (before and after the NCHW → NHWC layout transform);
2. ONNX Runtime's "Node placements" summary in the session log;
3. a one-inference ORT profile, which records the EP that executed each node.

| Model | Strict session | QNN validation, pass 1 / pass 2 (nodes accepted, rejected) | ORT placement | Executed (profile) |
|---|---|---|---|---|
| INT8 per-channel s8w | created | 313, 0 / 319, 0 | 1 fused node, QNN | 1 node, QNN |
| INT8 per-tensor | created | 313, 0 / 319, 0 | 1 fused node, QNN | 1 node, QNN |
| INT8 per-channel u8w | created | 313, 0 / 319, 0 | 1 fused node, QNN | 1 node, QNN |
| FP32 + `enable_htp_fp16_precision` | created | 100, 0 / 102, 0 | 1 fused node, QNN | 1 node, QNN |
| AI Hub w8a8 | created | 491, 0 / 494, 0 | 1 fused node, QNN | 1 node, QNN |
| Unpruned INT8 per-tensor (control) | created | 383, 0 / 389, 0 | 1 fused node, QNN | 1 node, QNN |

The same checks pass for sessions created from cached EP context models
(section 6).

**Why the node counts exceed the graph's.** The pruned INT8 models have 303
nodes. ONNX Runtime's `EnsureUniqueDQForNodeUnit` gives each of the 10
DequantizeLinear nodes with two consumers (the residual connections) its own
copy: 313. The layout transform then adds two Transpose node units of 3 nodes
each: 319. For FP32, ORT turns the 70 `Constant` nodes into initializers,
leaving 100 nodes, plus 2 Transposes in pass 2. In the unpruned control, the
QNN EP accepts the 70 dead DequantizeLinear nodes as standalone units, and its
NPU outputs are bit-identical to the pruned model's.

**A validated rejection.** `tools/qnn_rejection_check.py` builds a
`Relu → Det → Relu` model; the QNN EP does not implement `Det`.

- **Strict off**: the session is created. The EP logs
  `Validation FAILED for 1 nodes in OrtNodeUnit (Det)` with the node and the
  reason ("Operators of type `Det` are not supported by QNN EP"). ORT places
  `Det` on the CPU EP and the two Relus in two QNN partitions, and the output
  matches a CPU-only run (max abs diff 4.7e-10).
- **Strict on**: session creation fails with ORT error 1 (`ORT_FAIL`), "This
  session contains graph nodes that are assigned to the default CPU EP, but
  fallback to CPU EP has been explicitly disabled by the user", and the EP logs
  `Unsupported nodes in QNN EP: { name: unsupported_det, type: Det }`.

The log parser (`scripts/common/qnn_log.py`) reports the rejected node, its
reason and the CPU placement in both cases; trimmed copies of these real logs
are its test fixtures. Two other candidates were tried. `NonZero` is rejected,
but because its output shape is dynamic. A float64 `Cast → Add → Cast` chain is
*accepted* by the QNN EP, so double precision is not a reliable way to provoke
a rejection.

Harmless log noise: each NPU session logs "EPContext encryption config
unavailable (… is not available in this build)", and the QNN backend prints
`DSP_INFO UNSUPPORTED_KEY` lines.

## 5. Accuracy

500 eval images, shared preprocessing. Top-1 is over all 1000 ImageNet
classes, and over the 10 Imagenette classes (argmax over just those 10 logits).
Agreement is the fraction of images with the same predicted class, as
1000-class / 10-class. "Own float" is our FP32 model for our variants and AI
Hub's float model for AI Hub's. Results are deterministic: repeated runs and
separate scripts give identical values.

| Configuration | Top-1 (1000) | Top-1 (10) | Agreement with our CPU FP32 | Agreement with own float | Agreement with same model on CPU | Max abs logit diff vs same model on CPU |
|---|---|---|---|---|---|---|
| CPU FP32 | 71.0% | 98.4% | – | – | – | – |
| CPU INT8 per-tensor | 70.4% | 98.6% | 85.6% / 98.2% | same | – | – |
| CPU INT8 per-channel u8w | 69.4% | 98.4% | 92.6% / 99.2% | same | – | – |
| CPU INT8 per-channel s8w | 69.8% | 98.6% | 93.6% / 99.6% | same | – | – |
| NPU INT8 per-tensor | 70.6% | 98.6% | 86.2% / 99.0% | same | 97.2% / 99.2% | 0.684 |
| NPU INT8 per-channel u8w | 69.8% | 98.8% | 92.8% / 99.6% | same | 96.6% / 99.2% | 0.570 |
| **NPU INT8 per-channel s8w** | 70.4% | 98.4% | 93.6% / 99.4% | same | 97.6% / 99.8% | 0.627 |
| NPU FP16 (FP32 model) | 71.0% | 98.4% | 100.0% / 100.0% | same | 100.0% / 100.0% | 0.052 |
| CPU AI Hub float | 77.4% | 99.2% | 77.6% / 98.2% | – | – | – |
| CPU AI Hub w8a8 | 75.2% | 99.0% | 73.0% / 98.0% | 84.2% / 99.8% | – | – |
| NPU AI Hub w8a8 | 75.4% | 99.0% | 73.0% / 98.2% | 85.6% / 99.6% | 93.6% / 99.8% | 3.93 |

Held-out check (CPU, top-1 over 1000 classes), on the 134 Imagenette val
images that come from ImageNet val and that neither checkpoint was trained on.
One image is 0.75 points, and the standard error is about 4 points:

| | Ours FP32 | Ours INT8 per-tensor | Ours INT8 s8w | AI Hub float | AI Hub w8a8 |
|---|---|---|---|---|---|
| Correct / 134 | 96 (71.6%) | 95 (70.9%) | 94 (70.1%) | 100 (74.6%) | 101 (75.4%) |

- **Quantization cost.** Per-channel weights track FP32 much more closely than
  per-tensor (93.6% vs 85.6% agreement). Differences in top-1 between the INT8
  variants are within noise for 500 images.
- **NPU vs CPU.** INT8 on the NPU is as accurate as INT8 on the CPU but not
  numerically identical. QNN's integer kernels round differently from ORT's
  CPU QDQ kernels, so 2.4–3.4% of images change their 1000-class prediction.
  CPU/NPU comparisons therefore need a task-level metric (agreement), not a
  tight `allclose`. FP16 on the NPU matches CPU FP32 on every image.
- **AI Hub's higher accuracy comes from its checkpoint, not its quantization.**
  Its float model is 6.4 points above ours on the eval split, but its
  quantization costs 2.0–2.2 points and keeps 84–86% agreement with its float
  model, against 0.6–1.2 points and 93.6% for our s8w recipe. On the held-out
  images the float gap shrinks to 3.0 points (4 images), within noise, and
  published ImageNet accuracies are 71.8% vs 72.15%. Because the eval split is
  mostly ImageNet train images, it favours a checkpoint that fits its
  training data more closely.
- The most common FP32 errors (from a one-off analysis of the eval split) are
  near-synonym classes: cassette player → tape player (14 images) and church →
  monastery (12).

## 6. Initialization

Session creation time. *Cold* means compiling the graph for the NPU (no EP
context cache). *Context build* is a session created with
`ep.context_enable=1`, which writes the cache. *Cached* means created from the
resulting EP context model. Strict mode applies to all three.

Python (`scripts/npu_check.py`, median of 3 creations; context build measured
once):

| Configuration | Cold | Context build | Cached |
|---|---|---|---|
| NPU INT8 per-channel s8w | 2195 ms | 2092 ms | 273 ms |
| NPU INT8 per-tensor | 1953 ms | 1931 ms | 272 ms |
| NPU INT8 per-channel u8w | 2071 ms | 2077 ms | 250 ms |
| NPU FP16 | 1855 ms | 1764 ms | 251 ms |
| CPU FP32 / INT8 per-tensor / u8w / s8w | 11.7 / 24.3 / 29.0 / 26.0 ms | – | – |

In `scripts/compare_aihub.py` (3 rounds × 3 creations, median), cold / cached
was 2102 / 310 ms for per-tensor, 2160 / 299 ms for s8w, and 2230 / 304 ms for
AI Hub w8a8.

C++ (`ort_probe`, median of 3 creations; context build measured once):

| Configuration | Cold | Context build | Cached |
|---|---|---|---|
| NPU INT8 s8w (`Run` mode run) | 2139 ms | 2005 ms | 277 ms |
| NPU INT8 s8w (IoBinding mode run) | 2137 ms | 2168 ms | 314 ms |
| NPU FP16 (`Run` mode run) | 2073 ms | 2277 ms | 323 ms |
| NPU FP16 (IoBinding mode run) | 2099 ms | 2053 ms | 300 ms |
| CPU FP32 / INT8 s8w (single creation) | 19–26 ms / 27–28 ms | – | – |

- A cached session is created 6.4–8.3× faster than a cold one. The cache is a
  ~1 KB `<name>_ctx.onnx` with one `EPContext` node plus a `<name>_ctx_qnn.bin`
  (3.9 MB for INT8, 8.1 MB for FP16).
- Cached sessions give bit-identical outputs to uncached ones (Python: all 500
  eval images; C++: the 10 reference images), and pass strict mode with every
  node on QNN.
- **Cross-language compatibility.** Python loads context models generated by
  the C++ probe (every Python NPU timing in section 7.3 uses one). The probe
  loads a context model generated by the Python harness and still matches
  Python's outputs exactly. A context binary depends on the device and the QNN
  SDK version, not on the language binding.
- Sessions created from a context binary do not load `QnnHtpPrepare.dll`.
- Working set. The first NPU session in a configuration grows the Python
  process working set by 236–259 MB for INT8 and 186 MB for FP16, against
  9–17 MB for a CPU session. Later creations in the same process are dominated
  by memory being released and reused, so their deltas are not meaningful.
  Only the working set was measured, not NPU-side memory.

## 7. Latency sanity check

> These are **methodology sanity checks, not published benchmark numbers**.
> They are single-image `Run` calls on one fixed preprocessed image, measured
> with 20 warmup + 200 timed runs and nearest-rank percentiles, plugged in, in
> *Balanced* power mode. Latency covers model execution only (the `Run` call);
> image decoding, resizing and normalization are not timed. CPU sessions use
> 4 intra-op threads.

### 7.1 NPU and CPU from Python

`scripts/npu_check.py`, one round with CPU and NPU interleaved and a 5 s
cooldown between configurations. The script was run twice on the same day with
identical settings. Run B is the JSON summary kept in
`benchmarks/results/local/feasibility/`. Run A was recorded in an earlier
revision of this note; its summary file was overwritten by run B. The scripts
now write a new timestamped summary per run, so later runs no longer replace
earlier ones.

| Configuration | Run A mean | Run A p50 / p95 / min | Run B mean | Run B p50 / p95 / min |
|---|---|---|---|---|
| CPU FP32 | 9.86 | 7.89 / 16.32 / 5.88 | 4.07 | 3.95 / 4.87 / 3.62 |
| CPU INT8 per-tensor | 3.51 | 2.85 / 7.12 / 2.23 | 1.63 | 1.58 / 1.89 / 1.46 |
| CPU INT8 per-channel u8w | 3.33 | 2.85 / 5.43 / 2.40 | 1.64 | 1.60 / 1.93 / 1.43 |
| CPU INT8 per-channel s8w | 3.16 | 2.51 / 6.25 / 1.56 | 1.44 | 1.40 / 1.74 / 1.23 |
| NPU INT8 per-tensor | 0.477 | 0.433 / 0.667 / 0.350 | 0.356 | 0.343 / 0.409 / 0.321 |
| NPU INT8 per-channel u8w | 0.357 | 0.345 / 0.414 / 0.316 | 0.338 | 0.323 / 0.403 / 0.308 |
| NPU INT8 per-channel s8w | 0.352 | 0.343 / 0.393 / 0.318 | 0.399 | 0.371 / 0.500 / 0.353 |
| NPU FP16 | 0.661 | 0.630 / 0.727 / 0.593 | 0.614 | 0.601 / 0.682 / 0.582 |

All values in ms.

- **CPU run-to-run drift.** With identical settings and a fixed thread count,
  CPU latency changed by about 2× between the two runs, while NPU latency
  changed by at most about 25% (0.02–0.12 ms). The configuration did not
  change; the machine's background state did.
- **CPU thread count.** In a one-off experiment (output not kept as a summary
  file), ORT's default intra-op thread count (one per core, 10 here) made
  single-image latency bimodal under light background load. In back-to-back
  sessions, FP32 ran at p50 3.7 ms in one and 49 ms in the next, and INT8 s8w
  ran at p50 8 ms with p95 52 ms. With 4 threads, the same models ran at a
  steady p50 of 7.6–7.8 ms (FP32) and 2.4 ms (INT8 s8w). This experiment ran
  in the same period as run A.
- NPU latency is tight within a run (p95 0.05–0.25 ms above p50). Per-channel
  u8w and s8w differ by less than the run-to-run variation.

### 7.2 AI Hub comparison

`scripts/compare_aihub.py`: NPU only, strict, 3 rounds with the order rotated
each round (each model timed once in every position), 5 s cooldown, medians
over rounds.

| NPU, strict | Mean | p50 | p95 | Min | p50 per round |
|---|---|---|---|---|---|
| Ours INT8 per-tensor | 0.345 | 0.328 | 0.413 | 0.309 | 0.348 / 0.320 / 0.328 |
| Ours INT8 per-channel s8w | 0.367 | 0.339 | 0.411 | 0.322 | 0.339 / 0.336 / 0.356 |
| AI Hub w8a8 | 0.518 | 0.496 | 0.583 | 0.468 | 0.496 / 0.492 / 0.499 |

All values in ms. AI Hub's p50 is 46–51% higher than either of ours, in every
round. Its model card lists 0.456 ms for this asset on a Snapdragon X Elite.

**Cause: in-graph normalization.** The QNN detailed profiler
(`compare_aihub.py --qnn-profile`) puts AI Hub's accelerator execute time at
479 µs, against 298 µs for both of ours (105 vs 71 HTP ops). The `Div` takes
33.5% and the `Sub` 16.0% of its HTP cycles. These are element-wise quantized
operations on the full 3×224×224 input, and the 52 convolutions cost about the
same in both models. The kept `Clip` nodes are not the cause: in a one-off
ablation, our s8w model re-quantized with the Clips kept
(`keep_removable_activations=True`) compiled to the same 494-node QNN graph
shape and had the same p50 as our standard s8w model (0.327 vs 0.330 ms over 4
rounds). Our models normalize on the CPU instead, which takes 0.027 ms in NumPy
(one-off measurement) and is outside the timed `Run`. Folding normalization
into the first convolution would remove it from both paths.

### 7.3 Host overhead breakdown

`scripts/npu_overhead.py`: per-channel s8w INT8 on the NPU, strict, cached EP
context (the same context model for every mode), `htp_performance_mode=burst`.
3 rounds with the order rotated each round, 5 s cooldown; C++, Python, CPU and
NPU configurations are interleaved. Accelerator time per inference comes from
separate runs with the QNN profiler at `profiling_level=basic` (200 inferences
per mode). Profiling slows the host side down a great deal, so wall times
are never taken from profiled runs. Host overhead is wall-clock p50 minus
accelerator execute time.

| Mode | Wall p50 | p50 per round | Wall mean | Wall p95 | Accelerator execute | Host overhead | Share of wall p50 |
|---|---|---|---|---|---|---|---|
| C++ IoBinding, preallocated buffers | 316 µs | 318 / 316 / 316 | 340 µs | 407 µs | 273 µs | 43 µs | 14% |
| C++ `Session::Run` | 327 µs | 321 / 327 / 329 | 338 µs | 389 µs | 275 µs | 52 µs | 16% |
| Python `InferenceSession.run` | 339 µs | 353 / 339 / 334 | 379 µs | 525 µs | 277 µs | 63 µs | 19% |
| Python IoBinding, preallocated OrtValues | 351 µs | 351 / 354 / 348 | 384 µs | 437 µs | 276 µs | 75 µs | 21% |

- The accelerator time is the same in every mode (273–277 µs). The QNN
  profiler nests it inside "QNN accelerator (execute)" (289–293 µs) and
  "RPC (execute)" (298–303 µs), the call across to the DSP.
- About 80–86% of single-image latency is accelerator time. C++ IoBinding has
  the least host overhead. Plain C++ `Run` adds about 9 µs, for creating the
  input value and allocating the output. Python adds about another 10 µs.
- In Python, IoBinding is slower than plain `run` (75 vs 63 µs of overhead), so
  the binding API is not a win at this size.
- Python's mean and p95 sit further above its p50 than C++'s do.

CPU in the same run, 4 intra-op threads (ms, medians over rounds):

| Model | C++ p50 | Python p50 | C++ mean | Python mean | C++ p50 per round | Python p50 per round |
|---|---|---|---|---|---|---|
| INT8 s8w | 1.353 | 1.350 | 1.417 | 1.422 | 1.353 / 1.341 / 1.371 | 1.317 / 1.359 / 1.350 |
| FP32 | 3.958 | 3.967 | 4.207 | 4.114 | 3.926 / 3.985 / 3.958 | 3.936 / 3.970 / 3.967 |

On the CPU, C++ and Python are within 1% of each other, as expected when
compute dominates.

### 7.4 What the numbers support

- In both runs of section 7.1, INT8 on the NPU was 3.6–9.3× faster than INT8 on 4 CPU threads,
  and NPU INT8 or FP16 was 6.6–28× faster than CPU FP32 (ratios of means). The ranges are wide
  because CPU latency drifts between runs.
- NPU latencies are stable enough to compare configurations within a run.
  Without controls, CPU latencies are not.

## 8. C++ path

`tools/ort_probe` runs the models from native C++ with the NuGet ONNX Runtime
1.32.0 and the QNN plugin EP 2.8.0, using the same session settings as the
Python harness.

### DLL loading: the problem and the fix

Windows 11 ships an unrelated `onnxruntime.dll` in `C:\Windows\System32`. It
is a forwarder to the OS's own ONNX Runtime, version 1.17.1 on this machine,
and it is not a KnownDLL. The loader searches the application directory first,
so normally it finds the copy the build places next to the executable. But
when that copy is missing, the loader takes the System32 one. The ORT C++ API
initializes itself during static initialization and asks for C API version 32.
ORT 1.17 answers "The requested API version [32] is not available, only API
versions [1, 17] are supported", and the process crashes with an access
violation before `main()` runs, so no check inside the program can report it.

The fix used in `ort_probe`:

1. delay-load `onnxruntime.dll` (`/DELAYLOAD:onnxruntime.dll`) and build with
   `ORT_API_MANUAL_INIT`, so nothing touches ONNX Runtime before `main()`;
2. load `<exe dir>\onnxruntime.dll` by full path with
   `LoadLibraryExW(…, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`.
   Later delay-load resolution then finds this already-loaded module by name;
3. check that `OrtGetApiBase()->GetApi(ORT_API_VERSION)` succeeds before calling
   `Ort::InitApi(api)`, and otherwise exit with a clear message;
4. after creating the session, assert that the loaded `onnxruntime.dll` is the
   one next to the executable, and list every loaded ORT and QNN module.

| Situation | Result |
|---|---|
| Normal build output, started from the repository root, `%TEMP%`, `C:\`, or `C:\Windows\System32` as working directory (PowerShell and cmd) | Bundled `onnxruntime.dll` loaded, check OK |
| `ort_probe.exe` alone in a directory | "onnxruntime.dll not found (the build copies it next to the executable)", exit 1 |
| System32's `onnxruntime.dll` copied next to the executable | "… is ONNX Runtime 1.17.1, which does not provide C API version 32", exit 1, no crash |

With an NPU session from a context model alive, the loaded modules are, all
from the executable's directory: `onnxruntime.dll`,
`onnxruntime_providers_shared.dll`, `onnxruntime_providers_qnn.dll`,
`QnnHtp.dll`, `QnnHtpV73Stub.dll`, `QnnSystem.dll`. In Python, the ORT core is
`onnxruntime/capi/onnxruntime_pybind11_state.pyd` and no `onnxruntime.dll` is
loaded at all. With a compiling NPU session, Python additionally loads
`QnnHtpPrepare.dll`, all from the `onnxruntime_qnn` package directory.

### C++ vs Python agreement

`tools/ort_probe/make_reference.py` writes 10 preprocessed eval images (the
first of each class) and Python's outputs. The probe reads the same input
tensors, so C++ does no image decoding, and compares its outputs. CPU sessions
use 4 intra-op threads in both languages. Each Python configuration was also
run in a second, independent session.

| Configuration | Python repeat variation (same session / new session) | C++ vs Python max abs diff (`Run` / IoBinding) | Tolerance | Top-1 match |
|---|---|---|---|---|
| CPU FP32 | 0 / 0 | 0 / 0 | 1e-5 | 10 / 10 |
| CPU INT8 s8w | 0 / 0 | 0 / 0 | 1e-5 | 10 / 10 |
| NPU INT8 s8w, strict | 0 / 0 | 0 / 0 | 0 | 10 / 10 |
| NPU FP16, strict | 0 / 0 | 0 / 0 | 0 | 10 / 10 |

The outputs are bit-identical. NPU execution is deterministic run to run and
across sessions, which justifies a tolerance of 0 for the NPU.

### Strict-mode failure

Running the `Det` model above through the probe with strict mode on, session
creation throws `Ort::Exception` with `ORT_FAIL` (code 1) and the message
quoted in section 4. The probe prints the code and message, records them in
its JSON summary, and exits with status 3, without crashing. With strict mode
off, it runs and matches a CPU-only run (max abs diff 4.7e-10).

### Plugin EP API notes

- **Registration.** `Ort::Env::RegisterExecutionProviderLibrary(name, path)`
  takes an absolute wide-string path. Register once per `Env`, and release
  every session before `UnregisterExecutionProviderLibrary`.
- **Enumeration.** `Env::GetEpDevices()` returns `ConstEpDevice` views owned by
  the environment. Select by `EpName() == "QNNExecutionProvider"` and
  `Device().Type() == OrtHardwareDeviceType_NPU`, never by position.
  `Device().Vendor()` is empty for QNN devices; use `VendorId()` and
  `Metadata()`.
- **Attaching.** `SessionOptions::AppendExecutionProvider_V2(env, {device},
  options)` takes EP options as `std::unordered_map<std::string, std::string>`.
  The plugin EP never appears in `GetAvailableProviders()`.
- **Options vs config.** QNN options (`backend_path`, `htp_performance_mode`,
  `enable_htp_fp16_precision`, `profiling_level`, `profiling_file_path`) go in
  the EP options map. Strict mode (`session.disable_cpu_ep_fallback`) and EP
  context (`ep.context_enable`, `ep.context_file_path`, `ep.context_embed_mode`)
  are session config entries (`AddConfigEntry`). `backend_path` must point at
  the plugin's own `QnnHtp.dll`, which loads its stub and skel files from its
  own directory.
- **EP context.** The cache is written as a side effect of creating a session
  with `ep.context_enable=1`. Embed mode 0 (the default) writes
  `<name>_ctx.onnx` plus `<name>_ctx_qnn.bin`; delete both before
  regenerating. Load `_ctx.onnx` like any model, with the same device and
  options. `Ort::CompileModel` with `ModelCompilationOptions` is an explicit
  alternative (not tried here).
- **Errors.** A strict-mode refusal is an `Ort::Exception` with
  `GetOrtErrorCode() == ORT_FAIL`, raised at session creation, not at run time.
- **Type info lifetime.** `GetInputTypeInfo(i).GetTensorTypeAndShapeInfo()` on
  a temporary returns a dangling view, which shows up as garbage element
  types. Keep the `TypeInfo` alive while its tensor info is in use.
- **IoBinding.** Bind `Ort::Value`s over preallocated CPU buffers
  (`MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)`) once, then
  call `Session::Run(RunOptions, IoBinding)` repeatedly. This works with the QNN
  plugin EP and has the lowest overhead measured.
- **QNN profiling** is set per session through EP options, and the file is
  written when the session is released. Profiling adds roughly 12 ms of host
  time per inference.

## 9. Comparison matrix verdict

| Cell | Verdict | Reason |
|---|---|---|
| CPU FP32 | **Go** | Native from Python and C++, bit-identical across both; the accuracy reference (71.0% / 98.4%). Timing needs a fixed thread count and repeated trials (section 7.1) |
| CPU INT8 | **Go** | Per-channel s8w: 69.8% / 98.6%, 93.6% agreement with FP32. 2.8× faster than CPU FP32 at 4 threads in run B. Bit-identical across C++ and Python. Same timing controls as CPU FP32 |
| NPU INT8 | **Go** | Per-channel s8w runs 100% on the NPU in strict mode (verified three ways, from Python and C++, including cached sessions). 70.4% / 98.4%, 93.6% agreement with FP32, p50 0.32–0.37 ms. At parity with or better than Qualcomm AI Hub's model |
| NPU FP16 | **Conditional** | The FP32 model with `enable_htp_fp16_precision=1` runs 100% on the NPU and agrees with CPU FP32 on every image (max abs logit diff 0.052). Condition: the run must request FP16 explicitly and record its compute precision as FP16, not FP32. FP32 compute on the NPU without the option was not evaluated |

INT8 recipe: **per-channel int8 symmetric weights (s8w) is the default**.
Per-tensor is kept as a reference. Per-channel u8w is not carried forward,
because it performs the same as s8w and adds nothing.

## 10. Requirements for the engine

1. **Load ONNX Runtime explicitly.** Delay-load `onnxruntime.dll`, build with
   `ORT_API_MANUAL_INIT`, load the bundled DLL by full path, check
   `GetApi(ORT_API_VERSION)` before `Ort::InitApi`, and refuse to run with any
   other copy. This applies to `edgebench.exe` as well; today it has the same
   static-initialization exposure as the original probe.
2. **Record the path of every loaded ORT and QNN module** (ORT core, providers
   shared library, QNN plugin EP, QNN backend, stub and system libraries), and
   fail if any is not the bundled copy.
3. **Select the NPU by device type.** Pick the QNN EP device whose hardware type
   is NPU and require exactly one. Record the selected device (vendor ID, device
   ID, metadata) and the full enumerated device list.
4. **Strict mode on by default for NPU runs, and recorded.** A refused session
   is a failed run at the `session_creation` stage, carrying the ORT error code
   and message.
5. **Record placement evidence**: the verification method (strict session,
   QNN validation counts, ORT node placement, profile) and node counts per EP.
6. **Fixed, recorded provider options**: `backend_path` as the absolute path of
   the bundled `QnnHtp.dll`, `htp_performance_mode` (burst here), and FP16
   precision requested explicitly, with the resulting compute precision
   recorded.
7. **IoBinding with preallocated buffers as the default timed run mode**, with
   the run mode recorded. Plain `Run` remains available.
8. **An explicit CPU intra-op thread count for every timed session**, recorded.
   The ORT default is not used for timing.
9. **Report cold, context-build and cached initialization separately**, each
   from repeated session creations.
10. **Manage the EP context cache**: explicit file paths under a cache
    directory, and a cache key covering model SHA256, ORT version, QNN EP
    version, QNN SDK version, EP options and device, so stale caches are
    invalidated. Delete both `_ctx.onnx` and `_ctx_qnn.bin` before
    regenerating. Verify that cached outputs equal uncached ones, and apply
    strict mode to cached sessions.
11. **Never time a profiled session** (ORT or QNN profiling). Profiling runs
    are separate and labelled as such.
12. **Exclude preprocessing from latency, and say so in the result.** Timed runs
    use a fixed, already-preprocessed input.
13. **Run repeated trials** with interleaved or rotated configuration order and
    a cooldown, and report the spread across trials, not just a pooled
    statistic.
14. **Record system state**: power source, Windows power mode and power plan,
    battery saver, and background CPU load around the run.
15. **Compare CPU and NPU outputs with task-level metrics** (top-1 agreement
    against a reference run), not tight numeric tolerances. Same-model,
    same-EP comparisons across bindings can be exact.
16. **Model pipeline rules**: fixed input shapes, unused Q/DQ chains pruned, the
    quantization configuration recorded with each model (sidecar), and
    input normalization done outside the model or folded into the first
    convolution, not as in-graph element-wise operations. Record which one a
    model uses.
17. **Record the QNN SDK / backend version** reported by the loaded backend,
    not only the plugin version.
18. **Optional accelerator-time measurement**: a separate run with QNN
    `profiling_level=basic`, to report wall time minus accelerator time as host
    overhead.
19. **Record per-session memory**, at least the process working set or its peak,
    labelled as process memory: NPU-side memory is not visible.

## 11. Limitations

- One device (Snapdragon X Plus, Surface Laptop 7) and one model family
  (MobileNetV2, batch 1, 224×224).
- Balanced power mode only, always on AC. Best-performance mode and battery
  operation were not tested.
- No sustained-load or thermal testing: each timed run lasts well under a
  second.
- No power, energy or NPU-utilization measurement. Memory is the process
  working set only.
- The eval split overlaps the ImageNet-1k train set (488 of 500 images).
  Absolute accuracy is optimistic, and the held-out subset (134 images) is
  small.
- Latency sanity checks come from one or three rounds per script. The CPU
  thread-count experiment, the Clip ablation and the NumPy normalization
  timing were one-off measurements without retained summary files, and run A's
  summary file was overwritten.
- ORT 1.32.0 with QNN EP 2.8.0 only. No other version pairing was tested.
- FP32 compute on the NPU (without `enable_htp_fp16_precision`) was not
  evaluated.

## 12. Reproducing

The full sequence is in [setup.md](setup.md):

1. [Python environment](setup.md#python) and the environment check
   (`scripts/check_env.py`);
2. [data and models](setup.md#preparing-data-and-models): `scripts/get_data.py`,
   `scripts/export_mobilenetv2.py`, `scripts/quantize_qnn.py`,
   `scripts/eval_cpu.py`, `scripts/npu_check.py`, `scripts/get_aihub_baseline.py`
   and `scripts/compare_aihub.py` (`--qnn-profile` for the per-op profile);
3. [C++ build](setup.md#building-the-c-tools) (`scripts/fetch_deps.ps1`, CMake
   presets, `ctest`) and [ort_probe](setup.md#3-run-ort_probe), including
   `tools/ort_probe/make_reference.py` and `scripts/npu_overhead.py`;
4. the rejection check: `tools/qnn_rejection_check.py`.

The scripts write one timestamped JSON summary per run, plus a `<name>_latest.json`
copy, to `benchmarks/results/local/feasibility/` (not committed).
