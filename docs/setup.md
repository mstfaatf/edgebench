# Setup

EdgeBench targets Windows on Snapdragon (Windows 11 ARM64). Everything, Python
and C++, must run as **native ARM64** code. Results produced under x64
emulation are not representative and will be mislabeled.

## Python

### 1. Find the native ARM64 interpreter

Install Python 3.12 for ARM64 from [python.org](https://www.python.org/downloads/windows/)
(the "Windows installer (ARM64)"). Then list the installed interpreters:

```powershell
py -0p
```

```
 -V:3.13 *        C:\Users\you\AppData\Local\Programs\Python\Python313\python.exe
 -V:3.12          C:\Users\you\AppData\Local\Programs\Python\Python312\python.exe
 -V:3.12-arm64    C:\Users\you\AppData\Local\Programs\Python\Python312-arm64\python.exe
```

The ARM64 build is the one tagged `-arm64`. Confirm it with `sysconfig`:

```powershell
& 'C:\Users\you\AppData\Local\Programs\Python\Python312-arm64\python.exe' -c "import sysconfig; print(sysconfig.get_platform())"
# win-arm64
```

> **Warning: the x64 interpreter trap.** Windows on Arm runs x64 Python
> transparently under emulation, and `py` / `python` may well resolve to an
> x64 install (the `*` above marks the `py` default). Everything appears to
> work, but every library loads its x64 build and runs emulated.
>
> `platform.machine()` **cannot detect this**. It reports the host OS
> architecture, so an emulated x64 interpreter still returns `ARM64`. Use
> `sysconfig.get_platform()`, which reports what the interpreter was built for:
> `win-arm64` (native) or `win-amd64` (emulated x64).
>
> EdgeBench checks this with `scripts/common/platform_guard.py`. The check is
> scoped by what a script does:
>
> | Kind of script | Behaviour on a non-native interpreter | Examples |
> |---|---|---|
> | Runs models, loads ONNX Runtime, or measures the machine | Refuses to run (exit code 2) | `tools/ort_probe/reference.py`, `scripts/export_mobilenetv2.py`, `scripts/quantize_qnn.py`, `scripts/eval_cpu.py`, `scripts/npu_check.py`, `scripts/compare_aihub.py`, `scripts/npu_overhead.py`, `tools/qnn_rejection_check.py`, `tools/ort_probe/make_reference.py` |
> | Environment diagnostic | Runs and reports the problem prominently | `scripts/check_env.py` |
> | Platform-independent tooling | Not checked; runs on any OS/architecture | `scripts/validate_results.py` and its tests |
>
> Results from emulated or non-Snapdragon machines would be mislabeled, so only
> the first kind is blocked. Tooling that just reads or checks files must keep
> working on Linux/x64 CI runners and on contributors' other machines.

### 2. Create the virtual environment

Create the venv with the interpreter's **full path**, not `py` or `python`:

```powershell
& 'C:\Users\you\AppData\Local\Programs\Python\Python312-arm64\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

A venv inherits the architecture of the interpreter that created it, so
everything installed into `.venv` is native from here on.

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
```

Notes:

- `requirements.txt` pins exact versions. It adds the PyTorch package index
  (`download.pytorch.org/whl/cpu`) because torch and torchvision publish
  win-arm64 wheels only there, not on PyPI.
- `--only-binary=:all:` stops pip from trying to build anything from source.
- ONNX Runtime and the Qualcomm QNN execution provider share one environment:
  `onnxruntime-qnn` 2.x is a plugin that installs alongside the stock
  `onnxruntime` package.
- TensorFlow and tf2onnx are not included. They have no native win-arm64
  wheels, so Keras models are trained and converted to ONNX on another machine.

### 4. Check the environment

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe scripts\check_env.py --json tmp\check_env.json   # also save results
```

Every check should report `PASS`, except `tensorflow` and `tf2onnx`, which
report `NOT INSTALLED` (optional). The QNN check confirms that a model runs on
the NPU. See [feasibility.md](feasibility.md) for reference results.

### 5. Run the Python tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests live in `scripts/tests/` (configured in `pyproject.toml`). To validate
benchmark result files against the [results schema](results-schema.md):

```powershell
.\.venv\Scripts\python.exe scripts\validate_results.py benchmarks\results\published
```

## Preparing data and models

These steps produce the evaluation data and the MobileNetV2 models used by the
benchmarks. Run them from the repository root, in order. Nothing they produce
is committed: `data/`, `models/` (except its README), `cache/` and
`benchmarks/results/local/` are gitignored.

### 1. Download Imagenette and build the splits

```powershell
.\.venv\Scripts\python.exe scripts\get_data.py
```

Downloads [Imagenette](https://github.com/fastai/imagenette) (`imagenette2-320.tgz`,
~340 MB, 10 ImageNet classes) into `data/downloads/`, verifies its SHA256
(pinned in the script; checked again on every run), and extracts it to
`data/imagenette/imagenette2-320/`. It then writes two seeded, deterministic
image lists to `data/imagenette/splits/`:

| Split | Source | Size | Used for |
|---|---|---|---|
| `calibration.txt` | Imagenette train | 200 (20 per class) | Static quantization calibration |
| `eval.txt` | Imagenette val | 500 (50 per class) | Accuracy measurement |

The mapping from Imagenette WordNet IDs to ImageNet-1k class indices is in
`scripts/common/imagenette_classes.json`.

> Most Imagenette val images are drawn from the ImageNet-1k *train* set, which
> ImageNet-pretrained weights were trained on. Absolute accuracy on `eval` is
> therefore optimistic; it is meant for comparing variants of the same model
> (FP32 vs INT8, CPU vs NPU), not as an ImageNet accuracy figure.

All scripts that feed images to a model use the same preprocessing,
`scripts/common/preprocess.py` (resize shorter side to 256, center crop 224,
scale to [0, 1], ImageNet mean/std normalization, NCHW float32).

### 2. Export the FP32 model

```powershell
.\.venv\Scripts\python.exe scripts\export_mobilenetv2.py
```

Exports torchvision MobileNetV2 (`IMAGENET1K_V2` weights) to
`models/mobilenetv2/mobilenetv2_fp32_b1.onnx`: opset 17, fixed input
`input` [1, 3, 224, 224], output `logits` [1, 1000]. The script checks the
model with `onnx.checker`, compares it with PyTorch on an eval image, and checks
that an input filled with 0.5 (the input of `tools/ort_probe/reference.py` and
`ort_probe --fill 0.5`) gives top-1 class
677. It writes a JSON sidecar (`.json`, same name) with the export settings and
the model's SHA256.

### 3. Quantize to INT8 QDQ

```powershell
.\.venv\Scripts\python.exe scripts\quantize_qnn.py
```

Static INT8 quantization in QDQ format for the QNN HTP (NPU) backend, using
ONNX Runtime's QNN helpers (`qnn_preprocess_model`, `get_qnn_qdq_config`,
`quantize`) calibrated on the `calibration` split. All variants use uint8
activations and MinMax calibration:

| Model | Weights |
|---|---|
| `mobilenetv2_int8qdq_b1.onnx` | Helper defaults: uint8, per-tensor |
| `mobilenetv2_int8qdq_pc_b1.onnx` | uint8, per-channel |
| `mobilenetv2_int8qdq_pc_s8w_b1.onnx` | int8 symmetric, per-channel |

The quantizer leaves unused QuantizeLinear/DequantizeLinear chains on the
constants of the `Clip` (ReLU6) nodes it folds away. The script prunes nodes
whose outputs nothing consumes, and initializers nothing reads, until nothing
changes. It then checks that the pruned model gives bit-identical CPU outputs
to the unpruned one on the `eval` split. The unpruned models are kept in
`models/mobilenetv2/unpruned/`.

Each model gets a JSON sidecar recording the source model, calibration data,
the configuration applied, the pruning result, a summary of the quantized
graph, the ONNX Runtime version, and the model's SHA256.

### 4. Check accuracy on the CPU

```powershell
.\.venv\Scripts\python.exe scripts\eval_cpu.py
```

Runs the FP32 model and the three INT8 models on the `eval` split with the CPU
execution provider and prints top-1 accuracy (over all 1000 classes, and
restricted to the 10 Imagenette classes) and each INT8 model's top-1 agreement
with FP32. Each run saves a new summary,
`benchmarks/results/local/feasibility/cpu_accuracy_mobilenetv2_<timestamp>.json`
(UTC, `YYYYmmdd-HHMMSS`), and copies it to `cpu_accuracy_mobilenetv2_latest.json`.
Earlier summaries are never overwritten; the measurement scripts below follow the
same scheme.

### 5. Check the models on the NPU

```powershell
.\.venv\Scripts\python.exe scripts\npu_check.py
```

Runs the INT8 models, and the FP32 model with FP16 precision on HTP, on the
NPU through the QNN execution provider with CPU fallback disabled. It checks
that every node is placed on and executed by the QNN EP, compares accuracy
with the CPU, and measures session creation (with and without an EP context
cache) and single-image latency as a sanity check. Run it plugged in. Takes a
few minutes. The summary goes to
`benchmarks/results/local/feasibility/npu_check_mobilenetv2_<timestamp>.json`
(and `npu_check_mobilenetv2_latest.json`), with session logs and profiles in the
matching `npu_check_mobilenetv2_<timestamp>/` directory. EP context caches are written
to `cache/qnn/` (not committed; they are specific to the device and the QNN SDK
version). Results are discussed in [feasibility.md](feasibility.md).

CPU timing sessions use 4 intra-op threads (`--cpu-threads`); always record
the thread count with CPU numbers. `tools\qnn_rejection_check.py` checks that
the placement check detects nodes the QNN EP rejects (it runs a tiny model with
an unsupported `Det` node with and without strict mode).

### 6. Compare with Qualcomm AI Hub's model

```powershell
.\.venv\Scripts\python.exe scripts\get_aihub_baseline.py
.\.venv\Scripts\python.exe scripts\compare_aihub.py              # add --qnn-profile for per-op HTP cycles
```

`get_aihub_baseline.py` downloads Qualcomm AI Hub's pre-quantized (w8a8) and
float MobileNetV2 ONNX models (AI Hub Models release 0.67.0, from the
[`qualcomm/MobileNet-v2`](https://huggingface.co/qualcomm/MobileNet-v2) model
card's release assets; no account needed). They are pinned by URL, S3 object
version and SHA256, and extracted to `models/aihub/`. Each gets a JSON sidecar
with its source, license and I/O. `compare_aihub.py` runs the AI Hub model
through the same strict NPU checks as ours, next to our per-tensor and
per-channel s8w INT8 models. The AI Hub model takes a uint8 image and
normalizes internally, so it receives the shared preprocessing's [0, 1] tensor
quantized to uint8, and its uint8 output is dequantized. Timing runs 3 rounds
with the model order rotated. The summary goes to
`benchmarks/results/local/feasibility/aihub_comparison_mobilenetv2_<timestamp>.json`
(and `aihub_comparison_mobilenetv2_latest.json`), with logs in the matching
`aihub_comparison_mobilenetv2_<timestamp>/` directory.

## Building the C++ tools

### Prerequisites

- **Visual Studio 2022** (Community or Build Tools) with:
  - workload *Desktop development with C++*
  - component *MSVC v143 - VS 2022 C++ ARM64/ARM64EC build tools (Latest)*
  - a *Windows 11 SDK* (10.0.22621 or newer)
  - component *C++ CMake tools for Windows* (provides CMake and Ninja)

  The ARM64 build tools component provides the ARM64-hosted compiler
  (`VC\Tools\MSVC\<ver>\bin\Hostarm64\arm64\cl.exe`). Without it, the
  `arm64-release` preset fails to configure.
- CMake 3.25 or newer. The copy bundled with Visual Studio (the *C++ CMake
  tools for Windows* component) is enough.

### 0. Open a shell with CMake on `PATH`

The bundled `cmake` and `ctest` are **not** on `PATH` in a regular PowerShell.
Use a *Developer PowerShell for VS 2022* (Start menu), or load it into an
existing PowerShell session:

```powershell
# Adjust "Community" to your edition (e.g. BuildTools, Professional)
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch arm64 -SkipAutomaticLocation
cmake --version   # should print 3.25 or newer
```

The script may print `'vswhere.exe' is not recognized`; this is harmless.
`-SkipAutomaticLocation` keeps the current directory. A standalone CMake
(ARM64 build from cmake.org) on `PATH` works too. All commands below are run
from the repository root in this shell.

### 1. Fetch native dependencies

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch_deps.ps1
```

This downloads pinned, SHA256-verified archives into `third_party/` (which is
not committed):

| Package | Source | Extracted to |
|---|---|---|
| ONNX Runtime 1.32.0 (headers, win-arm64 `onnxruntime.dll`/`.lib`) | NuGet `Microsoft.ML.OnnxRuntime` | `third_party/onnxruntime/` |
| QNN execution provider 2.8.0 (plugin EP + QNN SDK 2.52.0 backends) | GitHub release `onnxruntime/onnxruntime-qnn`, `onnxruntime-qnn-2.8.0-win-arm64.zip` | `third_party/onnxruntime-qnn/` |

[`github.com/onnxruntime/onnxruntime-qnn`](https://github.com/onnxruntime/onnxruntime-qnn)
is maintained by Qualcomm and is the official source of the QNN plugin
execution provider and the QNN backend DLLs (`QnnHtp.dll` and friends). The
Python package `onnxruntime-qnn` ships the same files.

The script is safe to re-run. Use `-Force` to re-extract. CMake finds these
packages through `cmake/FindOnnxRuntime.cmake`; to use a different location,
set `OnnxRuntime_ROOT` / `QnnEp_ROOT` when configuring.

### 2. Build the engine and run the tests

```powershell
cmake --preset arm64-release          # or arm64-debug
cmake --build --preset arm64-release
ctest --preset arm64-release
.\build\arm64-release\bin\Release\edgebench.exe --version
```

The presets use the Visual Studio 2022 generator with the ARM64-hosted toolset,
so no `vcvarsall`/ARM64 compiler environment is needed, only `cmake` on `PATH`.
Configuration fails with a clear message if the compiler does not target
native ARM64. GoogleTest is downloaded at configure
time (pinned by SHA256). Executables are placed in
`build/<preset>/bin/<Config>/`, together with the ONNX Runtime and QNN runtime
DLLs. Project code (engine and tools) is compiled with `/W4 /WX`.

### 3. Run `ort_probe`

`tools/ort_probe` is a diagnostic that runs a model with ONNX Runtime from
native C++, on the CPU EP or on the NPU through the QNN plugin EP. It is built by
the root project (option `EDGEBENCH_BUILD_TOOLS`, on by default) into
`build/<preset>/bin/<Config>/ort_probe.exe`, next to the ONNX Runtime and QNN
runtime files. Its unit tests (`ort_probe_tests`) run with `ctest`. See
`ort_probe.exe --help` for all options. Typical runs, from the repository root:

```powershell
$probe = '.\build\arm64-release\bin\Release\ort_probe.exe'

# Per-channel s8w INT8 on the NPU, strict mode, with an EP context cache
& $probe --model models\mobilenetv2\mobilenetv2_int8qdq_pc_s8w_b1.onnx --target npu `
         --ep-context on --context-file cache\qnn\cpp_s8w_ctx.onnx --init-reps 3 --runs 200

# CPU EP; the intra-op thread count is always explicit (default 4) and printed
& $probe --model models\mobilenetv2\mobilenetv2_fp32_b1.onnx --target cpu --threads 4
```

To compare C++ with Python on identical inputs, write reference tensors first
(10 preprocessed eval images and Python outputs for `cpu_fp32`, `cpu_s8w`,
`npu_s8w`, `npu_fp16`). Each run writes a new
`data/ort_probe_reference/<timestamp>/` and refreshes
`data/ort_probe_reference/latest/` (not committed). Then point the probe at it:

```powershell
.\.venv\Scripts\python.exe tools\ort_probe\make_reference.py
& $probe --model models\mobilenetv2\mobilenetv2_int8qdq_pc_s8w_b1.onnx --target npu `
         --reference-dir data\ort_probe_reference\latest --reference-config npu_s8w
```

The probe exits with 0 if everything passed, 1 if a check failed (comparison,
DLL check, device selection), 2 on a usage error, and 3 on an ONNX Runtime error
(the error code and message are printed). It loads `onnxruntime.dll` from its own
directory by full path before any ONNX Runtime call and refuses to run with any
other copy. Windows ships an unrelated, older `onnxruntime.dll` in `System32`.

`scripts\npu_overhead.py` uses the probe to compare C++ and Python NPU latency
and to split it into accelerator time and host-side overhead. It reads the
reference inputs from `data/ort_probe_reference/latest/` and writes
`npu_overhead_mobilenetv2_<timestamp>.json` (and `_latest.json`) plus a matching
run directory with the probe's JSON output to `benchmarks/results/local/feasibility/`.

`tools\ort_probe\reference.py` is a quick Python check with an input filled
with 0.5; `ort_probe --fill 0.5` runs the same input from C++.
