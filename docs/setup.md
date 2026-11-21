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
> | Runs models, loads ONNX Runtime, or measures the machine | Refuses to run (exit code 2) | `tools/ort_probe/reference.py`, `scripts/export_mobilenetv2.py`, `scripts/quantize_qnn.py`, `scripts/eval_cpu.py` |
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
is committed: `data/`, model files (`*.onnx`) and `benchmarks/results/local/`
are gitignored.

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
that an input filled with 0.5 (the input `ort_probe` uses) gives top-1 class
677. It writes a JSON sidecar (`.json`, same name) with the export settings and
the model's SHA256.

### 3. Quantize to INT8 QDQ

```powershell
.\.venv\Scripts\python.exe scripts\quantize_qnn.py
```

Static INT8 quantization in QDQ format for the QNN HTP (NPU) backend, using
ONNX Runtime's QNN helpers (`qnn_preprocess_model`, `get_qnn_qdq_config`,
`quantize`) calibrated on the `calibration` split. Produces:

| Model | Configuration |
|---|---|
| `mobilenetv2_int8qdq_b1.onnx` | Helper defaults: uint8 activations and weights, per-tensor, MinMax calibration |
| `mobilenetv2_int8qdq_pc_b1.onnx` | Same, with per-channel weights |

Each model gets a JSON sidecar recording the source model, calibration data,
the configuration applied, a summary of the quantized graph, the ONNX Runtime
version, and the model's SHA256.

### 4. Check accuracy on the CPU

```powershell
.\.venv\Scripts\python.exe scripts\eval_cpu.py
```

Runs all three models on the `eval` split with the CPU execution provider and
prints top-1 accuracy (over all 1000 classes, and restricted to the 10
Imagenette classes) and each INT8 model's top-1 agreement with FP32. The
summary is saved to
`benchmarks/results/local/feasibility/cpu_accuracy_mobilenetv2.json`.

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
DLLs. Project code is compiled with `/W4 /WX`.

### 3. Build `ort_probe`

`tools/ort_probe` is a small diagnostic that checks ONNX Runtime and the QNN EP
from native C++. It is a separate CMake project with its own presets, so it is
configured from its own directory:

```powershell
Push-Location tools\ort_probe
cmake --preset arm64-release
cmake --build --preset arm64-release
Pop-Location   # back to the repository root
```

Output goes to `build/ort_probe/arm64-release/Release/` (under the repository
root). ONNX Runtime and all QNN runtime files
are copied next to `ort_probe.exe`.

### 4. Run `ort_probe`

From the repository root:

```powershell
# Exports tmp\mobilenetv2.onnx if missing, then runs CPU and NPU in Python
.\.venv\Scripts\python.exe tools\ort_probe\reference.py

# Same model, same input, from C++
.\build\ort_probe\arm64-release\Release\ort_probe.exe tmp\mobilenetv2.onnx
```

Both tools fill the input with `0.5` and print the top-1 index and first five
outputs for the CPU EP and the QNN EP on the NPU (with CPU fallback disabled),
so their output can be compared line by line. `ort_probe` also prints which
`onnxruntime.dll` it loaded. Windows ships its own copy in `System32`, and the
path should point at the copy next to the executable.
