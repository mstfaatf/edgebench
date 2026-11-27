# Setup

EdgeBench targets Windows on Snapdragon (Windows 11 ARM64). Everything that
measures, Python and C++, must run as **native ARM64** code. Results produced
under x64 emulation are not representative and will be mislabeled. (A CPU-only
x64 build of the engine exists for continuous integration; see
[x64 and Ninja builds](#x64-and-ninja-builds-cpu-only).)

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
> | Runs models, loads ONNX Runtime, or measures the machine | Refuses to run (exit code 2) | `scripts/export_models.py`, `scripts/export_mobilenetv2.py`, `scripts/fold_models.py`, `scripts/quantize_models.py`, `scripts/make_references.py` (except `--check`), `scripts/eval_accuracy.py`, `scripts/prepare_autoencoder.py`, `scripts/eval_anomaly.py`, `scripts/quantize_qnn.py`, `scripts/eval_cpu.py`, `scripts/npu_check.py`, `scripts/compare_aihub.py`, `scripts/npu_overhead.py`, `scripts/check_autoencoder_npu.py`, `scripts/make_probe_references.py`, `scripts/investigate_batch_equivalence.py`, `scripts/run_matrix.py`, `tools/qnn_rejection_check.py`, `tools/ort_probe/reference.py`, `tools/ort_probe/make_reference.py` |
> | Environment diagnostic | Runs and reports the problem prominently | `scripts/check_env.py` |
> | Downloads and platform-independent tooling | Not checked; runs on any OS/architecture | `scripts/get_data.py`, `scripts/get_aihub_baseline.py`, `scripts/validate_results.py`, `scripts/make_test_models.py`, `scripts/check_hygiene.py`, `scripts/coverage_summary.py`, `scripts/validate_accuracy.py`, `scripts/check_doc_numbers.py`, `scripts/check_matrix.py`, `scripts/model_recipes.py`, `scripts/model_info.py` (they read model files with onnx but never run them), and the tests in `scripts/tests/` |
> | Shared modules in `scripts/common/` | Not checked themselves. `npu_harness.py` and `ort_qnn.py` load ONNX Runtime and are imported only by guarded scripts, after the check; `onnx_export.py` and `onnx_quantize.py` import torch, onnx and ONNX Runtime inside their functions and are used by `export_models.py`, `fold_models.py`, `quantize_models.py` and their tests; `onnx_fold.py` needs onnx and NumPy only; `model_recipe.py` imports onnx inside the functions that read model files; `reference_outputs.py` imports NumPy only to generate the SplitMix64 input; `timeseries.py` is standard library only except for the two functions that build NumPy arrays; `autoencoder.py`, `autoencoder_pipeline.py` and `onnx_cleanup.py` import onnx and NumPy (the pipeline also ONNX Runtime) inside their functions, so the source-bundle checks of `autoencoder_pipeline.py` run anywhere; `anomaly_stats.py` is standard library only; `imagenet_eval.py` needs NumPy and Pillow; `accuracy_stats.py` and `matrix.py` are standard library only; the others (`preprocess.py`, `manifests.py`, `model_registry.py`, `accuracy.py`, `qnn_log.py`, `onnx_prune.py`, `qdq_inspect.py`, `results_io.py`, `cli.py`) do not | – |
>
> The C++ build accepts ARM64 (the measurement build) and x64 (CPU only, for
> continuous integration; it reports emulation at run time), and rejects
> anything else at configure time (`cmake/TargetArchitecture.cmake`).
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

`benchmarks/results/published/` holds no benchmark result files yet; the
accuracy and anomaly-detection results in `published/accuracy/` have their own
schemas, so this lists them as skipped (exit code 0); they are checked with
`scripts\validate_accuracy.py`. `docs\examples\v1.0.0`, `docs\examples\v1.1.0`
and `docs\examples\v1.2.0` hold example results that validate. The validator picks
the schema from each file's `schema_version`, reports errors and warnings
separately, and with `--strict` also fails on warnings. Its tests need only
`jsonschema` and `pytest` and run on any platform:

```powershell
.\.venv\Scripts\python.exe -m pytest scripts\tests\test_validate_results.py
```

CTest runs the same tests as `results_schema_tests` (label `schema`).

## Preparing data and models

These steps produce the evaluation data, every model file listed in
`models/manifest.json` (the three image classifiers in FP32, normalization-folded
FP32 and INT8, and the Conv1D autoencoder), their reference outputs and the
published accuracy results. Run them from the repository root, in order: each
step needs the files of the steps before it. Nothing they produce is committed:
`data/`, `models/` (except its README, `manifest.json` and `sources/`), `cache/`
and `benchmarks/results/local/` are gitignored.

### 1. Download the datasets

```powershell
.\.venv\Scripts\python.exe scripts\get_data.py --dataset all
```

`--dataset imagenette` (the default), `--dataset imagenetv2` and `--dataset nab`
fetch one of the three (`nab` is the time series for the autoencoder; see
[autoencoder.md](autoencoder.md)). Archives go to `data/downloads/` and are pinned by SHA256 in the
script. The images are never committed; what is committed is a manifest per
split in [`datasets/`](../datasets/README.md): every image's relative path,
ImageNet-1k class index and SHA256, so anyone can confirm they have exactly the
same images.

| Split | Manifest | Source | Size | Used for |
|---|---|---|---|---|
| eval-full | `datasets/imagenetv2-matched-frequency/eval-full.tsv` | ImageNetV2 matched-frequency | 10,000 (10 per class) | Published accuracy |
| eval-quick | `datasets/imagenetv2-matched-frequency/eval-quick.tsv` | Subset of eval-full | 1,000 (1 per class, seed 20261123) | Fast iteration |
| calibration | `datasets/imagenette/calibration.tsv` | Imagenette train | 200 (20 per class, seed 20261120) | Static quantization calibration |

**ImageNetV2.** [ImageNetV2](https://github.com/modestyachts/ImageNetV2) is a
test set for ImageNet-1k classifiers, built by its authors after the original
ImageNet with the same collection process, so it shares no images with the
ImageNet training set. The matched-frequency variant has 10 images for each of
the 1,000 classes. It is the accuracy set for published results.
`get_data.py --dataset imagenetv2` downloads
`imagenetv2-matched-frequency.tar.gz` (1.26 GB; despite the name, an
uncompressed tar) from the dataset's Hugging Face repository
([vaishaal/ImageNetV2](https://huggingface.co/datasets/vaishaal/ImageNetV2), the
location linked from the ImageNetV2 README) at a pinned revision, checks the
SHA256 that Hugging Face publishes for it, extracts it to `data/imagenetv2/`
and hashes every image against `eval-full.tsv`. Later runs skip the download and
only compare the file list with the manifest; `--verify` hashes the archive (if
still present) and all images again, `--force-extract` extracts again. The
archive can be deleted after extraction.

ImageNetV2 names its class directories `0` to `999` after the ImageNet-1k class
index, in the order torchvision's ImageNet weights use (WordNet IDs sorted).
Read them as integers: a generic image-folder loader sorts them as strings
(`0, 1, 10, 100, ...`) and mislabels 988 of the 1,000 classes. The class list
(index, WordNet ID, torchvision name) is committed as
`scripts/common/imagenet_classes.json`; the tests check it against the
torchvision weights metadata and the Imagenette mapping.

eval-quick takes one image per class from eval-full with Python's
`random.Random(20261123)`, visiting classes in index order and each class's
images in path order (`scripts/common/manifests.py`, `stratified_sample`).

**Imagenette.** [Imagenette](https://github.com/fastai/imagenette)
(`imagenette2-320.tgz`, ~340 MB, 10 ImageNet classes) is extracted to
`data/imagenette/imagenette2-320/` (its archive is verified on every run). The
script writes two seeded image lists to `data/imagenette/splits/`:
`calibration.txt` (the calibration split above) and `eval.txt` (500 val images,
50 per class), and checks the calibration images against their manifest. The
mapping from Imagenette WordNet IDs to ImageNet-1k class indices is in
`scripts/common/imagenette_classes.json`.

> Most Imagenette val images are drawn from the ImageNet-1k *train* set, which
> ImageNet-pretrained weights were trained on, so accuracy on `eval.txt` is
> optimistic. It was used for the variant comparisons (FP32 vs INT8, CPU vs
> NPU) in [feasibility.md](feasibility.md) and is kept for those scripts;
> published accuracy uses ImageNetV2.

Calibration and evaluation images must not overlap. The calibration images come
from a different dataset, and the check compares file contents (SHA256), not
names: `get_data.py --dataset imagenetv2` and the manifest tests fail on any
image shared by `calibration.tsv` and `eval-full.tsv`. That comparison only
catches byte-identical files; a re-encoded copy of the same photo would have a
different hash.

All scripts that feed images to a model use the same preprocessing,
`scripts/common/preprocess.py` (resize shorter side to 256, center crop 224,
scale to [0, 1], ImageNet mean/std normalization, NCHW float32).
Normalization is a separate step (`normalize()`, or `load_image(...,
normalized=False)` to skip it), for models that take [0, 1] input or have the
normalization folded into their first convolution. The tests check the output
against torchvision's transforms (synthetic images, and ImageNetV2 images when
downloaded) and pin its SHA256 for a fixed input.

### 2. Export the FP32 models

```powershell
.\.venv\Scripts\python.exe scripts\export_models.py
```

Exports MobileNetV2 (torchvision `IMAGENET1K_V2`), ResNet-18 (torchvision
`IMAGENET1K_V1`) and EfficientNet-Lite0 (timm `efficientnet_lite0.ra_in1k`) to
fixed-shape FP32 ONNX files, one per batch size (1, 4, 8):
`models/<model>/<model>_fp32_b<batch>.onnx`, opset 17, input `input`
[batch, 3, 224, 224] (already normalized with the model's own mean/std),
output `logits` [batch, 1000]. The
weights are checked against pinned SHA256s; every file is checked with
`onnx.checker` and shape inference, compared with PyTorch on ImageNetV2 images
(from step 1), and listed in the committed `models/manifest.json`. Each file
gets a recipe sidecar (`<name>.recipe.json`) built from its metadata;
`scripts\model_info.py <file.onnx>` prints it with the file's lineage, and
`scripts\model_recipes.py --check` checks sidecars, manifest and lineage
([models.md](models.md#recipes-and-the-manifest)).
`--verify` re-exports and compares with the manifest. Details, tolerances and
the ONNX metadata are in [models.md](models.md).
`scripts\export_mobilenetv2.py` exports only `mobilenetv2_fp32_b1.onnx`, the
input of the next step.

### 3. Fold the normalization into the first convolution

```powershell
.\.venv\Scripts\python.exe scripts\fold_models.py
```

Writes the normalization-folded FP32 variants from the files of step 2:
`models/<model>/<model>_fp32_folded_b<batch>.onnx`, with the mean/std folded
into the first convolution, so they take the [0, 1] image tensor. `--verify`
folds again and compares. See [models.md](models.md#normalization-folded-models).

### 4. Quantize to INT8 QDQ

```powershell
.\.venv\Scripts\python.exe scripts\quantize_models.py
```

Quantizes every FP32 file from steps 2 and 3 to static INT8 QDQ with the default
recipe (per-channel symmetric int8 weights, uint8 activations, MinMax
calibration on the 200 calibration images): `models/<model>/<model>_int8_b<batch>.onnx`
and `<model>_int8_folded_b<batch>.onnx`, pruned, plus the unpruned references
`<model>_int8_unpruned_b<batch>.onnx` and `<model>_int8_folded_unpruned_b<batch>.onnx`,
all listed in `models/manifest.json`. It needs the files of both step 2 and
step 3 (`--normalization external` or `folded` quantizes one kind only).
`--verify` re-quantizes and compares. Recipe, metadata and checks are in
[models.md](models.md#int8-qdq-models).

### 5. Prepare the autoencoder

The Conv1D time-series autoencoder is not exported here: its FP32 files come
from a Colab run and are committed in `models/sources/conv1d_autoencoder/`. Its
model files (the imported graph, the cleaned FP32 graph and the INT8 models,
with sidecars and manifest entries) are made from that source and the NAB data
of step 1:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_autoencoder.py            # --verify rebuilds and compares
.\.venv\Scripts\python.exe scripts\check_autoencoder_npu.py --models # strict NPU check, no timing
```

See [autoencoder.md](autoencoder.md).

### 6. Check the manifest and make the reference outputs

```powershell
.\.venv\Scripts\python.exe scripts\model_recipes.py --check
.\.venv\Scripts\python.exe scripts\make_references.py
```

`model_recipes.py --check` checks every model file, its sidecars and manifest
entry, and the lineage ([models.md](models.md#recipes-and-the-manifest)).
`make_references.py` writes the reference outputs of all 66 files (CPU, and NPU
for the INT8 files) to `models/references/`, used by the C++ `ReferenceModels`
tests (label `models`), and the committed batch-equivalence summary
`models/batch_equivalence.json` (`--batch-equivalence` rewrites only the
summary); `make_references.py --check` checks both against the model files.
See [models.md](models.md#reference-outputs).

### 7. Measure accuracy

Accuracy of every image-model variant on ImageNetV2 eval-full, CPU and NPU
(about 15 minutes; the first run also caches the resized images in
`data/cache/`), and the autoencoder's anomaly detection on the NAB test split
(under a minute):

```powershell
.\.venv\Scripts\python.exe scripts\eval_accuracy.py
.\.venv\Scripts\python.exe scripts\eval_anomaly.py
.\.venv\Scripts\python.exe scripts\validate_accuracy.py benchmarks\results\published\accuracy
.\.venv\Scripts\python.exe scripts\check_doc_numbers.py
```

Both evaluations write to `benchmarks/results/published/accuracy/`; to compare a
rerun with the committed results instead, pass `--output-dir` (and
`--predictions-dir` / `--scores-dir`). Metrics and the SHA256 of the prediction
and score files are reproducible. `check_doc_numbers.py` (standard library only,
also run in CI) checks that the numbers quoted in README.md, accuracy.md,
autoencoder.md, models.md and npu.md match the committed results, the manifest,
the split manifest and `docs/findings/`. See [accuracy.md](accuracy.md).

### 8. Feasibility-study models (optional)

The feasibility study's MobileNetV2 variants (used by `eval_cpu.py`,
`npu_check.py`, `compare_aihub.py`, `tools/ort_probe/make_reference.py`, and six
C++ tests with the `models` label that skip without them) come from an older
script:

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

#### Check accuracy on the CPU

```powershell
.\.venv\Scripts\python.exe scripts\eval_cpu.py
```

Runs the FP32 model and the three INT8 models on the `eval` split with the CPU
execution provider and prints top-1 accuracy (over all 1000 classes, and
restricted to the 10 Imagenette classes) and each INT8 model's top-1 agreement
with FP32. Each run saves a new summary,
`benchmarks/results/local/feasibility/cpu_accuracy_mobilenetv2_<timestamp>.json`
(UTC with milliseconds, `YYYYmmddTHHMMSS.mmmZ`), and copies it to `cpu_accuracy_mobilenetv2_latest.json`.
Earlier summaries are never overwritten; the measurement scripts below follow the
same scheme.

#### Check the models on the NPU

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
the thread count with CPU numbers.

To check that the placement check detects nodes the QNN EP rejects, run a tiny
model with an unsupported `Det` node with and without strict mode (all 12
checks should pass):

```powershell
.\.venv\Scripts\python.exe tools\qnn_rejection_check.py
```

#### Compare with Qualcomm AI Hub's model

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
| With `-Arch x64`: ONNX Runtime 1.32.0 (headers, win-x64 DLL/`.lib`), no QNN | the same NuGet package and SHA256 | `third_party/onnxruntime-x64/` |

[`github.com/onnxruntime/onnxruntime-qnn`](https://github.com/onnxruntime/onnxruntime-qnn)
is maintained by Qualcomm and is the official source of the QNN plugin
execution provider and the QNN backend DLLs (`QnnHtp.dll` and friends). The
Python package `onnxruntime-qnn` ships the same files.

The versions, archive URLs and SHA256 pins, of the archives and of every DLL
`edgebench doctor` checks, live in [`cmake/bundle-pins.json`](../cmake/bundle-pins.json):
the script reads it and checks the pinned files after every run, and CMake reads
it at configure time (`cmake/BundlePins.cmake`: the ONNX Runtime version of the
build, and a generated header with the file pins that doctor compares the bundle
with). One committed file for both, so a pin cannot be updated in one place only.

The script is safe to re-run. Use `-Force` to re-extract. CMake finds these
packages through `cmake/FindOnnxRuntime.cmake`; to use a different location,
set `OnnxRuntime_ROOT` / `QnnEp_ROOT` when configuring.

### 2. Build the engine and run the tests

```powershell
cmake --preset arm64-release          # or arm64-debug
cmake --build --preset arm64-release
ctest --preset arm64-release --output-on-failure
.\build\arm64-release\bin\Release\edgebench.exe doctor
```

`edgebench doctor` checks everything a run depends on: the bundle against its
pins, the backend versions, the devices, a strict probe session on the NPU, an
EP context, the context cache directory, power and background load (see
[cli.md](cli.md#edgebench-doctor) and
[How the executables find ONNX Runtime](#how-the-executables-find-onnx-runtime));
[cli.md](cli.md) describes every command and exit code. Its probe models and
their expected NPU outputs are in `engine/resources/doctor/` (made by
`scripts/make_test_models.py` and `scripts/make_probe_references.py`) and built
into the executable.

#### Test labels

Every test carries a CTest label saying what it needs, so each environment can
select the tests it can run:

| Label | Needs | Tests |
|---|---|---|
| `unit` | nothing beyond the build: no ONNX Runtime DLL | `edgebench_tests`, `ort_probe_tests` |
| `ort` | the bundled ONNX Runtime for the build's architecture (x64 or ARM64) and the committed fixture models in `engine/tests/fixtures/models/` | `edgebench_bundle_tests` (loader, `doctor`, `run`, `ort_probe.exe`, exit paths, missing/wrong/damaged `onnxruntime.dll`, registering the QNN EP library without using a device) |
| `models` | additionally the exported models (`models/`) or reference data (`data/ort_probe_reference/`); see [Preparing data and models](#preparing-data-and-models) | `edgebench_bundle_tests` suites ending in `Models`; they skip when the files are missing |
| `npu` | the QNN execution provider, its backend and an NPU (ARM64 Snapdragon only) | `edgebench_bundle_tests` suites ending in `Npu` (`QnnEpNpu`, `DoctorNpu`, `QnnProviderNpu`, `RunNpu`, `OrtProbeNpu`, `EpCacheNpu` use the committed fixture models; `ReferenceModelsNpu`, `ReferenceDeterminismNpu` and `ReferenceCachedNpu` also need the prepared models and `models/references/`, and skip without them; the `AllFilesManual` and `ImportedFilesManual` instances of `ReferenceCachedNpu`, every NPU reference checked from an EP context, skip unless `EDGEBENCH_CACHED_REFERENCES_ALL=1` is set). They skip, with the reason, in x64 builds and on machines without any Qualcomm hardware; on a Snapdragon machine whose NPU cannot be selected they fail |
| `schema` | the Python environment (`EDGEBENCH_PYTHON`, default `.venv`) | `results_schema_tests` (the validator suite through pytest) |

In the ONNX Runtime test program the label follows the test suite name (see
`edgebench_discover_ort_tests()` in `engine/CMakeLists.txt`): suites ending in
`Models` get `models`, suites ending in `Npu` get `npu`, all others `ort`. Name
new suites accordingly. `npu` tests also share the ctest resource lock `npu`,
so they run one at a time even with `-j` (the others still run in parallel),
and have a 300 s timeout (about 8x the slowest one), so a hang is reported as
a timeout. A test process that crashes prints the exception code on stderr,
which `--output-on-failure` shows next to ctest's own `***Exception` line.
Commands per environment:

| Environment | Command |
|---|---|
| Snapdragon device, everything prepared | `ctest --preset arm64-release --output-on-failure` |
| ARM64 without models or NPU | `ctest --preset arm64-release -L "unit\|ort\|schema" --output-on-failure` |
| x64 build (CI, or an x64 machine) | `ctest --preset x64-release -L "unit\|ort" --output-on-failure` |
| No ONNX Runtime at all | `ctest --preset <preset> -L unit --output-on-failure` |

Tests that need a model or reference file skip (not fail) when it is missing;
tests that send console control events skip without a console.

The presets use the Visual Studio 2022 generator with the ARM64-hosted toolset,
so no `vcvarsall`/ARM64 compiler environment is needed, only `cmake` on `PATH`.
Configuration fails with a clear message if the compiler targets neither ARM64
nor x64 (`cmake/TargetArchitecture.cmake`). Third-party C++ libraries are
downloaded at configure time into `build/_downloads/` (shared by all build
directories, `EDGEBENCH_DOWNLOAD_DIR`), pinned by release tag and SHA256:

| Library | Version | Used for |
|---|---|---|
| [CLI11](https://github.com/CLIUtils/CLI11) | 2.7.2 | Command-line parsing (`edgebench`) |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.12.0 | JSON output |
| [GoogleTest](https://github.com/google/googletest) | 1.18.0 | Unit and process tests (only with `EDGEBENCH_BUILD_TESTS`) |

Their headers are treated as external, so `/W4 /WX` applies to project code only.
Executables are placed in
`build/<preset>/bin/<Config>/`, together with the ONNX Runtime and QNN runtime
DLLs. Project code (engine and tools) is compiled with `/W4 /WX`. If the
repository is checked out under `%TEMP%`, MSBuild prints warning MSB8029 about
intermediate directories in a temporary folder; it is harmless, but a checkout
elsewhere avoids it.

### x64 and Ninja builds (CPU only)

Measurements come from the native ARM64 build. An x64 build exists so the engine
and its tests also build and run on x64 machines, in particular the hosted
continuous-integration runners. It uses the win-x64 ONNX Runtime from the same
NuGet package and has no QNN execution provider: `edgebench` offers the CPU
provider only, and `ort_probe --target npu` stops with
`error: [qnn_ep_not_bundled]` (exit 4). On Windows on Arm an x64 build runs under
emulation; `edgebench doctor` reports that as the `process_emulated` warning, and
its timings do not describe the device.

The `x64` and `arm64-ninja` presets use the Ninja Multi-Config generator and the
compiler of the current MSVC developer environment, whose target architecture
decides the build's (any Visual Studio version with that toolset works). From
the repository root:

```powershell
# x64 developer environment (on an ARM64 machine this uses the emulated x64-hosted compiler)
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -SkipAutomaticLocation
powershell -ExecutionPolicy Bypass -File scripts\fetch_deps.ps1 -Arch x64
cmake --preset x64
cmake --build --preset x64-release      # or x64-debug
ctest --preset x64-release -L "unit|ort" --output-on-failure
```

`arm64-ninja` / `arm64-ninja-release` work the same way from an ARM64 developer
environment (`-Arch arm64`). Build directories: `build/x64/`, `build/arm64-ninja/`.

### How the executables find ONNX Runtime

Windows ships an unrelated, older `onnxruntime.dll` in `System32` (1.17.1 on
current Windows 11), and the normal DLL search order can pick it up. So
`edgebench.exe`, `ort_probe.exe` and the tests never let Windows choose:

1. Every target linking ONNX Runtime is built with `ORT_API_MANUAL_INIT` and
   delay-loads `onnxruntime.dll` (set on the `OnnxRuntime::OnnxRuntime` target in
   `cmake/FindOnnxRuntime.cmake`), so nothing touches ONNX Runtime before `main()`.
2. At startup the program loads `onnxruntime.dll` from **its own directory**
   (from the executable's path, not the current directory or `PATH`) by absolute
   path, checks that it reports the ONNX Runtime version the build expects
   (`EDGEBENCH_ONNXRUNTIME_VERSION` in `CMakeLists.txt`, 1.32.0) and provides the
   C API version of the headers, and only then initializes the C++ API.
3. After creating sessions, `ort_probe` also checks that every loaded ONNX
   Runtime / QNN DLL (`onnxruntime*.dll`, `Qnn*.dll`) comes from that directory.

The build copies the bundled DLLs next to the executables. If loading fails,
`edgebench` (e.g. `edgebench doctor`) and `ort_probe` print the error kind and
exit with code 4. `edgebench doctor` names any module loaded from outside the
bundle directory, which is also exit code 4:

| Error | Meaning | Fix |
|---|---|---|
| `dll_not_found` | No `onnxruntime.dll` next to the executable, or a DLL it depends on is missing | Rebuild, or copy `onnxruntime.dll` and `onnxruntime_providers_shared.dll` from `third_party/onnxruntime/lib` next to the `.exe` |
| `load_failed` | The file exists but Windows cannot load it (wrong architecture, damaged) | Re-run `scripts\fetch_deps.ps1 -Force` and rebuild |
| `entry_point_missing` | The DLL is not ONNX Runtime (no `OrtGetApiBase` export) | Replace it with the bundled copy |
| `version_mismatch` | A different ONNX Runtime version, e.g. a copy of the one in `System32` | Replace it with the bundled copy |
| `api_version_unsupported` | Right version, but the headers and the DLL come from different packages | Re-run `scripts\fetch_deps.ps1 -Force` and rebuild |
| `foreign_module_loaded` | An `onnxruntime.dll` from another directory was already loaded into the process | Find what loads it first (e.g. an injected DLL); do not copy ONNX Runtime into `PATH` or `System32` |

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

The probe uses edgebench's [exit codes](cli.md#exit-codes): 0 if everything
passed, 1 if the output comparison failed, 2 on a usage error, 3 on an ONNX
Runtime error (the error code and message are printed) or when `--target npu`
finds no single QNN NPU device (`no_qnn_ep_devices`, `no_npu_device`,
`multiple_npu_devices`, with the device list), 4 if ONNX Runtime, the QNN
execution provider or its HTP backend (`QnnHtp.dll`, `QnnHtpPrepare.dll`,
checked before the first NPU session) could not be loaded or a loaded ONNX
Runtime/QNN DLL is not the bundled one, 5 if strict mode refused the NPU session
(`strict_mode_refusal`), and 6 if the `--json` file could not be written. It registers
the QNN EP and selects the NPU with the same code as `edgebench`. `--json` never
overwrites: if the path already exists, the probe exits with 6 before doing
anything. It loads ONNX Runtime the same way as
`edgebench.exe` (see [How the executables find ONNX Runtime](#how-the-executables-find-onnx-runtime))
and refuses to run with any other copy.

`scripts\npu_overhead.py` uses the probe to compare C++ and Python NPU latency
and to split it into accelerator time and host-side overhead. It reads the
reference inputs from `data/ort_probe_reference/latest/` and writes
`npu_overhead_mobilenetv2_<timestamp>.json` (and `_latest.json`) plus a matching
run directory with the probe's JSON output to `benchmarks/results/local/feasibility/`.

`tools\ort_probe\reference.py` is a quick Python check with an input filled
with 0.5; `ort_probe --fill 0.5` runs the same input from C++.

### 4. Run the CPU/NPU matrix (optional)

`scripts/run_matrix.py` runs `edgebench run` over the comparison matrix of
`scripts/common/matrix.py`: every model (the three image classifiers with
external normalization and pruned INT8 files, the autoencoder's cleaned graph)
at batch 1, 4 and 8 on CPU FP32, CPU INT8, NPU INT8 and NPU FP16, plus NPU INT8
at batch 8 with `--htp-graph-finalization-mode 1` and the folded INT8 files at
batch 1 on both devices. NPU runs use `--ep-cache use` (each entry built just
before its run, in a cache directory of the matrix), `--init-breakdown` and
`--profile-accelerator`. It runs `edgebench doctor` first and stops unless the
power and background-load checks pass (AC power, the Best performance power
mode, an idle machine). `scripts/check_matrix.py` then checks every result:
strict validation, the planned configuration, effective precision, batch
equivalence against `models/batch_equivalence.json`, the cache hit and output
verification, module checks, one QNN partition, accelerator time against
latency and an explicit list of expected warnings, and prints a sanity table.
The results stay local; they check the engine and are not published numbers.

```powershell
.\.venv\Scripts\python.exe scripts\run_matrix.py --trials 3 --iterations 100   # writes benchmarks\results\local\matrix\<stamp>\
.\.venv\Scripts\python.exe scripts\check_matrix.py benchmarks\results\local\matrix\<stamp>
.\.venv\Scripts\python.exe scripts\validate_results.py --strict benchmarks\results\local\matrix\<stamp>\results benchmarks\results\local\matrix\<stamp>\cache_build
```

`--only RUN_ID ...` runs a subset (a quick rerun of the check).

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests to `main`, and
on demand. Hosted runners have neither the exported models nor an NPU, so CI runs
the `unit` and `ort` labels only, never `models` or `npu`.

| Job | Runner | What it does |
|---|---|---|
| `python (validator, hygiene)` | `ubuntu-latest`, Python 3.12 | Installs `requirements-ci.txt` (jsonschema and pytest, pinned); runs the validator, hygiene-check, coverage-summary, dataset-manifest, model-registry, model-manifest/recipe, reference-output, accuracy, time-series, autoencoder (source bundle, recipes, lineage), anomaly-detection and documented-numbers tests; checks the numbers quoted in the docs against the published results (`scripts/check_doc_numbers.py`); validates `benchmarks/results/published/accuracy/` (classification and anomaly-detection results) with `scripts/validate_accuracy.py`; validates `docs/examples/` and `benchmarks/results/published/` with `--strict`; runs the hygiene check |
| `windows-x64` | `windows-latest` | MSVC x64 environment (`scripts/msvc_dev_env.ps1`: `vcvarsall.bat` of the newest Visual Studio, exported to the following steps); fetches the x64 ONNX Runtime; `x64` preset, Release, `/W4 /WX`; `ctest -L "unit\|ort"`; uploads the CTest log on failure |
| `windows-arm64` | `windows-11-arm` | The same for ARM64 (`arm64-ninja` preset). The QNN files are bundled and the QNN EP is registered by the `ort` tests, but the runner has no Qualcomm hardware: the job also runs `ctest -L npu` and fails unless every `npu` test reports itself skipped (the reasons are in the log, the count in the job summary). Both Windows jobs then run `edgebench doctor --json` and require exit code 0, no failed check and the NPU checks N/A or warning (`scripts/validate_doctor.py --runner`) |
| `coverage (x64)` | `windows-latest` | x64 Debug build; the `unit` and `ort` tests under [OpenCppCoverage](https://github.com/OpenCppCoverage/OpenCppCoverage) 0.9.9.0 (installer pinned by SHA256), including child processes, filtered to `engine/src`; uploads the HTML and Cobertura report (`coverage-x64` artifact) and prints per-file line coverage in the job summary. Report only, no threshold |

`third_party/` is cached per architecture, keyed by the hash of
`scripts/fetch_deps.ps1` and `cmake/bundle-pins.json`; `build/_downloads/` is cached keyed by the CMake files.
The Windows jobs pass a venv with `requirements-ci.txt` as `EDGEBENCH_PYTHON`, so
the engine tests that validate the result files they write run there too.
Third-party actions are pinned to full commit SHAs.

To run the same checks locally:

```powershell
# python job (any OS; with the full .venv or a venv with requirements-ci.txt)
python -m pytest scripts/tests/test_validate_results.py scripts/tests/test_check_hygiene.py scripts/tests/test_coverage_summary.py scripts/tests/test_manifests.py scripts/tests/test_export_models.py scripts/tests/test_model_recipes.py scripts/tests/test_references.py scripts/tests/test_accuracy_eval.py scripts/tests/test_timeseries.py scripts/tests/test_autoencoder.py scripts/tests/test_autoencoder_pipeline.py scripts/tests/test_anomaly_eval.py scripts/tests/test_doc_numbers.py scripts/tests/test_validate_doctor.py scripts/tests/test_matrix.py
python scripts/validate_accuracy.py benchmarks/results/published/accuracy
python scripts/validate_results.py --strict docs/examples benchmarks/results/published
python scripts/check_doc_numbers.py
python scripts/check_hygiene.py

# windows-x64 job: see "x64 and Ninja builds" above
ctest --preset x64-release -L "unit|ort" --output-on-failure
```

### Hygiene check

`scripts/check_hygiene.py` (standard library only) fails if a local-only notes
file is tracked, if a tracked text file contains development-process wording
(phase or prompt numbers, names of AI tools or assistants, attribution
trailers), or if a tracked file is larger than 1 MiB (the only allowed exception is the full ImageNetV2 manifest,
`datasets/imagenetv2-matched-frequency/eval-full.tsv`).
It also scans commit messages: those not yet on `origin/main` (or `--base REF`)
must be clean; older ones are only reported. The patterns are defined once, at
the top of the script. A line can be exempted with the marker `hygiene: allow`,
for genuine false positives only; there are none at present.

OpenCppCoverage instruments x64 processes only, so coverage is measured on the
x64 build; the code under test is the same as on ARM64.
