# EdgeBench

[![CI](https://github.com/mstfaatf/edgebench/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mstfaatf/edgebench/actions/workflows/ci.yml)

CPU vs NPU ML inference benchmarking for Windows on Snapdragon, built on
[ONNX Runtime](https://onnxruntime.ai/).

EdgeBench measures how fast real models (image classifiers and a time-series
autoencoder) run on the CPU and on the Qualcomm Hexagon NPU of a Snapdragon
laptop. It records the configuration, the software versions, the loaded libraries and the state of the
machine with every result, verifies where each node of the model actually ran,
and writes results as schema-validated JSON plus CSV.

## Status

- **CPU benchmarking works**: `edgebench run` benchmarks a model on the ONNX
  Runtime CPU execution provider with repeated trials, statistics, system-state
  recording and result files.
- **NPU support is in progress.** `edgebench run --provider npu` creates and
  times sessions on the NPU through the Qualcomm QNN execution provider, in
  strict mode by default, with the provider options and node placement recorded
  ([docs/cli.md](docs/cli.md#npu-runs)); its outputs are bit-identical to the
  Python ONNX Runtime references on the NPU ([docs/models.md](docs/models.md#what-the-c-tests-prove)).
  Results (schema 1.2.0) record the effective compute precision, the nodes a
  strict-mode refusal names, the loaded QNN backend's version, and whether the
  batch size computes the same per-item outputs and predictions as batch 1
  ([docs/models.md](docs/models.md#batch-equivalence)). An EP context cache,
  keyed by everything a compiled context depends on and verified against a
  compiled session, and a cold / context-build / cached initialization
  breakdown are available ([docs/cli.md](docs/cli.md#ep-context-cache-and-initialization)).
  The QNN SDK version and HTP architecture are recorded as the QNN execution
  provider reports them at run time, and an optional separate profiled pass
  records the accelerator time per inference
  ([docs/cli.md](docs/cli.md#accelerator-time---profile-accelerator)).
  See [NPU runs](#npu-runs) below. No NPU latency results are published yet.
- **Accuracy results are published**: every model variant on ImageNetV2,
  on the CPU and on the NPU (strict mode), with confidence intervals and paired
  CPU vs NPU comparisons ([docs/accuracy.md](docs/accuracy.md)). On the same
  INT8 file the NPU matches the CPU within 0.15 top-1 points. The Conv1D
  autoencoder's anomaly-detection results on the NAB machine-temperature series
  are published the same way ([docs/autoencoder.md](docs/autoencoder.md)).
- **No published latency results yet.** Latency numbers in the docs are
  methodology checks from one machine, not benchmark results.

## Requirements

- Windows 11 on Snapdragon (ARM64). Tested on a Snapdragon X Plus (Surface
  Laptop 7).
- Visual Studio 2022 or newer with the C++ workload and the ARM64 build tools;
  the CMake and Ninja bundled with Visual Studio are enough.
- PowerShell, for `scripts/fetch_deps.ps1` (downloads ONNX Runtime and the QNN
  execution provider).
- Native ARM64 Python 3.12, only for preparing models and running the Python
  scripts and tests.

A CPU-only x64 build also exists, mainly for continuous integration; it is not
used for measurements.

## Quick start

From the repository root, in a *Developer PowerShell for VS 2022* (which puts
`cmake` on `PATH`; see [docs/setup.md](docs/setup.md) for other shells):

```powershell
# Native dependencies (ONNX Runtime 1.32.0, QNN EP 2.8.0), pinned by SHA-256
powershell -ExecutionPolicy Bypass -File scripts\fetch_deps.ps1

# Configure, build and test
cmake --preset arm64-release
cmake --build --preset arm64-release
ctest --preset arm64-release -L "unit|ort" --output-on-failure

# Check the machine and the runtime
.\build\arm64-release\bin\Release\edgebench.exe doctor
```

Run `edgebench doctor` first on every machine, and again before any run whose
numbers you want to trust: it checks the bundle against its pinned hashes, runs a
tiny model on the NPU and compares the output bit for bit, and reports power and
background load ([docs/cli.md](docs/cli.md#edgebench-doctor)).

Until the Python environment (`.venv`, see the setup guide) exists, the tests
that check written result files with the Python validator skip that check, and
the `schema` test is not added; configure again after creating it.

To benchmark a model you need an ONNX file with fixed input shapes. The
[setup guide](docs/setup.md#preparing-data-and-models) creates FP32 models
(MobileNetV2, ResNet-18 and EfficientNet-Lite0, batch sizes 1, 4 and 8; see [docs/models.md](docs/models.md)),
their normalization-folded variants and their static INT8 QDQ versions with a
Python environment (`scripts\export_models.py`, `scripts\fold_models.py`,
`scripts\quantize_models.py`), and the files of the
Conv1D time-series autoencoder from its committed source
(`scripts\prepare_autoencoder.py`). Then:

```powershell
.\build\arm64-release\bin\Release\edgebench.exe run --model models\mobilenetv2\mobilenetv2_fp32_b1.onnx
```

Defaults: CPU, 4 intra-op threads, IoBinding, 3 trials of 200 measured
iterations after 20 warm-up iterations, 2 s idle before every trial. An excerpt
of the summary (FP32 MobileNetV2, Surface Laptop 7, plugged in, *Balanced*
power mode):

```text
Provider    CPUExecutionProvider, 4 intra-op threads, binding iobinding_preallocated, graph optimization enable_all
Trials      3 x 200 measured iterations (20 warm-up, 2000 ms cooldown), batch size 1
Power       ac, mode Balanced, plan Balanced, battery saver off, battery 80%
CPU load    pre-run 7.1% over 2.0 s, highest cooldown 7.5% (before trial 2), highest during trials 9.1% (trial 1, own process excluded)

ms               mean        p50        p95        p99        min        max inferences/s
trial 1        4.3883     4.0223     4.7335    15.0211     3.6030    26.1895        227.9
trial 2        4.0326     3.9751     4.4864     4.7538     3.5880     4.8543        248.0
trial 3        4.0293     4.0011     4.4871     4.7743     3.5916     5.1914        248.2
aggregate      4.1501     4.0038     4.6084     5.2121     3.5880    26.1895        241.0

Trial means 4.0293 .. 4.3883 ms, CV 4.97%
Placement   55 nodes: CPUExecutionProvider 55 (all on the requested EP)
```

Results land in `benchmarks\results\local\` (not committed):
`<name>_<UTC stamp>.json` (results schema 1.2.0), a directory with
`latencies.csv` and `summary.csv`, and `<name>_latest.json`. Existing files
are never overwritten. Check a result with:

```powershell
.\.venv\Scripts\python.exe scripts\validate_results.py --strict benchmarks\results\local\mobilenetv2_fp32_b1_cpu_iobinding_t4_latest.json
```

## NPU runs

Run `edgebench doctor` first: it checks the bundled QNN files, the NPU, a probe
model on the NPU, the EP context cache and the power settings. Then, for
example:

```powershell
# INT8 as stored; strict mode (no CPU fallback) is the default
.\build\arm64-release\bin\Release\edgebench.exe run --model models\mobilenetv2\mobilenetv2_int8_b1.onnx --provider npu --ep-cache use --init-breakdown --profile-accelerator
# an FP32 model needs FP16 compute: the NPU has no FP32
.\build\arm64-release\bin\Release\edgebench.exe run --model models\mobilenetv2\mobilenetv2_fp32_b1.onnx --provider npu --npu-fp16
```

[docs/npu.md](docs/npu.md) explains what an NPU result contains and how to read
it: device and bundle checks, strict mode, precision, the provider options
(including `--htp-graph-finalization-mode`), cold, build and cached
initialization, the EP context cache, accelerator time against latency, and
why INT8 image models at batch 8 compute something different from batch 1.

## Experiment suites

Benchmark plans run through `edgebench suite`: a JSON configuration lists the
models and configurations (or a matrix of providers, precisions and batch
sizes), run parameters, repetitions and cooldowns. Every run is its own
`edgebench run` process, the configurations run in a rotated order, a
pre-flight gate (AC power and Best performance, sleep settings, an idle
machine, no other NPU users, doctor, and a sustained-load run that detects
throttling) decides whether the results are publishable, the power state and
background load are checked again before every run, every result is
validated, and an interrupted suite resumes where it stopped.

```powershell
.\build\arm64-release\bin\Release\edgebench.exe suite benchmarks\configs\smoke.json --dry-run
.\build\arm64-release\bin\Release\edgebench.exe suite benchmarks\configs\smoke.json
```

See [docs/suite.md](docs/suite.md).

## Testing

Tests carry CTest labels by what they need: `unit` (nothing), `ort` (the ONNX
Runtime DLLs), `models` (locally prepared models), `npu` (the NPU) and
`schema` (the Python environment). Continuous integration builds with
`/W4 /WX` and runs `unit` and `ort` on Windows x64 and ARM64 runners, line
coverage on x64, and on Linux the result and accuracy validators, the
manifest and pipeline tests that need no model files, a check that the numbers
quoted in the documentation match the published results, and the repository
hygiene check. See [Test labels](docs/setup.md#test-labels) and
[Continuous integration](docs/setup.md#continuous-integration).

## Documentation

- [docs/methodology.md](docs/methodology.md): what is timed, trial structure,
  statistics, recorded system state, and what EdgeBench does not claim
- [docs/models.md](docs/models.md): supported models (MobileNetV2, ResNet-18,
  EfficientNet-Lite0, and the imported Conv1D autoencoder), weights, export
  settings and checks; which input each file expects (normalized, or [0, 1] for
  the normalization-folded files)
- [docs/accuracy.md](docs/accuracy.md): accuracy of every variant on ImageNetV2,
  CPU and NPU, and the CPU vs NPU comparison
- [docs/autoencoder.md](docs/autoencoder.md): time-series anomaly-detection
  autoencoder: NAB data, windows and splits, Colab training, the committed
  source and its exact graph clean-up, INT8, NPU checks and the
  anomaly-detection evaluation on CPU and NPU
- [docs/npu.md](docs/npu.md): NPU runs: doctor, device and bundle checks,
  strict mode, precision, provider options, initialization, the EP context
  cache, accelerator time, batch equivalence, and why NPU INT8 image models
  compute different outputs from batch 7 up
- [docs/cli.md](docs/cli.md): `edgebench` commands, options, output and exit
  codes
- [docs/suite.md](docs/suite.md): experiment suites: configuration files,
  matrix expansion, run order, run IDs, the pre-flight gate, the manifest,
  resuming and the duration estimate
- [docs/results-schema.md](docs/results-schema.md): result file format (JSON
  Schema in [schemas/](schemas/))
- [docs/architecture.md](docs/architecture.md): engine structure
- [docs/setup.md](docs/setup.md): environment, data and model preparation,
  building, tests and continuous integration
- [docs/feasibility.md](docs/feasibility.md): technical note on CPU and NPU
  execution, NPU verification, accuracy and the evidence behind the methodology

## Repository layout

| Path | Contents |
|---|---|
| `engine/` | C++ engine: `edgebench_core` library, `edgebench` CLI, tests |
| `tools/` | Diagnostics: `ort_probe` (C++ CPU/NPU runner), QNN rejection check |
| `scripts/` | Dependency fetch, data and model preparation, NPU checks, result validation |
| `schemas/` | JSON Schema for result, accuracy, doctor, suite and model-recipe files, one file per version |
| `datasets/` | Manifests of the evaluation and calibration data (images: path, label, SHA256; the NAB time series: windows and splits); the data is downloaded, not committed |
| `notebooks/` | Colab notebook that trains the Keras autoencoder and exports it to ONNX |
| `models/` | Model manifest (committed), the committed autoencoder source in `models/sources/`; model files are made locally |
| `benchmarks/` | Suite configurations (`configs/`) and results (`results/published/` is committed, `results/local/` is not) |
| `ml/`, `dashboard/` | Model training/export pipelines and the static dashboard (planned) |
| `docs/` | Documentation |

## License

EdgeBench is released under the [MIT License](LICENSE). Third-party components
(ONNX Runtime, the QNN execution provider and Qualcomm's QNN libraries, C++
libraries, models and datasets) are downloaded at build or run time, not stored
in the repository; their licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
