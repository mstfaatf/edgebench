# EdgeBench

CPU vs NPU inference benchmarking for Windows on Snapdragon, built on
[ONNX Runtime](https://onnxruntime.ai/).

## Status

**Early development.** The build, toolchain and environment checks are in
place, but nothing has been benchmarked yet and EdgeBench publishes no
performance numbers at this stage.

## Goals

- **Real inference**: benchmark real models (e.g. MobileNetV2, EfficientNet-Lite)
  end to end, not synthetic kernels.
- **Reproducible methodology**: pinned dependencies, fixed inputs, explicit
  warm-up and iteration counts, and recorded device and software versions.
- **CPU and Qualcomm NPU**: the ONNX Runtime CPU execution provider vs the
  Qualcomm QNN execution provider on the Hexagon NPU.
- **Verified NPU execution**: confirm that a model actually runs on the NPU
  (no silent CPU fallback) and that its outputs are correct.
- **Structured output**: results as JSON/CSV for analysis and for a static
  dashboard.

## Supported platforms

- Windows 11 on Snapdragon (ARM64), native ARM64 builds only.
- Designed to work on other Snapdragon devices by detecting the available
  execution providers and devices at run time.
- So far only tested on a Snapdragon X Plus (Surface Laptop 7).

## Repository layout

| Path | Contents |
|---|---|
| `engine/` | C++ benchmarking engine (`edgebench_core` library, `edgebench` CLI, tests) |
| `ml/` | Python model preparation: PyTorch, TensorFlow/Keras (trained off-device), export, quantization, validation |
| `models/` | Locally produced model files (not committed) |
| `benchmarks/` | Benchmark configurations and results (`results/published/` is committed, `results/local/` is not) |
| `schemas/` | JSON Schema for benchmark result files ([format spec](docs/results-schema.md)) |
| `dashboard/` | Static results dashboard |
| `scripts/` | Environment check and dependency fetch scripts |
| `tools/` | Standalone diagnostics, e.g. `ort_probe` for ONNX Runtime + QNN |
| `cmake/` | Shared CMake modules |
| `docs/` | Setup guide, results format, environment feasibility notes |

## Getting started

See [docs/setup.md](docs/setup.md) for the Python environment, native
dependencies, and building the C++ code. In short, from the repository root in
a *Developer PowerShell for VS 2022* (which puts `cmake` on `PATH`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch_deps.ps1
cmake --preset arm64-release
cmake --build --preset arm64-release
ctest --preset arm64-release
.\build\arm64-release\bin\Release\edgebench.exe --version
```

[docs/feasibility.md](docs/feasibility.md) records which parts of the toolchain
run natively on Windows on Snapdragon.

## License

[MIT](LICENSE)
