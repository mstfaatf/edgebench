# Command-line interface

```text
edgebench run --model <file.onnx> [options]    benchmark a model
edgebench doctor [--json]                      check the build, the machine and ONNX Runtime
edgebench --version | --help
edgebench <command> --help
```

- Results and JSON go to stdout; errors and diagnostics go to stderr. There are
  no color codes.
- `--help` and `--version` never load ONNX Runtime, so they work even when the
  DLLs are missing. Running `edgebench` with no command, or with an unknown
  command, prints the help to stderr and exits with code 2.
- Commands that need ONNX Runtime load the bundled `onnxruntime.dll` from the
  executable's directory and verify it first; see
  [setup.md](setup.md#how-the-executables-find-onnx-runtime).

## `edgebench run`

Benchmarks one model with one configuration. Only the CPU provider is available
in this version, and benchmark execution itself is not part of this build yet:
without `--dry-run`, `run` validates its options, prints
`error: benchmark execution is not available in this build` and exits with
code 1.

| Option | Default | Constraints / meaning |
|---|---|---|
| `--model PATH` | (required) | Must exist, be a regular file, and end in `.onnx` (any case) |
| `--provider NAME` | `cpu` | Supported providers: `cpu` |
| `--threads N` | `4` | CPU intra-op threads, ≥ 1. Always set explicitly and recorded; there is no "ONNX Runtime default" setting, because CPU latency depends strongly on it |
| `--binding MODE` | `iobinding` | `run`: `Session::Run` with input and output values created per call. `iobinding`: IoBinding with buffers allocated once. Recorded as `config.run_binding` = `run` / `iobinding_preallocated` |
| `--warmup N` | `20` | Unmeasured iterations before each trial, ≥ 0 |
| `--iterations N` | `200` | Measured iterations per trial, ≥ 1 |
| `--trials N` | `3` | Number of trials, ≥ 1 |
| `--cooldown-ms N` | `2000` | Pause between trials, ≥ 0 |
| `--input KIND` | `random` | `random` or `constant` |
| `--seed N` | `0` | RNG seed (unsigned integer). Only allowed with `--input random` |
| `--constant-value FLOAT` | `0.0` | Fill value, finite. Only allowed with `--input constant` |
| `--graph-opt LEVEL` | `all` | ONNX Runtime graph optimization: `disable`, `basic`, `extended`, `all` (recorded as `disable_all` … `enable_all`) |
| `--label TEXT` | none | Name for the run; must not be empty if given |
| `--output-dir PATH` | `benchmarks/results/local` | Directory for result files, relative to the current directory |
| `--dry-run` | off | Validate the options, load ONNX Runtime, print the resolved configuration as JSON, exit 0 |

All validation problems are reported together, one per line, as
`error: --<option>: <message>`, followed by a hint to run `edgebench run --help`.

### `--dry-run` output

The resolved configuration with every default filled in. `config` uses the
field names of the [results schema](results-schema.md)'s `config` block.
`config.batch_size` is `null` because it comes from the model's input shape when
the session is created. `model`, `label`, `output_dir` and `cooldown_ms` are run
settings the schema has no `config` field for. `runtime.onnxruntime` is the
loaded library, as in a result file.

```json
{
  "command": "run",
  "dry_run": true,
  "edgebench_version": "0.1.0",
  "results_schema_version": "1.1.0",
  "model": {
    "path": "C:\\dev\\edgebench\\models\\mobilenetv2\\mobilenetv2_fp32_b1.onnx",
    "file_name": "mobilenetv2_fp32_b1.onnx",
    "size_bytes": 13992263
  },
  "label": null,
  "output_dir": "C:\\dev\\edgebench\\benchmarks\\results\\local",
  "cooldown_ms": 2000,
  "config": {
    "requested": {
      "execution_provider": "CPUExecutionProvider",
      "device_type": "CPU",
      "compute_precision": null
    },
    "strict_mode": false,
    "batch_size": null,
    "input": {
      "kind": "random",
      "constant_value": null,
      "seed": 0,
      "dataset": null
    },
    "warmup_iterations": 20,
    "measured_iterations": 200,
    "trials": 3,
    "run_api": "sync",
    "ort_execution_mode": "sequential",
    "intra_op_threads": 4,
    "inter_op_threads": null,
    "graph_optimization_level": "enable_all",
    "ep_options": {},
    "session_config": {},
    "ep_context_cache": null,
    "run_binding": "iobinding_preallocated",
    "profiling": {
      "ort": false,
      "qnn_level": "off"
    }
  },
  "runtime": {
    "onnxruntime": {
      "version": "1.32.0",
      "library_path": "C:\\dev\\edgebench\\build\\arm64-release\\bin\\Release\\onnxruntime.dll",
      "api_version": 32,
      "load_check": "full_path_verified"
    }
  }
}
```

## `edgebench doctor`

Checks everything a benchmark depends on and reports it:

- **Build**: edgebench version, git commit and whether the tree had uncommitted
  changes at build time, build type, compiler, results schema version.
- **System**: OS name, version and build; OS and process architecture; CPU name,
  physical cores and logical processors; total physical memory; device
  manufacturer and model. An x64 (or ARM64EC) build running under emulation on
  ARM64 Windows is reported as `EMULATED` with a warning, because its timings
  would not represent native performance.
- **ONNX Runtime**: the load result (version, C API version, library path,
  `load_check`), or the loader's error kind and message.
- **Loaded modules**: every ONNX Runtime and QNN DLL in the process, with its
  path, and whether it is inside the bundle directory (the executable's
  directory).
- **EP devices**: the devices ONNX Runtime enumerates without any plugin EP
  registered (the CPU EP).

| Option | Meaning |
|---|---|
| `--json` | Print the report as JSON instead of text. `tool`, `system` and `runtime` use the [results schema](results-schema.md)'s field names where they overlap; `onnxruntime_load`, `problems`, `warnings`, `result` and `exit_code` summarize the checks |

Exit code 0 when every check passes. Exit code 4 when ONNX Runtime cannot be
loaded (the report still contains every section that does not need it) or when
an ONNX Runtime/QNN module is loaded from outside the bundle directory (the
module is named). Each problem is also printed to stderr as
`error: [<kind>] <message>`.

Sample output (Surface Laptop 7, development build):

```text
Build
  edgebench          0.1.0
  git commit         338e6266e7f863b182747b3ee6ca248820a1ca5a (dirty)
  build type         Release
  compiler           MSVC 19.42.34435.0
  results schema     1.1.0

System
  OS                 Windows 11 Home
  OS version         25H2 (build 26200.9974)
  OS architecture    arm64
  process            arm64 (native)
  CPU                Snapdragon(R) X 10-core X1P64100 @ 3.40 GHz
  cores              10 physical, 10 logical
  memory             15.6 GiB (16757260288 bytes)
  device             Microsoft Corporation / Microsoft Surface Laptop, 7th Edition

ONNX Runtime
  required           1.32.0 (C API 32)
  library            C:\dev\edgebench\build\arm64-release\bin\Release\onnxruntime.dll
  version            1.32.0 (C API 32)
  load check         full_path_verified

Loaded ONNX Runtime / QNN modules (bundle: C:\dev\edgebench\build\arm64-release\bin\Release)
  C:\dev\edgebench\build\arm64-release\bin\Release\onnxruntime.dll

EP devices (no plugin EP registered)
  CPUExecutionProvider  type CPU  ep vendor Microsoft  vendor Qualcomm  vendor_id 0x4d4f4351  device_id 0x7
      device metadata: {Description=Snapdragon(R) X 10-core X1P64100 @ 3.42 GHz}
      ep metadata:     {version=1.32.0}

Result: OK
```

With a copy of `C:\Windows\System32\onnxruntime.dll` next to the executable, the
ONNX Runtime section reads `load check FAILED (version_mismatch)` and
`found version 1.17.1`, stderr has
`error: [version_mismatch] ...\onnxruntime.dll is ONNX Runtime 1.17.1, but EdgeBench requires ONNX Runtime 1.32.0. ...`,
and the exit code is 4.

## Exit codes

Shared by `edgebench` and `tools/ort_probe` (defined in
`engine/include/edgebench/exit_codes.hpp`).

| Code | Name | Meaning |
|---|---|---|
| 0 | `success` | Success |
| 1 | `internal_error` | Internal or unexpected error. `ort_probe` also uses it when one of its checks fails (output comparison, device selection) |
| 2 | `usage_error` | Bad arguments, invalid values, or model file not found |
| 3 | `ort_runtime_error` | ONNX Runtime error: session creation or inference failed |
| 4 | `ort_load_failure` | ONNX Runtime could not be loaded (any loader error kind), or a loaded ONNX Runtime/QNN DLL is not the bundled one |
| 5 | `strict_mode_refusal` | Strict mode refused the session: not every node can run on the requested device. Reserved for the NPU provider |
| 6 | `output_error` | Results could not be written |
