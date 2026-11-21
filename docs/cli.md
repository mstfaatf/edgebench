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

Benchmarks one model with one configuration and prints a summary. Only the CPU
provider is available in this version. Results are kept in memory and printed;
`--output-dir` is validated but no result file is written yet. `--dry-run`
creates the real session and checks one inference without measuring.

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
| `--input KIND` | `random` | `random` or `constant`; see [Input data](#input-data) |
| `--seed N` | `0` | RNG seed (unsigned integer). Only allowed with `--input random` |
| `--constant-value FLOAT` | `0.0` | Fill value, finite. Only allowed with `--input constant` |
| `--graph-opt LEVEL` | `all` | ONNX Runtime graph optimization: `disable`, `basic`, `extended`, `all` (recorded as `disable_all` … `enable_all`) |
| `--label TEXT` | none | Name for the run; must not be empty if given |
| `--output-dir PATH` | `benchmarks/results/local` | Directory for result files, relative to the current directory |
| `--dry-run` | off | Validate the options, create the session, run one inference, print the resolved configuration as JSON, exit 0 |

All validation problems are reported together, one per line, as
`error: --<option>: <message>`, followed by a hint to run `edgebench run --help`.

### What `run` measures

A run consists of `--trials` trials. Each trial:

1. waits `--cooldown-ms` (before every trial except the first);
2. creates a **new session** (the creation time of the `Ort::Session` alone,
   without input generation, binding setup or model hashing, goes into
   `timing.session_init.cold_ms`, one value per trial);
3. runs the first warm-up iteration and times it (trial 1's value is
   `timing.first_inference_ms`, which often includes one-time setup);
4. runs the remaining warm-up iterations untimed;
5. runs `--iterations` measured iterations, each timed on its own around the
   inference call only (`run_once()`: `Session::Run`, or the IoBinding run) with
   `std::chrono::steady_clock`;
6. checks that every output value is finite, compares the final outputs with
   trial 1's, and releases the session.

After the last trial, node placement is recorded in a separate session with
verbose logging (see [`--dry-run`](#--dry-run)); doing it last means it cannot
warm up any timed session.

Not timed: model hashing, session creation (measured separately), input
generation, binding setup, warm-up iterations after the first, output checks,
and cooldowns. Inputs are generated once per session before the first
iteration (see [Input data](#input-data)); there is no image decoding, resizing
or normalization in a run.

Statistics, as defined in the [results schema](results-schema.md#timing): per
trial and pooled over all trials (`aggregate`): mean, nearest-rank p50/p95/p99,
sample standard deviation (n − 1), min, max, and throughput
(`inferences/s` = 1000 / mean, sequential). p95 needs at least 20 samples and
p99 at least 100; below that they are shown as `-`. The trial-to-trial spread
is the range and the coefficient of variation (sample stddev / mean) of the
trial means. `session_create_ms` is the nearest-rank median of the per-trial
creation times.

Stops with an error:

- exit 3 if an inference fails (the message names the trial and iteration, and
  contains the ONNX Runtime error code);
- exit 1 if an output contains NaN or infinity (names the output and trial), or
  if a trial's outputs have a different top-1 than trial 1's for the same input.
  Outputs that differ only in value but not in top-1 are reported as a
  `nondeterministic_outputs` warning instead.

Trial order comes from a scheduler that rotates the starting configuration
every round; with the single configuration of `run` the order is simply trial
1, 2, 3, … (see [architecture.md](architecture.md#trial-loop-and-scheduling)).

### Output

Progress (`trial 1 of 3: ...`, `cooldown 2000 ms`, `recording node placement`)
and warnings go to stderr; the summary goes to stdout. Times are milliseconds.
For FP32 MobileNetV2 with the defaults on a Surface Laptop 7 (plugged in,
*Balanced* power mode):

```text
Model       mobilenetv2_fp32_b1.onnx (fp32, 13992263 bytes, sha256 163fa832dc77)
Provider    CPUExecutionProvider, 4 intra-op threads, binding iobinding_preallocated, graph optimization enable_all
Trials      3 x 200 measured iterations (20 warm-up, 2000 ms cooldown), batch size 1
Input       random
Session     create ms: 21.7 50.2 48.1  (median 48.1)
First run   7.005 ms (first warm-up iteration of trial 1)

ms               mean        p50        p95        p99        min        max inferences/s
trial 1        4.0445     3.9595     4.6094     4.8339     3.5801     5.3162        247.3
trial 2        3.9776     3.8939     4.5425     4.8205     3.5677     5.5284        251.4
trial 3        4.0434     3.9115     4.7777     5.1508     3.5848     5.5991        247.3
aggregate      4.0218     3.9300     4.6420     4.9964     3.5677     5.5991        248.6

Trial means 3.9776 .. 4.0445 ms, CV 0.95%
Placement   55 nodes: CPUExecutionProvider 55 (all on the requested EP)
```

- `Session`: creation time of each trial's session, and their median. Session
  creation is sensitive to the CPU's idle state: after a 2 s cooldown it took
  about 50 ms on this machine, against about 16 ms when trials ran back to back
  (and about 22 ms for the first trial, which follows model hashing). The cooldown
  lets the CPU settle between trials; warm-up absorbs the effect before the
  measured iterations.
- `First run`: trial 1's first warm-up iteration.
- One row per trial, then `aggregate` over all measured iterations pooled.
- `Trial means`: range of the trial means and their coefficient of variation.
- `Placement`: node count per execution provider after graph optimization.

### `--dry-run`

A dry run does everything a benchmark does up to the first measurement:

1. Validates the options and loads ONNX Runtime.
2. Reads the model file's size and SHA-256.
3. Creates the provider and the session that would be timed, and checks the
   model: every input and output dimension must be fixed (a symbolic or
   dynamic dimension is a usage error naming the tensor and the dimension), and
   every input must be float32 (anything else is a usage error naming the input
   and its dtype). `batch_size` is dimension 0 of the first input.
4. Generates the inputs (see [Input data](#input-data)) and prepares the run
   binding.
5. Runs one inference and checks that every floating-point output value is
   finite.
6. Records node placement from a second session created with the same options
   and verbose logging (ONNX Runtime's "Node placements" log). It is a separate
   session because verbose logging slows session creation down; a timed
   session never logs verbosely.

It prints the result as JSON with the field names of the
[results schema](results-schema.md): `model` (with `sha256`, `inputs` and
`outputs`), `config` (with `batch_size` and `session_config`),
`runtime.execution_provider`, `runtime.ep_device` and `execution`
(`verification_method` is `node_assignment`, with the node counts per EP).
`label`, `output_dir` and `cooldown_ms` are run settings the schema's `config`
block has no field for; `dry_run_check` reports the session creation time of
this one session and the inference check.

Exit codes: 0 if everything passed; 2 for a usage error (including a model with
dynamic shapes or non-float32 inputs); 3 if ONNX Runtime fails to create the
session or run the inference (the message contains the ONNX Runtime error code,
e.g. `ONNX Runtime error ORT_INVALID_PROTOBUF (code 7)`); 4 if ONNX Runtime
cannot be loaded; 1 if an output contains NaN or infinity (the message names the
output). If any node is not placed on the requested execution provider, the
JSON has `full_device_execution_verified: false` and a `warnings` entry, and the
warning is also printed to stderr.

Output for the FP32 MobileNetV2 model (`run --model models\mobilenetv2\mobilenetv2_fp32_b1.onnx --dry-run`):

```json
{
  "command": "run",
  "dry_run": true,
  "edgebench_version": "0.1.0",
  "results_schema_version": "1.1.0",
  "model": {
    "path": "C:\\dev\\edgebench\\models\\mobilenetv2\\mobilenetv2_fp32_b1.onnx",
    "file_name": "mobilenetv2_fp32_b1.onnx",
    "size_bytes": 13992263,
    "sha256": "163fa832dc77de88f24dd0f81588bc9a81167b6e08d9631f4a33c776a7e31dce",
    "inputs": [
      {
        "name": "input",
        "dtype": "float32",
        "shape": [
          1,
          3,
          224,
          224
        ]
      }
    ],
    "outputs": [
      {
        "name": "logits",
        "dtype": "float32",
        "shape": [
          1,
          1000
        ]
      }
    ]
  },
  "label": null,
  "output_dir": "c:\\dev\\edgebench\\benchmarks\\results\\local",
  "cooldown_ms": 2000,
  "config": {
    "requested": {
      "execution_provider": "CPUExecutionProvider",
      "device_type": "CPU",
      "compute_precision": null
    },
    "strict_mode": false,
    "batch_size": 1,
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
    "session_config": {
      "session.intra_op.allow_spinning": "1"
    },
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
    },
    "execution_provider": {
      "name": "CPUExecutionProvider",
      "version": null,
      "library_path": null
    },
    "ep_device": {
      "ep_name": "CPUExecutionProvider",
      "ep_vendor": "Microsoft",
      "device_type": "CPU",
      "vendor": "Qualcomm",
      "vendor_id": 1297040209,
      "device_id": 7,
      "driver_version": null,
      "metadata": {
        "Description": "Snapdragon(R) X 10-core X1P64100 @ 3.42 GHz"
      },
      "selection": {
        "method": "by_device_type",
        "candidate_count": 1
      }
    }
  },
  "execution": {
    "full_device_execution_verified": true,
    "verification_method": "node_assignment",
    "verification_details": "Node placement from ONNX Runtime's verbose session log, recorded in a separate session with the same options as the measured one.",
    "node_assignment": {
      "total_nodes": 55,
      "nodes_by_ep": {
        "CPUExecutionProvider": 55
      }
    }
  },
  "warnings": [],
  "dry_run_check": {
    "session_create_ms": 25.647,
    "inferences": 1,
    "outputs_finite": true
  }
}
```

### Input data

Inputs are generated once, before the first inference, into buffers the
session owns, and the same data is used for every iteration.

- `--input constant`: every element of every input is `--constant-value`
  (converted to float32).
- `--input random`: one SplitMix64 generator, seeded with `--seed`, fills the
  inputs in model order, each in row-major order, one value per element.
  The result is bit-identical on every platform, compiler and standard library.

SplitMix64 (64-bit unsigned arithmetic, wrapping):

```text
state = seed
next():
    state = state + 0x9E3779B97F4A7C15
    z = state
    z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z XOR (z >> 27)) * 0x94D049BB133111EB
    return z XOR (z >> 31)
```

Each float32 is `(next() >> 40) * 2^-24`: the top 24 bits as an integer, scaled
to [0, 1). Every such value is exactly representable in float32, so no rounding
is involved. With seed 0, the first `next()` values are `0xE220A8397B1DCDAF`,
`0x6E789E6AA1B965F4`, `0x06C45D188009454F`, and the first float is
`14819496 / 16777216` (≈ 0.8833). In Python:

```python
import numpy as np

def edgebench_random(seed: int, count: int) -> np.ndarray:
    mask = (1 << 64) - 1
    state, out = seed, np.empty(count, dtype=np.float32)
    for i in range(count):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z ^= z >> 31
        out[i] = np.float32(z >> 40) * np.float32(2.0 ** -24)
    return out
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
