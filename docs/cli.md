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

Benchmarks one model with one configuration, prints a summary, and writes the
result as a results-schema 1.1.0 JSON document plus CSV files (see
[Result files](#result-files)). Only the CPU provider is available in this
version. `--dry-run` creates the real session and checks one inference without
measuring or writing anything.

| Option | Default | Constraints / meaning |
|---|---|---|
| `--model PATH` | (required) | Must exist, be a regular file that can be opened for reading, and end in `.onnx` (any case) |
| `--provider NAME` | `cpu` | Supported providers: `cpu` |
| `--threads N` | `4` | CPU intra-op threads, 1 to 256. Always set explicitly and recorded; there is no "ONNX Runtime default" setting, because CPU latency depends strongly on it. More threads than logical processors is allowed, with a `threads_exceed_logical_processors` warning |
| `--binding MODE` | `iobinding` | `run`: `Session::Run` with input and output values created per call. `iobinding`: IoBinding with buffers allocated once. Recorded as `config.run_binding` = `run` / `iobinding_preallocated` |
| `--warmup N` | `20` | Unmeasured iterations before each trial, 0 to 100,000 |
| `--iterations N` | `200` | Measured iterations per trial, 1 to 1,000,000 |
| `--trials N` | `3` | Number of trials, 1 to 1,000 |
| `--cooldown-ms N` | `2000` | Idle period before every trial (the first included), 0 to 600,000; background CPU load is sampled during it |
| `--load-sample-ms N` | `1000` | With `--cooldown-ms 0`: length of the background-load sample (and idle) before the first trial, 0 to 600,000. `0` takes no sample: `system.load` is null with a `value_unavailable` warning |
| `--input KIND` | `random` | `random` or `constant`; see [Input data](#input-data) |
| `--seed N` | `0` | RNG seed (unsigned integer). Only allowed with `--input random` |
| `--constant-value FLOAT` | `0.0` | Fill value, finite. Only allowed with `--input constant` |
| `--graph-opt LEVEL` | `all` | ONNX Runtime graph optimization: `disable`, `basic`, `extended`, `all` (recorded as `disable_all` … `enable_all`) |
| `--label TEXT` | none | Name for the run; must not be empty if given |
| `--output-dir PATH` | `benchmarks/results/local` | Directory for result files (created if missing), relative to the current directory |
| `--name TEXT` | `<model stem>_<provider>_<binding>_t<threads>` | Base name of the result files. Letters, digits, `-`, `_` and `.` only, not starting with `.`, at most 128 characters. In the default, other characters of the model file name become `_`, e.g. `mobilenetv2_fp32_b1_cpu_iobinding_t4` |
| `--format FORMAT` | `summary` | What goes to stdout: `summary` (human-readable) or `json` (the full result document). Progress and errors stay on stderr; the files are written either way |
| `--dry-run` | off | Validate the options, create the session, run one inference, print the resolved configuration as JSON, exit 0 |

All validation problems are reported together, one per line, as
`error: [usage_error] --<option>: <message>` (`[model_unreadable]` when the
model exists but cannot be opened, e.g. because another process holds it
exclusively; the message is the operating system's), followed by one
`hint: Run 'edgebench run --help' for the options.` line. The upper bounds keep
a run's memory and duration finite (one million iterations store 8 MB of
latencies per trial); they are not measurement recommendations.

### What `run` measures

A run consists of `--trials` trials. Each trial:

1. idles for `--cooldown-ms` while sampling system-wide CPU load (before
   **every** trial, including the first; with `--cooldown-ms 0`, for
   `--load-sample-ms` before the first trial only);
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

After the last trial, the process memory is read, node placement is recorded
in a separate session with verbose logging (see [`--dry-run`](#--dry-run);
doing it last means it cannot warm up any timed session), and the power state
is read again.

**Why the first trial idles too.** Session creation is much slower right after
the CPU has been idle, because the CPU drops to a lower power state: on the
reference machine a session took about 50 ms after a 2 s pause, against about
16 ms when trials ran back to back. Without the initial idle, the first trial
(which follows model hashing, a busy period) was created about twice as fast as
the others, and `cold_ms` mixed the two states. Starting every trial after the
same pause removes that systematic difference; session creation times still
vary from trial to trial. The idle period doubles as the background-load
sample.

**System state.** The run records, at its start, the power source, Windows power
mode, power plan, battery charge and battery saver (`system.power`), and the
background CPU load of the idle period before trial 1 (`system.load`). The power
state is read again at the end; if the source, mode, plan or battery saver
changed, the result gets a `power_state_changed` warning. If any idle sample
shows more than 10% CPU utilization, the result gets a
`background_cpu_load_high` warning with the highest value and when it occurred.
The working set is read just before and just after each session's constructor
(outside the timed bracket) for `memory.session_create_working_set_delta_bytes`,
the median over trials; `memory.peak_working_set_bytes` and
`peak_private_bytes` are read after the last trial.

Not timed: model hashing, session creation (measured separately), input
generation, binding setup, warm-up iterations after the first, output checks,
memory and power readings, and idle periods. Inputs are generated once per session before the first
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

Progress (`idle 2000 ms before trial 1 of 3 ...`, `trial 1 of 3: ...`,
`recording node placement`) and warnings go to stderr; the summary goes to
stdout. Times are milliseconds. For FP32 MobileNetV2 with the defaults on a
Surface Laptop 7 (plugged in, *Balanced* power mode, with an editor and a
terminal open):

```text
Model       mobilenetv2_fp32_b1.onnx (fp32, 13992263 bytes, sha256 163fa832dc77)
Provider    CPUExecutionProvider, 4 intra-op threads, binding iobinding_preallocated, graph optimization enable_all
Trials      3 x 200 measured iterations (20 warm-up, 2000 ms cooldown), batch size 1
Input       random
Session     create ms: 58.9 22.3 57.5  (median 57.5)
First run   6.854 ms (first warm-up iteration of trial 1)
Power       ac, mode Balanced, plan Balanced, battery saver off, battery 80%
CPU load    pre-run 5.1% over 2.0 s, highest cooldown 7.4% (before trial 2)
Memory      working set 37.7 MiB after the last trial, peak 64.1 MiB, session-create delta +21.0 MiB (median) (process only)

ms               mean        p50        p95        p99        min        max inferences/s
trial 1        3.9743     3.9131     4.4830     4.8493     3.5862     5.0992        251.6
trial 2        4.1811     3.9320     4.5574     6.1728     3.5681    37.6502        239.2
trial 3        4.0761     3.9642     4.6493     4.9814     3.6605     6.9566        245.3
aggregate      4.0772     3.9428     4.5840     5.0992     3.5681    37.6502        245.3

Trial means 3.9743 .. 4.1811 ms, CV 2.54%
Placement   55 nodes: CPUExecutionProvider 55 (all on the requested EP)
```

- `Session`: creation time of each trial's session, and their median.
- `First run`: trial 1's first warm-up iteration.
- `Power`: power source, Windows power mode, power plan, battery saver and
  battery charge at the start of the run.
- `CPU load`: system-wide CPU utilization during the idle period before trial 1
  (`system.load`), and the highest value seen during the cooldowns.
- `Memory`: the process working set after the last trial's session was
  released, its peak, and the median change of the working set across session
  creation. Process memory only; memory used by drivers or on an accelerator is
  not included.
- One row per trial, then `aggregate` over all measured iterations pooled.
- `Trial means`: range of the trial means and their coefficient of variation.
- `Placement`: node count per execution provider after graph optimization.
- `Results`: the last line, the path of the result file that was written. With
  `--format json`, stdout is only the document and the path goes to stderr.

### Interrupting a run

Ctrl+C, Ctrl+Break and closing the console window stop a run cleanly. The
console handler only sets a flag and signals an event; the run checks the flag
before and after every session creation, before every warm-up and measured
iteration (outside the timed bracket), and during the idle periods, which are
waits on that event and end at once. Then:

- the trial in progress is dropped; a `interrupted_trial_discarded` warning
  says how many measured iterations were discarded;
- node placement is skipped (`execution` stays unverified);
- a failed result is written with `error.code` `interrupted`, the stage the run
  was in, and every completed trial (with its latency files);
- the exit code is 130.

```text
trial 2 of 3: creating session, 20 warm-up + 200 measured iterations
warning: [interrupted_trial_discarded] trial 2 of 3 was interrupted after 46 of 200 measured iterations; the trial is not part of the result
error: [interrupted] interrupted by Ctrl+Break (trial 2 of 3, measured iteration 47)
failed result written to ...\mobilenetv2_fp32_b1_cpu_iobinding_t4_20261123T012641.138Z.json
```

A second Ctrl+C or Ctrl+Break ends the process immediately with exit code 130
and writes nothing more (an atomic write in progress leaves only a temporary
file). When the console window is closed, Windows ends the process shortly
after the handler returns; the handler waits up to 4 s for the result to be
written. `--dry-run`, `doctor` and `ort_probe` have nothing to save: there the
first Ctrl+C or Ctrl+Break exits 130 at once.

### Result files

Every run that gets past argument validation writes, in `--output-dir`:

```text
<name>_<stamp>.json            the result document (results schema 1.1.0)
<name>_<stamp>/latencies.csv   one row per measured iteration
<name>_<stamp>/summary.csv     one row per trial, plus the aggregate row
<name>_latest.json             a copy of the newest result for <name>
```

- `<stamp>` is the UTC start of the run with milliseconds,
  `YYYYmmddTHHMMSS.mmmZ` (e.g. `20261121T235210.123Z`); milliseconds are
  truncated, never rounded. The Python measurement scripts use the same format
  (`scripts/common/results_io.py`).
- **Nothing is overwritten.** Every timestamped file and folder is created
  exclusively. If one with the same stamp exists (two runs in the same
  millisecond), `-2`, `-3`, ... is appended to the stamp. Each file is written to
  a temporary name in its directory and then renamed into place, so an
  interrupted run never leaves a half-written file under a final name.
- `<name>_latest.json` is the only file that is ever replaced, atomically
  (temporary file, then `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING |
  MOVEFILE_WRITE_THROUGH`). It sits in the same directory as the stamped file,
  so its `raw.latencies_file` reference is valid too.
- The document's `raw` section links `latencies.csv` by a path relative to the
  result file, with the file's SHA-256 (see
  [results-schema.md](results-schema.md#raw)).
- CSV files follow RFC 4180: UTF-8 without BOM, CRLF line endings, a header row,
  `.` as decimal separator, fields quoted when needed, and numbers written in
  the same shortest round-trip form as the JSON. Empty fields are nulls (p95
  with fewer than 20 samples, p99 with fewer than 100).
- `latencies.csv` columns: `trial_index,iteration_index,latency_ms`.
  `summary.csv` columns: `row` (`trial` or `aggregate`), `trial_index`, then the
  latency statistics of the schema (`count`, `mean_ms`, `p50_ms`, `p95_ms`,
  `p99_ms`, `stddev_ms`, `min_ms`, `max_ms`, `inferences_per_sec`,
  `items_per_sec`) and `session_create_ms` (the trial's session creation time;
  the median for the aggregate row).

**Failed runs** also write a result, with `status: "failed"` and the `error`
object (`stage`, `message`, `code`), keeping whatever was established before
the failure (system, runtime, model, completed trials): ONNX Runtime could not
be loaded (stage `environment`, exit 4), session creation or an inference failed
(`session_creation`, `warmup` or `measurement`, exit 3), non-finite outputs or a
changed top-1 between trials (`measurement`, exit 1), an interrupt (exit
130, see [Interrupting a run](#interrupting-a-run)). Usage errors (exit 2),
including models with dynamic shapes or unsupported input types, write
nothing; neither does an `--output-dir` that is an existing file (exit 6,
checked before anything is measured). A failed result never contains made-up
values: what was not established is null (for example `config.batch_size` when
the model could not be read), with a `value_unavailable` warning. Completed trials keep their latency files; without one, `raw` is null
and there is no run folder.

If the results cannot be written, the summary has already been printed, the
error names the path and the operating system's message, and the exit code is
6. If writing a failed run's result also fails, that is reported on stderr and
the run's own exit code is kept.

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
`label`, `output_dir`, `cooldown_ms`, `load_sample_ms` and `name` (the result file name) are run settings the schema's `config`
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
  "load_sample_ms": 1000,
  "name": "mobilenetv2_fp32_b1_cpu_iobinding_t4",
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
- **Power**: power source, battery charge, battery saver, power plan (name and
  GUID) and Windows power mode (name and overlay GUID; an unknown overlay is
  shown as its GUID), plus system-wide CPU load sampled over 1 s. Doctor warns,
  without changing the exit code, when the machine runs on battery, battery
  saver is on, or the background load is above 10%, because each of them
  distorts benchmarks. It never changes a power setting.
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
| `--load-sample-ms N` | Length of the background CPU load sample, default `1000`; `0` skips it (`system.load` is null) |
| `--json` | Print the report as JSON instead of text. `tool`, `system` and `runtime` use the [results schema](results-schema.md)'s field names where they overlap; `onnxruntime_load`, `problems`, `warnings`, `result` and `exit_code` summarize the checks |

Exit code 0 when every check passes. Exit code 4 when ONNX Runtime cannot be
loaded (the report still contains every section that does not need it) or when
an ONNX Runtime/QNN module is loaded from outside the bundle directory (the
module is named). Each problem is also printed to stderr as
`error: [<kind>] <message>`, with a `hint:` line when there is a fix to suggest.
Warnings (battery power, battery saver, background load over 10%, an unknown
power mode, emulation) appear in the report as `warning: [<code>] <message>`
and do not change the exit code.

Sample output (Surface Laptop 7, development build):

```text
Build
  edgebench          0.1.0
  git commit         ee9fffbb54e7e9574f513f4a4737c5c18555dd91 (dirty)
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

Power
  source             ac
  battery            80%
  battery saver      off
  power plan         Balanced (381b4222-f694-41f0-9685-ff5bb260df2e)
  power mode         Balanced (overlay 00000000-0000-0000-0000-000000000000)
  background CPU     5.7% over 1.0 s

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
| 6 | `output_error` | Results could not be written (`edgebench run`: after the summary was printed, or before measuring if `--output-dir` is an existing file; `ort_probe`: the `--json` file exists or could not be written) |
| 130 | `interrupted` | Stopped by Ctrl+C, Ctrl+Break or closing the console (the conventional 128 + SIGINT) |

## Messages

Every error, from `edgebench` and `ort_probe` alike, goes to stderr through one
formatter (`engine/include/edgebench/diagnostics.hpp`):

```text
error: [<kind>] <message>
hint: <what to do>                (optional, at most one line)
warning: [<code>] <message>
```

`<kind>` is a stable identifier (the loader error kind, `usage_error`,
`inference_failed`, ...); for failed runs it is also the result's `error.code`,
except that ONNX Runtime errors record ONNX Runtime's own code name there (e.g.
`ORT_INVALID_ARGUMENT`) and keep it in the message. Messages from ONNX
Runtime's logger appear as `warning: [onnxruntime] ...` or
`warning: [onnxruntime_error] ...`. An exception that escapes every handler is
reported as `error: [internal_error] unexpected exception: ...` (exit 1) or,
for an ONNX Runtime exception, `error: [ort_runtime_error] ...` with its code
(exit 3). A crash (access violation and other structured exceptions) prints
`error: [internal_error] unhandled exception 0x<code> at <address>` and exits 1;
no dump is written and no Windows error dialog appears.

## Exit paths

Every way `edgebench` and `ort_probe` can end with an error. "Result" says
whether a result document is written. Tests are in
`engine/tests/exit_paths_test.cpp`, `failure_paths_test.cpp`,
`edgebench_exe_test.cpp`, `benchmark_test.cpp`, `run_config_test.cpp` and
`cli_test.cpp`; "not triggerable" marks paths that need ONNX Runtime or Windows
to fail internally and that a test cannot provoke.

`edgebench run`:

| Case | Kind | Exit | Result | Test |
|---|---|---|---|---|
| Unknown option, bad value, value out of range, option repeated | `usage_error` | 2 | no | `Cli.UsageErrorsExit2OnStderr`, `CliRun.*`, `Validate.UpperBounds` |
| No command, unknown command | `usage_error` | 2 | no | `Cli.NoCommandPrintsHelpAndFails`, `Cli.UnknownCommandPrintsHelpAndFails` |
| Model missing, not a regular file, not `.onnx` | `usage_error` | 2 | no | `Validate.ModelMustExist`, `Validate.ModelMustBeARegularFile`, `Validate.ModelMustHaveOnnxExtension` |
| Model exists but cannot be opened | `model_unreadable` | 2 | no | `ExitPaths.LockedModelExits2WithTheOsError`, `Validate.LockedModelIsUnreadable` |
| `--output-dir` is an existing file | `output_dir_not_a_directory` | 6 | no (nothing measured) | `ExitPaths.OutputDirThatIsAFileExits6BeforeMeasuring` |
| ONNX Runtime cannot be loaded | loader kind (`dll_not_found`, `version_mismatch`, ...) | 4 | yes, stage `environment` (dry run: no) | `Cli.RunLoadFailureWritesAFailedResult`, `Cli.RunDryRunNeedsOnnxRuntime` |
| ONNX Runtime environment cannot be created | `environment_creation_failed` | 3 | yes, `environment` | not triggerable |
| EP devices cannot be enumerated | `ep_enumeration_failed` | 3 | yes, `environment` / `ep_registration` | not triggerable |
| No CPU EP device | `device_selection_failed` | 1 | yes, `ep_registration` | not triggerable |
| Session options rejected | `session_options_failed` | 3 | yes, `session_creation` | not triggerable |
| Model is not a valid ONNX file | `session_creation_failed` (code e.g. `ORT_INVALID_PROTOBUF`) | 3 | yes, `session_creation` | `ExitPaths.InvalidModelExits3AndWritesAFailedResult`, `Cli.RunDryRunInvalidModelExits3WithOrtCode` |
| Session creation fails on a later trial | as reported by ONNX Runtime | 3 | yes, `session_creation`, earlier trials kept | `FailurePaths.SessionCreationFailsOnTheSecondTrial` |
| Dynamic dimension, non-float32 input, output without fixed size | `dynamic_shape`, `unsupported_input_dtype`, `unsupported_output_dtype` | 2 | no | `Cli.RunUsageErrorFromTheModelWritesNothing`, `Cli.RunDryRunRejectsDynamicAndInt64Models` |
| Model without inputs | `no_inputs` | 2 | no | not tested (no fixture) |
| Model inputs/outputs cannot be read, buffers cannot be bound | `session_metadata_failed`, `binding_failed` | 3 | yes, `session_creation` | not triggerable |
| An inference fails | `inference_failed` (code e.g. `ORT_INVALID_ARGUMENT`) | 3 | yes, `warmup` or `measurement` | `ExitPaths.RunFailureExits3AndWritesAFailedResult`, `FailurePaths.RunFailureDuringWarmUpAndMeasurement` |
| An output contains NaN or infinity | `non_finite_output` | 1 | yes, `measurement` | `Cli.RunStopsOnNonFiniteOutputAndWritesAFailedResult` |
| A trial's top-1 differs from trial 1's | `inconsistent_outputs` | 1 | yes, `measurement` | `FailurePaths.OutputsWithADifferentTop1FailTheRun` |
| ONNX Runtime loads the model but its metadata cannot be decoded | `invalid_onnx` | 1 | yes, `model_load` | not triggerable |
| The placement session fails | as reported by ONNX Runtime | 3 | yes, `session_creation`, all trials kept | `FailurePaths.PlacementFailureFailsTheRun` |
| The placement session logs no placement | `placement_unavailable` | 1 | yes, `session_creation` | not triggerable |
| Ctrl+C, Ctrl+Break, console closed | `interrupted` | 130 | yes, current stage, completed trials kept | `ExitPaths.CtrlBreakDuringARunWritesAnInterruptedResult`, `Interrupt.*` |
| Second Ctrl+C or Ctrl+Break | none | 130 | no | `ExitPaths.SecondCtrlBreakExitsAtOnce` |
| Ctrl+C or Ctrl+Break during `--dry-run` | none | 130 | no | same handler mode as `ExitPaths.OrtProbeCtrlBreakExits130WithoutWriting` |
| The result cannot be written | `write_failed` | 6 (a failed run keeps its own code) | no | `Cli.RunWithAnUnwritableOutputDirectoryExits6AfterTheSummary`, `ResultFiles.*` |
| An exception escapes | `internal_error` / `ort_runtime_error` | 1 / 3 | no | not triggerable |
| A crash (structured exception) | `internal_error` | 1 | no | `ExitPaths.UnhandledExceptionIsReportedAndExits1` |

Warnings that do not end the run: `nondeterministic_outputs`
(`FailurePaths.OutputsDifferingWithTheSameTop1AreAWarning`),
`nodes_not_on_requested_ep` (`FailurePaths.NodesOffTheRequestedEpAreAWarning`),
`threads_exceed_logical_processors`
(`ExitPaths.ThreadsAboveLogicalProcessorsWarnEverywhere`),
`background_cpu_load_high`, `power_state_changed`, `unknown_power_mode`,
`value_unavailable`.

`edgebench doctor`:

| Case | Kind | Exit | Test |
|---|---|---|---|
| ONNX Runtime cannot be loaded | loader kind | 4 | `Doctor.MissingDllIsReported`, `Doctor.WrongDllVersionIsReported`, `Doctor.CorruptDllIsReported`, ... |
| An ONNX Runtime/QNN module is loaded from outside the bundle | `module_outside_bundle` | 4 | not triggerable (the loader refuses a foreign `onnxruntime.dll` first) |
| EP devices cannot be enumerated | `ep_enumeration_failed` | 3 | not triggerable |
| Battery, battery saver, load over 10%, unknown power mode, emulation | warning only | 0 | `DoctorWarnings.*` |
| Ctrl+C or Ctrl+Break | none | 130 | same handler mode as the `ort_probe` test |

`ort_probe`:

| Case | Kind | Exit | Test |
|---|---|---|---|
| Bad arguments | `usage_error` | 2 | `ExitPaths.OrtProbeUsageErrorExits2`, `ParseArgs.Errors` |
| Model or context file not found | `usage_error` | 2 | not tested |
| `--json` file exists | `file_exists` | 6 | `OrtProbe.JsonRefusesToOverwrite` |
| `--json` cannot be written | `write_failed` | 6 | not tested |
| ONNX Runtime cannot be loaded | loader kind | 4 | `OrtProbe.LoadFailureExits4` |
| `onnxruntime.dll` not from the exe directory; an ORT/QNN module loaded from elsewhere | `foreign_module_loaded`; `module_outside_bundle` | 4 | not triggerable for the probe (the loader check is tested with `PreloadHelper.*`) |
| `--target npu` in a build without the QNN execution provider (x64) | `qnn_ep_not_bundled` | 4 | `OrtProbe.WithoutTheQnnEpOnlyTheCpuTargetWorks` |
| Output comparison or device selection fails | `device_selection_failed`, check output | 1 | manual (NPU) |
| ONNX Runtime error, e.g. strict mode refusing a session | `ort_runtime_error` with the ORT code | 3 | manual (`tools/qnn_rejection_check.py`, NPU) |
| Ctrl+C or Ctrl+Break | none | 130 | `ExitPaths.OrtProbeCtrlBreakExits130WithoutWriting` |
| A crash | `internal_error` | 1 | same handler as `ExitPaths.UnhandledExceptionIsReportedAndExits1` |
