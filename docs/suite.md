# Experiment suites (`edgebench suite`)

`edgebench suite` runs a whole experiment plan from a configuration file: it
expands the configurations, runs every one of them a given number of times in
a rotated order with a cooldown before each run, gates the start on a
pre-flight check, records everything in a manifest, and can resume after an
interruption. Every published benchmark run goes through it.

```powershell
# the plan and an estimated duration, nothing is run
.\build\arm64-release\bin\Release\edgebench.exe suite benchmarks\configs\smoke.json --dry-run
# run it
.\build\arm64-release\bin\Release\edgebench.exe suite benchmarks\configs\smoke.json
# continue an interrupted suite
.\build\arm64-release\bin\Release\edgebench.exe suite --resume benchmarks\results\local\suites\smoke_20261127T221323.347Z
```

| Option | Meaning |
|---|---|
| `CONFIG` | The configuration file. With `--resume` it is optional (default: the file the suite was started with) |
| `--dry-run` | Print the expanded, ordered plan and the estimated duration, then exit 0 without running anything |
| `--resume DIR` | Continue the suite in `DIR` (see [Resume](#resume)) |
| `--output-dir PATH` | Create the suite directory here instead of under the configuration's `output_dir` |
| `--fail-fast` | Stop at the first failed run; the remaining runs are marked `skipped`. By default a failed run is recorded and the suite continues |
| `--skip-preflight` | Development only: skip the pre-flight gate. The suite and every result from it are marked as not publishable |
| `--python PATH` | Interpreter that runs `scripts/validate_results.py` on every result (default: `.venv\Scripts\python.exe` of the repository) |

## How a suite runs

1. The configuration is read, validated and expanded (errors: exit 2, nothing
   is created).
2. The plan and the estimate are printed, the suite directory
   `<output_dir>/<name>_<UTC stamp>/` is created, and the configuration is
   copied into it.
3. The [pre-flight gate](#pre-flight-gate) runs. If it fails, no run starts
   (exit 1).
4. Every planned run, in order: the suite idles for the run's `cooldown_s`,
   then launches `edgebench run` as a **child process** of the same
   executable, with the run's output in its own directory and its stdout and
   stderr in `run.log`. Its exit code is recorded (named from the
   [exit-code table](cli.md#exit-codes)), and its result is validated with
   `scripts/validate_results.py`. A run is `done` when it exited 0 and its
   result validates; otherwise it is `failed`.
5. The summary: counts by status, wall time against the estimate, failed runs
   with exit codes and log paths, and runs whose results carry warnings (for
   example `trial_spread_high` or `background_cpu_load_high`).

The manifest is rewritten (atomically) after every change of state.

**Why one process per run.** Every run gets a real cold start (nothing cached
in the process), the QNN EP's compatibility query is only safe before the
first QNN session of a process (see [docs/npu.md](npu.md)), and ONNX Runtime
teardown problems stay inside the run that has them. The suite process itself
never loads ONNX Runtime. Every child is in a Windows job object that ends
with the suite: if the suite process is killed, its run is killed too.

**A run means the same as a run started by hand.** Each run keeps its own
trial loop (`trials`, `trial_cooldown_ms`) and writes a normal result file.
The suite only adds a `suite` block to the result (see
[Results](#results-link-back-to-the-suite)).

## The configuration file

JSON, validated against
[schemas/suite-config.v1.0.0.schema.json](../schemas/suite-config.v1.0.0.schema.json)
and then by the engine, which also checks what a schema cannot (model files,
every combination of a matrix, the rules of `edgebench run`). Relative paths
are relative to the configuration file's directory. Unknown keys are errors.

```json
{
  "suite_config_version": "1.0.0",
  "name": "smoke",
  "output_dir": "../results/local/suites",
  "models": ["conv1d_autoencoder"],
  "matrix": {
    "providers": ["cpu", "npu"],
    "precisions": ["fp32", "int8"],
    "batches": [1],
    "exclude": [{"provider": "npu", "precision": "fp32"}]
  },
  "defaults": {"warmup": 5, "iterations": 50, "trials": 2, "trial_cooldown_ms": 500},
  "repetitions": 2,
  "rotation": "rotated",
  "cooldown_s": 2,
  "preflight": {"checks": ["doctor"]}
}
```

| Key | Meaning |
|---|---|
| `suite_config_version` | `"1.0.0"` |
| `name` | Letters, digits, `-`, `_`, `.`; the suite directory is `<output_dir>/<name>_<stamp>` |
| `description` | Free text |
| `output_dir` | Where suite directories are created (`--output-dir` replaces it) |
| `models_dir` | Directory with `manifest.json` for models named by their manifest name. Default: the nearest `models/` directory above the configuration file |
| `models` | The models the configurations may use (below) |
| `configurations` or `matrix` | Exactly one of them (below) |
| `defaults` | [Run parameters](#run-parameters) for every configuration |
| `overrides` | `[{match, set}]`: run parameters for the configurations `match` selects, applied in order |
| `repetitions` | How often every configuration runs (default 1) |
| `rotation` | `rotated` (default) or `blocked` ([Run order](#run-order)) |
| `cooldown_s` | Default idle period before every run, in seconds (default 5) |
| `preflight` | `{checks, fail_on_warn}` ([Pre-flight gate](#pre-flight-gate)) |

### Models

A string names a model of `models/manifest.json` (for example
`"conv1d_autoencoder"`); the file of a configuration is the manifest entry
with that model, batch size and variant. The variant defaults to `fp32` for
precision `fp32` and to `int8` (pruned) for `int8`; give `variant` to use
another file of the model (`int8_folded`, `fp32_imported`, ...). The file on
disk must have the manifest's SHA256.

An object gives a model by its files, for models outside the manifest:

```json
{"name": "probe", "files": [{"path": "probe_int8.onnx", "precision": "int8", "batch": 1}]}
```

Either way the engine reads the file and refuses it if its stored precision or
its input's batch size is not the declared one.

### Configurations

A configuration says what is measured:

| Key | Meaning |
|---|---|
| `model`, `provider` (`cpu`, `npu`), `precision` (`fp32`, `int8`), `batch` | Required. `precision` is the stored precision of the model file |
| `variant` | Model file variant (above) |
| `npu_fp16` | NPU: compute an FP32 file in FP16 (`run --npu-fp16`) |
| `htp_performance_mode` | NPU: `run --htp-performance-mode` (default: burst) |
| `htp_graph_finalization_mode` | NPU: `run --htp-graph-finalization-mode` (default: not passed) |
| `strict` | NPU: `false` is `run --no-strict` (default `true`) |

plus any [run parameter](#run-parameters), which overrides `defaults` and
`overrides` for that configuration.

`configurations` is an explicit list of them. `matrix` is the cartesian
product of `models` (default: every listed model), `providers`, `precisions`
and `batches`; NPU configurations are further multiplied by the axes under
`npu` (`npu_fp16`, `htp_performance_mode`, `htp_graph_finalization_mode`,
`strict`, each a list; default one value: `false`, the EP default, not
passed, `true`). Then every configuration an `exclude` entry matches is
removed, and every `include` entry (a full configuration) is added.

A combination that `edgebench run` would refuse is an **error, not dropped**:
the suite never plans a run that cannot run, and never presents something
other than what was asked for. In particular FP32 on the NPU needs
`npu_fp16: true` (the NPU has no FP32 compute, so an FP32 file runs with FP16
compute or not at all), INT8 files cannot take `npu_fp16`, and NPU options or
an EP context cache on the CPU are refused. Exclude such combinations
explicitly. An `exclude` or `overrides` entry that matches nothing is an error
too (it is almost always a typo), and so are two identical configurations.

A `match` object (in `exclude` and `overrides`) holds any of the configuration
keys above; it selects the configurations whose values equal all of them
(`{}` in an override selects all).

### Run parameters

| Key | Default | Meaning (`edgebench run` option) |
|---|---|---|
| `warmup` | 20 | `--warmup` |
| `iterations` | 200 | `--iterations` |
| `trials` | 3 | `--trials` |
| `trial_cooldown_ms` | 2000 | `--cooldown-ms`: idle before every trial of the run |
| `load_sample_ms` | 1000 | `--load-sample-ms` |
| `threads` | 4 | `--threads` |
| `binding` | `iobinding` | `--binding` |
| `init` | `cold` | How sessions are initialized (below) |
| `init_breakdown` | `false` | NPU: `--init-breakdown` |
| `profile_accelerator` | `false` | NPU: `--profile-accelerator` (the profiled pass) |
| `profile_iterations` | 50 | NPU: `--profile-iterations` (with `profile_accelerator`) |
| `cooldown_s` | 5 (or the top-level `cooldown_s`) | The suite's idle period before this run, in seconds |

`init`:

- `cold`: every session is compiled from the model (`--ep-cache off`); the
  only mode on the CPU.
- `build`: the run rebuilds its EP context cache entry, then creates its
  timed sessions from it (`--ep-cache build`).
- `cached`: right before the timed run, the suite runs a short separate
  `--ep-cache build` run (one trial of one iteration, no warm-up, no idle
  periods) that builds the cache entry; the timed run then loads it
  (`--ep-cache use`, a cache hit). Its result is in the run's
  `cache_build/` directory and in the manifest.

The suite's EP context cache is `<suite dir>/qnn-context`, so a suite never
uses entries built outside it.

Parameters are resolved as: built-in defaults, then the top-level
`cooldown_s`, then `defaults`, then every matching `overrides` entry in
order, then the keys written in the configuration itself.

## Run order

Every configuration runs `repetitions` times. With `rotation: "blocked"` the
repetitions of a configuration run back to back (configuration 0 R times,
then configuration 1, ...): useful only for comparison.

With `rotation: "rotated"` (the default), round r runs every configuration
once, in the order of row r of a balanced Latin square (a Williams design):
the first row is 0, 1, N-1, 2, N-2, ..., and row r adds r to every entry
(mod N); for an odd N, rows N to 2N-1 are rows 0 to N-1 reversed. Over N
rounds (2N for an odd N) every configuration runs first after the pre-flight
equally often, and every configuration directly follows every other one
equally often, so neither warm-up after a cooldown nor a particular
neighbour is always charged to the same configuration. Rounds beyond that
repeat the pattern. The order depends only on the configuration file; there
is no random seed. The round that ends one repetition and the round that
starts the next are not balanced against each other.

The scheduler that orders trials of several configurations inside one
process (`schedule_trials`, docs/architecture.md) is a simpler cyclic
rotation; the suite works at the level of whole runs.

## Run IDs

Every planned run has an ID: the SHA256 of its fully resolved configuration
(the model file's SHA256, provider, stored and compute precision, batch size,
strict mode, the effective HTP performance mode, the compile-relevant QNN EP
options (the same definition the EP context cache key and the batch
equivalence summary use), every run parameter including `cooldown_s`, and the
fixed input settings), serialized canonically (keys sorted, no whitespace,
UTF-8), with the repetition index added. The configuration ID is the same
hash without the repetition. Tables show the first 12 hex digits, and a run's
output directory is `runs/<those 12 digits>/`.

IDs do not depend on the suite's name, on the run order or on the rotation:
the same configuration in two suites has the same IDs. A changed model file
changes them.

## Pre-flight gate

Before the first run (and again on every resume) the suite runs its gate: an
ordered list of checks, each with a name, a status (`pass`, `warn`, `fail`),
a summary and details, all recorded in the manifest. Any `fail` stops the
suite before its first run (exit 1); with `preflight.fail_on_warn` a `warn`
does too.

| Check | What it does |
|---|---|
| `doctor` | Runs `edgebench doctor --json --quick` as a child process and saves the report as `preflight/doctor-<n>.json`. `fail` if doctor exits non-zero or prints no report; `warn` if it passes with warnings (for example the power mode or background load); details hold every doctor check's status and the versions doctor reports (ONNX Runtime, QNN EP, QAIRT SDK) |

`preflight.checks` lists the checks to run (default `["doctor"]`). A suite is
publishable only if its gate included every required check (currently
`doctor`) and never was skipped.

`--skip-preflight` is for development. The manifest records it
(`preflight_ever_skipped`, which never becomes false again for that suite),
`publishable` is `false`, and every result from that suite has
`suite.preflight_skipped: true` and a `preflight_skipped` warning; the
validator reports `result_not_publishable` for such a result, so it fails
`validate_results.py --strict`, which published results are checked with.

## The manifest

`<suite dir>/suite-manifest.json`, validated against
[schemas/suite-manifest.v1.0.0.schema.json](../schemas/suite-manifest.v1.0.0.schema.json)
and by `scripts/validate_suite.py`. It holds:

- the suite: name, configuration file path, the configuration itself and its
  SHA256 (of the canonical JSON);
- `status`: `running`, `completed` (every run was attempted; some may have
  failed), `interrupted` or `aborted` (pre-flight failed, or `--fail-fast`);
- the tool (version, git commit, dirty flag, build type, results schema
  version) and the versions doctor reported;
- `publishable` and the reasons when not (pre-flight skipped or incomplete,
  or built from a working tree with uncommitted changes);
- every pre-flight attempt with every check, every invocation (start, end,
  resumed, options, exit code), the estimate, counts by status;
- every planned run: run ID, configuration ID, label, order index,
  repetition, the resolved configuration, status (`pending`, `running`,
  `done`, `failed`, `skipped`), attempts, exit code and its name, start and end
  time, duration, the cooldown before it, the estimate, its output directory,
  result file and its SHA256, log file, the reason it failed, the validation
  outcome and issues, the codes of the result's warnings, the cache-building
  run (`init: "cached"`), and the output of discarded attempts.

```text
<suite dir>/suite-manifest.json
<suite dir>/suite-config.json          the configuration file as read
<suite dir>/preflight/doctor-<n>.json  one doctor report (and log) per attempt
<suite dir>/runs/<short id>/           a run: result, latencies, run.log, validation.txt
<suite dir>/runs/<short id>/cache_build/  its cache-building run (init "cached")
<suite dir>/qnn-context/               the suite's EP context cache
<suite dir>/discarded/<short id>-<n>/  output of an interrupted or failed attempt
```

`scripts/check_matrix.py` accepts a suite directory as well as a
`scripts/run_matrix.py` directory.

## Resume

`edgebench suite --resume DIR` continues a suite:

- It refuses (exit 2, nothing changed) if the configuration file's SHA256
  differs from the manifest's, if the planned runs differ (a model file
  changed), or if the edgebench version or git commit differs. After the
  pre-flight gate it also refuses versions (ONNX Runtime, QNN EP, QAIRT SDK)
  other than the ones doctor reported before.
- It runs the pre-flight gate again: a resumed suite must still pass it.
- A `done` run is kept if its result file still has the recorded SHA256 and
  passes the validator; otherwise it runs again.
- A `running` run was interrupted mid-run: its partial output is moved to
  `discarded/` and it runs again. `failed` runs run again (their output is
  discarded too); `pending` and `skipped` runs run.

**Ctrl+C** (or Ctrl+Break, or closing the console) during a run is forwarded
to the run as Ctrl+Break; it stops at its next safe point (if it has not
stopped 30 s later, it is terminated). Its partial output is discarded, it is
marked `pending`, the manifest is written and the suite exits 130. During a
cooldown the suite stops at once. A second Ctrl+C ends the suite immediately;
the manifest then still says `running` for that run, which `--resume`
handles.

## Results link back to the suite

Every result of a suite run has a top-level `suite` block (results schema
1.2.0; absent from runs started by hand):

| Field | Meaning |
|---|---|
| `suite_name`, `suite_config_hash` | The suite and its configuration's SHA256 |
| `run_id` | The suite's run ID (not the result's own `run_id`, a UUID) |
| `repetition`, `order_index` | 0-based |
| `cooldown_s_before` | The suite's idle period before the run |
| `preflight_skipped` | `true` if the suite ran with `--skip-preflight` |

The suite passes these to `edgebench run` with hidden command-line options
(not environment variables), so a run started by hand never has the block.

## Duration estimate

`--dry-run` and every real start print an estimated duration, per run and in
total. It errs toward too long, for planning unattended runs. Per run it
adds: the process start and fixed costs (3 s, 4 s on the NPU), per trial the
trial cooldown, one timed session creation (cold, or cached for an EP
context) and `warmup + iterations` inferences (plus the init breakdown's
three idles and three session creations with `init_breakdown`), the load
sample when `trial_cooldown_ms` is 0, the placement session, the cached vs
compiled output check and context build for `build` / `cached`, the
cache-building run for `cached`, the profiled pass (one session plus
`warmup + profile_iterations` inferences at three times the latency, at least
15 ms more each), 1.5 s for validating the result, and the cooldown. Then
×1.25. The suite adds 5 s for the pre-flight gate (×1.25).

The latency and session times come from earlier results under
`benchmarks/results/` (local and published) with the same model SHA256,
provider, compute precision and batch size: the slowest mean latency and the
longest session times among them. A configuration without such a result uses
a heuristic from the model file's size (CPU: 1 ms + 1 ms per MB and batch
item per inference, 0.5 s + 0.05 s per MB per session; NPU: 1 ms + 0.2 ms
per MB and batch item, 3 s + 0.1 s per MB and batch item per compiled
session, 1.2× that to build a context, 1 s to load one). The output says how
many configurations were estimated each way.

## Exit codes

| Code | When |
|---|---|
| 0 | Every run is `done` (also a `--dry-run`) |
| 1 | A run failed (or was skipped by `--fail-fast`), or the pre-flight gate failed |
| 2 | The configuration is invalid, the validator cannot run, or `--resume` found a changed configuration, plan, build or versions |
| 6 | The suite directory or the manifest could not be written |
| 130 | Interrupted (resumable) |

## Checking suite files

```powershell
.\.venv\Scripts\python.exe scripts\validate_suite.py benchmarks\configs <suite dir>
.\.venv\Scripts\python.exe scripts\check_matrix.py <suite dir>
```

`validate_suite.py` checks configuration files against their schema, and
manifests against theirs plus: the configuration hash, counts, unique run IDs,
contiguous order, complete repetitions, the rotation (each round runs every
configuration once; blocked runs are one block per configuration), the
pre-flight and publishability flags, and that every `done` run's result exists,
has its SHA256 and names this suite, run, repetition and order
(`--no-files` skips the result files).
