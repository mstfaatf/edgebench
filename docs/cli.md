# Command-line interface

```text
edgebench run --model <file.onnx> [options]    benchmark a model
edgebench doctor [--json] [--quick] [--prune-cache] [--ep-cache-dir PATH] [--load-sample-ms N]
                                                check everything a run depends on
edgebench --version | --help
edgebench <command> --help
```

- Results and JSON go to stdout; errors and diagnostics go to stderr. There are
  no color codes.
- `--help` and `--version` never load ONNX Runtime, so they work even when the
  DLLs are missing. `--version` prints the edgebench version, the results
  schema version and the ONNX Runtime version the build requires. Running `edgebench` with no command, or with an unknown
  command, prints the help to stderr and exits with code 2.
- Commands that need ONNX Runtime load the bundled `onnxruntime.dll` from the
  executable's directory and verify it first; see
  [setup.md](setup.md#how-the-executables-find-onnx-runtime).

## `edgebench run`

Benchmarks one model with one configuration, prints a summary, and writes the
result as a results-schema 1.2.0 JSON document plus CSV files (see
[Result files](#result-files)). `--dry-run` creates the real session and checks
one inference without measuring or writing anything.

| Option | Default | Constraints / meaning |
|---|---|---|
| `--model PATH` | (required) | Must exist, be a regular file that can be opened for reading, and end in `.onnx` (any case) |
| `--provider NAME` | `cpu` | `cpu` (ONNX Runtime's CPU EP) or `npu` (the QNN EP on the NPU; ARM64 builds only, see [NPU runs](#npu-runs)) |
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
| `--no-strict` | off | NPU only: allow nodes the NPU cannot run to fall back to the CPU (recorded as `config.strict_mode` false, with `strict_mode_off` and `cpu_fallback_nodes` warnings). By default such a session is refused (exit 5) |
| `--npu-fp16` | off | NPU only: run an FP32 model with FP16 compute (`enable_htp_fp16_precision` = `1`). Required for FP32 models, a usage error for INT8 models |
| `--htp-performance-mode MODE` | `burst` | NPU only: the QNN EP's `htp_performance_mode`: `burst`, `balanced`, `default`, `high_performance`, `high_power_saver`, `low_balanced`, `low_power_saver`, `power_saver`, `extreme_power_saver`, `sustained_high_performance` |
| `--htp-graph-finalization-mode MODE` | not passed | NPU only: the QNN EP's `htp_graph_finalization_optimization_mode`, the values it documents: `0` (EP default), `1` (faster preparation, less optimized graph), `2` (longer preparation, more optimized graph), `3` (longest preparation). Passed and recorded only when given; without it the EP's default applies. See [npu.md](npu.md#provider-options) |
| `--ep-cache MODE` | `off` | NPU only: EP context cache. `off`: every session is compiled from the model. `build`: delete this key's cache entry, build it once before the trials, create every timed session from it. `use`: create the timed sessions from the entry whose key matches; build it first if there is none. See [EP context cache and initialization](#ep-context-cache-and-initialization) |
| `--ep-cache-dir PATH` | `%LOCALAPPDATA%\EdgeBench\qnn-context` | NPU only: the cache directory (created if missing; must be writable, exit 6 otherwise), one subdirectory per cache key. Only with `--ep-cache build`/`use` or `--init-breakdown` |
| `--init-breakdown` | off | NPU only: in every trial, also time a cold, a context-building and a cached session creation, each after the trial's idle period, before the timed session |
| `--profile-accelerator` | off | NPU only: after the trials and the placement session, run one more session with QNN profiling and record the accelerator time per inference in `timing.profiled_pass`; never part of the timed results. See [Accelerator time](#accelerator-time---profile-accelerator) |
| `--profile-iterations N` | `50` | 1 to 1,000,000. Inferences of the `--profile-accelerator` pass, after `--warmup` unrecorded ones. Only with `--profile-accelerator` |

All validation problems are reported together, one per line, as
`error: [usage_error] --<option>: <message>` (`[model_unreadable]` when the
model exists but cannot be opened, e.g. because another process holds it
exclusively; the message is the operating system's), followed by one
`hint: Run 'edgebench run --help' for the options.` line. The upper bounds keep
a run's memory and duration finite (one million iterations store 8 MB of
latencies per trial); they are not measurement recommendations.

### NPU runs

`--provider npu` registers the bundled QNN execution provider, selects the one
QNN device of type NPU (exactly one must exist; exit 3 otherwise), checks that
the HTP backend files `QnnHtp.dll` and `QnnHtpPrepare.dll` are in the bundle
(exit 4 otherwise: the EP lists the NPU even without them), and attaches every
session to that device. The trial loop, the IoBinding default and the timing are
the same as for the CPU.

edgebench passes exactly these settings and records them in the result; every
other QNN EP option stays at the EP's default:

| Where | Key | Value |
|---|---|---|
| `config.ep_options` | `backend_path` | absolute path of the bundled `QnnHtp.dll` |
| `config.ep_options` | `htp_performance_mode` | `--htp-performance-mode` (default `burst`) |
| `config.ep_options` | `enable_htp_fp16_precision` | `1`, only with `--npu-fp16` |
| `config.ep_options` | `htp_graph_finalization_optimization_mode` | `--htp-graph-finalization-mode`, only when given |
| `config.session_config` | `session.disable_cpu_ep_fallback` | `1` (strict mode), `0` with `--no-strict` |
| `config.session_config` | `ep.context_enable` | `0`: the session writes no EP context |
| `config.session_config` | `ep.context_embed_mode` | `0`, only with `--ep-cache build`/`use` or `--init-breakdown` (see below) |

The compute precision follows from the model: an INT8 QDQ model runs as stored
(`config.requested.compute_precision` `int8`); an FP32 model needs `--npu-fp16`
(`fp16`, next to `model.stored_precision` `fp32` and a `model.precision_notes`
entry), because the NPU has no FP32 compute. Without the flag it is refused with
`npu_fp32_needs_fp16` (exit 2); other stored precisions are
`npu_precision_unsupported`.

Session failures are told apart by what ONNX Runtime and the QNN EP report:
a backend the QNN EP could not load (`qnn_backend_load_failed`, exit 4); in
strict mode, an `ORT_FAIL` whose message says CPU fallback was disabled
(`strict_mode_refusal`, exit 5, with the unsupported nodes the QNN EP logged);
anything else exit 3. Node placement comes from a separate verbose session
after the trials (as for the CPU), together with the QNN EP's node validation
(`execution.qnn_validation`, `verification_method` `combined`). In strict mode
a node placed anywhere but on the QNN EP is an internal error (exit 1). After
every QNN session creation the loaded ONNX Runtime/QNN modules are checked
again: the HTP backend DLLs load at that point.

### EP context cache and initialization

Compiling a model for the HTP takes about 2 s; a session created from an EP
context (the precompiled graph the QNN EP writes) takes about 0.3 s. With
`--ep-cache build` or `use` the timed sessions are created from a context
(`timing.session_init.ep_context_state` `cached`); with `off` (the default) they
are compiled every time (`cold`). The latency of the steady-state iterations is
measured on the timed session in every mode, the same way.

**The cache.** One subdirectory of `--ep-cache-dir` per cache key, named by the
key id, holding `<model stem>_ctx.onnx`, its context binary
`<model stem>_ctx_qnn.bin` and `edgebench-cache.json` (key fields, key id,
creation time, every file with its size and SHA256). Entries are built in a
temporary subdirectory and moved into place when complete. edgebench only ever
deletes files listed in a manifest it wrote, and only in direct subdirectories
of the cache directory; a directory for the key without such a manifest stops
the run (`ep_cache_entry_foreign`, exit 6) instead of being touched.

**The key.** The key id is the SHA256 of the key fields serialized as JSON with
sorted keys, no whitespace and UTF-8 text (Python:
`json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`),
recorded as `config.ep_context_cache.key` and `key_id` (the validator recomputes
it). The fields: the model file's SHA256, the ONNX Runtime version, the QNN EP
version, the QNN SDK version reported at run time (`runtime.qnn.sdk_version`,
see [What the QNN EP reports](#what-the-qnn-ep-reports)), the HTP backend file's
version (`QnnHtp.dll` ProductVersion), every EP option
passed (including the backend path), every session config entry that affects
compilation (all but `ep.context_enable` and `ep.context_file_path`: strict
mode and the embed mode), the graph optimization level, and the device (type,
vendor id, device id). Changing any of them gives a new key.

**Invalidation.** An entry is stale when it is for the same model (SHA256) and
the same configuration (the compile-relevant EP options, i.e. all but
`backend_path`, `htp_performance_mode` and `enable_htp_fp16_precision`, the
session config and the graph optimization level) but its environment key
fields differ: the ONNX Runtime, QNN EP or QNN SDK version, the HTP backend
file version or the device. Such an entry can never be loaded again. With
`build` or `use`, stale entries of the run's model and configuration are
deleted first; the deleted files are listed in
`execution.verification_details.deleted_stale_cache_files`. Entries for other
options of the same model (another `--htp-graph-finalization-mode`, another
performance mode) are not stale and stay, so runs alternating between them
keep their cache hits. `build` also deletes the entry of the current key
before building it again (`state` `regenerated`; `generated` when nothing
existed for the key). `use`
loads a matching entry (`state` `loaded`, `hit` true) after checking every file
against its manifest; a file that is missing, truncated or changed fails the run
(`ep_context_corrupt`, exit 3), and so does a context ONNX Runtime cannot load
(`ep_context_load_failed`, exit 3): the message says to rebuild it with
`--ep-cache build`, and nothing is rebuilt silently. Without an entry, `use`
builds one first (`hit` false).

**Verified equal outputs.** Before the first trial of a `build` or `use` run,
the engine runs its generated input (the same SplitMix64 input as the timed
sessions) once through a session compiled from the model and once through a
session created from the context, and compares the output bytes. A difference
fails the run (`ep_context_outputs_differ`, exit 1); the result records
`verified_equal_outputs`. There is no tolerance.

**Embed mode.** Contexts are written with `ep.context_embed_mode` `0`: the
context binary in its own file next to a small context model. That is the mode
whose cached initialization and output equality were measured
([feasibility.md](feasibility.md)), and the binary is loaded from its file
rather than parsed out of the ONNX file (on MobileNetV2 INT8 the median cached
initialization was 336 ms with mode 0 and 368 ms with mode 1). Context sessions
load `QnnSystem.dll`, so with the cache or `--init-breakdown` it must be in the
bundle too (exit 4 otherwise).

**`--init-breakdown`.** Each trial times three more session creations before
its timed session, each after the same idle period as the trial
(`--cooldown-ms`): `cold_ms` (from the model, no context), `context_build_ms`
(a session that writes a context into a scratch directory of that trial: a full
build every trial) and `cached_init_ms` (a session from that scratch context,
which is then deleted). They go into `timing.trials[].session_init` and their
nearest-rank median, minimum and maximum into `timing.session_init.summary`.
Without `--init-breakdown` a trial records only its timed session's creation,
as `cold_ms` (cache off) or `cached_init_ms` (cache on).
`timing.session_init.session_create_ms` is always the median creation time of
the timed sessions; `cold_ms` and `cached_ms` list every creation of that kind
in order, and `context_build_ms` is the build of the shared cache entry before
the trials (null if none was built).

The HTP architecture is not a key field: it is only known from a compiled
context (below), after the key was needed. The device fields of the key and the
EP's own compatibility check on every context load cover a change of hardware.

**Compatibility.** The QNN EP stores compatibility info in every context model
it compiles (`ep_compatibility_info.QNNExecutionProvider`, e.g.
`v2:6:2.52.0:5.52.0:73:0:0:0`: format version, backend id, SDK version, backend
API version, HTP arch, SoC model option, VTCM size option, user-driver flag).
With `build`/`use` the result records it in `config.ep_context_cache.compatibility`
with the EP's verdict for the selected NPU (`optimal`, `prefer_recompilation`,
`unsupported`). The verdict is null when it was not asked for: the QNN EP 2.8.0
crashes when ONNX Runtime asks it to rate compatibility info after any QNN
session existed in the process, so it is only asked before the first session,
which happens when the cache entry already exists (`hit` true).

Every session from a context logs that the QNN EP rates it "supported but
suboptimal" and suggests recompiling; the result gets an
`ep_context_recompilation_suggested` note. The reason, read from the EP's
source and confirmed with its verdict on varied compatibility strings: the EP
rates a context optimal only if the VTCM size stored in it equals the device's,
and without the `vtcm_mb` option it stores 0. Compiling with `vtcm_mb` set to
the device's VTCM size (8 MB on the Snapdragon X Plus) makes the verdict
`optimal`, with bit-identical outputs and the same cached initialization and
latency, but the engine has no reported value to set it from (a larger value
than the device's makes the context unusable), so it is not set. The note is
harmless: the outputs of the context are verified bit-identical to a compiled
session.

### What the QNN EP reports

`runtime.qnn` records the QNN values as the QNN EP reports them while the run
is in progress, never derived from file names, package versions or a table;
`runtime.qnn.value_sources` says where each one came from, and a value nothing
reports is `null` with a `value_unavailable` warning:

| Field | Value (this machine) | Source (`value_sources`) |
|---|---|---|
| `backend_build` | `v2.52.0.261104115025` | `qnn_ep_log_backend_build_id`: the backend's build id, which the QNN EP logs when it sets up its backend. edgebench asks the EP to rate a fixed compatibility string once per process, before any session, with the log level raised, and reads the id from that log |
| `sdk_version` | `2.52.0` | the same: the `major.minor.patch` of the build id |
| `htp_arch` | `v73` | `ep_compatibility_info`: the compatibility info the QNN EP stores in every EP context it compiles. Only known when the run built or loaded a context (`--ep-cache build`/`use`, or `--init-breakdown`); null otherwise |
| `soc_model` | null | not reported: the QNN EP keeps only its `soc_model` option (0) in the context |

`runtime.qnn.backend_file_version` is a separate fact: the version resource of
the `QnnHtp.dll` that was loaded (source `loaded_module_version_resource`). The
validator checks that `sdk_version` is the `major.minor.patch` of
`backend_build` and warns if the build differs from the file's product version.

The QNN EP 2.8.0 crashes (access violation) if ONNX Runtime asks it to rate
compatibility info after any QNN session has existed in the process, so the
build id is read once per process before the first session and kept, and the
cache verdict (`config.ep_context_cache.compatibility.verdict`) is only asked
for before the first session.

### Accelerator time (`--profile-accelerator`)

`--profile-accelerator` (NPU only; a usage error with `--provider cpu`) adds
one more session after the trials and the placement session: created like the
timed sessions (from the model, or from the EP context when the cache is used)
with the QNN EP's `profiling_level` `basic` and a profile file in a temporary
directory, it runs `--warmup` unrecorded and `--profile-iterations` (default
50) recorded inferences and is then released, which makes the QNN EP write its
profile. edgebench reads the per-inference "Accelerator (execute) time" from it
(the HTP's own share of an inference, in microseconds) and records
`timing.profiled_pass`: `accelerator_ms` (count, median, minimum, maximum), the
level, the inference counts, every execute event of the profile, the
initialization events (graph finalization when compiled, the loading of the
context binary from an EP context) and the profile files, copied into the run
directory with their sizes and SHA256 (`qnn_profile.csv` and the binary log for
Qualcomm's profile viewer). The modules are checked while that session exists
(stage `profiled_session`); a failure of the pass fails the run with stage
`profiled_pass`.

Nothing from the profiled session enters the timed results: profiling adds
host-side work to every inference of that session, so only the
backend-reported accelerator time is used, never its wall time
([methodology.md](methodology.md#accelerator-time)). Warnings:
`accelerator_time_exceeds_latency` (the median accelerator time is above the
timed median latency, so one of the two is wrong) and
`profiled_pass_incomplete` (the profile holds fewer inferences than were run).
How to read accelerator time against latency is in
[npu.md](npu.md#accelerator-time-and-latency).

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
doing it last means it cannot warm up any timed session), the optional
profiled accelerator pass runs (`--profile-accelerator`), and the power state
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
changed, the result gets a `power_state_changed` warning. Every load sample (the
idle periods and each trial's measured iterations) counts other processes
only: edgebench's own CPU time over the window is subtracted. If any sample of
at least 0.5 s shows more than 10%, the result gets a `background_cpu_load_high` warning with
the highest value, when it occurred and the top other processes by CPU share,
and every such sample is recorded in
`execution.verification_details.background_load_samples` with its
attribution: the top other processes (image name and share only, no command
lines), edgebench's own share and the busy time no process accounts for
(interrupts, DPCs). Shorter samples are not judged: the system and process
counters advance in clock ticks (15.6 ms by default), so over the measured
iterations of a fast model (tens of milliseconds) a single tick of another
process is several percent. Their values still appear in the summary, and the
idle samples before every trial (1 s by default) catch sustained load.

System-wide CPU utilization is also sampled across each trial's measured
iterations (`GetSystemTimes` before the first and after the last iteration,
outside the timed brackets). The benchmark's own threads are busy then, so this
process's CPU time over the same window (`GetProcessTimes`, all threads) is
subtracted first: what remains is the load from other processes, as a fraction
of the window times the number of logical processors. If any trial's value is
above 10%, the result gets a second `background_cpu_load_high` warning naming
the trial ("during trial N of M"). These during-trial values appear in the
summary but not in the result document: `system.load` stays the pre-run sample.

If the trial medians (p50) are spread out, i.e. their coefficient of variation is
above 10% or the slowest trial median is more than 1.25 times the fastest, the
result gets a `trial_spread_high` warning (path `/timing/aggregate/trial_spread`):
the trials disagree, so the aggregate statistics alone do not describe the run.
The spread is judged on the medians because a single slow iteration moves a
trial's mean but not its median; such spikes show in each trial's max and p99.
`timing.aggregate.trial_variation` (the spread of the trial means) is kept as
data, and `timing.aggregate.trial_spread` records the medians' spread, the
thresholds and the verdict (`judged_on` `trial_p50`).

Right after the provider is created (for the NPU: after the QNN EP library is
registered), after every session is created, and while the node placement
session is alive, the run checks that each ONNX Runtime and QNN module in the
process comes from the bundle directory (the directory of the loaded
`onnxruntime.dll`). Every check is recorded in
`execution.verification_details.module_checks`. A module from anywhere else
stops the run with exit code 4 and a failed result (`error.code`
`module_outside_bundle`; stage `ep_registration` for the first check,
`session_creation` for the others) that names the module and its path.

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

### Batch equivalence

A model with batch size N > 1 does not always compute, per item, what the
batch-1 file of the same model computes: on the NPU the INT8 image models at
batch 8 give different outputs, and on the CPU the FP32 MobileNetV2 and
EfficientNet-Lite0 files at batch 4 and 8 differ in the last bits. `run` looks
the model up by its SHA256, the provider, the compute precision and the
compile-relevant EP options (every entry of `config.ep_options` except
`backend_path`, `htp_performance_mode` and `enable_htp_fp16_precision`; none
by default) in `batch_equivalence.json` (the file in the model's directory or the nearest
parent directory that has one: `models/batch_equivalence.json`, written by
`scripts/make_references.py`; see [models.md](models.md#batch-equivalence)) and
records the answer in `execution.batch_equivalence`: whether the outputs are
bit-identical to batch 1's (`bitwise_equal_to_b1`) and, if not, whether any
prediction changes (`predictions_equal_to_b1`: top-1 on the 1,000 eval-quick
images, or the anomaly flags of the autoencoder's test split).

- Predictions differ: a `batch_not_equivalent` warning. The latency is that of
  a different computation and must not be averaged or compared with batch-1
  results as the same one.
- Only the bits differ: a `batch_rounding_only` note (in `notes`, not a
  warning). The result may be compared with batch 1.
- Not listed: `not_available` with a `value_unavailable` warning.
- Listed, but only with other compile-relevant EP options than the run's (e.g.
  `--htp-graph-finalization-mode 2`): `not_available` with a `value_unavailable`
  warning and a `batch_equivalence_options_not_covered` note. Facts measured
  with other options are never reused: options can change how the backend
  compiles the graph. `--htp-graph-finalization-mode 1` has its own entries for
  the INT8 image models.

The summary prints a `Batch` line for batch sizes above 1, and the notes after
the warnings.

### Output

Progress (`idle 2000 ms before trial 1 of 3 ...`, `trial 1 of 3: ...`,
`recording node placement`) and warnings go to stderr; the summary goes to
stdout. Times are milliseconds. For FP32 MobileNetV2 with the defaults on a
Surface Laptop 7 (plugged in, *Balanced* power mode, with an editor and a
terminal open):

```text
Model       mobilenetv2_fp32_b1.onnx (fp32, 13993244 bytes, sha256 bcf75698a82b)
Provider    CPUExecutionProvider, 4 intra-op threads, binding iobinding_preallocated, graph optimization enable_all
Trials      3 x 200 measured iterations (20 warm-up, 2000 ms cooldown), batch size 1
Input       random
Session     create ms: 65.6 50.8 54.6  (median 54.6)
First run   7.734 ms (first warm-up iteration of trial 1)
Power       ac, mode Balanced, plan Balanced, battery saver off, battery 80%
CPU load    pre-run 7.1% over 2.0 s, highest cooldown 7.5% (before trial 2), highest during trials 9.1% (trial 1, own process excluded)
Memory      working set 39.2 MiB after the last trial, peak 64.5 MiB, session-create delta +21.0 MiB (median) (process only)

ms               mean        p50        p95        p99        min        max inferences/s
trial 1        4.3883     4.0223     4.7335    15.0211     3.6030    26.1895        227.9
trial 2        4.0326     3.9751     4.4864     4.7538     3.5880     4.8543        248.0
trial 3        4.0293     4.0011     4.4871     4.7743     3.5916     5.1914        248.2
aggregate      4.1501     4.0038     4.6084     5.2121     3.5880    26.1895        241.0

Trial means 4.0293 .. 4.3883 ms, CV 4.97%
Placement   55 nodes: CPUExecutionProvider 55 (all on the requested EP)
```

- `Session`: creation time of each trial's session, and their median.
- `First run`: trial 1's first warm-up iteration.
- `Power`: power source, Windows power mode, power plan, battery saver and
  battery charge at the start of the run.
- `CPU load`: the CPU load of other processes during the idle period before trial 1
  (`system.load`), the highest value seen during the cooldowns, and the highest
  background load during a trial's measured iterations (`highest during trials
  N% (trial k, own process excluded)`).
- `Memory`: the process working set after the last trial's session was
  released, its peak, and the median change of the working set across session
  creation. Process memory only; memory used by drivers or on an accelerator is
  not included.
- One row per trial, then `aggregate` over all measured iterations pooled.
- `Trial means`: range of the trial means and their coefficient of variation.
- `Trial p50s`: the same for the trial medians, on which `trial_spread_high` is
  judged.
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
<name>_<stamp>.json            the result document (results schema 1.2.0)
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
  "results_schema_version": "1.2.0",
  "model": {
    "path": "C:\\dev\\edgebench\\models\\mobilenetv2\\mobilenetv2_fp32_b1.onnx",
    "file_name": "mobilenetv2_fp32_b1.onnx",
    "size_bytes": 13993244,
    "sha256": "bcf75698a82b096f63ce9d2048b0f50cb152ba3ea60882cdac06be3c12b6b674",
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

Checks everything a run depends on, before you trust one. Run it on every new
machine and before any run whose numbers you want to keep. Each check reports
`PASS`, `WARN`, `FAIL` or `N/A` (not applicable to this build or machine) with
one summary line, details below it, and for a warning or failure what is wrong
(`[code]`) and what to do (`hint:`). Doctor never changes a setting and, without
`--prune-cache`, never deletes anything.

| Option | Meaning |
|---|---|
| `--json` | Print the report as JSON ([schema](../schemas/doctor-report.v1.0.0.schema.json); check it with `scripts/validate_doctor.py`) instead of text |
| `--load-sample-ms N` | Length of the background CPU load sample, default `1000`; `0` skips it (`background_load` is N/A) |
| `--ep-cache-dir PATH` | The EP context cache directory to check, default `%LOCALAPPDATA%\EdgeBench\qnn-context` |
| `--prune-cache` | Delete the stale cache entries (see `ep_cache_dir`), with the same manifest-only deletion as runs: only files a manifest edgebench wrote lists |
| `--quick` | Skip the three checks that use the NPU backend (`backend_versions`, `npu_probe`, `npu_context`, reported N/A, `skipped (--quick)`): about 1 s instead of about 8 s |

The checks, in report order. They also run in an order that respects the QNN
execution provider: the session-free backend query (`backend_versions`) runs
before the first QNN session in the process, because the EP crashes when asked
about compatibility info after one (see [architecture.md](architecture.md)).

| Check | What it checks | Warn / fail, and what to do |
|---|---|---|
| `ort_bundle` | `onnxruntime.dll` loads from the executable's directory (long path, file version, C API version), every pinned ONNX Runtime file matches its size and SHA256 in [`cmake/bundle-pins.json`](../cmake/bundle-pins.json), and every ONNX Runtime / QNN module loaded by all the checks comes from the bundle | Fail (exit 4): ONNX Runtime cannot be loaded (the loader's error kind, e.g. `dll_not_found`, `version_mismatch`), `bundle_file_mismatch`, `module_outside_bundle`. Rebuild, or rerun `scripts/fetch_deps.ps1 -Force` and rebuild. Warn: `process_emulated` (use the native build) |
| `qnn_bundle` | The QNN EP plugin, `QnnHtp.dll`, `QnnHtpPrepare.dll`, the HTP stubs and `QnnSystem.dll` against their pins (and their file versions); the plugin registers with ONNX Runtime | N/A in x64 builds (no QNN EP). Fail (exit 4): `qnn_bundle_mismatch` (a missing or changed file), or the registration's error kind |
| `backend_versions` | The backend build id and SDK version the QNN EP reports before any session, as recorded in results (`runtime.qnn`), against the SDK version the QNN EP release declares and the version resource of `QnnHtp.dll` | Never fails. Warn: `qnn_sdk_version_disagreement`, `qnn_backend_version_disagreement` (check that the bundle comes from the pinned release), `backend_build_unavailable`. N/A without an NPU |
| `devices` | Every EP device ONNX Runtime enumerates, and the NPU an NPU run would select (the single `QNNExecutionProvider` device of type NPU) | Fail (exit 3): `multiple_npu_devices`, `ep_enumeration_failed`. Warn: `no_qnn_ep_devices`, `no_npu_device` (CPU runs still work) |
| `npu_probe` | Two tiny probe models, built into the executable ([`engine/resources/doctor/`](../engine/resources/doctor/)): a 3x3 convolution as static INT8 QDQ, and in FP32 run with FP16 compute (`--npu-fp16` semantics). Each runs once in a strict NPU session, with the generated input of `run` (SplitMix64, seed 0), and its output must be byte-identical to the expected output made with the Python reference sessions (`scripts/make_probe_references.py`). The modules are checked while each session exists | Fail: a session failure with its `run` exit code (backend 4, strict refusal 5, other ONNX Runtime errors 3), `npu_probe_mismatch` (1: the NPU computes something else than the reference: check the QNN files and the NPU driver), `module_outside_bundle` (4) |
| `npu_context` | An EP context of the INT8 probe built in a temporary directory under `%TEMP%` (never in the user's cache), loaded, and run: its output must equal the compiled session's. Reports the HTP arch from the context's compatibility info. The EP's verdict on it is not available: it can only be asked before the first QNN session | Fail: the error's exit code (a context that cannot be loaded: 3), `ep_context_outputs_differ` (1: do not use `--ep-cache`). Warn: `htp_arch_unavailable`, `temporary_directory_left` |
| `ep_cache_dir` | The cache directory: exists (missing is fine: the first `--ep-cache` run creates it), writable, size, entries, and stale entries: their key's ONNX Runtime, QNN EP or QNN SDK version, HTP backend file version or device differs from this machine's, so no run will ever use them (the rule runs use, applied to every model and configuration) | Fail (exit 6): `ep_cache_dir_not_writable` (pass a writable `--ep-cache-dir`). Warn: `stale_cache_entries` (delete them with `--prune-cache`), `cache_directories_without_manifest` (never touched by edgebench: remove them yourself) |
| `power` | AC or battery, battery saver, the Windows power mode | Warn: `battery_power`, `battery_saver_on`, `power_mode_not_best_performance`, `power_source_unknown`. Published latency runs require AC power and the power mode Best performance |
| `background_load` | System-wide CPU load over `--load-sample-ms`, sampled first, before doctor does any work | Warn: `background_cpu_load_high` above 10% (close background apps) |

**Exit code.** 0 when no check fails (warnings allowed). When a check fails,
the exit code of the first failing check in report order, which is the code
`edgebench run` gives for the same problem (the one a run would hit first):

| Exit | Failures |
|---|---|
| 1 | `npu_probe_mismatch`, `ep_context_outputs_differ` |
| 3 | `multiple_npu_devices`, `ep_enumeration_failed`, probe or context sessions ONNX Runtime refuses (`ep_context_load_failed` included) |
| 4 | ONNX Runtime load failures, `bundle_file_mismatch`, `qnn_bundle_mismatch`, `module_outside_bundle`, QNN EP registration failures, backend load failures |
| 5 | A strict-mode refusal of a probe |
| 6 | `ep_cache_dir_not_writable` |

Failed checks are also printed to stderr as
`error: [<code>] <check>: <summary>` with their hint. With `--json` the report
has `checks[]` (`name`, `status`, `summary`, `code`, `hint`, `exit_code` for a
failure, `duration_ms`, `details`), `overall` (`status`, `exit_code`), the tool,
the pinned and found versions, the system block (results schema field names)
and `duration_ms`. The CI jobs run the full doctor on runners without an NPU
and require exit code 0 with the NPU checks N/A or warning
(`scripts/validate_doctor.py --runner`).

A full doctor takes about 8 s on a Surface Laptop 7: four NPU session creations
(two probes, the context build and its load, about 2 s each but the load)
make up almost all of it. `--quick` takes about 1 s.

Sample output (Surface Laptop 7, development build):

```text
edgebench 0.1.0 (0e964f9ff138587eeab6787f9c25589203a7a7e0, dirty, Release)
ONNX Runtime 1.32.0 (pinned 1.32.0), QNN EP 2.8.0 (pinned 2.8.0), QAIRT SDK 2.52.0 (release declares 2.52.0)
Windows 11 Home 25H2 (build 26200.9974, arm64), Snapdragon(R) X 10-core X1P64100 @ 3.40 GHz, Microsoft Corporation Microsoft Surface Laptop, 7th Edition
bundle C:\dev\edgebench\build\arm64-release\bin\Release

PASS  ort_bundle        onnxruntime 1.32.0 (C API 32) from the bundle, 2 files match their SHA256 pins, 4 ONNX Runtime / QNN modules loaded, all from the bundle
                        file version 1.32.0.20261112.3.fab30a4, C:\dev\edgebench\build\arm64-release\bin\Release\onnxruntime.dll
PASS  qnn_bundle        7 QNN files match their SHA256 pins; the QNN execution provider registers
                        onnxruntime_providers_qnn.dll, QnnHtp.dll, QnnHtpPrepare.dll, QnnHtpV68Stub.dll, QnnHtpV73Stub.dll, QnnHtpV81Stub.dll, QnnSystem.dll
                        QnnHtp.dll version 2.52.0.261104115025
PASS  backend_versions  backend build v2.52.0.261104115025, SDK 2.52.0 (reported by the QNN EP before any session)
                        agrees with the release's SDK 2.52.0 and QnnHtp.dll 2.52.0.261104115025
PASS  devices           3 EP devices; NPU for runs: QNNExecutionProvider NPU (ep vendor Qualcomm, vendor (unset), vendor_id 0x4d4f4351, device_id 0x41304430)
                        CPUExecutionProvider CPU (ep vendor Microsoft, vendor Qualcomm, vendor_id 0x4d4f4351, device_id 0x7)
                        QNNExecutionProvider NPU (ep vendor Qualcomm, vendor (unset), vendor_id 0x4d4f4351, device_id 0x41304430)
                        QNNExecutionProvider GPU (ep vendor Qualcomm, vendor (unset), vendor_id 0x4d4f4351, device_id 0x36334330)
PASS  npu_probe         strict NPU sessions, outputs bit-identical to the reference: npu_probe_int8.onnx (int8), npu_probe_fp32.onnx (fp16); every module from the bundle
                        npu_probe_int8.onnx: 256 values bit-identical, session 1850 ms
                        npu_probe_fp32.onnx: 256 values bit-identical, session 2039 ms
PASS  npu_context       built in 1894 ms, loaded in 418 ms, outputs equal; HTP arch v73
                        compatibility info v2:6:2.52.0:5.52.0:73:0:0:0
                        verdict (not rated): the QNN EP cannot rate compatibility info once a QNN session has existed in the process, which building the context needs
PASS  ep_cache_dir      C:\Users\you\AppData\Local\EdgeBench\qnn-context: 4 entries, 19.1 MiB
                        28eb1b68924a  mobilenetv2_int8_b8_ctx.onnx  4.1 MiB
                        359422e638b5  efficientnet_lite0_fp32_b1_ctx.onnx  11.2 MiB
                        4c992b58713a  mobilenetv2_int8_b1_ctx.onnx  3.8 MiB
                        b9ee82ef69ba  conv1d_autoencoder_int8_b1_ctx.onnx  0.1 MiB
WARN  power             power mode is Balanced, not Best performance
                        [power_mode_not_best_performance]
                        hint: Set the power mode to Best performance (Settings > System > Power). Published latency runs require AC power and the Windows power mode Best performance.
PASS  background_load   system-wide CPU load 4.7% over 1.0 s

Result: OK with 1 warning in 8.4 s
```

Loading the probe's EP context makes ONNX Runtime print its "supported but
suboptimal" warning on stderr (see
[EP context cache and initialization](#ep-context-cache-and-initialization));
the QNN EP's own diagnostics (`DSP_INFO ...`) may appear there too.

## Exit codes

Shared by `edgebench` and `tools/ort_probe` (defined in
`engine/include/edgebench/exit_codes.hpp`).

| Code | Name | Meaning |
|---|---|---|
| 0 | `success` | Success |
| 1 | `internal_error` | Internal or unexpected error. `ort_probe` also uses it when its output comparison fails |
| 2 | `usage_error` | Bad arguments, invalid values, or model file not found |
| 3 | `ort_runtime_error` | ONNX Runtime error: no usable EP device (none, or more than one, matches the requested device), or session creation or inference failed |
| 4 | `ort_load_failure` | ONNX Runtime, the QNN execution provider or its HTP backend could not be loaded, or a loaded ONNX Runtime/QNN DLL is not the bundled one |
| 5 | `strict_mode_refusal` | Strict mode refused the session: not every node can run on the requested device (NPU runs without `--no-strict`; `ort_probe --target npu`) |
| 6 | `output_error` | Results could not be written (`edgebench run`: after the summary was printed, or before measuring if `--output-dir` is an existing file, or the EP context cache directory is not writable; `ort_probe`: the `--json` file exists or could not be written) |
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
`edgebench_exe_test.cpp`, `benchmark_test.cpp`, `qnn_ep_test.cpp`,
`qnn_provider_test.cpp`, `ep_cache_npu_test.cpp`, `ep_context_cache_test.cpp`,
`npu_options_test.cpp`, `npu_selection_test.cpp`,
`run_config_test.cpp` and `cli_test.cpp`; "not triggerable" marks paths that need ONNX Runtime or Windows
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
| No CPU EP device, or more than one | `device_selection_failed` | 3 | yes, `ep_registration` | not triggerable |
| Session options rejected | `session_options_failed` | 3 | yes, `session_creation` | not triggerable |
| Model is not a valid ONNX file | `session_creation_failed` (code e.g. `ORT_INVALID_PROTOBUF`) | 3 | yes, `session_creation` | `ExitPaths.InvalidModelExits3AndWritesAFailedResult`, `Cli.RunDryRunInvalidModelExits3WithOrtCode` |
| Session creation fails on a later trial | as reported by ONNX Runtime | 3 | yes, `session_creation`, earlier trials kept | `FailurePaths.SessionCreationFailsOnTheSecondTrial` |
| An ONNX Runtime/QNN module from outside the bundle directory (the directory of the loaded `onnxruntime.dll`) is loaded when the provider has been created, or after a session is created | `module_outside_bundle` | 4 | yes, `ep_registration` (already loaded when the run starts) or `session_creation`, names each module and its path; `runtime.loaded_modules` lists it | `FailurePaths.ModuleOutsideTheBundleStopsTheRun` (`ep_registration`), `QnnProviderNpu.ModulesAreCheckedAfterEveryQnnSessionCreation` (`session_creation`) |
| Dynamic dimension, non-float32 input, output without fixed size | `dynamic_shape`, `unsupported_input_dtype`, `unsupported_output_dtype` | 2 | no | `Cli.RunUsageErrorFromTheModelWritesNothing`, `Cli.RunDryRunRejectsDynamicAndInt64Models` |
| Model without inputs | `no_inputs` | 2 | no | not tested (no fixture) |
| Model inputs/outputs cannot be read, buffers cannot be bound | `session_metadata_failed`, `binding_failed` | 3 | yes, `session_creation` | not triggerable |
| An inference fails | `inference_failed` (code e.g. `ORT_INVALID_ARGUMENT`) | 3 | yes, `warmup` or `measurement` | `ExitPaths.RunFailureExits3AndWritesAFailedResult`, `FailurePaths.RunFailureDuringWarmUpAndMeasurement` |
| An output contains NaN or infinity | `non_finite_output` | 1 | yes, `measurement` | `Cli.RunStopsOnNonFiniteOutputAndWritesAFailedResult` |
| A trial's top-1 differs from trial 1's | `inconsistent_outputs` | 1 | yes, `measurement` | `FailurePaths.OutputsWithADifferentTop1FailTheRun` |
| ONNX Runtime loads the model but its metadata cannot be decoded | `invalid_onnx` | 1 | yes, `model_load` | not triggerable |
| The placement session fails | as reported by ONNX Runtime | 3 | yes, `session_creation`, all trials kept | `FailurePaths.PlacementFailureFailsTheRun` |
| The placement session logs no placement | `placement_unavailable` | 1 | yes, `session_creation` | not triggerable |
| `--npu-fp16`, `--no-strict`, `--htp-performance-mode` or `--htp-graph-finalization-mode` without `--provider npu`; unknown `--htp-performance-mode`; `--htp-graph-finalization-mode` other than `0` to `3` | `usage_error` | 2 | no | `NpuCli.NpuOptionsWithTheCpuProviderAreUsageErrors`, `NpuCli.UnknownPerformanceModeIsAUsageError`, `NpuCli.GraphFinalizationModeIsParsedAndValidated` |
| NPU: FP32 model without `--npu-fp16` | `npu_fp32_needs_fp16` | 2 | no | `Cli.RunNpuPrecisionRulesAreUsageErrors`, `ComputePrecision.Fp32WithoutTheFlagIsRefused` |
| NPU: `--npu-fp16` with an INT8 model | `npu_fp16_needs_fp32_model` | 2 | no | `ComputePrecision.TheFlagOnAnInt8ModelIsAUsageError` |
| NPU: other stored precision (FP16, mixed, dynamic or QOperator quantization) | `npu_precision_unsupported` | 2 | no | `ComputePrecision.OtherStoredPrecisionsAreNotSupported` |
| NPU: no `onnxruntime_providers_qnn.dll` in the bundle (also every x64 build) | `qnn_ep_not_bundled` | 4 | yes, `ep_registration` | `Cli.RunNpuWithoutTheQnnEpExits4AndWritesAFailedResult` |
| NPU: no QNN EP device, none of type NPU, more than one | `no_qnn_ep_devices`, `no_npu_device`, `multiple_npu_devices` | 3 | yes, `ep_registration` | `Cli.RunNpuWithoutASingleNpuExits3` (runs only where the QNN EP offers no single NPU, e.g. the ARM64 CI runner), `NpuSelection.*` |
| NPU: `QnnHtp.dll` or `QnnHtpPrepare.dll` missing from the bundle | `qnn_backend_not_bundled` | 4 | yes, `ep_registration` | `RunNpu.MissingHtpBackendExits4NamingTheFile`, `QnnHtpBackend.MissingFilesAreALoadFailureNamingThem` |
| NPU: the QNN EP logs that its backend could not be loaded | `qnn_backend_load_failed` (ORT code kept) | 4 | yes, `session_creation` | `QnnSessionFailure.BackendLoadFailureIsExit4EvenThoughOrtReportsARefusal` |
| NPU, strict mode: a node cannot run on the NPU | `strict_mode_refusal` (code `ORT_FAIL`) | 5 | yes, `session_creation` | `RunNpu.StrictRefusalExits5AndWritesAFailedResult`, `QnnProviderNpu.StrictRefusalIsExit5WithTheOrtError`, `QnnSessionFailure.*` |
| NPU: after a QNN session is created, a QNN module is loaded from outside the bundle (e.g. the display driver's HTP stub when the bundled one is missing) | `module_outside_bundle` | 4 | yes, `session_creation` | `QnnProviderNpu.ModulesAreCheckedAfterEveryQnnSessionCreation`, `RunNpu.BackendStubFromOutsideTheBundleExits4` |
| NPU, strict mode: the placement session reports a node off the QNN EP | `strict_placement_mismatch` | 1 | yes, `session_creation` | not triggerable (ONNX Runtime refuses such a session) |
| `--ep-cache`, `--init-breakdown` without `--provider npu`; `--ep-cache-dir` without either; unknown `--ep-cache` mode | `usage_error` | 2 | no | `NpuCli.CacheOptionErrors` |
| NPU with EP contexts: `QnnSystem.dll` missing from the bundle | `qnn_backend_not_bundled` | 4 | yes, `ep_registration` | `QnnHtpBackend.MissingFilesAreALoadFailureNamingThem` |
| EP context cache directory cannot be created or written (or `LOCALAPPDATA` is unset) | `ep_cache_dir_not_writable` | 6 | yes, `configuration` | `EpCacheNpu.UnwritableCacheDirectoryExits6`, `EpCacheNpu.ExeExitCodesForCacheErrors`, `EpCacheDirectory.MustBeWritable` |
| The directory for the cache key exists without a manifest edgebench wrote | `ep_cache_entry_foreign` | 6 | yes, `configuration` | `EpCacheNpu.ADirectoryWithoutAManifestIsNeverTouched` |
| A cache file cannot be deleted | `ep_cache_delete_failed` | 6 | yes, `configuration` | `EpCacheEntries.RefusesToDeleteOutsideTheCacheDirectory` (the refusal); deletion failures not triggerable |
| `--ep-cache use`: a file of the entry is missing, truncated or changed (checked against its manifest) | `ep_context_corrupt` | 3 | yes, `session_creation`; nothing rebuilt | `EpCacheNpu.UseWithATruncatedContextExits3WithoutRebuilding`, `EpCacheNpu.ExeExitCodesForCacheErrors` |
| A session cannot be created from an EP context model | `ep_context_load_failed` (ORT code kept) | 3 | yes, `session_creation` | `EpCacheNpu.UseWithAContextOnnxRuntimeCannotLoadExits3` |
| The context-building session writes no context model | `ep_context_not_written` | 3 | yes, `session_creation` | not triggerable |
| A session from the EP context gives other outputs than a compiled session | `ep_context_outputs_differ` | 1 | yes, `measurement`, `verified_equal_outputs` false | not triggerable (the QNN EP gives bit-identical outputs; `ReferenceCachedNpu.*`) |
| Ctrl+C, Ctrl+Break, console closed | `interrupted` | 130 | yes, current stage, completed trials kept | `ExitPaths.CtrlBreakDuringARunWritesAnInterruptedResult`, `Interrupt.*` |
| Second Ctrl+C or Ctrl+Break | none | 130 | no | `ExitPaths.SecondCtrlBreakExitsAtOnce` |
| Ctrl+C or Ctrl+Break during `--dry-run` | none | 130 | no | same handler mode as `ExitPaths.OrtProbeCtrlBreakExits130WithoutWriting` |
| The result cannot be written | `write_failed` | 6 (a failed run keeps its own code) | no | `Cli.RunWithAnUnwritableOutputDirectoryExits6AfterTheSummary`, `ResultFiles.*` |
| An exception escapes | `internal_error` / `ort_runtime_error` | 1 / 3 | no | not triggerable |
| A crash (structured exception) | `internal_error` | 1 | no | `ExitPaths.UnhandledExceptionIsReportedAndExits1` |

Notes (in `notes`, not warnings): `batch_rounding_only`
(`BenchmarkModels.BatchEquivalenceComesFromTheReferenceSummary`),
`batch_equivalence_options_not_covered`
(`QnnProviderNpu.BatchEquivalenceFollowsTheGraphFinalizationMode`) and
`ep_context_recompilation_suggested`
(`EpCacheNpu.LoadingAContextIsNotedWhenOnnxRuntimeSuggestsRecompiling`).

Warnings that do not end the run: `nondeterministic_outputs`
(`FailurePaths.OutputsDifferingWithTheSameTop1AreAWarning`), `strict_mode_off`
and `cpu_fallback_nodes` (NPU with `--no-strict`;
`QnnProviderNpu.WithoutStrictModeUnsupportedNodesRunOnTheCpuWithAWarning`),
`qnn_validation_rejected_nodes` (every node on the QNN EP, but its validation
log reports failed nodes; not triggerable),
`nodes_not_on_requested_ep` (`FailurePaths.NodesOffTheRequestedEpAreAWarning`),
`threads_exceed_logical_processors`
(`ExitPaths.ThreadsAboveLogicalProcessorsWarnEverywhere`),
`background_cpu_load_high` (idle and during-trial samples;
`FailurePaths.BackgroundLoadDuringATrialIsAWarning`), `trial_spread_high`
(`FailurePaths.TrialSpreadIsAWarningInTheResultAndSummary`, `TrialSpread.*`),
`batch_not_equivalent` (`BatchEquivalence.LooksUpByShaProviderAndComputePrecision`; the warning itself: validator fixture `batch_not_equivalent_unflagged`),
`accelerator_time_exceeds_latency` and `profiled_pass_incomplete`
(`--profile-accelerator`; checked in `ProfiledPassNpu.IsASeparateSessionThatLeavesTheTimedResultsAlone`,
not triggered on this hardware; validator fixtures `accelerator_exceeds_latency_unflagged`,
`profiled_pass_count_mismatch`),
`power_state_changed`, `unknown_power_mode`, `value_unavailable` (for the QNN
values: `QnnBackendInfoNpu.WithoutAContextTheHtpArchIsUnavailable`).

`edgebench doctor` (the check and its code; see [`edgebench doctor`](#edgebench-doctor)):

| Case | Kind | Exit | Test |
|---|---|---|---|
| ONNX Runtime cannot be loaded (`ort_bundle`) | loader kind | 4 | `Doctor.MissingDllIsReported`, `Doctor.MissingDllJson`, `Doctor.WrongDllVersionIsReported`, `Doctor.DllWithoutOrtEntryPointIsReported`, `Doctor.CorruptDllIsReported` |
| A bundled ONNX Runtime file differs from its pin (`ort_bundle`) | `bundle_file_mismatch` | 4 | `Doctor.AModifiedBundleFileFailsItsPin`, `DoctorOrtBundle.PassWarnAndFail` |
| An ONNX Runtime/QNN module is loaded from outside the bundle (`ort_bundle`, `npu_probe`) | `module_outside_bundle` | 4 | `DoctorOrtBundle.PassWarnAndFail`, `DoctorNpuProbe.Outcomes` (not triggerable end to end: the loader refuses a foreign `onnxruntime.dll` first) |
| ARM64 build with a QNN file missing or changed (x64: `qnn_bundle` N/A, exit 0) | `qnn_bundle_mismatch` | 4 | `Doctor.MissingQnnFilesFailOnlyBuildsThatBundleThem`, `DoctorNpu.ExitCodesFollowTheFirstFailure`, `DoctorQnnBundle.FilesAndRegistration` |
| ONNX Runtime refuses to register the QNN EP library | `ep_registration_failed` | 4 | `Doctor.QnnEpThatIsNotAnEpLibraryIsReported` (x64), `DoctorQnnBundle.FilesAndRegistration` |
| EP devices cannot be enumerated | `ep_enumeration_failed` | 3 | `DoctorDevices.NpuSelectionOutcomes` (not triggerable end to end) |
| More than one QNN NPU | `multiple_npu_devices` | 3 | `DoctorDevices.NpuSelectionOutcomes` (not triggerable on real hardware) |
| No QNN EP device, none of type NPU | warning `no_qnn_ep_devices`, `no_npu_device` | 0 | `DoctorDevices.NpuSelectionOutcomes`, CI doctor step (ARM64 runner) |
| A probe session is refused or fails | the session's kind | 3, 4 or 5 | `DoctorNpuProbe.Outcomes` (not triggerable end to end: the probe models are built in) |
| A probe output differs from the expected one | `npu_probe_mismatch` | 1 | `DoctorNpuProbe.Outcomes`, `DoctorNpuProbe.ByteComparison` (not triggerable end to end) |
| The probe's EP context cannot be loaded, or computes other outputs | `ep_context_load_failed`, `ep_context_outputs_differ` | 3, 1 | `DoctorNpuContext.Outcomes` (not triggerable end to end) |
| The cache directory cannot be created or written | `ep_cache_dir_not_writable` | 6 | `Doctor.CacheDirectoryThatCannotBeCreatedExits6`, `DoctorNpu.ExitCodesFollowTheFirstFailure`, `DoctorEpCacheDir.Outcomes` |
| Stale cache entries; directories without a manifest | warning `stale_cache_entries`, `cache_directories_without_manifest` | 0 | `DoctorNpu.PruneCacheDeletesOnlyStaleEntries`, `Doctor.CacheDirectoryIsReportedAndNeverChangedWithoutPrune` |
| Battery, battery saver, not Best performance; load over 10%; emulation; backend version disagreement | warnings | 0 | `DoctorPower.OnlyAcAndBestPerformancePass`, `DoctorBackgroundLoad.ThresholdAndNoSample`, `DoctorOrtBundle.PassWarnAndFail`, `DoctorBackendVersions.AgreementAndDisagreement` |
| Several checks fail | the first failing check's | its code | `DoctorOverall.StatusAndExitCodeOfTheFirstFailure`, `DoctorNpu.ExitCodesFollowTheFirstFailure` |
| `--ep-cache-dir ""` | `usage_error` | 2 | `Doctor.EmptyCacheDirectoryIsAUsageError` |
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
| ONNX Runtime refuses to register the bundled QNN EP library | `ep_registration_failed` | 4 | `OrtProbe.QnnEpThatIsNotAnEpLibraryExits4` |
| After registering the QNN EP, an ONNX Runtime/QNN module is loaded from outside the bundle | `module_outside_bundle` | 4 | `QnnEpRegistration.AForeignQnnModuleFailsTheRegistration` (same check, in process) |
| `--target npu`: the QNN EP reports no device | `no_qnn_ep_devices` | 3 | `OrtProbe.NpuTargetWithoutASingleNpuExits3` (runs only on a machine without a QNN NPU, e.g. the ARM64 CI runner), `NpuSelection.NoDevicesAtAll`, `NpuSelection.CpuOnlyMeansNoQnnDevices` |
| `--target npu`: the QNN EP reports devices, none of type NPU | `no_npu_device` | 3 | `NpuSelection.GpuOnlyMeansNoNpu`, `NpuSelection.NpuOfAnotherEpIsNotSelected` (and the probe test above where it applies) |
| `--target npu`: the QNN EP reports more than one NPU | `multiple_npu_devices` | 3 | `NpuSelection.TwoNpusAreRefused` |
| `--target npu`: `QnnHtp.dll` or `QnnHtpPrepare.dll` missing from the bundle | `qnn_backend_not_bundled` | 4 | `QnnHtpBackend.MissingFilesAreALoadFailureNamingThem` (same check, in process) |
| `--target npu`, `--strict on`: a node cannot run on the NPU | `strict_mode_refusal` | 5 | `OrtProbeNpu.StrictRefusalExits5` |
| Output comparison fails | check output | 1 | manual (NPU) |
| Other ONNX Runtime error | `ort_runtime_error` with the ORT code | 3 | not tested |
| Ctrl+C or Ctrl+Break | none | 130 | `ExitPaths.OrtProbeCtrlBreakExits130WithoutWriting` |
| A crash | `internal_error` | 1 | same handler as `ExitPaths.UnhandledExceptionIsReportedAndExits1` |
