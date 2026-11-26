# Engine architecture

The C++ engine (`engine/`) is a static library, `edgebench_core`, plus a thin
`edgebench.exe`. This note describes how a benchmark session comes together and
how a new execution provider plugs in.

```text
load_onnxruntime()  ->  Runtime  ->  Provider  ->  BenchSession  ->  run_once() ...
   (ort_loader)        (Ort::Env)   (CpuProvider)  (OrtBenchSession)

run_benchmark()     ->  BenchmarkResult  ->  format_summary()
   (benchmark)          (schema-shaped)
```

## Loading and the Runtime

`load_onnxruntime()` (`ort_loader.hpp`) loads the bundled `onnxruntime.dll` by
full path and verifies it ([setup.md](setup.md#how-the-executables-find-onnx-runtime)).
Only then may anything touch the ONNX Runtime API.

`Runtime` (`runtime.hpp`) owns the process's single `Ort::Env`, created after a
successful load. ONNX Runtime keeps one environment per process, so there is one
Runtime at a time, and every session and provider must be released before it.
The Runtime installs a logging callback: warnings and errors go to stderr, and a
`Runtime::LogCapture` collects every message (including verbose ones) while it
is alive. Node placement uses this to read ONNX Runtime's "Node placements" log.

The Runtime also owns the execution-provider plugin libraries registered with
its environment (`Runtime::register_ep_library()`, `ep_library.hpp`). A plugin
library is registered by absolute path only and is unregistered before the
environment is released. Every session on one of its devices is meant to hold
an `EpLibrary::Lease`; unregistering refuses while a lease is held, so a library
is never pulled from under a live session, on error paths and after an
interrupt included (the Runtime is released last, after every session).

### Release the Runtime before `main()` returns

The ONNX Runtime environment, every session and every EP registration must be
released before `main()` returns. Never keep a `Runtime` (or anything holding
sessions) in a static or function-static object: in a process that has created
a QNN session, releasing the environment during static destruction after
`main()` intermittently never returns, and the process hangs with no CPU use.
`edgebench.exe` keeps its Runtime in a local of the command it runs. Test
programs use the one Runtime of `test_runtime.hpp` (`shared_runtime()`), which a
GoogleTest global environment releases when the tests finish, before `main()`
returns; that environment also fails the test program if any other `Runtime`
is still alive at that point (`Runtime::live_count()`), so a Runtime parked in
a static is caught by the next test run rather than by a hang.

### The QNN execution provider and the NPU device

`register_qnn_ep()` (`qnn_ep.hpp`) registers the bundled
`onnxruntime_providers_qnn.dll` (next to `onnxruntime.dll`) under
`QNNExecutionProvider` and then repeats the bundle-module check
(`bundle_modules.hpp`): registering loads the plugin DLL, and a module from
outside the bundle undoes the registration and fails with
`module_outside_bundle` (exit code 4). A missing plugin is `qnn_ep_not_bundled`
and a refused one `ep_registration_failed`, both exit code 4. The QNN backend
DLLs (`QnnHtp.dll` and the rest) are loaded only by NPU sessions, so the module
check runs again after each of those.

The devices are read into plain `EpDeviceInfo` structs (`ep_devices.hpp`: EP
name and vendor, hardware type, vendor, vendor and device IDs, hardware and EP
metadata as reported). `select_qnn_npu()` (`npu_selection.hpp`) is a pure
function over that list: the single device whose EP is `QNNExecutionProvider`
and whose hardware type is `NPU`. It returns the device's index, or one of
`no_qnn_ep_devices`, `no_npu_device` and `multiple_npu_devices` with a message
that lists every enumerated device; nothing in it is specific to one machine.
`ep_device_handle()` turns the index back into the `Ort::ConstEpDevice` a
session is attached to, after checking that the device list has not changed.

The QNN EP enumerates its NPU and GPU devices from the hardware, not from its
backend libraries: the devices are listed even when `QnnHtp.dll` is missing. A
selected NPU therefore says nothing about whether the HTP backend can be loaded;
that is only known when a session is created.

What the backend reports about itself is read without the QNN SDK headers,
through ONNX Runtime and the QNN EP only (`qnn_backend_info.hpp`). When
`QnnProvider` is created it raises the environment's log level to verbose and
asks the EP to rate a compatibility string whose every field is 0
(`GetModelCompatibilityForEpDevices`): the EP sets up its HTP backend for that
and logs the backend's build id, which gives `runtime.qnn.backend_build` and
`sdk_version`. The HTP arch is read from the compatibility info the EP stores
in an EP context model (`GetCompatibilityInfoFromModel`). The QNN EP 2.8.0
crashes with an access violation when it is asked to rate compatibility info
after any QNN session has existed in the process, so the provider asks only
before the first session: the build id is probed once per process and backend
and kept, and the verdict on a context (`context_compatibility()`) is only
requested on a cache hit, before any session.

Device-selection failures are exit code 3 (`ort_runtime_error`), like the CPU
provider's: ONNX Runtime and the plugin loaded correctly (that would be 4), but
the runtime offers no single usable device for the request.

## Provider

`Provider` (`provider.hpp`) is one execution-provider configuration:

| Member | Purpose |
|---|---|
| `name()`, `device_type()` | CLI name (`cpu`) and results-schema device type (`CPU`) |
| `describe()` | Everything the result file records about the provider: EP name, version and library path, the selected EP device with how it was selected (`by_device_type`, candidate count), `ep_options`, `session_config` |
| `create_session(model, config)` | Builds the session options for the device and returns a ready `BenchSession`, or an `Error`. Does nothing else, so its timing is the session's alone |
| `record_placement(model, config)` | Creates a separate session with the same options and verbose logging and returns the node counts per EP |

`make_provider(runtime, config)` returns the provider for `config.provider`.

A provider is responsible for exactly the device-specific part: registering
and selecting the EP device, the session options and config entries, and any
EP options. It records every config entry it sets in `describe()`, so the result
file shows the actual configuration (for example `CpuProvider` sets
`session.intra_op.allow_spinning=1` explicitly, the ONNX Runtime default, so the
spin setting is visible).

`CpuProvider` uses ONNX Runtime's built-in CPU EP: no plugin library, the CPU EP
device from the device enumeration as `ep_device`, intra-op threads always set
from the configuration, sequential execution mode, and the configured graph
optimization level.

## BenchSession

`BenchSession` (`bench_session.hpp`) is one session ready to be timed:

| Member | Purpose |
|---|---|
| `inputs()`, `outputs()` | Names, dtypes and fixed shapes |
| `batch_size()` | Dimension 0 of the first input |
| `run_binding()` | `run` or `iobinding_preallocated` |
| `input_data(i)` | The owned float32 input buffer (stable address) |
| `run_once()` | One inference; the only call inside the timed region |
| `output(i)` | Read-only view of the latest outputs |
| `session_create_ms()` | Creation time of this session |

`OrtBenchSession` (`ort_bench_session.hpp`) implements it for every ONNX Runtime
provider. Given the provider's `Ort::SessionOptions`, it:

- creates and times the session;
- reads the inputs and outputs and rejects dynamic or symbolic dimensions and
  non-float32 inputs (usage errors, exit code 2);
- generates the inputs once into owned buffers (SplitMix64 or a constant;
  [cli.md](cli.md#input-data));
- for `run`, creates the input `Ort::Value`s once over those buffers and lets
  `Session::Run` allocate the outputs; for `iobinding_preallocated`, binds the
  inputs and preallocated output buffers once, so `run_once()` allocates no
  tensors.

Node placement is not part of a BenchSession: `Provider::record_placement()`
creates its own session with verbose logging, so verbose logging never affects a
session that is timed.

## Trial loop and scheduling

`run_benchmark()` (`benchmark.hpp`) runs one configuration:

```text
hash + inspect model file            (not timed; warms the file cache for every trial)
read power state
NPU, --ep-cache build|use:           (before the first trial, not timed as trials)
    key fields -> key id; delete stale entries of this model
    load the entry for the key, or build it (context_build_ms)
    compiled session vs session from the context, same input -> verified_equal_outputs
for each trial (scheduler order):
    --init-breakdown: for cold, context build (scratch entry), cached:
        idle as below; create the session -> trial session_init; release
    idle --cooldown-ms (1 s before the first trial if 0), sampling CPU load   [stop: idle]
                                                                               [stop: before_session]
    provider.create_session()        -> timed creation (from the model, or the
                                        cached context), working-set delta
                                                                               [stop: after_session]
    warm-up 1 (timed)                -> first_inference_ms (trial 0)          [stop: warmup, each]
    warm-up 2..n (untimed)
    measured iterations              -> one latency per run_once()            [stop: measured, each]
    finite check, compare outputs with trial 0, release the session
read process memory
                                                                               [stop: before_placement]
provider.record_placement()          (after every timed session)
                                                                               [stop: after_placement]
read power state again               -> power_state_changed warning
statistics: per trial, pooled aggregate, trial variation
```

Every trial gets a fresh session, so session creation is measured once per
trial and no trial inherits another's warm state. Every trial, the first
included, starts after the same idle period, because session creation is
slower when the CPU has just been idle: idling only between trials made the
first trial systematically faster to create. Placement runs last for the
same reason: its verbose session must not warm anything before a timed session
is created.

The EP context cache (`ep_context_cache.hpp`) is pure file handling: the key
fields and their canonical serialization, the manifest each entry carries, the
manifest-only deletion of entries and stale entries, and the lookup that tells
a missing, valid, corrupt or foreign entry apart. `benchmark.cpp` decides what
to build and load; the provider only knows how to create a session that writes
a context (`Provider::create_context_building_session()`, implemented by
`QnnProvider` with `qnn_context_build_config()`), and a session from a context
model is an ordinary `create_session()`.

`schedule_trials(configurations, trials)` (`scheduler.hpp`) produces the order:
trials run in rounds, round r runs every configuration once starting at
configuration r mod N (`0 1 2 | 1 2 0 | 2 0 1`). No configuration always runs
first or always right after a particular other one. `run` uses it with one
configuration; a suite of configurations uses the same scheduler. Each trial
records its `trial_index` and its `sequence` in the overall order.

`BenchmarkResult` mirrors the results schema 1.2.0 (tool, system, runtime,
model, config, execution, timing, warnings) and keeps every per-iteration
latency, so exporters only translate it. `stats.hpp` holds the statistics as pure
functions (nearest-rank percentiles, sample standard deviation, trial variation);
`timer_info.hpp` reports the clock and its effective resolution
(`steady_clock::period` checked against `QueryPerformanceFrequency`).
`onnx_inspect.hpp` reads the model file's opset, IR version, producer and
quantization nodes, which ONNX Runtime does not expose.

## Results and failed runs

`run_benchmark()` always returns a `BenchmarkResult`. If the run fails, its
`failure` holds the error and the schema's `error.stage`, and `known` records
which parts were established (runtime, provider, model, a session, placement),
so a failed result keeps everything collected so far. `begin_result()` fills the
parts that need no ONNX Runtime (identity, tool, system, power, the model file's
hash and decoded metadata, including its tensors), and `finish_result()` the
parts derived at the end (statistics of the completed trials, memory, power
changes, warnings). A run whose ONNX Runtime cannot be loaded gets its result
from `failed_load_result()`.

`result_json.hpp` turns a result into the schema 1.2.0 document (every key, in
schema order) and the two CSV files; `result_files.hpp` writes them without
overwriting anything (exclusive creation through a temporary file and a
no-replace rename, collision suffixes, atomic replacement of `<name>_latest.json`).
`main` prints the summary first, then writes, so a write error (exit code 6)
never loses the measurements. See [cli.md](cli.md#result-files).

### Interrupts

`interrupt.hpp` installs the console control handler. In `stop_gracefully`
mode (a benchmark) the handler only stores the event, sets an atomic flag and
signals a manual-reset event; a second Ctrl+C or Ctrl+Break calls
`TerminateProcess(130)`. For a console close, logoff or shutdown it waits up to
4 s on a second event that `main` signals once the result is written
(`notify_shutdown_complete()`), because Windows ends the process when the
handler returns. In `exit_immediately` mode (dry run, `doctor`, `ort_probe`) the
first event ends the process with 130.

The trial loop polls `stop_requested()` at the `[stop: ...]` points above, all
outside the timed brackets, and the idle periods wait on the event
(`wait_or_stop()`), so no timing changes when nobody interrupts. The polls go
through `should_stop(hook, StopPoint, trial, iteration)`: tests pass a
`StopHook` that calls `request_stop()` at a chosen point, which exercises
exactly the path a real Ctrl+C takes. An interrupted run becomes a failed
result (`interrupted`, exit 130) with the completed trials; the partial trial
is dropped and placement is skipped.

## Errors

Fallible operations return `Result<T>` (`result.hpp`): a value or an `Error`
with the exit code ([cli.md](cli.md#exit-codes)), a stable `kind`, a message and,
for ONNX Runtime failures, the ORT error code and name. Usage problems with the
model are exit code 2, ONNX Runtime failures 3, load failures 4.

Everything that reaches the user goes through `diagnostics.hpp`
(`format_error`, `format_warning`, `print_error`, `print_warning`):
`error: [<kind>] <message>`, an optional `hint:` line, and
`warning: [<code>] <message>`, in `edgebench` and `ort_probe` alike. `wmain`
catches `Ort::Exception` (exit 3) and `std::exception` (exit 1) as a last
resort. `crash_handler.hpp` installs an unhandled-exception filter that writes
`error: [internal_error] unhandled exception 0x<code> at <address>` with
`WriteFile` from a stack buffer (no allocation in a crashed process) and ends
the process with exit code 1; `SetErrorMode` suppresses the Windows error
dialog, and no dump is written. The full list of failure cases with their kind,
exit code and whether a result is written is in
[cli.md](cli.md#exit-paths).

## Adding a provider

1. Implement `Provider` for the device: select the EP device (exactly one
   candidate), build the `Ort::SessionOptions` (append the EP with its options
   and config entries), and fill `ProviderDescription` with everything that was
   set.
2. In `create_session`, create an `OrtBenchSession` with those options. In
   `record_placement`, call `record_node_placement()` with the same options.
   Nodes that are not on the requested EP are reported by the caller; in strict
   mode a refused session is exit code 5.
3. Add the value to `ProviderKind`, `parse_provider()` and `make_provider()`.

The timing, statistics and export code only use `Provider::describe()` and the
`BenchSession` interface, so they need no change.

## Supporting modules

- `model_info.hpp`: model file size and SHA-256; `sha256.hpp`: SHA-256 (CNG) for
  files and buffers, also for cache keys.
- `tensor_info.hpp`: tensor metadata, shape and dtype checks, non-finite checks.
- `placement.hpp`: parsing ONNX Runtime's node placement log.
- `system_info.hpp`: OS, CPU, memory, device and architecture queries; power
  state (GetSystemPowerStatus, PowerGetActiveScheme, PowerGetEffectiveOverlayScheme),
  background CPU load (GetSystemTimes deltas) and process memory
  (GetProcessMemoryInfo). The decoding and arithmetic are pure functions.
- `run_config.hpp`, `cli.hpp`: the run configuration and command-line parsing.
- `dry_run.hpp`: `run --dry-run`.
- `run_identity.hpp`: run IDs, RFC 3339 timestamps and the file stamp
  (`YYYYmmddTHHMMSS.mmmZ`).
