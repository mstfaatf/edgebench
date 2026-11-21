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
for each trial (scheduler order):
    idle --cooldown-ms (1 s before the first trial if 0), sampling CPU load
    provider.create_session()        -> cold_ms[trial], working-set delta
    warm-up 1 (timed)                -> first_inference_ms (trial 0)
    warm-up 2..n (untimed)
    measured iterations              -> one latency per run_once()
    finite check, compare outputs with trial 0, release the session
read process memory
provider.record_placement()          (after every timed session)
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

`schedule_trials(configurations, trials)` (`scheduler.hpp`) produces the order:
trials run in rounds, round r runs every configuration once starting at
configuration r mod N (`0 1 2 | 1 2 0 | 2 0 1`). No configuration always runs
first or always right after a particular other one. `run` uses it with one
configuration; a suite of configurations uses the same scheduler. Each trial
records its `trial_index` and its `sequence` in the overall order.

`BenchmarkResult` mirrors the results schema 1.1.0 (tool, system, runtime,
model, config, execution, timing, warnings) and keeps every per-iteration
latency, so exporters only translate it. `stats.hpp` holds the statistics as pure
functions (nearest-rank percentiles, sample standard deviation, trial variation);
`timer_info.hpp` reports the clock and its effective resolution
(`steady_clock::period` checked against `QueryPerformanceFrequency`).
`onnx_inspect.hpp` reads the model file's opset, IR version, producer and
quantization nodes, which ONNX Runtime does not expose.

## Errors

Fallible operations return `Result<T>` (`result.hpp`): a value or an `Error`
with the exit code ([cli.md](cli.md#exit-codes)), a stable `kind`, a message and,
for ONNX Runtime failures, the ORT error code and name. Usage problems with the
model are exit code 2, ONNX Runtime failures 3, load failures 4.

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
