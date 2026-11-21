# Engine architecture

The C++ engine (`engine/`) is a static library, `edgebench_core`, plus a thin
`edgebench.exe`. This note describes how a benchmark session comes together and
how a new execution provider plugs in.

```text
load_onnxruntime()  ->  Runtime  ->  Provider  ->  BenchSession  ->  run_once() ...
   (ort_loader)        (Ort::Env)   (CpuProvider)  (OrtBenchSession)
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
| `create_session(model, config)` | Builds the session options for the device and returns a ready `BenchSession`, or an `Error` |

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
| `node_assignment()` | Nodes per EP, from a separate verbose placement session |
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

The provider records node placement in a second session with the same options
and verbose logging, so verbose logging never affects the session that is timed.

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
2. In `create_session`, create an `OrtBenchSession` with those options and
   record node placement with `record_node_placement()` using the same options.
   Report nodes that are not on the requested EP; in strict mode a refused
   session is exit code 5.
3. Add the value to `ProviderKind`, `parse_provider()` and `make_provider()`.

The timing, statistics and export code only use `Provider::describe()` and the
`BenchSession` interface, so they need no change.

## Supporting modules

- `model_info.hpp`: model file size and SHA-256; `sha256.hpp`: SHA-256 (CNG) for
  files and buffers, also for cache keys.
- `tensor_info.hpp`: tensor metadata, shape and dtype checks, non-finite checks.
- `placement.hpp`: parsing ONNX Runtime's node placement log.
- `system_info.hpp`: OS, CPU, memory, device and architecture queries.
- `run_config.hpp`, `cli.hpp`: the run configuration and command-line parsing.
- `dry_run.hpp`: `run --dry-run`.
