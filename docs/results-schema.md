# Benchmark Results Schema

This document specifies the format of EdgeBench benchmark results: the
contract between the C++ engine that writes them, the Python analysis scripts,
and the dashboard that reads them.

- Machine-readable schema: [`schemas/benchmark-result.schema.json`](../schemas/benchmark-result.schema.json) (JSON Schema draft 2020-12)
- Current version: **1.0.0**
- Examples: [`docs/examples/example-result.json`](examples/example-result.json) (successful run) and
  [`docs/examples/example-result-failed.json`](examples/example-result-failed.json) (failed run).
  Both are illustrative only, with placeholder values.
- Validator: `python scripts/validate_results.py FILE_OR_DIR ...`

## Principles

1. **Never guess.** Anything that was not measured or cannot be determined is
   `null`, never a plausible-looking default. Where the reason matters, add a
   `warnings` entry whose `path` points at the null field. A `0` always means a
   measured or configured zero.
2. **Units in field names.** Every quantity with a unit carries it as a suffix:
   `_ms` (milliseconds), `_ns` (nanoseconds), `_bytes`, `_per_sec`. Unitless
   fractions end in `_ratio` (0–1, not percent). Counts end in `_count`,
   `_iterations`, `_threads` or are plainly countable (`trials`, `batch_size`).
3. **Requested vs. verified.** What the user asked for (`config.requested`,
   `config.strict_mode`) is kept separate from what the run established
   (`runtime.execution_provider`, `runtime.ep_device`, `execution`). In
   particular, "requested the NPU" is not the same as "verified that the whole
   model ran on the NPU".
4. **Precision honesty.** `model.stored_precision` is the precision of the model
   file as stored. How the hardware actually computes may differ: for example,
   an NPU backend may run an FP32 graph at reduced precision internally, so its
   outputs differ from the CPU EP by more than FP32 rounding. Such known
   differences go in `model.precision_notes`. No field claims a computed
   precision that cannot be observed.
5. **One file = one run.** A result file describes one model with one provider
   configuration, measured over one or more trials. Comparisons (CPU vs NPU,
   FP32 vs INT8) are made across files.
6. **Per-iteration data stays out of the main file.** Individual latencies go in
   a separate CSV referenced by `raw.latencies_file`. The JSON holds summary
   statistics only.
7. **Versioned.** Every file declares `schema_version` (semver); see
   [Versioning](#versioning).

## Conventions

- Every field in the schema is **required to be present**. Optional
  information is expressed as `null`, not by omitting the key. This makes
  "not known" explicit and keeps readers simple.
- Objects reject unknown keys (`additionalProperties: false`), except
  free-form string maps (`config.ep_options`, `config.session_config`,
  `execution.node_assignment.nodes_by_ep`).
- Timestamps are RFC 3339 in UTC with a `Z` suffix, e.g.
  `2026-11-20T17:34:27.053Z`.
- UUIDs are lowercase, hyphenated.
- SHA256 digests are 64 lowercase hex characters.
- Paths inside the document (`raw.latencies_file`) are relative to the result
  file. Absolute paths appear only where the field records where something was
  loaded from (`*_path`, `library_path`).
- Device types are `CPU`, `GPU`, `NPU`. Tensor element types use ONNX names in
  lowercase (`float32`, `float16`, `int8`, `uint8`, …).
- The nil UUID (`00000000-0000-0000-0000-000000000000`) is reserved for
  illustrative examples, which must also carry an `illustrative_example`
  warning.

In the tables below, *nullable* means the value may be `null`; the key itself
is always present.

## Top level

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `schema_version` | string | no | Schema version, `"1.0.0"` |
| `run_id` | string (UUID) | no | Unique ID of this run |
| `timestamp_utc` | string (RFC 3339, UTC) | no | Start time of the run |
| `label` | string | yes | Optional human-readable name |
| `status` | `"success"` \| `"failed"` | no | Whether the run completed |
| `error` | object | yes | Set if and only if `status` is `"failed"` |
| `warnings` | array of objects | no (may be empty) | Caveats, unavailable values, anomalies |
| `tool` | object | no | The EdgeBench build that produced the file |
| `system` | object | no | Machine and OS |
| `runtime` | object | yes¹ | ONNX Runtime, execution provider, device |
| `model` | object | yes¹ | The model that was run |
| `config` | object | no | Requested benchmark configuration |
| `execution` | object | yes¹ | What was verified about where the model ran |
| `timing` | object | yes¹ | Latency measurements |
| `memory` | object | yes | Process memory usage |
| `accuracy` | object | yes | Optional accuracy evaluation |
| `raw` | object | yes | Reference to per-iteration latencies |

¹ Required (non-null) when `status` is `"success"`. A failed run fills in
whatever was established before the failure and leaves the rest `null`.

### `error`

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `stage` | enum | no | Where the run failed: `configuration`, `environment`, `model_load`, `ep_registration`, `session_creation`, `warmup`, `measurement`, `accuracy`, `output` |
| `message` | string | no | Human-readable error message |
| `code` | string | yes | Error code from the failing component, e.g. an ONNX Runtime `OrtErrorCode` name |

### `warnings[]`

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `code` | string (`snake_case`) | no | Machine-readable category |
| `message` | string | no | Human-readable explanation |
| `path` | string (JSON Pointer) | yes | Field the warning refers to, e.g. `/runtime/ep_device/driver_version` |

Recommended codes:

| Code | Use |
|---|---|
| `illustrative_example` | The file is an example, not a measurement |
| `value_unavailable` | A value could not be determined and is `null` |
| `insufficient_samples` | A statistic is `null` because there were too few samples |
| `outlier_trial` | A trial deviates strongly from the others |
| `power_not_on_ac` | The device was on battery during the run |

## `tool`

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `name` | `"edgebench"` | no | Tool name |
| `version` | string (semver) | no | EdgeBench version |
| `git_commit` | string (40 hex) | yes | Commit the binary was built from |
| `git_dirty` | boolean | yes | Whether the working tree had uncommitted changes at build time |
| `build_type` | string | yes | CMake configuration, e.g. `Release` |
| `compiler` | string | yes | Compiler and version, e.g. `MSVC 19.42.34435` |

## `system`

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `os.name` | string | no | e.g. `Windows 11` |
| `os.version` | string | yes | Display version, e.g. `25H2` |
| `os.build` | string | yes | Full build number, e.g. `26200.9974` |
| `os.architecture` | `arm64` \| `x64` \| `arm64ec` \| `x86` | yes | OS architecture |
| `process_architecture` | same enum | yes | Architecture the edgebench process was built for. If this differs from `os.architecture`, the run was emulated |
| `cpu.name` | string | yes | CPU name as reported by the OS |
| `cpu.physical_cores` | integer ≥ 1 | yes | Physical cores |
| `cpu.logical_processors` | integer ≥ 1 | yes | Logical processors |
| `memory_total_bytes` | integer | yes | Installed physical memory visible to the OS |
| `device.manufacturer` | string | yes | e.g. from SMBIOS |
| `device.model` | string | yes | e.g. from SMBIOS |
| `power.source` | `ac` \| `battery` | yes | Power source during the run |
| `power.power_mode` | string | yes | Windows power mode as reported by the OS |

## `runtime`

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `onnxruntime.version` | string (semver) | no | Version reported by the loaded library (not the headers) |
| `onnxruntime.library_path` | string | yes | Absolute path of the loaded `onnxruntime.dll` |
| `execution_provider.name` | string | no | EP the session used, e.g. `QNNExecutionProvider`, `CPUExecutionProvider` |
| `execution_provider.version` | string | yes | EP version (e.g. the QNN plugin EP version); `null` for built-in EPs without their own version |
| `execution_provider.library_path` | string | yes | Plugin EP library, if registered from a library |
| `qnn` | object | yes | `null` unless the QNN EP is used |
| `qnn.sdk_version` | string | yes | QNN SDK (QAIRT) version the EP was built with |
| `qnn.backend` | `htp` \| `gpu` \| `cpu` \| `saver` \| `ir` | yes | QNN backend |
| `qnn.backend_path` | string | yes | Backend library path as passed to the EP |
| `ep_device` | object | yes | The EP device the session was attached to; `null` if no device-based selection was used |
| `ep_device.ep_name` | string | no | EP name of the device entry |
| `ep_device.ep_vendor` | string | yes | EP vendor |
| `ep_device.device_type` | `CPU` \| `GPU` \| `NPU` | no | Hardware device type |
| `ep_device.vendor` | string | yes | Hardware vendor string; some EPs leave it unset |
| `ep_device.vendor_id` | integer | yes | PCI-style vendor ID, if reported |
| `ep_device.device_id` | integer | yes | Device ID, if reported |
| `ep_device.driver_version` | string | yes | Device driver version (e.g. the NPU driver) |

## `model`

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `name` | string | no | Logical model name, e.g. `mobilenet_v2` |
| `file_name` | string | no | File name of the ONNX model (no directory) |
| `sha256` | string | no | SHA256 of the model file |
| `size_bytes` | integer | no | Size of the model file |
| `opset` | integer | yes | Opset of the default ONNX domain |
| `ir_version` | integer | yes | ONNX IR version |
| `stored_precision` | `fp32` \| `fp16` \| `int8` \| `mixed` | no | Precision as stored in the file. `mixed` covers models with more than one compute precision (e.g. INT8 with FP32 islands, or 8/16-bit mixes) |
| `inputs[]`, `outputs[]` | array of tensor info | no | See below |
| `source.library` | string | yes | e.g. `torchvision`, `timm`, `keras`, `custom` |
| `source.identifier` | string | yes | Model/weights identifier, e.g. `mobilenet_v2/IMAGENET1K_V2` |
| `source.export_tool` | string | yes | Tool and version that produced the ONNX file |
| `quantization` | object | no | See below |
| `precision_notes` | string | yes | Known differences between stored precision and how hardware computes |

Tensor info:

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Tensor name |
| `dtype` | string | ONNX element type, lowercase |
| `shape` | array | One entry per dimension: integer (fixed), string (named symbolic dimension, e.g. `"batch"`), or `null` (unnamed unknown dimension). Shapes are as declared in the model; `config.batch_size` gives the value used |

`quantization`:

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `method` | `none` \| `static` \| `dynamic` \| `qat` | no | Quantization method |
| `format` | `qdq` \| `qoperator` | yes | ONNX quantization format |
| `activation_dtype` | dtype | yes | Activation type, e.g. `uint8`, `uint16` |
| `weight_dtype` | dtype | yes | Weight type, e.g. `uint8`, `int8` |
| `calibration` | object | yes | `{dataset, sample_count}` (both nullable) for static quantization |
| `tool` | string | yes | Quantization tool and version |

When `method` is `none`, all other quantization fields must be `null`.

## `config`

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `requested.execution_provider` | string | no | EP the user asked for |
| `requested.device_type` | `CPU` \| `GPU` \| `NPU` | no | Device the user asked for |
| `requested.compute_precision` | `fp32` \| `fp16` \| `int8` | yes | Compute precision explicitly requested from the EP (e.g. via an EP option); `null` = EP default |
| `strict_mode` | boolean | no | `true` if CPU fallback was disabled (`session.disable_cpu_ep_fallback = 1`) |
| `batch_size` | integer ≥ 1 | no | Items per inference |
| `input.kind` | `constant` \| `random` \| `dataset` | no | How inputs were generated |
| `input.constant_value` | number | yes | Fill value for `constant` |
| `input.seed` | integer | yes | RNG seed for `random` |
| `input.dataset` | string | yes | Dataset name for `dataset` |
| `warmup_iterations` | integer ≥ 0 | no | Unmeasured inferences before each trial |
| `measured_iterations` | integer ≥ 1 | no | Measured inferences per trial |
| `trials` | integer ≥ 1 | no | Number of trials |
| `run_api` | `sync` \| `async` | no | `Session::Run` or `RunAsync` |
| `ort_execution_mode` | `sequential` \| `parallel` | no | ONNX Runtime graph execution mode |
| `intra_op_threads` | integer ≥ 0 | yes | Requested intra-op threads; `null` = left at the ONNX Runtime default (the effective count is not observable) |
| `inter_op_threads` | integer ≥ 0 | yes | Requested inter-op threads; `null` = default |
| `graph_optimization_level` | `disable_all` \| `enable_basic` \| `enable_extended` \| `enable_layout` \| `enable_all` | no | ONNX Runtime graph optimization level |
| `ep_options` | map string → string | no (may be empty) | EP options passed to the provider |
| `session_config` | map string → string | no (may be empty) | Session config entries set explicitly |
| `ep_context_cache.enabled` | boolean | no | Whether an EP context (precompiled) model was used or generated |
| `ep_context_cache.cache_file` | string | yes | File name of the EP context model |

## `execution`

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `full_device_execution_verified` | boolean | yes | `true`: verified that the whole graph ran on the requested device. `false`: verified that it did not (some nodes fell back). `null`: not verified |
| `verification_method` | enum | yes | How it was established. Required (non-null) when the flag is `true` or `false`, and `null` when the flag is `null` |
| `verification_details` | string | yes | Free-text explanation |
| `node_assignment` | object | yes | `{total_nodes, nodes_by_ep: {ep_name: count}}`, if available |

`verification_method` values:

| Value | Meaning |
|---|---|
| `strict_session` | CPU fallback was disabled and session creation succeeded, so every node was assigned to the requested EP |
| `node_assignment` | Determined from per-node EP assignment (e.g. profiling or session logs); `node_assignment` holds the counts |
| `cpu_provider_only` | The CPU EP was requested and is the only EP in the session, so CPU execution is guaranteed |

## `timing`

All durations are milliseconds of wall-clock time measured around the inference
call, using a monotonic clock.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `timer.source` | string | no | Clock used, e.g. `std::chrono::steady_clock` |
| `timer.resolution_ns` | number | yes | Clock tick period |
| `percentile_method` | `nearest_rank` | no | How percentiles are computed (see below) |
| `session_init.session_create_ms` | number | yes | Time to create the inference session (includes EP graph compilation) |
| `session_init.ep_context_state` | `cold` \| `cached` \| `not_used` | yes | `cold`: compiled from the ONNX graph; `cached`: loaded from an EP context model; `not_used`: the EP has no context cache (e.g. CPU) |
| `first_inference_ms` | number | yes | Latency of the very first inference (first warm-up iteration), which often includes one-time setup |
| `trials[]` | array | no | One entry per trial: `{trial_index, stats}`. `trial_index` is 0-based and in order. Non-empty for successful runs |
| `aggregate` | object | yes | Required for successful runs |
| `aggregate.stats` | latency stats | no | Statistics over all measured iterations of all trials pooled |
| `aggregate.trial_variation.trial_mean_min_ms` | number | no | Smallest trial mean |
| `aggregate.trial_variation.trial_mean_max_ms` | number | no | Largest trial mean |
| `aggregate.trial_variation.trial_mean_stddev_ms` | number | yes | Sample std dev of the trial means; `null` with fewer than 2 trials |
| `aggregate.trial_variation.trial_mean_cv_ratio` | number | yes | Coefficient of variation of the trial means (stddev / mean of trial means); `null` with fewer than 2 trials |

Latency stats (used for each trial and for the aggregate):

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `count` | integer ≥ 1 | no | Number of measured inferences covered |
| `mean_ms` | number | no | Arithmetic mean |
| `p50_ms` | number | no | Median (nearest-rank 50th percentile) |
| `p95_ms` | number | yes | 95th percentile; `null` if `count < 20` |
| `p99_ms` | number | yes | 99th percentile; `null` if `count < 100` |
| `stddev_ms` | number | yes | Sample standard deviation (n − 1); `null` if `count < 2` |
| `min_ms`, `max_ms` | number | no | Extremes |
| `inferences_per_sec` | number | no | `1000 / mean_ms`: sequential, latency-derived throughput, not concurrent throughput |
| `items_per_sec` | number | no | `inferences_per_sec × config.batch_size` |

Definitions:

- **Nearest-rank percentile**: for sorted samples `x[1..n]`, the p-th percentile
  is `x[ceil(p/100 × n)]`. It is always an observed value. The median is
  `p50_ms`; there is no separate median field.
- The sample-count thresholds for `p95_ms`, `p99_ms` and `stddev_ms` are
  strict in both directions: below the threshold the value must be `null`
  (with an optional `insufficient_samples` warning), and at or above it the
  value must be present.
- Warm-up iterations are never included in any statistic.

## `memory`

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `scope` | `"process"` | no | Measurements cover the edgebench process only. Memory used on the NPU or by drivers outside the process is not included |
| `peak_working_set_bytes` | integer | yes | Peak working set of the process |
| `peak_private_bytes` | integer | yes | Peak private (committed) bytes of the process |
| `source` | string | yes | API used, e.g. `GetProcessMemoryInfo` |

## `accuracy`

Optional. `null` unless an accuracy evaluation was run.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `dataset` | string | no | Dataset name/split |
| `sample_count` | integer ≥ 1 | no | Samples evaluated |
| `metrics[]` | array of `{name, value}` | no | e.g. `{"name": "top1_accuracy_ratio", "value": 0.71}`. Metric names follow the unit-suffix convention; `value` may be `null` if it could not be computed |
| `reference_agreement` | object | yes | Agreement with a reference run on the same samples |
| `reference_agreement.reference_run_id` | UUID | no | `run_id` of the reference run (e.g. the CPU FP32 run) |
| `reference_agreement.top1_agreement_ratio` | number 0–1 | no | Fraction of samples whose top-1 prediction matches the reference |
| `reference_agreement.sample_count` | integer ≥ 1 | no | Samples compared |

## `raw`

`null` if per-iteration latencies were not saved.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `latencies_file` | string | no | Path of the latency CSV, relative to the result file |
| `format` | `"csv"` | no | File format |
| `sha256` | string | yes | SHA256 of the latency file |

Latency CSV format: UTF-8, header row `trial_index,iteration_index,latency_ms`,
then one row per measured iteration (warm-up excluded), ordered by trial and
then iteration. Both indices are 0-based. The number of rows equals
`timing.aggregate.stats.count`.

## Consistency rules

`scripts/validate_results.py` checks the JSON Schema and, in addition, rules
that JSON Schema cannot express:

- `min_ms ≤ p50_ms ≤ p95_ms ≤ p99_ms ≤ max_ms` (skipping nulls), and
  `min_ms ≤ mean_ms ≤ max_ms`.
- The sample-count thresholds above, in both directions.
- `inferences_per_sec ≈ 1000 / mean_ms` and `items_per_sec ≈ inferences_per_sec × batch_size`
  (relative tolerance 1e-3).
- `trial_index` runs `0..n-1`. Each trial's `count` equals
  `config.measured_iterations`. For successful runs, the number of trials equals
  `config.trials`. The aggregate `count` equals the sum of trial counts.
- `trial_mean_min_ms` / `trial_mean_max_ms` match the trial means, and the
  trial-variation fields are `null` exactly when there are fewer than 2 trials.
- If `raw` is set, the latency file exists next to the result, has the expected
  header and row count, and matches `raw.sha256` if given.
- The nil UUID is only used together with an `illustrative_example` warning.

Statistics are not recomputed from the latency file; the engine is responsible
for computing them from the same data.

## CSV summary format

For spreadsheets and quick analysis, runs can be flattened into a CSV with
**one row per run**. It is a lossy, documented subset of the JSON. The JSON
file remains the source of truth.

Encoding: UTF-8 without BOM, comma-separated, LF line endings, header row,
fields quoted per RFC 4180 when they contain a comma, quote or newline.

Flattening rules:

- Each column maps to exactly one JSON field (or a simple derivation, marked
  below). Values are copied as-is; no unit conversion.
- `null` (or a missing parent block, e.g. `timing` in a failed run) becomes an
  empty field.
- Booleans are written as `true` / `false`.
- Numbers use `.` as the decimal separator, with no thousands separators, at
  full precision.
- Column names are the JSON field name, prefixed with its block where the bare
  name would be ambiguous (e.g. `model_name`, `requested_device_type`). The
  headline latency statistics come from `timing.aggregate.stats` and are
  unprefixed.
- Columns appear in the order below. New columns are only appended at the end
  (a minor schema version bump).

| # | Column | Source (JSON Pointer) |
|---|---|---|
| 1 | `schema_version` | `/schema_version` |
| 2 | `run_id` | `/run_id` |
| 3 | `timestamp_utc` | `/timestamp_utc` |
| 4 | `status` | `/status` |
| 5 | `label` | `/label` |
| 6 | `error_stage` | `/error/stage` |
| 7 | `warning_count` | number of entries in `/warnings` (derived) |
| 8 | `edgebench_version` | `/tool/version` |
| 9 | `git_commit` | `/tool/git_commit` |
| 10 | `device_model` | `/system/device/model` |
| 11 | `cpu_name` | `/system/cpu/name` |
| 12 | `os_build` | `/system/os/build` |
| 13 | `process_architecture` | `/system/process_architecture` |
| 14 | `power_source` | `/system/power/source` |
| 15 | `onnxruntime_version` | `/runtime/onnxruntime/version` |
| 16 | `execution_provider` | `/runtime/execution_provider/name` |
| 17 | `execution_provider_version` | `/runtime/execution_provider/version` |
| 18 | `device_type` | `/runtime/ep_device/device_type` |
| 19 | `model_name` | `/model/name` |
| 20 | `model_sha256` | `/model/sha256` |
| 21 | `stored_precision` | `/model/stored_precision` |
| 22 | `quantization_method` | `/model/quantization/method` |
| 23 | `requested_execution_provider` | `/config/requested/execution_provider` |
| 24 | `requested_device_type` | `/config/requested/device_type` |
| 25 | `strict_mode` | `/config/strict_mode` |
| 26 | `full_device_execution_verified` | `/execution/full_device_execution_verified` |
| 27 | `batch_size` | `/config/batch_size` |
| 28 | `warmup_iterations` | `/config/warmup_iterations` |
| 29 | `measured_iterations` | `/config/measured_iterations` |
| 30 | `trials` | `/config/trials` |
| 31 | `session_create_ms` | `/timing/session_init/session_create_ms` |
| 32 | `ep_context_state` | `/timing/session_init/ep_context_state` |
| 33 | `mean_ms` | `/timing/aggregate/stats/mean_ms` |
| 34 | `p50_ms` | `/timing/aggregate/stats/p50_ms` |
| 35 | `p95_ms` | `/timing/aggregate/stats/p95_ms` |
| 36 | `p99_ms` | `/timing/aggregate/stats/p99_ms` |
| 37 | `stddev_ms` | `/timing/aggregate/stats/stddev_ms` |
| 38 | `min_ms` | `/timing/aggregate/stats/min_ms` |
| 39 | `max_ms` | `/timing/aggregate/stats/max_ms` |
| 40 | `inferences_per_sec` | `/timing/aggregate/stats/inferences_per_sec` |
| 41 | `items_per_sec` | `/timing/aggregate/stats/items_per_sec` |
| 42 | `trial_mean_cv_ratio` | `/timing/aggregate/trial_variation/trial_mean_cv_ratio` |
| 43 | `peak_working_set_bytes` | `/memory/peak_working_set_bytes` |

## Versioning

`schema_version` follows semantic versioning:

- **Patch** (`1.0.x`): documentation or schema fixes that do not change which
  documents are valid in practice (e.g. clarified descriptions, a corrected
  pattern that was meant to accept a value).
- **Minor** (`1.x.0`): backwards-compatible additions. New fields (nullable,
  so writers that cannot fill them set `null`), new enum values, new CSV
  columns appended at the end. Every document valid under `1.y` stays valid
  under `1.(y+1)`; the schema's `schema_version` enum lists all accepted
  versions.
- **Major** (`x.0.0`): anything that changes the meaning or shape of existing
  data: renaming, removing or retyping a field, changing a unit or a
  statistic's definition, removing an enum value. Readers must check the major
  version and reject versions they do not support. A new major version gets a
  new schema file.

Because objects reject unknown keys, a document written to a newer minor
version will fail validation against an older schema. Validate with the schema
from the same or a newer EdgeBench version.

## Storage

Result files live under `benchmarks/results/`:

- `benchmarks/results/published/`: committed, curated runs that pass validation.
- `benchmarks/results/local/`: gitignored, for scratch and experimental runs.

Validate everything that is published with:

```powershell
.\.venv\Scripts\python.exe scripts\validate_results.py benchmarks\results\published
```
