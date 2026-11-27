// The benchmark trial loop and its in-memory result.
//
// BenchmarkResult mirrors the results schema 1.2.0 (docs/results-schema.md):
// each member corresponds to a schema block, and every value that the schema
// records is kept here, including the raw per-iteration latencies and the
// per-trial idle samples. Accuracy is not measured by `run` (null).
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/batch_equivalence.hpp"
#include "edgebench/bundle_modules.hpp"
#include "edgebench/ep_context_cache.hpp"
#include "edgebench/model_info.hpp"
#include "edgebench/npu_options.hpp"
#include "edgebench/onnx_inspect.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/provider.hpp"
#include "edgebench/qnn_profile.hpp"
#include "edgebench/stats.hpp"
#include "edgebench/system_info.hpp"
#include "edgebench/tensor_info.hpp"
#include "edgebench/timer_info.hpp"
#include "edgebench/version.hpp"

namespace edgebench {

class Runtime;

/// warnings[]: {code, message, path (JSON Pointer)}.
struct ResultWarning {
    std::string code;
    std::string message;
    std::optional<std::string> path;
};

struct RuntimeResult {
    OrtLoadResult onnxruntime;     // version, library_path, api_version, load_check
    ProviderDescription provider;  // execution_provider, ep_device, ep_options, session_config
    std::vector<LoadedModule> loaded_modules;  // captured while the first session was alive
    std::optional<std::vector<EpDeviceInfo>> ep_devices;  // null if not enumerated
    /// runtime.qnn.backend_file_version: the loaded QnnHtp.dll, read while the first
    /// session was alive (QNN runs only).
    std::optional<ModuleFileVersion> qnn_backend_file_version;
};

struct ModelResult {
    std::string name;  // file name without extension
    ModelFileInfo file;
    std::optional<std::int64_t> opset, ir_version;
    ModelPrecision precision;
    std::optional<std::string> export_tool;      // producer name and version from the file
    std::optional<std::string> normalization;    // from the file's metadata, if recorded
    QuantizationRecord quantization_record;      // from the file's metadata, if recorded
    std::optional<std::string> precision_notes;  // e.g. FP32 stored, FP16 computed on the NPU
    std::vector<TensorInfo> inputs, outputs;
};

struct ExecutionResult {
    bool full_device_execution_verified = false;
    std::string verification_details;
    NodeAssignment node_assignment;  // verification_method is node_assignment.verification_method
    /// Every bundle module check of the run, in order (verification_details.module_checks).
    std::vector<ModuleCheckpoint> module_checks;
};

struct SessionInitResult {
    /// Nearest-rank median of the timed sessions' creation times (one per trial).
    double session_create_ms = 0;
    /// What the timed sessions were created from. CPU: not_used (no EP context
    /// cache); NPU: cold (from the ONNX model) or cached (from the EP context model).
    std::string ep_context_state = "not_used";
    std::vector<double> timed_ms;  // the timed sessions' creation times, in trial order
    /// Every creation from the ONNX model without a cache (--init-breakdown
    /// sessions, and the timed sessions when they are compiled), in order.
    std::vector<double> cold_ms;
    /// The creation that built the shared cache before the trials (null: none).
    std::optional<double> context_build_ms;
    /// Every creation from an EP context model (--init-breakdown sessions, and the
    /// timed sessions when they load the cache), in order; empty: none.
    std::vector<double> cached_ms;
};

/// timing.trials[].session_init: one value per kind of initialization, or null if
/// the trial did not measure it.
struct TrialSessionInit {
    std::optional<double> cold_ms;
    std::optional<double> context_build_ms;
    std::optional<double> cached_init_ms;
};

/// config.ep_context_cache of an NPU run.
struct EpContextCacheResult {
    EpCacheMode mode = EpCacheMode::off;
    std::optional<std::filesystem::path> cache_dir;  // absolute; null if no context was used
    /// disabled | generated | loaded | regenerated; null: the run failed before the
    /// cache was examined.
    std::optional<std::string> state;
    std::optional<std::string> key_id;
    std::optional<nlohmann::json> key;  // cache_key_json()
    std::optional<std::filesystem::path> context_path;
    std::vector<CacheFile> files;
    std::optional<bool> hit;
    std::optional<bool> verified_equal_outputs;
    /// Files deleted because their key differed for the same model (stale).
    std::vector<std::filesystem::path> deleted_stale;
    /// The context's compatibility info and the EP's verdict on it (build|use only).
    std::optional<ContextCompatibility> compatibility;
    bool enabled() const { return mode != EpCacheMode::off; }
};

/// timing.profiled_pass: one separate session with QNN profiling, created after
/// the trials and the placement session from the same model (or EP context) as
/// the timed sessions. Nothing here is part of the timed results.
struct ProfiledPassResult {
    int warmup_iterations = 0;  // run first, their profile entries dropped
    int iterations = 0;         // measured inferences requested
    std::string level = "basic";
    std::string source = "qnn_profiling_basic";
    QnnProfileSummary profile;  // times from the profile, warm-up removed
    /// Where the EP wrote its output (a temporary directory until the result files
    /// are written, which copy the files into the run directory).
    std::filesystem::path staging_dir;
    std::vector<std::filesystem::path> files;  // in staging_dir, sorted by name
    /// Accelerator execute time per measured inference, in order (may be empty).
    const std::vector<double>& accelerator_ms() const;
};

struct TrialResult {
    int trial_index = 0;
    int sequence = 0;  // position in the overall trial order (scheduler)
    /// Background CPU load sampled during the idle period before this trial
    /// (null if there was none: --cooldown-ms 0 after the first trial).
    std::optional<LoadSample> idle_before;
    /// Background CPU load across the measured iterations: system-wide utilization
    /// minus this process's own CPU time (background_cpu_utilization()). Not part of
    /// the results schema; used for warnings and the summary. Null if a counter
    /// could not be read.
    std::optional<LoadSample> during;
    double session_create_ms = 0;  // the timed session's creation
    std::optional<std::int64_t> session_create_working_set_delta_bytes;
    TrialSessionInit session_init;
    std::optional<double> first_inference_ms;  // first warm-up iteration; null without warm-up
    LatencyStats stats;
    std::vector<double> latencies_ms;  // every measured iteration, in order
};

/// memory: process memory only (the schema's scope "process").
struct MemoryResult {
    std::string scope = "process";
    std::uint64_t peak_working_set_bytes = 0;  // after the last trial
    std::uint64_t peak_private_bytes = 0;
    std::string source = "GetProcessMemoryInfo";
    /// Nearest-rank median of the per-trial deltas (null if none was measured).
    std::optional<std::int64_t> session_create_working_set_delta_bytes;
    std::uint64_t working_set_after_last_trial_bytes = 0;  // not in the schema; for the summary
};

struct AggregateResult {
    LatencyStats stats;  // all measured iterations of all trials pooled
    TrialVariation trial_variation;
};

struct TimingResult {
    TimerInfo timer;
    std::string percentile_method = "nearest_rank";
    std::string scope = "model_execution";
    SessionInitResult session_init;
    std::optional<double> first_inference_ms;  // trial 0's first warm-up iteration
    std::vector<TrialResult> trials;
    std::optional<AggregateResult> aggregate;
    /// --profile-accelerator only. timing.accelerator (superseded by it) stays null.
    std::optional<ProfiledPassResult> profiled_pass;
};

/// Why a run failed (results schema: error).
struct RunFailure {
    Error error;        // exit code, kind, message, ORT error code
    std::string stage;  // error.stage: environment, model_load, session_creation, warmup, ...
};

/// Which parts of a result were established before the run ended. A failed run
/// keeps what it collected; the exporter writes null for the rest.
struct KnownParts {
    bool runtime = false;    // ONNX Runtime was loaded
    bool provider = false;   // the provider was created (describe() is valid)
    bool model = false;      // the model file was read and decoded
    bool session = false;    // at least one session was created
    bool placement = false;  // node placement was recorded
};

struct BenchmarkResult {
    std::string schema_version;
    std::string run_id;
    std::chrono::system_clock::time_point started_at;
    std::string timestamp_utc;  // started_at, RFC 3339
    std::optional<std::string> label;
    std::optional<RunFailure> failure;  // set: status "failed"
    KnownParts known;
    std::vector<ResultWarning> warnings;
    /// notes[]: facts that do not limit how the result may be used (same shape).
    std::vector<ResultWarning> notes;
    /// execution.verification_details.background_load_samples: every load sample
    /// above kHighBackgroundLoad, with its per-process attribution.
    struct BackgroundLoadRecord {
        std::string phase;  // idle_before_trial | measured_iterations
        int trial_index = 0;
        LoadSample sample;
    };
    std::vector<BackgroundLoadRecord> background_load_samples;

    std::string status() const { return failure ? "failed" : "success"; }

    std::string tool_version;
    BuildInfo tool;
    SystemInfo system;        // power: at the start of the run; load: the pre-run idle sample
    PowerState power_at_end;  // compared with system.power for the power_state_changed warning
    std::optional<ProcessMemory> memory_before_placement;  // after the last trial's session
    RuntimeResult runtime;
    ModelResult model;
    std::optional<OnnxGraphSummary> graph;  // decoded model file

    RunConfig config;  // as requested; the schema's config block derives from it
    /// config.requested.compute_precision: null for CPU runs; "int8" or "fp16" on the
    /// NPU (requested_compute_precision()).
    std::optional<std::string> compute_precision;
    /// execution.effective_precision (effective_precision()); null if unknown.
    std::optional<EffectivePrecision> effective_precision;
    /// execution.batch_equivalence, looked up once the batch size is known.
    std::optional<BatchEquivalence> batch_equivalence;
    std::int64_t batch_size = 0;      // 0: unknown (written as null; failed results only)
    std::string input_preprocessing;  // config.input.preprocessing

    ExecutionResult execution;
    TimingResult timing;
    MemoryResult memory;
    EpContextCacheResult ep_cache;  // NPU runs only
};

/// Receives progress messages ("idle 2000 ms before trial 1 of 3", "trial 1 of 3: ...").
using ProgressFn = std::function<void(std::string_view)>;

/// The points where a run checks for a stop request (stop_requested()).
enum class StopPoint {
    idle,              // the idle period before a trial
    before_session,    // before a trial's session is created
    after_session,     // after it was created
    warmup,            // before a warm-up iteration
    measured,          // before a measured iteration (outside the timed bracket)
    before_placement,  // before the node placement session
    after_placement,   // after it returned
    profiled_pass,     // before each inference of the --profile-accelerator session
};

/// Called at every stop point, just before the stop flag is checked (tests use it
/// to raise the flag at a chosen point). `iteration` is 0 where it does not apply.
using StopHook = std::function<void(StopPoint, int trial_index, int iteration)>;

/// A result for `config` before anything runs: identity, tool, system, the
/// power state, and the model file's facts (its hash and decoded metadata). A
/// model file that cannot be read sets `failure` (stage model_load).
BenchmarkResult begin_result(const RunConfig& config);

/// Marks `result` failed.
void set_failure(BenchmarkResult& result, Error error, std::string stage);

/// Fills in what is derived at the end of a run, successful or not: statistics
/// of the completed trials, process memory, the power state at the end, and the
/// warnings for changed power state, high background load and unknown values.
void finish_result(BenchmarkResult& result);

/// The failed result of a run whose ONNX Runtime could not be loaded (stage
/// environment, code = the loader's error kind).
BenchmarkResult failed_load_result(const RunConfig& config, const OrtLoadResult& load);

/// Creates the provider for `config` and runs the benchmark with it. For the NPU
/// the model's precision is checked first (requested_compute_precision()), so a
/// usage error is reported before the QNN EP is registered.
BenchmarkResult run_benchmark(Runtime& runtime, const RunConfig& config,
                              const ProgressFn& progress = {}, const StopHook& hook = {});

/// Runs the benchmark described by `config` with `provider`:
///
/// record the power state; for every trial (scheduler order): idle for
/// --cooldown-ms while sampling background CPU load (before the first trial too,
/// for 1 s if the cooldown is 0, so every trial starts from the same idle state),
/// create a fresh session (timed: cold_ms; working-set delta), time the first
/// warm-up iteration, run the other warm-up iterations, time each measured
/// iteration (run_once() only), check that every output value is finite and
/// matches trial 0, release the session. After the last trial, read the process
/// memory, record node placement in a separate session, and compare the power
/// state with the one at the start.
///
/// In strict mode every node must be placed on the requested EP; otherwise the
/// run fails with internal_error (strict_placement_mismatch), because ONNX
/// Runtime should have refused the session. Without strict mode, nodes on the
/// CPU are a cpu_fallback_nodes warning.
///
/// Never throws for run failures: the result's `failure` is set and everything
/// collected so far is kept. run_once() failures are ort_runtime_error naming
/// trial and iteration (stage warmup or measurement); non-finite outputs, and
/// outputs whose top-1 differs from trial 0, are internal_error naming output
/// and trial (stage measurement); session creation failures keep their exit code
/// (usage_error for unsupported models, ort_runtime_error otherwise).
///
/// Interrupts (interrupt.hpp) are honoured at every StopPoint: the run ends there
/// with error kind "interrupted" (exit code interrupted), no placement session,
/// and only the trials that completed; a trial cut off partway is dropped with
/// an interrupted_trial_discarded warning.
BenchmarkResult run_benchmark(Runtime& runtime, Provider& provider, const RunConfig& config,
                              const ProgressFn& progress = {}, const StopHook& hook = {});

/// Human-readable summary of a result (what `edgebench run` prints).
std::string format_summary(const BenchmarkResult& result);

/// config.input.preprocessing for generated inputs.
std::string generated_input_description(const RunConfig& config);

/// Idle period before each trial: --cooldown-ms, or --load-sample-ms before the
/// first trial if the cooldown is 0 (then none between trials).
std::chrono::milliseconds idle_before_trial(const RunConfig& config, int sequence);

/// The note of timing.profiled_pass.
inline constexpr std::string_view kProfiledPassNote =
    "Accelerator time from a separate session with QNN profiling (basic), created after the "
    "timed trials; profiling adds host-side overhead to that session, so this is the "
    "accelerator's share of an inference, not a latency, and must not be compared with the "
    "timed wall-clock latency as if they measured the same thing.";

/// Background CPU load above which a run gets a background_cpu_load_high warning.
inline constexpr double kHighBackgroundLoad = 0.10;
/// The shortest load sample judged against kHighBackgroundLoad. The system and
/// process CPU counters advance in clock ticks (15.6 ms by default), so over a
/// window of a few ticks a single tick of another process is several percent:
/// the measured iterations of a fast model (tens of milliseconds) cannot tell
/// 10% background load from noise. Shorter samples are still shown in the
/// summary; the idle samples before every trial (1 s by default) catch sustained
/// load either way.
inline constexpr double kMinJudgedLoadWindowS = 0.5;
/// Whether `s` is long enough to be judged and above kHighBackgroundLoad.
inline bool background_load_too_high(const LoadSample& s) {
    return s.sample_window_s >= kMinJudgedLoadWindowS &&
           s.cpu_utilization_ratio > kHighBackgroundLoad;
}

}  // namespace edgebench
