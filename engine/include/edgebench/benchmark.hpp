// The benchmark trial loop and its in-memory result.
//
// BenchmarkResult mirrors the results schema 1.1.0 (docs/results-schema.md):
// each member corresponds to a schema block, and every value that the schema
// records is kept here, including the raw per-iteration latencies. Blocks that
// are not measured yet (system power and load, memory, accuracy) are absent or
// null, and the result carries value_unavailable warnings for them.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/model_info.hpp"
#include "edgebench/onnx_inspect.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/provider.hpp"
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
    std::vector<EpDeviceInfo> ep_devices;
};

struct ModelResult {
    std::string name;  // file name without extension
    ModelFileInfo file;
    std::optional<std::int64_t> opset, ir_version;
    ModelPrecision precision;
    std::optional<std::string> export_tool;  // producer name and version from the file
    std::vector<TensorInfo> inputs, outputs;
};

struct ExecutionResult {
    bool full_device_execution_verified = false;
    std::string verification_details;
    NodeAssignment node_assignment;  // verification_method is node_assignment.verification_method
};

struct SessionInitResult {
    double session_create_ms = 0;                  // nearest-rank median of cold_ms
    std::string ep_context_state = "not_used";     // no EP context cache (CPU)
    std::vector<double> cold_ms;                   // one per trial, in trial order
    std::optional<double> context_build_ms;        // null: no cache
    std::optional<std::vector<double>> cached_ms;  // null: no cache
};

struct TrialResult {
    int trial_index = 0;
    int sequence = 0;  // position in the overall trial order (scheduler)
    double session_create_ms = 0;
    std::optional<double> first_inference_ms;  // first warm-up iteration; null without warm-up
    LatencyStats stats;
    std::vector<double> latencies_ms;  // every measured iteration, in order
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
    // timing.accelerator: always null here (no profiled accelerator pass).
};

struct BenchmarkResult {
    std::string schema_version;
    std::string run_id;
    std::string timestamp_utc;  // start of the run
    std::optional<std::string> label;
    std::string status = "success";
    std::vector<ResultWarning> warnings;

    std::string tool_version;
    BuildInfo tool;
    SystemInfo system;  // OS, CPU, memory, device; power and load are not queried yet
    RuntimeResult runtime;
    ModelResult model;

    RunConfig config;  // as requested; the schema's config block derives from it
    std::int64_t batch_size = 0;
    std::string input_preprocessing;  // config.input.preprocessing

    ExecutionResult execution;
    TimingResult timing;
};

/// Receives progress messages ("trial 1 of 3", "cooldown 2000 ms").
using ProgressFn = std::function<void(std::string_view)>;

/// Runs the benchmark described by `config` with `provider`:
///
/// for every trial (scheduler order): sleep --cooldown-ms (not before the first),
/// create a fresh session (timed: cold_ms), time the first warm-up iteration, run
/// the other warm-up iterations, time each measured iteration (run_once() only),
/// check that every output value is finite and matches trial 0, release the
/// session. After the last trial, record node placement in a separate session.
///
/// Errors: run_once() failures are ort_runtime_error naming trial and iteration;
/// non-finite outputs, and outputs whose top-1 differs from trial 0, are
/// internal_error naming output and trial.
Result<BenchmarkResult> run_benchmark(Runtime& runtime, Provider& provider, const RunConfig& config,
                                      const ProgressFn& progress = {});

/// Human-readable summary of a result (what `edgebench run` prints).
std::string format_summary(const BenchmarkResult& result);

/// config.input.preprocessing for generated inputs.
std::string generated_input_description(const RunConfig& config);

}  // namespace edgebench
