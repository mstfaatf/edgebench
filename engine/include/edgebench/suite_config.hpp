// Suite configuration files (`edgebench suite CONFIG`, docs/suite.md): parsing,
// validation, expansion into configurations, the run order, and run IDs.
//
// A configuration file (schemas/suite-config.v1.0.0.schema.json) names models,
// configurations (an explicit list, or a matrix expanded as a cartesian product
// with include / exclude lists), run parameters with defaults and overrides, the
// number of repetitions, the rotation and the cooldown between runs. Loading it
// resolves every configuration to a model file and to the `edgebench run`
// configuration it stands for, and checks that configuration with the same rules
// as `edgebench run` (validate(), requested_compute_precision()), so a suite can
// never plan a run that `run` would refuse. Everything here is pure except that
// model files are read (SHA256, ONNX header) and models/manifest.json is read for
// models named by their manifest name.
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/result.hpp"
#include "edgebench/run_config.hpp"

namespace edgebench::suite {

inline constexpr std::string_view kConfigVersion = "1.0.0";
inline constexpr std::string_view kConfigVersionKey = "suite_config_version";
/// Version of the run-ID derivation (part of the hashed object).
inline constexpr int kRunIdVersion = 1;
/// Characters of a run ID shown in tables and used in directory names.
inline constexpr std::size_t kShortIdLength = 12;
inline constexpr int kMaxRepetitions = 1000;
inline constexpr double kMaxCooldownS = 3600.0;

/// How a run's sessions are initialized. cold: every session is compiled from the
/// model (--ep-cache off; the only mode on the CPU); build: the run rebuilds the
/// EP context cache, then creates its timed sessions from it (--ep-cache build);
/// cached: the suite builds the cache entry in a separate short run right before,
/// and the timed run must load it (--ep-cache use).
enum class InitMode { cold, build, cached };
enum class Rotation { rotated, blocked };

std::string_view to_string(InitMode mode);
std::string_view to_string(Rotation rotation);

/// Run parameters a configuration may override (docs/suite.md, "Run parameters").
struct RunParams {
    int warmup = 20;
    int iterations = 200;
    int trials = 3;
    int trial_cooldown_ms = 2000;  // `run --cooldown-ms`: idle before every trial
    int load_sample_ms = 1000;
    int threads = 4;
    RunBinding binding = RunBinding::iobinding_preallocated;
    InitMode init = InitMode::cold;
    bool init_breakdown = false;
    bool profile_accelerator = false;
    std::optional<int> profile_iterations;
    double cooldown_s = 5.0;  // the suite's idle period before the run
    ProcessPriority process_priority = ProcessPriority::normal;
    bool operator==(const RunParams&) const = default;
};

/// What a configuration measures (the keys `match` objects compare).
struct Identity {
    std::string model;
    ProviderKind provider = ProviderKind::cpu;
    std::string precision;  // stored precision of the model file: fp32 | int8
    int batch = 1;
    std::optional<std::string> variant;  // model file variant; default fp32 / int8
    bool npu_fp16 = false;
    std::optional<std::string> htp_performance_mode;
    std::optional<std::string> htp_graph_finalization_mode;
    bool strict = true;
    bool operator==(const Identity&) const = default;
};

/// A configuration resolved to its model file and its `edgebench run` settings.
struct ResolvedConfiguration {
    Identity identity;
    RunParams params;
    std::size_t source_index = 0;      // position in the expanded list (0-based)
    std::string source;                // e.g. "configurations[2]", "matrix", "matrix.include[0]"
    std::filesystem::path model_file;  // absolute
    std::string model_file_display;    // relative to the models directory, or as configured
    std::string model_sha256;
    std::uint64_t model_size_bytes = 0;
    std::optional<std::string> variant;  // the file's variant, if known
    std::string compute_precision;       // fp32 | int8 | fp16
    /// compile_relevant_ep_options() of the run's QNN EP options (empty on the CPU).
    std::map<std::string, std::string> ep_compile_options;
    std::string label;                // e.g. "conv1d_autoencoder_b1_npu_int8"
    std::string configuration_id;     // SHA256 of the resolved configuration (no repetition)
    nlohmann::ordered_json resolved;  // what configuration_id hashes (also in the manifest)
    double sustained_duration_s = 0;  // sustained-load legs only
};

struct PlannedRun {
    std::string run_id;             // SHA256 of the resolved configuration and the repetition
    std::size_t configuration = 0;  // index into SuitePlan::configurations
    int repetition = 0;             // 0-based
    int order_index = 0;            // 0-based position in the run order
    std::string short_id() const { return run_id.substr(0, kShortIdLength); }
};

/// The checks of the pre-flight gate, in the order they run (the default list)
/// and the ones a suite must pass to be publishable.
const std::vector<std::string>& gate_check_names();

/// One continuous run of the sustained-load check, as configured.
struct SustainedLegSpec {
    Identity identity;  // model (manifest name), provider, precision, batch, NPU options
    std::optional<std::string> path;  // a model file instead of a manifest name
    double duration_s = 0;
    bool operator==(const SustainedLegSpec&) const = default;
};

/// The pre-flight gate and its thresholds (configuration: `preflight`).
struct PreflightSettings {
    std::vector<std::string> checks = gate_check_names();
    bool fail_on_warn = false;
    double idle_window_s = 30;
    double idle_max_cpu_ratio = 0.05;
    double sleep_margin_s = 1800;  // sleep/hibernate must not come within estimate + this
    double sustained_window_s = 30;
    double sustained_max_drift_ratio = 0.05;
    /// Default: ResNet-18 FP32 on the CPU for 3 min, then ResNet-18 INT8 on the NPU for 2 min.
    std::vector<SustainedLegSpec> sustained_legs = default_sustained_legs();
    static std::vector<SustainedLegSpec> default_sustained_legs();
    bool operator==(const PreflightSettings&) const = default;
};

/// The check before every run (configuration: `in_flight`).
struct InFlightSettings {
    double load_sample_s = 5;  // load sample before a run (during its cooldown if long enough)
    double max_wait_s = 600;   // longest wait for the load to drop (then the run starts anyway)
    double power_max_wait_s = 600;  // longest wait for AC + Best performance to come back
    /// "wait": wait up to power_max_wait_s for the power state, then stop; "stop": stop at once.
    std::string on_gate_loss = "wait";
    bool operator==(const InFlightSettings&) const = default;
};

struct SuitePlan {
    std::filesystem::path config_file;  // absolute
    std::string config_sha256;          // SHA256 of the canonical JSON of the file
    nlohmann::json config;              // the parsed file
    std::string name;
    std::filesystem::path output_root;  // absolute: where suite directories are created
    std::optional<std::filesystem::path> models_dir;  // absolute, if one was used
    int repetitions = 1;
    Rotation rotation = Rotation::rotated;
    PreflightSettings preflight;
    InFlightSettings in_flight;
    /// How many times a run whose result is flagged for background load or trial
    /// spread runs again after the main pass (configuration: retry_flagged).
    int retry_flagged = 1;
    /// The sustained-load legs, resolved like configurations (run parameters:
    /// one trial, 20 warm-up, no idle periods; iterations set at run time).
    std::vector<ResolvedConfiguration> sustained;
    std::vector<ResolvedConfiguration> configurations;
    std::vector<PlannedRun> runs;  // in run order
};

/// One problem in a configuration file. `pointer` is a JSON Pointer into the file
/// (empty for the file as a whole); `kind` is stable, for tests and messages.
struct SuiteConfigError {
    std::string pointer;
    std::string message;
    std::string kind = "suite_config_invalid";
};

/// usage_error (exit 2) listing every problem, prefixed with the file name.
Error to_error(const std::filesystem::path& config_file,
               const std::vector<SuiteConfigError>& errors);

/// Reads, validates, expands and resolves a configuration file. `output_dir`
/// replaces the file's output_dir if given. Errors are usage errors (exit 2).
Result<SuitePlan> load_suite(const std::filesystem::path& config_file,
                             const std::optional<std::filesystem::path>& output_dir = std::nullopt);

/// The same for an already parsed document; relative paths resolve against
/// `base_dir` (the configuration file's directory). Returns every problem found.
struct LoadOutcome {
    std::optional<SuitePlan> plan;
    std::vector<SuiteConfigError> errors;
};
LoadOutcome plan_suite(const nlohmann::json& doc, const std::filesystem::path& config_file,
                       const std::optional<std::filesystem::path>& output_dir = std::nullopt);

/// The `edgebench run` configuration of a resolved configuration, writing into
/// `output_dir` under `name`. For InitMode::cached this is the timed run
/// (--ep-cache use); cache_build_config() is the run that prepares it.
RunConfig to_run_config(const ResolvedConfiguration& c, const std::filesystem::path& output_dir,
                        const std::string& name,
                        const std::optional<std::filesystem::path>& ep_cache_dir);

/// The short run that builds the EP context cache entry of an InitMode::cached
/// configuration: same model and NPU options, --ep-cache build, one trial of one
/// iteration, no warm-up, no idle periods, no profiled pass.
RunConfig cache_build_config(const ResolvedConfiguration& c,
                             const std::filesystem::path& output_dir, const std::string& name,
                             const std::filesystem::path& ep_cache_dir);

/// `edgebench run` arguments for `config` (everything validate() checks, plus
/// --output-dir, --name, --label).
std::vector<std::string> run_arguments(const RunConfig& config);

/// The order of `repetitions` repetitions of `configurations` configurations, as
/// (configuration, repetition) pairs. blocked: every repetition of configuration
/// 0, then of 1, ... rotated: round r runs every configuration once, in row r of
/// a balanced (Williams) Latin square: over N rounds (2N for an odd N) every
/// configuration runs first equally often and directly follows every other
/// configuration equally often. Later rounds repeat the pattern.
std::vector<std::pair<std::size_t, int>> run_order(std::size_t configurations, int repetitions,
                                                   Rotation rotation);

/// SHA256 of canonical_json(`resolved` plus the repetition): the run ID.
std::string run_id(const nlohmann::ordered_json& resolved, int repetition);

/// SHA256 of the canonical JSON of a parsed configuration file (whitespace and
/// key order do not change it).
std::string config_hash(const nlohmann::json& doc);

/// Characters allowed in a suite name (as `run --name`).
bool is_valid_suite_name(std::string_view name);

}  // namespace edgebench::suite
