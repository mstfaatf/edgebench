// Configuration of `edgebench run`, independent of how it was parsed.
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace edgebench {

enum class ProviderKind { cpu, npu };

// Upper bounds of the run options (docs/cli.md); larger values are usage errors.
inline constexpr int kMaxIterations = 1'000'000;
inline constexpr int kMaxTrials = 1'000;
inline constexpr int kMaxWarmup = 100'000;
inline constexpr int kMaxCooldownMs = 600'000;
inline constexpr int kMaxLoadSampleMs = 600'000;
inline constexpr int kMaxThreads = 256;
/// Inferences of the profiled accelerator pass unless --profile-iterations is given.
inline constexpr int kDefaultProfileIterations = 50;
enum class RunBinding { run, iobinding_preallocated };
enum class InputKind { random, constant };
enum class GraphOptLevel { disable_all, enable_basic, enable_extended, enable_all };
enum class OutputFormat { summary, json };
/// --ep-cache: off (compile every session from the ONNX model), build (rebuild the
/// EP context cache once before the trials, then create every timed session from
/// it), use (create the timed sessions from an existing cache with a matching key;
/// build it first if there is none).
enum class EpCacheMode { off, build, use };

/// Every value `edgebench run` uses. Defaults match the documented CLI defaults
/// (docs/cli.md). `seed` and `constant_value` are only set when given explicitly;
/// effective_seed() / effective_constant_value() apply the defaults.
struct RunConfig {
    std::filesystem::path model;
    ProviderKind provider = ProviderKind::cpu;
    int threads = 4;
    RunBinding binding = RunBinding::iobinding_preallocated;
    int warmup = 20;
    int iterations = 200;
    int trials = 3;
    int cooldown_ms = 2000;
    int load_sample_ms = 1000;  // background-load sample before trial 1 when cooldown_ms is 0
    InputKind input = InputKind::random;
    std::optional<std::uint64_t> seed;
    std::optional<double> constant_value;
    GraphOptLevel graph_opt = GraphOptLevel::enable_all;
    std::optional<std::string> label;
    std::filesystem::path output_dir = "benchmarks/results/local";
    std::optional<std::string> name;  // result file name; default: default_result_name()
    OutputFormat format = OutputFormat::summary;
    bool dry_run = false;

    // NPU only (--provider npu); usage errors with any other provider.
    bool strict = true;     // session.disable_cpu_ep_fallback = 1; --no-strict clears it
    bool npu_fp16 = false;  // --npu-fp16: FP32 model with FP16 compute
    std::optional<std::string> htp_performance_mode;  // default: burst
    EpCacheMode ep_cache = EpCacheMode::off;
    /// --ep-cache-dir; default: default_ep_cache_dir() (%LOCALAPPDATA%\EdgeBench\qnn-context).
    std::optional<std::filesystem::path> ep_cache_dir;
    /// --init-breakdown: per trial, time a cold, a context-building and a cached
    /// session creation before the timed session.
    bool init_breakdown = false;
    /// --profile-accelerator: after the trials and the placement session, run a
    /// separate session with QNN profiling and record its accelerator time
    /// (timing.profiled_pass). Nothing from it is part of the timed results.
    bool profile_accelerator = false;
    /// --profile-iterations: measured inferences of that pass (after --warmup
    /// unrecorded ones); only set when given.
    std::optional<int> profile_iterations;

    int effective_profile_iterations() const {
        return profile_iterations.value_or(kDefaultProfileIterations);
    }

    /// Whether the run creates EP context files (and so needs the cache directory).
    bool uses_ep_context() const { return ep_cache != EpCacheMode::off || init_breakdown; }

    std::uint64_t effective_seed() const { return seed.value_or(0); }
    double effective_constant_value() const { return constant_value.value_or(0.0); }
};

/// One validation problem. `field` is the option name without dashes, e.g. "threads".
struct ConfigError {
    std::string field;
    std::string message;
    std::string kind = "usage_error";  // error kind printed with it, e.g. "model_unreadable"
    friend bool operator==(const ConfigError&, const ConfigError&) = default;
};

/// All problems with `config`; empty if it is valid. Checks the model file on disk.
std::vector<ConfigError> validate(const RunConfig& config);

/// Characters allowed in --name: letters, digits, '-', '_' and '.'.
bool is_valid_result_name(std::string_view name);

/// <model stem>_<provider>_<binding>_t<threads>, e.g. "mobilenetv2_fp32_b1_cpu_iobinding_t4".
/// Characters of the model stem outside the --name set become '_'.
std::string default_result_name(const RunConfig& config);

/// config.name if set, otherwise default_result_name().
std::string result_name(const RunConfig& config);

// CLI spellings (`--provider cpu`, `--binding iobinding`, ...) <-> values.
std::optional<ProviderKind> parse_provider(std::string_view text);
std::optional<RunBinding> parse_binding(std::string_view text);
std::optional<InputKind> parse_input_kind(std::string_view text);
std::optional<GraphOptLevel> parse_graph_opt(std::string_view text);
std::optional<OutputFormat> parse_output_format(std::string_view text);
std::optional<EpCacheMode> parse_ep_cache_mode(std::string_view text);
std::string_view cli_name(RunBinding binding);     // "run" or "iobinding"
std::string_view cli_name(ProviderKind provider);  // "cpu" or "npu"

/// Comma-separated list of the accepted `--provider` values, e.g. "cpu, npu".
std::string_view supported_providers();

// Results-schema spellings (config.run_binding, config.graph_optimization_level, ...).
std::string_view schema_name(RunBinding binding);
std::string_view schema_name(InputKind kind);
std::string_view schema_name(GraphOptLevel level);
std::string_view schema_name(EpCacheMode mode);                   // "off", "build" or "use"
std::string_view execution_provider_name(ProviderKind provider);  // e.g. "CPUExecutionProvider"
std::string_view device_type_name(ProviderKind provider);         // e.g. "CPU"

/// True if CPU fallback is disabled (results schema: config.strict_mode): NPU runs
/// unless --no-strict was given. The CPU provider has no fallback to disable.
bool strict_mode(const RunConfig& config);

/// The resolved configuration, with every default filled in. `config` holds the
/// fields of the results schema's config block under the schema's names
/// (batch_size is null: it comes from the model's input shape when the session is
/// created); the other keys hold the run settings the schema has no field for.
nlohmann::ordered_json resolved_config_json(const RunConfig& config);

}  // namespace edgebench
