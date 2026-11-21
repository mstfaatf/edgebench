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

enum class ProviderKind { cpu };
enum class RunBinding { run, iobinding_preallocated };
enum class InputKind { random, constant };
enum class GraphOptLevel { disable_all, enable_basic, enable_extended, enable_all };

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
    InputKind input = InputKind::random;
    std::optional<std::uint64_t> seed;
    std::optional<double> constant_value;
    GraphOptLevel graph_opt = GraphOptLevel::enable_all;
    std::optional<std::string> label;
    std::filesystem::path output_dir = "benchmarks/results/local";
    bool dry_run = false;

    std::uint64_t effective_seed() const { return seed.value_or(0); }
    double effective_constant_value() const { return constant_value.value_or(0.0); }
};

/// One validation problem. `field` is the option name without dashes, e.g. "threads".
struct ConfigError {
    std::string field;
    std::string message;
    friend bool operator==(const ConfigError&, const ConfigError&) = default;
};

/// All problems with `config`; empty if it is valid. Checks the model file on disk.
std::vector<ConfigError> validate(const RunConfig& config);

// CLI spellings (`--provider cpu`, `--binding iobinding`, ...) <-> values.
std::optional<ProviderKind> parse_provider(std::string_view text);
std::optional<RunBinding> parse_binding(std::string_view text);
std::optional<InputKind> parse_input_kind(std::string_view text);
std::optional<GraphOptLevel> parse_graph_opt(std::string_view text);

/// Comma-separated list of the accepted `--provider` values, e.g. "cpu".
std::string_view supported_providers();

// Results-schema spellings (config.run_binding, config.graph_optimization_level, ...).
std::string_view schema_name(RunBinding binding);
std::string_view schema_name(InputKind kind);
std::string_view schema_name(GraphOptLevel level);
std::string_view execution_provider_name(ProviderKind provider);  // e.g. "CPUExecutionProvider"
std::string_view device_type_name(ProviderKind provider);         // e.g. "CPU"

/// The resolved configuration, with every default filled in. `config` holds the
/// fields of the results schema's config block under the schema's names
/// (batch_size is null: it comes from the model's input shape when the session is
/// created); the other keys hold the run settings the schema has no field for.
nlohmann::ordered_json resolved_config_json(const RunConfig& config);

}  // namespace edgebench
