#include "edgebench/run_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <system_error>

#include "edgebench/model_info.hpp"
#include "edgebench/npu_options.hpp"
#include "edgebench/ort_loader.hpp"

namespace fs = std::filesystem;

namespace edgebench {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

std::vector<ConfigError> validate(const RunConfig& c) {
    std::vector<ConfigError> errors;
    auto add = [&errors](std::string field, std::string message) {
        errors.push_back({std::move(field), std::move(message)});
    };

    std::error_code ec;
    if (c.model.empty()) {
        add("model", "is required");
    } else if (!fs::exists(c.model, ec)) {
        add("model", "file not found: " + path_to_utf8(c.model));
    } else if (!fs::is_regular_file(c.model, ec)) {
        add("model", "not a regular file: " + path_to_utf8(c.model));
    } else if (lower(path_to_utf8(c.model.extension())) != ".onnx") {
        add("model", "must be an .onnx file: " + path_to_utf8(c.model));
    } else if (Result<void> readable = check_file_readable(c.model); !readable) {
        errors.push_back({"model", readable.error().message, readable.error().kind});
    }

    auto range = [&add](const char* field, int value, int lo, int hi) {
        if (value < lo || value > hi) {
            add(field, "must be between " + std::to_string(lo) + " and " + std::to_string(hi) +
                           " (got " + std::to_string(value) + ")");
        }
    };
    range("threads", c.threads, 1, kMaxThreads);
    range("warmup", c.warmup, 0, kMaxWarmup);
    range("iterations", c.iterations, 1, kMaxIterations);
    range("trials", c.trials, 1, kMaxTrials);
    range("cooldown-ms", c.cooldown_ms, 0, kMaxCooldownMs);
    range("load-sample-ms", c.load_sample_ms, 0, kMaxLoadSampleMs);
    if (c.name && !is_valid_result_name(*c.name)) {
        add("name",
            "may only contain letters, digits, '-', '_' and '.', and must not start "
            "with '.' (got '" +
                *c.name + "')");
    }

    if (c.seed && c.input != InputKind::random)
        add("seed", "is only allowed with --input random");
    if (c.constant_value && c.input != InputKind::constant)
        add("constant-value", "is only allowed with --input constant");
    if (c.constant_value && !std::isfinite(*c.constant_value))
        add("constant-value", "must be a finite number");

    if (c.provider != ProviderKind::npu) {
        const std::string only_npu = "is only allowed with --provider npu";
        if (c.npu_fp16)
            add("npu-fp16", only_npu);
        if (!c.strict)
            add("no-strict", only_npu + " (the CPU provider has no fallback to disable)");
        if (c.htp_performance_mode)
            add("htp-performance-mode", only_npu);
        if (c.ep_cache != EpCacheMode::off)
            add("ep-cache", only_npu + " (the CPU provider has no EP context cache)");
        if (c.init_breakdown)
            add("init-breakdown", only_npu + " (the CPU provider has no EP context cache)");
    }
    if (c.ep_cache_dir && !c.uses_ep_context())
        add("ep-cache-dir", "is only used with --ep-cache build|use or --init-breakdown");
    if (c.ep_cache_dir && c.ep_cache_dir->empty())
        add("ep-cache-dir", "must not be empty");
    if (c.htp_performance_mode && !is_htp_performance_mode(*c.htp_performance_mode)) {
        add("htp-performance-mode", "unsupported mode '" + *c.htp_performance_mode +
                                        "'; supported modes: " + htp_performance_modes_list());
    }

    if (c.label && c.label->empty())
        add("label", "must not be empty");
    if (c.output_dir.empty())
        add("output-dir", "must not be empty");
    return errors;
}

bool is_valid_result_name(std::string_view name) {
    if (name.empty() || name.size() > 128 || name.front() == '.')
        return false;
    return std::all_of(name.begin(), name.end(), [](char ch) {
        const auto c = static_cast<unsigned char>(ch);
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '-' || c == '_' || c == '.';
    });
}

std::string default_result_name(const RunConfig& c) {
    std::string stem = path_to_utf8(c.model.stem());
    for (char& ch : stem) {
        const auto u = static_cast<unsigned char>(ch);
        const bool ok = (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
                        (u >= '0' && u <= '9') || u == '-' || u == '_' || u == '.';
        if (!ok)
            ch = '_';
    }
    if (stem.empty() || stem.front() == '.')
        stem = "model" + stem;
    return stem + "_" + std::string(cli_name(c.provider)) + "_" + std::string(cli_name(c.binding)) +
           "_t" + std::to_string(c.threads);
}

std::string result_name(const RunConfig& c) {
    return c.name ? *c.name : default_result_name(c);
}

std::optional<OutputFormat> parse_output_format(std::string_view text) {
    if (text == "summary")
        return OutputFormat::summary;
    if (text == "json")
        return OutputFormat::json;
    return std::nullopt;
}

std::optional<EpCacheMode> parse_ep_cache_mode(std::string_view text) {
    if (text == "off")
        return EpCacheMode::off;
    if (text == "build")
        return EpCacheMode::build;
    if (text == "use")
        return EpCacheMode::use;
    return std::nullopt;
}

std::string_view schema_name(EpCacheMode mode) {
    switch (mode) {
        case EpCacheMode::off:
            return "off";
        case EpCacheMode::build:
            return "build";
        case EpCacheMode::use:
            return "use";
    }
    return "off";
}

std::string_view cli_name(RunBinding binding) {
    return binding == RunBinding::run ? "run" : "iobinding";
}

std::string_view cli_name(ProviderKind provider) {
    return provider == ProviderKind::npu ? "npu" : "cpu";
}

std::optional<ProviderKind> parse_provider(std::string_view text) {
    if (text == "cpu")
        return ProviderKind::cpu;
    if (text == "npu")
        return ProviderKind::npu;
    return std::nullopt;
}

std::optional<RunBinding> parse_binding(std::string_view text) {
    if (text == "run")
        return RunBinding::run;
    if (text == "iobinding")
        return RunBinding::iobinding_preallocated;
    return std::nullopt;
}

std::optional<InputKind> parse_input_kind(std::string_view text) {
    if (text == "random")
        return InputKind::random;
    if (text == "constant")
        return InputKind::constant;
    return std::nullopt;
}

std::optional<GraphOptLevel> parse_graph_opt(std::string_view text) {
    if (text == "disable")
        return GraphOptLevel::disable_all;
    if (text == "basic")
        return GraphOptLevel::enable_basic;
    if (text == "extended")
        return GraphOptLevel::enable_extended;
    if (text == "all")
        return GraphOptLevel::enable_all;
    return std::nullopt;
}

std::string_view supported_providers() {
    return "cpu, npu";
}

std::string_view schema_name(RunBinding binding) {
    return binding == RunBinding::run ? "run" : "iobinding_preallocated";
}

std::string_view schema_name(InputKind kind) {
    return kind == InputKind::random ? "random" : "constant";
}

std::string_view schema_name(GraphOptLevel level) {
    switch (level) {
        case GraphOptLevel::disable_all:
            return "disable_all";
        case GraphOptLevel::enable_basic:
            return "enable_basic";
        case GraphOptLevel::enable_extended:
            return "enable_extended";
        case GraphOptLevel::enable_all:
            return "enable_all";
    }
    return "enable_all";
}

std::string_view execution_provider_name(ProviderKind provider) {
    return provider == ProviderKind::npu ? "QNNExecutionProvider" : "CPUExecutionProvider";
}

std::string_view device_type_name(ProviderKind provider) {
    return provider == ProviderKind::npu ? "NPU" : "CPU";
}

bool strict_mode(const RunConfig& config) {
    return config.provider == ProviderKind::npu && config.strict;
}

nlohmann::ordered_json resolved_config_json(const RunConfig& c) {
    using json = nlohmann::ordered_json;
    std::error_code ec;
    const fs::path model = fs::absolute(c.model, ec);
    const fs::path output_dir = fs::absolute(c.output_dir, ec);
    const auto size = fs::file_size(model, ec);

    json input = {
        {"kind", schema_name(c.input)},
        {"constant_value",
         c.input == InputKind::constant ? json(c.effective_constant_value()) : json(nullptr)},
        {"seed", c.input == InputKind::random ? json(c.effective_seed()) : json(nullptr)},
        {"dataset", nullptr},
    };
    json config = {
        {"requested",
         {{"execution_provider", execution_provider_name(c.provider)},
          {"device_type", device_type_name(c.provider)},
          {"compute_precision", nullptr}}},
        {"strict_mode", strict_mode(c)},
        {"batch_size", nullptr},
        {"input", input},
        {"warmup_iterations", c.warmup},
        {"measured_iterations", c.iterations},
        {"trials", c.trials},
        {"run_api", "sync"},
        {"ort_execution_mode", "sequential"},
        {"intra_op_threads", c.threads},
        {"inter_op_threads", nullptr},
        {"graph_optimization_level", schema_name(c.graph_opt)},
        {"ep_options", json::object()},
        {"session_config", json::object()},
        {"ep_context_cache", c.provider == ProviderKind::npu ? json{{"enabled", false},
                                                                    {"cache_file", nullptr},
                                                                    {"state", "disabled"},
                                                                    {"files", json::array()},
                                                                    {"key", nullptr}}
                                                             : json(nullptr)},
        {"run_binding", schema_name(c.binding)},
        {"profiling", {{"ort", false}, {"qnn_level", "off"}}},
    };
    return {
        {"model",
         {{"path", path_to_utf8(model)},
          {"file_name", path_to_utf8(model.filename())},
          {"size_bytes", ec ? json(nullptr) : json(size)}}},
        {"label", c.label ? json(*c.label) : json(nullptr)},
        {"output_dir", path_to_utf8(output_dir)},
        {"cooldown_ms", c.cooldown_ms},
        {"load_sample_ms", c.load_sample_ms},
        {"name", result_name(c)},
        {"config", config},
    };
}

}  // namespace edgebench
