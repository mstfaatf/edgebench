#include "edgebench/run_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <system_error>

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
    }

    if (c.threads < 1)
        add("threads", "must be at least 1 (got " + std::to_string(c.threads) + ")");
    if (c.warmup < 0)
        add("warmup", "must be 0 or more (got " + std::to_string(c.warmup) + ")");
    if (c.iterations < 1)
        add("iterations", "must be at least 1 (got " + std::to_string(c.iterations) + ")");
    if (c.trials < 1)
        add("trials", "must be at least 1 (got " + std::to_string(c.trials) + ")");
    if (c.cooldown_ms < 0)
        add("cooldown-ms", "must be 0 or more (got " + std::to_string(c.cooldown_ms) + ")");
    if (c.load_sample_ms < 0)
        add("load-sample-ms", "must be 0 or more (got " + std::to_string(c.load_sample_ms) + ")");
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

std::string_view cli_name(RunBinding binding) {
    return binding == RunBinding::run ? "run" : "iobinding";
}

std::string_view cli_name(ProviderKind) {
    return "cpu";
}

std::optional<ProviderKind> parse_provider(std::string_view text) {
    if (text == "cpu")
        return ProviderKind::cpu;
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
    return "cpu";
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

std::string_view execution_provider_name(ProviderKind) {
    return "CPUExecutionProvider";
}

std::string_view device_type_name(ProviderKind) {
    return "CPU";
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
        {"strict_mode", false},
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
        {"ep_context_cache", nullptr},
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
