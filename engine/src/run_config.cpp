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
        {"config", config},
    };
}

}  // namespace edgebench
