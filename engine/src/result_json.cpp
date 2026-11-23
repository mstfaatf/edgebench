#include "edgebench/result_json.hpp"

#include <sstream>

namespace edgebench {

namespace {

using json = nlohmann::ordered_json;

json opt(const std::optional<std::string>& v) {
    return v ? json(*v) : json(nullptr);
}

template <typename T>
json opt(const std::optional<T>& v) {
    return v ? json(*v) : json(nullptr);
}

json stats_json(const LatencyStats& s) {
    return {
        {"count", s.count},
        {"mean_ms", s.mean_ms},
        {"p50_ms", s.p50_ms},
        {"p95_ms", opt(s.p95_ms)},
        {"p99_ms", opt(s.p99_ms)},
        {"stddev_ms", opt(s.stddev_ms)},
        {"min_ms", s.min_ms},
        {"max_ms", s.max_ms},
        {"inferences_per_sec", s.inferences_per_sec},
        {"items_per_sec", s.items_per_sec},
    };
}

json tensors_json(const std::vector<TensorInfo>& tensors) {
    json out = json::array();
    for (const TensorInfo& t : tensors) {
        json shape = json::array();
        for (std::size_t d = 0; d < t.shape.size(); ++d) {
            const std::string param = d < t.dim_params.size() ? t.dim_params[d] : "";
            if (!param.empty())
                shape.push_back(param);
            else if (t.shape[d] >= 0)
                shape.push_back(t.shape[d]);
            else
                shape.push_back(nullptr);
        }
        out.push_back({{"name", t.name}, {"dtype", dtype_name(t.dtype)}, {"shape", shape}});
    }
    return out;
}

json error_json(const BenchmarkResult& r) {
    if (!r.failure)
        return nullptr;
    const Error& e = r.failure->error;
    return {
        {"stage", r.failure->stage},
        {"message", e.message},
        {"code", e.ort_error_name ? json(*e.ort_error_name)
                                  : (e.kind.empty() ? json(nullptr) : json(e.kind))},
    };
}

json runtime_json(const BenchmarkResult& r) {
    if (!r.known.runtime)
        return nullptr;
    const OrtLoadResult& ort = r.runtime.onnxruntime;
    const ProviderDescription& p = r.runtime.provider;
    json execution_provider = nullptr;  // no provider was created
    json ep_device = nullptr;
    if (r.known.provider) {
        execution_provider = {{"name", p.execution_provider},
                              {"version", opt(p.ep_version)},
                              {"library_path", opt(p.ep_library_path)}};
        if (p.ep_device) {
            const EpDeviceInfo& d = p.ep_device->device;
            ep_device = {
                {"ep_name", d.ep_name},
                {"ep_vendor", d.ep_vendor},
                {"device_type", d.device_type},
                {"vendor", d.vendor},
                {"vendor_id", d.vendor_id},
                {"device_id", d.device_id},
                {"driver_version", opt(p.ep_device->driver_version)},
                {"metadata", d.metadata},
                {"selection",
                 {{"method", p.ep_device->selection_method},
                  {"candidate_count", p.ep_device->candidate_count}}},
            };
        }
    }
    json modules = json::array();
    for (const LoadedModule& m : r.runtime.loaded_modules)
        modules.push_back({{"name", m.name}, {"path", path_to_utf8(m.path)}});
    json devices = nullptr;
    if (r.runtime.ep_devices) {
        devices = json::array();
        for (const EpDeviceInfo& d : *r.runtime.ep_devices)
            devices.push_back(to_json(d));
    }
    return {
        {"onnxruntime",
         {{"version", ort.ort_version},
          {"library_path", path_to_utf8(ort.dll_path)},
          {"api_version", ort.api_version},
          {"load_check", ort.load_check()}}},
        {"execution_provider", execution_provider},
        {"qnn", nullptr},
        {"ep_device", ep_device},
        {"loaded_modules", modules},
        {"ep_devices", devices},
    };
}

json model_json(const BenchmarkResult& r) {
    if (!r.known.model)
        return nullptr;
    const ModelResult& m = r.model;
    const ModelPrecision& p = m.precision;
    return {
        {"name", m.name},
        {"file_name", m.file.file_name},
        {"sha256", m.file.sha256},
        {"size_bytes", m.file.size_bytes},
        {"opset", opt(m.opset)},
        {"ir_version", opt(m.ir_version)},
        {"stored_precision", p.stored_precision},
        {"inputs", tensors_json(m.inputs)},
        {"outputs", tensors_json(m.outputs)},
        {"source",
         {{"library", nullptr}, {"identifier", nullptr}, {"export_tool", opt(m.export_tool)}}},
        {"quantization",
         {{"method", p.quantization_method},
          {"format", opt(p.quantization_format)},
          {"activation_dtype", opt(p.activation_dtype)},
          {"weight_dtype", opt(p.weight_dtype)},
          {"calibration", nullptr},
          {"tool", nullptr}}},
        {"precision_notes", nullptr},
        {"normalization", nullptr},
    };
}

json config_json(const BenchmarkResult& r) {
    const RunConfig& c = r.config;
    const ProviderDescription& p = r.runtime.provider;
    return {
        {"requested",
         {{"execution_provider", execution_provider_name(c.provider)},
          {"device_type", device_type_name(c.provider)},
          {"compute_precision", nullptr}}},
        {"strict_mode", false},
        {"batch_size", r.batch_size > 0 ? json(r.batch_size) : json(nullptr)},
        {"input",
         {{"kind", schema_name(c.input)},
          {"constant_value",
           c.input == InputKind::constant ? json(c.effective_constant_value()) : json(nullptr)},
          {"seed", c.input == InputKind::random ? json(c.effective_seed()) : json(nullptr)},
          {"dataset", nullptr},
          {"preprocessing", r.input_preprocessing}}},
        {"warmup_iterations", c.warmup},
        {"measured_iterations", c.iterations},
        {"trials", c.trials},
        {"run_api", "sync"},
        {"ort_execution_mode", "sequential"},
        {"intra_op_threads", c.threads},
        {"inter_op_threads", nullptr},
        {"graph_optimization_level", schema_name(c.graph_opt)},
        {"ep_options", r.known.provider ? json(p.ep_options) : json::object()},
        {"session_config", r.known.provider ? json(p.session_config) : json::object()},
        {"ep_context_cache", nullptr},
        {"run_binding", schema_name(c.binding)},
        {"profiling", {{"ort", false}, {"qnn_level", "off"}}},
    };
}

json execution_json(const BenchmarkResult& r) {
    if (r.known.placement) {
        const NodeAssignment& na = r.execution.node_assignment;
        return {
            {"full_device_execution_verified", r.execution.full_device_execution_verified},
            {"verification_method", na.verification_method},
            {"verification_details", r.execution.verification_details},
            {"node_assignment", {{"total_nodes", na.total_nodes}, {"nodes_by_ep", na.nodes_by_ep}}},
            {"qnn_validation", nullptr},
        };
    }
    if (!r.known.session)
        return nullptr;
    return {
        {"full_device_execution_verified", nullptr},
        {"verification_method", nullptr},
        {"verification_details", "Node placement was not recorded because the run failed."},
        {"node_assignment", nullptr},
        {"qnn_validation", nullptr},
    };
}

json timing_json(const BenchmarkResult& r) {
    const TimingResult& t = r.timing;
    if (t.trials.empty())
        return nullptr;
    json trials = json::array();
    for (const TrialResult& trial : t.trials)
        trials.push_back({{"trial_index", trial.trial_index}, {"stats", stats_json(trial.stats)}});
    json aggregate = nullptr;
    if (t.aggregate) {
        const TrialVariation& v = t.aggregate->trial_variation;
        aggregate = {
            {"stats", stats_json(t.aggregate->stats)},
            {"trial_variation",
             {{"trial_mean_min_ms", v.trial_mean_min_ms},
              {"trial_mean_max_ms", v.trial_mean_max_ms},
              {"trial_mean_stddev_ms", opt(v.trial_mean_stddev_ms)},
              {"trial_mean_cv_ratio", opt(v.trial_mean_cv_ratio)}}},
        };
    }
    const SessionInitResult& si = t.session_init;
    return {
        {"timer", {{"source", t.timer.source}, {"resolution_ns", opt(t.timer.resolution_ns)}}},
        {"percentile_method", t.percentile_method},
        {"session_init",
         {{"session_create_ms", si.session_create_ms},
          {"ep_context_state", si.ep_context_state},
          {"cold_ms", si.cold_ms},
          {"context_build_ms", opt(si.context_build_ms)},
          {"cached_ms", si.cached_ms ? json(*si.cached_ms) : json(nullptr)}}},
        {"first_inference_ms", opt(t.first_inference_ms)},
        {"trials", trials},
        {"aggregate", aggregate},
        {"scope", t.scope},
        {"accelerator", nullptr},
    };
}

json memory_json(const BenchmarkResult& r) {
    const MemoryResult& m = r.memory;
    return {
        {"scope", m.scope},
        {"peak_working_set_bytes", m.peak_working_set_bytes},
        {"peak_private_bytes", m.peak_private_bytes},
        {"source", m.source},
        {"session_create_working_set_delta_bytes", opt(m.session_create_working_set_delta_bytes)},
    };
}

}  // namespace

json to_result_json(const BenchmarkResult& r, const std::optional<RawLatencyLink>& raw) {
    json warnings = json::array();
    for (const ResultWarning& w : r.warnings)
        warnings.push_back({{"code", w.code}, {"message", w.message}, {"path", opt(w.path)}});

    return {
        {"schema_version", r.schema_version},
        {"run_id", r.run_id},
        {"timestamp_utc", r.timestamp_utc},
        {"label", opt(r.label)},
        {"status", r.status()},
        {"error", error_json(r)},
        {"warnings", warnings},
        {"tool",
         {{"name", "edgebench"},
          {"version", r.tool_version},
          {"git_commit", opt(r.tool.git_commit)},
          {"git_dirty", opt(r.tool.git_dirty)},
          {"build_type", r.tool.build_type.empty() ? json(nullptr) : json(r.tool.build_type)},
          {"compiler", r.tool.compiler.empty() ? json(nullptr) : json(r.tool.compiler)},
          {"implementation", "cpp"}}},
        {"system", to_json(r.system)},
        {"runtime", runtime_json(r)},
        {"model", model_json(r)},
        {"config", config_json(r)},
        {"execution", execution_json(r)},
        {"timing", timing_json(r)},
        {"memory", memory_json(r)},
        {"accuracy", nullptr},
        {"raw", raw ? json{{"latencies_file", raw->latencies_file},
                           {"format", "csv"},
                           {"sha256", raw->sha256}}
                    : json(nullptr)},
    };
}

std::string format_number(double value) {
    return json(value).dump();
}

std::string csv_field(std::string_view text) {
    if (text.find_first_of(",\"\r\n") == std::string_view::npos)
        return std::string(text);
    std::string out = "\"";
    for (char c : text) {
        if (c == '"')
            out += '"';
        out += c;
    }
    return out + "\"";
}

std::string latencies_csv(const BenchmarkResult& r) {
    std::string out = "trial_index,iteration_index,latency_ms\r\n";
    for (const TrialResult& trial : r.timing.trials) {
        for (std::size_t i = 0; i < trial.latencies_ms.size(); ++i) {
            out += std::to_string(trial.trial_index);
            out += ',';
            out += std::to_string(i);
            out += ',';
            out += format_number(trial.latencies_ms[i]);
            out += "\r\n";
        }
    }
    return out;
}

std::string summary_csv(const BenchmarkResult& r) {
    auto num = [](const std::optional<double>& v) { return v ? format_number(*v) : ""; };
    auto row = [&](std::string_view kind, std::string trial_index, const LatencyStats& s,
                   std::optional<double> create_ms) {
        std::string line = csv_field(kind) + "," + trial_index + "," + std::to_string(s.count) +
                           "," + format_number(s.mean_ms) + "," + format_number(s.p50_ms) + "," +
                           num(s.p95_ms) + "," + num(s.p99_ms) + "," + num(s.stddev_ms) + "," +
                           format_number(s.min_ms) + "," + format_number(s.max_ms) + "," +
                           format_number(s.inferences_per_sec) + "," +
                           format_number(s.items_per_sec) + "," + num(create_ms) + "\r\n";
        return line;
    };
    std::string out =
        "row,trial_index,count,mean_ms,p50_ms,p95_ms,p99_ms,stddev_ms,min_ms,max_ms,"
        "inferences_per_sec,items_per_sec,session_create_ms\r\n";
    for (const TrialResult& trial : r.timing.trials)
        out +=
            row("trial", std::to_string(trial.trial_index), trial.stats, trial.session_create_ms);
    if (r.timing.aggregate) {
        out += row("aggregate", "", r.timing.aggregate->stats,
                   r.timing.trials.empty()
                       ? std::nullopt
                       : std::optional<double>(r.timing.session_init.session_create_ms));
    }
    return out;
}

}  // namespace edgebench
