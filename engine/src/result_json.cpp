#include "edgebench/result_json.hpp"

#include <algorithm>
#include <sstream>

#include "edgebench/sha256.hpp"

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
    json unsupported = nullptr;
    if (e.unsupported_nodes) {
        unsupported = json::array();
        for (const NamedNode& n : *e.unsupported_nodes)
            unsupported.push_back({{"name", n.name}, {"op_type", n.op_type}});
    }
    return {
        {"stage", r.failure->stage},
        {"message", e.message},
        {"code", e.ort_error_name ? json(*e.ort_error_name)
                                  : (e.kind.empty() ? json(nullptr) : json(e.kind))},
        {"kind", e.kind.empty() ? json(nullptr) : json(e.kind)},
        {"unsupported_nodes", unsupported},
    };
}

// The trials' values of one kind of initialization (timing.trials[].session_init).
std::vector<double> per_trial(const TimingResult& t,
                              std::optional<double> TrialSessionInit::*kind) {
    std::vector<double> values;
    for (const TrialResult& trial : t.trials)
        if (const std::optional<double>& v = trial.session_init.*kind)
            values.push_back(*v);
    return values;
}

json init_summary_json(std::vector<double> values) {
    if (values.empty())
        return nullptr;
    std::sort(values.begin(), values.end());
    return {{"count", values.size()},
            {"median_ms", nearest_rank(values, 50)},
            {"min_ms", values.front()},
            {"max_ms", values.back()}};
}

// Nodes ONNX Runtime places on the QNN EP: each one is a fused partition.
std::optional<int> qnn_partition_count(const BenchmarkResult& r) {
    if (!r.known.placement)
        return std::nullopt;
    const auto& by_ep = r.execution.node_assignment.nodes_by_ep;
    const auto it = by_ep.find("QNNExecutionProvider");
    return it == by_ep.end() ? 0 : it->second;
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
    json qnn = nullptr;
    if (r.known.provider && p.qnn) {
        json file_version = nullptr;
        if (const std::optional<ModuleFileVersion>& v = r.runtime.qnn_backend_file_version) {
            file_version = {{"module_path", path_to_utf8(v->module_path)},
                            {"product_version", opt(v->product_version)},
                            {"file_version", opt(v->file_version)},
                            {"source", "loaded_module_version_resource"}};
        }
        qnn = {{"sdk_version", opt(p.qnn->sdk_version)},
               {"backend", p.qnn->backend},
               {"backend_path", p.qnn->backend_path},
               {"backend_build", opt(p.qnn->backend_build)},
               {"backend_file_version", file_version},
               {"htp_arch", opt(p.qnn->htp_arch)},
               {"soc_model", opt(p.qnn->soc_model)},
               {"partition_count", opt(qnn_partition_count(r))},
               {"unlisted_ep_options",
                p.qnn->unlisted_options_default ? json("ep_default") : json(nullptr)},
               {"value_sources",
                {{"sdk_version", opt(p.qnn->sdk_version_source)},
                 {"backend_build", opt(p.qnn->backend_build_source)},
                 {"htp_arch", opt(p.qnn->htp_arch_source)},
                 {"soc_model", opt(p.qnn->soc_model_source)}}}};
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
        {"qnn", qnn},
        {"ep_device", ep_device},
        {"loaded_modules", modules},
        {"ep_devices", devices},
    };
}

// Calibration and tool only describe a quantized model (the schema requires
// null for method "none"); both come from the file's edgebench.quant.* metadata.
bool quantized(const ModelResult& m) {
    return m.precision.quantization_method != "none";
}

json calibration_json(const ModelResult& m) {
    const QuantizationRecord& q = m.quantization_record;
    if (!quantized(m) || (!q.calibration_dataset && !q.calibration_count))
        return nullptr;
    return {{"dataset", opt(q.calibration_dataset)}, {"sample_count", opt(q.calibration_count)}};
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
          {"calibration", calibration_json(m)},
          {"tool", quantized(m) ? opt(m.quantization_record.tool) : json(nullptr)}}},
        {"precision_notes", opt(m.precision_notes)},
        {"normalization", opt(m.normalization)},
    };
}

// config.ep_context_cache: null for the CPU EP (it has none).
json ep_context_cache_json(const BenchmarkResult& r) {
    if (r.config.provider != ProviderKind::npu)
        return nullptr;
    const EpContextCacheResult& c = r.ep_cache;
    json files = json::array();
    for (const CacheFile& f : c.files)
        files.push_back({{"name", f.name}, {"size_bytes", f.size_bytes}, {"sha256", f.sha256}});
    json key = nullptr;
    if (c.key) {
        // The key fields in the schema's key order (the key id hashes them sorted).
        const json& k = *c.key;
        key = {{"model_sha256", k.at("model_sha256")},
               {"ort_version", k.at("ort_version")},
               {"ep_version", k.at("ep_version")},
               {"qnn_sdk_version", k.at("qnn_sdk_version")},
               {"ep_options_sha256", k.at("ep_options_sha256")},
               {"device", k.at("device")},
               {"qnn_backend_file_version", k.at("qnn_backend_file_version")},
               {"ep_options", k.at("ep_options")},
               {"session_config", k.at("session_config")},
               {"graph_optimization_level", k.at("graph_optimization_level")}};
    }
    // Before the cache was examined (a run that failed early) the state is unknown;
    // with the cache off it is "disabled" from the start.
    const std::optional<std::string> state =
        c.state ? c.state : (c.enabled() ? std::nullopt : std::optional<std::string>("disabled"));
    return {{"enabled", c.enabled()},
            {"cache_file",
             c.context_path ? json(path_to_utf8(c.context_path->filename())) : json(nullptr)},
            {"state", opt(state)},
            {"files", files},
            {"mode", schema_name(c.mode)},
            {"key_id", opt(c.key_id)},
            {"context_path", c.context_path ? json(path_to_utf8(*c.context_path)) : json(nullptr)},
            {"hit", opt(c.hit)},
            {"verified_equal_outputs", opt(c.verified_equal_outputs)},
            {"key", key},
            {"cache_dir", c.cache_dir ? json(path_to_utf8(*c.cache_dir)) : json(nullptr)},
            {"embed_mode", c.cache_dir ? json(std::string(kEpContextEmbedMode)) : json(nullptr)},
            {"compatibility", c.compatibility
                                  ? json{{"info", opt(c.compatibility->info)},
                                         {"verdict", opt(c.compatibility->verdict)},
                                         {"source", "ort_model_compatibility_for_ep_devices"}}
                                  : json(nullptr)}};
}

json qnn_validation_json(const std::optional<QnnValidation>& v) {
    if (!v)
        return nullptr;
    json passes = json::array();
    for (const QnnValidationPass& p : v->passes)
        passes.push_back({{"passed_nodes", p.passed_nodes}, {"failed_nodes", p.failed_nodes}});
    json rejected = json::array();
    for (const QnnRejectedNode& n : v->rejected_nodes)
        rejected.push_back({{"op_type", n.op_type},
                            {"node_name", n.node_name},
                            {"reason", opt(n.reason)},
                            {"node_index", opt(n.node_index)}});
    return {{"passes", passes}, {"rejected_nodes", rejected}};
}

json verification_details_json(const BenchmarkResult& r, const std::string& summary) {
    json passes = nullptr;
    if (r.known.placement && r.execution.node_assignment.qnn_validation) {
        passes = json::array();
        for (const QnnValidationPass& p : r.execution.node_assignment.qnn_validation->passes)
            passes.push_back({{"passed_units_by_op", p.passed_units_by_op},
                              {"failed_units_by_op", p.failed_units_by_op}});
    }
    json checks = json::array();
    for (const ModuleCheckpoint& c : r.execution.module_checks) {
        json outside = json::array();
        for (const LoadedModule& m : c.outside)
            outside.push_back({{"name", m.name}, {"path", path_to_utf8(m.path)}});
        checks.push_back({{"stage", c.stage},
                          {"trial_index", opt(c.trial_index)},
                          {"modules_outside_bundle", outside}});
    }
    json deleted = json::array();
    for (const std::filesystem::path& p : r.ep_cache.deleted_stale)
        deleted.push_back(path_to_utf8(p));
    json load = json::array();
    for (const BenchmarkResult::BackgroundLoadRecord& b : r.background_load_samples) {
        json attribution = nullptr;
        if (b.sample.attribution) {
            json top = json::array();
            for (const ProcessShare& p : b.sample.attribution->top_processes)
                top.push_back({{"name", p.name}, {"cpu_share", p.cpu_share}});
            attribution = {{"top_processes", top},
                           {"own_process_share", b.sample.attribution->own_share},
                           {"unattributed_share", b.sample.attribution->unattributed_share}};
        }
        load.push_back({{"phase", b.phase},
                        {"trial_index", b.trial_index},
                        {"cpu_utilization_ratio", b.sample.cpu_utilization_ratio},
                        {"sample_window_s", b.sample.sample_window_s},
                        {"attribution", attribution}});
    }
    return {{"summary", summary},
            {"qnn_validation_passes", passes},
            {"module_checks", checks},
            {"deleted_stale_cache_files", deleted},
            {"background_load_samples", load}};
}

json effective_precision_json(const BenchmarkResult& r) {
    if (!r.effective_precision)
        return nullptr;
    const EffectivePrecision& p = *r.effective_precision;
    return {{"stored", p.stored}, {"compute", p.compute}, {"source", p.source}};
}

json batch_equivalence_json(const BenchmarkResult& r) {
    if (!r.batch_equivalence)
        return nullptr;
    const BatchEquivalence& b = *r.batch_equivalence;
    return {{"bitwise_equal_to_b1", opt(b.bitwise_equal_to_b1)},
            {"bitwise_basis", b.bitwise_basis},
            {"predictions_equal_to_b1", opt(b.predictions_equal_to_b1)},
            {"predictions_basis", b.predictions_basis},
            {"predictions_compared", opt(b.predictions_compared)},
            {"predictions_differing", opt(b.predictions_differing)},
            {"reference_batch", 1}};
}

json config_json(const BenchmarkResult& r) {
    const RunConfig& c = r.config;
    const ProviderDescription& p = r.runtime.provider;
    return {
        {"requested",
         {{"execution_provider", execution_provider_name(c.provider)},
          {"device_type", device_type_name(c.provider)},
          {"compute_precision", opt(r.compute_precision)}}},
        {"strict_mode", strict_mode(c)},
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
        {"ep_context_cache", ep_context_cache_json(r)},
        {"run_binding", schema_name(c.binding)},
        {"profiling", {{"ort", false}, {"qnn_level", "off"}}},
        {"init_breakdown", c.init_breakdown},
    };
}

json execution_json(const BenchmarkResult& r) {
    if (r.known.placement) {
        const NodeAssignment& na = r.execution.node_assignment;
        return {
            {"full_device_execution_verified", r.execution.full_device_execution_verified},
            {"verification_method", na.verification_method},
            {"verification_details",
             verification_details_json(r, r.execution.verification_details)},
            {"node_assignment", {{"total_nodes", na.total_nodes}, {"nodes_by_ep", na.nodes_by_ep}}},
            {"qnn_validation", qnn_validation_json(na.qnn_validation)},
            {"effective_precision", effective_precision_json(r)},
            {"batch_equivalence", batch_equivalence_json(r)},
        };
    }
    if (!r.known.session)
        return nullptr;
    return {
        {"full_device_execution_verified", nullptr},
        {"verification_method", nullptr},
        {"verification_details",
         verification_details_json(r, "Node placement was not recorded because the run failed.")},
        {"node_assignment", nullptr},
        {"qnn_validation", nullptr},
        {"effective_precision", effective_precision_json(r)},
        {"batch_equivalence", batch_equivalence_json(r)},
    };
}

json profiled_pass_json(const BenchmarkResult& r,
                        const std::optional<std::vector<LinkedFile>>& linked) {
    if (!r.timing.profiled_pass)
        return nullptr;
    const ProfiledPassResult& p = *r.timing.profiled_pass;
    json events = json::object();
    for (const auto& [id, values] : p.profile.execute_ms)
        events[id] = init_summary_json(values);
    json init = json::array();
    for (const auto& [id, ms] : p.profile.init_ms)
        init.push_back({{"event", id}, {"time_ms", ms}});
    json files = json::array();
    if (linked) {
        for (const LinkedFile& f : *linked)
            files.push_back({{"name", f.name},
                             {"path", f.path},
                             {"size_bytes", f.size_bytes},
                             {"sha256", f.sha256}});
    } else {
        for (const std::filesystem::path& f : p.files) {
            std::error_code ec;
            const std::uintmax_t size = std::filesystem::file_size(f, ec);
            Result<std::string> sha = sha256_file(f);
            files.push_back({{"name", path_to_utf8(f.filename())},
                             {"path", path_to_utf8(f)},
                             {"size_bytes", ec ? json(nullptr) : json(size)},
                             {"sha256", sha ? json(sha.value()) : json(nullptr)}});
        }
    }
    return {{"enabled", true},
            {"iterations", p.iterations},
            {"accelerator_ms", init_summary_json(p.accelerator_ms())},
            {"source", p.source},
            {"note", std::string(kProfiledPassNote)},
            {"level", p.level},
            {"warmup_iterations", p.warmup_iterations},
            {"inferences_in_profile", p.profile.inferences},
            {"reported_unit", "us"},
            {"execute_events_ms", events},
            {"init_events", init},
            {"files", files}};
}

json timing_json(const BenchmarkResult& r, const std::optional<std::vector<LinkedFile>>& linked) {
    const TimingResult& t = r.timing;
    if (t.trials.empty())
        return nullptr;
    json trials = json::array();
    for (const TrialResult& trial : t.trials) {
        trials.push_back({{"trial_index", trial.trial_index},
                          {"stats", stats_json(trial.stats)},
                          {"session_init",
                           {{"cold_ms", opt(trial.session_init.cold_ms)},
                            {"context_build_ms", opt(trial.session_init.context_build_ms)},
                            {"cached_init_ms", opt(trial.session_init.cached_init_ms)}}}});
    }
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
        const TrialVariation& m = t.aggregate->trial_spread;
        aggregate["trial_spread"] = {{"judged_on", std::string(kTrialSpreadStatistic)},
                                     {"trial_min_ms", m.trial_mean_min_ms},
                                     {"trial_max_ms", m.trial_mean_max_ms},
                                     {"trial_stddev_ms", opt(m.trial_mean_stddev_ms)},
                                     {"trial_cv_ratio", opt(m.trial_mean_cv_ratio)},
                                     {"cv_threshold_ratio", kHighTrialMedianCv},
                                     {"max_to_min_threshold", kHighTrialMedianRatio},
                                     {"high", trial_spread_high(m)}};
    }
    const SessionInitResult& si = t.session_init;
    return {
        {"timer", {{"source", t.timer.source}, {"resolution_ns", opt(t.timer.resolution_ns)}}},
        {"percentile_method", t.percentile_method},
        {"session_init",
         {{"session_create_ms", si.session_create_ms},
          {"ep_context_state", si.ep_context_state},
          {"cold_ms", si.cold_ms.empty() ? json(nullptr) : json(si.cold_ms)},
          {"context_build_ms", opt(si.context_build_ms)},
          {"cached_ms", si.cached_ms.empty() ? json(nullptr) : json(si.cached_ms)},
          {"summary",
           {{"cold_ms", init_summary_json(per_trial(t, &TrialSessionInit::cold_ms))},
            {"context_build_ms",
             init_summary_json(per_trial(t, &TrialSessionInit::context_build_ms))},
            {"cached_init_ms",
             init_summary_json(per_trial(t, &TrialSessionInit::cached_init_ms))}}}}},
        {"first_inference_ms", opt(t.first_inference_ms)},
        {"trials", trials},
        {"aggregate", aggregate},
        {"scope", t.scope},
        {"accelerator", nullptr},
        {"profiled_pass", profiled_pass_json(r, linked)},
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

json to_result_json(const BenchmarkResult& r, const std::optional<RawLatencyLink>& raw,
                    const std::optional<std::vector<LinkedFile>>& profile_files) {
    auto entries = [](const std::vector<ResultWarning>& list) {
        json out = json::array();
        for (const ResultWarning& w : list)
            out.push_back({{"code", w.code}, {"message", w.message}, {"path", opt(w.path)}});
        return out;
    };

    json doc = {
        {"schema_version", r.schema_version},
        {"run_id", r.run_id},
        {"timestamp_utc", r.timestamp_utc},
        {"label", opt(r.label)},
        {"status", r.status()},
        {"error", error_json(r)},
        {"warnings", entries(r.warnings)},
        {"notes", entries(r.notes)},
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
        {"timing", timing_json(r, profile_files)},
        {"memory", memory_json(r)},
        {"accuracy", nullptr},
        {"raw", raw ? json{{"latencies_file", raw->latencies_file},
                           {"format", "csv"},
                           {"sha256", raw->sha256}}
                    : json(nullptr)},
    };
    // Only runs launched by `edgebench suite` carry the suite block.
    if (const std::optional<SuiteLink>& s = r.config.suite) {
        doc["suite"] = {{"suite_name", s->suite_name},
                        {"suite_config_hash", s->config_hash},
                        {"run_id", s->run_id},
                        {"repetition", s->repetition},
                        {"order_index", s->order_index},
                        {"cooldown_s_before", s->cooldown_s_before},
                        {"preflight_skipped", s->preflight_skipped}};
    }
    return doc;
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
