#include "edgebench/dry_run.hpp"

#include "edgebench/bundle_modules.hpp"
#include "edgebench/model_info.hpp"
#include "edgebench/npu_options.hpp"
#include "edgebench/onnx_inspect.hpp"
#include "edgebench/provider.hpp"
#include "edgebench/runtime.hpp"
#include "edgebench/version.hpp"

namespace edgebench {

namespace {

using json = nlohmann::ordered_json;

json tensors_json(const std::vector<TensorInfo>& tensors) {
    json out = json::array();
    for (const TensorInfo& t : tensors)
        out.push_back({{"name", t.name}, {"dtype", dtype_name(t.dtype)}, {"shape", t.shape}});
    return out;
}

json opt(const std::optional<std::string>& v) {
    return v ? json(*v) : json(nullptr);
}

json ep_device_json(const std::optional<SelectedEpDevice>& selected) {
    if (!selected)
        return nullptr;
    const EpDeviceInfo& d = selected->device;
    return {
        {"ep_name", d.ep_name},
        {"ep_vendor", d.ep_vendor},
        {"device_type", d.device_type},
        {"vendor", d.vendor},
        {"vendor_id", d.vendor_id},
        {"device_id", d.device_id},
        {"driver_version", opt(selected->driver_version)},
        {"metadata", d.metadata},
        {"selection",
         {{"method", selected->selection_method}, {"candidate_count", selected->candidate_count}}},
    };
}

}  // namespace

Result<json> dry_run(Runtime& runtime, const RunConfig& config) {
    Result<ModelFileInfo> model = read_model_file_info(config.model);
    if (!model)
        return model.error();

    // The NPU compute precision follows from the model's stored precision.
    std::optional<std::string> compute_precision;
    if (config.provider == ProviderKind::npu) {
        Result<OnnxGraphSummary> graph = inspect_onnx_file(model.value().path);
        if (!graph)
            return graph.error();
        Result<std::optional<std::string>> precision =
            requested_compute_precision(config, infer_precision(graph.value()));
        if (!precision)
            return precision.error();
        compute_precision = precision.value();
    }

    Result<std::unique_ptr<Provider>> provider = make_provider(runtime, config);
    if (!provider)
        return provider.error();
    const ProviderDescription& desc = provider.value()->describe();

    Result<std::unique_ptr<BenchSession>> created =
        provider.value()->create_session(model.value().path, config);
    if (!created)
        return created.error();
    BenchSession& session = *created.value();
    // Execution providers load their libraries with the session.
    if (Result<void> modules = check_bundle_modules(runtime.bundle_dir()).result(); !modules)
        return modules.error();

    if (Result<void> r = session.run_once(); !r)
        return r.error();
    for (std::size_t i = 0; i < session.outputs().size(); ++i) {
        if (const auto bad = first_non_finite(session.output(i))) {
            return Error{ExitCode::internal_error, "non_finite_output",
                         "output '" + session.outputs()[i].name +
                             "' has a non-finite value (NaN or infinity) at element " +
                             std::to_string(*bad)};
        }
    }

    // Placement after the session check: nothing is timed in a dry run.
    Result<NodeAssignment> placed = provider.value()->record_placement(model.value().path, config);
    if (!placed)
        return placed.error();
    const NodeAssignment& placement = placed.value();
    const std::string requested_ep(execution_provider_name(config.provider));
    const bool all_on_requested = placement.all_on(requested_ep);
    if (!all_on_requested && strict_mode(config)) {
        return Error{ExitCode::internal_error, "strict_placement_mismatch",
                     "strict mode is on, but " +
                         std::to_string(placement.nodes_not_on(requested_ep)) + " of " +
                         std::to_string(placement.total_nodes) + " nodes are not on " +
                         requested_ep + "; ONNX Runtime should have refused the session"};
    }

    json out = {{"command", "run"},
                {"dry_run", true},
                {"edgebench_version", version()},
                {"results_schema_version", results_schema_version()}};
    out.update(resolved_config_json(config));
    out["model"] = {
        {"path", path_to_utf8(model.value().path)}, {"file_name", model.value().file_name},
        {"size_bytes", model.value().size_bytes},   {"sha256", model.value().sha256},
        {"inputs", tensors_json(session.inputs())}, {"outputs", tensors_json(session.outputs())},
    };
    json& cfg = out["config"];
    cfg["batch_size"] = session.batch_size();
    cfg["ep_options"] = desc.ep_options;
    cfg["session_config"] = desc.session_config;
    cfg["requested"]["compute_precision"] = opt(compute_precision);

    const OrtLoadResult& ort = runtime.load_result();
    out["runtime"] = {
        {"onnxruntime",
         {{"version", ort.ort_version},
          {"library_path", path_to_utf8(ort.dll_path)},
          {"api_version", ort.api_version},
          {"load_check", ort.load_check()}}},
        {"execution_provider",
         {{"name", desc.execution_provider},
          {"version", opt(desc.ep_version)},
          {"library_path", opt(desc.ep_library_path)}}},
        {"qnn", desc.qnn
                    ? json{{"backend", desc.qnn->backend}, {"backend_path", desc.qnn->backend_path}}
                    : json(nullptr)},
        {"ep_device", ep_device_json(desc.ep_device)},
    };
    out["execution"] = {
        {"full_device_execution_verified", all_on_requested},
        {"verification_method", placement.verification_method},
        {"verification_details",
         "Node placement from ONNX Runtime's verbose session log, recorded in a separate session "
         "with the same options as the measured one."},
        {"node_assignment",
         {{"total_nodes", placement.total_nodes}, {"nodes_by_ep", placement.nodes_by_ep}}},
    };
    json warnings = json::array();
    if (!all_on_requested) {
        warnings.push_back(
            {{"code", config.provider == ProviderKind::cpu ? "nodes_not_on_requested_ep"
                                                           : "cpu_fallback_nodes"},
             {"message", std::to_string(placement.nodes_not_on(requested_ep)) + " of " +
                             std::to_string(placement.total_nodes) + " nodes are not on " +
                             requested_ep},
             {"path", "/execution/node_assignment"}});
    }
    out["warnings"] = warnings;
    out["dry_run_check"] = {
        {"session_create_ms", session.session_create_ms()},
        {"inferences", 1},
        {"outputs_finite", true},
    };
    return out;
}

}  // namespace edgebench
