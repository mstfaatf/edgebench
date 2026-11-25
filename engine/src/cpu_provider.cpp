#include "edgebench/cpu_provider.hpp"

#include <onnxruntime_cxx_api.h>

#include "edgebench/ort_bench_session.hpp"
#include "edgebench/runtime.hpp"

namespace edgebench {

namespace {

GraphOptimizationLevel ort_level(GraphOptLevel level) {
    switch (level) {
        case GraphOptLevel::disable_all:
            return ORT_DISABLE_ALL;
        case GraphOptLevel::enable_basic:
            return ORT_ENABLE_BASIC;
        case GraphOptLevel::enable_extended:
            return ORT_ENABLE_EXTENDED;
        case GraphOptLevel::enable_all:
            return ORT_ENABLE_ALL;
    }
    return ORT_ENABLE_ALL;
}

}  // namespace

Result<std::unique_ptr<CpuProvider>> CpuProvider::create(Runtime& runtime) {
    std::vector<EpDeviceInfo> devices;
    try {
        devices = runtime.ep_devices();
    } catch (const Ort::Exception& e) {
        return ort_error("ep_enumeration_failed", "enumerating EP devices", e.GetOrtErrorCode(),
                         e.what());
    }
    std::vector<EpDeviceInfo> matches;
    for (const EpDeviceInfo& d : devices) {
        if (d.ep_name == kExecutionProvider && d.device_type == "CPU")
            matches.push_back(d);
    }
    if (matches.size() != 1) {
        return Error{ExitCode::ort_runtime_error, "device_selection_failed",
                     "expected exactly one " + std::string(kExecutionProvider) +
                         " device of type CPU, found " + std::to_string(matches.size())};
    }

    ProviderDescription d;
    d.execution_provider = kExecutionProvider;
    d.ep_device = SelectedEpDevice{matches.front(), std::nullopt, "by_device_type",
                                   static_cast<int>(matches.size())};
    // Set explicitly (to ONNX Runtime's default) so the spin setting is recorded.
    d.session_config["session.intra_op.allow_spinning"] = "1";
    return std::unique_ptr<CpuProvider>(new CpuProvider(runtime, std::move(d)));
}

CpuProvider::CpuProvider(Runtime& runtime, ProviderDescription description)
    : runtime_(runtime), description_(std::move(description)) {}

Ort::SessionOptions CpuProvider::session_options(const RunConfig& config) const {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(config.threads);
    options.SetExecutionMode(ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(ort_level(config.graph_opt));
    for (const auto& [key, value] : description_.session_config)
        options.AddConfigEntry(key.c_str(), value.c_str());
    return options;
}

Result<std::unique_ptr<BenchSession>> CpuProvider::create_session(
    const std::filesystem::path& model, const RunConfig& config) {
    Result<std::unique_ptr<OrtBenchSession>> session =
        [&]() -> Result<std::unique_ptr<OrtBenchSession>> {
        try {
            return OrtBenchSession::create(runtime_.env(), model, session_options(config), config);
        } catch (const Ort::Exception& e) {
            return ort_error("session_options_failed", "setting session options",
                             e.GetOrtErrorCode(), e.what());
        }
    }();
    if (!session)
        return session.error();
    return std::unique_ptr<BenchSession>(std::move(session).value());
}

Result<NodeAssignment> CpuProvider::record_placement(const std::filesystem::path& model,
                                                     const RunConfig& config) {
    try {
        Ort::SessionOptions options = session_options(config);
        return record_node_placement(runtime_, model, options);
    } catch (const Ort::Exception& e) {
        return ort_error("session_options_failed", "setting session options", e.GetOrtErrorCode(),
                         e.what());
    }
}

Result<std::unique_ptr<Provider>> make_provider(Runtime& runtime, const RunConfig& config) {
    switch (config.provider) {
        case ProviderKind::cpu:
            break;
    }
    Result<std::unique_ptr<CpuProvider>> cpu = CpuProvider::create(runtime);
    if (!cpu)
        return cpu.error();
    return std::unique_ptr<Provider>(std::move(cpu).value());
}

}  // namespace edgebench
