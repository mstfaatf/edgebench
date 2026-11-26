#include "edgebench/qnn_provider.hpp"

#include <unordered_map>

#include "edgebench/bundle_modules.hpp"
#include "edgebench/npu_options.hpp"
#include "edgebench/ort_bench_session.hpp"
#include "edgebench/qnn_ep.hpp"
#include "edgebench/qnn_log.hpp"
#include "edgebench/runtime.hpp"

namespace fs = std::filesystem;

namespace edgebench {

namespace {

// A session that holds a lease on the EP library for as long as it exists. The
// lease is declared first, so it is released after the session.
class LeasedBenchSession final : public BenchSession {
public:
    LeasedBenchSession(EpLibrary::Lease lease, std::unique_ptr<BenchSession> session)
        : lease_(std::move(lease)), session_(std::move(session)) {}

    const std::vector<TensorInfo>& inputs() const override { return session_->inputs(); }
    const std::vector<TensorInfo>& outputs() const override { return session_->outputs(); }
    std::int64_t batch_size() const override { return session_->batch_size(); }
    RunBinding run_binding() const override { return session_->run_binding(); }
    std::span<float> input_data(std::size_t index) override { return session_->input_data(index); }
    Result<void> run_once() override { return session_->run_once(); }
    TensorView output(std::size_t index) const override { return session_->output(index); }
    double session_create_ms() const override { return session_->session_create_ms(); }
    std::optional<std::int64_t> session_create_working_set_delta_bytes() const override {
        return session_->session_create_working_set_delta_bytes();
    }

private:
    EpLibrary::Lease lease_;
    std::unique_ptr<BenchSession> session_;
};

// The QNN EP logged that its backend did not load, although the session was
// created (without strict mode every node then silently runs on the CPU).
Error backend_failure(const std::string& context, const std::vector<std::string>& logged) {
    Error e{ExitCode::ort_load_failure, "qnn_backend_load_failed",
            context + ": the QNN execution provider could not load its HTP backend"};
    for (const std::string& message : logged) {
        if (mentions_qnn_backend_failure({message})) {
            e.message += " (" + message + ")";
            break;
        }
    }
    return e;
}

Error classify(Error e, const RunConfig& config, const std::vector<std::string>& logged) {
    if (e.exit_code != ExitCode::ort_runtime_error || !e.ort_error_code)
        return e;  // not an ONNX Runtime error, e.g. a model with dynamic shapes
    return classify_qnn_session_failure(std::move(e), config.strict, logged);
}

}  // namespace

Result<std::unique_ptr<QnnProvider>> QnnProvider::create(Runtime& runtime,
                                                         const RunConfig& config) {
    Result<EpLibrary*> library = register_qnn_ep(runtime);
    if (!library)
        return library.error();

    std::vector<EpDeviceInfo> devices;
    try {
        devices = runtime.ep_devices();
    } catch (const Ort::Exception& e) {
        return ort_error("ep_enumeration_failed", "enumerating EP devices", e.GetOrtErrorCode(),
                         e.what());
    }
    const NpuSelection selection = select_qnn_npu(devices);
    if (!selection.ok())
        return selection.error();

    // The EP lists the NPU whether or not the backend exists.
    Result<fs::path> backend = check_qnn_htp_backend(runtime.bundle_dir());
    if (!backend)
        return backend.error();

    const EpDeviceInfo& device = devices[*selection.index];
    ProviderDescription d;
    d.execution_provider = std::string(kQnnEpName);
    if (const auto it = device.ep_metadata.find("version"); it != device.ep_metadata.end())
        d.ep_version = it->second;
    d.ep_library_path = path_to_utf8(library.value()->path());
    d.ep_device =
        SelectedEpDevice{device, std::nullopt, "by_device_type", selection.candidate_count};
    d.ep_options = qnn_ep_options(config, backend.value());
    d.session_config = qnn_session_config(config);
    d.qnn = QnnBackendInfo{"htp", path_to_utf8(backend.value()), std::nullopt, std::nullopt};
    return std::unique_ptr<QnnProvider>(
        new QnnProvider(runtime, *library.value(), *selection.index, std::move(d)));
}

QnnProvider::QnnProvider(Runtime& runtime, EpLibrary& library, std::size_t device_index,
                         ProviderDescription description)
    : runtime_(runtime),
      library_(library),
      device_index_(device_index),
      description_(std::move(description)) {}

Result<Ort::SessionOptions> QnnProvider::session_options(const RunConfig& config) {
    Result<Ort::ConstEpDevice> device =
        ep_device_handle(runtime_.env(), device_index_, description_.ep_device->device);
    if (!device)
        return device.error();
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(config.threads);
        options.SetExecutionMode(ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(ort_graph_optimization_level(config.graph_opt));
        for (const auto& [key, value] : description_.session_config)
            options.AddConfigEntry(key.c_str(), value.c_str());
        const std::unordered_map<std::string, std::string> ep_options(
            description_.ep_options.begin(), description_.ep_options.end());
        options.AppendExecutionProvider_V2(runtime_.env(), {device.value()}, ep_options);
        return options;
    } catch (const Ort::Exception& e) {
        return ort_error("session_options_failed", "setting the NPU session options",
                         e.GetOrtErrorCode(), e.what());
    }
}

Result<std::unique_ptr<BenchSession>> QnnProvider::create_session(const fs::path& model,
                                                                  const RunConfig& config) {
    EpLibrary::Lease lease = library_.lease();  // before the session exists
    Result<Ort::SessionOptions> options = session_options(config);
    if (!options)
        return options.error();

    const std::uint64_t log_start = runtime_.log_position();
    Result<std::unique_ptr<OrtBenchSession>> session =
        OrtBenchSession::create(runtime_.env(), model, options.value(), config);
    const std::vector<std::string> logged = runtime_.warnings_since(log_start);
    if (!session)
        return classify(session.error(), config, logged);
    if (mentions_qnn_backend_failure(logged))
        return backend_failure("creating a session for " + path_to_utf8(model), logged);
    return std::unique_ptr<BenchSession>(
        new LeasedBenchSession(std::move(lease), std::move(session).value()));
}

Result<NodeAssignment> QnnProvider::record_placement(const fs::path& model,
                                                     const RunConfig& config) {
    EpLibrary::Lease lease = library_.lease();
    Result<Ort::SessionOptions> options = session_options(config);
    if (!options)
        return options.error();

    const std::uint64_t log_start = runtime_.log_position();
    const fs::path bundle = runtime_.bundle_dir();
    Result<PlacementCapture> captured =
        capture_node_placement(runtime_, model, options.value(),
                               [&bundle]() { return check_bundle_modules(bundle).result(); });
    if (!captured)
        return classify(captured.error(), config, runtime_.warnings_since(log_start));
    PlacementCapture& c = captured.value();
    if (mentions_qnn_backend_failure(c.messages))
        return backend_failure("creating the node placement session for " + path_to_utf8(model),
                               c.messages);

    NodeAssignment assignment = std::move(c.assignment);
    assignment.modules_checked_while_alive = true;
    assignment.qnn_validation = parse_qnn_validation(c.messages);
    if (assignment.qnn_validation)
        assignment.verification_method = "combined";  // ORT placement + QNN validation log
    return assignment;
}

}  // namespace edgebench
