#include "edgebench/qnn_provider.hpp"

#include <atomic>
#include <map>
#include <mutex>
#include <unordered_map>

#include "edgebench/bundle_modules.hpp"
#include "edgebench/npu_options.hpp"
#include "edgebench/ort_bench_session.hpp"
#include "edgebench/qnn_backend_info.hpp"
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

// Raises the environment's log level to verbose; restores WARNING when it goes
// out of scope.
class VerboseEnvLog {
public:
    explicit VerboseEnvLog(Ort::Env& env) : env_(env) {
        env_.UpdateEnvWithCustomLogLevel(ORT_LOGGING_LEVEL_VERBOSE);
    }
    ~VerboseEnvLog() {
        try {
            env_.UpdateEnvWithCustomLogLevel(ORT_LOGGING_LEVEL_WARNING);
        } catch (const Ort::Exception&) {
            // keeps the verbose level; nothing else can be done here
        }
    }
    VerboseEnvLog(const VerboseEnvLog&) = delete;
    VerboseEnvLog& operator=(const VerboseEnvLog&) = delete;

private:
    Ort::Env& env_;
};

// The QNN EP (2.8.0) crashes with an access violation when ONNX Runtime asks it
// to rate compatibility info (GetModelCompatibilityForEpDevices) after any QNN
// session has existed in the process, compiled or loaded from a context, even
// once it was released. So that call is only made before the first QNN session,
// and the backend build id it yields is kept for the process.
std::atomic<bool> g_qnn_session_created{false};
std::mutex g_build_id_mutex;
std::map<std::string, std::optional<std::string>> g_build_ids;  // by backend path

void note_qnn_session() {
    g_qnn_session_created = true;
}

// What the QNN EP logs while it sets up its backend to rate kQnnBackendProbeInfo.
std::vector<std::string> backend_probe_messages(Runtime& runtime, Ort::ConstEpDevice device) {
    Runtime::LogCapture capture(runtime);
    try {
        const VerboseEnvLog verbose(runtime.env());
        (void)Ort::GetModelCompatibilityForEpDevices({device},
                                                     std::string(kQnnBackendProbeInfo).c_str());
    } catch (const Ort::Exception&) {
        // no rating: whatever was logged before the failure is still used
    }
    return capture.messages();
}

// The backend build id for `backend_path`: probed once per process, before any
// QNN session; null if it was not logged or can no longer be probed.
std::optional<std::string> backend_build_id(Runtime& runtime, Ort::ConstEpDevice device,
                                            const std::string& backend_path) {
    const std::lock_guard<std::mutex> lock(g_build_id_mutex);
    if (const auto it = g_build_ids.find(backend_path); it != g_build_ids.end())
        return it->second;
    if (g_qnn_session_created)
        return std::nullopt;  // too late to ask (see above); not remembered
    std::optional<std::string> id = find_backend_build_id(backend_probe_messages(runtime, device));
    g_build_ids[backend_path] = id;
    return id;
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
    Result<fs::path> backend =
        check_qnn_htp_backend(runtime.bundle_dir(), config.uses_ep_context());
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
    QnnBackendInfo qnn;
    qnn.backend = "htp";
    qnn.backend_path = path_to_utf8(backend.value());
    if (Result<Ort::ConstEpDevice> handle =
            ep_device_handle(runtime.env(), *selection.index, device);
        handle) {
        if (std::optional<std::string> build =
                backend_build_id(runtime, handle.value(), qnn.backend_path)) {
            qnn.backend_build = *build;
            qnn.backend_build_source = std::string(kSourceBackendBuildIdLog);
            qnn.sdk_version = sdk_version_from_build_id(*build);
            if (qnn.sdk_version)
                qnn.sdk_version_source = std::string(kSourceBackendBuildIdLog);
        }
    }
    d.qnn = std::move(qnn);
    return std::unique_ptr<QnnProvider>(
        new QnnProvider(runtime, *library.value(), *selection.index, std::move(d)));
}

QnnProvider::QnnProvider(Runtime& runtime, EpLibrary& library, std::size_t device_index,
                         ProviderDescription description)
    : runtime_(runtime),
      library_(library),
      device_index_(device_index),
      description_(std::move(description)) {}

Result<Ort::SessionOptions> QnnProvider::session_options(
    const RunConfig& config, const std::map<std::string, std::string>& session_config,
    const std::map<std::string, std::string>& ep_options_map) {
    Result<Ort::ConstEpDevice> device =
        ep_device_handle(runtime_.env(), device_index_, description_.ep_device->device);
    if (!device)
        return device.error();
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(config.threads);
        options.SetExecutionMode(ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(ort_graph_optimization_level(config.graph_opt));
        for (const auto& [key, value] : session_config)
            options.AddConfigEntry(key.c_str(), value.c_str());
        const std::unordered_map<std::string, std::string> ep_options(ep_options_map.begin(),
                                                                      ep_options_map.end());
        options.AppendExecutionProvider_V2(runtime_.env(), {device.value()}, ep_options);
        return options;
    } catch (const Ort::Exception& e) {
        return ort_error("session_options_failed", "setting the NPU session options",
                         e.GetOrtErrorCode(), e.what());
    }
}

Result<std::unique_ptr<BenchSession>> QnnProvider::create_session(const fs::path& model,
                                                                  const RunConfig& config) {
    return create(model, config, description_.session_config, description_.ep_options);
}

Result<std::unique_ptr<BenchSession>> QnnProvider::create_context_building_session(
    const fs::path& model, const RunConfig& config, const fs::path& context_model) {
    return create(model, config, qnn_context_build_config(config, context_model),
                  description_.ep_options);
}

Result<std::unique_ptr<BenchSession>> QnnProvider::create_profiling_session(
    const fs::path& model, const RunConfig& config, const fs::path& profile_file) {
    std::map<std::string, std::string> ep_options = description_.ep_options;
    ep_options["profiling_level"] = "basic";
    ep_options["profiling_file_path"] = path_to_utf8(profile_file);
    return create(model, config, description_.session_config, ep_options);
}

ContextCompatibility QnnProvider::context_compatibility(const fs::path& context_model) {
    ContextCompatibility c;
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr info = Ort::GetCompatibilityInfoFromModelAllocated(
            context_model.c_str(), std::string(kQnnEpName).c_str(), allocator);
        if (!info)
            return c;
        c.info = std::string(info.get());
    } catch (const Ort::Exception&) {
        return c;
    }
    if (g_qnn_session_created)
        return c;  // the EP cannot rate it any more in this process (see above)
    Result<Ort::ConstEpDevice> device =
        ep_device_handle(runtime_.env(), device_index_, description_.ep_device->device);
    if (!device)
        return c;
    try {
        c.verdict = compatibility_verdict_name(static_cast<int>(
            Ort::GetModelCompatibilityForEpDevices({device.value()}, c.info->c_str())));
    } catch (const Ort::Exception&) {
        // not rated
    }
    return c;
}

Result<std::unique_ptr<BenchSession>> QnnProvider::create(
    const fs::path& model, const RunConfig& config,
    const std::map<std::string, std::string>& session_config,
    const std::map<std::string, std::string>& ep_options) {
    EpLibrary::Lease lease = library_.lease();  // before the session exists
    Result<Ort::SessionOptions> options = session_options(config, session_config, ep_options);
    note_qnn_session();
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
    Result<Ort::SessionOptions> options =
        session_options(config, description_.session_config, description_.ep_options);
    if (!options)
        return options.error();
    note_qnn_session();

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
