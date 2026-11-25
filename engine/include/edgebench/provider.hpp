// Provider: an execution provider configuration that creates BenchSessions.
//
// A provider knows how to configure ONNX Runtime for one device (session
// options, EP options, device selection) and describes that configuration for
// the result file. Everything after session creation (inputs, Run or IoBinding,
// timing, export) is shared and provider-independent. See docs/architecture.md.
#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "edgebench/bench_session.hpp"
#include "edgebench/ep_devices.hpp"
#include "edgebench/placement.hpp"
#include "edgebench/result.hpp"
#include "edgebench/run_config.hpp"

namespace edgebench {

class Runtime;

/// The EP device a provider runs on, and how it was chosen
/// (results schema: runtime.ep_device).
struct SelectedEpDevice {
    EpDeviceInfo device;
    std::optional<std::string> driver_version;  // null if it cannot be determined
    std::string selection_method;               // e.g. "by_device_type"
    int candidate_count = 0;                    // devices that matched; exactly 1 is valid
};

/// The QNN backend a provider uses (results schema: runtime.qnn).
struct QnnBackendInfo {
    std::string backend;                     // "htp"
    std::string backend_path;                // absolute, as passed in backend_path
    std::optional<std::string> sdk_version;  // null: not read from the backend
    std::optional<std::string> backend_build;
};

/// Everything the results schema records about the provider configuration.
struct ProviderDescription {
    std::string execution_provider;              // runtime.execution_provider.name
    std::optional<std::string> ep_version;       // null for built-in EPs
    std::optional<std::string> ep_library_path;  // null unless registered from a library
    std::optional<SelectedEpDevice> ep_device;
    std::map<std::string, std::string> ep_options;      // config.ep_options
    std::map<std::string, std::string> session_config;  // config.session_config
    std::optional<QnnBackendInfo> qnn;                  // null unless the QNN EP is used
};

class Provider {
public:
    virtual ~Provider() = default;

    /// CLI name, e.g. "cpu".
    virtual std::string_view name() const = 0;
    /// Results-schema device type, e.g. "CPU".
    virtual std::string_view device_type() const = 0;

    virtual const ProviderDescription& describe() const = 0;

    /// Creates a session for `model`: checks its inputs and outputs (fixed shapes,
    /// supported dtypes; usage_error otherwise), generates the inputs and prepares
    /// the run binding. ONNX Runtime failures are ort_runtime_error with the ORT
    /// error code. Does nothing else, so its creation time is the session's alone.
    virtual Result<std::unique_ptr<BenchSession>> create_session(const std::filesystem::path& model,
                                                                 const RunConfig& config) = 0;

    /// Node placement for `model` with `config`'s session options, from a separate
    /// session created with verbose logging. Call it when no timed session follows
    /// (`run` calls it after the last trial), so it cannot warm anything up.
    virtual Result<NodeAssignment> record_placement(const std::filesystem::path& model,
                                                    const RunConfig& config) = 0;
};

/// The provider selected by `config.provider`. `runtime` must outlive it.
Result<std::unique_ptr<Provider>> make_provider(Runtime& runtime, const RunConfig& config);

}  // namespace edgebench
