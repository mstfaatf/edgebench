// QnnProvider: the QNN execution provider on the NPU (QNN HTP backend).
//
// Created once per run: registers the bundled QNN EP library (qnn_ep.hpp),
// selects the single QNN NPU device (npu_selection.hpp) and checks that the HTP
// backend files exist, because the EP lists the NPU even without them. Every
// session is attached to that device, holds a lease on the EP library (so it
// cannot be unregistered under a live session), and uses exactly the provider
// options and session config entries of npu_options.hpp, which describe()
// records. Session creation failures are classified by
// classify_qnn_session_failure(): backend load failure (exit code 4), strict
// mode refusal (5), other ONNX Runtime errors (3).
#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <string>

#include "edgebench/ep_library.hpp"
#include "edgebench/provider.hpp"

namespace edgebench {

class Runtime;

class QnnProvider final : public Provider {
public:
    /// Errors: those of register_qnn_ep() and check_qnn_htp_backend() (exit code 4),
    /// of the device enumeration and select_qnn_npu() (exit code 3).
    static Result<std::unique_ptr<QnnProvider>> create(Runtime& runtime, const RunConfig& config);

    std::string_view name() const override { return "npu"; }
    std::string_view device_type() const override { return "NPU"; }
    const ProviderDescription& describe() const override { return description_; }
    Result<std::unique_ptr<BenchSession>> create_session(const std::filesystem::path& model,
                                                         const RunConfig& config) override;
    /// The same session with qnn_context_build_config(): writes the EP context.
    Result<std::unique_ptr<BenchSession>> create_context_building_session(
        const std::filesystem::path& model, const RunConfig& config,
        const std::filesystem::path& context_model) override;
    /// Placement from a separate verbose session with the same options, plus the
    /// QNN EP's node validation (NodeAssignment::qnn_validation). The modules are
    /// checked while that session exists, as after every QNN session.
    Result<NodeAssignment> record_placement(const std::filesystem::path& model,
                                            const RunConfig& config) override;

private:
    QnnProvider(Runtime& runtime, EpLibrary& library, std::size_t device_index,
                ProviderDescription description);

    /// Session options attached to the selected NPU: intra-op threads, sequential
    /// execution, graph optimization level, `session_config` and
    /// describe().ep_options.
    Result<Ort::SessionOptions> session_options(
        const RunConfig& config, const std::map<std::string, std::string>& session_config);

    Result<std::unique_ptr<BenchSession>> create(
        const std::filesystem::path& model, const RunConfig& config,
        const std::map<std::string, std::string>& session_config);

    Runtime& runtime_;
    EpLibrary& library_;
    std::size_t device_index_;  // in Runtime::ep_devices()
    ProviderDescription description_;
};

}  // namespace edgebench
