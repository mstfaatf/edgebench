// CpuProvider: ONNX Runtime's built-in CPU execution provider.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>

#include "edgebench/provider.hpp"

namespace edgebench {

class Runtime;

class CpuProvider final : public Provider {
public:
    static constexpr const char* kExecutionProvider = "CPUExecutionProvider";

    /// Fails (internal_error) unless ONNX Runtime enumerates exactly one CPU EP
    /// device of type CPU.
    static Result<std::unique_ptr<CpuProvider>> create(Runtime& runtime);

    std::string_view name() const override { return "cpu"; }
    std::string_view device_type() const override { return "CPU"; }
    const ProviderDescription& describe() const override { return description_; }
    Result<std::unique_ptr<BenchSession>> create_session(const std::filesystem::path& model,
                                                         const RunConfig& config) override;

    /// Session options for `config`: explicit intra-op threads, sequential
    /// execution, graph optimization level, and describe().session_config.
    Ort::SessionOptions session_options(const RunConfig& config) const;

private:
    CpuProvider(Runtime& runtime, ProviderDescription description);

    Runtime& runtime_;
    ProviderDescription description_;
};

}  // namespace edgebench
