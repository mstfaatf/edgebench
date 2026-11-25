// OrtBenchSession: the BenchSession implementation shared by every ONNX Runtime
// provider. A provider supplies its Ort::SessionOptions; this class creates the
// session, checks the model's inputs and outputs, owns the input (and, with
// IoBinding, output) buffers, and runs inferences.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "edgebench/bench_session.hpp"

namespace edgebench {

class OrtBenchSession final : public BenchSession {
public:
    /// Creates the session from `model` with `options` (timed: session_create_ms()),
    /// reads and checks its inputs and outputs, fills the inputs as `config` says
    /// and prepares `config.binding`. session_create_ms() covers the Ort::Session
    /// construction only.
    static Result<std::unique_ptr<OrtBenchSession>> create(Ort::Env& env,
                                                           const std::filesystem::path& model,
                                                           const Ort::SessionOptions& options,
                                                           const RunConfig& config);

    const std::vector<TensorInfo>& inputs() const override { return inputs_; }
    const std::vector<TensorInfo>& outputs() const override { return outputs_; }
    std::int64_t batch_size() const override;
    RunBinding run_binding() const override { return binding_mode_; }
    std::span<float> input_data(std::size_t index) override;
    Result<void> run_once() override;
    TensorView output(std::size_t index) const override;
    double session_create_ms() const override { return create_ms_; }
    std::optional<std::int64_t> session_create_working_set_delta_bytes() const override {
        return create_ws_delta_;
    }

private:
    OrtBenchSession(Ort::Session session, RunBinding binding, double create_ms);
    Result<void> prepare(const RunConfig& config);

    Ort::Session session_;
    RunBinding binding_mode_;
    double create_ms_;
    std::optional<std::int64_t> create_ws_delta_;

    std::vector<TensorInfo> inputs_, outputs_;
    std::vector<std::string> input_names_, output_names_;
    std::vector<const char*> input_name_ptrs_, output_name_ptrs_;
    Ort::MemoryInfo cpu_memory_;
    Ort::RunOptions run_options_;

    std::vector<std::vector<float>> input_buffers_;
    std::vector<Ort::Value> input_values_;  // created once over input_buffers_

    // RunBinding::iobinding_preallocated
    std::vector<std::vector<std::byte>> output_buffers_;
    std::vector<Ort::Value> output_values_;  // created once over output_buffers_
    Ort::IoBinding binding_{nullptr};

    // RunBinding::run: outputs allocated by ONNX Runtime on each run
    std::vector<Ort::Value> run_outputs_;
    bool has_run_ = false;
};

/// ONNX Runtime's graph optimization level for `level`.
GraphOptimizationLevel ort_graph_optimization_level(GraphOptLevel level);

/// Inputs and outputs of an ONNX Runtime session as TensorInfo.
std::vector<TensorInfo> session_inputs(const Ort::Session& session);
std::vector<TensorInfo> session_outputs(const Ort::Session& session);

}  // namespace edgebench
