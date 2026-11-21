#include "edgebench/ort_bench_session.hpp"

#include <chrono>

#include "edgebench/input_fill.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/system_info.hpp"

namespace edgebench {

namespace {

TensorInfo tensor_info(const std::string& name, const Ort::TypeInfo& type) {
    TensorInfo t;
    t.name = name;
    if (type.GetONNXType() != ONNX_TYPE_TENSOR)
        return t;  // dtype stays UNDEFINED and is rejected by check_supported_dtypes
    // GetTensorTypeAndShapeInfo() is a view into `type`, which must outlive it.
    const auto info = type.GetTensorTypeAndShapeInfo();
    t.dtype = info.GetElementType();
    t.shape = info.GetShape();
    t.dim_params.resize(t.shape.size());
    std::vector<const char*> symbolic(t.shape.size(), nullptr);
    info.GetSymbolicDimensions(symbolic.data(), symbolic.size());
    for (std::size_t i = 0; i < symbolic.size(); ++i)
        t.dim_params[i] = symbolic[i] ? symbolic[i] : "";
    return t;
}

}  // namespace

std::vector<TensorInfo> session_inputs(const Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<TensorInfo> out;
    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
        const std::string name = session.GetInputNameAllocated(i, allocator).get();
        out.push_back(tensor_info(name, session.GetInputTypeInfo(i)));
    }
    return out;
}

std::vector<TensorInfo> session_outputs(const Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<TensorInfo> out;
    for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
        const std::string name = session.GetOutputNameAllocated(i, allocator).get();
        out.push_back(tensor_info(name, session.GetOutputTypeInfo(i)));
    }
    return out;
}

OrtBenchSession::OrtBenchSession(Ort::Session session, RunBinding binding, double create_ms)
    : session_(std::move(session)),
      binding_mode_(binding),
      create_ms_(create_ms),
      cpu_memory_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {}

Result<std::unique_ptr<OrtBenchSession>> OrtBenchSession::create(Ort::Env& env,
                                                                 const std::filesystem::path& model,
                                                                 const Ort::SessionOptions& options,
                                                                 const RunConfig& config) {
    std::unique_ptr<OrtBenchSession> s;
    try {
        // Working set read just outside the clock reads, so it is not timed.
        const ProcessMemory before = query_process_memory();
        const auto t0 = std::chrono::steady_clock::now();
        Ort::Session session(env, model.c_str(), options);
        const auto t1 = std::chrono::steady_clock::now();
        const ProcessMemory after = query_process_memory();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        s.reset(new OrtBenchSession(std::move(session), config.binding, ms));
        if (before.working_set_bytes > 0 && after.working_set_bytes > 0) {
            s->create_ws_delta_ = static_cast<std::int64_t>(after.working_set_bytes) -
                                  static_cast<std::int64_t>(before.working_set_bytes);
        }
    } catch (const Ort::Exception& e) {
        return ort_error("session_creation_failed", "creating a session for " + path_to_utf8(model),
                         e.GetOrtErrorCode(), e.what());
    }
    if (Result<void> r = s->prepare(config); !r)
        return r.error();
    return s;
}

Result<void> OrtBenchSession::prepare(const RunConfig& config) {
    try {
        inputs_ = session_inputs(session_);
        outputs_ = session_outputs(session_);
    } catch (const Ort::Exception& e) {
        return ort_error("session_metadata_failed", "reading the model's inputs and outputs",
                         e.GetOrtErrorCode(), e.what());
    }
    if (inputs_.empty())
        return Error{ExitCode::usage_error, "no_inputs", "the model has no inputs"};
    if (Result<void> r = check_fixed_shapes(inputs_, outputs_); !r)
        return r;
    if (Result<void> r = check_supported_dtypes(inputs_, outputs_); !r)
        return r;

    for (const TensorInfo& t : inputs_)
        input_names_.push_back(t.name);
    for (const TensorInfo& t : outputs_)
        output_names_.push_back(t.name);
    for (const std::string& n : input_names_)
        input_name_ptrs_.push_back(n.c_str());
    for (const std::string& n : output_names_)
        output_name_ptrs_.push_back(n.c_str());

    // Inputs: generated once, in model order, from one generator (docs/cli.md).
    SplitMix64 rng(config.effective_seed());
    input_buffers_.reserve(inputs_.size());
    for (const TensorInfo& t : inputs_) {
        std::vector<float>& buf = input_buffers_.emplace_back(t.element_count());
        if (config.input == InputKind::random)
            fill_random(buf, rng);
        else
            fill_constant(buf, static_cast<float>(config.effective_constant_value()));
    }

    try {
        for (std::size_t i = 0; i < inputs_.size(); ++i) {
            input_values_.push_back(Ort::Value::CreateTensor<float>(
                cpu_memory_, input_buffers_[i].data(), input_buffers_[i].size(),
                inputs_[i].shape.data(), inputs_[i].shape.size()));
        }
        if (binding_mode_ == RunBinding::iobinding_preallocated) {
            binding_ = Ort::IoBinding(session_);
            for (std::size_t i = 0; i < inputs_.size(); ++i)
                binding_.BindInput(input_names_[i].c_str(), input_values_[i]);
            for (std::size_t i = 0; i < outputs_.size(); ++i) {
                std::vector<std::byte>& buf = output_buffers_.emplace_back(outputs_[i].byte_size());
                output_values_.push_back(Ort::Value::CreateTensor(
                    cpu_memory_, buf.data(), buf.size(), outputs_[i].shape.data(),
                    outputs_[i].shape.size(), outputs_[i].dtype));
                binding_.BindOutput(output_names_[i].c_str(), output_values_[i]);
            }
        }
    } catch (const Ort::Exception& e) {
        return ort_error("binding_failed", "preparing input and output tensors",
                         e.GetOrtErrorCode(), e.what());
    }
    return {};
}

std::int64_t OrtBenchSession::batch_size() const {
    const std::vector<std::int64_t>& shape = inputs_.front().shape;
    return shape.empty() ? 1 : shape.front();
}

std::span<float> OrtBenchSession::input_data(std::size_t index) {
    return input_buffers_.at(index);
}

Result<void> OrtBenchSession::run_once() {
    try {
        if (binding_mode_ == RunBinding::iobinding_preallocated) {
            session_.Run(run_options_, binding_);
        } else {
            run_outputs_ = session_.Run(run_options_, input_name_ptrs_.data(), input_values_.data(),
                                        input_values_.size(), output_name_ptrs_.data(),
                                        output_name_ptrs_.size());
        }
    } catch (const Ort::Exception& e) {
        return ort_error("inference_failed", "running an inference", e.GetOrtErrorCode(), e.what());
    }
    has_run_ = true;
    return {};
}

TensorView OrtBenchSession::output(std::size_t index) const {
    const TensorInfo& info = outputs_.at(index);
    if (!has_run_)
        return {info.dtype, {}};
    if (binding_mode_ == RunBinding::iobinding_preallocated) {
        const std::vector<std::byte>& buf = output_buffers_[index];
        return {info.dtype, {buf.data(), buf.size()}};
    }
    const Ort::Value& v = run_outputs_.at(index);
    const auto* data = static_cast<const std::byte*>(v.GetTensorRawData());
    const std::size_t bytes =
        v.GetTensorTypeAndShapeInfo().GetElementCount() * TensorInfo::dtype_size(info.dtype);
    return {info.dtype, {data, bytes}};
}

}  // namespace edgebench
