// ONNX Runtime + QNN plugin EP plumbing for ort_probe: device listing, session
// options, and a runner for plain Run / IoBinding. Loading ONNX Runtime, registering
// the QNN EP library, checking the loaded DLLs and selecting the NPU device are done
// by edgebench_core (ort_loader.hpp, ep_library.hpp, bundle_modules.hpp,
// npu_selection.hpp).
#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "probe_util.hpp"

namespace probe {

inline constexpr const char* kQnnHtpBackend = "QnnHtp.dll";

// --------------------------------------------------------------------------- devices

const char* DeviceTypeName(OrtHardwareDeviceType type);
void PrintDevices(const std::vector<Ort::ConstEpDevice>& devices);
void WriteDevices(JsonWriter& json, const std::vector<Ort::ConstEpDevice>& devices);

// --------------------------------------------------------------------------- sessions

// QNN provider options matching the Python harness (scripts/common/npu_harness.py).
std::map<std::string, std::string> QnnProviderOptions(const Options& o,
                                                      const std::filesystem::path& exe_dir);
// Session config entries, e.g. session.disable_cpu_ep_fallback.
std::map<std::string, std::string> SessionConfig(const Options& o, bool generate_context);

// CPU session options: intra-op threads only (strict mode and fallback do not apply).
Ort::SessionOptions CpuSessionOptions(int threads);
// NPU session options: QNN EP attached to `npu`, plus SessionConfig(o, generate_context).
Ort::SessionOptions NpuSessionOptions(Ort::Env& env, const Ort::ConstEpDevice& npu,
                                      const Options& o, const std::filesystem::path& exe_dir,
                                      bool generate_context);

// One float32 input -> first float32 output, with a fully static input shape.
class ModelRunner {
public:
    explicit ModelRunner(Ort::Session session);

    const std::vector<int64_t>& input_shape() const { return input_shape_; }
    size_t input_count() const { return input_count_; }
    size_t output_count() const { return output_count_; }

    // Session::Run with an Ort::Value created over `input` on every call.
    std::vector<float> Run(std::span<const float> input);
    // IoBinding with input and output buffers allocated once; Prepare() copies the input.
    void PrepareBinding(std::span<const float> input);
    void RunBound();
    std::vector<float> BoundOutput() const { return output_buffer_; }

    // Runs once in the given mode and returns the output.
    std::vector<float> RunWith(RunMode mode, std::span<const float> input);

    Ort::Session& session() { return session_; }

private:
    Ort::Session session_;
    std::string input_name_, output_name_;
    std::vector<int64_t> input_shape_, output_shape_;
    size_t input_count_ = 0, output_count_ = 0;
    Ort::MemoryInfo cpu_memory_;
    Ort::RunOptions run_options_;
    // IoBinding state
    std::vector<float> input_buffer_, output_buffer_;
    Ort::Value bound_input_{nullptr}, bound_output_{nullptr};
    Ort::IoBinding binding_{nullptr};
};

const char* OrtErrorCodeName(OrtErrorCode code);

}  // namespace probe
