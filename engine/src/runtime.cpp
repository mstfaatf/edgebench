#include "edgebench/runtime.hpp"

#include <cstdio>
#include <stdexcept>

namespace edgebench {

Result<std::unique_ptr<Runtime>> Runtime::create(const OrtLoadResult& load) {
    if (!load.ok()) {
        return Error{ExitCode::ort_load_failure, std::string(to_string(load.error_kind)),
                     "cannot create the ONNX Runtime environment: " + load.message};
    }
    try {
        return std::unique_ptr<Runtime>(new Runtime(load));
    } catch (const Ort::Exception& e) {
        return ort_error("environment_creation_failed", "creating the ONNX Runtime environment",
                         e.GetOrtErrorCode(), e.what());
    }
}

Runtime::Runtime(const OrtLoadResult& load)
    : load_(load), env_(ORT_LOGGING_LEVEL_WARNING, "edgebench", &Runtime::log_callback, this) {}

Runtime::~Runtime() = default;

std::vector<EpDeviceInfo> Runtime::ep_devices() {
    return list_ep_devices(env_);
}

void ORT_API_CALL Runtime::log_callback(void* param, OrtLoggingLevel severity, const char*,
                                        const char*, const char*, const char* message) {
    auto* self = static_cast<Runtime*>(param);
    const char* text = message ? message : "";
    {
        const std::lock_guard<std::mutex> lock(self->log_mutex_);
        if (self->capture_) {
            self->capture_->emplace_back(text);
            return;  // captured sessions log verbosely; keep stderr clean
        }
    }
    if (severity >= ORT_LOGGING_LEVEL_WARNING) {
        const char* level = severity >= ORT_LOGGING_LEVEL_ERROR ? "error" : "warning";
        std::fprintf(stderr, "onnxruntime %s: %s\n", level, text);
    }
}

Runtime::LogCapture::LogCapture(Runtime& runtime) : runtime_(runtime) {
    const std::lock_guard<std::mutex> lock(runtime_.log_mutex_);
    if (runtime_.capture_)
        throw std::logic_error("Runtime::LogCapture: a capture is already active");
    runtime_.capture_ = &messages_;
}

Runtime::LogCapture::~LogCapture() {
    const std::lock_guard<std::mutex> lock(runtime_.log_mutex_);
    runtime_.capture_ = nullptr;
}

std::vector<std::string> Runtime::LogCapture::messages() const {
    const std::lock_guard<std::mutex> lock(runtime_.log_mutex_);
    return messages_;
}

}  // namespace edgebench
