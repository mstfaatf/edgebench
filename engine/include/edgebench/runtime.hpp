// Runtime: owns the process's single ONNX Runtime environment (Ort::Env).
//
// Create it only after load_onnxruntime() succeeded. Every session, provider and
// Ort object created from it must be released before the Runtime.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "edgebench/ep_devices.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/result.hpp"

namespace edgebench {

class Runtime {
public:
    /// Fails (ort_load_failure) if `load` is not a successful load result.
    static Result<std::unique_ptr<Runtime>> create(const OrtLoadResult& load);

    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    Ort::Env& env() { return env_; }
    const OrtLoadResult& load_result() const { return load_; }

    /// Every EP device ONNX Runtime enumerates for the EPs registered so far.
    std::vector<EpDeviceInfo> ep_devices();

    /// Collects every ONNX Runtime log message delivered while it is alive (only
    /// one capture at a time). Messages from sessions with verbose log severity are
    /// included; outside a capture only warnings and errors are printed to stderr.
    class LogCapture {
    public:
        explicit LogCapture(Runtime& runtime);
        ~LogCapture();
        LogCapture(const LogCapture&) = delete;
        LogCapture& operator=(const LogCapture&) = delete;
        std::vector<std::string> messages() const;

    private:
        Runtime& runtime_;
        std::vector<std::string> messages_;
    };

private:
    explicit Runtime(const OrtLoadResult& load);
    static void ORT_API_CALL log_callback(void* param, OrtLoggingLevel severity,
                                          const char* category, const char* logid,
                                          const char* code_location, const char* message);

    OrtLoadResult load_;
    std::mutex log_mutex_;
    std::vector<std::string>* capture_ = nullptr;  // guarded by log_mutex_
    Ort::Env env_;                                 // last: its logger uses the members above
};

}  // namespace edgebench
