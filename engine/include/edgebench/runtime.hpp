// Runtime: owns the process's single ONNX Runtime environment (Ort::Env).
//
// Create it only after load_onnxruntime() succeeded. Every session, provider and
// Ort object created from it must be released before the Runtime.
//
// Release the Runtime before main() returns: own it in a local (as edgebench.exe
// does) or reset it explicitly, never in a static or function-static object. In
// a process that has created a QNN session, releasing the ONNX Runtime
// environment and its EP registrations during static destruction after main()
// can hang the process forever. live_count() lets test programs check that no
// Runtime is left when they finish (engine/tests/test_runtime.cpp).
//
// The Runtime also owns the execution-provider plugin libraries registered with
// its environment (register_ep_library()). They are unregistered before the
// environment is released, and never while a session still holds a lease on
// one (EpLibrary::Lease), so the library outlives every session using it, on
// error paths and after an interrupt as well.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/ep_devices.hpp"
#include "edgebench/ep_library.hpp"
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

    /// Runtime objects alive in this process.
    static int live_count();

    Ort::Env& env() { return env_; }
    const OrtLoadResult& load_result() const { return load_; }

    /// Directory of the loaded onnxruntime.dll: every ONNX Runtime / QNN module the
    /// process loads must come from here.
    std::filesystem::path bundle_dir() const { return load_.dll_path.parent_path(); }

    /// Every EP device ONNX Runtime enumerates for the EPs registered so far.
    std::vector<EpDeviceInfo> ep_devices();

    /// Registers an EP plugin library by absolute path (EpLibrary::register_library()).
    /// Registering the same name and path again returns the existing registration;
    /// the same name with another path is an error (ep_library_conflict, exit 1).
    Result<EpLibrary*> register_ep_library(std::string registration_name,
                                           const std::filesystem::path& path);

    /// The registered library with this name, or nullptr.
    EpLibrary* ep_library(std::string_view registration_name);

    /// Unregisters and forgets the library (EpLibrary::unregister()); unknown names
    /// are a no-op. On error the registration is kept.
    Result<void> unregister_ep_library(std::string_view registration_name);

    /// Warning and error messages ONNX Runtime has logged so far (captured or not),
    /// numbered from 0; the most recent kLogHistory are kept. Read the position
    /// before an operation and the messages since it afterwards, e.g. to classify
    /// a failed session creation by what the execution provider reported.
    static constexpr std::size_t kLogHistory = 256;
    std::uint64_t log_position();
    std::vector<std::string> warnings_since(std::uint64_t position);

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
    std::uint64_t logged_warnings_ = 0;            // guarded by log_mutex_
    std::deque<std::string> warning_history_;      // guarded by log_mutex_; the newest last
    Ort::Env env_;                                 // after the members its logger uses
    // After env_, so the libraries are unregistered before the environment is released.
    std::vector<std::unique_ptr<EpLibrary>> ep_libraries_;
};

}  // namespace edgebench
