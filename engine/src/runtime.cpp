#include "edgebench/runtime.hpp"

#include "edgebench/diagnostics.hpp"

#include <atomic>
#include <cstdio>
#include <stdexcept>

namespace edgebench {

namespace {
std::atomic<int> g_live_runtimes{0};
}  // namespace

int Runtime::live_count() {
    return g_live_runtimes.load();
}

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
    : load_(load), env_(ORT_LOGGING_LEVEL_WARNING, "edgebench", &Runtime::log_callback, this) {
    ++g_live_runtimes;
}

Runtime::~Runtime() {
    // Most recently registered first; each EpLibrary unregisters itself.
    while (!ep_libraries_.empty())
        ep_libraries_.pop_back();
    --g_live_runtimes;
}

std::vector<EpDeviceInfo> Runtime::ep_devices() {
    return list_ep_devices(env_);
}

Result<EpLibrary*> Runtime::register_ep_library(std::string registration_name,
                                                const std::filesystem::path& path) {
    if (EpLibrary* existing = ep_library(registration_name)) {
        if (is_same_path(existing->path().native(), path.native()))
            return existing;
        return Error{ExitCode::internal_error, "ep_library_conflict",
                     registration_name + " is already registered from " +
                         path_to_utf8(existing->path()) + ", not " + path_to_utf8(path)};
    }
    Result<std::unique_ptr<EpLibrary>> library =
        EpLibrary::register_library(env_, std::move(registration_name), path);
    if (!library)
        return library.error();
    ep_libraries_.push_back(std::move(library).value());
    return ep_libraries_.back().get();
}

EpLibrary* Runtime::ep_library(std::string_view registration_name) {
    for (const std::unique_ptr<EpLibrary>& library : ep_libraries_) {
        if (library->registration_name() == registration_name)
            return library.get();
    }
    return nullptr;
}

Result<void> Runtime::unregister_ep_library(std::string_view registration_name) {
    for (auto it = ep_libraries_.begin(); it != ep_libraries_.end(); ++it) {
        if ((*it)->registration_name() != registration_name)
            continue;
        if (Result<void> r = (*it)->unregister(); !r)
            return r;
        ep_libraries_.erase(it);
        return {};
    }
    return {};
}

std::uint64_t Runtime::log_position() {
    const std::lock_guard<std::mutex> lock(log_mutex_);
    return logged_warnings_;
}

std::vector<std::string> Runtime::warnings_since(std::uint64_t position) {
    const std::lock_guard<std::mutex> lock(log_mutex_);
    const std::uint64_t kept = warning_history_.size();
    const std::uint64_t first_kept = logged_warnings_ - kept;
    const std::uint64_t from = position > first_kept ? position - first_kept : 0;
    if (from >= kept)
        return {};
    return {warning_history_.begin() + static_cast<std::ptrdiff_t>(from), warning_history_.end()};
}

void ORT_API_CALL Runtime::log_callback(void* param, OrtLoggingLevel severity, const char*,
                                        const char*, const char*, const char* message) {
    auto* self = static_cast<Runtime*>(param);
    const char* text = message ? message : "";
    {
        const std::lock_guard<std::mutex> lock(self->log_mutex_);
        if (severity >= ORT_LOGGING_LEVEL_WARNING) {
            ++self->logged_warnings_;
            self->warning_history_.emplace_back(text);
            if (self->warning_history_.size() > kLogHistory)
                self->warning_history_.pop_front();
        }
        if (self->capture_) {
            self->capture_->emplace_back(text);
            return;  // captured sessions log verbosely; keep stderr clean
        }
    }
    if (severity >= ORT_LOGGING_LEVEL_WARNING) {
        // ONNX Runtime's own log lines; failures are reported separately as errors.
        print_warning(severity >= ORT_LOGGING_LEVEL_ERROR ? "onnxruntime_error" : "onnxruntime",
                      text);
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
