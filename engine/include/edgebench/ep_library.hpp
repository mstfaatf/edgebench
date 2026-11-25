// Registration of an execution-provider plugin library (e.g. the QNN EP,
// onnxruntime_providers_qnn.dll) with an ONNX Runtime environment.
//
// A plugin EP library is loaded by ONNX Runtime when it is registered, adds its
// EP devices to the environment, and is unloaded when it is unregistered. Every
// session created on one of its devices must be released before that: a session
// holds a Lease, and unregistering refuses while any lease is held.
//
// Usually owned by the Runtime (Runtime::register_ep_library()), which releases
// its registrations before the environment itself; tools with their own Ort::Env
// can use EpLibrary directly.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>

#include "edgebench/result.hpp"

namespace edgebench {

class EpLibrary {
public:
    /// Registers the library at `path` under `registration_name`. The path must be
    /// absolute (no search-path loading). Errors: ep_library_not_found and
    /// ep_registration_failed (exit code 4, the library could not be loaded or is not
    /// an EP library), ep_library_path_not_absolute (exit code 1, a programming error).
    static Result<std::unique_ptr<EpLibrary>> register_library(Ort::Env& env,
                                                               std::string registration_name,
                                                               const std::filesystem::path& path);

    /// Unregisters the library if it is still registered and no lease is held.
    /// Errors are printed as warnings; with a lease still held the library is left
    /// registered (and loaded) rather than pulled from under a live session.
    ~EpLibrary();
    EpLibrary(const EpLibrary&) = delete;
    EpLibrary& operator=(const EpLibrary&) = delete;

    const std::string& registration_name() const { return name_; }
    const std::filesystem::path& path() const { return path_; }
    bool registered() const { return registered_; }

    /// Held for as long as a session uses one of the library's devices.
    class Lease {
    public:
        Lease() = default;
        ~Lease() { release(); }
        Lease(Lease&& other) noexcept : count_(std::move(other.count_)) {}
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release();
                count_ = std::move(other.count_);
            }
            return *this;
        }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        bool held() const { return count_ != nullptr; }
        void release() {
            if (count_) {
                --*count_;
                count_.reset();
            }
        }

    private:
        friend class EpLibrary;
        explicit Lease(std::shared_ptr<std::atomic<int>> count) : count_(std::move(count)) {
            ++*count_;
        }
        std::shared_ptr<std::atomic<int>> count_;
    };

    /// A lease for a session about to be created on one of the library's devices.
    Lease lease() { return Lease(leases_); }
    int active_leases() const { return leases_->load(); }

    /// Unregisters now. Errors: ep_library_in_use (exit code 1) while a lease is held,
    /// ep_unregistration_failed (exit code 3) if ONNX Runtime reports an error.
    /// Unregistering twice is a no-op.
    Result<void> unregister();

private:
    EpLibrary(Ort::Env& env, std::string name, std::filesystem::path path);

    Ort::Env& env_;
    std::string name_;
    std::filesystem::path path_;
    bool registered_ = true;
    std::shared_ptr<std::atomic<int>> leases_ = std::make_shared<std::atomic<int>>(0);
};

}  // namespace edgebench
