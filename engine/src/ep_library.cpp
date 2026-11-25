#include "edgebench/ep_library.hpp"

#include <system_error>

#include "edgebench/diagnostics.hpp"
#include "edgebench/ort_loader.hpp"

namespace fs = std::filesystem;

namespace edgebench {

Result<std::unique_ptr<EpLibrary>> EpLibrary::register_library(Ort::Env& env,
                                                               std::string registration_name,
                                                               const fs::path& path) {
    const std::string where = path_to_utf8(path);
    if (!path.is_absolute()) {
        return Error{
            ExitCode::internal_error, "ep_library_path_not_absolute",
            "execution provider libraries are registered by absolute path only, got " + where};
    }
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) {
        return Error{ExitCode::ort_load_failure, "ep_library_not_found",
                     "execution provider library " + where + " not found"};
    }
    try {
        env.RegisterExecutionProviderLibrary(registration_name.c_str(), path.native());
    } catch (const Ort::Exception& e) {
        Error error = ort_error(
            "ep_registration_failed",
            "registering the execution provider library " + where + " as " + registration_name,
            e.GetOrtErrorCode(), e.what());
        error.exit_code = ExitCode::ort_load_failure;  // the library could not be used at all
        return error;
    }
    return std::unique_ptr<EpLibrary>(new EpLibrary(env, std::move(registration_name), path));
}

EpLibrary::EpLibrary(Ort::Env& env, std::string name, fs::path path)
    : env_(env), name_(std::move(name)), path_(std::move(path)) {}

EpLibrary::~EpLibrary() {
    if (!registered_)
        return;
    if (const Result<void> r = unregister(); !r)
        print_warning(r.error().kind, r.error().message);
}

Result<void> EpLibrary::unregister() {
    if (!registered_)
        return {};
    if (const int held = active_leases(); held > 0) {
        return Error{ExitCode::internal_error, "ep_library_in_use",
                     "cannot unregister " + name_ + " (" + path_to_utf8(path_) + "): " +
                         std::to_string(held) + " session(s) using its devices are still alive"};
    }
    try {
        env_.UnregisterExecutionProviderLibrary(name_.c_str());
    } catch (const Ort::Exception& e) {
        return ort_error("ep_unregistration_failed",
                         "unregistering the execution provider library " + name_,
                         e.GetOrtErrorCode(), e.what());
    }
    registered_ = false;
    return {};
}

}  // namespace edgebench
