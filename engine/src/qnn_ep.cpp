#include "edgebench/qnn_ep.hpp"

#include <string>
#include <system_error>

#include "edgebench/bundle_modules.hpp"

namespace fs = std::filesystem;

namespace edgebench {

fs::path qnn_ep_library_path(const fs::path& bundle_dir) {
    return bundle_dir / kQnnEpLibraryFile;
}

Result<EpLibrary*> register_qnn_ep(Runtime& runtime) {
    const fs::path bundle = runtime.bundle_dir();
    const fs::path library = qnn_ep_library_path(bundle);
    std::error_code ec;
    if (!fs::is_regular_file(library, ec)) {
        return Error{ExitCode::ort_load_failure, "qnn_ep_not_bundled",
                     "the QNN execution provider " + path_to_utf8(library) + " does not exist"};
    }

    Result<EpLibrary*> registered = runtime.register_ep_library(std::string(kQnnEpName), library);
    if (!registered) {
        Error e = registered.error();
        if (e.kind == "ep_library_not_found")
            e.kind = "qnn_ep_not_bundled";  // removed between the check and the registration
        return e;
    }

    // The EP library is loaded now; nothing it pulled in may come from elsewhere.
    if (Result<void> modules = check_bundle_modules(bundle).result(); !modules) {
        if (Result<void> undone = runtime.unregister_ep_library(kQnnEpName); !undone) {
            Error e = modules.error();
            e.message += " (unregistering the QNN execution provider also failed: " +
                         undone.error().message + ")";
            return e;
        }
        return modules.error();
    }
    return registered;
}

std::string_view qnn_ep_hint(std::string_view kind) {
    if (kind == "qnn_ep_not_bundled") {
        return qnn_ep_in_build()
                   ? "The build copies the QNN execution provider from the package fetched by "
                     "scripts/fetch_deps.ps1 next to the executable; rebuild, or re-run "
                     "scripts/fetch_deps.ps1 -Force and rebuild."
                   : "The QNN execution provider is bundled with ARM64 builds only; build with an "
                     "arm64 preset.";
    }
    if (kind == "ep_registration_failed") {
        return "Check that onnxruntime_providers_qnn.dll is the QNN execution provider fetched by "
               "scripts/fetch_deps.ps1 for this architecture; re-run scripts/fetch_deps.ps1 -Force "
               "and rebuild.";
    }
    if (kind == "module_outside_bundle")
        return load_error_hint(OrtLoadError::foreign_module_loaded);
    for (NpuSelectionFailure f :
         {NpuSelectionFailure::no_qnn_ep_devices, NpuSelectionFailure::no_npu_device,
          NpuSelectionFailure::multiple_npu_devices}) {
        if (kind == to_string(f))
            return hint(f);
    }
    return {};
}

}  // namespace edgebench
