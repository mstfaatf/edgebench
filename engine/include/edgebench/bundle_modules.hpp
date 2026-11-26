// Verification that every ONNX Runtime / QNN module in the process comes from the
// bundle directory (the directory of the loaded onnxruntime.dll).
//
// QNN DLLs load late: the EP library when it is registered, the HTP backend
// DLLs with the first NPU session. The check therefore runs again after each of
// those steps, not only after ONNX Runtime itself was loaded.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/ort_loader.hpp"
#include "edgebench/result.hpp"

namespace edgebench {

struct BundleModuleCheck {
    std::filesystem::path bundle_dir;
    std::vector<LoadedModule> modules;  // every ORT/QNN module loaded at the time of the check
    std::vector<LoadedModule> outside;  // the ones that are not inside bundle_dir

    bool ok() const { return outside.empty(); }
    /// Success, or module_outside_bundle (exit code 4) naming each module and its path.
    Result<void> result() const;
};

/// Lists the ORT/QNN modules loaded now (list_ort_modules(): long-path normalized,
/// same file name filter) and compares them with `bundle_dir` (find_modules_outside()).
BundleModuleCheck check_bundle_modules(const std::filesystem::path& bundle_dir);

/// One recorded module check of a run (results schema:
/// execution.verification_details.module_checks).
struct ModuleCheckpoint {
    std::string stage;                  // after_provider_creation | after_session_creation |
                                        // placement_session
    std::optional<int> trial_index;     // after_session_creation only
    std::vector<LoadedModule> outside;  // modules found outside the bundle (empty: passed)
};

/// The version resource of a module's file (results schema:
/// runtime.qnn.backend_file_version).
struct ModuleFileVersion {
    std::filesystem::path module_path;
    std::optional<std::string> product_version;  // StringFileInfo ProductVersion
    std::optional<std::string> file_version;     // VS_FIXEDFILEINFO, "a.b.c.d"
};

/// Reads the version resource of `path`; nullopt if the file has none.
std::optional<ModuleFileVersion> read_module_file_version(const std::filesystem::path& path);

/// The module named `file_name` (case-insensitive) in `modules`, or nullptr.
const LoadedModule* find_loaded_module(const std::vector<LoadedModule>& modules,
                                       std::string_view file_name);

}  // namespace edgebench
