// Verification that every ONNX Runtime / QNN module in the process comes from the
// bundle directory (the directory of the loaded onnxruntime.dll).
//
// QNN DLLs load late: the EP library when it is registered, the HTP backend
// DLLs with the first NPU session. The check therefore runs again after each of
// those steps, not only after ONNX Runtime itself was loaded.
#pragma once

#include <filesystem>
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

}  // namespace edgebench
