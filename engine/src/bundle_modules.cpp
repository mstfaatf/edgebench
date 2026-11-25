#include "edgebench/bundle_modules.hpp"

#include <string>

namespace edgebench {

Result<void> BundleModuleCheck::result() const {
    if (outside.empty())
        return {};
    std::string names;
    for (const LoadedModule& m : outside)
        names += (names.empty() ? "" : ", ") + m.name + " (" + path_to_utf8(m.path) + ")";
    return Error{ExitCode::ort_load_failure, "module_outside_bundle",
                 "ONNX Runtime/QNN module loaded from outside the bundle directory " +
                     path_to_utf8(bundle_dir) + ": " + names};
}

BundleModuleCheck check_bundle_modules(const std::filesystem::path& bundle_dir) {
    BundleModuleCheck check;
    check.bundle_dir = bundle_dir;
    check.modules = list_ort_modules();
    check.outside = find_modules_outside(check.modules, bundle_dir);
    return check;
}

}  // namespace edgebench
