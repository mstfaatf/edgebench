#include "edgebench/bundle_modules.hpp"

#include <windows.h>

#include <cstdint>
#include <string>
#include <vector>

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

std::optional<ModuleFileVersion> read_module_file_version(const std::filesystem::path& path) {
    DWORD ignored = 0;
    const DWORD size = GetFileVersionInfoSizeW(path.c_str(), &ignored);
    if (size == 0)
        return std::nullopt;
    std::vector<std::uint8_t> data(size);
    if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data()))
        return std::nullopt;
    ModuleFileVersion out;
    out.module_path = path;
    VS_FIXEDFILEINFO* fixed = nullptr;
    UINT length = 0;
    if (VerQueryValueW(data.data(), LR"(\)", reinterpret_cast<void**>(&fixed), &length) && fixed &&
        length >= sizeof(VS_FIXEDFILEINFO)) {
        out.file_version = std::to_string(HIWORD(fixed->dwFileVersionMS)) + "." +
                           std::to_string(LOWORD(fixed->dwFileVersionMS)) + "." +
                           std::to_string(HIWORD(fixed->dwFileVersionLS)) + "." +
                           std::to_string(LOWORD(fixed->dwFileVersionLS));
    }
    // The first language/code page the resource declares (US English / Unicode,
    // 040904B0, for the QNN backends).
    struct Translation {
        WORD language;
        WORD code_page;
    };
    Translation* translations = nullptr;
    std::vector<std::wstring> keys;
    if (VerQueryValueW(data.data(), LR"(\VarFileInfo\Translation)",
                       reinterpret_cast<void**>(&translations), &length) &&
        translations && length >= sizeof(Translation)) {
        wchar_t key[64];
        swprintf_s(key, LR"(\StringFileInfo\%04x%04x\ProductVersion)", translations[0].language,
                   translations[0].code_page);
        keys.emplace_back(key);
    }
    keys.emplace_back(LR"(\StringFileInfo\040904b0\ProductVersion)");
    for (const std::wstring& key : keys) {
        wchar_t* text = nullptr;
        if (VerQueryValueW(data.data(), key.c_str(), reinterpret_cast<void**>(&text), &length) &&
            text && length > 1) {
            const std::wstring value(text, wcsnlen(text, length));
            out.product_version = path_to_utf8(std::filesystem::path(value));
            break;
        }
    }
    return out;
}

const LoadedModule* find_loaded_module(const std::vector<LoadedModule>& modules,
                                       std::string_view file_name) {
    for (const LoadedModule& m : modules) {
        if (m.name.size() == file_name.size() &&
            CompareStringOrdinal(std::filesystem::path(m.name).c_str(), -1,
                                 std::filesystem::path(file_name).c_str(), -1, TRUE) == CSTR_EQUAL)
            return &m;
    }
    return nullptr;
}

}  // namespace edgebench
