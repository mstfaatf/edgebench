#ifndef _WIN32
#error \
    "The ONNX Runtime loader (ort_loader.cpp) supports Windows only: it relies on LoadLibraryExW, delay-loading and the PSAPI module list."
#endif

#include "edgebench/ort_loader.hpp"

#include <onnxruntime_cxx_api.h>

#ifndef ORT_API_MANUAL_INIT
#error \
    "ORT_API_MANUAL_INIT must be defined for every target that uses ONNX Runtime; it is set on the OnnxRuntime::OnnxRuntime target in cmake/FindOnnxRuntime.cmake."
#endif

#include <algorithm>
#include <mutex>
#include <optional>
#include <system_error>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// windows.h first
#include <psapi.h>

#include "version_config.hpp"

namespace fs = std::filesystem;

namespace edgebench {

namespace {

constexpr const wchar_t* kOrtDllName = L"onnxruntime.dll";

std::string to_utf8(std::wstring_view w) {
    if (w.empty())
        return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0,
                                      nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), n, nullptr,
                        nullptr);
    return out;
}

fs::path module_file_name(HMODULE module) {
    std::wstring buf(MAX_PATH, L'\0');
    DWORD n = 0;
    while ((n = GetModuleFileNameW(module, buf.data(), static_cast<DWORD>(buf.size()))) ==
           buf.size()) {
        buf.resize(buf.size() * 2);
    }
    buf.resize(n);
    return fs::path(buf);
}

std::string system_error_text(DWORD code) {
    return std::system_category().message(static_cast<int>(code));
}

// Removes the "\\?\" prefix GetFinalPathNameByHandleW adds, for readable paths.
std::wstring strip_extended_prefix(std::wstring p) {
    if (p.starts_with(L"\\\\?\\UNC\\"))
        return L"\\\\" + p.substr(8);
    if (p.starts_with(L"\\\\?\\"))
        return p.substr(4);
    return p;
}

bool same_file_path(const fs::path& a, const fs::path& b) {
    return is_same_path(resolve_path(a).native(), resolve_path(b).native());
}

std::mutex g_load_mutex;
std::optional<OrtLoadResult> g_loaded;  // set after the first successful load

}  // namespace

std::string_view to_string(OrtLoadError error) {
    switch (error) {
        case OrtLoadError::none:
            return "none";
        case OrtLoadError::dll_not_found:
            return "dll_not_found";
        case OrtLoadError::load_failed:
            return "load_failed";
        case OrtLoadError::entry_point_missing:
            return "entry_point_missing";
        case OrtLoadError::version_mismatch:
            return "version_mismatch";
        case OrtLoadError::api_version_unsupported:
            return "api_version_unsupported";
        case OrtLoadError::foreign_module_loaded:
            return "foreign_module_loaded";
    }
    return "unknown";
}

std::string_view load_error_hint(OrtLoadError error) {
    switch (error) {
        case OrtLoadError::dll_not_found:
        case OrtLoadError::entry_point_missing:
        case OrtLoadError::version_mismatch:
            return "The build copies the bundled ONNX Runtime (lib/ of the package fetched by "
                   "scripts/fetch_deps.ps1) next to the executable; rebuild, or copy "
                   "onnxruntime.dll and onnxruntime_providers_shared.dll from there.";
        case OrtLoadError::load_failed:
            return "Check that the DLL is an ONNX Runtime build for this architecture and not "
                   "damaged; re-run scripts/fetch_deps.ps1 -Force and rebuild.";
        case OrtLoadError::api_version_unsupported:
            return "The headers and the DLL come from different packages; re-run "
                   "scripts/fetch_deps.ps1 -Force and rebuild.";
        case OrtLoadError::foreign_module_loaded:
            return "Something loaded another onnxruntime.dll first (an injected DLL, or a copy on "
                   "PATH or in System32); remove it from the process.";
        case OrtLoadError::none:
            break;
    }
    return {};
}

std::string_view OrtLoadResult::load_check() const {
    return ok() ? std::string_view("full_path_verified") : to_string(error_kind);
}

std::string_view expected_onnxruntime_version() noexcept {
    return EDGEBENCH_ONNXRUNTIME_VERSION_STRING;
}

int expected_onnxruntime_api_version() noexcept {
    return ORT_API_VERSION;
}

std::string path_to_utf8(const fs::path& path) {
    return to_utf8(path.native());
}

fs::path executable_dir() {
    return module_file_name(nullptr).parent_path();
}

fs::path resolve_path(const fs::path& path) {
    std::error_code ec;
    fs::path abs = fs::absolute(path, ec);
    if (ec)
        abs = path;
    // FILE_FLAG_BACKUP_SEMANTICS allows opening directories; no access rights are requested.
    HANDLE h = CreateFileW(abs.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        std::wstring buf(MAX_PATH, L'\0');
        DWORD n = GetFinalPathNameByHandleW(h, buf.data(), static_cast<DWORD>(buf.size()),
                                            FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        if (n >= buf.size()) {
            buf.resize(n + 1);
            n = GetFinalPathNameByHandleW(h, buf.data(), static_cast<DWORD>(buf.size()),
                                          FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        }
        CloseHandle(h);
        if (n > 0 && n < buf.size()) {
            buf.resize(n);
            return fs::path(strip_extended_prefix(std::move(buf)));
        }
    }
    // Could not open it (e.g. access denied): at least expand short names.
    const DWORD n = GetLongPathNameW(abs.c_str(), nullptr, 0);
    if (n > 0) {
        std::wstring buf(n, L'\0');
        const DWORD m = GetLongPathNameW(abs.c_str(), buf.data(), n);
        if (m > 0 && m < n) {
            buf.resize(m);
            return fs::path(buf);
        }
    }
    return abs;
}

fs::path loaded_module_path(std::wstring_view file_name) {
    const std::wstring name(file_name);
    HMODULE module = GetModuleHandleW(name.c_str());
    return module ? module_file_name(module) : fs::path();
}

std::vector<LoadedModule> list_ort_modules() {
    HANDLE process = GetCurrentProcess();
    std::vector<HMODULE> modules(1024);
    DWORD needed = 0;
    while (true) {
        const auto bytes = static_cast<DWORD>(modules.size() * sizeof(HMODULE));
        if (!K32EnumProcessModules(process, modules.data(), bytes, &needed))
            return {};
        if (needed <= bytes)
            break;
        modules.resize(needed / sizeof(HMODULE));
    }
    modules.resize(needed / sizeof(HMODULE));

    std::vector<LoadedModule> out;
    for (HMODULE m : modules) {
        std::wstring buf(MAX_PATH, L'\0');
        DWORD n = 0;
        while ((n = K32GetModuleFileNameExW(process, m, buf.data(),
                                            static_cast<DWORD>(buf.size()))) == buf.size()) {
            buf.resize(buf.size() * 2);
        }
        if (n == 0)
            continue;  // unloaded between the two calls
        buf.resize(n);
        const fs::path path(buf);
        const std::wstring name = path.filename().native();
        if (is_ort_or_qnn_module(name))
            out.push_back({to_utf8(name), path});
    }
    std::sort(out.begin(), out.end(),
              [](const LoadedModule& a, const LoadedModule& b) { return a.path < b.path; });
    return out;
}

std::vector<LoadedModule> find_modules_outside(const std::vector<LoadedModule>& modules,
                                               const fs::path& bundle_dir) {
    const std::wstring dir = resolve_path(bundle_dir).native();
    std::vector<LoadedModule> outside;
    for (const LoadedModule& m : modules) {
        if (!is_path_within(resolve_path(m.path).native(), dir))
            outside.push_back(m);
    }
    return outside;
}

OrtLoadResult load_onnxruntime() {
    return load_onnxruntime(executable_dir());
}

OrtLoadResult load_onnxruntime(const fs::path& bundle_dir) {
    const std::lock_guard<std::mutex> lock(g_load_mutex);
    if (g_loaded)
        return *g_loaded;

    OrtLoadResult r;
    std::error_code ec;
    const fs::path dir = fs::absolute(bundle_dir, ec);
    r.dll_path = (ec ? bundle_dir : dir) / kOrtDllName;
    const std::string dll = path_to_utf8(r.dll_path);
    const std::string expected(expected_onnxruntime_version());

    auto fail = [&r](OrtLoadError kind, std::string message) {
        r.status = OrtLoadStatus::failed;
        r.error_kind = kind;
        r.message = std::move(message);
        r.loaded_modules = list_ort_modules();
        return r;
    };

    // Another onnxruntime.dll already in the process (loaded by something else, or
    // through the normal search order) would be used by any code importing from it.
    for (const LoadedModule& m : list_ort_modules()) {
        if (is_same_path(m.path.filename().native(), kOrtDllName) &&
            !same_file_path(m.path, r.dll_path)) {
            return fail(OrtLoadError::foreign_module_loaded,
                        "an onnxruntime.dll from another location is already loaded in this "
                        "process: " +
                            path_to_utf8(m.path) + " (expected " + dll + ")");
        }
    }

    if (!fs::is_regular_file(r.dll_path, ec))
        return fail(OrtLoadError::dll_not_found, dll + " not found");

    // Absolute path; the DLL's own dependencies resolve from its directory first,
    // then the system directories. The current directory and PATH are not searched.
    HMODULE module =
        LoadLibraryExW(r.dll_path.c_str(), nullptr,
                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module) {
        const DWORD err = GetLastError();
        const std::string base = "LoadLibraryExW(" + dll + ") failed with error " +
                                 std::to_string(err) + ": " + system_error_text(err);
        if (err == ERROR_MOD_NOT_FOUND) {
            return fail(OrtLoadError::dll_not_found, base + " (a DLL it depends on is missing)");
        }
        return fail(OrtLoadError::load_failed, base);
    }
    auto fail_and_unload = [&](OrtLoadError kind, std::string message) {
        FreeLibrary(module);
        return fail(kind, std::move(message));
    };

    const fs::path loaded = module_file_name(module);
    if (!same_file_path(loaded, r.dll_path)) {
        return fail_and_unload(
            OrtLoadError::foreign_module_loaded,
            "loading " + dll + " returned a module from another location: " + path_to_utf8(loaded));
    }

    using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
    const auto get_api_base =
        reinterpret_cast<GetApiBaseFn>(GetProcAddress(module, "OrtGetApiBase"));
    const OrtApiBase* api_base = get_api_base ? get_api_base() : nullptr;
    if (!api_base) {
        return fail_and_unload(
            OrtLoadError::entry_point_missing,
            dll + " does not export OrtGetApiBase, so it is not an ONNX Runtime library");
    }
    const char* version = api_base->GetVersionString ? api_base->GetVersionString() : nullptr;
    r.ort_version = version ? version : "";

    if (!version_matches(r.ort_version, expected)) {
        return fail_and_unload(
            OrtLoadError::version_mismatch,
            dll + " is ONNX Runtime " +
                (r.ort_version.empty() ? std::string("(no version)") : r.ort_version) +
                ", but EdgeBench requires ONNX Runtime " + expected);
    }

    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    if (!api) {
        return fail_and_unload(
            OrtLoadError::api_version_unsupported,
            dll + " (ONNX Runtime " + r.ort_version + ") does not provide C API version " +
                std::to_string(ORT_API_VERSION) + ", which EdgeBench's headers were built for");
    }

    Ort::InitApi(api);
    r.status = OrtLoadStatus::ok;
    r.error_kind = OrtLoadError::none;
    r.api_version = ORT_API_VERSION;
    r.loaded_modules = list_ort_modules();
    g_loaded = r;
    return r;
}

}  // namespace edgebench
