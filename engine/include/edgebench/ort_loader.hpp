// Explicit loading of the bundled ONNX Runtime.
//
// Every target that links ONNX Runtime is built with ORT_API_MANUAL_INIT and
// delay-loads onnxruntime.dll (see cmake/FindOnnxRuntime.cmake), so nothing
// touches ONNX Runtime until load_onnxruntime() runs. It loads the
// onnxruntime.dll next to the executable by absolute path, checks its version
// and C API version, and only then initializes the ORT C++ API. A missing,
// foreign, or incompatible DLL becomes an error result instead of a crash.
//
// Windows ships an unrelated, older onnxruntime.dll in System32; without these
// steps the normal DLL search order can pick it up.
#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/module_paths.hpp"

namespace edgebench {

enum class OrtLoadStatus { ok, failed };

enum class OrtLoadError {
    none,
    /// onnxruntime.dll (or one of its dependencies) is not where it should be.
    dll_not_found,
    /// The file exists but Windows could not load it (e.g. wrong architecture, corrupt).
    load_failed,
    /// The DLL does not export OrtGetApiBase, so it is not ONNX Runtime.
    entry_point_missing,
    /// The DLL is ONNX Runtime, but not the version EdgeBench was built against.
    version_mismatch,
    /// The DLL has the expected version but does not provide the C API version
    /// the headers were built for.
    api_version_unsupported,
    /// An onnxruntime.dll from another directory was already loaded into the process.
    foreign_module_loaded,
};

/// Stable identifier, e.g. "dll_not_found"; "none" for OrtLoadError::none.
std::string_view to_string(OrtLoadError error);

/// One-line advice on how to fix a load error (empty for OrtLoadError::none).
std::string_view load_error_hint(OrtLoadError error);

/// A loaded DLL: file name and absolute path (results schema: runtime.loaded_modules[]).
struct LoadedModule {
    std::string name;  // UTF-8 file name, e.g. "onnxruntime.dll"
    std::filesystem::path path;
};

struct OrtLoadResult {
    OrtLoadStatus status = OrtLoadStatus::failed;
    OrtLoadError error_kind = OrtLoadError::none;
    std::string message;             // human-readable explanation; empty on success
    std::string ort_version;         // as reported by the DLL; empty if it could not be read
    int api_version = 0;             // C API version obtained (ORT_API_VERSION) on success, else 0
    std::filesystem::path dll_path;  // the onnxruntime.dll that was examined
    std::vector<LoadedModule> loaded_modules;  // ORT/QNN modules right after loading

    bool ok() const { return status == OrtLoadStatus::ok; }
    /// Results schema runtime.onnxruntime.load_check: "full_path_verified" on
    /// success, otherwise the error kind.
    std::string_view load_check() const;
};

/// The ONNX Runtime version EdgeBench is built against, e.g. "1.32.0".
std::string_view expected_onnxruntime_version() noexcept;

/// The C API version the ONNX Runtime headers define (ORT_API_VERSION).
int expected_onnxruntime_api_version() noexcept;

/// Directory of the running executable (from the module file name, not the
/// current working directory).
std::filesystem::path executable_dir();

/// Loads <executable_dir()>/onnxruntime.dll and initializes the ORT C++ API.
/// Must be called before any Ort:: API call. Thread-safe; after a successful
/// load, later calls return the same result without loading again.
OrtLoadResult load_onnxruntime();

/// Same, with an explicit bundle directory instead of executable_dir().
OrtLoadResult load_onnxruntime(const std::filesystem::path& bundle_dir);

/// Absolute path of a loaded module by file name, or empty if it is not loaded.
std::filesystem::path loaded_module_path(std::wstring_view file_name);

/// Every ONNX Runtime / QNN module loaded in this process (see
/// is_ort_or_qnn_module()), sorted by path. Call it while sessions are alive:
/// the QNN backend DLLs are unloaded together with the last session using them.
std::vector<LoadedModule> list_ort_modules();

/// The modules that are not inside `bundle_dir`. Both sides are resolved to
/// their final long-name form (resolve_path()) and compared case-insensitively.
std::vector<LoadedModule> find_modules_outside(const std::vector<LoadedModule>& modules,
                                               const std::filesystem::path& bundle_dir);

/// Absolute, final form of an existing path: links and short (8.3) names are
/// resolved. Returns the absolute input path if it cannot be opened.
std::filesystem::path resolve_path(const std::filesystem::path& path);

/// UTF-8 text of a path, for printing and JSON.
std::string path_to_utf8(const std::filesystem::path& path);

}  // namespace edgebench
