// The QNN execution provider (onnxruntime-qnn), an EP plugin library bundled
// with ARM64 builds next to the executable.
//
// Registration is by absolute path from the bundle directory, through the
// Runtime (which owns it and unregisters it after every session is released).
// Registering loads onnxruntime_providers_qnn.dll into the process, so the
// bundle-module check runs again right after it.
#pragma once

#include <filesystem>
#include <string_view>

#include "edgebench/ep_library.hpp"
#include "edgebench/npu_selection.hpp"
#include "edgebench/result.hpp"
#include "edgebench/runtime.hpp"

namespace edgebench {

inline constexpr std::wstring_view kQnnEpLibraryFile = L"onnxruntime_providers_qnn.dll";

/// True for builds that bundle the QNN execution provider (native ARM64); x64
/// builds are CPU-only.
constexpr bool qnn_ep_in_build() noexcept {
#if defined(_M_ARM64) && !defined(_M_ARM64EC)
    return true;
#else
    return false;
#endif
}

/// <bundle_dir>/onnxruntime_providers_qnn.dll.
std::filesystem::path qnn_ep_library_path(const std::filesystem::path& bundle_dir);

/// Registers the bundled QNN EP library with the runtime under kQnnEpName, then
/// verifies that every loaded ONNX Runtime / QNN module is inside the bundle.
/// Already registered: returns the registration (and checks the modules again).
/// Errors, all exit code 4: qnn_ep_not_bundled (the DLL is not in the bundle
/// directory), ep_registration_failed (ONNX Runtime refused it), and
/// module_outside_bundle (the library is unregistered again before returning).
Result<EpLibrary*> register_qnn_ep(Runtime& runtime);

/// One-line advice for the errors of register_qnn_ep() and select_qnn_npu()
/// (empty for other kinds).
std::string_view qnn_ep_hint(std::string_view kind);

}  // namespace edgebench
