// Process exit codes shared by edgebench and ort_probe (see docs/cli.md).
#pragma once

#include <array>
#include <string_view>

namespace edgebench {

enum class ExitCode : int {
    success = 0,
    /// Internal or unexpected error (also: a check that ort_probe runs did not pass).
    internal_error = 1,
    /// Bad arguments, invalid values, or a model file that does not exist.
    usage_error = 2,
    /// ONNX Runtime failed at run time: session creation or inference.
    ort_runtime_error = 3,
    /// ONNX Runtime could not be loaded, or a loaded ORT/QNN DLL is not the bundled one.
    ort_load_failure = 4,
    /// Strict mode refused the session: not every node can run on the requested device.
    strict_mode_refusal = 5,
    /// Results could not be written.
    output_error = 6,
    /// Stopped by Ctrl+C, Ctrl+Break or closing the console (128 + SIGINT, as in shells).
    interrupted = 130,
};

inline constexpr std::array<ExitCode, 8> kAllExitCodes = {
    ExitCode::success,           ExitCode::internal_error,   ExitCode::usage_error,
    ExitCode::ort_runtime_error, ExitCode::ort_load_failure, ExitCode::strict_mode_refusal,
    ExitCode::output_error,      ExitCode::interrupted,
};

constexpr int to_int(ExitCode code) noexcept {
    return static_cast<int>(code);
}

/// Stable identifier, e.g. "ort_load_failure".
std::string_view name(ExitCode code) noexcept;

/// One-line description, as in docs/cli.md.
std::string_view describe(ExitCode code) noexcept;

}  // namespace edgebench
