#include "edgebench/exit_codes.hpp"

namespace edgebench {

std::string_view name(ExitCode code) noexcept {
    switch (code) {
        case ExitCode::success:
            return "success";
        case ExitCode::internal_error:
            return "internal_error";
        case ExitCode::usage_error:
            return "usage_error";
        case ExitCode::ort_runtime_error:
            return "ort_runtime_error";
        case ExitCode::ort_load_failure:
            return "ort_load_failure";
        case ExitCode::strict_mode_refusal:
            return "strict_mode_refusal";
        case ExitCode::output_error:
            return "output_error";
        case ExitCode::interrupted:
            return "interrupted";
    }
    return "unknown";
}

std::string_view describe(ExitCode code) noexcept {
    switch (code) {
        case ExitCode::success:
            return "Success";
        case ExitCode::internal_error:
            return "Internal or unexpected error";
        case ExitCode::usage_error:
            return "Usage error: bad arguments, invalid values, or model file not found";
        case ExitCode::ort_runtime_error:
            return "ONNX Runtime error: no usable EP device, or session creation or inference "
                   "failed";
        case ExitCode::ort_load_failure:
            return "ONNX Runtime, the QNN execution provider or its HTP backend could not be "
                   "loaded, or a loaded ONNX Runtime/QNN DLL is not the bundled one";
        case ExitCode::strict_mode_refusal:
            return "Strict mode refused the session: not every node can run on the requested "
                   "device";
        case ExitCode::output_error:
            return "Results could not be written";
        case ExitCode::interrupted:
            return "Interrupted by Ctrl+C, Ctrl+Break or closing the console";
    }
    return "Unknown exit code";
}

}  // namespace edgebench
