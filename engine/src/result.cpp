#include "edgebench/result.hpp"

#include <onnxruntime_c_api.h>

namespace edgebench {

const char* ort_error_code_name(int code) noexcept {
    switch (static_cast<OrtErrorCode>(code)) {
        case ORT_OK:
            return "ORT_OK";
        case ORT_FAIL:
            return "ORT_FAIL";
        case ORT_INVALID_ARGUMENT:
            return "ORT_INVALID_ARGUMENT";
        case ORT_NO_SUCHFILE:
            return "ORT_NO_SUCHFILE";
        case ORT_NO_MODEL:
            return "ORT_NO_MODEL";
        case ORT_ENGINE_ERROR:
            return "ORT_ENGINE_ERROR";
        case ORT_RUNTIME_EXCEPTION:
            return "ORT_RUNTIME_EXCEPTION";
        case ORT_INVALID_PROTOBUF:
            return "ORT_INVALID_PROTOBUF";
        case ORT_MODEL_LOADED:
            return "ORT_MODEL_LOADED";
        case ORT_NOT_IMPLEMENTED:
            return "ORT_NOT_IMPLEMENTED";
        case ORT_INVALID_GRAPH:
            return "ORT_INVALID_GRAPH";
        case ORT_EP_FAIL:
            return "ORT_EP_FAIL";
        default:
            return "ORT_UNKNOWN_ERROR_CODE";
    }
}

Error ort_error(std::string kind, const std::string& context, int ort_code,
                const std::string& ort_message) {
    Error e;
    e.exit_code = ExitCode::ort_runtime_error;
    e.kind = std::move(kind);
    e.ort_error_code = ort_code;
    e.ort_error_name = ort_error_code_name(ort_code);
    e.message = context + ": ONNX Runtime error " + *e.ort_error_name + " (code " +
                std::to_string(ort_code) + "): " + ort_message;
    return e;
}

}  // namespace edgebench
