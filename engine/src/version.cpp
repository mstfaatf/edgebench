#include "edgebench/version.hpp"

#include <onnxruntime_c_api.h>

#include "version_config.hpp"

namespace edgebench {

std::string_view version() noexcept {
    return EDGEBENCH_VERSION_STRING;
}

std::string onnxruntime_version() {
    // Asks the loaded onnxruntime.dll, not the headers, so a mismatched DLL is visible.
    const OrtApiBase* base = OrtGetApiBase();
    const char* v = base ? base->GetVersionString() : nullptr;
    return v ? std::string(v) : std::string();
}

}  // namespace edgebench
