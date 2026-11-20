#pragma once

#include <string>
#include <string_view>

namespace edgebench {

/// EdgeBench version, e.g. "0.1.0". Set from the CMake project version.
std::string_view version() noexcept;

/// Version of the ONNX Runtime library loaded at run time, e.g. "1.32.0".
std::string onnxruntime_version();

}  // namespace edgebench
