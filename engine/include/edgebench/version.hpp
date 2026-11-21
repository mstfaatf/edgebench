#pragma once

#include <string_view>

namespace edgebench {

/// EdgeBench version, e.g. "0.1.0". Set from the CMake project version.
std::string_view version() noexcept;

}  // namespace edgebench
