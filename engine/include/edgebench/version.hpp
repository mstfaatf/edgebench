#pragma once

#include <string_view>

namespace edgebench {

/// EdgeBench version, e.g. "0.1.0". Set from the CMake project version.
std::string_view version() noexcept;

/// Results schema version that result documents are written with, e.g. "1.1.0".
/// The schema is schemas/benchmark-result.v<version>.schema.json.
std::string_view results_schema_version() noexcept;

}  // namespace edgebench
