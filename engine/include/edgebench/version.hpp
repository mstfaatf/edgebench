#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace edgebench {

/// EdgeBench version, e.g. "0.1.0". Set from the CMake project version.
std::string_view version() noexcept;

/// Results schema version that result documents are written with, e.g. "1.1.0".
/// The schema is schemas/benchmark-result.v<version>.schema.json.
std::string_view results_schema_version() noexcept;

/// How this binary was built (results schema: tool.*).
struct BuildInfo {
    std::optional<std::string> git_commit;  // 40 hex digits; empty if not built from git
    std::optional<bool> git_dirty;          // uncommitted changes to tracked files at build time
    std::string build_type;                 // CMake configuration, e.g. "Release"
    std::string compiler;                   // e.g. "MSVC 19.42.34435"
};

const BuildInfo& build_info();

}  // namespace edgebench
