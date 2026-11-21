// Platform-independent helpers for checking which ONNX Runtime / QNN DLLs a
// process loaded: Windows-style path normalization, directory containment,
// module-name filtering, and version strings. No file system or OS calls, so
// these are unit-tested on any platform.
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace edgebench {

/// Lexically normalizes a Windows path for comparison: '/' becomes '\', the
/// "\\?\" and "\\?\UNC\" prefixes are removed, repeated separators are
/// collapsed, "." and ".." segments are resolved, trailing separators are
/// dropped, and ASCII letters are lower-cased. Does not touch the file system:
/// short (8.3) names and links must be resolved beforehand (see resolve_path()).
std::wstring normalize_path(std::wstring_view path);

/// True if `path` is `dir` or lies anywhere below it, after normalize_path() on both.
bool is_path_within(std::wstring_view path, std::wstring_view dir);

/// True if both paths are equal after normalize_path().
bool is_same_path(std::wstring_view a, std::wstring_view b);

/// True for the DLL file names of ONNX Runtime and the QNN execution provider:
/// onnxruntime*.dll (including onnxruntime_providers_*.dll and the QNN EP plugin
/// onnxruntime_providers_qnn.dll) and Qnn*.dll (QNN backends). Case-insensitive;
/// `file_name` is a bare file name, not a path.
bool is_ort_or_qnn_module(std::wstring_view file_name);

/// Major.minor.patch of a version string.
struct Version {
    int major = 0;
    int minor = 0;
    int patch = 0;
    friend bool operator==(const Version&, const Version&) = default;
};

/// Parses exactly "<major>.<minor>.<patch>" (decimal, no sign, no suffix).
std::optional<Version> parse_version(std::string_view text);

/// True if both strings parse and name the same major.minor.patch.
bool version_matches(std::string_view found, std::string_view expected);

}  // namespace edgebench
