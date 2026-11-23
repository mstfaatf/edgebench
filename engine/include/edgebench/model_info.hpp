// Facts about a model file (results schema: model.file_name, size_bytes, sha256).
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "edgebench/result.hpp"

namespace edgebench {

struct ModelFileInfo {
    std::filesystem::path path;  // absolute
    std::string file_name;       // UTF-8, no directory
    std::uint64_t size_bytes = 0;
    std::string sha256;  // 64 lowercase hex characters
};

/// Checks that `path` can be opened for reading (it may be locked by another
/// process or not readable for this user). usage_error, kind "model_unreadable",
/// with the operating system's message otherwise.
Result<void> check_file_readable(const std::filesystem::path& path);

/// Size and SHA-256 of `path`. usage_error if the file cannot be read.
Result<ModelFileInfo> read_model_file_info(const std::filesystem::path& path);

}  // namespace edgebench
