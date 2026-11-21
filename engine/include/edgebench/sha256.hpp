// SHA-256 digests (model files, cache keys), as 64 lowercase hex characters.
#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

#include "edgebench/result.hpp"

namespace edgebench {

std::string sha256_hex(std::span<const std::byte> data);
std::string sha256_hex(std::string_view text);

/// Digest of a file's contents, read in chunks. Error kind "file_read_failed"
/// (exit code usage_error) if the file cannot be read.
Result<std::string> sha256_file(const std::filesystem::path& path);

}  // namespace edgebench
