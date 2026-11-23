#include "edgebench/model_info.hpp"

#include <system_error>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "edgebench/ort_loader.hpp"
#include "edgebench/sha256.hpp"

namespace edgebench {

Result<void> check_file_readable(const std::filesystem::path& path) {
    // Share everything: only an exclusive lock or missing permission makes this fail.
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD err = GetLastError();
        return Error{ExitCode::usage_error, "model_unreadable",
                     "cannot open " + path_to_utf8(path) + " for reading: " +
                         std::system_category().message(static_cast<int>(err))};
    }
    CloseHandle(h);
    return {};
}

Result<ModelFileInfo> read_model_file_info(const std::filesystem::path& path) {
    std::error_code ec;
    ModelFileInfo info;
    info.path = std::filesystem::absolute(path, ec);
    if (ec)
        info.path = path;
    info.file_name = path_to_utf8(info.path.filename());
    info.size_bytes = std::filesystem::file_size(info.path, ec);
    if (ec) {
        return Error{ExitCode::usage_error, "file_read_failed",
                     "cannot read the size of " + path_to_utf8(info.path) + ": " + ec.message()};
    }
    Result<std::string> digest = sha256_file(info.path);
    if (!digest)
        return digest.error();
    info.sha256 = std::move(digest).value();
    return info;
}

}  // namespace edgebench
