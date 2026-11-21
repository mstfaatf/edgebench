#include "edgebench/model_info.hpp"

#include <system_error>

#include "edgebench/ort_loader.hpp"
#include "edgebench/sha256.hpp"

namespace edgebench {

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
