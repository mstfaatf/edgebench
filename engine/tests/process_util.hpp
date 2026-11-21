// Test helpers: run a child process and capture its output, and temporary directories.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace edgebench::test {

struct ProcessResult {
    uint32_t exit_code = 0;
    std::string out;     // stdout
    std::string err;     // stderr
    std::string output;  // stdout followed by stderr
    bool timed_out = false;
};

/// Runs `exe` with `args` in `working_dir` and waits up to `timeout_ms`.
/// `path_prefix`, if not empty, is prepended to PATH for the child.
ProcessResult run_process(const std::filesystem::path& exe, const std::vector<std::wstring>& args,
                          const std::filesystem::path& working_dir,
                          const std::filesystem::path& path_prefix = {},
                          uint32_t timeout_ms = 60000);

/// A new, empty directory under the system temp directory, deleted on destruction.
class TempDir {
public:
    TempDir();
    ~TempDir();
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace edgebench::test
