// Test helpers: run a child process and capture its output, and temporary directories.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
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

/// A child process started in its own process group on this process's console,
/// so console control events can be sent to it. Output is collected while it runs.
class ChildProcess {
public:
    ChildProcess(const std::filesystem::path& exe, const std::vector<std::wstring>& args,
                 const std::filesystem::path& working_dir);
    ~ChildProcess();
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;

    /// True if this process has a console to share (otherwise events cannot be sent).
    static bool console_available();

    /// Waits until stderr contains `text`; false on timeout or if the child exited.
    bool wait_for_stderr(const std::string& text, uint32_t timeout_ms = 30000);
    /// Sends CTRL_BREAK_EVENT to the child's process group.
    bool send_ctrl_break();
    /// Waits for the child to exit and returns its exit code and output.
    ProcessResult wait(uint32_t timeout_ms = 60000);

private:
    struct State;
    std::unique_ptr<State> state_;
};

/// A new, empty directory under the system temp directory, deleted on destruction.
class TempDir {
public:
    TempDir();
    /// Under the system temp directory, named `name` plus a unique suffix.
    explicit TempDir(const std::wstring& name);
    ~TempDir();
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace edgebench::test
