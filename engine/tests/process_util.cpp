#include "process_util.hpp"

#include <atomic>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace fs = std::filesystem;

namespace edgebench::test {

namespace {

// Quotes one argument for CommandLineToArgvW-style parsing.
std::wstring quote(const std::wstring& arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\"") == std::wstring::npos)
        return arg;
    std::wstring out = L"\"";
    size_t backslashes = 0;
    for (wchar_t c : arg) {
        if (c == L'\\') {
            ++backslashes;
            continue;
        }
        if (c == L'"')
            out.append(backslashes * 2 + 1, L'\\');
        else
            out.append(backslashes, L'\\');
        backslashes = 0;
        out += c;
    }
    out.append(backslashes * 2, L'\\');
    return out + L"\"";
}

struct Handle {
    HANDLE h = nullptr;
    ~Handle() {
        if (h && h != INVALID_HANDLE_VALUE)
            CloseHandle(h);
    }
};

// Sets an environment variable for the lifetime of the object.
class ScopedEnv {
public:
    ScopedEnv(const wchar_t* name, const std::wstring& value) : name_(name) {
        const DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
        if (n > 0) {
            old_.resize(n);
            old_.resize(GetEnvironmentVariableW(name, old_.data(), n));
            had_ = true;
        }
        SetEnvironmentVariableW(name, value.c_str());
    }
    ~ScopedEnv() { SetEnvironmentVariableW(name_, had_ ? old_.c_str() : nullptr); }

private:
    const wchar_t* name_;
    std::wstring old_;
    bool had_ = false;
};

}  // namespace

ProcessResult run_process(const fs::path& exe, const std::vector<std::wstring>& args,
                          const fs::path& working_dir, const fs::path& path_prefix,
                          uint32_t timeout_ms) {
    std::wstring cmd = quote(exe.native());
    for (const std::wstring& a : args)
        cmd += L" " + quote(a);

    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    Handle out_read, out_write, err_read, err_write;
    if (!CreatePipe(&out_read.h, &out_write.h, &sa, 0) ||
        !CreatePipe(&err_read.h, &err_write.h, &sa, 0)) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                "CreatePipe");
    }
    SetHandleInformation(out_read.h, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_read.h, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nullptr;
    si.hStdOutput = out_write.h;
    si.hStdError = err_write.h;
    PROCESS_INFORMATION pi{};

    std::optional<ScopedEnv> path_env;
    if (!path_prefix.empty()) {
        const DWORD n = GetEnvironmentVariableW(L"PATH", nullptr, 0);
        std::wstring path(n, L'\0');
        path.resize(GetEnvironmentVariableW(L"PATH", path.data(), n));
        path_env.emplace(L"PATH", path_prefix.native() + L";" + path);
    }
    // Report crashes through the exit code instead of the Windows error dialog.
    const UINT old_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    const BOOL created = CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE,
                                        CREATE_NO_WINDOW, nullptr, working_dir.c_str(), &si, &pi);
    const DWORD create_error = GetLastError();
    SetErrorMode(old_mode);
    path_env.reset();
    if (!created) {
        throw std::system_error(static_cast<int>(create_error), std::system_category(),
                                "CreateProcessW");
    }
    Handle process{pi.hProcess}, thread{pi.hThread};
    CloseHandle(out_write.h);
    out_write.h = nullptr;
    CloseHandle(err_write.h);
    err_write.h = nullptr;

    ProcessResult r;
    auto drain = [](HANDLE h, std::string& into) {
        char buf[4096];
        DWORD got = 0;
        while (ReadFile(h, buf, sizeof(buf), &got, nullptr) && got > 0)
            into.append(buf, got);
    };
    // Read both pipes at once so a child filling one of them cannot block.
    std::thread err_reader(drain, err_read.h, std::ref(r.err));
    drain(out_read.h, r.out);
    err_reader.join();
    if (WaitForSingleObject(process.h, timeout_ms) != WAIT_OBJECT_0) {
        TerminateProcess(process.h, 0xFFFFFFFF);
        r.timed_out = true;
    }
    DWORD code = 0;
    GetExitCodeProcess(process.h, &code);
    r.exit_code = code;
    // Text-mode stdio in the child writes CRLF; compare with plain "\n".
    std::erase(r.out, '\r');
    std::erase(r.err, '\r');
    r.output = r.out + r.err;
    return r;
}

TempDir::TempDir() {
    static std::atomic<int> counter{0};
    const fs::path base = fs::temp_directory_path();
    for (int attempt = 0; attempt < 100; ++attempt) {
        const fs::path p = base / ("edgebench_test_" + std::to_string(GetCurrentProcessId()) + "_" +
                                   std::to_string(counter++));
        if (fs::create_directory(p)) {
            path_ = p;
            return;
        }
    }
    throw std::runtime_error("cannot create a temporary directory");
}

TempDir::~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
}

}  // namespace edgebench::test
