#ifndef _WIN32
#error "result_files.cpp supports Windows only: it uses CreateFileW and MoveFileExW."
#endif

#include "edgebench/result_files.hpp"

#include <algorithm>
#include <atomic>
#include <system_error>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "edgebench/ort_loader.hpp"
#include "edgebench/result_json.hpp"
#include "edgebench/run_identity.hpp"
#include "edgebench/sha256.hpp"

namespace fs = std::filesystem;

namespace edgebench {

namespace {

constexpr int kMaxCollisionSuffix = 1000;

Error os_error(const std::string& what, const fs::path& path, DWORD code,
               std::string kind = "write_failed") {
    return Error{ExitCode::output_error, std::move(kind),
                 what + " " + path_to_utf8(path) + ": " +
                     std::system_category().message(static_cast<int>(code))};
}

bool is_exists_error(DWORD code) {
    return code == ERROR_ALREADY_EXISTS || code == ERROR_FILE_EXISTS;
}

// A new, empty temporary file next to `target`, created exclusively.
Result<fs::path> create_temp_next_to(const fs::path& target, HANDLE& handle) {
    static std::atomic<unsigned> counter{0};
    for (int attempt = 0; attempt < 100; ++attempt) {
        const fs::path temp = target.parent_path() / (L"." + target.filename().native() + L".tmp-" +
                                                      std::to_wstring(GetCurrentProcessId()) +
                                                      L"-" + std::to_wstring(counter++));
        handle = CreateFileW(temp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle != INVALID_HANDLE_VALUE)
            return temp;
        const DWORD err = GetLastError();
        if (!is_exists_error(err))
            return os_error("cannot create a temporary file for", target, err);
    }
    return Error{ExitCode::output_error, "write_failed",
                 "cannot find a free temporary file name for " + path_to_utf8(target)};
}

Result<void> write_all(HANDLE h, const fs::path& temp, std::string_view content) {
    std::size_t done = 0;
    while (done < content.size()) {
        const DWORD chunk =
            static_cast<DWORD>(std::min<std::size_t>(content.size() - done, 1 << 30));
        DWORD written = 0;
        if (!WriteFile(h, content.data() + done, chunk, &written, nullptr) || written == 0)
            return os_error("cannot write", temp, GetLastError());
        done += written;
    }
    if (!FlushFileBuffers(h))
        return os_error("cannot flush", temp, GetLastError());
    return {};
}

Result<void> write_via_temp(const fs::path& path, std::string_view content, DWORD move_flags) {
    HANDLE h = INVALID_HANDLE_VALUE;
    Result<fs::path> temp = create_temp_next_to(path, h);
    if (!temp)
        return temp.error();
    Result<void> written = write_all(h, temp.value(), content);
    CloseHandle(h);
    if (!written) {
        DeleteFileW(temp.value().c_str());
        return written;
    }
    if (!MoveFileExW(temp.value().c_str(), path.c_str(), move_flags)) {
        const DWORD err = GetLastError();
        DeleteFileW(temp.value().c_str());
        if (is_exists_error(err))
            return os_error("refusing to overwrite", path, err, "file_exists");
        return os_error("cannot move the finished file to", path, err);
    }
    return {};
}

bool path_exists(const fs::path& p) {
    return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES;
}

void remove_quietly(const fs::path& p) {
    std::error_code ec;
    fs::remove(p, ec);
}

}  // namespace

Result<void> create_file_atomically(const fs::path& path, std::string_view content) {
    if (path_exists(path))
        return os_error("refusing to overwrite", path, ERROR_FILE_EXISTS, "file_exists");
    return write_via_temp(path, content, MOVEFILE_WRITE_THROUGH);
}

Result<void> replace_file_atomically(const fs::path& path, std::string_view content) {
    return write_via_temp(path, content, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
}

Result<WrittenResult> write_result_files(
    const BenchmarkResult& result, const fs::path& output_dir, const std::string& name,
    std::optional<std::chrono::system_clock::time_point> when) {
    std::error_code ec;
    const fs::path dir = fs::absolute(output_dir, ec);
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir)) {
        return Error{ExitCode::output_error, "write_failed",
                     "cannot create the output directory " + path_to_utf8(dir) + ": " +
                         (ec ? ec.message() : std::string("not a directory"))};
    }
    const bool with_latencies = !result.timing.trials.empty();
    const std::string base_stamp = file_stamp(when.value_or(result.started_at));
    const std::string latencies = with_latencies ? latencies_csv(result) : std::string();
    const std::string summary = with_latencies ? summary_csv(result) : std::string();

    for (int k = 1; k <= kMaxCollisionSuffix; ++k) {
        const std::string stamp = k == 1 ? base_stamp : base_stamp + "-" + std::to_string(k);
        const std::string base = name + "_" + stamp;
        const fs::path json_path = dir / (base + ".json");
        const fs::path run_dir = dir / base;
        if (path_exists(json_path) || path_exists(run_dir))
            continue;

        std::optional<RawLatencyLink> raw;
        if (with_latencies) {
            // The run directory is created exclusively: it reserves the stamp.
            if (!CreateDirectoryW(run_dir.c_str(), nullptr)) {
                const DWORD err = GetLastError();
                if (is_exists_error(err))
                    continue;
                return os_error("cannot create the run directory", run_dir, err);
            }
            auto undo_dir = [&run_dir] {
                remove_quietly(run_dir / "latencies.csv");
                remove_quietly(run_dir / "summary.csv");
                remove_quietly(run_dir);
            };
            if (Result<void> r = create_file_atomically(run_dir / "latencies.csv", latencies); !r) {
                undo_dir();
                return r.error();
            }
            if (Result<void> r = create_file_atomically(run_dir / "summary.csv", summary); !r) {
                undo_dir();
                return r.error();
            }
            raw = RawLatencyLink{base + "/latencies.csv", sha256_hex(std::string_view(latencies))};
        }

        WrittenResult out;
        out.document = to_result_json(result, raw);
        const std::string text = out.document.dump(2) + "\n";
        if (Result<void> r = create_file_atomically(json_path, text); !r) {
            if (with_latencies) {
                remove_quietly(run_dir / "latencies.csv");
                remove_quietly(run_dir / "summary.csv");
                remove_quietly(run_dir);
            }
            if (r.error().kind == "file_exists")
                continue;  // another writer took this stamp meanwhile
            return r.error();
        }
        out.result_file = json_path;
        out.latest_file = dir / (name + "_latest.json");
        out.stamp = stamp;
        if (with_latencies)
            out.run_dir = run_dir;
        if (Result<void> r = replace_file_atomically(out.latest_file, text); !r)
            return r.error();
        return out;
    }
    return Error{ExitCode::output_error, "write_failed",
                 "no free file name for " + name + "_" + base_stamp + " in " + path_to_utf8(dir)};
}

}  // namespace edgebench
