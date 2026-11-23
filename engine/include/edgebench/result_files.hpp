// Writing result files without ever overwriting one (docs/cli.md, "Result files").
//
//   <dir>/<name>_<stamp>.json            the result document
//   <dir>/<name>_<stamp>/latencies.csv   one row per measured iteration
//   <dir>/<name>_<stamp>/summary.csv     one row per trial plus the aggregate
//   <dir>/<name>_latest.json             copy of the newest result for <name>
//
// <stamp> is file_stamp() of the run's start. If a file or folder with that stamp
// already exists, "-2", "-3", ... is appended to the stamp. Every file is written
// to a temporary name in its final directory and renamed into place, so a crash
// never leaves a partly written file under a final name. Only <name>_latest.json
// is ever replaced, atomically.
#pragma once

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "edgebench/benchmark.hpp"
#include "edgebench/result.hpp"

namespace edgebench {

/// Creates `path` with `content`: written to a temporary file in the same
/// directory, then renamed without replacing. Fails (output_error, kind
/// "file_exists") if `path` already exists, and never modifies an existing file.
Result<void> create_file_atomically(const std::filesystem::path& path, std::string_view content);

/// Replaces (or creates) `path` with `content` atomically: temporary file, then
/// MoveFileExW with MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH.
Result<void> replace_file_atomically(const std::filesystem::path& path, std::string_view content);

struct WrittenResult {
    std::filesystem::path result_file;             // <name>_<stamp>.json
    std::filesystem::path latest_file;             // <name>_latest.json
    std::optional<std::filesystem::path> run_dir;  // only when there are latencies
    std::string stamp;                             // including a collision suffix, if any
    nlohmann::ordered_json document;               // what was written
};

/// Writes `result` into `output_dir` (created if missing) under `name`. The
/// latency and summary CSVs are written only if at least one trial completed.
/// Errors are output_error naming the path and the OS error.
/// `stamp_time` defaults to result.started_at.
Result<WrittenResult> write_result_files(
    const BenchmarkResult& result, const std::filesystem::path& output_dir, const std::string& name,
    std::optional<std::chrono::system_clock::time_point> stamp_time = std::nullopt);

}  // namespace edgebench
