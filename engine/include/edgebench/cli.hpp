// Command-line parsing for edgebench: `edgebench run ...`, `edgebench doctor ...`.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "edgebench/exit_codes.hpp"
#include "edgebench/run_config.hpp"

namespace edgebench {

enum class Command { none, run, doctor };

struct DoctorOptions {
    bool json = false;
    int load_sample_ms = 1000;  // background-load sample length; 0: no sample
    std::optional<std::filesystem::path> ep_cache_dir;  // --ep-cache-dir
    bool prune_cache = false;                           // --prune-cache
    bool quick = false;                                 // --quick
};

/// Outcome of parsing. If `exit` is set, the program prints `out` to stdout and
/// `err` to stderr and exits with that code (help, --version, usage errors).
/// Otherwise `command` is run or doctor and the matching options are filled in;
/// a run configuration has already passed validate().
struct ParseResult {
    Command command = Command::none;
    RunConfig run;
    DoctorOptions doctor;
    std::optional<ExitCode> exit;
    std::string out;
    std::string err;
};

/// Parses the arguments after the program name.
ParseResult parse_command_line(const std::vector<std::string>& args);

/// Text printed by `edgebench --version`.
std::string version_text();

}  // namespace edgebench
