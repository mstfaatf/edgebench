// edgebench: CPU vs NPU inference benchmarking for Windows on Snapdragon.
//
//   edgebench run --model <file.onnx> [options]   benchmark a model (--dry-run: resolve only)
//   edgebench doctor [--json]                     check the build, system and ONNX Runtime
//
// ONNX Runtime is delay-loaded and initialized manually, and is only loaded by
// the commands that need it (never for --help or --version), by full path from
// the executable's directory; see edgebench/ort_loader.hpp. Exit codes are
// listed in edgebench/exit_codes.hpp and docs/cli.md.

#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "edgebench/cli.hpp"
#include "edgebench/doctor.hpp"
#include "edgebench/exit_codes.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/run_config.hpp"
#include "edgebench/version.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace eb = edgebench;

namespace {

std::string to_utf8(const wchar_t* w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1)
        return {};
    std::string out(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), n, nullptr, nullptr);
    return out;
}

int fail(eb::ExitCode code, const std::string& message) {
    std::fprintf(stderr, "error: %s\n", message.c_str());
    return eb::to_int(code);
}

int load_failure(const eb::OrtLoadResult& r) {
    return fail(eb::ExitCode::ort_load_failure, "cannot load ONNX Runtime (" +
                                                    std::string(eb::to_string(r.error_kind)) +
                                                    ")\n  " + r.message);
}

int run_command(const eb::RunConfig& config) {
    if (!config.dry_run) {
        return fail(eb::ExitCode::internal_error,
                    "benchmark execution is not available in this build");
    }
    const eb::OrtLoadResult ort = eb::load_onnxruntime();
    if (!ort.ok())
        return load_failure(ort);
    nlohmann::ordered_json resolved = {{"command", "run"},
                                       {"dry_run", true},
                                       {"edgebench_version", eb::version()},
                                       {"results_schema_version", eb::results_schema_version()}};
    resolved.update(eb::resolved_config_json(config));
    resolved["runtime"] = {{"onnxruntime",
                            {{"version", ort.ort_version},
                             {"library_path", eb::path_to_utf8(ort.dll_path)},
                             {"api_version", ort.api_version},
                             {"load_check", ort.load_check()}}}};
    std::printf("%s\n", resolved.dump(2).c_str());
    return eb::to_int(eb::ExitCode::success);
}

int doctor_command(const eb::DoctorOptions& options) {
    const eb::DoctorReport report = eb::run_doctor();
    if (options.json)
        std::printf("%s\n", eb::to_json(report).dump(2).c_str());
    else
        std::fputs(eb::format_text(report).c_str(), stdout);
    std::fflush(stdout);
    for (const eb::DoctorProblem& p : report.problems)
        std::fprintf(stderr, "error: [%s] %s\n", p.code.c_str(), p.message.c_str());
    return eb::to_int(report.exit_code);
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
        args.push_back(to_utf8(argv[i]));

    const eb::ParseResult parsed = eb::parse_command_line(args);
    if (parsed.exit) {
        std::fputs(parsed.out.c_str(), stdout);
        std::fputs(parsed.err.c_str(), stderr);
        return eb::to_int(*parsed.exit);
    }
    try {
        switch (parsed.command) {
            case eb::Command::run:
                return run_command(parsed.run);
            case eb::Command::doctor:
                return doctor_command(parsed.doctor);
            case eb::Command::none:
                break;
        }
        return fail(eb::ExitCode::internal_error, "no command to run");
    } catch (const Ort::Exception& e) {
        return fail(eb::ExitCode::ort_runtime_error,
                    std::string("ONNX Runtime error: ") + e.what());
    } catch (const std::exception& e) {
        return fail(eb::ExitCode::internal_error, e.what());
    }
}
