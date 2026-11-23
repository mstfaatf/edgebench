// edgebench: CPU vs NPU inference benchmarking for Windows on Snapdragon.
//
//   edgebench run --model <file.onnx> [options]   benchmark a model (--dry-run: check only)
//   edgebench doctor [--json]                     check the build, system and ONNX Runtime
//
// ONNX Runtime is delay-loaded and initialized manually, and is only loaded by
// the commands that need it (never for --help or --version), by full path from
// the executable's directory; see edgebench/ort_loader.hpp. Errors and warnings
// go through edgebench/diagnostics.hpp. Exit codes are listed in
// edgebench/exit_codes.hpp and docs/cli.md.

#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <exception>
#include <string>
#include <system_error>
#include <vector>

#include "edgebench/benchmark.hpp"
#include "edgebench/cli.hpp"
#include "edgebench/crash_handler.hpp"
#include "edgebench/diagnostics.hpp"
#include "edgebench/doctor.hpp"
#include "edgebench/dry_run.hpp"
#include "edgebench/exit_codes.hpp"
#include "edgebench/interrupt.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/result_files.hpp"
#include "edgebench/result_json.hpp"
#include "edgebench/run_config.hpp"
#include "edgebench/runtime.hpp"
#include "edgebench/version.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace eb = edgebench;
namespace fs = std::filesystem;

namespace {

constexpr const char* kRunHelpHint = "Run 'edgebench run --help' for the options.";

std::string to_utf8(const wchar_t* w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1)
        return {};
    std::string out(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), n, nullptr, nullptr);
    return out;
}

// A hint for an error, or empty.
std::string hint_for(const eb::Error& e) {
    switch (e.exit_code) {
        case eb::ExitCode::usage_error:
            return kRunHelpHint;
        case eb::ExitCode::output_error:
            return "Choose another --output-dir, or check that it is writable.";
        default:
            return {};
    }
}

int fail(const eb::Error& e, const std::string& hint) {
    eb::print_error(e, hint);
    return eb::to_int(e.exit_code);
}

int fail(const eb::Error& e) {
    return fail(e, hint_for(e));
}

eb::Error load_error(const eb::OrtLoadResult& r) {
    return eb::Error{eb::ExitCode::ort_load_failure, std::string(eb::to_string(r.error_kind)),
                     "cannot load ONNX Runtime: " + r.message};
}

int load_failure(const eb::OrtLoadResult& r) {
    return fail(load_error(r), std::string(eb::load_error_hint(r.error_kind)));
}

// Prints a finished run (summary or JSON), writes its result files, and returns
// the exit code: the run's own failure code if it failed, otherwise 0, or 6 if
// the results could not be written. Usage errors write nothing.
int report_result(const eb::BenchmarkResult& r) {
    const eb::RunConfig& config = r.config;
    const int run_code = r.failure ? eb::to_int(r.failure->error.exit_code) : 0;
    if (r.failure && r.failure->error.exit_code == eb::ExitCode::usage_error)
        return fail(r.failure->error);

    for (const eb::ResultWarning& w : r.warnings) {
        if (w.code != "value_unavailable")
            eb::print_warning(w.code, w.message);
    }
    if (r.failure) {
        const eb::Error& e = r.failure->error;
        if (e.exit_code == eb::ExitCode::ort_load_failure && r.failure->stage == "environment") {
            eb::print_error(e.kind, "cannot load ONNX Runtime: " + e.message,
                            r.runtime.onnxruntime.ok()
                                ? std::string_view()
                                : eb::load_error_hint(r.runtime.onnxruntime.error_kind));
        } else {
            eb::print_error(e, hint_for(e));
        }
    } else if (config.format == eb::OutputFormat::summary) {
        // Printed before writing, so the measurements are never lost to a write error.
        std::fputs(eb::format_summary(r).c_str(), stdout);
    }
    std::fflush(stdout);

    const eb::Result<eb::WrittenResult> written =
        eb::write_result_files(r, config.output_dir, eb::result_name(config));
    if (config.format == eb::OutputFormat::json) {
        const nlohmann::ordered_json doc =
            written ? written.value().document : eb::to_result_json(r, std::nullopt);
        std::printf("%s\n", doc.dump(2).c_str());
        std::fflush(stdout);
    }
    if (!written) {
        eb::print_error(written.error().kind, "cannot write the result: " + written.error().message,
                        hint_for(written.error()));
        eb::notify_shutdown_complete();
        return r.failure ? run_code : eb::to_int(eb::ExitCode::output_error);
    }
    const std::string path = eb::path_to_utf8(written.value().result_file);
    if (!r.failure && config.format == eb::OutputFormat::summary)
        std::printf("Results     %s\n", path.c_str());
    else
        std::fprintf(stderr, "%s result written to %s\n", r.failure ? "failed" : "the",
                     path.c_str());
    std::fflush(stdout);
    std::fflush(stderr);
    eb::notify_shutdown_complete();
    return run_code;
}

int benchmark_command(eb::Runtime& runtime, const eb::RunConfig& config) {
    return report_result(eb::run_benchmark(runtime, config, [](std::string_view message) {
        std::fprintf(stderr, "%.*s\n", static_cast<int>(message.size()), message.data());
        std::fflush(stderr);
    }));
}

int run_command(const eb::RunConfig& config) {
    // An existing file where the output directory should be: nothing can be written,
    // so do not measure anything.
    std::error_code ec;
    if (!config.dry_run && fs::exists(config.output_dir, ec) &&
        !fs::is_directory(config.output_dir, ec)) {
        return fail(eb::Error{eb::ExitCode::output_error, "output_dir_not_a_directory",
                              "--output-dir " + eb::path_to_utf8(fs::absolute(config.output_dir)) +
                                  " is an existing file, not a directory"});
    }
    // A benchmark stops gracefully on Ctrl+C and writes a partial result; a dry
    // run has nothing to save.
    eb::install_interrupt_handler(config.dry_run ? eb::InterruptMode::exit_immediately
                                                 : eb::InterruptMode::stop_gracefully);

    const eb::OrtLoadResult ort = eb::load_onnxruntime();
    if (!ort.ok()) {
        if (config.dry_run)
            return load_failure(ort);
        eb::BenchmarkResult r = eb::failed_load_result(config, ort);
        r.runtime.onnxruntime = ort;  // for the hint; runtime stays unknown in the result
        return report_result(r);
    }
    // Declared first, destroyed last: every session is released before the Runtime.
    eb::Result<std::unique_ptr<eb::Runtime>> runtime = eb::Runtime::create(ort);
    if (!runtime) {
        if (config.dry_run)
            return fail(runtime.error());
        eb::BenchmarkResult r = eb::begin_result(config);
        if (!r.failure)
            eb::set_failure(r, runtime.error(), "environment");
        eb::finish_result(r);
        return report_result(r);
    }
    if (!config.dry_run)
        return benchmark_command(*runtime.value(), config);
    const eb::Result<nlohmann::ordered_json> resolved = eb::dry_run(*runtime.value(), config);
    if (!resolved)
        return fail(resolved.error());
    for (const auto& w : resolved.value().at("warnings"))
        eb::print_warning(w.at("code").get<std::string>(), w.at("message").get<std::string>());
    std::printf("%s\n", resolved.value().dump(2).c_str());
    return eb::to_int(eb::ExitCode::success);
}

int doctor_command(const eb::DoctorOptions& options) {
    eb::install_interrupt_handler(eb::InterruptMode::exit_immediately);
    const eb::DoctorReport report =
        eb::run_doctor(std::chrono::milliseconds(options.load_sample_ms));
    if (options.json)
        std::printf("%s\n", eb::to_json(report).dump(2).c_str());
    else
        std::fputs(eb::format_text(report).c_str(), stdout);
    std::fflush(stdout);
    for (const eb::DoctorWarning& w : report.warnings)
        eb::print_warning(w.code, w.message, w.hint);
    for (const eb::DoctorProblem& p : report.problems)
        eb::print_error(p.code, p.message, p.hint);
    return eb::to_int(report.exit_code);
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    eb::install_crash_handler();
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
        return fail(eb::Error{eb::ExitCode::internal_error, "internal_error", "no command to run"});
    } catch (const Ort::Exception& e) {
        return fail(eb::ort_error("ort_runtime_error", "unexpected ONNX Runtime exception",
                                  e.GetOrtErrorCode(), e.what()));
    } catch (const std::exception& e) {
        return fail(eb::Error{eb::ExitCode::internal_error, "internal_error",
                              std::string("unexpected exception: ") + e.what()});
    }
}
