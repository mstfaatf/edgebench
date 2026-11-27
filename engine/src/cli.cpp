#include "edgebench/cli.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <sstream>

#include "edgebench/diagnostics.hpp"
#include "edgebench/npu_options.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/version.hpp"

namespace edgebench {

namespace {

constexpr const char* kDescription =
    "edgebench: CPU vs NPU inference benchmarking for Windows on Snapdragon";

// A CLI11 check from a parse function: empty string if the value is accepted.
template <typename ParseFn>
CLI::Validator choice(ParseFn parse, std::string choices, std::string what) {
    return CLI::Validator(
        [parse, choices, what](std::string& value) -> std::string {
            if (parse(value))
                return {};
            return "unsupported " + what + " '" + value + "'; supported " + what + "s: " + choices;
        },
        "{" + choices + "}");
}

// " (lo to hi)" for an option's help text, from the limits that validate() enforces.
std::string range(int lo, int hi) {
    return " (" + std::to_string(lo) + " to " + std::to_string(hi) + ")";
}

}  // namespace

std::string version_text() {
    std::string text = "edgebench " + std::string(version()) + "\n";
    text += "results schema " + std::string(results_schema_version()) + "\n";
    text += "onnxruntime " + std::string(expected_onnxruntime_version()) +
            " (required; 'edgebench doctor' checks the loaded library)\n";
    return text;
}

ParseResult parse_command_line(const std::vector<std::string>& args) {
    ParseResult result;
    CLI::App app{kDescription, "edgebench"};
    std::string version_flag_text = version_text();
    version_flag_text.pop_back();  // CLI11 adds the final newline
    app.set_version_flag("--version", version_flag_text);
    app.require_subcommand(0, 1);

    // ---------------------------------------------------------------- run
    CLI::App* run = app.add_subcommand(
        "run",
        "Benchmark a model. With --dry-run, validate the options, create the session, run "
        "one inference and print the resolved configuration as JSON.");
    RunConfig defaults;
    std::string model, provider = "cpu", binding = "iobinding", input = "random", graph = "all";
    std::string label, name, format = "summary", output_dir = defaults.output_dir.string();
    std::string htp_mode(kDefaultHtpPerformanceMode), finalization_mode;
    std::string ep_cache = "off", ep_cache_dir;
    int profile_iterations = kDefaultProfileIterations;
    bool no_strict = false;
    std::uint64_t seed = 0;
    double constant_value = 0.0;
    RunConfig& cfg = result.run;

    run->add_option("--model", model, "ONNX model file (.onnx); required")->type_name("PATH");
    run->add_option("--provider", provider, "Execution provider")
        ->type_name("NAME")
        ->check(choice(parse_provider, std::string(supported_providers()), "provider"))
        ->capture_default_str();
    run->add_option("--threads", cfg.threads,
                    "CPU intra-op threads" + range(1, kMaxThreads) + "; always recorded")
        ->type_name("N")
        ->capture_default_str();
    run->add_option("--binding", binding,
                    "run: Session::Run per call; iobinding: IoBinding with buffers allocated once")
        ->type_name("MODE")
        ->check(choice(parse_binding, "run, iobinding", "binding"))
        ->capture_default_str();
    run->add_option("--warmup", cfg.warmup,
                    "Unmeasured iterations before each trial" + range(0, kMaxWarmup))
        ->type_name("N")
        ->capture_default_str();
    run->add_option("--iterations", cfg.iterations,
                    "Measured iterations per trial" + range(1, kMaxIterations))
        ->type_name("N")
        ->capture_default_str();
    run->add_option("--trials", cfg.trials, "Number of trials" + range(1, kMaxTrials))
        ->type_name("N")
        ->capture_default_str();
    run->add_option("--cooldown-ms", cfg.cooldown_ms,
                    "Idle period before every trial in ms, sampling background CPU load" +
                        range(0, kMaxCooldownMs))
        ->type_name("N")
        ->capture_default_str();
    run->add_option("--input", input, "Input data: random or constant")
        ->type_name("KIND")
        ->check(choice(parse_input_kind, "random, constant", "input kind"))
        ->capture_default_str();
    CLI::Option* seed_opt =
        run->add_option("--seed", seed, "RNG seed for --input random [default: 0]")->type_name("N");
    CLI::Option* constant_opt = run->add_option("--constant-value", constant_value,
                                                "Fill value for --input constant [default: 0.0]")
                                    ->type_name("FLOAT");
    run->add_option("--graph-opt", graph, "ONNX Runtime graph optimization level")
        ->type_name("LEVEL")
        ->check(
            choice(parse_graph_opt, "disable, basic, extended, all", "graph optimization level"))
        ->capture_default_str();
    CLI::Option* label_opt =
        run->add_option("--label", label, "Name for the run")->type_name("TEXT");
    run->add_option("--load-sample-ms", cfg.load_sample_ms,
                    "Background-load sample in ms before trial 1 when --cooldown-ms is 0" +
                        range(0, kMaxLoadSampleMs) + "; 0: none")
        ->type_name("N")
        ->capture_default_str();
    run->add_option("--output-dir", output_dir, "Directory for result files")
        ->type_name("PATH")
        ->capture_default_str();
    CLI::Option* name_opt = run->add_option("--name", name,
                                            "Result file name (letters, digits, '-', '_', '.') "
                                            "[default: <model>_<provider>_<binding>_t<threads>]")
                                ->type_name("TEXT");
    run->add_option("--format", format, "What to print to stdout: summary or json")
        ->type_name("FORMAT")
        ->check(choice(parse_output_format, "summary, json", "format"))
        ->capture_default_str();
    run->add_flag("--no-strict", no_strict,
                  "NPU: allow nodes the NPU cannot run to fall back to the CPU (recorded and "
                  "warned); by default the session is refused instead");
    run->add_flag("--npu-fp16", cfg.npu_fp16,
                  "NPU: run an FP32 model with FP16 compute (enable_htp_fp16_precision = 1); the "
                  "NPU has no FP32 compute, so FP32 models require it");
    CLI::Option* htp_opt =
        run->add_option("--htp-performance-mode", htp_mode, "NPU: QNN HTP performance mode")
            ->type_name("MODE")
            ->check(choice([](std::string_view v) { return is_htp_performance_mode(v); },
                           htp_performance_modes_list(), "HTP performance mode"))
            ->capture_default_str();
    CLI::Option* finalization_opt =
        run->add_option("--htp-graph-finalization-mode", finalization_mode,
                        "NPU: QNN HTP graph finalization optimization mode "
                        "(htp_graph_finalization_optimization_mode), passed only when given "
                        "[default: not passed, the EP's default]")
            ->type_name("MODE")
            ->check(choice([](std::string_view v) { return is_htp_graph_finalization_mode(v); },
                           htp_graph_finalization_modes_list(), "HTP graph finalization mode"));
    run->add_option("--ep-cache", ep_cache,
                    "NPU: EP context cache. off: compile every session from the model; build: "
                    "rebuild the cache once, then create every timed session from it; use: create "
                    "the timed sessions from the cache whose key matches (build it if missing)")
        ->type_name("MODE")
        ->check(choice(parse_ep_cache_mode, "off, build, use", "EP cache mode"))
        ->capture_default_str();
    CLI::Option* cache_dir_opt =
        run->add_option("--ep-cache-dir", ep_cache_dir,
                        "NPU: EP context cache directory, one subdirectory per cache key "
                        "[default: %LOCALAPPDATA%\\EdgeBench\\qnn-context]")
            ->type_name("PATH");
    run->add_flag("--init-breakdown", cfg.init_breakdown,
                  "NPU: per trial, also time a cold, a context-building and a cached session "
                  "creation (each after the same idle period) before the timed session");
    run->add_flag("--profile-accelerator", cfg.profile_accelerator,
                  "NPU: after the trials, run a separate session with QNN profiling and record "
                  "the accelerator time per inference; never part of the timed results");
    CLI::Option* profile_iterations_opt =
        run->add_option("--profile-iterations", profile_iterations,
                        "NPU: inferences of the --profile-accelerator pass (after --warmup "
                        "unrecorded ones)" +
                            range(1, kMaxIterations))
            ->type_name("N")
            ->capture_default_str();
    run->add_flag("--dry-run", cfg.dry_run,
                  "Validate, create the session, run one inference, print the resolved "
                  "configuration as JSON");
    // Hidden: set by `edgebench suite` for the runs it launches (results: the suite block).
    SuiteLink link;
    CLI::Option* suite_name_opt = run->add_option("--suite-name", link.suite_name)->group("");
    CLI::Option* suite_hash_opt =
        run->add_option("--suite-config-hash", link.config_hash)->group("");
    CLI::Option* suite_run_opt = run->add_option("--suite-run-id", link.run_id)->group("");
    CLI::Option* suite_rep_opt = run->add_option("--suite-repetition", link.repetition)->group("");
    CLI::Option* suite_order_opt =
        run->add_option("--suite-order-index", link.order_index)->group("");
    CLI::Option* suite_cooldown_opt =
        run->add_option("--suite-cooldown-s-before", link.cooldown_s_before)->group("");
    CLI::Option* suite_skipped_opt =
        run->add_flag("--suite-preflight-skipped", link.preflight_skipped)->group("");

    // ---------------------------------------------------------------- doctor
    CLI::App* doctor = app.add_subcommand(
        "doctor",
        "Check everything a run depends on: the ONNX Runtime and QNN bundle, the backend, "
        "the devices, the NPU (probe sessions), the EP context cache, power and load.");
    doctor->add_flag("--json", result.doctor.json, "Print the report as JSON");
    doctor
        ->add_option("--load-sample-ms", result.doctor.load_sample_ms,
                     "Background CPU load sample length in ms (0: no sample)")
        ->type_name("N")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();
    std::string doctor_cache_dir;
    CLI::Option* doctor_cache_opt =
        doctor
            ->add_option("--ep-cache-dir", doctor_cache_dir,
                         "EP context cache directory to check "
                         "[default: %LOCALAPPDATA%\\EdgeBench\\qnn-context]")
            ->type_name("PATH");
    doctor->add_flag("--quick", result.doctor.quick,
                     "Skip the checks that use the NPU (backend versions, probe sessions, EP "
                     "context): about a second instead of several");
    doctor->add_flag("--prune-cache", result.doctor.prune_cache,
                     "Delete stale EP context cache entries (only files their manifest lists); "
                     "without it doctor deletes nothing");

    // ---------------------------------------------------------------- suite
    CLI::App* suite = app.add_subcommand(
        "suite",
        "Run an experiment plan from a configuration file (docs/suite.md): every run in its "
        "own `edgebench run` process, in a rotated order with cooldowns, after a pre-flight "
        "check; resumable after an interruption.");
    std::string suite_config, suite_resume, suite_output, suite_python;
    CLI::Option* suite_config_opt =
        suite->add_option("config", suite_config, "Suite configuration file (.json)")
            ->type_name("CONFIG");
    CLI::Option* suite_resume_opt =
        suite
            ->add_option("--resume", suite_resume,
                         "Continue the suite in DIR: keep its finished runs and run the rest "
                         "(CONFIG defaults to the configuration file the suite was started with)")
            ->type_name("DIR");
    CLI::Option* suite_output_opt =
        suite
            ->add_option("--output-dir", suite_output,
                         "Create the suite directory here instead of the configuration's "
                         "output_dir")
            ->type_name("PATH");
    CLI::Option* suite_python_opt =
        suite
            ->add_option("--python", suite_python,
                         "Python interpreter that runs scripts/validate_results.py on every "
                         "result [default: .venv\\Scripts\\python.exe of the repository]")
            ->type_name("PATH");
    suite->add_flag("--dry-run", result.suite.dry_run,
                    "Print the expanded, ordered run plan and the estimated duration; run nothing");
    suite->add_flag("--fail-fast", result.suite.fail_fast,
                    "Stop at the first failed run (the rest are marked skipped); by default "
                    "failures are recorded and the suite continues");
    suite->add_flag("--skip-preflight", result.suite.skip_preflight,
                    "Development only: skip the pre-flight check. The suite and every result "
                    "from it are marked as not publishable");

    std::vector<std::string> reversed(args.rbegin(), args.rend());  // CLI11 consumes from the back
    try {
        app.parse(reversed);
    } catch (const CLI::ParseError& e) {
        if (e.get_exit_code() == 0) {  // --help or --version
            std::ostringstream out, err;
            app.exit(e, out, err);
            result.out = out.str();
            result.err = err.str();
            result.exit = ExitCode::success;
            return result;
        }
        std::string message = e.what();
        const std::string app_prefix = "edgebench: ";  // CLI11 prefixes some messages
        if (message.rfind(app_prefix, 0) == 0)
            message.erase(0, app_prefix.size());
        const std::vector<CLI::App*> selected = app.get_subcommands();
        if (selected.empty()) {  // no or unknown command: show the commands
            result.err =
                format_error("usage_error", message, "Run 'edgebench --help' for the commands.") +
                "\n" + app.help();
        } else {
            result.err = format_error(
                "usage_error", message,
                "Run 'edgebench " + selected.front()->get_name() + " --help' for the options.");
        }
        result.exit = ExitCode::usage_error;
        return result;
    }

    if (run->parsed()) {
        result.command = Command::run;
        cfg.model = std::filesystem::path(std::u8string(model.begin(), model.end()));
        cfg.provider = *parse_provider(provider);
        cfg.binding = *parse_binding(binding);
        cfg.input = *parse_input_kind(input);
        cfg.graph_opt = *parse_graph_opt(graph);
        if (seed_opt->count() > 0)
            cfg.seed = seed;
        if (constant_opt->count() > 0)
            cfg.constant_value = constant_value;
        if (label_opt->count() > 0)
            cfg.label = label;
        if (name_opt->count() > 0)
            cfg.name = name;
        cfg.format = *parse_output_format(format);
        cfg.strict = !no_strict;
        if (htp_opt->count() > 0)
            cfg.htp_performance_mode = htp_mode;
        if (finalization_opt->count() > 0)
            cfg.htp_graph_finalization_mode = finalization_mode;
        cfg.ep_cache = *parse_ep_cache_mode(ep_cache);
        if (cache_dir_opt->count() > 0)
            cfg.ep_cache_dir =
                std::filesystem::path(std::u8string(ep_cache_dir.begin(), ep_cache_dir.end()));
        cfg.output_dir = std::filesystem::path(std::u8string(output_dir.begin(), output_dir.end()));
        if (profile_iterations_opt->count() > 0)
            cfg.profile_iterations = profile_iterations;
        const std::size_t suite_given = suite_name_opt->count() + suite_hash_opt->count() +
                                        suite_run_opt->count() + suite_rep_opt->count() +
                                        suite_order_opt->count() + suite_cooldown_opt->count();
        if (suite_given > 0 && suite_given != 6) {
            result.err = format_error(
                "usage_error",
                "--suite-name, --suite-config-hash, --suite-run-id, --suite-repetition, "
                "--suite-order-index and --suite-cooldown-s-before go together",
                "They are set by 'edgebench suite'.");
            result.exit = ExitCode::usage_error;
            return result;
        }
        if (suite_given == 0 && suite_skipped_opt->count() > 0) {
            result.err = format_error("usage_error",
                                      "--suite-preflight-skipped needs the other --suite options",
                                      "They are set by 'edgebench suite'.");
            result.exit = ExitCode::usage_error;
            return result;
        }
        if (suite_given > 0)
            cfg.suite = link;

        const std::vector<ConfigError> errors = validate(cfg);
        if (!errors.empty()) {
            for (std::size_t i = 0; i < errors.size(); ++i) {
                const ConfigError& e = errors[i];
                // One hint, after the last error.
                result.err += format_error(
                    e.kind, "--" + e.field + ": " + e.message,
                    i + 1 == errors.size() ? "Run 'edgebench run --help' for the options." : "");
            }
            result.exit = ExitCode::usage_error;
        }
        return result;
    }
    if (doctor->parsed()) {
        result.command = Command::doctor;
        if (doctor_cache_opt->count() > 0) {
            if (doctor_cache_dir.empty()) {
                result.err = format_error("usage_error", "--ep-cache-dir: must not be empty",
                                          "Run 'edgebench doctor --help' for the options.");
                result.exit = ExitCode::usage_error;
                return result;
            }
            result.doctor.ep_cache_dir = std::filesystem::path(
                std::u8string(doctor_cache_dir.begin(), doctor_cache_dir.end()));
        }
        return result;
    }
    if (suite->parsed()) {
        result.command = Command::suite;
        auto path = [](const std::string& s) {
            return std::filesystem::path(std::u8string(s.begin(), s.end()));
        };
        SuiteOptions& o = result.suite;
        if (suite_config_opt->count() > 0)
            o.config = path(suite_config);
        if (suite_resume_opt->count() > 0)
            o.resume = path(suite_resume);
        if (suite_output_opt->count() > 0)
            o.output_dir = path(suite_output);
        if (suite_python_opt->count() > 0)
            o.python = path(suite_python);
        std::string problem;
        if (!o.config && !o.resume)
            problem = "give a suite configuration file, or --resume DIR";
        else if (o.resume && o.dry_run)
            problem = "--dry-run cannot be combined with --resume";
        else if (o.resume && o.output_dir)
            problem = "--output-dir cannot be combined with --resume (the suite stays in DIR)";
        else if ((o.config && o.config->empty()) || (o.resume && o.resume->empty()) ||
                 (o.output_dir && o.output_dir->empty()) || (o.python && o.python->empty()))
            problem = "paths must not be empty";
        if (!problem.empty()) {
            result.err = format_error("usage_error", problem,
                                      "Run 'edgebench suite --help' for the options.");
            result.exit = ExitCode::usage_error;
        }
        return result;
    }
    result.err = format_error("usage_error", "no command given",
                              "Run 'edgebench --help' for the commands.") +
                 "\n" + app.help();
    result.exit = ExitCode::usage_error;
    return result;
}

}  // namespace edgebench
