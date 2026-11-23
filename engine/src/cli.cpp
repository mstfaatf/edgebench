#include "edgebench/cli.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <sstream>

#include "edgebench/diagnostics.hpp"
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
    run->add_flag("--dry-run", cfg.dry_run,
                  "Validate, create the session, run one inference, print the resolved "
                  "configuration as JSON");

    // ---------------------------------------------------------------- doctor
    CLI::App* doctor = app.add_subcommand(
        "doctor", "Check the build, the system and the ONNX Runtime installation.");
    doctor->add_flag("--json", result.doctor.json, "Print the report as JSON");
    doctor
        ->add_option("--load-sample-ms", result.doctor.load_sample_ms,
                     "Background CPU load sample length in ms (0: no sample)")
        ->type_name("N")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();

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
        cfg.output_dir = std::filesystem::path(std::u8string(output_dir.begin(), output_dir.end()));

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
        return result;
    }
    result.err = format_error("usage_error", "no command given",
                              "Run 'edgebench --help' for the commands.") +
                 "\n" + app.help();
    result.exit = ExitCode::usage_error;
    return result;
}

}  // namespace edgebench
