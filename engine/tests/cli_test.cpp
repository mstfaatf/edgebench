#include "edgebench/cli.hpp"

#include <gtest/gtest.h>

#include "edgebench/ort_loader.hpp"
#include "edgebench/version.hpp"
#include "test_files.hpp"

namespace eb = edgebench;
using eb::test::TempModel;

namespace {

eb::ParseResult parse(std::vector<std::string> args) {
    return eb::parse_command_line(args);
}

std::vector<std::string> run_args(const TempModel& model, std::vector<std::string> extra = {}) {
    std::vector<std::string> args = {"run", "--model", eb::path_to_utf8(model.path())};
    args.insert(args.end(), extra.begin(), extra.end());
    return args;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

void expect_usage_error(const eb::ParseResult& r, const std::string& message) {
    ASSERT_TRUE(r.exit.has_value());
    EXPECT_EQ(*r.exit, eb::ExitCode::usage_error);
    EXPECT_TRUE(r.out.empty()) << r.out;
    EXPECT_TRUE(contains(r.err, message)) << r.err;
}

}  // namespace

// --------------------------------------------------------------------------- top level

TEST(Cli, NoCommandPrintsHelpAndFails) {
    const eb::ParseResult r = parse({});
    expect_usage_error(r, "no command given");
    EXPECT_TRUE(contains(r.err, "run")) << r.err;
    EXPECT_TRUE(contains(r.err, "doctor")) << r.err;
}

TEST(Cli, UnknownCommandPrintsHelpAndFails) {
    const eb::ParseResult r = parse({"bench"});
    expect_usage_error(r, "bench");
    EXPECT_TRUE(contains(r.err, "doctor")) << r.err;  // the command list
}

TEST(Cli, UnknownTopLevelOption) {
    expect_usage_error(parse({"--frobnicate"}), "--frobnicate");
}

TEST(Cli, Help) {
    for (const char* flag : {"--help", "-h"}) {
        const eb::ParseResult r = parse({flag});
        ASSERT_EQ(r.exit, eb::ExitCode::success);
        EXPECT_TRUE(contains(r.out, "run")) << r.out;
        EXPECT_TRUE(contains(r.out, "doctor")) << r.out;
        EXPECT_TRUE(r.err.empty());
    }
}

TEST(Cli, Version) {
    const eb::ParseResult r = parse({"--version"});
    ASSERT_EQ(r.exit, eb::ExitCode::success);
    EXPECT_TRUE(contains(r.out, "edgebench " + std::string(eb::version()))) << r.out;
    EXPECT_TRUE(contains(r.out, "results schema " + std::string(eb::results_schema_version())));
    EXPECT_EQ(r.out, eb::version_text());
}

TEST(Cli, SubcommandHelp) {
    const eb::ParseResult run = parse({"run", "--help"});
    ASSERT_EQ(run.exit, eb::ExitCode::success);
    for (const char* option :
         {"--model", "--provider", "--threads", "--binding", "--warmup", "--iterations", "--trials",
          "--cooldown-ms", "--input", "--seed", "--constant-value", "--graph-opt", "--label",
          "--output-dir", "--dry-run"}) {
        EXPECT_TRUE(contains(run.out, option)) << option;
    }
    const eb::ParseResult doctor = parse({"doctor", "--help"});
    ASSERT_EQ(doctor.exit, eb::ExitCode::success);
    EXPECT_TRUE(contains(doctor.out, "--json")) << doctor.out;
}

// --------------------------------------------------------------------------- doctor

TEST(Cli, Doctor) {
    eb::ParseResult r = parse({"doctor"});
    EXPECT_FALSE(r.exit.has_value());
    EXPECT_EQ(r.command, eb::Command::doctor);
    EXPECT_FALSE(r.doctor.json);
    r = parse({"doctor", "--json"});
    EXPECT_EQ(r.command, eb::Command::doctor);
    EXPECT_TRUE(r.doctor.json);
    expect_usage_error(parse({"doctor", "--model", "x.onnx"}), "--model");
}

// --------------------------------------------------------------------------- run

TEST(CliRun, DefaultsWithOnlyModel) {
    TempModel model;
    const eb::ParseResult r = parse(run_args(model));
    ASSERT_FALSE(r.exit.has_value()) << r.err;
    EXPECT_EQ(r.command, eb::Command::run);
    const eb::RunConfig& c = r.run;
    const eb::RunConfig d;
    EXPECT_EQ(c.model, model.path());
    EXPECT_EQ(c.provider, d.provider);
    EXPECT_EQ(c.threads, d.threads);
    EXPECT_EQ(c.binding, d.binding);
    EXPECT_EQ(c.warmup, d.warmup);
    EXPECT_EQ(c.iterations, d.iterations);
    EXPECT_EQ(c.trials, d.trials);
    EXPECT_EQ(c.cooldown_ms, d.cooldown_ms);
    EXPECT_EQ(c.input, d.input);
    EXPECT_FALSE(c.seed.has_value());
    EXPECT_FALSE(c.constant_value.has_value());
    EXPECT_EQ(c.graph_opt, d.graph_opt);
    EXPECT_FALSE(c.label.has_value());
    EXPECT_EQ(c.output_dir, d.output_dir);
    EXPECT_FALSE(c.dry_run);
}

TEST(CliRun, EveryOption) {
    TempModel model;
    const eb::ParseResult r = parse(run_args(
        model, {"--provider",    "cpu",   "--threads",    "8",        "--binding",        "run",
                "--warmup",      "0",     "--iterations", "5",        "--trials",         "1",
                "--cooldown-ms", "0",     "--input",      "constant", "--constant-value", "1.5",
                "--graph-opt",   "basic", "--label",      "my run",   "--output-dir",     "out",
                "--dry-run"}));
    ASSERT_FALSE(r.exit.has_value()) << r.err;
    const eb::RunConfig& c = r.run;
    EXPECT_EQ(c.threads, 8);
    EXPECT_EQ(c.binding, eb::RunBinding::run);
    EXPECT_EQ(c.warmup, 0);
    EXPECT_EQ(c.iterations, 5);
    EXPECT_EQ(c.trials, 1);
    EXPECT_EQ(c.cooldown_ms, 0);
    EXPECT_EQ(c.input, eb::InputKind::constant);
    EXPECT_EQ(c.constant_value, 1.5);
    EXPECT_EQ(c.graph_opt, eb::GraphOptLevel::enable_basic);
    EXPECT_EQ(c.label, "my run");
    EXPECT_EQ(c.output_dir, std::filesystem::path("out"));
    EXPECT_TRUE(c.dry_run);
}

TEST(CliRun, SeedWithRandomInput) {
    TempModel model;
    const eb::ParseResult r = parse(run_args(model, {"--input", "random", "--seed", "42"}));
    ASSERT_FALSE(r.exit.has_value()) << r.err;
    EXPECT_EQ(r.run.seed, 42u);
}

TEST(CliRun, ProviderValues) {
    TempModel model;
    EXPECT_FALSE(parse(run_args(model, {"--provider", "cpu"})).exit.has_value());
    for (const char* bad : {"npu", "qnn", "gpu", "CPU"}) {
        const eb::ParseResult r = parse(run_args(model, {"--provider", bad}));
        expect_usage_error(r, "supported providers: cpu");
        EXPECT_TRUE(contains(r.err, bad)) << r.err;
    }
}

TEST(CliRun, ModelErrors) {
    expect_usage_error(parse({"run"}), "--model: is required");
    expect_usage_error(parse({"run", "--model", "C:\\definitely\\missing\\model.onnx"}),
                       "--model: file not found");
    TempModel txt("model.txt");
    expect_usage_error(parse(run_args(txt)), "--model: must be an .onnx file");
}

TEST(CliRun, InvalidNumbers) {
    TempModel model;
    expect_usage_error(parse(run_args(model, {"--threads", "0"})), "error: [usage_error] --threads: must be between 1 and 256 (got 0)");
    expect_usage_error(parse(run_args(model, {"--threads", "-1"})), "--threads");
    expect_usage_error(parse(run_args(model, {"--threads", "four"})), "--threads");
    expect_usage_error(parse(run_args(model, {"--iterations", "0"})), "--iterations");
    expect_usage_error(parse(run_args(model, {"--trials", "0"})), "--trials");
    expect_usage_error(parse(run_args(model, {"--warmup", "-1"})), "--warmup");
    expect_usage_error(parse(run_args(model, {"--cooldown-ms", "-5"})), "--cooldown-ms");
    expect_usage_error(parse(run_args(model, {"--seed", "-1"})), "--seed");
    expect_usage_error(parse(run_args(model, {"--input", "constant", "--constant-value", "x"})),
                       "--constant-value");
}

TEST(CliRun, InputKindMismatch) {
    TempModel model;
    expect_usage_error(parse(run_args(model, {"--input", "constant", "--seed", "1"})),
                       "--seed: is only allowed with --input random");
    expect_usage_error(parse(run_args(model, {"--constant-value", "0.5"})),
                       "--constant-value: is only allowed with --input constant");
    expect_usage_error(parse(run_args(model, {"--input", "random", "--constant-value", "0.5"})),
                       "--constant-value: is only allowed with --input constant");
}

TEST(CliRun, InvalidChoices) {
    TempModel model;
    expect_usage_error(parse(run_args(model, {"--binding", "iobinding_preallocated"})),
                       "--binding");
    expect_usage_error(parse(run_args(model, {"--input", "dataset"})), "--input");
    expect_usage_error(parse(run_args(model, {"--graph-opt", "layout"})), "--graph-opt");
}

TEST(CliRun, UnknownOptionAndMissingValue) {
    TempModel model;
    expect_usage_error(parse(run_args(model, {"--bogus"})), "--bogus");
    expect_usage_error(parse(run_args(model, {"--threads"})), "--threads");
}

TEST(CliRun, AllValidationErrorsAreReported) {
    TempModel model;
    const eb::ParseResult r = parse(run_args(model, {"--threads", "0", "--trials", "0"}));
    expect_usage_error(r, "--threads");
    EXPECT_TRUE(contains(r.err, "--trials")) << r.err;
    EXPECT_TRUE(contains(r.err, "edgebench run --help")) << r.err;
}

TEST(CliRun, NameFormatAndLoadSample) {
    TempModel model;
    eb::ParseResult r = parse(run_args(model));
    ASSERT_FALSE(r.exit.has_value()) << r.err;
    EXPECT_FALSE(r.run.name.has_value());
    EXPECT_EQ(r.run.format, eb::OutputFormat::summary);
    EXPECT_EQ(r.run.load_sample_ms, 1000);

    r = parse(
        run_args(model, {"--name", "fp32-cpu_t4.v2", "--format", "json", "--load-sample-ms", "0"}));
    ASSERT_FALSE(r.exit.has_value()) << r.err;
    EXPECT_EQ(r.run.name, "fp32-cpu_t4.v2");
    EXPECT_EQ(r.run.format, eb::OutputFormat::json);
    EXPECT_EQ(r.run.load_sample_ms, 0);

    expect_usage_error(parse(run_args(model, {"--name", "has space"})), "--name: may only contain");
    expect_usage_error(parse(run_args(model, {"--name", "a/b"})), "--name");
    expect_usage_error(parse(run_args(model, {"--name", ".hidden"})), "--name");
    expect_usage_error(parse(run_args(model, {"--format", "xml"})), "--format");
    expect_usage_error(parse(run_args(model, {"--load-sample-ms", "-1"})), "--load-sample-ms");
}

TEST(Cli, DoctorLoadSample) {
    eb::ParseResult r = parse({"doctor", "--load-sample-ms", "0"});
    ASSERT_FALSE(r.exit.has_value()) << r.err;
    EXPECT_EQ(r.doctor.load_sample_ms, 0);
    EXPECT_EQ(parse({"doctor"}).doctor.load_sample_ms, 1000);
    expect_usage_error(parse({"doctor", "--load-sample-ms", "-5"}), "--load-sample-ms");
}
