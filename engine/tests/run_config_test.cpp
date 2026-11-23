#include "edgebench/run_config.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <fstream>
#include <limits>

#include "test_files.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace eb = edgebench;
namespace fs = std::filesystem;
using eb::test::TempModel;

namespace {

std::vector<std::string> fields(const std::vector<eb::ConfigError>& errors) {
    std::vector<std::string> out;
    for (const auto& e : errors)
        out.push_back(e.field);
    return out;
}

eb::RunConfig valid_config(const TempModel& model) {
    eb::RunConfig c;
    c.model = model.path();
    return c;
}

}  // namespace

TEST(RunConfig, Defaults) {
    const eb::RunConfig c;
    EXPECT_TRUE(c.model.empty());
    EXPECT_EQ(c.provider, eb::ProviderKind::cpu);
    EXPECT_EQ(c.threads, 4);
    EXPECT_EQ(c.binding, eb::RunBinding::iobinding_preallocated);
    EXPECT_EQ(c.warmup, 20);
    EXPECT_EQ(c.iterations, 200);
    EXPECT_EQ(c.trials, 3);
    EXPECT_EQ(c.cooldown_ms, 2000);
    EXPECT_EQ(c.input, eb::InputKind::random);
    EXPECT_FALSE(c.seed.has_value());
    EXPECT_EQ(c.effective_seed(), 0u);
    EXPECT_FALSE(c.constant_value.has_value());
    EXPECT_EQ(c.effective_constant_value(), 0.0);
    EXPECT_EQ(c.graph_opt, eb::GraphOptLevel::enable_all);
    EXPECT_FALSE(c.label.has_value());
    EXPECT_EQ(c.output_dir, fs::path("benchmarks/results/local"));
    EXPECT_FALSE(c.dry_run);
}

TEST(Validate, ValidConfigHasNoErrors) {
    TempModel model;
    EXPECT_TRUE(eb::validate(valid_config(model)).empty());
}

TEST(Validate, ModelIsRequired) {
    const auto errors = eb::validate(eb::RunConfig{});
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_EQ(errors[0], (eb::ConfigError{"model", "is required"}));
}

TEST(Validate, ModelMustExist) {
    TempModel model;
    eb::RunConfig c = valid_config(model);
    c.model = model.path().parent_path() / "does-not-exist.onnx";
    const auto errors = eb::validate(c);
    ASSERT_EQ(fields(errors), std::vector<std::string>{"model"});
    EXPECT_NE(errors[0].message.find("file not found"), std::string::npos);
}

TEST(Validate, ModelMustBeARegularFile) {
    TempModel model;
    eb::RunConfig c = valid_config(model);
    const fs::path dir = model.path().parent_path() / "a_directory.onnx";
    fs::create_directory(dir);
    c.model = dir;
    const auto errors = eb::validate(c);
    fs::remove(dir);
    ASSERT_EQ(fields(errors), std::vector<std::string>{"model"});
    EXPECT_NE(errors[0].message.find("not a regular file"), std::string::npos);
}

TEST(Validate, ModelMustHaveOnnxExtension) {
    TempModel wrong("model.txt");
    const auto errors = eb::validate(valid_config(wrong));
    ASSERT_EQ(fields(errors), std::vector<std::string>{"model"});
    EXPECT_NE(errors[0].message.find(".onnx"), std::string::npos);

    TempModel upper("MODEL.ONNX");  // the extension check ignores case
    EXPECT_TRUE(eb::validate(valid_config(upper)).empty());
}

TEST(Validate, Threads) {
    TempModel model;
    eb::RunConfig c = valid_config(model);
    for (int bad : {0, -1}) {
        c.threads = bad;
        EXPECT_EQ(fields(eb::validate(c)), std::vector<std::string>{"threads"}) << bad;
    }
    c.threads = 1;
    EXPECT_TRUE(eb::validate(c).empty());
}

TEST(Validate, IterationCounts) {
    TempModel model;
    const eb::RunConfig base = valid_config(model);
    struct Case {
        int eb::RunConfig::*member;
        const char* field;
        int bad;
        int smallest_ok;
    };
    const Case cases[] = {
        {&eb::RunConfig::warmup, "warmup", -1, 0},
        {&eb::RunConfig::iterations, "iterations", 0, 1},
        {&eb::RunConfig::trials, "trials", 0, 1},
        {&eb::RunConfig::cooldown_ms, "cooldown-ms", -1, 0},
    };
    for (const Case& k : cases) {
        eb::RunConfig c = base;
        c.*k.member = k.bad;
        EXPECT_EQ(fields(eb::validate(c)), std::vector<std::string>{k.field}) << k.field;
        c.*k.member = k.smallest_ok;
        EXPECT_TRUE(eb::validate(c).empty()) << k.field;
    }
}

TEST(Validate, SeedOnlyWithRandomInput) {
    TempModel model;
    eb::RunConfig c = valid_config(model);
    c.seed = 7;
    EXPECT_TRUE(eb::validate(c).empty());  // random is the default
    c.input = eb::InputKind::constant;
    const auto errors = eb::validate(c);
    ASSERT_EQ(fields(errors), std::vector<std::string>{"seed"});
    EXPECT_EQ(errors[0].message, "is only allowed with --input random");
}

TEST(Validate, ConstantValueOnlyWithConstantInput) {
    TempModel model;
    eb::RunConfig c = valid_config(model);
    c.constant_value = 0.5;
    const auto errors = eb::validate(c);
    ASSERT_EQ(fields(errors), std::vector<std::string>{"constant-value"});
    EXPECT_EQ(errors[0].message, "is only allowed with --input constant");
    c.input = eb::InputKind::constant;
    EXPECT_TRUE(eb::validate(c).empty());
}

TEST(Validate, BothInputOptionsWithEitherKind) {
    TempModel model;
    eb::RunConfig c = valid_config(model);
    c.seed = 1;
    c.constant_value = 1.0;
    EXPECT_EQ(fields(eb::validate(c)), std::vector<std::string>{"constant-value"});
    c.input = eb::InputKind::constant;
    EXPECT_EQ(fields(eb::validate(c)), std::vector<std::string>{"seed"});
}

TEST(Validate, ConstantValueMustBeFinite) {
    TempModel model;
    eb::RunConfig c = valid_config(model);
    c.input = eb::InputKind::constant;
    for (double bad : {std::numeric_limits<double>::infinity(), std::nan("")}) {
        c.constant_value = bad;
        EXPECT_EQ(fields(eb::validate(c)), std::vector<std::string>{"constant-value"});
    }
}

TEST(Validate, LabelAndOutputDir) {
    TempModel model;
    eb::RunConfig c = valid_config(model);
    c.label = "";
    c.output_dir.clear();
    EXPECT_EQ(fields(eb::validate(c)), (std::vector<std::string>{"label", "output-dir"}));
}

TEST(Validate, ReportsEveryProblem) {
    eb::RunConfig c;
    c.threads = 0;
    c.iterations = 0;
    c.trials = 0;
    c.seed = 1;
    c.input = eb::InputKind::constant;
    EXPECT_EQ(fields(eb::validate(c)),
              (std::vector<std::string>{"model", "threads", "iterations", "trials", "seed"}));
}

TEST(Provider, OnlyCpuIsAccepted) {
    EXPECT_EQ(eb::parse_provider("cpu"), eb::ProviderKind::cpu);
    for (const char* bad : {"npu", "qnn", "gpu", "dml", "CPU", "", "cpu "})
        EXPECT_FALSE(eb::parse_provider(bad).has_value()) << "'" << bad << "'";
    EXPECT_EQ(eb::supported_providers(), "cpu");
    EXPECT_EQ(eb::execution_provider_name(eb::ProviderKind::cpu), "CPUExecutionProvider");
    EXPECT_EQ(eb::device_type_name(eb::ProviderKind::cpu), "CPU");
}

TEST(Spellings, CliToSchemaNames) {
    EXPECT_EQ(eb::schema_name(*eb::parse_binding("run")), "run");
    EXPECT_EQ(eb::schema_name(*eb::parse_binding("iobinding")), "iobinding_preallocated");
    EXPECT_FALSE(eb::parse_binding("iobinding_preallocated").has_value());
    EXPECT_EQ(eb::schema_name(*eb::parse_input_kind("random")), "random");
    EXPECT_EQ(eb::schema_name(*eb::parse_input_kind("constant")), "constant");
    EXPECT_FALSE(eb::parse_input_kind("dataset").has_value());
    EXPECT_EQ(eb::schema_name(*eb::parse_graph_opt("disable")), "disable_all");
    EXPECT_EQ(eb::schema_name(*eb::parse_graph_opt("basic")), "enable_basic");
    EXPECT_EQ(eb::schema_name(*eb::parse_graph_opt("extended")), "enable_extended");
    EXPECT_EQ(eb::schema_name(*eb::parse_graph_opt("all")), "enable_all");
    EXPECT_FALSE(eb::parse_graph_opt("layout").has_value());
}

TEST(ResolvedConfig, UsesSchemaFieldNamesAndDefaults) {
    TempModel model;
    const nlohmann::ordered_json j = eb::resolved_config_json(valid_config(model));
    const nlohmann::ordered_json& c = j.at("config");
    EXPECT_EQ(c.at("requested").at("execution_provider"), "CPUExecutionProvider");
    EXPECT_EQ(c.at("requested").at("device_type"), "CPU");
    EXPECT_TRUE(c.at("requested").at("compute_precision").is_null());
    EXPECT_EQ(c.at("strict_mode"), false);
    EXPECT_TRUE(c.at("batch_size").is_null());
    EXPECT_EQ(c.at("input").at("kind"), "random");
    EXPECT_EQ(c.at("input").at("seed"), 0);
    EXPECT_TRUE(c.at("input").at("constant_value").is_null());
    EXPECT_EQ(c.at("warmup_iterations"), 20);
    EXPECT_EQ(c.at("measured_iterations"), 200);
    EXPECT_EQ(c.at("trials"), 3);
    EXPECT_EQ(c.at("intra_op_threads"), 4);
    EXPECT_EQ(c.at("graph_optimization_level"), "enable_all");
    EXPECT_EQ(c.at("run_binding"), "iobinding_preallocated");
    EXPECT_EQ(c.at("profiling"), (nlohmann::ordered_json{{"ort", false}, {"qnn_level", "off"}}));
    EXPECT_TRUE(c.at("ep_context_cache").is_null());
    EXPECT_EQ(j.at("cooldown_ms"), 2000);
    EXPECT_TRUE(j.at("label").is_null());
    EXPECT_EQ(j.at("model").at("size_bytes"), model.size());
    EXPECT_TRUE(fs::path(j.at("output_dir").get<std::string>()).is_absolute());
}

TEST(ResolvedConfig, ConstantInput) {
    TempModel model;
    eb::RunConfig c = valid_config(model);
    c.input = eb::InputKind::constant;
    c.constant_value = 0.5;
    const nlohmann::ordered_json input = eb::resolved_config_json(c).at("config").at("input");
    EXPECT_EQ(input.at("kind"), "constant");
    EXPECT_EQ(input.at("constant_value"), 0.5);
    EXPECT_TRUE(input.at("seed").is_null());
}

TEST(Validate, UpperBounds) {
    TempModel model;
    const eb::RunConfig base = valid_config(model);
    struct Case {
        int eb::RunConfig::*member;
        const char* field;
        int largest_ok;
    };
    const Case cases[] = {
        {&eb::RunConfig::iterations, "iterations", eb::kMaxIterations},
        {&eb::RunConfig::trials, "trials", eb::kMaxTrials},
        {&eb::RunConfig::warmup, "warmup", eb::kMaxWarmup},
        {&eb::RunConfig::cooldown_ms, "cooldown-ms", eb::kMaxCooldownMs},
        {&eb::RunConfig::load_sample_ms, "load-sample-ms", eb::kMaxLoadSampleMs},
        {&eb::RunConfig::threads, "threads", eb::kMaxThreads},
    };
    EXPECT_EQ(eb::kMaxIterations, 1'000'000);
    EXPECT_EQ(eb::kMaxTrials, 1'000);
    EXPECT_EQ(eb::kMaxWarmup, 100'000);
    EXPECT_EQ(eb::kMaxCooldownMs, 600'000);
    EXPECT_EQ(eb::kMaxThreads, 256);
    for (const Case& k : cases) {
        eb::RunConfig c = base;
        c.*k.member = k.largest_ok;
        EXPECT_TRUE(eb::validate(c).empty()) << k.field;
        c.*k.member = k.largest_ok + 1;
        const std::vector<eb::ConfigError> errors = eb::validate(c);
        ASSERT_EQ(fields(errors), std::vector<std::string>{k.field}) << k.field;
        EXPECT_NE(errors[0].message.find("must be between"), std::string::npos)
            << errors[0].message;
        EXPECT_EQ(errors[0].kind, "usage_error");
    }
}

TEST(Validate, LockedModelIsUnreadable) {
    TempModel model;
    // No sharing at all: every other open fails with a sharing violation.
    HANDLE lock = CreateFileW(model.path().c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    ASSERT_NE(lock, INVALID_HANDLE_VALUE);
    const std::vector<eb::ConfigError> errors = eb::validate(valid_config(model));
    CloseHandle(lock);
    ASSERT_EQ(fields(errors), std::vector<std::string>{"model"});
    EXPECT_EQ(errors[0].kind, "model_unreadable");
    EXPECT_NE(errors[0].message.find("for reading"), std::string::npos) << errors[0].message;
    EXPECT_TRUE(eb::validate(valid_config(model)).empty());  // readable again
}
