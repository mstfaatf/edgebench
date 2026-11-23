// run_benchmark() against the bundled ONNX Runtime on the fixture models: trial
// structure, call order (placement after the timed trials), failed runs, and the
// result files (written by the real exporter and checked with
// scripts/validate_results.py).

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <charconv>
#include <fstream>
#include <sstream>

#include "edgebench/benchmark.hpp"
#include "edgebench/onnx_inspect.hpp"
#include "edgebench/result_files.hpp"
#include "edgebench/result_json.hpp"
#include "edgebench/runtime.hpp"
#include "edgebench/sha256.hpp"
#include "process_util.hpp"
#include "test_runtime.hpp"
#include "validator.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using json = nlohmann::ordered_json;

namespace {

const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();
const fs::path kFixtures = kSourceDir / "engine" / "tests" / "fixtures" / "models";

eb::Runtime& runtime() {
    return eb::test::shared_runtime();
}

// Forwards to a real provider and records the order of the calls.
class RecordingProvider final : public eb::Provider {
public:
    explicit RecordingProvider(std::unique_ptr<eb::Provider> inner) : inner_(std::move(inner)) {}

    std::string_view name() const override { return inner_->name(); }
    std::string_view device_type() const override { return inner_->device_type(); }
    const eb::ProviderDescription& describe() const override { return inner_->describe(); }
    eb::Result<std::unique_ptr<eb::BenchSession>> create_session(const fs::path& model,
                                                                 const eb::RunConfig& c) override {
        calls.push_back("create_session");
        return inner_->create_session(model, c);
    }
    eb::Result<eb::NodeAssignment> record_placement(const fs::path& model,
                                                    const eb::RunConfig& c) override {
        calls.push_back("record_placement");
        return inner_->record_placement(model, c);
    }

    std::vector<std::string> calls;

private:
    std::unique_ptr<eb::Provider> inner_;
};

// Short and fast: no cooldown and no background-load sample.
eb::RunConfig short_run(const fs::path& model) {
    eb::RunConfig c;
    c.model = model;
    c.trials = 2;
    c.warmup = 3;
    c.iterations = 10;
    c.cooldown_ms = 0;
    c.load_sample_ms = 0;
    return c;
}

RecordingProvider make_provider(const eb::RunConfig& c) {
    auto p = eb::make_provider(runtime(), c);
    if (!p)
        throw std::runtime_error(p.error().message);
    return RecordingProvider(std::move(p).value());
}

eb::BenchmarkResult run(const eb::RunConfig& c, RecordingProvider* recorder = nullptr,
                        std::vector<std::string>* progress = nullptr) {
    std::optional<RecordingProvider> local;
    if (!recorder)
        local.emplace(make_provider(c));
    RecordingProvider& p = recorder ? *recorder : *local;
    return eb::run_benchmark(runtime(), p, c, [progress](std::string_view m) {
        if (progress)
            progress->emplace_back(m);
    });
}

// Runs scripts/validate_results.py --strict on `path`; returns its stdout + stderr.
eb::test::ProcessResult validate(const fs::path& path) {
    return eb::test::run_validator(path);
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::vector<std::string>> read_csv(const fs::path& p) {
    std::vector<std::vector<std::string>> rows;
    std::istringstream in(read_file(p));
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        std::vector<std::string> cells;
        std::stringstream ss(line);
        for (std::string cell; std::getline(ss, cell, ',');)
            cells.push_back(cell);
        if (!line.empty() && line.back() == ',')
            cells.emplace_back();
        rows.push_back(cells);
    }
    return rows;
}

double parse_double(const std::string& s) {
    double v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc() || ptr != s.data() + s.size())
        throw std::runtime_error("not a number: " + s);
    return v;
}

}  // namespace

// --------------------------------------------------------------------------- trial loop

TEST(Benchmark, ShortRunOnFixture) {
    const eb::RunConfig c = short_run(kFixtures / "fixed_float32.onnx");
    RecordingProvider provider = make_provider(c);
    std::vector<std::string> progress;
    const eb::BenchmarkResult r = run(c, &provider, &progress);
    ASSERT_FALSE(r.failure) << r.failure->error.message;
    EXPECT_EQ(r.status(), "success");

    // Fresh session per trial, placement only after the last trial.
    EXPECT_EQ(provider.calls,
              (std::vector<std::string>{"create_session", "create_session", "record_placement"}));
    EXPECT_EQ(progress.back(), "recording node placement");

    EXPECT_EQ(r.schema_version, "1.1.0");
    EXPECT_EQ(r.batch_size, 1);
    const eb::TimingResult& t = r.timing;
    ASSERT_EQ(t.trials.size(), 2u);
    for (int i = 0; i < 2; ++i) {
        const eb::TrialResult& trial = t.trials[static_cast<std::size_t>(i)];
        EXPECT_EQ(trial.trial_index, i);
        EXPECT_EQ(trial.sequence, i);
        EXPECT_EQ(trial.latencies_ms.size(), 10u);
        EXPECT_EQ(trial.stats.count, 10u);
        EXPECT_TRUE(trial.first_inference_ms.has_value());
        EXPECT_GT(trial.session_create_ms, 0.0);
    }
    ASSERT_EQ(t.session_init.cold_ms.size(), 2u);
    EXPECT_EQ(t.session_init.cold_ms[0], t.trials[0].session_create_ms);
    EXPECT_EQ(t.session_init.ep_context_state, "not_used");
    EXPECT_FALSE(t.session_init.context_build_ms.has_value());
    EXPECT_FALSE(t.session_init.cached_ms.has_value());
    EXPECT_EQ(t.first_inference_ms, t.trials[0].first_inference_ms);
    ASSERT_TRUE(t.aggregate.has_value());
    EXPECT_EQ(t.aggregate->stats.count, 20u);
    EXPECT_TRUE(t.aggregate->stats.p95_ms.has_value());  // 20 pooled samples
    EXPECT_EQ(t.scope, "model_execution");
    EXPECT_EQ(t.percentile_method, "nearest_rank");
    EXPECT_EQ(t.timer.source, "std::chrono::steady_clock");

    EXPECT_TRUE(r.execution.full_device_execution_verified);
    EXPECT_TRUE(r.execution.node_assignment.all_on("CPUExecutionProvider"));
    EXPECT_EQ(r.model.precision.stored_precision, "fp32");
    EXPECT_EQ(r.model.opset, 17);
    EXPECT_FALSE(r.runtime.loaded_modules.empty());
    EXPECT_EQ(r.input_preprocessing, eb::generated_input_description(c));

    const std::string summary = eb::format_summary(r);
    EXPECT_NE(summary.find("aggregate"), std::string::npos) << summary;
    EXPECT_NE(summary.find("trial 2"), std::string::npos) << summary;
}

TEST(Benchmark, RecordsPowerLoadAndMemory) {
    eb::RunConfig c = short_run(kFixtures / "fixed_float32.onnx");
    c.load_sample_ms = 100;  // cooldown 0: one sample before trial 1
    const eb::BenchmarkResult r = run(c);
    ASSERT_FALSE(r.failure) << r.failure->error.message;

    ASSERT_TRUE(r.system.power.has_value());
    EXPECT_TRUE(r.system.power->source.has_value());
    EXPECT_TRUE(r.system.power->power_mode.has_value());
    EXPECT_TRUE(r.system.power->power_plan.has_value());
    EXPECT_TRUE(r.system.power->battery_saver.has_value());

    ASSERT_TRUE(r.system.load.has_value());
    EXPECT_NEAR(r.system.load->sample_window_s, 0.1, 0.05);
    EXPECT_GE(r.system.load->cpu_utilization_ratio, 0.0);
    EXPECT_LE(r.system.load->cpu_utilization_ratio, 1.0);
    ASSERT_EQ(r.timing.trials.size(), 2u);
    EXPECT_TRUE(r.timing.trials[0].idle_before.has_value());
    EXPECT_FALSE(r.timing.trials[1].idle_before.has_value());  // no cooldown between trials

    EXPECT_EQ(r.memory.scope, "process");
    EXPECT_GT(r.memory.peak_working_set_bytes, 0u);
    EXPECT_GT(r.memory.peak_private_bytes, 0u);
    EXPECT_GT(r.memory.working_set_after_last_trial_bytes, 0u);
    EXPECT_LE(r.memory.working_set_after_last_trial_bytes, r.memory.peak_working_set_bytes);
    EXPECT_TRUE(r.memory.session_create_working_set_delta_bytes.has_value());
    for (const eb::TrialResult& trial : r.timing.trials)
        EXPECT_TRUE(trial.session_create_working_set_delta_bytes.has_value());

    const std::string summary = eb::format_summary(r);
    EXPECT_NE(summary.find("\nPower       "), std::string::npos) << summary;
    EXPECT_NE(summary.find("\nCPU load    pre-run "), std::string::npos) << summary;
    EXPECT_NE(summary.find("\nMemory      working set "), std::string::npos) << summary;
}

TEST(Benchmark, LoadSampleCanBeDisabled) {
    const eb::BenchmarkResult r = run(short_run(kFixtures / "fixed_float32.onnx"));
    ASSERT_FALSE(r.failure);
    EXPECT_FALSE(r.system.load.has_value());
    bool warned = false;
    for (const eb::ResultWarning& w : r.warnings)
        warned |= w.code == "value_unavailable" && w.path == "/system/load";
    EXPECT_TRUE(warned);
}

TEST(Benchmark, IdlesBeforeEveryTrialIncludingTheFirst) {
    eb::RunConfig c = short_run(kFixtures / "fixed_float32.onnx");
    c.cooldown_ms = 50;
    RecordingProvider provider = make_provider(c);
    std::vector<std::string> progress;
    const eb::BenchmarkResult r = run(c, &provider, &progress);
    ASSERT_FALSE(r.failure);

    // The timeline: idle, trial 1, idle, trial 2, placement.
    ASSERT_EQ(progress.size(), 5u);
    EXPECT_EQ(progress[0].rfind("idle 50 ms before trial 1 of 2", 0), 0u) << progress[0];
    EXPECT_EQ(progress[1].rfind("trial 1 of 2: creating session", 0), 0u) << progress[1];
    EXPECT_EQ(progress[2].rfind("idle 50 ms before trial 2 of 2", 0), 0u) << progress[2];
    EXPECT_EQ(progress[3].rfind("trial 2 of 2: creating session", 0), 0u) << progress[3];
    EXPECT_EQ(progress[4], "recording node placement");

    for (const eb::TrialResult& trial : r.timing.trials) {
        ASSERT_TRUE(trial.idle_before.has_value());
        EXPECT_GE(trial.idle_before->sample_window_s, 0.045);
    }
    EXPECT_EQ(r.system.load->cpu_utilization_ratio,
              r.timing.trials[0].idle_before->cpu_utilization_ratio);
}

TEST(Benchmark, NonFiniteOutputFailsTheRun) {
    const eb::RunConfig c = short_run(kFixtures / "nan_output.onnx");
    RecordingProvider provider = make_provider(c);
    const eb::BenchmarkResult r = run(c, &provider);
    ASSERT_TRUE(r.failure.has_value());
    EXPECT_EQ(r.status(), "failed");
    EXPECT_EQ(r.failure->error.exit_code, eb::ExitCode::internal_error);
    EXPECT_EQ(r.failure->error.kind, "non_finite_output");
    EXPECT_EQ(r.failure->stage, "measurement");
    EXPECT_NE(r.failure->error.message.find("trial 1 of 2: output 'y' has a non-finite value"),
              std::string::npos)
        << r.failure->error.message;
    // Stopped in the first trial: no second session, no placement.
    EXPECT_EQ(provider.calls, (std::vector<std::string>{"create_session"}));
    EXPECT_TRUE(r.known.runtime && r.known.model && r.known.session);
    EXPECT_FALSE(r.known.placement);
    EXPECT_TRUE(r.timing.trials.empty());
}

TEST(Benchmark, UsageErrorsComeFromSessionCreation) {
    const eb::BenchmarkResult r = run(short_run(kFixtures / "dynamic_batch.onnx"));
    ASSERT_TRUE(r.failure.has_value());
    EXPECT_EQ(r.failure->error.exit_code, eb::ExitCode::usage_error);
    EXPECT_EQ(r.failure->stage, "session_creation");
    EXPECT_NE(r.failure->error.message.find("trial 1 of 2: input 'x' has a dynamic dimension"),
              std::string::npos)
        << r.failure->error.message;
}

TEST(Benchmark, ZeroWarmupHasNoFirstInference) {
    eb::RunConfig c = short_run(kFixtures / "fixed_float32.onnx");
    c.warmup = 0;
    c.trials = 1;
    const eb::BenchmarkResult r = run(c);
    ASSERT_FALSE(r.failure);
    EXPECT_FALSE(r.timing.first_inference_ms.has_value());
    EXPECT_FALSE(r.timing.aggregate->trial_variation.trial_mean_stddev_ms.has_value());
}

// --------------------------------------------------------------------------- result files

TEST(ResultFiles, RealRunWritesValidatedFiles) {
    const eb::RunConfig c = short_run(kFixtures / "fixed_float32.onnx");
    const eb::BenchmarkResult r = run(c);
    ASSERT_FALSE(r.failure);
    eb::test::TempDir dir;
    const auto written = eb::write_result_files(r, dir.path(), "fixture");
    ASSERT_TRUE(written.ok()) << written.error().message;
    const eb::WrittenResult& w = written.value();

    ASSERT_TRUE(fs::is_regular_file(w.result_file));
    ASSERT_TRUE(fs::is_regular_file(w.latest_file));
    ASSERT_TRUE(w.run_dir.has_value());
    EXPECT_EQ(w.result_file.filename().string(), "fixture_" + w.stamp + ".json");
    EXPECT_EQ(w.latest_file.filename().string(), "fixture_latest.json");
    EXPECT_EQ(read_file(w.latest_file), read_file(w.result_file));
    for (const auto& entry : fs::directory_iterator(dir.path()))  // no temporary files left
        EXPECT_NE(entry.path().filename().string().front(), '.') << entry.path();

    // The document: every key, raw linked to latencies.csv with its digest.
    const json doc = json::parse(read_file(w.result_file));
    EXPECT_EQ(doc, w.document);
    EXPECT_EQ(doc.at("status"), "success");
    EXPECT_EQ(doc.at("raw").at("latencies_file"), "fixture_" + w.stamp + "/latencies.csv");
    EXPECT_EQ(doc.at("raw").at("sha256").get<std::string>(),
              eb::sha256_file(*w.run_dir / "latencies.csv").value());

    // latencies.csv: trials x iterations rows, values exactly the in-memory ones.
    const auto lat = read_csv(*w.run_dir / "latencies.csv");
    ASSERT_EQ(lat.size(), 1u + 2 * 10);
    EXPECT_EQ(lat[0], (std::vector<std::string>{"trial_index", "iteration_index", "latency_ms"}));
    std::size_t row = 1;
    for (const eb::TrialResult& trial : r.timing.trials) {
        for (std::size_t i = 0; i < trial.latencies_ms.size(); ++i, ++row) {
            EXPECT_EQ(lat[row][0], std::to_string(trial.trial_index));
            EXPECT_EQ(lat[row][1], std::to_string(i));
            EXPECT_EQ(parse_double(lat[row][2]), trial.latencies_ms[i]);  // exact
        }
    }

    // summary.csv matches the JSON statistics.
    const auto sum = read_csv(*w.run_dir / "summary.csv");
    ASSERT_EQ(sum.size(), 1u + 2 + 1);
    const std::vector<std::string>& header = sum[0];
    EXPECT_EQ(header.front(), "row");
    EXPECT_EQ(header.back(), "session_create_ms");
    auto check_row = [&](const std::vector<std::string>& cells, const json& stats) {
        for (std::size_t k = 2; k + 1 < header.size(); ++k) {
            const json& v = stats.at(header[k]);
            if (v.is_null())
                EXPECT_EQ(cells[k], "") << header[k];
            else
                EXPECT_EQ(parse_double(cells[k]), v.get<double>()) << header[k];
        }
    };
    for (int i = 0; i < 2; ++i) {
        const auto& cells = sum[static_cast<std::size_t>(1 + i)];
        EXPECT_EQ(cells[0], "trial");
        EXPECT_EQ(cells[1], std::to_string(i));
        check_row(cells, doc.at("timing").at("trials").at(static_cast<std::size_t>(i)).at("stats"));
        EXPECT_EQ(parse_double(cells.back()),
                  r.timing.trials[static_cast<std::size_t>(i)].session_create_ms);
    }
    EXPECT_EQ(sum[3][0], "aggregate");
    check_row(sum[3], doc.at("timing").at("aggregate").at("stats"));

    if (const std::string& why = eb::test::validator_unavailable_reason(); !why.empty())
        GTEST_SKIP() << why;
    const eb::test::ProcessResult v = validate(w.result_file);
    EXPECT_EQ(v.exit_code, 0u) << v.output;
    EXPECT_NE(v.out.find("1 passed, 0 failed, 0 with warnings"), std::string::npos) << v.output;
    // Only values that are null for a real reason carry value_unavailable warnings.
    for (const eb::ResultWarning& warn : r.warnings) {
        if (warn.code == "value_unavailable")
            EXPECT_TRUE(warn.path == "/runtime/ep_device/driver_version" ||
                        warn.path == "/system/load")  // --load-sample-ms 0
                << warn.message;
        else  // the machine's state may legitimately trigger these (short trials vary)
            EXPECT_TRUE(warn.code == "background_cpu_load_high" ||
                        warn.code == "power_state_changed" || warn.code == "trial_spread_high")
                << warn.code << ": " << warn.message;
    }
}

TEST(ResultFiles, SameMillisecondGetsASuffix) {
    const eb::BenchmarkResult r = run(short_run(kFixtures / "fixed_float32.onnx"));
    ASSERT_FALSE(r.failure);
    eb::test::TempDir dir;
    const auto when = r.started_at;  // the same stamp twice: a forced collision
    const auto first = eb::write_result_files(r, dir.path(), "same", when);
    const auto second = eb::write_result_files(r, dir.path(), "same", when);
    ASSERT_TRUE(first.ok() && second.ok());
    EXPECT_EQ(second.value().stamp, first.value().stamp + "-2");
    EXPECT_NE(first.value().result_file, second.value().result_file);
    EXPECT_TRUE(fs::is_regular_file(first.value().result_file));
    EXPECT_TRUE(fs::is_regular_file(*second.value().run_dir / "latencies.csv"));
    EXPECT_EQ(read_file(second.value().latest_file), read_file(second.value().result_file));
    // The suffixed result links its own latency file.
    EXPECT_EQ(second.value().document.at("raw").at("latencies_file"),
              "same_" + second.value().stamp + "/latencies.csv");
}

TEST(ResultFiles, FailedRunWritesAValidFailedResult) {
    const eb::BenchmarkResult r = run(short_run(kFixtures / "nan_output.onnx"));
    ASSERT_TRUE(r.failure.has_value());
    eb::test::TempDir dir;
    const auto written = eb::write_result_files(r, dir.path(), "nan");
    ASSERT_TRUE(written.ok()) << written.error().message;
    const json& doc = written.value().document;
    EXPECT_EQ(doc.at("status"), "failed");
    EXPECT_EQ(doc.at("error").at("stage"), "measurement");
    EXPECT_EQ(doc.at("error").at("code"), "non_finite_output");
    EXPECT_TRUE(doc.at("raw").is_null());  // no completed trial, no latencies
    EXPECT_FALSE(written.value().run_dir.has_value());
    EXPECT_TRUE(doc.at("timing").is_null());
    EXPECT_FALSE(doc.at("runtime").is_null());  // partial data is kept
    EXPECT_FALSE(doc.at("model").is_null());
    EXPECT_TRUE(doc.at("execution").at("verification_method").is_null());
    if (const std::string& why = eb::test::validator_unavailable_reason(); !why.empty())
        GTEST_SKIP() << why;
    const eb::test::ProcessResult v = validate(written.value().result_file);
    EXPECT_EQ(v.exit_code, 0u) << v.output;
}

TEST(ResultFiles, FailedLoadResultHasNoRuntime) {
    eb::OrtLoadResult load;
    load.status = eb::OrtLoadStatus::failed;
    load.error_kind = eb::OrtLoadError::version_mismatch;
    load.message = "onnxruntime.dll is ONNX Runtime 1.17.1, but EdgeBench requires 1.32.0";
    eb::RunConfig c = short_run(kFixtures / "fixed_float32.onnx");
    const eb::BenchmarkResult r = eb::failed_load_result(c, load);
    eb::test::TempDir dir;
    const auto written = eb::write_result_files(r, dir.path(), "load");
    ASSERT_TRUE(written.ok()) << written.error().message;
    const json& doc = written.value().document;
    EXPECT_EQ(
        doc.at("error"),
        (json{{"stage", "environment"}, {"message", load.message}, {"code", "version_mismatch"}}));
    EXPECT_TRUE(doc.at("runtime").is_null());
    EXPECT_EQ(doc.at("model").at("inputs").at(0).at("shape"), (json{1, 4}));  // from the file
    EXPECT_EQ(doc.at("config").at("batch_size"), 1);
    if (const std::string& why = eb::test::validator_unavailable_reason(); !why.empty())
        GTEST_SKIP() << why;
    EXPECT_EQ(validate(written.value().result_file).exit_code, 0u);
}

TEST(OnnxInspectModels, MobileNetV2) {
    const fs::path models = kSourceDir / "models" / "mobilenetv2";
    const fs::path fp32 = models / "mobilenetv2_fp32_b1.onnx";
    const fs::path s8w = models / "mobilenetv2_int8qdq_pc_s8w_b1.onnx";
    if (!fs::is_regular_file(fp32) || !fs::is_regular_file(s8w))
        GTEST_SKIP() << "MobileNetV2 models not present (see docs/setup.md)";
    const auto f = eb::inspect_onnx_file(fp32);
    ASSERT_TRUE(f.ok());
    EXPECT_EQ(f.value().opset, 17);
    EXPECT_EQ(f.value().ir_version, 8);
    EXPECT_EQ(f.value().producer_name, "pytorch");
    EXPECT_EQ(eb::infer_precision(f.value()).stored_precision, "fp32");
    ASSERT_EQ(f.value().inputs.size(), 1u);
    EXPECT_EQ(f.value().inputs[0].shape, (std::vector<std::int64_t>{1, 3, 224, 224}));

    const auto q = eb::inspect_onnx_file(s8w);
    ASSERT_TRUE(q.ok());
    const eb::ModelPrecision p = eb::infer_precision(q.value());
    EXPECT_EQ(p.stored_precision, "int8");
    EXPECT_EQ(p.quantization_method, "static");
    EXPECT_EQ(p.quantization_format, "qdq");
    EXPECT_EQ(p.weight_dtype, "int8");       // per-channel symmetric int8 weights
    EXPECT_EQ(p.activation_dtype, "uint8");  // QUInt8 activations
}
