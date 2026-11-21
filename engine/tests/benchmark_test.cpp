// run_benchmark() against the bundled ONNX Runtime on the fixture models: trial
// structure, call order (placement after the timed trials), error paths, and a
// round trip of the in-memory result through scripts/validate_results.py.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <fstream>

#include "edgebench/benchmark.hpp"
#include "edgebench/onnx_inspect.hpp"
#include "edgebench/runtime.hpp"
#include "process_util.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using json = nlohmann::ordered_json;

namespace {

const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();
const fs::path kFixtures = kSourceDir / "engine" / "tests" / "fixtures" / "models";

eb::Runtime& runtime() {
    static std::unique_ptr<eb::Runtime> rt = [] {
        const eb::OrtLoadResult load = eb::load_onnxruntime();
        if (!load.ok())
            throw std::runtime_error(load.message);
        return std::move(eb::Runtime::create(load)).value();
    }();
    return *rt;
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

eb::RunConfig short_run(const fs::path& model) {
    eb::RunConfig c;
    c.model = model;
    c.trials = 2;
    c.warmup = 3;
    c.iterations = 10;
    c.cooldown_ms = 0;
    return c;
}

RecordingProvider make_provider(const eb::RunConfig& c) {
    auto p = eb::make_provider(runtime(), c);
    if (!p)
        throw std::runtime_error(p.error().message);
    return RecordingProvider(std::move(p).value());
}

// ------------------------------------------------------------------- serializer
// A minimal mapping of BenchmarkResult onto the 1.1.0 schema, so the in-memory
// shape is validated against the schema. The real exporter replaces it.

json opt(const std::optional<std::string>& v) {
    return v ? json(*v) : json(nullptr);
}
template <typename T>
json opt(const std::optional<T>& v) {
    return v ? json(*v) : json(nullptr);
}

json stats_json(const eb::LatencyStats& s) {
    return {{"count", s.count},
            {"mean_ms", s.mean_ms},
            {"p50_ms", s.p50_ms},
            {"p95_ms", opt(s.p95_ms)},
            {"p99_ms", opt(s.p99_ms)},
            {"stddev_ms", opt(s.stddev_ms)},
            {"min_ms", s.min_ms},
            {"max_ms", s.max_ms},
            {"inferences_per_sec", s.inferences_per_sec},
            {"items_per_sec", s.items_per_sec}};
}

json tensors_json(const std::vector<eb::TensorInfo>& tensors) {
    json out = json::array();
    for (const eb::TensorInfo& t : tensors)
        out.push_back({{"name", t.name}, {"dtype", eb::dtype_name(t.dtype)}, {"shape", t.shape}});
    return out;
}

json to_schema_json(const eb::BenchmarkResult& r) {
    json warnings = json::array();
    for (const eb::ResultWarning& w : r.warnings)
        warnings.push_back({{"code", w.code}, {"message", w.message}, {"path", opt(w.path)}});

    json system = eb::to_json(r.system);
    system["power"] = {{"source", nullptr},
                       {"power_mode", nullptr},
                       {"power_plan", nullptr},
                       {"battery_percent", nullptr},
                       {"battery_saver", nullptr}};
    system["load"] = nullptr;

    const eb::ProviderDescription& p = r.runtime.provider;
    json ep_device = nullptr;
    if (p.ep_device) {
        const eb::EpDeviceInfo& d = p.ep_device->device;
        ep_device = {{"ep_name", d.ep_name},
                     {"ep_vendor", d.ep_vendor},
                     {"device_type", d.device_type},
                     {"vendor", d.vendor},
                     {"vendor_id", d.vendor_id},
                     {"device_id", d.device_id},
                     {"driver_version", opt(p.ep_device->driver_version)},
                     {"metadata", d.metadata},
                     {"selection",
                      {{"method", p.ep_device->selection_method},
                       {"candidate_count", p.ep_device->candidate_count}}}};
    }
    json modules = json::array();
    for (const eb::LoadedModule& m : r.runtime.loaded_modules)
        modules.push_back({{"name", m.name}, {"path", eb::path_to_utf8(m.path)}});
    json devices = json::array();
    for (const eb::EpDeviceInfo& d : r.runtime.ep_devices)
        devices.push_back(eb::to_json(d));

    const eb::ModelPrecision& mp = r.model.precision;
    const eb::RunConfig& c = r.config;
    const eb::TimingResult& t = r.timing;
    json trials = json::array();
    for (const eb::TrialResult& trial : t.trials)
        trials.push_back({{"trial_index", trial.trial_index}, {"stats", stats_json(trial.stats)}});
    const eb::TrialVariation& tv = t.aggregate->trial_variation;
    const eb::NodeAssignment& na = r.execution.node_assignment;

    return {
        {"schema_version", r.schema_version},
        {"run_id", r.run_id},
        {"timestamp_utc", r.timestamp_utc},
        {"label", opt(r.label)},
        {"status", r.status},
        {"error", nullptr},
        {"warnings", warnings},
        {"tool",
         {{"name", "edgebench"},
          {"version", r.tool_version},
          {"git_commit", opt(r.tool.git_commit)},
          {"git_dirty", opt(r.tool.git_dirty)},
          {"build_type", r.tool.build_type},
          {"compiler", r.tool.compiler},
          {"implementation", "cpp"}}},
        {"system", system},
        {"runtime",
         {{"onnxruntime",
           {{"version", r.runtime.onnxruntime.ort_version},
            {"library_path", eb::path_to_utf8(r.runtime.onnxruntime.dll_path)},
            {"api_version", r.runtime.onnxruntime.api_version},
            {"load_check", r.runtime.onnxruntime.load_check()}}},
          {"execution_provider",
           {{"name", p.execution_provider},
            {"version", opt(p.ep_version)},
            {"library_path", opt(p.ep_library_path)}}},
          {"qnn", nullptr},
          {"ep_device", ep_device},
          {"loaded_modules", modules},
          {"ep_devices", devices}}},
        {"model",
         {{"name", r.model.name},
          {"file_name", r.model.file.file_name},
          {"sha256", r.model.file.sha256},
          {"size_bytes", r.model.file.size_bytes},
          {"opset", opt(r.model.opset)},
          {"ir_version", opt(r.model.ir_version)},
          {"stored_precision", mp.stored_precision},
          {"inputs", tensors_json(r.model.inputs)},
          {"outputs", tensors_json(r.model.outputs)},
          {"source",
           {{"library", nullptr},
            {"identifier", nullptr},
            {"export_tool", opt(r.model.export_tool)}}},
          {"quantization",
           {{"method", mp.quantization_method},
            {"format", opt(mp.quantization_format)},
            {"activation_dtype", opt(mp.activation_dtype)},
            {"weight_dtype", opt(mp.weight_dtype)},
            {"calibration", nullptr},
            {"tool", nullptr}}},
          {"precision_notes", nullptr},
          {"normalization", nullptr}}},
        {"config",
         {{"requested",
           {{"execution_provider", eb::execution_provider_name(c.provider)},
            {"device_type", eb::device_type_name(c.provider)},
            {"compute_precision", nullptr}}},
          {"strict_mode", false},
          {"batch_size", r.batch_size},
          {"input",
           {{"kind", eb::schema_name(c.input)},
            {"constant_value", c.input == eb::InputKind::constant
                                   ? json(c.effective_constant_value())
                                   : json(nullptr)},
            {"seed", c.input == eb::InputKind::random ? json(c.effective_seed()) : json(nullptr)},
            {"dataset", nullptr},
            {"preprocessing", r.input_preprocessing}}},
          {"warmup_iterations", c.warmup},
          {"measured_iterations", c.iterations},
          {"trials", c.trials},
          {"run_api", "sync"},
          {"ort_execution_mode", "sequential"},
          {"intra_op_threads", c.threads},
          {"inter_op_threads", nullptr},
          {"graph_optimization_level", eb::schema_name(c.graph_opt)},
          {"ep_options", p.ep_options},
          {"session_config", p.session_config},
          {"ep_context_cache", nullptr},
          {"run_binding", eb::schema_name(c.binding)},
          {"profiling", {{"ort", false}, {"qnn_level", "off"}}}}},
        {"execution",
         {{"full_device_execution_verified", r.execution.full_device_execution_verified},
          {"verification_method", na.verification_method},
          {"verification_details", r.execution.verification_details},
          {"node_assignment", {{"total_nodes", na.total_nodes}, {"nodes_by_ep", na.nodes_by_ep}}},
          {"qnn_validation", nullptr}}},
        {"timing",
         {{"timer", {{"source", t.timer.source}, {"resolution_ns", opt(t.timer.resolution_ns)}}},
          {"percentile_method", t.percentile_method},
          {"scope", t.scope},
          {"session_init",
           {{"session_create_ms", t.session_init.session_create_ms},
            {"ep_context_state", t.session_init.ep_context_state},
            {"cold_ms", t.session_init.cold_ms},
            {"context_build_ms", opt(t.session_init.context_build_ms)},
            {"cached_ms", nullptr}}},
          {"first_inference_ms", opt(t.first_inference_ms)},
          {"trials", trials},
          {"aggregate",
           {{"stats", stats_json(t.aggregate->stats)},
            {"trial_variation",
             {{"trial_mean_min_ms", tv.trial_mean_min_ms},
              {"trial_mean_max_ms", tv.trial_mean_max_ms},
              {"trial_mean_stddev_ms", opt(tv.trial_mean_stddev_ms)},
              {"trial_mean_cv_ratio", opt(tv.trial_mean_cv_ratio)}}}}},
          {"accelerator", nullptr}}},
        {"memory", nullptr},
        {"accuracy", nullptr},
        {"raw", nullptr},
    };
}

}  // namespace

TEST(Benchmark, ShortRunOnFixture) {
    const eb::RunConfig c = short_run(kFixtures / "fixed_float32.onnx");
    RecordingProvider provider = make_provider(c);
    std::vector<std::string> progress;
    auto result = eb::run_benchmark(runtime(), provider, c,
                                    [&progress](std::string_view m) { progress.emplace_back(m); });
    ASSERT_TRUE(result.ok()) << result.error().message;
    const eb::BenchmarkResult& r = result.value();

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

TEST(Benchmark, InMemoryResultValidatesAgainstTheSchema) {
    const fs::path python = fs::path(EDGEBENCH_PYTHON).make_preferred();
    if (!fs::is_regular_file(python))
        GTEST_SKIP() << python << " not found";
    const eb::RunConfig c = short_run(kFixtures / "fixed_float32.onnx");
    RecordingProvider provider = make_provider(c);
    auto result = eb::run_benchmark(runtime(), provider, c);
    ASSERT_TRUE(result.ok()) << result.error().message;

    eb::test::TempDir dir;
    const fs::path file = dir.path() / "result.json";
    std::ofstream(file) << to_schema_json(result.value()).dump(2);
    const eb::test::ProcessResult v = eb::test::run_process(
        python,
        {(kSourceDir / "scripts" / "validate_results.py").native(), L"--strict", file.native()},
        kSourceDir);
    EXPECT_EQ(v.exit_code, 0u) << v.output;
    EXPECT_NE(v.out.find("1 passed, 0 failed, 0 with warnings"), std::string::npos) << v.output;
}

TEST(Benchmark, NonFiniteOutputStopsTheRun) {
    const eb::RunConfig c = short_run(kFixtures / "nan_output.onnx");
    RecordingProvider provider = make_provider(c);
    auto result = eb::run_benchmark(runtime(), provider, c);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().exit_code, eb::ExitCode::internal_error);
    EXPECT_EQ(result.error().kind, "non_finite_output");
    EXPECT_NE(result.error().message.find("trial 1 of 2: output 'y' has a non-finite value"),
              std::string::npos)
        << result.error().message;
    // Stopped in the first trial: no second session, no placement.
    EXPECT_EQ(provider.calls, (std::vector<std::string>{"create_session"}));
}

TEST(Benchmark, UsageErrorsComeFromSessionCreation) {
    const eb::RunConfig c = short_run(kFixtures / "dynamic_batch.onnx");
    RecordingProvider provider = make_provider(c);
    auto result = eb::run_benchmark(runtime(), provider, c);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().exit_code, eb::ExitCode::usage_error);
    EXPECT_NE(result.error().message.find("trial 1 of 2: input 'x' has a dynamic dimension"),
              std::string::npos)
        << result.error().message;
}

TEST(Benchmark, ZeroWarmupHasNoFirstInference) {
    eb::RunConfig c = short_run(kFixtures / "fixed_float32.onnx");
    c.warmup = 0;
    c.trials = 1;
    RecordingProvider provider = make_provider(c);
    auto result = eb::run_benchmark(runtime(), provider, c);
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_FALSE(result.value().timing.first_inference_ms.has_value());
    EXPECT_FALSE(result.value().timing.aggregate->trial_variation.trial_mean_stddev_ms.has_value());
}

TEST(OnnxInspect, MobileNetV2Models) {
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

    const auto q = eb::inspect_onnx_file(s8w);
    ASSERT_TRUE(q.ok());
    const eb::ModelPrecision p = eb::infer_precision(q.value());
    EXPECT_EQ(p.stored_precision, "int8");
    EXPECT_EQ(p.quantization_method, "static");
    EXPECT_EQ(p.quantization_format, "qdq");
    EXPECT_EQ(p.weight_dtype, "int8");       // per-channel symmetric int8 weights
    EXPECT_EQ(p.activation_dtype, "uint8");  // QUInt8 activations
}
