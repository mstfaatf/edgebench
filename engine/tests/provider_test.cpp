// CpuProvider and OrtBenchSession against the bundled ONNX Runtime: fixture
// models, input generation, Run vs IoBinding, node placement, and agreement with
// the Python (onnxruntime) reference outputs.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include "edgebench/cpu_provider.hpp"
#include "edgebench/input_fill.hpp"
#include "edgebench/provider.hpp"
#include "edgebench/runtime.hpp"
#include "process_util.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;

namespace {

const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();
const fs::path kFixtures = kSourceDir / "engine" / "tests" / "fixtures" / "models";
const fs::path kModels = kSourceDir / "models" / "mobilenetv2";
const fs::path kFp32 = kModels / "mobilenetv2_fp32_b1.onnx";
const fs::path kS8w = kModels / "mobilenetv2_int8qdq_pc_s8w_b1.onnx";
const fs::path kReference = kSourceDir / "data" / "ort_probe_reference" / "latest";

// One Runtime (and so one Ort::Env) for the whole test program.
eb::Runtime& runtime() {
    static std::unique_ptr<eb::Runtime> rt = [] {
        const eb::OrtLoadResult load = eb::load_onnxruntime();
        if (!load.ok())
            throw std::runtime_error(load.message);
        return std::move(eb::Runtime::create(load)).value();
    }();
    return *rt;
}

eb::RunConfig config_for(const fs::path& model, eb::RunBinding binding) {
    eb::RunConfig c;
    c.model = model;
    c.binding = binding;
    c.threads = 4;
    return c;
}

std::unique_ptr<eb::BenchSession> make_session(const eb::RunConfig& config) {
    auto provider = eb::make_provider(runtime(), config);
    if (!provider)
        throw std::runtime_error(provider.error().message);
    auto session = provider.value()->create_session(config.model, config);
    if (!session)
        throw std::runtime_error(session.error().message);
    return std::move(session).value();
}

eb::Error session_error(const eb::RunConfig& config) {
    auto provider = eb::make_provider(runtime(), config);
    auto session = provider.value()->create_session(config.model, config);
    if (session.ok())
        throw std::runtime_error("expected session creation to fail");
    return session.error();
}

std::vector<std::byte> copy_output(const eb::BenchSession& s, std::size_t i) {
    const eb::TensorView v = s.output(i);
    return {v.bytes.begin(), v.bytes.end()};
}

// --------------------------------------------------------------------------- reference data

struct Record {
    std::string kind, config, image_id;
    fs::path file;
    std::size_t elements = 0;
};

std::vector<Record> read_index(const fs::path& dir) {
    std::ifstream in(dir / "index.tsv");
    std::vector<Record> out;
    std::string line;
    std::getline(in, line);  // header: kind config image image_id file dtype shape
    while (std::getline(in, line)) {
        std::vector<std::string> f;
        std::stringstream ss(line);
        for (std::string cell; std::getline(ss, cell, '\t');)
            f.push_back(cell);
        if (f.size() < 7)
            continue;
        Record r{f[0], f[1], f[3], dir / fs::path(f[4]).make_preferred(), 1};
        std::stringstream dims(f[6]);
        for (std::string d; std::getline(dims, d, ',');)
            r.elements *= static_cast<std::size_t>(std::stoll(d));
        out.push_back(r);
    }
    return out;
}

std::vector<float> read_f32(const Record& r) {
    std::vector<float> v(r.elements);
    std::ifstream in(r.file, std::ios::binary);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    if (in.gcount() != static_cast<std::streamsize>(v.size() * 4))
        throw std::runtime_error("short read: " + r.file.string());
    return v;
}

std::size_t argmax(std::span<const float> v) {
    return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

}  // namespace

// --------------------------------------------------------------------------- provider

TEST(CpuProvider, DescribesTheCpuEp) {
    auto provider = eb::make_provider(runtime(), eb::RunConfig{});
    ASSERT_TRUE(provider.ok()) << provider.error().message;
    const eb::Provider& p = *provider.value();
    EXPECT_EQ(p.name(), "cpu");
    EXPECT_EQ(p.device_type(), "CPU");
    const eb::ProviderDescription& d = p.describe();
    EXPECT_EQ(d.execution_provider, "CPUExecutionProvider");
    EXPECT_FALSE(d.ep_version.has_value());
    EXPECT_FALSE(d.ep_library_path.has_value());
    ASSERT_TRUE(d.ep_device.has_value());
    EXPECT_EQ(d.ep_device->device.ep_name, "CPUExecutionProvider");
    EXPECT_EQ(d.ep_device->device.device_type, "CPU");
    EXPECT_EQ(d.ep_device->selection_method, "by_device_type");
    EXPECT_EQ(d.ep_device->candidate_count, 1);
    EXPECT_TRUE(d.ep_options.empty());
    EXPECT_EQ(d.session_config,
              (std::map<std::string, std::string>{{"session.intra_op.allow_spinning", "1"}}));
}

TEST(CpuProvider, SessionOnFixedFixture) {
    auto s = make_session(config_for(kFixtures / "fixed_float32.onnx", eb::RunBinding::run));
    ASSERT_EQ(s->inputs().size(), 1u);
    EXPECT_EQ(s->inputs()[0].name, "x");
    EXPECT_EQ(s->inputs()[0].shape, (std::vector<std::int64_t>{1, 4}));
    ASSERT_EQ(s->outputs().size(), 1u);
    EXPECT_EQ(s->outputs()[0].name, "y");
    EXPECT_EQ(s->outputs()[0].shape, (std::vector<std::int64_t>{1, 3}));
    EXPECT_EQ(s->batch_size(), 1);
    EXPECT_EQ(s->run_binding(), eb::RunBinding::run);
    EXPECT_GT(s->session_create_ms(), 0.0);
    EXPECT_TRUE(s->output(0).empty());  // no run yet

    ASSERT_TRUE(s->run_once().ok());
    const std::span<const float> y = s->output(0).as_float();
    ASSERT_EQ(y.size(), 3u);

    // y = relu(x W + b) with the fixture's constants (scripts/make_test_models.py).
    const float w[4][3] = {
        {0.5f, -1.0f, 0.25f}, {1.0f, 0.5f, -0.5f}, {-0.25f, 1.0f, 0.75f}, {0.125f, -0.5f, 1.0f}};
    const float b[3] = {0.1f, -0.2f, 0.3f};
    const std::span<const float> x = s->input_data(0);
    for (int j = 0; j < 3; ++j) {
        float acc = b[j];
        for (int i = 0; i < 4; ++i)
            acc += x[static_cast<std::size_t>(i)] * w[i][j];
        EXPECT_NEAR(y[static_cast<std::size_t>(j)], std::max(acc, 0.0f), 1e-6f) << j;
    }
}

TEST(CpuProvider, NodePlacementIsAllCpu) {
    for (const fs::path& model : {kFixtures / "fixed_float32.onnx", kFp32}) {
        if (!fs::is_regular_file(model))
            continue;
        auto s = make_session(config_for(model, eb::RunBinding::run));
        const eb::NodeAssignment& p = s->node_assignment();
        EXPECT_GT(p.total_nodes, 0) << model;
        EXPECT_TRUE(p.all_on("CPUExecutionProvider")) << model;
        EXPECT_EQ(p.nodes_by_ep.size(), 1u) << model;
        EXPECT_EQ(p.verification_method, "node_assignment");
        RecordProperty(model.filename().string() + "_nodes", p.total_nodes);
    }
}

TEST(CpuProvider, DynamicBatchIsAUsageError) {
    const eb::Error e =
        session_error(config_for(kFixtures / "dynamic_batch.onnx", eb::RunBinding::run));
    EXPECT_EQ(e.exit_code, eb::ExitCode::usage_error);
    EXPECT_EQ(e.kind, "dynamic_shape");
    EXPECT_NE(e.message.find("input 'x' has a dynamic dimension 0 ('batch')"), std::string::npos)
        << e.message;
}

TEST(CpuProvider, Int64InputIsAUsageError) {
    const eb::Error e =
        session_error(config_for(kFixtures / "int64_input.onnx", eb::RunBinding::run));
    EXPECT_EQ(e.exit_code, eb::ExitCode::usage_error);
    EXPECT_EQ(e.kind, "unsupported_input_dtype");
    EXPECT_NE(e.message.find("input 'ids' has dtype int64"), std::string::npos) << e.message;
}

TEST(CpuProvider, InvalidModelIsAnOrtError) {
    eb::test::TempDir dir;
    const fs::path bad = dir.path() / "garbage.onnx";
    std::ofstream(bad) << "this is not an ONNX model";
    const eb::Error e = session_error(config_for(bad, eb::RunBinding::run));
    EXPECT_EQ(e.exit_code, eb::ExitCode::ort_runtime_error);
    EXPECT_EQ(e.kind, "session_creation_failed");
    ASSERT_TRUE(e.ort_error_code.has_value());
    EXPECT_NE(*e.ort_error_code, 0);
    EXPECT_NE(e.message.find(*e.ort_error_name), std::string::npos) << e.message;
}

// --------------------------------------------------------------------------- inputs

TEST(CpuProvider, RandomInputsFollowTheDocumentedGenerator) {
    eb::RunConfig c = config_for(kFixtures / "fixed_float32.onnx", eb::RunBinding::run);
    c.seed = 99;
    auto s = make_session(c);
    std::vector<float> expected(4);
    eb::SplitMix64 rng(99);
    eb::fill_random(expected, rng);
    const std::span<const float> x = s->input_data(0);
    ASSERT_EQ(x.size(), expected.size());
    EXPECT_EQ(std::memcmp(x.data(), expected.data(), expected.size() * sizeof(float)), 0);
}

TEST(CpuProvider, ConstantInputs) {
    eb::RunConfig c = config_for(kFixtures / "fixed_float32.onnx", eb::RunBinding::run);
    c.input = eb::InputKind::constant;
    c.constant_value = 0.5;
    auto s = make_session(c);
    for (float v : s->input_data(0))
        EXPECT_EQ(v, 0.5f);
}

// --------------------------------------------------------------------------- run modes

TEST(CpuProvider, RunAndIoBindingAreBitIdentical) {
    for (const fs::path& model : {kFixtures / "fixed_float32.onnx", kFp32, kS8w}) {
        if (!fs::is_regular_file(model))
            continue;
        auto run = make_session(config_for(model, eb::RunBinding::run));
        auto bound = make_session(config_for(model, eb::RunBinding::iobinding_preallocated));
        ASSERT_EQ(run->input_data(0).size(), bound->input_data(0).size());
        EXPECT_EQ(std::memcmp(run->input_data(0).data(), bound->input_data(0).data(),
                              run->input_data(0).size_bytes()),
                  0);  // same seed, same inputs
        ASSERT_TRUE(run->run_once().ok());
        ASSERT_TRUE(bound->run_once().ok());
        for (std::size_t i = 0; i < run->outputs().size(); ++i)
            EXPECT_EQ(copy_output(*run, i), copy_output(*bound, i)) << model << " output " << i;
    }
}

TEST(CpuProvider, IoBindingReusesItsBuffers) {
    const fs::path model = fs::is_regular_file(kFp32) ? kFp32 : kFixtures / "fixed_float32.onnx";
    auto s = make_session(config_for(model, eb::RunBinding::iobinding_preallocated));
    const float* input = s->input_data(0).data();
    ASSERT_TRUE(s->run_once().ok());
    const std::byte* output = s->output(0).bytes.data();
    const std::vector<std::byte> first = copy_output(*s, 0);
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(s->run_once().ok());
        EXPECT_EQ(s->input_data(0).data(), input);
        EXPECT_EQ(s->output(0).bytes.data(), output);
        EXPECT_EQ(copy_output(*s, 0), first);  // same input, same output
    }
}

TEST(CpuProvider, IoBindingSeesInputChanges) {
    auto s = make_session(
        config_for(kFixtures / "fixed_float32.onnx", eb::RunBinding::iobinding_preallocated));
    ASSERT_TRUE(s->run_once().ok());
    const std::vector<std::byte> before = copy_output(*s, 0);
    eb::fill_constant(s->input_data(0), 2.0f);
    ASSERT_TRUE(s->run_once().ok());
    EXPECT_NE(copy_output(*s, 0), before);
}

// --------------------------------------------------------------------------- Python reference

struct ReferenceCase {
    const char* name;
    fs::path model;
    const char* config;  // output config in index.tsv
    eb::RunBinding binding;
};

class PythonReference : public ::testing::TestWithParam<ReferenceCase> {};

TEST_P(PythonReference, MatchesOrtProbeTolerance) {
    const ReferenceCase& c = GetParam();
    if (!fs::is_regular_file(c.model) || !fs::is_regular_file(kReference / "index.tsv"))
        GTEST_SKIP() << "model or reference data not present (see docs/setup.md)";
    const std::vector<Record> index = read_index(kReference);
    std::vector<Record> inputs, refs;
    for (const Record& r : index) {
        if (r.kind == "input" && r.config == "normalized")
            inputs.push_back(r);
        if (r.kind == "output" && r.config == c.config)
            refs.push_back(r);
    }
    ASSERT_FALSE(inputs.empty());
    ASSERT_EQ(inputs.size(), refs.size());

    auto s = make_session(config_for(c.model, c.binding));
    constexpr double kTolerance = 1e-5;  // ort_probe's CPU tolerance (max abs diff)
    double worst = 0.0;
    int top1_matches = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::vector<float> x = read_f32(inputs[i]);
        ASSERT_EQ(x.size(), s->input_data(0).size());
        std::copy(x.begin(), x.end(), s->input_data(0).begin());
        ASSERT_TRUE(s->run_once().ok());
        const std::span<const float> y = s->output(0).as_float();
        const std::vector<float> ref = read_f32(refs[i]);
        ASSERT_EQ(y.size(), ref.size());
        double diff = 0.0;
        for (std::size_t k = 0; k < y.size(); ++k)
            diff = std::max(diff, static_cast<double>(std::fabs(y[k] - ref[k])));
        worst = std::max(worst, diff);
        top1_matches += argmax(y) == argmax(ref);
    }
    std::printf("[ reference ] %s: %zu images, max abs diff %.3g, top-1 %d/%zu\n", c.name,
                inputs.size(), worst, top1_matches, inputs.size());
    RecordProperty("max_abs_diff", std::to_string(worst));
    RecordProperty("top1_matches", top1_matches);
    EXPECT_LE(worst, kTolerance);
    EXPECT_EQ(top1_matches, static_cast<int>(inputs.size()));
}

INSTANTIATE_TEST_SUITE_P(
    CpuProvider, PythonReference,
    ::testing::Values(
        ReferenceCase{"fp32_run", kFp32, "cpu_fp32", eb::RunBinding::run},
        ReferenceCase{"fp32_iobinding", kFp32, "cpu_fp32", eb::RunBinding::iobinding_preallocated},
        ReferenceCase{"s8w_run", kS8w, "cpu_s8w", eb::RunBinding::run},
        ReferenceCase{"s8w_iobinding", kS8w, "cpu_s8w", eb::RunBinding::iobinding_preallocated}),
    [](const ::testing::TestParamInfo<ReferenceCase>& info) { return info.param.name; });
