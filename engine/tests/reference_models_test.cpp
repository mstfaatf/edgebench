// The engine computes exactly what Python's ONNX Runtime computes, for every
// exported model file.
//
// models/references/ (scripts/make_references.py) holds, for each file in
// models/manifest.json, the outputs of Python onnxruntime (CPU EP, 4 intra-op
// threads, the CPU provider's session settings) on two fixed input sets: 8
// preprocessed ImageNetV2 images (8 normalized reference windows for the
// time-series autoencoder) and the SplitMix64 input `edgebench run` generates
// by default. Each test creates the session through the same
// Provider / BenchSession path `edgebench run` uses, in the timed mode
// (IoBinding with preallocated buffers), and requires:
//
// - the generated input to equal the Python reproduction of the documented
//   generator byte for byte, and
// - every output to equal the Python output byte for byte (same ONNX Runtime
//   version, same thread count and session settings, so no tolerance).
//
// Suite "ReferenceModels" (ctest label `models`): skipped, with the reason,
// when the model files or the references are not present.
//
// Suite "ReferenceModelsNpu" (ctest label `npu`) does the same on the NPU: for
// every INT8 file against its `npu` reference and every FP32 file (run with
// FP16 compute, --npu-fp16) against its `npu_fp16` reference, through the QNN
// provider in strict mode with the options `edgebench run` records. Both sides
// use the same ONNX Runtime, QNN execution provider and HTP backend on the same
// device, so the outputs must be identical, again without a tolerance. Batch-8
// INT8 image files are included: their NPU outputs differ from batch 1 (the
// HTP computes that graph differently), but C++ and Python must still agree.
// "ReferenceDeterminismNpu" checks, on a few files, that fresh sessions and
// both run bindings give identical outputs. "ReferenceCachedNpu" checks that a
// session created from an EP context the engine built matches the references
// too (a subset by default; every file with EDGEBENCH_CACHED_REFERENCES_ALL=1).
// The NPU suites skip, with the
// reason, without a usable NPU (npu_skip_reason()) and, like the CPU suite,
// without the model files or the references.

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>

#include <nlohmann/json.hpp>

#include "edgebench/ep_context_cache.hpp"
#include "edgebench/npu_selection.hpp"
#include "edgebench/provider.hpp"
#include "edgebench/runtime.hpp"
#include "edgebench/sha256.hpp"
#include "process_util.hpp"
#include "test_runtime.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;

namespace {

const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();
const fs::path kModels = kSourceDir / "models";
const fs::path kReferences = kModels / "references";
constexpr int kThreads = 4;  // as recorded in every CPU reference (intra_op_threads)

// The files of models/manifest.json: models x variants x batch sizes
// (scripts/common/model_registry.py), for the exported image models and the
// imported autoencoder. IndexListsExactlyTheseFiles keeps the two in step.
const std::vector<std::string> kModelNames = {"efficientnet_lite0", "mobilenetv2", "resnet18"};
const std::vector<std::string> kVariants = {"fp32",          "fp32_folded", "int8",
                                            "int8_unpruned", "int8_folded", "int8_folded_unpruned"};
const std::vector<std::string> kImportedModelNames = {"conv1d_autoencoder"};
const std::vector<std::string> kImportedVariants = {"fp32_imported", "fp32", "int8",
                                                    "int8_unpruned"};
const std::vector<int> kBatches = {1, 4, 8};

std::string relative_model(const std::string& model, const std::string& variant, int batch) {
    return model + "/" + model + "_" + variant + "_b" + std::to_string(batch) + ".onnx";
}

struct Row {
    std::string kind, input_set, model, device;
    int run = 0;
    std::vector<std::string> inputs;
    std::string file;
    std::size_t elements = 1;
};

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string cell; std::getline(ss, cell, sep);)
        out.push_back(cell);
    return out;
}

// index.tsv: kind input_set model device run inputs file dtype shape sha256
std::vector<Row> read_index() {
    std::ifstream in(kReferences / "index.tsv");
    std::vector<Row> rows;
    std::string line;
    std::getline(in, line);
    if (line != "kind\tinput_set\tmodel\tdevice\trun\tinputs\tfile\tdtype\tshape\tsha256")
        throw std::runtime_error("index.tsv: unexpected header: " + line);
    while (std::getline(in, line)) {
        const std::vector<std::string> f = split(line, '\t');
        if (f.size() != 10)
            throw std::runtime_error("index.tsv: bad line: " + line);
        if (f[7] != "float32")
            throw std::runtime_error("index.tsv: unexpected dtype " + f[7]);
        Row r{f[0], f[1], f[2], f[3], std::stoi(f[4]), {}, f[6]};
        if (f[5] != "-")
            r.inputs = split(f[5], ',');
        for (const std::string& d : split(f[8], ','))
            r.elements *= static_cast<std::size_t>(std::stoll(d));
        rows.push_back(std::move(r));
    }
    return rows;
}

std::vector<std::byte> read_bytes(const std::string& rel) {
    const fs::path path = kReferences / fs::path(rel).make_preferred();
    std::ifstream in(path, std::ios::binary);
    std::vector<std::byte> data(static_cast<std::size_t>(fs::file_size(path)));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!in)
        throw std::runtime_error("cannot read " + path.string());
    return data;
}

double max_abs_diff(std::span<const std::byte> a, std::span<const std::byte> b) {
    const std::size_t n = std::min(a.size(), b.size()) / sizeof(float);
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        float x = 0.0f;
        float y = 0.0f;
        std::memcpy(&x, a.data() + i * sizeof(float), sizeof(float));
        std::memcpy(&y, b.data() + i * sizeof(float), sizeof(float));
        worst = std::max(worst, static_cast<double>(std::fabs(x - y)));
    }
    return worst;
}

// The configuration `edgebench run` uses by default (random input, seed 0,
// IoBinding with preallocated buffers); on the NPU in strict mode, with
// --npu-fp16 for FP32 files.
eb::RunConfig engine_config(const fs::path& model, const std::string& device) {
    eb::RunConfig c;
    c.model = model;
    c.threads = kThreads;
    c.binding = eb::RunBinding::iobinding_preallocated;
    if (device != "cpu") {
        c.provider = eb::ProviderKind::npu;
        c.npu_fp16 = device == "npu_fp16";
        if (device == "npu_finalization1")
            c.htp_graph_finalization_mode = "1";
    }
    return c;
}

// A session created as `edgebench run` creates it; `options` receives the
// provider options it recorded.
std::unique_ptr<eb::BenchSession> engine_session(
    const eb::RunConfig& c, std::map<std::string, std::string>* options = nullptr) {
    auto provider = eb::make_provider(eb::test::shared_runtime(), c);
    if (!provider)
        throw std::runtime_error(provider.error().kind + ": " + provider.error().message);
    if (options)
        *options = provider.value()->describe().ep_options;
    auto session = provider.value()->create_session(c.model, c);
    if (!session)
        throw std::runtime_error(session.error().kind + ": " + session.error().message);
    return std::move(session).value();
}

std::unique_ptr<eb::BenchSession> engine_session(const fs::path& model) {
    return engine_session(engine_config(model, "cpu"));
}

// The output rows of `rel` for `device`: the SplitMix64 run, then the data runs.
std::vector<Row> reference_runs(const std::string& rel, const std::string& device) {
    std::vector<Row> splitmix, data;
    for (const Row& r : read_index()) {
        if (r.kind == "output" && r.model == rel && r.device == device)
            (r.input_set.starts_with("splitmix64") ? splitmix : data).push_back(r);
    }
    splitmix.insert(splitmix.end(), data.begin(), data.end());
    return splitmix;
}

std::vector<std::byte> run_inputs(const Row& r) {
    std::vector<std::byte> x;
    for (const std::string& f : r.inputs) {
        const std::vector<std::byte> part = read_bytes(f);
        x.insert(x.end(), part.begin(), part.end());
    }
    return x;
}

// Feeds `x` to the session's (single) input and returns a copy of its output.
std::vector<std::byte> run_on(eb::BenchSession& session, const std::vector<std::byte>& x) {
    const std::span<float> input = session.input_data(0);
    if (x.size() != std::as_bytes(input).size())
        throw std::runtime_error("input size differs from the session's input");
    std::memcpy(input.data(), x.data(), x.size());
    if (eb::Result<void> r = session.run_once(); !r)
        throw std::runtime_error(r.error().message);
    const std::span<const std::byte> y = session.output(0).bytes;
    return {y.begin(), y.end()};
}

// Elements that differ, and the largest difference in units in the last place.
struct Difference {
    std::size_t elements = 0;
    std::uint32_t max_ulp = 0;
};

Difference difference(std::span<const std::byte> a, std::span<const std::byte> b) {
    Difference d;
    const std::size_t n = std::min(a.size(), b.size()) / sizeof(float);
    for (std::size_t i = 0; i < n; ++i) {
        std::int32_t x = 0;
        std::int32_t y = 0;
        std::memcpy(&x, a.data() + i * sizeof(float), sizeof(float));
        std::memcpy(&y, b.data() + i * sizeof(float), sizeof(float));
        if (x == y)
            continue;
        ++d.elements;
        // Map the sign-magnitude float bits onto a monotonic integer line.
        const std::int64_t ox = x < 0 ? std::int64_t{INT32_MIN} - x : x;
        const std::int64_t oy = y < 0 ? std::int64_t{INT32_MIN} - y : y;
        d.max_ulp = std::max(d.max_ulp, static_cast<std::uint32_t>(std::min<std::int64_t>(
                                            std::llabs(ox - oy), UINT32_MAX)));
    }
    return d;
}

using Param = std::tuple<std::string, std::string, int>;

class ReferenceModels : public ::testing::TestWithParam<Param> {};

}  // namespace

TEST_P(ReferenceModels, EngineMatchesPythonBitForBit) {
    const auto& [name, variant, batch] = GetParam();
    const std::string rel = relative_model(name, variant, batch);
    const fs::path model = kModels / fs::path(rel).make_preferred();
    if (!fs::is_regular_file(model))
        GTEST_SKIP() << rel << " is not present (see docs/models.md)";
    if (!fs::is_regular_file(kReferences / "index.tsv"))
        GTEST_SKIP() << "no reference outputs; run scripts/make_references.py";

    // A reference made from another version of the file is stale, not a mismatch.
    const fs::path record_path =
        kReferences / "outputs" / fs::path(rel).replace_extension() / "cpu.json";
    ASSERT_TRUE(fs::is_regular_file(record_path))
        << rel << " has no CPU reference; run scripts/make_references.py";
    std::ifstream record_file(record_path);
    const nlohmann::json record = nlohmann::json::parse(record_file);
    const auto model_sha = eb::sha256_file(model);
    ASSERT_TRUE(model_sha.ok());
    ASSERT_EQ(record.at("model_sha256").get<std::string>(), model_sha.value())
        << rel << ": the reference is stale; run scripts/make_references.py";
    ASSERT_EQ(record.at("intra_op_threads").get<int>(), kThreads);

    std::vector<Row> splitmix, images;
    for (const Row& r : read_index()) {
        if (r.kind == "output" && r.model == rel && r.device == "cpu")
            (r.input_set.starts_with("splitmix64") ? splitmix : images).push_back(r);
    }
    ASSERT_EQ(splitmix.size(), 1u) << rel;
    ASSERT_EQ(images.size(), static_cast<std::size_t>(8 / batch)) << rel;

    auto session = engine_session(model);
    const std::span<float> input = session->input_data(0);
    const std::span<const std::byte> input_bytes = std::as_bytes(input);

    // The timed input: what the engine generated must be the Python reproduction.
    ASSERT_EQ(splitmix[0].inputs.size(), 1u);
    const std::vector<std::byte> generated = read_bytes(splitmix[0].inputs[0]);
    ASSERT_EQ(generated.size(), input_bytes.size()) << rel;
    EXPECT_EQ(std::memcmp(generated.data(), input_bytes.data(), generated.size()), 0)
        << rel << ": generated input differs from the SplitMix64 reproduction";

    int identical = 0;
    double worst = 0.0;
    std::vector<Row> runs = splitmix;
    runs.insert(runs.end(), images.begin(), images.end());
    for (const Row& r : runs) {
        std::vector<std::byte> x;
        for (const std::string& f : r.inputs) {
            const std::vector<std::byte> part = read_bytes(f);
            x.insert(x.end(), part.begin(), part.end());
        }
        ASSERT_EQ(x.size(), input_bytes.size()) << r.file;
        std::memcpy(input.data(), x.data(), x.size());
        ASSERT_TRUE(session->run_once().ok()) << r.file;
        const std::span<const std::byte> y = session->output(0).bytes;
        const std::vector<std::byte> ref = read_bytes(r.file);
        ASSERT_EQ(y.size(), ref.size()) << r.file;
        const bool same = std::memcmp(y.data(), ref.data(), ref.size()) == 0;
        identical += same;
        if (!same)
            worst = std::max(worst, max_abs_diff(y, ref));
        EXPECT_TRUE(same) << rel << " " << r.input_set << " run " << r.run
                          << ": output differs from Python, max abs diff " << max_abs_diff(y, ref);
    }
    std::printf("[ reference ] %s: %d/%zu runs bit-identical (max abs diff %.3g)\n", rel.c_str(),
                identical, runs.size(), worst);
    RecordProperty("bit_identical_runs", identical);
}

// --------------------------------------------------------------------------- NPU

namespace {

// Variants with an NPU reference: INT8 files (`npu`) and FP32 files (`npu_fp16`).
const std::vector<std::string> kNpuVariants = {
    "fp32", "fp32_folded", "int8", "int8_unpruned", "int8_folded", "int8_folded_unpruned"};
const std::vector<std::string> kImportedNpuVariants = {"fp32_imported", "fp32", "int8",
                                                       "int8_unpruned"};

std::string npu_device(const std::string& variant) {
    return variant.starts_with("int8") ? "npu" : "npu_fp16";
}

// Skips without a usable NPU; every test leaves the QNN EP unregistered again.
class NpuTest {
protected:
    static std::optional<std::string> skip_reason() {
        return eb::test::npu_skip_reason(eb::test::probe_qnn_npu());
    }
    static void unregister() {
        const eb::Result<void> r = eb::test::shared_runtime().unregister_ep_library(eb::kQnnEpName);
        EXPECT_TRUE(r.ok()) << r.error().message;
    }
};

class ReferenceModelsNpu : public ::testing::TestWithParam<Param>, protected NpuTest {
protected:
    void TearDown() override { unregister(); }
};

// The record of `rel` for `device`, checked to be current and made with the
// options the engine passes; fails the test otherwise.
nlohmann::json npu_record(const std::string& rel, const fs::path& model,
                          const std::string& device) {
    const fs::path path =
        kReferences / "outputs" / fs::path(rel).replace_extension() / (device + ".json");
    if (!fs::is_regular_file(path))
        throw std::runtime_error(rel + " has no " + device +
                                 " reference; run scripts/make_references.py");
    std::ifstream file(path);
    nlohmann::json record = nlohmann::json::parse(file);
    const auto sha = eb::sha256_file(model);
    if (!sha || record.at("model_sha256").get<std::string>() != sha.value())
        throw std::runtime_error(rel + ": the " + device +
                                 " reference is stale; run scripts/make_references.py");
    return record;
}

}  // namespace

namespace {

// The engine's timed NPU session for `param` with the options of reference
// device `device` computes every saved output of that device bit for bit.
void check_npu_reference(const Param& param, const std::string& device) {
    const auto& [name, variant, batch] = param;
    const std::string rel = relative_model(name, variant, batch);
    const fs::path model = kModels / fs::path(rel).make_preferred();
    if (!fs::is_regular_file(model))
        GTEST_SKIP() << rel << " is not present (see docs/models.md)";
    if (!fs::is_regular_file(kReferences / "index.tsv"))
        GTEST_SKIP() << "no reference outputs; run scripts/make_references.py";
    const nlohmann::json record = npu_record(rel, model, device);

    // The reference was made with the options the engine passes (backend_path
    // aside: Python's is the onnxruntime-qnn package's copy of the same backend).
    std::map<std::string, std::string> options;
    auto session = engine_session(engine_config(model, device), &options);
    ASSERT_EQ(record.at("strict").get<bool>(), true);
    ASSERT_EQ(record.at("session").at("session.disable_cpu_ep_fallback"), "1");
    std::map<std::string, std::string> recorded =
        record.at("provider_options").get<std::map<std::string, std::string>>();
    recorded.erase("backend_path");
    options.erase("backend_path");
    ASSERT_EQ(options, recorded) << rel;

    const std::vector<Row> runs = reference_runs(rel, device);
    ASSERT_EQ(runs.size(), static_cast<std::size_t>(1 + 8 / batch)) << rel;
    ASSERT_TRUE(runs[0].input_set.starts_with("splitmix64")) << rel;

    // The timed input: what the engine generated must be the Python reproduction.
    ASSERT_EQ(runs[0].inputs.size(), 1u);
    const std::vector<std::byte> generated = read_bytes(runs[0].inputs[0]);
    const std::span<const std::byte> engine_input = std::as_bytes(session->input_data(0));
    ASSERT_EQ(generated.size(), engine_input.size()) << rel;
    EXPECT_EQ(std::memcmp(generated.data(), engine_input.data(), generated.size()), 0)
        << rel << ": generated input differs from the SplitMix64 reproduction";

    int identical = 0;
    for (const Row& r : runs) {
        const std::vector<std::byte> y = run_on(*session, run_inputs(r));
        const std::vector<std::byte> ref = read_bytes(r.file);
        ASSERT_EQ(y.size(), ref.size()) << r.file;
        const bool same = y == ref;
        identical += same;
        const Difference d = difference(y, ref);
        EXPECT_TRUE(same) << rel << " " << r.input_set << " run " << r.run << ": " << d.elements
                          << " of " << ref.size() / sizeof(float)
                          << " elements differ from Python, max abs diff " << max_abs_diff(y, ref)
                          << ", max " << d.max_ulp << " ULP";
    }
    std::printf("[ reference ] %s (%s): %d/%zu runs bit-identical\n", rel.c_str(), device.c_str(),
                identical, runs.size());
    ::testing::Test::RecordProperty("bit_identical_runs", identical);
}

}  // namespace

TEST_P(ReferenceModelsNpu, EngineMatchesPythonBitForBit) {
    if (const std::optional<std::string> reason = skip_reason())
        GTEST_SKIP() << *reason;
    check_npu_reference(GetParam(), npu_device(std::get<1>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(AllFiles, ReferenceModelsNpu,
                         ::testing::Combine(::testing::ValuesIn(kModelNames),
                                            ::testing::ValuesIn(kNpuVariants),
                                            ::testing::ValuesIn(kBatches)),
                         [](const ::testing::TestParamInfo<Param>& info) {
                             return std::get<0>(info.param) + "_" + std::get<1>(info.param) + "_b" +
                                    std::to_string(std::get<2>(info.param));
                         });

INSTANTIATE_TEST_SUITE_P(ImportedFiles, ReferenceModelsNpu,
                         ::testing::Combine(::testing::ValuesIn(kImportedModelNames),
                                            ::testing::ValuesIn(kImportedNpuVariants),
                                            ::testing::ValuesIn(kBatches)),
                         [](const ::testing::TestParamInfo<Param>& info) {
                             return std::get<0>(info.param) + "_" + std::get<1>(info.param) + "_b" +
                                    std::to_string(std::get<2>(info.param));
                         });

namespace {

// INT8 variants of the image models: they also have an `npu_finalization1`
// reference (htp_graph_finalization_optimization_mode 1).
const std::vector<std::string> kFinalizationVariants = {"int8", "int8_unpruned", "int8_folded",
                                                        "int8_folded_unpruned"};
constexpr const char* kFinalizationDevice = "npu_finalization1";
constexpr const char* kAllFinalizationEnv = "EDGEBENCH_FINALIZATION_REFERENCES_ALL";

class ReferenceFinalizationNpu : public ::testing::TestWithParam<Param>, protected NpuTest {
protected:
    void TearDown() override { unregister(); }
};

}  // namespace

// With --htp-graph-finalization-mode 1 the engine computes the saved
// npu_finalization1 references bit for bit (the Subset runs by default, every
// file with EDGEBENCH_FINALIZATION_REFERENCES_ALL=1).
TEST_P(ReferenceFinalizationNpu, EngineMatchesPythonBitForBit) {
    if (const std::optional<std::string> reason = skip_reason())
        GTEST_SKIP() << *reason;
    const std::string suite =
        ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
    char* value = nullptr;
    std::size_t length = 0;
    const bool all = _dupenv_s(&value, &length, kAllFinalizationEnv) == 0 && value &&
                     std::string(value) == "1";
    std::free(value);
    if (suite.find("Manual") != std::string::npos && !all)
        GTEST_SKIP() << "set " << kAllFinalizationEnv << "=1 to check every file";
    check_npu_reference(GetParam(), kFinalizationDevice);
}

INSTANTIATE_TEST_SUITE_P(Subset, ReferenceFinalizationNpu,
                         ::testing::Combine(::testing::ValuesIn(kModelNames),
                                            ::testing::Values(std::string("int8")),
                                            ::testing::Values(1, 8)),
                         [](const ::testing::TestParamInfo<Param>& info) {
                             return std::get<0>(info.param) + "_" + std::get<1>(info.param) + "_b" +
                                    std::to_string(std::get<2>(info.param));
                         });

INSTANTIATE_TEST_SUITE_P(AllFilesManual, ReferenceFinalizationNpu,
                         ::testing::Combine(::testing::ValuesIn(kModelNames),
                                            ::testing::ValuesIn(kFinalizationVariants),
                                            ::testing::ValuesIn(kBatches)),
                         [](const ::testing::TestParamInfo<Param>& info) {
                             return std::get<0>(info.param) + "_" + std::get<1>(info.param) + "_b" +
                                    std::to_string(std::get<2>(info.param));
                         });

namespace {

class ReferenceDeterminismNpu : public ::testing::TestWithParam<Param>, protected NpuTest {
protected:
    void TearDown() override { unregister(); }
};

}  // namespace

// Three fresh IoBinding sessions and one Run session give the same bytes for
// every reference input, and the same input twice in one session too.
TEST_P(ReferenceDeterminismNpu, FreshSessionsAndBothBindingsAreBitIdentical) {
    if (const std::optional<std::string> reason = skip_reason())
        GTEST_SKIP() << *reason;
    const auto& [name, variant, batch] = GetParam();
    const std::string rel = relative_model(name, variant, batch);
    const fs::path model = kModels / fs::path(rel).make_preferred();
    if (!fs::is_regular_file(model))
        GTEST_SKIP() << rel << " is not present (see docs/models.md)";
    if (!fs::is_regular_file(kReferences / "index.tsv"))
        GTEST_SKIP() << "no reference outputs; run scripts/make_references.py";
    const std::string device = npu_device(variant);
    const std::vector<Row> runs = reference_runs(rel, device);
    ASSERT_FALSE(runs.empty()) << rel;

    std::vector<std::vector<std::byte>> first;
    for (int s = 0; s < 4; ++s) {
        eb::RunConfig c = engine_config(model, device);
        if (s == 3)
            c.binding = eb::RunBinding::run;
        auto session = engine_session(c);
        ASSERT_EQ(session->run_binding(), c.binding);
        for (std::size_t i = 0; i < runs.size(); ++i) {
            const std::vector<std::byte> x = run_inputs(runs[i]);
            const std::vector<std::byte> y = run_on(*session, x);
            EXPECT_EQ(run_on(*session, x), y)
                << rel << " session " << s << ": repeated run differs";
            if (s == 0) {
                first.push_back(y);
                continue;
            }
            const Difference d = difference(y, first[i]);
            EXPECT_EQ(d.elements, 0u)
                << rel << " " << (s == 3 ? "Run" : "fresh IoBinding") << " session " << s << ", "
                << runs[i].input_set << " run " << runs[i].run << ": " << d.elements
                << " elements differ from the first session, max " << d.max_ulp << " ULP";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Subset, ReferenceDeterminismNpu,
                         ::testing::Values(Param{"mobilenetv2", "int8", 1},
                                           Param{"mobilenetv2", "int8", 8},
                                           Param{"conv1d_autoencoder", "int8", 1},
                                           Param{"efficientnet_lite0", "fp32", 1}),
                         [](const ::testing::TestParamInfo<Param>& info) {
                             return std::get<0>(info.param) + "_" + std::get<1>(info.param) + "_b" +
                                    std::to_string(std::get<2>(info.param));
                         });

namespace {

class ReferenceCachedNpu : public ::testing::TestWithParam<Param>, protected NpuTest {
protected:
    void TearDown() override { unregister(); }
};

// Set to run every file of AllFilesManual / ImportedFilesManual (slow: one
// context build per file).
constexpr const char* kAllCachedEnv = "EDGEBENCH_CACHED_REFERENCES_ALL";

bool all_cached_requested() {
    char* value = nullptr;
    std::size_t size = 0;
    const bool set = _dupenv_s(&value, &size, kAllCachedEnv) == 0 && value && *value &&
                     std::string(value) != "0";
    std::free(value);
    return set;
}

void check_cached_session(const Param& param) {
    const auto& [name, variant, batch] = param;
    const std::string rel = relative_model(name, variant, batch);
    const fs::path model = kModels / fs::path(rel).make_preferred();
    if (!fs::is_regular_file(model))
        GTEST_SKIP() << rel << " is not present (see docs/models.md)";
    if (!fs::is_regular_file(kReferences / "index.tsv"))
        GTEST_SKIP() << "no reference outputs; run scripts/make_references.py";
    const std::string device = npu_device(variant);
    (void)npu_record(rel, model, device);  // current, else it throws

    // Build the EP context as `edgebench run --ep-cache build` does, in a
    // temporary cache directory, then load a session from it.
    eb::test::TempDir cache(L"cached_reference");
    eb::RunConfig c = engine_config(model, device);
    c.ep_cache = eb::EpCacheMode::build;
    c.ep_cache_dir = cache.path();
    auto provider = eb::make_provider(eb::test::shared_runtime(), c);
    ASSERT_TRUE(provider) << provider.error().message;
    const fs::path context = cache.path() / eb::context_model_name(model);
    {
        auto building = provider.value()->create_context_building_session(model, c, context);
        ASSERT_TRUE(building) << building.error().kind << ": " << building.error().message;
    }
    ASSERT_TRUE(fs::is_regular_file(context)) << rel;
    auto session = provider.value()->create_session(context, c);
    ASSERT_TRUE(session) << session.error().kind << ": " << session.error().message;

    const std::vector<Row> runs = reference_runs(rel, device);
    ASSERT_EQ(runs.size(), static_cast<std::size_t>(1 + 8 / batch)) << rel;
    int identical = 0;
    for (const Row& r : runs) {
        const std::vector<std::byte> y = run_on(*session.value(), run_inputs(r));
        const std::vector<std::byte> ref = read_bytes(r.file);
        ASSERT_EQ(y.size(), ref.size()) << r.file;
        const bool same = y == ref;
        identical += same;
        const Difference d = difference(y, ref);
        EXPECT_TRUE(same) << rel << " (cached) " << r.input_set << " run " << r.run << ": "
                          << d.elements << " of " << ref.size() / sizeof(float)
                          << " elements differ from Python, max abs diff " << max_abs_diff(y, ref)
                          << ", max " << d.max_ulp << " ULP";
    }
    std::printf("[ reference ] %s (%s, from an EP context): %d/%zu runs bit-identical\n",
                rel.c_str(), device.c_str(), identical, runs.size());
    ::testing::Test::RecordProperty("bit_identical_runs", identical);
}

std::string cached_param_name(const ::testing::TestParamInfo<Param>& info) {
    return std::get<0>(info.param) + "_" + std::get<1>(info.param) + "_b" +
           std::to_string(std::get<2>(info.param));
}

}  // namespace

// A session created from an EP context built by the engine computes exactly
// the saved NPU references, like a compiled session.
TEST_P(ReferenceCachedNpu, SessionFromAContextMatchesPythonBitForBit) {
    if (const std::optional<std::string> reason = skip_reason())
        GTEST_SKIP() << *reason;
    const std::string suite =
        ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
    if (suite.find("Manual") != std::string::npos && !all_cached_requested())
        GTEST_SKIP() << "set " << kAllCachedEnv << "=1 to check every file from a context";
    check_cached_session(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Subset, ReferenceCachedNpu,
                         ::testing::Values(Param{"mobilenetv2", "int8", 1},
                                           Param{"mobilenetv2", "int8", 8},
                                           Param{"efficientnet_lite0", "fp32", 1},
                                           Param{"conv1d_autoencoder", "int8", 1},
                                           Param{"conv1d_autoencoder", "fp32", 1}),
                         cached_param_name);

INSTANTIATE_TEST_SUITE_P(AllFilesManual, ReferenceCachedNpu,
                         ::testing::Combine(::testing::ValuesIn(kModelNames),
                                            ::testing::ValuesIn(kNpuVariants),
                                            ::testing::ValuesIn(kBatches)),
                         cached_param_name);

INSTANTIATE_TEST_SUITE_P(ImportedFilesManual, ReferenceCachedNpu,
                         ::testing::Combine(::testing::ValuesIn(kImportedModelNames),
                                            ::testing::ValuesIn(kImportedNpuVariants),
                                            ::testing::ValuesIn(kBatches)),
                         cached_param_name);

TEST(ReferenceModelsNpu, IndexListsExactlyTheseFiles) {
    if (!fs::is_regular_file(kReferences / "index.tsv"))
        GTEST_SKIP() << "no reference outputs; run scripts/make_references.py";
    std::set<std::string> indexed, expected;
    for (const Row& r : read_index())
        if (r.kind == "output" && r.device.starts_with("npu"))
            indexed.insert(r.model + " " + r.device);
    for (const std::string& m : kModelNames)
        for (const std::string& v : kNpuVariants)
            for (int b : kBatches)
                expected.insert(relative_model(m, v, b) + " " + npu_device(v));
    for (const std::string& m : kImportedModelNames)
        for (const std::string& v : kImportedNpuVariants)
            for (int b : kBatches)
                expected.insert(relative_model(m, v, b) + " " + npu_device(v));
    for (const std::string& m : kModelNames)
        for (const std::string& v : kFinalizationVariants)
            for (int b : kBatches)
                expected.insert(relative_model(m, v, b) + " " + kFinalizationDevice);
    EXPECT_EQ(indexed, expected);
}

namespace {

std::string param_name(const ::testing::TestParamInfo<Param>& info) {
    return std::get<0>(info.param) + "_" + std::get<1>(info.param) + "_b" +
           std::to_string(std::get<2>(info.param));
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(AllFiles, ReferenceModels,
                         ::testing::Combine(::testing::ValuesIn(kModelNames),
                                            ::testing::ValuesIn(kVariants),
                                            ::testing::ValuesIn(kBatches)),
                         param_name);

INSTANTIATE_TEST_SUITE_P(ImportedFiles, ReferenceModels,
                         ::testing::Combine(::testing::ValuesIn(kImportedModelNames),
                                            ::testing::ValuesIn(kImportedVariants),
                                            ::testing::ValuesIn(kBatches)),
                         param_name);

TEST(ReferenceModels, IndexListsExactlyTheseFiles) {
    if (!fs::is_regular_file(kReferences / "index.tsv"))
        GTEST_SKIP() << "no reference outputs; run scripts/make_references.py";
    std::set<std::string> indexed, expected;
    for (const Row& r : read_index())
        if (r.kind == "output" && r.device == "cpu")
            indexed.insert(r.model);
    for (const std::string& m : kModelNames)
        for (const std::string& v : kVariants)
            for (int b : kBatches)
                expected.insert(relative_model(m, v, b));
    for (const std::string& m : kImportedModelNames)
        for (const std::string& v : kImportedVariants)
            for (int b : kBatches)
                expected.insert(relative_model(m, v, b));
    EXPECT_EQ(indexed, expected);
}
