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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <tuple>

#include <nlohmann/json.hpp>

#include "edgebench/provider.hpp"
#include "edgebench/runtime.hpp"
#include "edgebench/sha256.hpp"
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

std::unique_ptr<eb::BenchSession> engine_session(const fs::path& model) {
    eb::RunConfig c;  // the defaults of `edgebench run`: random input, seed 0
    c.model = model;
    c.threads = kThreads;
    c.binding = eb::RunBinding::iobinding_preallocated;
    auto provider = eb::make_provider(eb::test::shared_runtime(), c);
    if (!provider)
        throw std::runtime_error(provider.error().message);
    auto session = provider.value()->create_session(model, c);
    if (!session)
        throw std::runtime_error(session.error().message);
    return std::move(session).value();
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
