// Unit tests for the provider-independent building blocks: Result/Error, the
// input generator, SHA-256, tensor checks and the node placement parser.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#include "edgebench/input_fill.hpp"
#include "edgebench/placement.hpp"
#include "edgebench/result.hpp"
#include "edgebench/sha256.hpp"
#include "edgebench/tensor_info.hpp"
#include "test_files.hpp"

namespace eb = edgebench;

// --------------------------------------------------------------------------- Result

TEST(Result, HoldsValue) {
    eb::Result<int> r = 42;
    ASSERT_TRUE(r.ok());
    EXPECT_TRUE(static_cast<bool>(r));
    EXPECT_EQ(r.value(), 42);
    EXPECT_THROW((void)r.error(), eb::BadResultAccess);
}

TEST(Result, HoldsError) {
    eb::Result<std::string> r = eb::Error{eb::ExitCode::usage_error, "bad", "it is bad"};
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().exit_code, eb::ExitCode::usage_error);
    EXPECT_EQ(r.error().kind, "bad");
    EXPECT_EQ(r.error().message, "it is bad");
    EXPECT_FALSE(r.error().ort_error_code.has_value());
    EXPECT_THROW((void)r.value(), eb::BadResultAccess);
}

TEST(Result, VoidAndMoveOnly) {
    eb::Result<void> ok;
    EXPECT_TRUE(ok.ok());
    EXPECT_THROW((void)ok.error(), eb::BadResultAccess);
    eb::Result<void> failed = eb::Error{eb::ExitCode::output_error, "x", "y"};
    EXPECT_FALSE(failed.ok());
    EXPECT_EQ(failed.error().exit_code, eb::ExitCode::output_error);

    eb::Result<std::unique_ptr<int>> p = std::make_unique<int>(7);
    std::unique_ptr<int> owned = std::move(p).value();
    EXPECT_EQ(*owned, 7);
}

TEST(Result, OrtError) {
    const eb::Error e =
        eb::ort_error("session_creation_failed", "creating a session", 7, "bad proto");
    EXPECT_EQ(e.exit_code, eb::ExitCode::ort_runtime_error);
    EXPECT_EQ(e.ort_error_code, 7);
    EXPECT_EQ(e.ort_error_name, "ORT_INVALID_PROTOBUF");
    EXPECT_EQ(e.message,
              "creating a session: ONNX Runtime error ORT_INVALID_PROTOBUF (code 7): bad proto");
    EXPECT_STREQ(eb::ort_error_code_name(1), "ORT_FAIL");
    EXPECT_STREQ(eb::ort_error_code_name(999), "ORT_UNKNOWN_ERROR_CODE");
}

// --------------------------------------------------------------------------- inputs

TEST(SplitMix64, KnownSequenceForSeedZero) {
    // Reference values of SplitMix64 seeded with 0.
    eb::SplitMix64 rng(0);
    EXPECT_EQ(rng.next(), 0xE220A8397B1DCDAFull);
    EXPECT_EQ(rng.next(), 0x6E789E6AA1B965F4ull);
    EXPECT_EQ(rng.next(), 0x06C45D188009454Full);
}

TEST(SplitMix64, FloatMappingIsTop24Bits) {
    eb::SplitMix64 a(0), b(0);
    for (int i = 0; i < 1000; ++i) {
        const std::uint64_t bits = a.next();
        const float f = b.next_float();
        EXPECT_EQ(f, static_cast<float>(bits >> 40) / 16777216.0f);
        EXPECT_GE(f, 0.0f);
        EXPECT_LT(f, 1.0f);
    }
    // 0xE220A8397B1DCDAF >> 40 = 0xE220A8 = 14819496.
    EXPECT_EQ(eb::SplitMix64(0).next_float(), 14819496.0f / 16777216.0f);
}

TEST(SplitMix64, SameSeedSameStreamAcrossCalls) {
    std::vector<float> first(1000), second(1000);
    eb::SplitMix64 r1(1234), r2(1234);
    eb::fill_random(first, r1);
    eb::fill_random(second, r2);
    EXPECT_EQ(std::memcmp(first.data(), second.data(), first.size() * sizeof(float)), 0);

    // Filling in two parts continues the same stream.
    std::vector<float> split(1000);
    eb::SplitMix64 r3(1234);
    eb::fill_random(std::span<float>(split).first(300), r3);
    eb::fill_random(std::span<float>(split).subspan(300), r3);
    EXPECT_EQ(std::memcmp(first.data(), split.data(), first.size() * sizeof(float)), 0);

    std::vector<float> other(1000);
    eb::SplitMix64 r4(1235);
    eb::fill_random(other, r4);
    EXPECT_NE(std::memcmp(first.data(), other.data(), first.size() * sizeof(float)), 0);
}

TEST(FillConstant, FillsEveryElement) {
    std::vector<float> v(17, 3.0f);
    eb::fill_constant(v, -0.5f);
    for (float x : v)
        EXPECT_EQ(x, -0.5f);
}

// --------------------------------------------------------------------------- SHA-256

TEST(Sha256, KnownVectors) {
    EXPECT_EQ(eb::sha256_hex(std::string_view("")),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(eb::sha256_hex(std::string_view("abc")),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(eb::sha256_hex(
                  std::string_view("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(Sha256, FileMatchesBuffer) {
    eb::test::TempModel file;  // contains "not a real model"
    const eb::Result<std::string> digest = eb::sha256_file(file.path());
    ASSERT_TRUE(digest.ok());
    EXPECT_EQ(digest.value(), eb::sha256_hex(std::string_view("not a real model")));

    const eb::Result<std::string> missing = eb::sha256_file(file.path().parent_path() / "nope");
    ASSERT_FALSE(missing.ok());
    EXPECT_EQ(missing.error().exit_code, eb::ExitCode::usage_error);
}

TEST(Sha256, LargeFileInChunks) {
    eb::test::TempModel file;
    std::string data(3 * 1024 * 1024 + 17, '\0');
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<char>(i * 31 + 7);
    std::ofstream(file.path(), std::ios::binary)
        .write(data.data(), static_cast<std::streamsize>(data.size()));
    EXPECT_EQ(eb::sha256_file(file.path()).value(), eb::sha256_hex(std::string_view(data)));
}

// --------------------------------------------------------------------------- tensors

namespace {

eb::TensorInfo tensor(std::string name, ONNXTensorElementDataType dtype,
                      std::vector<std::int64_t> shape, std::vector<std::string> params = {}) {
    params.resize(shape.size());
    return {std::move(name), dtype, std::move(shape), std::move(params)};
}

constexpr auto kFloat = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto kInt64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

}  // namespace

TEST(TensorInfo, SizesAndNames) {
    const eb::TensorInfo t = tensor("x", kFloat, {1, 3, 224, 224});
    EXPECT_EQ(t.element_count(), 150528u);
    EXPECT_EQ(t.byte_size(), 602112u);
    EXPECT_EQ(eb::dtype_name(kFloat), "float32");
    EXPECT_EQ(eb::dtype_name(kInt64), "int64");
    EXPECT_EQ(eb::dtype_name(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16), "float16");
    EXPECT_EQ(eb::TensorInfo::dtype_size(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING), 0u);
}

TEST(TensorInfo, FixedShapesPass) {
    EXPECT_TRUE(
        eb::check_fixed_shapes({tensor("x", kFloat, {1, 4})}, {tensor("y", kFloat, {1, 3})}).ok());
    EXPECT_TRUE(eb::check_fixed_shapes({tensor("s", kFloat, {})}, {}).ok());  // scalar
}

TEST(TensorInfo, SymbolicDimensionIsRejected) {
    const auto r = eb::check_fixed_shapes({tensor("x", kFloat, {-1, 4}, {"batch", ""})}, {});
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().exit_code, eb::ExitCode::usage_error);
    EXPECT_EQ(r.error().kind, "dynamic_shape");
    EXPECT_EQ(r.error().message,
              "input 'x' has a dynamic dimension 0 ('batch'); only models with fixed input and "
              "output shapes are supported");
}

TEST(TensorInfo, UnnamedDynamicDimensionIsRejected) {
    const auto r = eb::check_fixed_shapes({tensor("x", kFloat, {1, 3, -1, -1})}, {});
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().message,
              "input 'x' has a dynamic dimension 2; only models with fixed input and output "
              "shapes are supported");
}

TEST(TensorInfo, DynamicOutputIsRejected) {
    const auto r =
        eb::check_fixed_shapes({tensor("x", kFloat, {1, 4})}, {tensor("y", kFloat, {-1}, {"n"})});
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().message.rfind("output 'y' has a dynamic dimension 0 ('n')", 0), 0u)
        << r.error().message;
}

TEST(TensorInfo, OnlyFloat32InputsAreSupported) {
    const auto r = eb::check_supported_dtypes({tensor("ids", kInt64, {1, 4})}, {});
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().exit_code, eb::ExitCode::usage_error);
    EXPECT_EQ(r.error().kind, "unsupported_input_dtype");
    EXPECT_EQ(r.error().message, "input 'ids' has dtype int64; only float32 inputs are supported");
    // Non-float outputs with a fixed element size are fine.
    EXPECT_TRUE(
        eb::check_supported_dtypes({tensor("x", kFloat, {1})}, {tensor("i", kInt64, {1})}).ok());
    const auto s = eb::check_supported_dtypes(
        {tensor("x", kFloat, {1})}, {tensor("s", ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, {1})});
    ASSERT_FALSE(s.ok());
    EXPECT_EQ(s.error().kind, "unsupported_output_dtype");
}

TEST(TensorView, NonFiniteDetection) {
    std::vector<float> v = {0.0f, 1.0f, -2.0f};
    auto view = [&v] { return eb::TensorView{kFloat, std::as_bytes(std::span<const float>(v))}; };
    EXPECT_FALSE(eb::first_non_finite(view()).has_value());
    EXPECT_EQ(view().as_float().size(), 3u);
    v[2] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(eb::first_non_finite(view()), 2u);
    v[2] = 0.0f;
    v[1] = -std::numeric_limits<float>::infinity();
    EXPECT_EQ(eb::first_non_finite(view()), 1u);

    const std::vector<std::uint16_t> half = {0x3C00, 0x7C00};  // 1.0, +inf
    EXPECT_EQ(eb::first_non_finite({ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                                    std::as_bytes(std::span<const std::uint16_t>(half))}),
              1u);
    const std::vector<std::int64_t> ints = {1, 2};  // not floating point
    EXPECT_FALSE(eb::first_non_finite({kInt64, std::as_bytes(std::span<const std::int64_t>(ints))})
                     .has_value());
}

// --------------------------------------------------------------------------- placement

TEST(Placement, AllNodesOnOneEp) {
    const auto p = eb::parse_node_placements(
        {"Some other message", "Node placements",
         " All nodes placed on [CPUExecutionProvider]. Number of nodes: 65"});
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->total_nodes, 65);
    EXPECT_EQ(p->nodes_by_ep, (std::map<std::string, int>{{"CPUExecutionProvider", 65}}));
    EXPECT_TRUE(p->all_on("CPUExecutionProvider"));
    EXPECT_EQ(p->verification_method, "node_assignment");
}

TEST(Placement, SeveralEps) {
    // As logged for a Relu -> Det -> Relu graph with Det unsupported by the QNN EP.
    const auto p = eb::parse_node_placements(
        {"Node placements", " Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1",
         "  Det (unsupported_det)", " Node(s) placed on [QNNExecutionProvider]. Number of nodes: 2",
         "  QNNExecutionProvider_1 (QNNExecutionProvider_1_0)"});
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->total_nodes, 3);
    EXPECT_FALSE(p->all_on("QNNExecutionProvider"));
    EXPECT_EQ(p->nodes_not_on("QNNExecutionProvider"), 1);
    EXPECT_EQ(p->nodes_not_on("CPUExecutionProvider"), 2);
}

TEST(Placement, NoBlockMeansNoResult) {
    EXPECT_FALSE(eb::parse_node_placements({}).has_value());
    EXPECT_FALSE(eb::parse_node_placements({"Node placements"}).has_value());
    // A placement line outside a block is ignored.
    EXPECT_FALSE(eb::parse_node_placements(
                     {" All nodes placed on [CPUExecutionProvider]. Number of nodes: 3"})
                     .has_value());
}
