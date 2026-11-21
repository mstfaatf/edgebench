#include "edgebench/version.hpp"

#include <gtest/gtest.h>

#include "edgebench/ort_loader.hpp"

// EDGEBENCH_EXPECTED_VERSION and EDGEBENCH_EXPECTED_ORT_VERSION are passed in by
// CMake from the project version and the ONNX Runtime package found at configure time.

TEST(Version, MatchesCMakeProjectVersion) {
    const std::string_view v = edgebench::version();
    ASSERT_FALSE(v.empty());
    EXPECT_EQ(v, EDGEBENCH_EXPECTED_VERSION);
}

TEST(Version, ExpectedOnnxRuntimeVersionMatchesPackage) {
    EXPECT_EQ(edgebench::expected_onnxruntime_version(), EDGEBENCH_EXPECTED_ORT_VERSION);
    EXPECT_TRUE(edgebench::parse_version(edgebench::expected_onnxruntime_version()).has_value());
}

TEST(Version, ExpectedApiVersionComesFromHeaders) {
    EXPECT_GT(edgebench::expected_onnxruntime_api_version(), 0);
}
