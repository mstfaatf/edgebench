#include "edgebench/version.hpp"

#include <gtest/gtest.h>

// EDGEBENCH_EXPECTED_VERSION and EDGEBENCH_EXPECTED_ORT_VERSION are passed in by
// CMake from the project version and the ONNX Runtime package found at configure time.

TEST(Version, MatchesCMakeProjectVersion) {
    const std::string_view v = edgebench::version();
    ASSERT_FALSE(v.empty());
    EXPECT_EQ(v, EDGEBENCH_EXPECTED_VERSION);
}

TEST(Version, ReportsLoadedOnnxRuntimeVersion) {
    const std::string ort = edgebench::onnxruntime_version();
    ASSERT_FALSE(ort.empty());
    EXPECT_EQ(ort, EDGEBENCH_EXPECTED_ORT_VERSION);
}
