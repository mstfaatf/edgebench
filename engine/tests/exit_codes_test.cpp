#include "edgebench/exit_codes.hpp"

#include <gtest/gtest.h>

#include <set>
#include <string>

namespace eb = edgebench;

TEST(ExitCodes, DocumentedValues) {
    EXPECT_EQ(eb::to_int(eb::ExitCode::success), 0);
    EXPECT_EQ(eb::to_int(eb::ExitCode::internal_error), 1);
    EXPECT_EQ(eb::to_int(eb::ExitCode::usage_error), 2);
    EXPECT_EQ(eb::to_int(eb::ExitCode::ort_runtime_error), 3);
    EXPECT_EQ(eb::to_int(eb::ExitCode::ort_load_failure), 4);
    EXPECT_EQ(eb::to_int(eb::ExitCode::strict_mode_refusal), 5);
    EXPECT_EQ(eb::to_int(eb::ExitCode::output_error), 6);
}

TEST(ExitCodes, AllCodesInOrder) {
    for (size_t i = 0; i < eb::kAllExitCodes.size(); ++i)
        EXPECT_EQ(eb::to_int(eb::kAllExitCodes[i]), static_cast<int>(i));
}

TEST(ExitCodes, NamesAndDescriptionsAreUniqueAndSet) {
    std::set<std::string> names, descriptions;
    for (eb::ExitCode code : eb::kAllExitCodes) {
        const std::string n(eb::name(code));
        const std::string d(eb::describe(code));
        EXPECT_NE(n, "unknown");
        EXPECT_FALSE(d.empty());
        names.insert(n);
        descriptions.insert(d);
    }
    EXPECT_EQ(names.size(), eb::kAllExitCodes.size());
    EXPECT_EQ(descriptions.size(), eb::kAllExitCodes.size());
    EXPECT_EQ(eb::name(eb::ExitCode::ort_load_failure), "ort_load_failure");
    EXPECT_EQ(eb::describe(eb::ExitCode::usage_error),
              "Usage error: bad arguments, invalid values, or model file not found");
}
