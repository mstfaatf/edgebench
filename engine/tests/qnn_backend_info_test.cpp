// What the QNN EP reports at run time, without ONNX Runtime: the backend build id
// from its log, the SDK version read from it, the compatibility info stored in
// EP context models, and the profiling CSV of the profiled accelerator pass (real
// output of a run in engine/tests/fixtures/qnn_profile/).

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "edgebench/qnn_backend_info.hpp"
#include "edgebench/qnn_profile.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;

namespace {

std::string read_fixture(const std::string& name) {
    const fs::path path =
        fs::path(EDGEBENCH_SOURCE_DIR) / "engine" / "tests" / "fixtures" / "qnn_profile" / name;
    std::ifstream in(path, std::ios::binary);
    EXPECT_TRUE(in) << path;
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

}  // namespace

TEST(QnnBackendInfo, BuildIdFromTheBackendSetupLog) {
    // Messages as the Runtime's callback receives them (no prefix), from a real run.
    const std::vector<std::string> messages = {
        "User specified soc_model: 0",
        "QnnHtp.dll interface version: 2.41.0",
        "Found valid interface, version: 2.41.0 backend provider name: HTP_QTI_AISW backend id: 6",
        "Backend build version: v2.52.0.261104115025",
        "InitializeQnnLog succeed.",
    };
    EXPECT_EQ(eb::find_backend_build_id(messages), "v2.52.0.261104115025");
    EXPECT_FALSE(eb::find_backend_build_id({"QnnHtp.dll interface version: 2.41.0"}));
    EXPECT_FALSE(eb::find_backend_build_id({"Backend build version: "}));
    // The last one wins (one per backend setup).
    EXPECT_EQ(eb::find_backend_build_id(
                  {"Backend build version: v2.51.0.1", "Backend build version: v2.52.0.2\n"}),
              "v2.52.0.2");
}

TEST(QnnBackendInfo, SdkVersionIsTheBuildIdsMajorMinorPatch) {
    EXPECT_EQ(eb::sdk_version_from_build_id("v2.52.0.261104115025"), "2.52.0");
    EXPECT_EQ(eb::sdk_version_from_build_id("v10.0.3.1"), "10.0.3");
    for (const char* bad : {"2.52.0.261104115025", "v2.52.0", "v2.52.0.x", "v2..0.1", "", "v"})
        EXPECT_FALSE(eb::sdk_version_from_build_id(bad)) << bad;
}

TEST(QnnBackendInfo, CompatibilityInfoOfARealContext) {
    const auto info = eb::parse_qnn_compatibility_info("v2:6:2.52.0:5.52.0:73:0:0:0");
    ASSERT_TRUE(info);
    EXPECT_EQ(info->backend_id, 6u);
    EXPECT_EQ(info->sdk_version, "2.52.0");
    EXPECT_EQ(info->backend_api_version, "5.52.0");
    EXPECT_EQ(info->htp_archs, std::vector<std::uint32_t>{73});
    EXPECT_EQ(info->soc_models, std::vector<std::uint32_t>{0});
    EXPECT_EQ(info->vtcm_mbs, std::vector<std::uint32_t>{0});
    EXPECT_FALSE(info->htp_user_driver);
    EXPECT_EQ(eb::htp_arch_name(*info), "v73");
    EXPECT_TRUE(eb::recompilation_reason(*info));
}

TEST(QnnBackendInfo, CompatibilityInfoVariants) {
    // A context compiled with vtcm_mb=8: nothing to explain.
    const auto vtcm = eb::parse_qnn_compatibility_info("v2:6:2.52.0:5.52.0:73:0:8:0");
    ASSERT_TRUE(vtcm);
    EXPECT_FALSE(eb::recompilation_reason(*vtcm));
    // Several SoCs (one context for each): no single arch.
    const auto multi = eb::parse_qnn_compatibility_info("v2:6:2.52.0:5.52.0:73,81:60,88:0,0:1");
    ASSERT_TRUE(multi);
    EXPECT_EQ(multi->htp_archs, (std::vector<std::uint32_t>{73, 81}));
    EXPECT_TRUE(multi->htp_user_driver);
    EXPECT_FALSE(eb::htp_arch_name(*multi));
    // Arch 0: unknown.
    const auto unknown = eb::parse_qnn_compatibility_info(eb::kQnnBackendProbeInfo);
    ASSERT_TRUE(unknown);
    EXPECT_FALSE(eb::htp_arch_name(*unknown));
    for (const char* bad : {"", "6:2.52.0:5.52.0:0.0.0:73:0", "v1:6:2.52.0:5.52.0:73:0:0:0",
                            "v2:6:2.52:5.52.0:73:0:0:0", "v2:6:2.52.0:5.52.0:73,81:0:0:0",
                            "v2:6:2.52.0:5.52.0:73:0:0", "v2:x:2.52.0:5.52.0:73:0:0:0",
                            "v2:6:2.52.0:5.52.0:73:0:0:2", "v2:6:2.52.0:5.52.0::0:0:0"})
        EXPECT_FALSE(eb::parse_qnn_compatibility_info(bad)) << bad;
}

TEST(QnnBackendInfo, VerdictNames) {
    EXPECT_EQ(eb::compatibility_verdict_name(0), "not_applicable");
    EXPECT_EQ(eb::compatibility_verdict_name(1), "optimal");
    EXPECT_EQ(eb::compatibility_verdict_name(2), "prefer_recompilation");
    EXPECT_EQ(eb::compatibility_verdict_name(3), "unsupported");
    EXPECT_FALSE(eb::compatibility_verdict_name(4));
}

TEST(QnnProfile, CompiledSessionProfile) {
    // --warmup 2 --profile-iterations 3 on the autoencoder (INT8, batch 1).
    const auto events = eb::parse_qnn_profile_csv(read_fixture("compiled_basic.csv"));
    ASSERT_TRUE(events) << events.error().message;
    const eb::QnnProfileSummary s = eb::summarize_qnn_profile(events.value(), 2);
    EXPECT_EQ(s.inferences, 5);
    const auto& acc = s.execute_ms.at(std::string(eb::kAcceleratorExecuteEvent));
    ASSERT_EQ(acc.size(), 3u);
    for (double ms : acc) {
        EXPECT_GT(ms, 0.0);
        EXPECT_LT(ms, 1.0);
    }
    for (const char* id : {"RPC (execute) time", "QNN accelerator (execute) time",
                           "Accelerator (execute excluding wait) time", "QNN (execute) time"})
        EXPECT_EQ(s.execute_ms.at(id).size(), 3u) << id;
    // Graph finalization while the session was created, in file order.
    ASSERT_EQ(s.init_ms.size(), 4u);
    EXPECT_EQ(s.init_ms[0].first, "RPC (finalize) time");
    EXPECT_EQ(s.init_ms[3].first, "QNN (finalize) time");
    EXPECT_NEAR(s.init_ms[3].second, 70.361, 1e-9);
}

TEST(QnnProfile, ContextSessionProfile) {
    const auto events = eb::parse_qnn_profile_csv(read_fixture("context_basic.csv"));
    ASSERT_TRUE(events) << events.error().message;
    const eb::QnnProfileSummary s = eb::summarize_qnn_profile(events.value(), 2);
    EXPECT_EQ(s.inferences, 5);
    EXPECT_EQ(s.execute_ms.at(std::string(eb::kAcceleratorExecuteEvent)),
              (std::vector<double>{0.031, 0.031, 0.032}));
    // A session from an EP context loads the binary instead of finalizing a graph.
    ASSERT_EQ(s.init_ms.size(), 4u);
    EXPECT_EQ(s.init_ms[3].first, "QNN (load binary) time");
    // Without skipping, the slow first inference is there.
    const eb::QnnProfileSummary all = eb::summarize_qnn_profile(events.value(), 0);
    EXPECT_EQ(all.execute_ms.at(std::string(eb::kAcceleratorExecuteEvent)).front(), 0.347);
}

TEST(QnnProfile, MalformedProfilesAreErrors) {
    EXPECT_FALSE(eb::parse_qnn_profile_csv(""));
    EXPECT_FALSE(eb::parse_qnn_profile_csv("a,b,c\n0,BACKEND,1,US,BACKEND,ROOT,x\n"));
    const std::string header = std::string(eb::kQnnProfileHeader) + "\r\n";
    EXPECT_FALSE(eb::parse_qnn_profile_csv(header + "0,BACKEND,1,US\n"));
    EXPECT_FALSE(eb::parse_qnn_profile_csv(header + "0,BACKEND,fast,US,BACKEND,ROOT,x\n"));
    const auto ok = eb::parse_qnn_profile_csv(header + "0,BACKEND,7,US,BACKEND,ROOT,a, b\r\n");
    ASSERT_TRUE(ok);
    ASSERT_EQ(ok.value().size(), 1u);
    EXPECT_EQ(ok.value()[0].identifier, "a, b");  // commas in the identifier are kept
    const auto bad = eb::parse_qnn_profile_csv("not a profile\n");
    ASSERT_FALSE(bad);
    EXPECT_EQ(bad.error().kind, "qnn_profile_unreadable");
}
