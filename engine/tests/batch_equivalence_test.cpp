// Batch equivalence lookup (edgebench/batch_equivalence.hpp) and the module
// helpers used for runtime.qnn (bundle_modules.hpp).
#include <gtest/gtest.h>

#include <windows.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

#include "edgebench/batch_equivalence.hpp"
#include "edgebench/bundle_modules.hpp"
#include "test_files.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using json = nlohmann::json;

namespace {

const std::string kSha4(64, '4');
const std::string kSha8(64, '8');
const eb::EpCompileOptions kDefaults;
const eb::EpCompileOptions kMode1 = {{"htp_graph_finalization_optimization_mode", "1"}};

json entry(const std::string& sha, const char* provider, const char* precision, int batch,
           bool bitwise, json predictions_equal, const char* basis, json compared = nullptr,
           json differing = nullptr, json options = json::object()) {
    return {{"sha256", sha},
            {"provider", provider},
            {"compute_precision", precision},
            {"ep_compile_options", options},
            {"batch_size", batch},
            {"bitwise_equal_to_b1", bitwise},
            {"predictions_equal_to_b1", predictions_equal},
            {"predictions_basis", basis},
            {"predictions_compared", compared},
            {"predictions_differing", differing}};
}

json summary() {
    return json{
        {"batch_equivalence_version", "1.2.0"},
        {"entries", json::array({
                        entry(kSha4, "npu", "int8", 4, true, true, "bitwise"),
                        entry(kSha8, "npu", "int8", 8, false, false, "eval_quick_top1", 1000, 6),
                        entry(kSha8, "npu", "int8", 8, true, true, "bitwise", nullptr, nullptr,
                              json{{"htp_graph_finalization_optimization_mode", "1"}}),
                        entry(kSha8, "cpu", "int8", 8, true, true, "bitwise"),
                        entry(kSha8, "cpu", "fp32", 8, false, true, "eval_quick_top1", 1000, 0),
                        entry(kSha4, "npu", "fp16", 4, false, nullptr, "not_available"),
                    })}};
}

}  // namespace

TEST(BatchEquivalence, BatchOneIsEqualByDefinition) {
    const eb::BatchEquivalence b =
        eb::batch_equivalence_from(json::object(), kSha4, "npu", "int8", kDefaults, 1);
    EXPECT_EQ(b.bitwise_equal_to_b1, true);
    EXPECT_EQ(b.bitwise_basis, "batch_size_1");
    EXPECT_EQ(b.predictions_equal_to_b1, true);
    EXPECT_EQ(b.predictions_basis, "bitwise");
    EXPECT_FALSE(b.rounding_only());
}

TEST(BatchEquivalence, LooksUpByShaProviderAndComputePrecision) {
    const json s = summary();
    const eb::BatchEquivalence b4 = eb::batch_equivalence_from(s, kSha4, "npu", "int8", kDefaults, 4);
    EXPECT_EQ(b4.bitwise_equal_to_b1, true);
    EXPECT_EQ(b4.predictions_basis, "bitwise");
    const eb::BatchEquivalence b8 = eb::batch_equivalence_from(s, kSha8, "npu", "int8", kDefaults, 8);
    EXPECT_EQ(b8.bitwise_equal_to_b1, false);
    EXPECT_EQ(b8.bitwise_basis, "reference_outputs");
    EXPECT_EQ(b8.predictions_equal_to_b1, false);
    EXPECT_EQ(b8.predictions_basis, "eval_quick_top1");
    EXPECT_EQ(b8.predictions_compared, 1000);
    EXPECT_EQ(b8.predictions_differing, 6);
    EXPECT_FALSE(b8.rounding_only());
    EXPECT_EQ(eb::batch_equivalence_from(s, kSha8, "cpu", "int8", kDefaults, 8).bitwise_equal_to_b1, true);
}

TEST(BatchEquivalence, BitsThatDifferWithEqualPredictionsAreRoundingOnly) {
    const eb::BatchEquivalence b = eb::batch_equivalence_from(summary(), kSha8, "cpu", "fp32", kDefaults, 8);
    EXPECT_EQ(b.bitwise_equal_to_b1, false);
    EXPECT_EQ(b.predictions_equal_to_b1, true);
    EXPECT_EQ(b.predictions_differing, 0);
    EXPECT_TRUE(b.rounding_only());
}

TEST(BatchEquivalence, MissingPredictionComparisonKeepsTheBitwiseAnswer) {
    const eb::BatchEquivalence b = eb::batch_equivalence_from(summary(), kSha4, "npu", "fp16", kDefaults, 4);
    EXPECT_EQ(b.bitwise_equal_to_b1, false);
    EXPECT_FALSE(b.predictions_equal_to_b1);
    EXPECT_EQ(b.predictions_basis, "not_available");
    EXPECT_FALSE(b.reason.empty());
}

TEST(BatchEquivalence, UnlistedOrInconsistentEntriesAreNotAvailable) {
    const json s = summary();
    json old_version = s;
    old_version["batch_equivalence_version"] = "1.1.0";
    json bad = s;
    bad["entries"][1]["predictions_differing"] = 0;  // equal_to_b1 false with nothing differing
    json bad_bitwise = s;
    bad_bitwise["entries"][0]["predictions_basis"] = "eval_quick_top1";  // bitwise but counted
    for (const eb::BatchEquivalence& b : {
             eb::batch_equivalence_from(s, kSha4, "cpu", "int8", kDefaults, 4),  // other provider
             eb::batch_equivalence_from(s, kSha4, "npu", "fp32", kDefaults, 4),  // other precision
             eb::batch_equivalence_from(s, std::string(64, 'a'), "npu", "int8", kDefaults, 4),
             eb::batch_equivalence_from(s, kSha4, "npu", "int8", kDefaults, 8),  // wrong batch size
             eb::batch_equivalence_from(json::object(), kSha4, "npu", "int8", kDefaults, 4),
             eb::batch_equivalence_from(old_version, kSha4, "npu", "int8", kDefaults, 4),
             eb::batch_equivalence_from(bad, kSha8, "npu", "int8", kDefaults, 8),
             eb::batch_equivalence_from(bad_bitwise, kSha4, "npu", "int8", kDefaults, 4),
         }) {
        EXPECT_FALSE(b.bitwise_equal_to_b1);
        EXPECT_EQ(b.bitwise_basis, "not_available");
        EXPECT_FALSE(b.predictions_equal_to_b1);
        EXPECT_EQ(b.predictions_basis, "not_available");
        EXPECT_FALSE(b.reason.empty());
    }
}

TEST(BatchEquivalence, EntriesMatchOnlyTheirOwnCompileOptions) {
    const json s = summary();
    // Default options: the default-mode entry (predictions differ at b8).
    const eb::BatchEquivalence defaults = eb::batch_equivalence_from(s, kSha8, "npu", "int8",
                                                                     kDefaults, 8);
    EXPECT_EQ(defaults.predictions_equal_to_b1, false);
    // Finalization mode 1: its own entry (bitwise equal).
    const eb::BatchEquivalence mode1 = eb::batch_equivalence_from(s, kSha8, "npu", "int8", kMode1,
                                                                  8);
    EXPECT_EQ(mode1.bitwise_equal_to_b1, true);
    EXPECT_EQ(mode1.predictions_equal_to_b1, true);
    EXPECT_FALSE(mode1.options_not_covered);
    // Any other combination: not available, never the default-mode answer.
    for (const eb::EpCompileOptions& other :
         {eb::EpCompileOptions{{"htp_graph_finalization_optimization_mode", "2"}},
          eb::EpCompileOptions{{"htp_graph_finalization_optimization_mode", "1"},
                               {"vtcm_mb", "8"}},
          eb::EpCompileOptions{{"vtcm_mb", "8"}}}) {
        const eb::BatchEquivalence b = eb::batch_equivalence_from(s, kSha8, "npu", "int8", other,
                                                                  8);
        EXPECT_FALSE(b.bitwise_equal_to_b1);
        EXPECT_EQ(b.bitwise_basis, "not_available");
        EXPECT_FALSE(b.predictions_equal_to_b1);
        EXPECT_TRUE(b.options_not_covered);
        EXPECT_NE(b.reason.find("not reused"), std::string::npos) << b.reason;
        EXPECT_NE(b.reason.find(eb::describe_compile_options(other)), std::string::npos)
            << b.reason;
    }
    // Options on a file listed only with the defaults: also not covered.
    EXPECT_TRUE(eb::batch_equivalence_from(s, kSha4, "npu", "int8", kMode1, 4).options_not_covered);
    // A file not listed at all is not "options not covered".
    EXPECT_FALSE(eb::batch_equivalence_from(s, std::string(64, 'a'), "npu", "int8", kMode1, 4)
                     .options_not_covered);
    // Batch 1 is equal by definition whatever the options.
    EXPECT_EQ(eb::batch_equivalence_from(s, kSha8, "npu", "int8", kMode1, 1).bitwise_basis,
              "batch_size_1");
}

TEST(BatchEquivalence, AnEntryWithoutCompileOptionsIsInconsistent) {
    json s = summary();
    s["entries"][0].erase("ep_compile_options");
    json not_strings = summary();
    not_strings["entries"][0]["ep_compile_options"] = json{{"vtcm_mb", 8}};
    for (const json& bad : {s, not_strings}) {
        const eb::BatchEquivalence b = eb::batch_equivalence_from(bad, kSha4, "npu", "int8",
                                                                  kDefaults, 4);
        EXPECT_FALSE(b.bitwise_equal_to_b1);
        EXPECT_NE(b.reason.find("inconsistent"), std::string::npos) << b.reason;
    }
}

TEST(BatchEquivalence, DescribesCompileOptions) {
    EXPECT_EQ(eb::describe_compile_options(kDefaults), "the EP defaults");
    EXPECT_EQ(eb::describe_compile_options({{"b", "2"}, {"a", "1"}}), "a=1, b=2");
}

TEST(BatchEquivalence, SummaryFileIsFoundInAParentDirectory) {
    eb::test::TempModel model("m_int8_b8.onnx");
    const fs::path models = model.path().parent_path();
    const fs::path nested = models / "sub" / "m_int8_b8.onnx";
    fs::create_directories(nested.parent_path());
    std::ofstream(nested, std::ios::binary) << "x";
    EXPECT_FALSE(eb::find_batch_equivalence_file(nested) &&
                 eb::find_batch_equivalence_file(nested)->parent_path() == models);
    std::ofstream(models / "batch_equivalence.json", std::ios::binary) << summary().dump();
    ASSERT_EQ(eb::find_batch_equivalence_file(nested), models / "batch_equivalence.json");
    EXPECT_EQ(eb::lookup_batch_equivalence(nested, kSha8, "npu", "int8", kDefaults, 8).predictions_equal_to_b1,
              false);

    std::ofstream(models / "batch_equivalence.json", std::ios::binary | std::ios::trunc) << "{ not";
    const eb::BatchEquivalence broken =
        eb::lookup_batch_equivalence(nested, kSha8, "npu", "int8", kDefaults, 8);
    EXPECT_EQ(broken.bitwise_basis, "not_available");
    EXPECT_NE(broken.reason.find("not a valid JSON object"), std::string::npos) << broken.reason;
}

TEST(ModuleFileVersion, ReadsTheVersionResourceOfASystemDll) {
    wchar_t system[MAX_PATH];
    ASSERT_GT(GetSystemDirectoryW(system, MAX_PATH), 0u);
    const auto v = eb::read_module_file_version(fs::path(system) / L"kernel32.dll");
    ASSERT_TRUE(v);
    ASSERT_TRUE(v->file_version);
    EXPECT_EQ(std::count(v->file_version->begin(), v->file_version->end(), '.'), 3)
        << *v->file_version;
    ASSERT_TRUE(v->product_version);
    EXPECT_FALSE(v->product_version->empty());
    eb::test::TempModel no_resource("plain.dll");
    EXPECT_FALSE(eb::read_module_file_version(no_resource.path()));
}

TEST(ModuleFileVersion, FindsALoadedModuleByNameIgnoringCase) {
    const std::vector<eb::LoadedModule> modules = {{"onnxruntime.dll", "C:/b/onnxruntime.dll"},
                                                   {"QnnHtp.dll", "C:/b/QnnHtp.dll"}};
    ASSERT_NE(eb::find_loaded_module(modules, "qnnhtp.DLL"), nullptr);
    EXPECT_EQ(eb::find_loaded_module(modules, "QnnHtp.dll")->path, fs::path("C:/b/QnnHtp.dll"));
    EXPECT_EQ(eb::find_loaded_module(modules, "QnnHtpPrepare.dll"), nullptr);
}
