// What the QNN EP reports about its backend and the NPU (runtime.qnn), the cache
// key that now carries the SDK version, and the profiled accelerator pass
// (timing.profiled_pass). Every suite ends in "Npu" (label `npu`): the tests need
// the QNN EP, its backend and an NPU, and skip where there is none
// (npu_skip_reason()). They use a committed fixture model and temporary
// directories only.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "edgebench/benchmark.hpp"
#include "edgebench/ep_context_cache.hpp"
#include "edgebench/qnn_backend_info.hpp"
#include "edgebench/qnn_ep.hpp"
#include "edgebench/result_files.hpp"
#include "edgebench/result_json.hpp"
#include "process_util.hpp"
#include "test_runtime.hpp"
#include "validator.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using eb::test::shared_runtime;
using eb::test::TempDir;
using json = nlohmann::ordered_json;

namespace {

const fs::path kModel = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred() / "engine" / "tests" /
                        "fixtures" / "models" / "fixed_float32.onnx";

eb::RunConfig npu_config() {
    eb::RunConfig c;
    c.model = kModel;
    c.provider = eb::ProviderKind::npu;
    c.npu_fp16 = true;  // FP32 fixture: FP16 compute on the NPU
    c.trials = 2;
    c.warmup = 2;
    c.iterations = 5;
    c.cooldown_ms = 0;
    c.load_sample_ms = 0;
    return c;
}

std::string failure(const eb::BenchmarkResult& r) {
    return r.failure ? r.failure->error.kind + ": " + r.failure->error.message : std::string();
}

bool has_warning_at(const eb::BenchmarkResult& r, const std::string& code,
                    const std::string& path) {
    return std::any_of(r.warnings.begin(), r.warnings.end(), [&](const eb::ResultWarning& w) {
        return w.code == code && w.path == path;
    });
}

bool has_warning(const eb::BenchmarkResult& r, const std::string& code) {
    return std::any_of(r.warnings.begin(), r.warnings.end(),
                       [&](const eb::ResultWarning& w) { return w.code == code; });
}

eb::WrittenResult write_and_validate(const eb::BenchmarkResult& r, const fs::path& dir) {
    eb::Result<eb::WrittenResult> written = eb::write_result_files(r, dir, "backend_info_test");
    EXPECT_TRUE(written.ok()) << written.error().message;
    if (!written)
        return {};
    eb::test::validate_or_skip(written.value().result_file);
    return std::move(written).value();
}

class NpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        const eb::test::QnnNpuProbe probe = eb::test::probe_qnn_npu();
        if (const std::optional<std::string> reason = eb::test::npu_skip_reason(probe))
            GTEST_SKIP() << *reason;
    }
    void TearDown() override {
        const eb::Result<void> r = shared_runtime().unregister_ep_library(eb::kQnnEpName);
        EXPECT_TRUE(r.ok()) << r.error().message;
    }
    TempDir work_{L"backend_info"};
    fs::path cache_ = work_.path() / "cache";
};

class QnnBackendInfoNpu : public NpuTest {};
class ProfiledPassNpu : public NpuTest {};

// The timing block without what only the profiled pass adds, and without the
// measured values: the structure of the timed results.
json timed_structure(const eb::BenchmarkResult& r) {
    json t = eb::to_result_json(r, std::nullopt).at("timing");
    t.erase("profiled_pass");
    std::function<void(json&)> blank = [&blank](json& j) {
        if (j.is_object() || j.is_array()) {
            for (json& v : j)
                blank(v);
        } else if (j.is_number()) {
            j = 0;
        }
    };
    blank(t);
    return t;
}

int module_checks_at(const eb::BenchmarkResult& r, const std::string& stage) {
    return static_cast<int>(
        std::count_if(r.execution.module_checks.begin(), r.execution.module_checks.end(),
                      [&stage](const eb::ModuleCheckpoint& c) { return c.stage == stage; }));
}

}  // namespace

TEST_F(QnnBackendInfoNpu, SdkVersionAndHtpArchComeFromTheEpAtRunTime) {
    eb::RunConfig c = npu_config();
    c.ep_cache = eb::EpCacheMode::use;
    c.ep_cache_dir = cache_;
    const eb::BenchmarkResult r = eb::run_benchmark(shared_runtime(), c);
    ASSERT_FALSE(r.failure) << failure(r);
    ASSERT_TRUE(r.runtime.provider.qnn);
    const eb::QnnBackendInfo& q = *r.runtime.provider.qnn;

    // The backend build id, from the EP's backend setup log, and the SDK version in it.
    ASSERT_TRUE(q.backend_build);
    EXPECT_TRUE(std::regex_match(*q.backend_build, std::regex(R"(v\d+\.\d+\.\d+\.\d+)")))
        << *q.backend_build;
    ASSERT_TRUE(q.sdk_version);
    EXPECT_EQ(eb::sdk_version_from_build_id(*q.backend_build), q.sdk_version);
    EXPECT_EQ(q.backend_build_source, eb::kSourceBackendBuildIdLog);
    EXPECT_EQ(q.sdk_version_source, eb::kSourceBackendBuildIdLog);
    // The backend file loaded in the process has the same version.
    ASSERT_TRUE(r.runtime.qnn_backend_file_version);
    EXPECT_EQ(r.runtime.qnn_backend_file_version->product_version, q.backend_build->substr(1));

    // The HTP arch, from the compatibility info of the context the run built.
    ASSERT_TRUE(q.htp_arch);
    EXPECT_TRUE(std::regex_match(*q.htp_arch, std::regex(R"(v\d+)"))) << *q.htp_arch;
    EXPECT_EQ(q.htp_arch_source, eb::kSourceEpCompatibilityInfo);
    ASSERT_TRUE(r.ep_cache.compatibility && r.ep_cache.compatibility->info);
    const auto info = eb::parse_qnn_compatibility_info(*r.ep_cache.compatibility->info);
    ASSERT_TRUE(info) << *r.ep_cache.compatibility->info;
    EXPECT_EQ(info->sdk_version, q.sdk_version);
    EXPECT_EQ(eb::htp_arch_name(*info), q.htp_arch);

    // The SoC model is never reported; only it stays value_unavailable.
    EXPECT_FALSE(q.soc_model);
    EXPECT_FALSE(q.soc_model_source);
    EXPECT_TRUE(has_warning_at(r, "value_unavailable", "/runtime/qnn/soc_model"));
    for (const char* filled :
         {"/runtime/qnn/sdk_version", "/runtime/qnn/backend_build", "/runtime/qnn/htp_arch"})
        EXPECT_FALSE(has_warning_at(r, "value_unavailable", filled)) << filled;

    // The SDK version is a cache key field now.
    ASSERT_TRUE(r.ep_cache.key);
    EXPECT_EQ(r.ep_cache.key->at("qnn_sdk_version"), *q.sdk_version);
    write_and_validate(r, work_.path() / "out");
}

TEST_F(QnnBackendInfoNpu, WithoutAContextTheHtpArchIsUnavailable) {
    const eb::BenchmarkResult r = eb::run_benchmark(shared_runtime(), npu_config());
    ASSERT_FALSE(r.failure) << failure(r);
    const eb::QnnBackendInfo& q = *r.runtime.provider.qnn;
    EXPECT_TRUE(q.sdk_version);
    EXPECT_TRUE(q.backend_build);
    EXPECT_FALSE(q.htp_arch);  // never derived from a file or module name
    EXPECT_FALSE(q.htp_arch_source);
    EXPECT_TRUE(has_warning_at(r, "value_unavailable", "/runtime/qnn/htp_arch"));
    EXPECT_FALSE(r.ep_cache.compatibility);
    const json doc = eb::to_result_json(r, std::nullopt);
    EXPECT_TRUE(doc.at("runtime").at("qnn").at("value_sources").at("htp_arch").is_null());
    EXPECT_TRUE(doc.at("config").at("ep_context_cache").at("compatibility").is_null());
    write_and_validate(r, work_.path() / "out");
}

// Entries built before the SDK version was a key field have another key id for
// the same model: the next run deletes them as stale and records their files.
TEST_F(QnnBackendInfoNpu, EntriesWithoutTheSdkVersionInTheirKeyAreDeleted) {
    eb::RunConfig c = npu_config();
    c.ep_cache = eb::EpCacheMode::build;
    c.ep_cache_dir = cache_;
    const eb::BenchmarkResult built = eb::run_benchmark(shared_runtime(), c);
    ASSERT_FALSE(built.failure) << failure(built);
    ASSERT_TRUE(built.ep_cache.key && built.ep_cache.key_id);

    // The same entry under the key it had without the SDK version.
    std::optional<eb::CacheKeyFields> old_fields = eb::cache_key_from_json(*built.ep_cache.key);
    ASSERT_TRUE(old_fields);
    ASSERT_TRUE(old_fields->qnn_sdk_version);
    old_fields->qnn_sdk_version.reset();
    const std::string old_id = eb::cache_key_id(*old_fields);
    ASSERT_NE(old_id, *built.ep_cache.key_id);
    const fs::path old_dir = cache_ / old_id;
    fs::create_directory(old_dir);
    for (const eb::CacheFile& f : built.ep_cache.files)
        fs::copy_file(cache_ / *built.ep_cache.key_id / f.name, old_dir / f.name);
    const eb::Result<eb::CacheManifest> manifest =
        eb::write_cache_manifest(old_dir, old_id, eb::cache_key_json(*old_fields),
                                 built.ep_cache.files.front().name, "2026-11-25T00:00:00.000Z");
    ASSERT_TRUE(manifest.ok()) << manifest.error().message;

    c.ep_cache = eb::EpCacheMode::use;
    const eb::BenchmarkResult used = eb::run_benchmark(shared_runtime(), c);
    ASSERT_FALSE(used.failure) << failure(used);
    EXPECT_EQ(used.ep_cache.state, "loaded");  // the current entry is still valid
    EXPECT_EQ(used.ep_cache.key_id, built.ep_cache.key_id);
    ASSERT_EQ(used.ep_cache.deleted_stale.size(), 3u);  // two context files and the manifest
    for (const fs::path& p : used.ep_cache.deleted_stale)
        EXPECT_EQ(p.parent_path(), old_dir);
    EXPECT_FALSE(fs::exists(old_dir));
    const json doc = eb::to_result_json(used, std::nullopt);
    EXPECT_EQ(doc.at("execution").at("verification_details").at("deleted_stale_cache_files").size(),
              3u);
    write_and_validate(used, work_.path() / "out");
}

TEST_F(ProfiledPassNpu, IsASeparateSessionThatLeavesTheTimedResultsAlone) {
    const eb::RunConfig plain = npu_config();
    eb::RunConfig profiled = plain;
    profiled.profile_accelerator = true;
    profiled.profile_iterations = 7;
    const eb::BenchmarkResult a = eb::run_benchmark(shared_runtime(), plain);
    const eb::BenchmarkResult b = eb::run_benchmark(shared_runtime(), profiled);
    ASSERT_FALSE(a.failure) << failure(a);
    ASSERT_FALSE(b.failure) << failure(b);

    // The timed results have the same structure, and the pass is not a trial.
    EXPECT_FALSE(a.timing.profiled_pass);
    ASSERT_TRUE(b.timing.profiled_pass);
    EXPECT_EQ(timed_structure(a), timed_structure(b));
    ASSERT_EQ(b.timing.trials.size(), static_cast<std::size_t>(profiled.trials));
    for (const eb::TrialResult& t : b.timing.trials)
        EXPECT_EQ(t.latencies_ms.size(), static_cast<std::size_t>(profiled.iterations));
    EXPECT_EQ(b.timing.session_init.timed_ms.size(), static_cast<std::size_t>(profiled.trials));
    EXPECT_EQ(b.timing.session_init.cold_ms.size(), static_cast<std::size_t>(profiled.trials));
    EXPECT_EQ(b.timing.aggregate->stats.count, profiled.trials * profiled.iterations);
    // The profiled session is checked for foreign modules like every other one.
    EXPECT_EQ(module_checks_at(a, "profiled_session"), 0);
    EXPECT_EQ(module_checks_at(b, "profiled_session"), 1);
    // The timed sessions themselves are never profiled.
    for (const auto& [key, value] : b.runtime.provider.ep_options)
        EXPECT_EQ(key.find("profiling"), std::string::npos) << key;

    const eb::ProfiledPassResult& p = *b.timing.profiled_pass;
    EXPECT_EQ(p.iterations, 7);
    EXPECT_EQ(p.warmup_iterations, profiled.warmup);
    EXPECT_EQ(p.profile.inferences, profiled.warmup + 7);
    ASSERT_EQ(p.accelerator_ms().size(), 7u);
    for (double ms : p.accelerator_ms())
        EXPECT_GT(ms, 0.0);
    std::vector<double> acc = p.accelerator_ms();
    std::sort(acc.begin(), acc.end());
    // Either below the median latency, or flagged.
    EXPECT_TRUE(acc[acc.size() / 2] <= b.timing.aggregate->stats.p50_ms ||
                has_warning(b, "accelerator_time_exceeds_latency"));
    EXPECT_FALSE(p.profile.init_ms.empty());  // graph finalization of the profiled session
    ASSERT_FALSE(p.files.empty());
    EXPECT_TRUE(std::any_of(p.files.begin(), p.files.end(),
                            [](const fs::path& f) { return f.filename() == "qnn_profile.csv"; }));

    // Writing the result moves the profile into the run directory.
    const eb::WrittenResult w = write_and_validate(b, work_.path() / "out");
    ASSERT_TRUE(w.run_dir);
    EXPECT_TRUE(fs::is_regular_file(*w.run_dir / "qnn_profile.csv"));
    EXPECT_FALSE(fs::exists(p.staging_dir));
    const json& files = w.document.at("timing").at("profiled_pass").at("files");
    ASSERT_EQ(files.size(), p.files.size());
    for (const json& f : files) {
        const fs::path rel = fs::path(f.at("path").get<std::string>());
        EXPECT_TRUE(rel.is_relative());
        EXPECT_TRUE(fs::is_regular_file(w.result_file.parent_path() / rel)) << rel;
    }
}

TEST_F(ProfiledPassNpu, FromAnEpContextTheProfileShowsTheBinaryLoad) {
    eb::RunConfig c = npu_config();
    c.ep_cache = eb::EpCacheMode::use;
    c.ep_cache_dir = cache_;
    c.profile_accelerator = true;
    c.profile_iterations = 3;
    const eb::BenchmarkResult r = eb::run_benchmark(shared_runtime(), c);
    ASSERT_FALSE(r.failure) << failure(r);
    ASSERT_TRUE(r.timing.profiled_pass);
    const eb::ProfiledPassResult& p = *r.timing.profiled_pass;
    EXPECT_EQ(p.accelerator_ms().size(), 3u);
    // Created from the same context as the timed sessions: it loads the binary.
    EXPECT_TRUE(std::any_of(p.profile.init_ms.begin(), p.profile.init_ms.end(), [](const auto& e) {
        return e.first.find("(load binary)") != std::string::npos;
    }));
    write_and_validate(r, work_.path() / "out");
}
