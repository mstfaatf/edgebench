// The EP context cache and the initialization breakdown on the NPU: build then
// use, stale entries, a corrupt context, an unwritable cache directory, and
// --init-breakdown. Every suite ends in "Npu" (label `npu`): the tests need the
// QNN EP, its backend and an NPU, and skip where there is none
// (npu_skip_reason()). They use a committed fixture model and temporary cache
// directories only.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "edgebench/benchmark.hpp"
#include "edgebench/ep_context_cache.hpp"
#include "edgebench/qnn_ep.hpp"
#include "edgebench/result_files.hpp"
#include "edgebench/result_json.hpp"
#include "edgebench/sha256.hpp"
#include "process_util.hpp"
#include "test_runtime.hpp"
#include "validator.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using eb::ExitCode;
using eb::test::shared_runtime;
using eb::test::TempDir;
using json = nlohmann::json;

namespace {

const fs::path kModel = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred() / "engine" / "tests" /
                        "fixtures" / "models" / "fixed_float32.onnx";
const fs::path kEdgebenchExe = fs::path(EDGEBENCH_EXE_PATH).make_preferred();

eb::RunConfig cache_config(const fs::path& cache_dir, eb::EpCacheMode mode) {
    eb::RunConfig c;
    c.model = kModel;
    c.provider = eb::ProviderKind::npu;
    c.npu_fp16 = true;  // FP32 fixture: FP16 compute on the NPU
    c.trials = 2;
    c.warmup = 2;
    c.iterations = 5;
    c.cooldown_ms = 0;
    c.load_sample_ms = 0;
    c.ep_cache = mode;
    c.ep_cache_dir = cache_dir;
    return c;
}

void write_and_validate(const eb::BenchmarkResult& r, const TempDir& dir) {
    const eb::Result<eb::WrittenResult> written =
        eb::write_result_files(r, dir.path(), "ep_cache_test");
    ASSERT_TRUE(written.ok()) << written.error().message;
    eb::test::validate_or_skip(written.value().result_file);
}

std::string failure(const eb::BenchmarkResult& r) {
    return r.failure ? r.failure->error.kind + ": " + r.failure->error.message : std::string();
}

std::vector<fs::path> entries(const fs::path& cache_dir) {
    std::vector<fs::path> out;
    for (const fs::directory_entry& e : fs::directory_iterator(cache_dir))
        out.push_back(e.path());
    return out;
}

bool has_note(const eb::BenchmarkResult& r, const std::string& code) {
    return std::any_of(r.notes.begin(), r.notes.end(),
                       [&code](const eb::ResultWarning& n) { return n.code == code; });
}

class EpCacheNpu : public ::testing::Test {
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
    TempDir work_{L"ep_cache"};
    fs::path cache_ = work_.path() / "cache";
};

}  // namespace

TEST_F(EpCacheNpu, BuildThenUseHitsTheCacheAndVerifiesTheOutputs) {
    const eb::BenchmarkResult built =
        eb::run_benchmark(shared_runtime(), cache_config(cache_, eb::EpCacheMode::build));
    ASSERT_FALSE(built.failure) << failure(built);
    const eb::EpContextCacheResult& b = built.ep_cache;
    EXPECT_EQ(b.state, "generated");
    EXPECT_EQ(b.hit, false);
    EXPECT_EQ(b.verified_equal_outputs, true);
    ASSERT_TRUE(b.key_id && b.context_path && b.cache_dir);
    EXPECT_TRUE(b.cache_dir->is_absolute());
    EXPECT_EQ(b.context_path->parent_path(), *b.cache_dir / *b.key_id);
    EXPECT_TRUE(fs::is_regular_file(*b.context_path));
    ASSERT_EQ(b.files.size(), 2u);  // embed mode 0: the context model and its binary
    EXPECT_EQ(b.files[0].name, "fixed_float32_ctx.onnx");
    EXPECT_EQ(b.files[1].name, "fixed_float32_ctx_qnn.bin");
    EXPECT_TRUE(fs::is_regular_file(b.context_path->parent_path() / eb::kEpCacheManifestName));
    ASSERT_TRUE(built.timing.session_init.context_build_ms);
    EXPECT_GT(*built.timing.session_init.context_build_ms, 0.0);
    EXPECT_EQ(built.timing.session_init.ep_context_state, "cached");
    EXPECT_EQ(built.runtime.provider.session_config.at("ep.context_embed_mode"), "0");
    for (const eb::TrialResult& t : built.timing.trials) {
        EXPECT_FALSE(t.session_init.cold_ms);
        ASSERT_TRUE(t.session_init.cached_init_ms);
        EXPECT_EQ(*t.session_init.cached_init_ms, t.session_create_ms);
    }
    // Cached sessions load the QNN system library, which therefore must be bundled.
    EXPECT_TRUE(std::any_of(
        built.runtime.loaded_modules.begin(), built.runtime.loaded_modules.end(),
        [](const eb::LoadedModule& m) { return _stricmp(m.name.c_str(), "QnnSystem.dll") == 0; }));
    write_and_validate(built, work_);

    const eb::BenchmarkResult used =
        eb::run_benchmark(shared_runtime(), cache_config(cache_, eb::EpCacheMode::use));
    ASSERT_FALSE(used.failure) << failure(used);
    const eb::EpContextCacheResult& u = used.ep_cache;
    EXPECT_EQ(u.state, "loaded");
    EXPECT_EQ(u.hit, true);
    EXPECT_EQ(u.verified_equal_outputs, true);
    EXPECT_EQ(u.key_id, b.key_id);
    EXPECT_EQ(u.files, b.files);  // the same files, not rebuilt
    EXPECT_FALSE(used.timing.session_init.context_build_ms);
    // The timed sessions of both runs compute the same outputs.
    write_and_validate(used, work_);
    EXPECT_EQ(entries(cache_).size(), 1u);
}

TEST_F(EpCacheNpu, UseBuildsAMissingCacheOnce) {
    const eb::BenchmarkResult r =
        eb::run_benchmark(shared_runtime(), cache_config(cache_, eb::EpCacheMode::use));
    ASSERT_FALSE(r.failure) << failure(r);
    EXPECT_EQ(r.ep_cache.hit, false);
    EXPECT_EQ(r.ep_cache.state, "generated");
    EXPECT_TRUE(r.timing.session_init.context_build_ms);
    write_and_validate(r, work_);
}

// Another EP option is another configuration of the same model: it gets its own
// entry, and the first one survives (and is still a hit afterwards).
TEST_F(EpCacheNpu, AnotherOptionOfTheSameModelKeepsTheFirstEntry) {
    eb::RunConfig first = cache_config(cache_, eb::EpCacheMode::build);
    const eb::BenchmarkResult a = eb::run_benchmark(shared_runtime(), first);
    ASSERT_FALSE(a.failure) << failure(a);
    eb::RunConfig second = first;
    second.htp_performance_mode = "balanced";  // an EP option: part of the key
    second.ep_cache = eb::EpCacheMode::use;
    const eb::BenchmarkResult b = eb::run_benchmark(shared_runtime(), second);
    ASSERT_FALSE(b.failure) << failure(b);
    EXPECT_NE(a.ep_cache.key_id, b.ep_cache.key_id);
    EXPECT_EQ(b.ep_cache.hit, false);
    EXPECT_EQ(b.ep_cache.state, "generated");
    EXPECT_TRUE(b.ep_cache.deleted_stale.empty());
    EXPECT_EQ(entries(cache_).size(), 2u);
    EXPECT_TRUE(fs::exists(cache_ / *a.ep_cache.key_id / eb::kEpCacheManifestName));

    eb::RunConfig again = first;
    again.ep_cache = eb::EpCacheMode::use;
    const eb::BenchmarkResult c = eb::run_benchmark(shared_runtime(), again);
    ASSERT_FALSE(c.failure) << failure(c);
    EXPECT_EQ(c.ep_cache.key_id, a.ep_cache.key_id);
    EXPECT_EQ(c.ep_cache.hit, true);
    EXPECT_EQ(c.ep_cache.state, "loaded");
    write_and_validate(b, work_);
    write_and_validate(c, work_);
}

TEST_F(EpCacheNpu, UseWithATruncatedContextExits3WithoutRebuilding) {
    const eb::BenchmarkResult built =
        eb::run_benchmark(shared_runtime(), cache_config(cache_, eb::EpCacheMode::build));
    ASSERT_FALSE(built.failure) << failure(built);
    const fs::path bin = built.ep_cache.context_path->parent_path() / "fixed_float32_ctx_qnn.bin";
    const auto size = fs::file_size(bin);
    fs::resize_file(bin, size / 2);

    const eb::BenchmarkResult r =
        eb::run_benchmark(shared_runtime(), cache_config(cache_, eb::EpCacheMode::use));
    ASSERT_TRUE(r.failure);
    EXPECT_EQ(r.failure->error.exit_code, ExitCode::ort_runtime_error);
    EXPECT_EQ(r.failure->error.kind, "ep_context_corrupt");
    EXPECT_NE(r.failure->error.message.find("--ep-cache build"), std::string::npos)
        << r.failure->error.message;
    EXPECT_EQ(fs::file_size(bin), size / 2);  // not silently rebuilt
    EXPECT_TRUE(r.timing.trials.empty());
    write_and_validate(r, work_);
}

TEST_F(EpCacheNpu, UseWithAContextOnnxRuntimeCannotLoadExits3) {
    const eb::BenchmarkResult built =
        eb::run_benchmark(shared_runtime(), cache_config(cache_, eb::EpCacheMode::build));
    ASSERT_FALSE(built.failure) << failure(built);
    // Damage the context binary and record the damaged file in the manifest, so
    // that only ONNX Runtime can tell.
    const fs::path entry = built.ep_cache.context_path->parent_path();
    const fs::path bin = entry / "fixed_float32_ctx_qnn.bin";
    {
        std::fstream f(bin, std::ios::binary | std::ios::in | std::ios::out);
        const std::string junk(256, '\x5a');
        f.seekp(0);
        f.write(junk.data(), static_cast<std::streamsize>(junk.size()));
    }
    const eb::Result<eb::CacheManifest> m =
        eb::write_cache_manifest(entry, *built.ep_cache.key_id, *built.ep_cache.key,
                                 "fixed_float32_ctx.onnx", "2026-11-26T00:00:00.000Z");
    ASSERT_TRUE(m);

    const eb::BenchmarkResult r =
        eb::run_benchmark(shared_runtime(), cache_config(cache_, eb::EpCacheMode::use));
    ASSERT_TRUE(r.failure);
    EXPECT_EQ(r.failure->error.exit_code, ExitCode::ort_runtime_error) << failure(r);
    EXPECT_EQ(r.failure->error.kind, "ep_context_load_failed") << failure(r);
    EXPECT_NE(r.failure->error.message.find("--ep-cache build"), std::string::npos);
    EXPECT_EQ(r.ep_cache.hit, true);
    EXPECT_FALSE(r.ep_cache.verified_equal_outputs);
    write_and_validate(r, work_);
}

TEST_F(EpCacheNpu, UnwritableCacheDirectoryExits6) {
    const fs::path file = work_.path() / "a_file";
    std::ofstream(file) << "x";
    const eb::BenchmarkResult r =
        eb::run_benchmark(shared_runtime(), cache_config(file / "cache", eb::EpCacheMode::use));
    ASSERT_TRUE(r.failure);
    EXPECT_EQ(r.failure->error.exit_code, ExitCode::output_error);
    EXPECT_EQ(r.failure->error.kind, "ep_cache_dir_not_writable");
    EXPECT_EQ(r.failure->stage, "configuration");
    write_and_validate(r, work_);
}

TEST_F(EpCacheNpu, ADirectoryWithoutAManifestIsNeverTouched) {
    const eb::BenchmarkResult built =
        eb::run_benchmark(shared_runtime(), cache_config(cache_, eb::EpCacheMode::build));
    ASSERT_FALSE(built.failure) << failure(built);
    const fs::path entry = built.ep_cache.context_path->parent_path();
    fs::remove(entry / eb::kEpCacheManifestName);  // now it is not an entry edgebench wrote
    const eb::BenchmarkResult r =
        eb::run_benchmark(shared_runtime(), cache_config(cache_, eb::EpCacheMode::build));
    ASSERT_TRUE(r.failure);
    EXPECT_EQ(r.failure->error.exit_code, ExitCode::output_error);
    EXPECT_EQ(r.failure->error.kind, "ep_cache_entry_foreign");
    EXPECT_TRUE(fs::is_regular_file(entry / "fixed_float32_ctx_qnn.bin"));
}

TEST_F(EpCacheNpu, InitBreakdownMeasuresEveryKindInEveryTrial) {
    for (eb::EpCacheMode mode : {eb::EpCacheMode::off, eb::EpCacheMode::use}) {
        eb::RunConfig c = cache_config(cache_, mode);
        c.init_breakdown = true;
        c.cooldown_ms = 50;
        const eb::BenchmarkResult r = eb::run_benchmark(shared_runtime(), c);
        ASSERT_FALSE(r.failure) << failure(r);
        ASSERT_EQ(r.timing.trials.size(), 2u);
        for (const eb::TrialResult& t : r.timing.trials) {
            ASSERT_TRUE(t.session_init.cold_ms && t.session_init.context_build_ms &&
                        t.session_init.cached_init_ms);
            EXPECT_GT(*t.session_init.cold_ms, 0.0);
            EXPECT_GT(*t.session_init.context_build_ms, 0.0);
            EXPECT_GT(*t.session_init.cached_init_ms, 0.0);
        }
        const json doc = eb::to_result_json(r, std::nullopt);
        const json& summary = doc.at("timing").at("session_init").at("summary");
        for (const char* kind : {"cold_ms", "context_build_ms", "cached_init_ms"}) {
            EXPECT_EQ(summary.at(kind).at("count"), 2) << kind;
            EXPECT_GT(summary.at(kind).at("median_ms").get<double>(), 0.0) << kind;
        }
        EXPECT_EQ(doc.at("config").at("init_breakdown"), true);
        EXPECT_EQ(r.timing.session_init.ep_context_state,
                  mode == eb::EpCacheMode::off ? "cold" : "cached");
        // The per-trial scratch contexts are gone; only a shared cache entry remains.
        EXPECT_EQ(entries(cache_).size(), mode == eb::EpCacheMode::off ? 0u : 1u);
        // Every untimed session's modules were checked.
        const auto untimed = std::count_if(
            r.execution.module_checks.begin(), r.execution.module_checks.end(),
            [](const eb::ModuleCheckpoint& m) { return m.stage == "after_untimed_session"; });
        EXPECT_EQ(untimed, (mode == eb::EpCacheMode::off ? 0 : 3) + 2 * 3);
        write_and_validate(r, work_);
    }
}

TEST_F(EpCacheNpu, LoadingAContextIsNotedWhenOnnxRuntimeSuggestsRecompiling) {
    const eb::BenchmarkResult r =
        eb::run_benchmark(shared_runtime(), cache_config(cache_, eb::EpCacheMode::use));
    ASSERT_FALSE(r.failure) << failure(r);
    // Whatever ONNX Runtime said about the context models is a note, never a warning.
    for (const eb::ResultWarning& w : r.warnings)
        EXPECT_NE(w.code, "ep_context_recompilation_suggested");
    if (has_note(r, "ep_context_recompilation_suggested"))
        SUCCEED() << "ONNX Runtime suggested recompiling the context models";
}

// The executable: exit codes of the cache errors, with a failed result written.
TEST_F(EpCacheNpu, ExeExitCodesForCacheErrors) {
    const fs::path out = work_.path() / "out";
    const fs::path file = work_.path() / "not_a_dir";
    std::ofstream(file) << "x";
    std::vector<std::wstring> args = {
        L"run",           L"--model",    kModel.native(),     L"--provider",
        L"npu",           L"--npu-fp16", L"--trials",         L"1",
        L"--warmup",      L"1",          L"--iterations",     L"2",
        L"--cooldown-ms", L"0",          L"--load-sample-ms", L"0",
        L"--output-dir",  out.native(),  L"--ep-cache",       L"use",
        L"--ep-cache-dir"};
    std::vector<std::wstring> unwritable = args;
    unwritable.push_back((file / "cache").native());
    const eb::test::ProcessResult r6 =
        eb::test::run_process(kEdgebenchExe, unwritable, work_.path());
    EXPECT_EQ(r6.exit_code, 6u) << r6.output;
    EXPECT_NE(r6.err.find("[ep_cache_dir_not_writable]"), std::string::npos) << r6.err;
    EXPECT_NE(r6.err.find("--ep-cache-dir"), std::string::npos) << r6.err;

    std::vector<std::wstring> good = args;
    good.push_back(cache_.native());
    const eb::test::ProcessResult ok = eb::test::run_process(kEdgebenchExe, good, work_.path());
    ASSERT_EQ(ok.exit_code, 0u) << ok.output;
    for (const fs::directory_entry& e : fs::directory_iterator(cache_)) {
        if (e.is_directory())
            fs::resize_file(e.path() / "fixed_float32_ctx_qnn.bin", 10);
    }
    const eb::test::ProcessResult r3 = eb::test::run_process(kEdgebenchExe, good, work_.path());
    EXPECT_EQ(r3.exit_code, 3u) << r3.output;
    EXPECT_NE(r3.err.find("[ep_context_corrupt]"), std::string::npos) << r3.err;
    EXPECT_NE(r3.err.find("--ep-cache build"), std::string::npos) << r3.err;
}
