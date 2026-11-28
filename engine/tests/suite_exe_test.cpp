// `edgebench suite` end to end on small fixture models: every run a child
// process, the manifest, the suite block in the results, interruption and
// --resume, --fail-fast, --skip-preflight. CPU suites run everywhere (label ort);
// the SuiteNpu tests need the NPU (label npu, serialized by the NPU lock).

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "edgebench/interrupt.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/suite.hpp"
#include "process_util.hpp"
#include "test_runtime.hpp"
#include "validator.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
namespace su = edgebench::suite;
using eb::test::ProcessResult;
using eb::test::run_process;
using eb::test::TempDir;
using json = nlohmann::json;

namespace {

const fs::path kEdgebenchExe = fs::path(EDGEBENCH_EXE_PATH).make_preferred();
const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();
const fs::path kPython = fs::path(EDGEBENCH_PYTHON).make_preferred();
const fs::path kTiny =
    kSourceDir / "engine" / "tests" / "fixtures" / "models" / "fixed_float32.onnx";
const fs::path kRuntimeError =
    kSourceDir / "engine" / "tests" / "fixtures" / "models" / "runtime_error.onnx";
const fs::path kProbeInt8 = kSourceDir / "engine" / "resources" / "doctor" / "npu_probe_int8.onnx";

std::string utf8(const fs::path& p) {
    return eb::path_to_utf8(p);
}

json read(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return json::parse(in);
}

void write(const fs::path& p, const json& j) {
    std::ofstream(p, std::ios::binary) << j.dump(2) << "\n";
}

json model(const std::string& name, const fs::path& file, const std::string& precision) {
    return {{"name", name},
            {"files", {{{"path", utf8(file)}, {"precision", precision}, {"batch", 1}}}}};
}

json config_entry(const std::string& name, const std::string& provider,
                  const std::string& precision) {
    return {{"model", name}, {"provider", provider}, {"precision", precision}, {"batch", 1}};
}

// A suite of short CPU runs: tiny (FP32) and probe (INT8), two repetitions.
json cpu_suite(const fs::path& out) {
    return {{"suite_config_version", "1.0.0"},
            {"name", "cpu-fixture"},
            {"output_dir", utf8(out)},
            {"models", {model("tiny", kTiny, "fp32"), model("probe", kProbeInt8, "int8")}},
            {"configurations",
             {config_entry("tiny", "cpu", "fp32"), config_entry("probe", "cpu", "int8")}},
            {"defaults",
             {{"warmup", 1},
              {"iterations", 5},
              {"trials", 1},
              {"trial_cooldown_ms", 0},
              {"load_sample_ms", 0}}},
            {"cooldown_s", 0},
            {"repetitions", 2},
            {"preflight", {{"checks", {"doctor"}}}}};
}

std::optional<std::string> skip_reason() {
    if (const std::string& r = eb::test::validator_unavailable_reason(); !r.empty())
        return r;
    return std::nullopt;
}

ProcessResult suite(const std::vector<std::wstring>& args, uint32_t timeout_ms = 300000) {
    std::vector<std::wstring> all = {L"suite"};
    all.insert(all.end(), args.begin(), args.end());
    all.insert(all.end(), {L"--python", kPython.native()});
    return run_process(kEdgebenchExe, all, kSourceDir, {}, timeout_ms);
}

// The one suite directory under `out`.
fs::path suite_dir(const fs::path& out) {
    std::vector<fs::path> dirs;
    for (const auto& e : fs::directory_iterator(out))
        if (e.is_directory())
            dirs.push_back(e.path());
    EXPECT_EQ(dirs.size(), 1u);
    return dirs.empty() ? out : dirs.front();
}

json manifest_of(const fs::path& dir) {
    return read(dir / std::string(su::kManifestFileName));
}

void validate_manifest(const fs::path& dir) {
    const ProcessResult v = run_process(kPython,
                                        {(kSourceDir / "scripts" / "validate_suite.py").native(),
                                         (dir / std::string(su::kManifestFileName)).native()},
                                        kSourceDir);
    EXPECT_EQ(v.exit_code, 0u) << v.output;
}

std::string describe(const ProcessResult& r) {
    return "exit " + std::to_string(r.exit_code) + "\n--- stdout\n" + r.out + "\n--- stderr\n" +
           r.err;
}

// In-process run of the suite (for the hooks); the stop flag is reset afterwards.
int run_in_process(su::SuiteOptions o, const su::SuiteHooks& hooks) {
    o.exe = kEdgebenchExe;
    o.python = kPython;
    const int code = su::run_suite(o, hooks);
    eb::reset_stop();
    return code;
}

}  // namespace

TEST(SuiteRun, DryRunPrintsThePlanAndWritesNothing) {
    TempDir dir(L"suite-dry");
    const fs::path out = dir.path() / "out";
    write(dir.path() / "suite.json", cpu_suite(out));
    const ProcessResult r = suite({(dir.path() / "suite.json").native(), L"--dry-run"});
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    EXPECT_NE(r.out.find("2 configurations x 2 repetitions = 4 runs"), std::string::npos) << r.out;
    EXPECT_NE(r.out.find("Estimate"), std::string::npos);
    EXPECT_NE(r.out.find("tiny"), std::string::npos);
    EXPECT_FALSE(fs::exists(out));
}

TEST(SuiteRun, AnInvalidConfigurationExits2BeforeAnything) {
    TempDir dir(L"suite-invalid");
    json c = cpu_suite(dir.path() / "out");
    c["configurations"][0]["provider"] = "npu";  // FP32 on the NPU without npu_fp16
    write(dir.path() / "suite.json", c);
    const ProcessResult r = suite({(dir.path() / "suite.json").native()});
    EXPECT_EQ(r.exit_code, 2u) << describe(r);
    EXPECT_NE(r.err.find("[invalid_combination]"), std::string::npos) << r.err;
    EXPECT_NE(r.err.find("npu_fp16"), std::string::npos) << r.err;
    EXPECT_FALSE(fs::exists(dir.path() / "out"));
}

TEST(SuiteRun, TinyCpuSuiteEndToEnd) {
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-cpu");
    const fs::path out = dir.path() / "out";
    write(dir.path() / "suite.json", cpu_suite(out));
    const ProcessResult r = suite({(dir.path() / "suite.json").native()});
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    EXPECT_NE(r.out.find("4 done, 0 failed"), std::string::npos) << r.out;

    const fs::path sd = suite_dir(out);
    const json m = manifest_of(sd);
    EXPECT_EQ(m["status"], "completed");
    EXPECT_EQ(m["counts"]["done"], 4);
    EXPECT_EQ(m["preflight"].size(), 1u);
    EXPECT_EQ(m["preflight"][0]["checks"][0]["name"], "doctor");
    EXPECT_TRUE(fs::is_regular_file(sd / "preflight" / "doctor-1.json"));
    EXPECT_TRUE(fs::is_regular_file(sd / std::string(su::kConfigCopyName)));
    std::set<std::string> configs;
    for (const json& run : m["runs"]) {
        EXPECT_EQ(run["status"], "done");
        EXPECT_EQ(run["exit_code"], 0);
        EXPECT_EQ(run["validation"], "passed");
        configs.insert(run["configuration_id"].get<std::string>());
        const json result =
            read(sd / eb::path_to_utf8(fs::path(run["result_file"].get<std::string>())));
        ASSERT_TRUE(result.contains("suite"));
        EXPECT_EQ(result["suite"]["run_id"], run["run_id"]);
        EXPECT_EQ(result["suite"]["order_index"], run["order_index"]);
        EXPECT_EQ(result["suite"]["repetition"], run["repetition"]);
        EXPECT_EQ(result["suite"]["suite_config_hash"], m["suite"]["config_sha256"]);
        EXPECT_EQ(result["suite"]["preflight_skipped"], false);
        EXPECT_EQ(result["label"], run["label"]);
        EXPECT_TRUE(fs::is_regular_file(
            sd / eb::path_to_utf8(fs::path(run["log_file"].get<std::string>()))));
    }
    EXPECT_EQ(configs.size(), 2u);
    // Rotated: the second round starts with the other configuration.
    EXPECT_NE(m["runs"][0]["configuration_id"], m["runs"][2]["configuration_id"]);
    validate_manifest(sd);
}

TEST(SuiteRun, InterruptedMidRunThenResumed) {
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-interrupt");
    const fs::path out = dir.path() / "out";
    write(dir.path() / "suite.json", cpu_suite(out));

    su::SuiteOptions o;
    o.config = dir.path() / "suite.json";
    su::SuiteHooks hooks;
    hooks.run_started = [](const su::RunRecord& r) {
        if (r.order_index == 1)
            eb::request_stop("a test");
    };
    EXPECT_EQ(run_in_process(o, hooks), 130);
    const fs::path sd = suite_dir(out);
    json m = manifest_of(sd);
    EXPECT_EQ(m["status"], "interrupted");
    EXPECT_EQ(m["runs"][0]["status"], "done");
    EXPECT_EQ(m["runs"][1]["status"], "pending");  // the interrupted run
    EXPECT_EQ(m["runs"][1]["discarded"].size(), 1u);
    EXPECT_EQ(m["runs"][1]["attempts"], 1);
    EXPECT_EQ(m["runs"][2]["status"], "pending");
    EXPECT_FALSE(fs::exists(sd / "runs" / m["runs"][1]["short_id"].get<std::string>()));
    validate_manifest(sd);

    const ProcessResult r = suite({L"--resume", sd.native()});
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    m = manifest_of(sd);
    EXPECT_EQ(m["status"], "completed");
    EXPECT_EQ(m["counts"]["done"], 4);
    EXPECT_EQ(m["runs"][0]["attempts"], 1);  // kept
    EXPECT_EQ(m["runs"][1]["attempts"], 2);  // run again
    EXPECT_EQ(m["invocations"].size(), 2u);
    EXPECT_TRUE(m["invocations"][1]["resumed"].get<bool>());
    EXPECT_EQ(m["preflight"].size(), 2u);  // the gate ran again
    validate_manifest(sd);
}

TEST(SuiteRun, InterruptedBetweenRunsLeavesTheRestPending) {
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-between");
    const fs::path out = dir.path() / "out";
    write(dir.path() / "suite.json", cpu_suite(out));
    su::SuiteOptions o;
    o.config = dir.path() / "suite.json";
    su::SuiteHooks hooks;
    hooks.run_finished = [](const su::RunRecord& r) {
        if (r.order_index == 0)
            eb::request_stop("a test");
    };
    EXPECT_EQ(run_in_process(o, hooks), 130);
    const json m = manifest_of(suite_dir(out));
    EXPECT_EQ(m["counts"]["done"], 1);
    EXPECT_EQ(m["counts"]["pending"], 3);
    EXPECT_TRUE(m["runs"][1]["discarded"].empty());
}

TEST(SuiteRun, ResumeRerunsMissingAndRunningRunsAndRefusesAChangedConfig) {
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-resume");
    const fs::path out = dir.path() / "out";
    const fs::path config = dir.path() / "suite.json";
    write(config, cpu_suite(out));
    ASSERT_EQ(suite({config.native()}).exit_code, 0u);
    const fs::path sd = suite_dir(out);

    // As if killed: one run left "running", one done run's result deleted.
    json m = manifest_of(sd);
    m["runs"][2]["status"] = "running";
    m["counts"]["done"] = 3;
    m["counts"]["running"] = 1;
    fs::remove(sd / eb::path_to_utf8(fs::path(m["runs"][1]["result_file"].get<std::string>())));
    write(sd / std::string(su::kManifestFileName), m);

    ProcessResult r = suite({L"--resume", sd.native()});
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    m = manifest_of(sd);
    EXPECT_EQ(m["counts"]["done"], 4);
    EXPECT_EQ(m["runs"][0]["attempts"], 1);
    EXPECT_EQ(m["runs"][1]["attempts"], 2);  // result missing: run again
    EXPECT_EQ(m["runs"][2]["attempts"], 2);  // was running: discarded, run again
    EXPECT_EQ(m["runs"][2]["discarded"].size(), 1u);
    EXPECT_EQ(m["runs"][3]["attempts"], 1);
    validate_manifest(sd);

    // A changed configuration file: refused, nothing touched.
    json changed = cpu_suite(out);
    changed["defaults"]["iterations"] = 6;
    write(config, changed);
    const std::string before = manifest_of(sd).dump();
    r = suite({L"--resume", sd.native()});
    EXPECT_EQ(r.exit_code, 2u) << describe(r);
    EXPECT_NE(r.err.find("[suite_config_changed]"), std::string::npos) << r.err;
    EXPECT_EQ(manifest_of(sd).dump(), before);
}

TEST(SuiteRun, FailedRunsAreRecordedAndFailFastStops) {
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-fail");
    json c = cpu_suite(dir.path() / "out");
    c["models"].push_back(model("broken", kRuntimeError, "fp32"));
    c["configurations"] = {config_entry("broken", "cpu", "fp32"),
                           config_entry("tiny", "cpu", "fp32")};
    c["rotation"] = "blocked";
    write(dir.path() / "suite.json", c);

    ProcessResult r = suite({(dir.path() / "suite.json").native()});
    EXPECT_EQ(r.exit_code, 1u) << describe(r);
    json m = manifest_of(suite_dir(dir.path() / "out"));
    EXPECT_EQ(m["counts"]["failed"], 2);  // both repetitions of the broken model
    EXPECT_EQ(m["counts"]["done"], 2);    // the suite went on
    EXPECT_EQ(m["runs"][0]["exit_code"], 3);
    EXPECT_EQ(m["runs"][0]["exit_code_name"], "ort_runtime_error");
    EXPECT_NE(r.out.find("Failed runs"), std::string::npos);
    EXPECT_NE(r.out.find(m["runs"][0]["log_file"].get<std::string>()), std::string::npos) << r.out;

    c["output_dir"] = utf8(dir.path() / "out2");
    write(dir.path() / "suite2.json", c);
    r = suite({(dir.path() / "suite2.json").native(), L"--fail-fast"});
    EXPECT_EQ(r.exit_code, 1u) << describe(r);
    m = manifest_of(suite_dir(dir.path() / "out2"));
    EXPECT_EQ(m["status"], "aborted");
    EXPECT_EQ(m["counts"]["failed"], 1);
    EXPECT_EQ(m["counts"]["skipped"], 3);
    validate_manifest(suite_dir(dir.path() / "out2"));
}

TEST(SuiteRun, SkippedPreflightMarksEverythingNotPublishable) {
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-skip");
    json c = cpu_suite(dir.path() / "out");
    c["repetitions"] = 1;
    c["configurations"] = json::array({config_entry("tiny", "cpu", "fp32")});
    write(dir.path() / "suite.json", c);
    const ProcessResult r = suite({(dir.path() / "suite.json").native(), L"--skip-preflight"});
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    EXPECT_NE(r.err.find("[preflight_skipped]"), std::string::npos);
    const fs::path sd = suite_dir(dir.path() / "out");
    const json m = manifest_of(sd);
    EXPECT_FALSE(m["publishable"].get<bool>());
    EXPECT_TRUE(m["preflight_ever_skipped"].get<bool>());
    EXPECT_FALSE(fs::exists(sd / "preflight"));
    const json result =
        read(sd / eb::path_to_utf8(fs::path(m["runs"][0]["result_file"].get<std::string>())));
    EXPECT_TRUE(result["suite"]["preflight_skipped"].get<bool>());
    bool warned = false;
    for (const json& w : result["warnings"])
        warned = warned || w["code"] == "preflight_skipped";
    EXPECT_TRUE(warned);
    // Valid, but never with --strict (which the published results are checked with).
    const ProcessResult strict = eb::test::run_validator(
        sd / eb::path_to_utf8(fs::path(m["runs"][0]["result_file"].get<std::string>())));
    EXPECT_NE(strict.exit_code, 0u) << strict.output;
    validate_manifest(sd);
}

TEST(SuiteRun, StandaloneRunsCarryNoSuiteBlock) {
    TempDir dir(L"suite-standalone");
    const ProcessResult r =
        run_process(kEdgebenchExe,
                    {L"run", L"--model", kTiny.native(), L"--iterations", L"5", L"--trials", L"1",
                     L"--warmup", L"0", L"--cooldown-ms", L"0", L"--load-sample-ms", L"0",
                     L"--output-dir", dir.path().native(), L"--name", L"standalone"},
                    kSourceDir);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const json result = read(dir.path() / "standalone_latest.json");
    EXPECT_FALSE(result.contains("suite"));
    for (const json& w : result["warnings"])
        EXPECT_NE(w["code"], "preflight_skipped");
}

// --------------------------------------------------------------------------- NPU

TEST(SuiteNpu, CachedInitBuildsTheEntryJustBeforeTheTimedRun) {
    if (auto reason = eb::test::npu_skip_reason(eb::test::probe_qnn_npu()))
        GTEST_SKIP() << *reason;
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-npu");
    json c = cpu_suite(dir.path() / "out");
    c["models"] = json::array({model("probe", kProbeInt8, "int8")});
    json npu = config_entry("probe", "npu", "int8");
    npu["init"] = "cached";
    npu["profile_accelerator"] = true;
    npu["profile_iterations"] = 5;
    c["configurations"] = {npu, config_entry("probe", "cpu", "int8")};
    write(dir.path() / "suite.json", c);
    const ProcessResult r = suite({(dir.path() / "suite.json").native()}, 600000);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const fs::path sd = suite_dir(dir.path() / "out");
    const json m = manifest_of(sd);
    int npu_runs = 0;
    for (const json& run : m["runs"]) {
        ASSERT_EQ(run["status"], "done") << run.dump(2);
        if (run["configuration"]["provider"] != "npu")
            continue;
        ++npu_runs;
        ASSERT_TRUE(run["cache_build"].is_object());
        EXPECT_EQ(run["cache_build"]["exit_code"], 0);
        const json build = read(
            sd / eb::path_to_utf8(fs::path(run["cache_build"]["result_file"].get<std::string>())));
        const json timed =
            read(sd / eb::path_to_utf8(fs::path(run["result_file"].get<std::string>())));
        const json& cache = timed["config"]["ep_context_cache"];
        EXPECT_EQ(cache["mode"], "use");
        EXPECT_EQ(cache["hit"], true);
        EXPECT_EQ(cache["state"], "loaded");
        EXPECT_EQ(cache["key_id"], build["config"]["ep_context_cache"]["key_id"]);
        EXPECT_TRUE(timed["timing"]["profiled_pass"]["enabled"].get<bool>());
        EXPECT_FALSE(build.contains("suite") && build["suite"].is_null());
    }
    EXPECT_EQ(npu_runs, 2);
    EXPECT_TRUE(fs::is_directory(sd / "qnn-context"));
    validate_manifest(sd);
}

TEST(SuiteNpu, SmokeConfigurationEndToEnd) {
    if (auto reason = eb::test::npu_skip_reason(eb::test::probe_qnn_npu()))
        GTEST_SKIP() << *reason;
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    const fs::path smoke = kSourceDir / "benchmarks" / "configs" / "smoke.json";
    const eb::Result<su::SuitePlan> plan = su::load_suite(smoke);
    if (!plan)
        GTEST_SKIP() << "the smoke configuration's models are not available: "
                     << plan.error().message;
    TempDir dir(L"suite-smoke");
    // The gate has its own tests (and needs an idle machine); this one is about the runs.
    const ProcessResult r =
        suite({smoke.native(), L"--output-dir", dir.path().native(), L"--skip-preflight"}, 900000);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const fs::path sd = suite_dir(dir.path());
    const json m = manifest_of(sd);
    EXPECT_EQ(m["counts"]["done"], 6);
    validate_manifest(sd);
}

// ------------------------------------------------------------- gate and retries

namespace {

// A sustained-load leg on a fixture model: short, with a lenient drift limit (the
// test checks that the series is recorded, not this machine's throttling).
json fixture_leg(const fs::path& model, const std::string& provider, const std::string& precision) {
    return {{"path", utf8(model)},
            {"provider", provider},
            {"precision", precision},
            {"duration_s", 10}};
}

json short_gate(std::vector<std::string> checks, json legs) {
    return {{"checks", checks},
            {"idle", {{"window_s", 3}}},
            {"sustained_load", {{"window_s", 2}, {"max_drift_ratio", 10}, {"legs", legs}}}};
}

const json* check_named(const json& manifest, const std::string& name) {
    for (const json& c : manifest["preflight"].back()["checks"])
        if (c["name"] == name)
            return &c;
    return nullptr;
}

}  // namespace

TEST(SuiteRun, RetryFlaggedRunsAgainAndKeepsTheFirstAttempt) {
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-retry");
    json c = cpu_suite(dir.path() / "out");
    c["repetitions"] = 1;
    write(dir.path() / "suite.json", c);
    su::SuiteOptions o;
    o.config = dir.path() / "suite.json";
    su::SuiteHooks hooks;
    hooks.run_finished = [](su::RunRecord& r) {
        if (r.order_index == 0 && r.attempts == 1)
            r.warnings.push_back("trial_spread_high");  // as if the result had been flagged
    };
    EXPECT_EQ(run_in_process(o, hooks), 0);
    const fs::path sd = suite_dir(dir.path() / "out");
    const json m = manifest_of(sd);
    EXPECT_EQ(m["runs"][0]["attempts"], 2);
    EXPECT_EQ(m["runs"][0]["retries"], 1);
    ASSERT_EQ(m["runs"][0]["history"].size(), 1u);
    EXPECT_NE(m["runs"][0]["history"][0]["reason"].get<std::string>().find("retry_flagged"),
              std::string::npos);
    EXPECT_TRUE(fs::is_directory(
        sd /
        eb::path_to_utf8(fs::path(m["runs"][0]["history"][0]["moved_to"].get<std::string>()))));
    EXPECT_EQ(m["runs"][1]["attempts"], 1);  // not flagged: not retried
    const json result =
        read(sd / eb::path_to_utf8(fs::path(m["runs"][0]["result_file"].get<std::string>())));
    EXPECT_EQ(result["suite"]["attempt"], 2);
    validate_manifest(sd);
}

TEST(SuiteRun, GateChecksAreRecordedWithTheirMeasurements) {
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-gate");
    json c = cpu_suite(dir.path() / "out");
    c["repetitions"] = 1;
    c["configurations"] = json::array({config_entry("tiny", "cpu", "fp32")});
    c["preflight"] = short_gate({"npu_users", "doctor", "sustained_load"},
                                json::array({fixture_leg(kProbeInt8, "cpu", "int8")}));
    c["preflight"]["sustained_load"]["window_s"] = 1;
    write(dir.path() / "suite.json", c);
    const ProcessResult r = suite({(dir.path() / "suite.json").native()});
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    EXPECT_NE(r.out.find("Gate        "), std::string::npos) << r.out;
    const fs::path sd = suite_dir(dir.path() / "out");
    const json m = manifest_of(sd);
    const json& attempt = m["preflight"].back();
    EXPECT_EQ(attempt["checks"].size(), 3u);
    EXPECT_EQ(attempt["settings"]["sustained_load"]["window_s"], 1.0);
    const json* sustained = check_named(m, "sustained_load");
    ASSERT_NE(sustained, nullptr);
    const json& leg = (*sustained)["details"]["legs"][0];
    EXPECT_GE(leg["p50_series_ms"].size(), 2u) << sustained->dump(2);
    EXPECT_TRUE(leg["drift_ratio"].is_number());
    EXPECT_GT(leg["iterations"].get<int>(), 100);
    EXPECT_TRUE(check_named(m, "npu_users")->at("details").contains("processes_scanned"));
    // Not the full gate: never publishable, and every result says so.
    EXPECT_FALSE(m["gate_passed"].get<bool>());
    EXPECT_FALSE(m["publishable"].get<bool>());
    const json result =
        read(sd / eb::path_to_utf8(fs::path(m["runs"][0]["result_file"].get<std::string>())));
    EXPECT_FALSE(result["suite"]["gate_passed"].get<bool>());
    EXPECT_FALSE(result["suite"]["publishable"].get<bool>());
    validate_manifest(sd);
}

TEST(SuiteRun, FullGateOnThisMachine) {
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-full-gate");
    json c = cpu_suite(dir.path() / "out");
    c["repetitions"] = 1;
    c["configurations"] = json::array({config_entry("tiny", "cpu", "fp32")});
    c["preflight"] = short_gate(su::gate_check_names(),
                                json::array({fixture_leg(kProbeInt8, "cpu", "int8")}));
    c["preflight"]["sustained_load"]["window_s"] = 1;
    c["in_flight"] = {{"load_sample_s", 1}, {"max_wait_s", 0}, {"power_max_wait_s", 0}};
    write(dir.path() / "suite.json", c);
    const ProcessResult r = suite({(dir.path() / "suite.json").native()});
    const json m = manifest_of(suite_dir(dir.path() / "out"));
    ASSERT_EQ(m["preflight"].back()["checks"].size(), su::gate_check_names().size()) << describe(r);
    // Power, sleep settings and background load are this machine's state: when it is
    // not set up for publishable runs, say which check stopped it instead of passing.
    std::string machine_state;
    for (const char* name : {"power", "sleep", "idle"}) {
        const json* check = check_named(m, name);
        if (check->at("status") == "fail")
            machine_state +=
                std::string(name) + ": " + check->at("summary").get<std::string>() + "; ";
    }
    if (!machine_state.empty()) {
        EXPECT_EQ(r.exit_code, 1u) << describe(r);
        EXPECT_EQ(m["status"], "aborted");
        GTEST_SKIP() << "this machine is not set up for publishable runs: " << machine_state;
    }
    EXPECT_EQ(r.exit_code, 0u) << describe(r);
    const json& run = m["runs"][0];
    EXPECT_TRUE(run["in_flight"]["power_ready"].get<bool>());
    EXPECT_FALSE(run["in_flight"]["load_samples"].empty());
}

TEST(SuiteNpu, SustainedLoadRunsAnNpuLeg) {
    if (auto reason = eb::test::npu_skip_reason(eb::test::probe_qnn_npu()))
        GTEST_SKIP() << *reason;
    if (auto reason = skip_reason())
        GTEST_SKIP() << *reason;
    TempDir dir(L"suite-npu-sustained");
    json c = cpu_suite(dir.path() / "out");
    c["repetitions"] = 1;
    c["configurations"] = json::array({config_entry("tiny", "cpu", "fp32")});
    c["preflight"] =
        short_gate({"sustained_load"}, json::array({fixture_leg(kProbeInt8, "npu", "int8")}));
    write(dir.path() / "suite.json", c);
    const ProcessResult r = suite({(dir.path() / "suite.json").native()}, 600000);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const json m = manifest_of(suite_dir(dir.path() / "out"));
    const json& leg = check_named(m, "sustained_load")->at("details")["legs"][0];
    EXPECT_NE(leg["label"].get<std::string>().find("_npu_int8"), std::string::npos);
    EXPECT_GE(leg["p50_series_ms"].size(), 2u);
}
