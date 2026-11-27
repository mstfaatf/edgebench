// `edgebench suite` without running anything: configuration files (the fixtures
// shared with scripts/tests/test_validate_suite.py), matrix expansion, the run
// order, run IDs, the estimate, the manifest and the resume decisions.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <map>
#include <set>

#include "edgebench/cli.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/suite.hpp"
#include "edgebench/suite_config.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
namespace su = edgebench::suite;
using json = nlohmann::json;

namespace {

const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();
const fs::path kFixtures = kSourceDir / "scripts" / "tests" / "fixtures" / "suite";
// Pinned in scripts/tests/test_validate_suite.py too: both sides hash the same way.
constexpr const char* kExplicitCpuHash =
    "3234ed5a4fe45e164cf6e0e42aced9fc3b0eb382a1321afe8918062cbc04f571";

json read(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return json::parse(in);
}

su::SuitePlan load(const fs::path& p) {
    eb::Result<su::SuitePlan> plan = su::load_suite(p);
    if (!plan)
        ADD_FAILURE() << plan.error().message;
    return plan ? std::move(plan.value()) : su::SuitePlan{};
}

std::vector<std::string> kinds(const su::LoadOutcome& o) {
    std::vector<std::string> k;
    for (const su::SuiteConfigError& e : o.errors)
        k.push_back(e.kind);
    return k;
}

std::string describe(const su::LoadOutcome& o) {
    std::string s;
    for (const su::SuiteConfigError& e : o.errors)
        s += e.pointer + ": " + e.message + " [" + e.kind + "]\n";
    return s;
}

}  // namespace

// ------------------------------------------------------------------- configuration

TEST(SuiteConfig, ValidFixturesExpandAsWritten) {
    const su::SuitePlan a = load(kFixtures / "valid" / "explicit_cpu.json");
    ASSERT_EQ(a.configurations.size(), 2u);
    EXPECT_EQ(a.runs.size(), 4u);
    EXPECT_EQ(a.config_sha256, kExplicitCpuHash);
    EXPECT_EQ(a.configurations[1].params.iterations, 7);  // the entry's own key wins
    EXPECT_EQ(a.configurations[0].params.iterations, 5);  // defaults
    EXPECT_EQ(a.configurations[0].label, "tiny_b1_cpu_fp32");
    EXPECT_EQ(a.configurations[1].compute_precision, "int8");

    const su::SuitePlan m = load(kFixtures / "valid" / "matrix_cpu_npu.json");
    // probe: cpu fp32, cpu int8, npu fp32 (fp16 compute), npu int8; plus the include.
    ASSERT_EQ(m.configurations.size(), 5u);
    EXPECT_EQ(m.runs.size(), 15u);
    std::set<std::string> labels;
    for (const su::ResolvedConfiguration& c : m.configurations)
        labels.insert(c.label);
    EXPECT_EQ(labels,
              (std::set<std::string>{"probe_b1_cpu_fp32", "probe_b1_cpu_int8", "probe_b1_npu_fp16",
                                     "probe_b1_npu_int8", "tiny_b1_cpu_fp32"}));
    for (const su::ResolvedConfiguration& c : m.configurations) {
        const bool npu = c.identity.provider == eb::ProviderKind::npu;
        EXPECT_EQ(c.params.init, npu ? su::InitMode::cached : su::InitMode::cold) << c.label;
        EXPECT_EQ(c.params.profile_accelerator, npu) << c.label;
        if (c.label == "probe_b1_npu_fp16") {
            EXPECT_TRUE(c.identity.npu_fp16);
            EXPECT_EQ(c.compute_precision, "fp16");
        }
    }
}

TEST(SuiteConfig, EveryInvalidFixtureIsRejectedWithItsKind) {
    int count = 0;
    for (const auto& entry : fs::directory_iterator(kFixtures / "invalid")) {
        const json fixture = read(entry.path());
        const su::LoadOutcome o = su::plan_suite(fixture["config"], entry.path());
        const std::string kind = fixture["expect"]["engine_kind"];
        EXPECT_FALSE(o.plan.has_value()) << entry.path().filename();
        const std::vector<std::string> k = kinds(o);
        EXPECT_NE(std::find(k.begin(), k.end(), kind), k.end())
            << entry.path().filename() << " expected " << kind << ", got:\n"
            << describe(o);
        ++count;
    }
    EXPECT_GE(count, 15);
}

TEST(SuiteConfig, RejectionsSayWhatToChange) {
    const json fixture = read(kFixtures / "invalid" / "npu_fp32_without_fp16.json");
    const su::LoadOutcome o = su::plan_suite(fixture["config"], kFixtures / "invalid" / "x.json");
    ASSERT_EQ(o.errors.size(), 1u) << describe(o);
    EXPECT_NE(o.errors[0].message.find("probe npu fp32 b1"), std::string::npos);
    EXPECT_NE(o.errors[0].message.find("npu_fp16"), std::string::npos) << o.errors[0].message;
    EXPECT_EQ(o.errors[0].message.find("--npu-fp16"), std::string::npos) << o.errors[0].message;

    const eb::Error e = su::to_error("x.json", o.errors);
    EXPECT_EQ(e.exit_code, eb::ExitCode::usage_error);
    EXPECT_NE(e.message.find("/configurations/0"), std::string::npos);
}

TEST(SuiteConfig, LoadingAFileThatIsNotJsonIsAUsageError) {
    const fs::path p = fs::temp_directory_path() / "edgebench-suite-not-json.json";
    std::ofstream(p) << "{ not json";
    const eb::Result<su::SuitePlan> r = su::load_suite(p);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().kind, "suite_config_not_json");
    EXPECT_EQ(r.error().exit_code, eb::ExitCode::usage_error);
    fs::remove(p);
    EXPECT_EQ(su::load_suite(p).error().kind, "suite_config_not_found");
}

// ----------------------------------------------------------------------- run order

TEST(SuiteOrder, BlockedRunsEveryRepetitionOfAConfigurationTogether) {
    const auto order = su::run_order(3, 2, su::Rotation::blocked);
    const std::vector<std::pair<std::size_t, int>> want = {{0, 0}, {0, 1}, {1, 0},
                                                           {1, 1}, {2, 0}, {2, 1}};
    EXPECT_EQ(order, want);
}

TEST(SuiteOrder, RotatedRoundsArePermutationsWithChangingFirstRun) {
    for (std::size_t n = 1; n <= 7; ++n) {
        const int rounds = static_cast<int>(n % 2 == 0 ? n : 2 * n);
        const auto order = su::run_order(n, rounds, su::Rotation::rotated);
        ASSERT_EQ(order.size(), n * static_cast<std::size_t>(rounds));
        std::map<std::size_t, int> first;
        for (int r = 0; r < rounds; ++r) {
            std::set<std::size_t> seen;
            for (std::size_t k = 0; k < n; ++k) {
                const auto [c, rep] = order[r * n + k];
                EXPECT_EQ(rep, r);
                seen.insert(c);
            }
            EXPECT_EQ(seen.size(), n) << "n " << n << " round " << r;
            ++first[order[r * n].first];
        }
        // Every configuration is first equally often (once, or twice for an odd n).
        for (std::size_t c = 0; c < n; ++c)
            EXPECT_EQ(first[c], n % 2 == 0 ? 1 : 2) << "n " << n << " config " << c;
        if (n >= 2) {  // no configuration is always first after a cooldown
            EXPECT_NE(order[0].first, order[n].first);
        }
    }
}

TEST(SuiteOrder, RotatedNeighboursAreBalanced) {
    for (std::size_t n = 2; n <= 7; ++n) {
        const int rounds = static_cast<int>(n % 2 == 0 ? n : 2 * n);
        const auto order = su::run_order(n, rounds, su::Rotation::rotated);
        std::map<std::pair<std::size_t, std::size_t>, int> follows;
        for (int r = 0; r < rounds; ++r) {
            for (std::size_t k = 1; k < n; ++k)
                ++follows[{order[r * n + k - 1].first, order[r * n + k].first}];
        }
        // Every ordered pair of different configurations is adjacent equally often.
        const int want = n % 2 == 0 ? 1 : 2;
        for (std::size_t a = 0; a < n; ++a) {
            for (std::size_t b = 0; b < n; ++b) {
                if (a != b)
                    EXPECT_EQ(follows[std::make_pair(a, b)], want)
                        << "n " << n << ": " << a << " -> " << b;
            }
        }
    }
}

TEST(SuiteOrder, NothingToOrder) {
    EXPECT_TRUE(su::run_order(0, 3, su::Rotation::rotated).empty());
    EXPECT_TRUE(su::run_order(3, 0, su::Rotation::blocked).empty());
}

// ------------------------------------------------------------------------ run IDs

TEST(SuiteRunIds, StableAcrossLoadsKeyOrderAndRotation) {
    const fs::path file = kFixtures / "valid" / "explicit_cpu.json";
    const su::SuitePlan a = load(file);
    const su::SuitePlan b = load(file);
    ASSERT_EQ(a.runs.size(), b.runs.size());
    for (std::size_t i = 0; i < a.runs.size(); ++i)
        EXPECT_EQ(a.runs[i].run_id, b.runs[i].run_id);

    json doc = read(file);
    // Whitespace and key order of the file do not matter: the hash is of canonical JSON.
    const json reparsed = json::parse(doc.dump(7));
    const su::LoadOutcome c = su::plan_suite(reparsed, file);
    ASSERT_TRUE(c.plan) << describe(c);
    EXPECT_EQ(c.plan->config_sha256, a.config_sha256);

    // Blocked instead of rotated: other order and config hash, the same set of IDs.
    doc["rotation"] = "blocked";
    const su::LoadOutcome d = su::plan_suite(doc, file);
    ASSERT_TRUE(d.plan);
    EXPECT_NE(d.plan->config_sha256, a.config_sha256);
    std::set<std::string> ids_a, ids_d;
    for (const su::PlannedRun& r : a.runs)
        ids_a.insert(r.run_id);
    for (const su::PlannedRun& r : d.plan->runs)
        ids_d.insert(r.run_id);
    EXPECT_EQ(ids_a, ids_d);
    EXPECT_EQ(ids_a.size(), 4u);
}

TEST(SuiteRunIds, ChangeWithEveryResolvedSettingAndTheRepetition) {
    const fs::path file = kFixtures / "valid" / "explicit_cpu.json";
    const su::SuitePlan base = load(file);
    const std::string id0 = base.runs[0].run_id;
    const su::ResolvedConfiguration& c = base.configurations[base.runs[0].configuration];
    EXPECT_NE(su::run_id(c.resolved, 0), su::run_id(c.resolved, 1));
    EXPECT_EQ(su::run_id(c.resolved, base.runs[0].repetition), id0);
    for (const char* key : {"model_sha256", "iterations", "trials", "threads", "init",
                            "compute_precision", "batch", "ep_compile_options"}) {
        nlohmann::ordered_json changed = c.resolved;
        changed[key] = "changed";
        EXPECT_NE(su::run_id(changed, base.runs[0].repetition), id0) << key;
    }
    EXPECT_EQ(id0.size(), 64u);
    EXPECT_EQ(base.runs[0].short_id(), id0.substr(0, 12));
}

TEST(SuiteRunIds, NpuCompileOptionsComeFromTheEngineOptions) {
    json doc = read(kFixtures / "valid" / "matrix_cpu_npu.json");
    doc["matrix"]["npu"]["htp_graph_finalization_mode"] = {nullptr, "1"};
    const su::LoadOutcome o = su::plan_suite(doc, kFixtures / "valid" / "matrix_cpu_npu.json");
    ASSERT_TRUE(o.plan) << describe(o);
    int with_mode = 0;
    for (const su::ResolvedConfiguration& c : o.plan->configurations) {
        if (c.identity.htp_graph_finalization_mode) {
            ++with_mode;
            EXPECT_EQ(c.ep_compile_options,
                      (std::map<std::string, std::string>{
                          {"htp_graph_finalization_optimization_mode", "1"}}));
            EXPECT_NE(c.label.find("_finalization1"), std::string::npos);
        } else {
            EXPECT_TRUE(c.ep_compile_options.empty()) << c.label;
        }
    }
    EXPECT_EQ(with_mode, 2);  // npu fp16 and npu int8
}

// ---------------------------------------------------------------- run arguments

TEST(SuiteRunArguments, RoundTripThroughTheRunParser) {
    const su::SuitePlan m = load(kFixtures / "valid" / "matrix_cpu_npu.json");
    for (const su::ResolvedConfiguration& c : m.configurations) {
        eb::RunConfig cfg = su::to_run_config(c, "out dir", "name_rep0", fs::path("cache dir"));
        eb::SuiteLink link{"fixture", std::string(64, 'a'), std::string(64, 'b'), 2, 7, 2.5, true};
        cfg.suite = link;
        const eb::ParseResult p = eb::parse_command_line(su::run_arguments(cfg));
        ASSERT_FALSE(p.exit) << c.label << ": " << p.err;
        const eb::RunConfig& r = p.run;
        EXPECT_EQ(r.model, cfg.model);
        EXPECT_EQ(r.provider, cfg.provider);
        EXPECT_EQ(r.iterations, cfg.iterations);
        EXPECT_EQ(r.trials, cfg.trials);
        EXPECT_EQ(r.warmup, cfg.warmup);
        EXPECT_EQ(r.cooldown_ms, cfg.cooldown_ms);
        EXPECT_EQ(r.load_sample_ms, cfg.load_sample_ms);
        EXPECT_EQ(r.npu_fp16, cfg.npu_fp16);
        EXPECT_EQ(r.ep_cache, cfg.ep_cache);
        EXPECT_EQ(r.ep_cache_dir, cfg.ep_cache_dir);
        EXPECT_EQ(r.profile_accelerator, cfg.profile_accelerator);
        EXPECT_EQ(r.profile_iterations, cfg.profile_iterations);
        EXPECT_EQ(r.name, cfg.name);
        EXPECT_EQ(r.label, cfg.label);
        EXPECT_EQ(r.suite, cfg.suite);
    }
}

TEST(SuiteRunArguments, CacheBuildRunIsShortAndBuilds) {
    const su::SuitePlan m = load(kFixtures / "valid" / "matrix_cpu_npu.json");
    const auto npu = std::find_if(m.configurations.begin(), m.configurations.end(),
                                  [](const su::ResolvedConfiguration& c) {
                                      return c.identity.provider == eb::ProviderKind::npu;
                                  });
    ASSERT_NE(npu, m.configurations.end());
    const eb::RunConfig b = su::cache_build_config(*npu, "out", "n", "cache");
    EXPECT_EQ(b.ep_cache, eb::EpCacheMode::build);
    EXPECT_EQ(b.trials, 1);
    EXPECT_EQ(b.iterations, 1);
    EXPECT_EQ(b.warmup, 0);
    EXPECT_FALSE(b.profile_accelerator);
    EXPECT_TRUE(eb::validate(b).empty());
    const eb::RunConfig t = su::to_run_config(*npu, "out", "n", fs::path("cache"));
    EXPECT_EQ(t.ep_cache, eb::EpCacheMode::use);
}

// ----------------------------------------------------------------------- the CLI

TEST(SuiteCli, OptionsAndErrors) {
    eb::ParseResult p = eb::parse_command_line({"suite", "plan.json", "--dry-run"});
    ASSERT_FALSE(p.exit) << p.err;
    EXPECT_EQ(p.command, eb::Command::suite);
    EXPECT_EQ(p.suite.config, fs::path("plan.json"));
    EXPECT_TRUE(p.suite.dry_run);

    p = eb::parse_command_line({"suite", "--resume", "dir", "--fail-fast", "--skip-preflight"});
    ASSERT_FALSE(p.exit) << p.err;
    EXPECT_EQ(p.suite.resume, fs::path("dir"));
    EXPECT_FALSE(p.suite.config);
    EXPECT_TRUE(p.suite.fail_fast);
    EXPECT_TRUE(p.suite.skip_preflight);

    for (const std::vector<std::string>& bad : {std::vector<std::string>{"suite"},
                                                {"suite", "--resume", "d", "--dry-run"},
                                                {"suite", "--resume", "d", "--output-dir", "o"},
                                                {"suite", "a.json", "--bogus"}}) {
        const eb::ParseResult e = eb::parse_command_line(bad);
        ASSERT_TRUE(e.exit);
        EXPECT_EQ(*e.exit, eb::ExitCode::usage_error);
        EXPECT_NE(e.err.find("error: [usage_error]"), std::string::npos) << e.err;
    }
}

TEST(SuiteCli, HiddenRunOptionsGoTogether) {
    const fs::path model =
        kSourceDir / "engine" / "tests" / "fixtures" / "models" / "fixed_float32.onnx";
    const std::string m = eb::path_to_utf8(model);
    const eb::ParseResult partial =
        eb::parse_command_line({"run", "--model", m, "--suite-name", "x"});
    ASSERT_TRUE(partial.exit);
    EXPECT_EQ(*partial.exit, eb::ExitCode::usage_error);
    const eb::ParseResult skipped =
        eb::parse_command_line({"run", "--model", m, "--suite-preflight-skipped"});
    ASSERT_TRUE(skipped.exit);
    const eb::ParseResult bad_hash =
        eb::parse_command_line({"run", "--model", m, "--suite-name", "x", "--suite-config-hash",
                                "abc", "--suite-run-id", std::string(64, 'a'), "--suite-repetition",
                                "0", "--suite-order-index", "0", "--suite-cooldown-s-before", "1"});
    ASSERT_TRUE(bad_hash.exit);
    EXPECT_NE(bad_hash.err.find("--suite-config-hash"), std::string::npos) << bad_hash.err;
    // Hidden: not in the help text.
    const eb::ParseResult help = eb::parse_command_line({"run", "--help"});
    EXPECT_EQ(help.out.find("--suite-"), std::string::npos);
}

// ---------------------------------------------------------------------- estimate

TEST(SuiteEstimate, FollowsTheDocumentedFormula) {
    su::SuitePlan plan = load(kFixtures / "valid" / "explicit_cpu.json");
    su::ResolvedConfiguration c = plan.configurations[0];
    c.params.trials = 3;
    c.params.warmup = 10;
    c.params.iterations = 90;
    c.params.trial_cooldown_ms = 1000;
    c.params.cooldown_s = 4;
    su::TimingBasis b;
    b.iteration_ms = 2.0;
    b.cold_init_s = 0.1;
    // overhead + trials * (idle + cold + 100 iterations) + placement + validation + cooldown
    const double want =
        (su::kProcessOverheadS + 3 * (1.0 + 0.1 + 100 * 0.002) + 0.1 + su::kValidationS + 4) *
        su::kEstimateMargin;
    EXPECT_NEAR(su::estimate_run_s(c, b), want, 1e-9);

    // More trials, iterations or a cooldown never shorten it.
    su::ResolvedConfiguration longer = c;
    longer.params.iterations = 900;
    EXPECT_GT(su::estimate_run_s(longer, b), su::estimate_run_s(c, b));
}

TEST(SuiteEstimate, PriorResultsMatchByModelProviderPrecisionAndBatch) {
    const su::SuitePlan plan = load(kFixtures / "valid" / "explicit_cpu.json");
    const su::ResolvedConfiguration& c = plan.configurations[0];
    auto result = [&](double mean, double cold) {
        return json{{"schema_version", "1.2.0"},
                    {"status", "success"},
                    {"model", {{"sha256", c.model_sha256}}},
                    {"config",
                     {{"requested", {{"execution_provider", "CPUExecutionProvider"}}},
                      {"batch_size", 1},
                      {"measured_iterations", 100}}},
                    {"execution", {{"effective_precision", {{"compute", "fp32"}}}}},
                    {"timing",
                     {{"aggregate", {{"stats", {{"mean_ms", mean}}}}},
                      {"trials", json::array({{{"stats", {{"mean_ms", mean}}}}})},
                      {"session_init", {{"cold_ms", {cold}}, {"session_create_ms", cold}}}}}};
    };
    su::PriorResults prior;
    prior.add(result(0.5, 40), "a.json");
    prior.add(result(0.7, 20), "b.json");
    json other = result(9.0, 900);
    other["config"]["batch_size"] = 8;
    prior.add(other, "c.json");
    const std::optional<su::TimingBasis> found = prior.find(c);
    ASSERT_TRUE(found);
    EXPECT_TRUE(found->from_prior);
    EXPECT_DOUBLE_EQ(found->iteration_ms, 0.7);  // the slowest match
    EXPECT_DOUBLE_EQ(found->cold_init_s, 0.04);
    EXPECT_FALSE(prior.find(plan.configurations[1]));

    const su::SuiteEstimate e = su::estimate_suite(plan, prior);
    EXPECT_EQ(e.from_prior, 1);
    EXPECT_EQ(e.from_heuristic, 1);
    ASSERT_EQ(e.run_s.size(), plan.runs.size());
    double sum = su::kPreflightS * su::kEstimateMargin;
    for (double s : e.run_s)
        sum += s;
    EXPECT_NEAR(e.total_s, sum, 1e-9);
}

TEST(SuiteEstimate, HeuristicIsConservative) {
    const su::SuitePlan plan = load(kFixtures / "valid" / "explicit_cpu.json");
    su::ResolvedConfiguration c = plan.configurations[0];
    c.model_size_bytes = 14'000'000;  // a MobileNetV2-sized FP32 file (about 4 ms on the CPU)
    EXPECT_GE(su::heuristic_basis(c).iteration_ms, 10.0);
    c.identity.provider = eb::ProviderKind::npu;
    EXPECT_GE(su::heuristic_basis(c).cold_init_s, 2.0);  // NPU cold sessions take about 2 s
}

// ---------------------------------------------------------------------- manifest

namespace {

su::SuiteManifest sample_manifest() {
    su::SuiteManifest m;
    m.suite_name = "s";
    m.config_file = "C:\\x\\s.json";
    m.config = json{{"name", "s"}};
    m.config_sha256 = su::config_hash(m.config);
    m.created_utc = m.updated_utc = "2026-11-27T00:00:00.000Z";
    m.tool = {{"name", "edgebench"},     {"version", "0.1.0"},
              {"git_commit", nullptr},   {"git_dirty", false},
              {"build_type", "Release"}, {"results_schema_version", "1.2.0"}};
    m.rotation = "rotated";
    m.repetitions = 1;
    m.invocations.push_back(
        {"2026-11-27T00:00:00.000Z", std::nullopt, false, false, false, std::nullopt});
    su::PreflightAttempt a;
    a.started_utc = "2026-11-27T00:00:00.000Z";
    a.passed = true;
    a.checks.push_back({"doctor", su::GateStatus::warn, "doctor exit 0, warn", {{"exit_code", 0}}});
    m.preflight.push_back(a);
    for (int i = 0; i < 3; ++i) {
        su::RunRecord r;
        r.run_id = std::string(63, 'a') + std::to_string(i);
        r.configuration_id = std::string(64, 'c');
        r.label = "x_b1_cpu_fp32";
        r.order_index = i;
        r.repetition = 0;
        r.configuration = {{"run_id_version", 1}};
        m.runs.push_back(r);
    }
    m.runs[0].status = su::RunStatus::done;
    m.runs[0].exit_code = 0;
    m.runs[0].result_file = "runs/a/x.json";
    m.runs[1].status = su::RunStatus::failed;
    m.runs[1].failure = "edgebench run exited 3";
    m.runs[2].status = su::RunStatus::running;
    return m;
}

}  // namespace

TEST(SuiteManifest, RoundTripsThroughJson) {
    const su::SuiteManifest m = sample_manifest();
    const json j = json::parse(su::to_json(m).dump());
    const eb::Result<su::SuiteManifest> back = su::manifest_from_json(j);
    ASSERT_TRUE(back) << back.error().message;
    EXPECT_EQ(json::parse(su::to_json(back.value()).dump()), j);  // same content
    EXPECT_EQ(j["counts"]["done"], 1);
    EXPECT_EQ(j["counts"]["failed"], 1);
    EXPECT_EQ(j["counts"]["running"], 1);
    EXPECT_TRUE(j["publishable"].get<bool>());

    json wrong = j;
    wrong["suite_manifest_version"] = "9.0.0";
    EXPECT_FALSE(su::manifest_from_json(wrong));
    json missing = j;
    missing.erase("runs");
    EXPECT_EQ(su::manifest_from_json(missing).error().kind, "suite_manifest_invalid");
}

TEST(SuiteManifest, SkippedPreflightOrDirtyBuildIsNotPublishable) {
    su::SuiteManifest m = sample_manifest();
    EXPECT_TRUE(m.publishable());
    m.preflight_ever_skipped = true;
    EXPECT_FALSE(m.publishable());
    m.preflight_ever_skipped = false;
    m.tool["git_dirty"] = true;
    EXPECT_FALSE(m.publishable());
    m.tool["git_dirty"] = nullptr;  // not built from git: unknown, not a reason
    EXPECT_TRUE(m.publishable());
    m.preflight.back().checks.clear();  // the gate ran without its doctor check
    EXPECT_FALSE(m.publishable());
}

TEST(SuiteResume, DecidesPerRunStatus) {
    su::RunRecord r;
    auto valid = [](const su::RunRecord&) { return true; };
    auto invalid = [](const su::RunRecord&) { return false; };
    r.status = su::RunStatus::done;
    EXPECT_EQ(su::resume_action(r, valid), su::ResumeAction::keep);
    EXPECT_EQ(su::resume_action(r, invalid), su::ResumeAction::rerun);  // missing or changed
    r.status = su::RunStatus::failed;
    EXPECT_EQ(su::resume_action(r, valid), su::ResumeAction::rerun);
    r.status = su::RunStatus::running;  // interrupted mid-run
    EXPECT_EQ(su::resume_action(r, valid), su::ResumeAction::discard);
    r.status = su::RunStatus::pending;
    EXPECT_EQ(su::resume_action(r, valid), su::ResumeAction::run);
    r.status = su::RunStatus::skipped;
    EXPECT_EQ(su::resume_action(r, valid), su::ResumeAction::run);
    // The validity callback is only asked for done runs.
    int asked = 0;
    for (su::RunStatus s : {su::RunStatus::pending, su::RunStatus::running, su::RunStatus::failed,
                            su::RunStatus::skipped}) {
        r.status = s;
        su::resume_action(r, [&](const su::RunRecord&) { return ++asked, true; });
    }
    EXPECT_EQ(asked, 0);
}

TEST(SuiteGate, FailOnWarnTurnsWarningsIntoFailures) {
    std::vector<su::GateResult> g = {{"doctor", su::GateStatus::warn, "", {}}};
    EXPECT_TRUE(su::gate_passed(g, false));
    EXPECT_FALSE(su::gate_passed(g, true));
    g[0].status = su::GateStatus::fail;
    EXPECT_FALSE(su::gate_passed(g, false));
    g[0].status = su::GateStatus::pass;
    EXPECT_TRUE(su::gate_passed(g, true));
    EXPECT_TRUE(su::gate_passed({}, true));
    EXPECT_EQ(su::make_preflight_checks({"doctor"}).size(), 1u);
    EXPECT_EQ(su::make_preflight_checks({"doctor"})[0]->name(), "doctor");
}

TEST(SuitePlanText, ShowsEveryRunAndTheEstimateSource) {
    const su::SuitePlan plan = load(kFixtures / "valid" / "explicit_cpu.json");
    const su::SuiteEstimate e = su::estimate_suite(plan, su::PriorResults{});
    const std::string text = su::format_plan(plan, e);
    for (const su::PlannedRun& r : plan.runs)
        EXPECT_NE(text.find(r.short_id()), std::string::npos);
    EXPECT_NE(text.find("0 of 2 configurations estimated from earlier results, 2 from a heuristic"),
              std::string::npos)
        << text;
}
