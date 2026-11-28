// The publish gate's judgements on injected machine readings, the in-flight
// backoff, retries, publishability, and the gate settings of a configuration.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <numeric>
#include <set>

#include "edgebench/suite.hpp"
#include "edgebench/suite_config.hpp"
#include "edgebench/suite_gate.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
namespace su = edgebench::suite;
using json = nlohmann::json;

namespace {

const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();
const fs::path kFixtures = kSourceDir / "scripts" / "tests" / "fixtures" / "suite";

eb::PowerState ready_power() {
    eb::PowerState p;
    p.source = "ac";
    p.power_mode = "Best performance";
    p.power_mode_guid = "ded574b5-45a0-4f42-8737-46345c09c238";
    p.power_mode_known = true;
    p.battery_saver = false;
    p.power_plan = "Balanced";
    return p;
}

su::IdleReading idle(double ratio, std::vector<su::ProcessActivity> processes = {}) {
    su::IdleReading r;
    r.cpu_ratio = ratio;
    r.window_s = 30;
    r.processes = std::move(processes);
    return r;
}

std::string offender_status(const su::GateResult& g, const std::string& name) {
    for (const auto& o : g.details["offenders"])
        if (o["name"] == name)
            return o["status"];
    return "absent";
}

json read(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return json::parse(in);
}

}  // namespace

// ------------------------------------------------------------------- power

TEST(GatePower, PassesOnlyOnAcWithBestPerformanceAndNoSaver) {
    EXPECT_EQ(su::evaluate_power(ready_power()).status, su::GateStatus::pass);
    EXPECT_TRUE(su::power_ready(ready_power()));

    eb::PowerState battery = ready_power();
    battery.source = "battery";
    EXPECT_EQ(su::evaluate_power(battery).status, su::GateStatus::fail);
    EXPECT_NE(su::evaluate_power(battery).summary.find("not on AC"), std::string::npos);

    eb::PowerState balanced = ready_power();
    balanced.power_mode = "Balanced";
    const su::GateResult g = su::evaluate_power(balanced);
    EXPECT_EQ(g.status, su::GateStatus::fail);
    EXPECT_NE(g.summary.find("Best performance"), std::string::npos);
    EXPECT_EQ(g.details["power_mode_overlay_guid"], "ded574b5-45a0-4f42-8737-46345c09c238");
    EXPECT_EQ(g.details["power_plan"], "Balanced");

    eb::PowerState saver = ready_power();
    saver.battery_saver = true;
    EXPECT_EQ(su::evaluate_power(saver).status, su::GateStatus::fail);
    eb::PowerState unknown;  // nothing could be read
    EXPECT_EQ(su::evaluate_power(unknown).status, su::GateStatus::fail);
    EXPECT_FALSE(su::power_ready(unknown));
}

// ------------------------------------------------------------------- sleep

TEST(GateSleep, NeverOrLongerThanTheSuitePasses) {
    EXPECT_EQ(su::evaluate_sleep({0u, 0u}, 3600).status, su::GateStatus::pass);
    EXPECT_EQ(su::evaluate_sleep({7200u, 0u}, 3600).status, su::GateStatus::pass);
    const su::GateResult g = su::evaluate_sleep({900u, 0u}, 3600);
    EXPECT_EQ(g.status, su::GateStatus::fail);
    EXPECT_NE(g.summary.find("powercfg /change standby-timeout-ac 0"), std::string::npos)
        << g.summary;
    const su::GateResult h = su::evaluate_sleep({0u, 1800u}, 3600);
    EXPECT_EQ(h.status, su::GateStatus::fail);
    EXPECT_NE(h.summary.find("powercfg /change hibernate-timeout-ac 0"), std::string::npos);
    EXPECT_EQ(su::evaluate_sleep({std::nullopt, 0u}, 10).status, su::GateStatus::fail);
    EXPECT_EQ(g.details["required_s"], 3600.0);
}

// -------------------------------------------------------------------- idle

TEST(GateIdle, LoadAboveTheLimitFails) {
    EXPECT_EQ(su::evaluate_idle(idle(0.02), 0.05).status, su::GateStatus::pass);
    const su::GateResult g = su::evaluate_idle(idle(0.08, {{1, "foo.exe", 0.07}}), 0.05);
    EXPECT_EQ(g.status, su::GateStatus::fail);
    EXPECT_EQ(g.details["top_processes"][0]["name"], "foo.exe");
    su::IdleReading unread = idle(0);
    unread.cpu_ratio.reset();
    EXPECT_EQ(su::evaluate_idle(unread, 0.05).status, su::GateStatus::fail);
}

TEST(GateIdle, BusyUpdateAndServicingProcessesFailByName) {
    // Quiet overall, but Windows Update is downloading: fail.
    su::GateResult g = su::evaluate_idle(idle(0.03, {{10, "BackgroundDownload.exe", 0.02}}), 0.05);
    EXPECT_EQ(g.status, su::GateStatus::fail);
    EXPECT_EQ(offender_status(g, "BackgroundDownload.exe"), "fail");
    // Present but idle: a warning.
    g = su::evaluate_idle(idle(0.01, {{11, "TiWorker.exe", 0.0}}), 0.05);
    EXPECT_EQ(g.status, su::GateStatus::warn);
    g = su::evaluate_idle(idle(0.04, {{12, "mscorsvw.exe", 0.03}}), 0.05);
    EXPECT_EQ(offender_status(g, "mscorsvw.exe"), "fail");
    // Names match case-insensitively; same-named processes add up.
    g = su::evaluate_idle(idle(0.01, {{13, "MSMPENG.EXE", 0.012}, {14, "msmpeng.exe", 0.012}}),
                          0.05);
    EXPECT_EQ(offender_status(g, "MsMpEng.exe"), "fail");  // 2.4 %: an active scan
    g = su::evaluate_idle(idle(0.01, {{13, "MsMpEng.exe", 0.001}}), 0.05);
    EXPECT_EQ(g.status, su::GateStatus::pass);  // always running; idle is fine
}

TEST(GateIdle, TheUpdateServiceIsJudgedOnItsHostProcess) {
    su::IdleReading r = idle(0.02, {{900, "svchost.exe", 0.01}});
    r.running_services["wuauserv"] = 900;
    EXPECT_EQ(offender_status(su::evaluate_idle(r, 0.05), "wuauserv"), "fail");
    r.processes[0].share = 0.0;
    EXPECT_EQ(offender_status(su::evaluate_idle(r, 0.05), "wuauserv"), "warn");
}

TEST(GateIdle, OpenInteractiveApplicationsWarn) {
    const su::GateResult g = su::evaluate_idle(idle(0.02, {{20, "Code.exe", 0.004}}), 0.05);
    EXPECT_EQ(g.status, su::GateStatus::warn);
    EXPECT_EQ(offender_status(g, "Code.exe"), "warn");
    EXPECT_NE(g.summary.find("Code.exe"), std::string::npos);
}

TEST(GateIdle, TheOffenderListCoversEveryGroup) {
    std::set<std::string> groups;
    for (const su::Offender& o : su::known_offenders())
        groups.insert(std::string(o.group));
    EXPECT_EQ(groups, (std::set<std::string>{"windows_update", "servicing", "antivirus", "indexer",
                                             "interactive"}));
}

// --------------------------------------------------------------- NPU users

TEST(GateNpuUsers, OtherUsersWarnAndNothingFails) {
    su::NpuUserScan scan;
    scan.scanned = 120;
    scan.inaccessible = 30;
    su::GateResult g = su::evaluate_npu_users(scan);
    EXPECT_EQ(g.status, su::GateStatus::pass);
    EXPECT_EQ(g.details["processes_not_inspectable"], 30);
    scan.users.push_back({4242, "python.exe", {"qnnhtp.dll"}});
    g = su::evaluate_npu_users(scan);
    EXPECT_EQ(g.status, su::GateStatus::warn);
    EXPECT_NE(g.summary.find("python.exe (pid 4242)"), std::string::npos);
}

// ---------------------------------------------------------- sustained load

TEST(GateSustained, WindowsFollowTheMeasuredTime) {
    // 10 ms per inference: a 100 ms window holds 10 inferences.
    std::vector<double> lat(35, 10.0);
    for (int i = 10; i < 20; ++i)
        lat[i] = 12.5;  // slower in the second window (keeps 10 x 12.5 >= 100 ms per window)
    const std::vector<double> s = su::windowed_p50(lat, 100.0);
    ASSERT_EQ(s.size(), 3u);  // 10 + 8 + 10 inferences; the partial rest is dropped
    EXPECT_DOUBLE_EQ(s[0], 10.0);
    EXPECT_DOUBLE_EQ(s[1], 12.5);
    EXPECT_DOUBLE_EQ(s[2], 10.0);
    EXPECT_FALSE(su::drift_ratio(std::vector<double>{1.0}));
    EXPECT_NEAR(*su::drift_ratio(std::vector<double>{2.0, 3.0, 2.2}), 0.1, 1e-12);
    EXPECT_TRUE(su::windowed_p50(lat, 0).empty());
}

TEST(GateSustained, DriftAboveTheLimitFails) {
    su::SustainedLeg steady{"cpu", 180, std::nullopt, {12.9, 13.0, 12.95}, {}};
    su::SustainedLeg throttled{"npu", 120, std::nullopt, {0.35, 0.36, 0.38}, {}};
    EXPECT_EQ(su::evaluate_sustained({steady}, 30, 0.05).status, su::GateStatus::pass);
    const su::GateResult g = su::evaluate_sustained({steady, throttled}, 30, 0.05);
    EXPECT_EQ(g.status, su::GateStatus::fail);  // +8.6 %
    EXPECT_NEAR(g.details["legs"][1]["drift_ratio"].get<double>(), 0.38 / 0.35 - 1, 1e-12);
    EXPECT_EQ(g.details["legs"][1]["p50_series_ms"].size(), 3u);
    // Faster at the end (warming up) is not throttling.
    su::SustainedLeg faster{"cpu", 180, std::nullopt, {14.0, 13.0}, {}};
    EXPECT_EQ(su::evaluate_sustained({faster}, 30, 0.05).status, su::GateStatus::pass);
    su::SustainedLeg short_leg{"cpu", 20, std::nullopt, {13.0}, {}};
    EXPECT_EQ(su::evaluate_sustained({short_leg}, 30, 0.05).status, su::GateStatus::fail);
    su::SustainedLeg failed{"npu", 120, std::string("the calibration run failed"), {}, {}};
    EXPECT_EQ(su::evaluate_sustained({failed}, 30, 0.05).status, su::GateStatus::fail);
    EXPECT_EQ(su::evaluate_sustained({}, 30, 0.05).status, su::GateStatus::fail);
}

// ---------------------------------------------------------------- in flight

TEST(GateInFlight, BackoffDoublesAndStopsAtTheLimit) {
    EXPECT_EQ(su::backoff_schedule(600), (std::vector<double>{30, 60, 120, 240, 150}));
    EXPECT_EQ(su::backoff_schedule(90), (std::vector<double>{30, 60}));
    EXPECT_EQ(su::backoff_schedule(20), (std::vector<double>{20}));
    EXPECT_TRUE(su::backoff_schedule(0).empty());
    const std::vector<double> w = su::backoff_schedule(3600);
    EXPECT_DOUBLE_EQ(std::accumulate(w.begin(), w.end(), 0.0), 3600);
}

TEST(GateInFlight, OnlyLoadAndSpreadWarningsAreRetried) {
    su::RunRecord r;
    r.status = su::RunStatus::done;
    r.warnings = {"latency_spikes_high"};
    EXPECT_FALSE(su::retry_worthy(r));
    r.warnings.push_back("background_cpu_load_high");
    EXPECT_TRUE(su::retry_worthy(r));
    r.warnings = {"trial_spread_high"};
    EXPECT_TRUE(su::retry_worthy(r));
    r.status = su::RunStatus::failed;
    EXPECT_FALSE(su::retry_worthy(r));
}

// ----------------------------------------------------------- publishability

TEST(GatePublishable, EveryGateCheckMustPass) {
    su::SuiteManifest m;
    m.tool = {{"git_dirty", false}};
    su::PreflightAttempt a;
    a.passed = true;
    for (const std::string& name : su::publish_gate_checks())
        a.checks.push_back({name, su::GateStatus::pass, "", {}});
    m.preflight.push_back(a);
    EXPECT_TRUE(m.gate_passed());
    EXPECT_TRUE(m.publishable());

    m.preflight.back().checks[3].status = su::GateStatus::warn;  // e.g. idle: Code.exe open
    EXPECT_FALSE(m.gate_passed());
    EXPECT_FALSE(m.publishable());
    ASSERT_FALSE(m.not_publishable_reasons().empty());
    EXPECT_NE(m.not_publishable_reasons()[0].find("warn"), std::string::npos);

    m.preflight.back().checks[3].status = su::GateStatus::pass;
    m.gate_lost = "not on AC power";
    EXPECT_TRUE(m.gate_passed());
    EXPECT_FALSE(m.publishable());

    m.gate_lost.reset();
    m.preflight.back().checks.pop_back();  // sustained_load did not run
    EXPECT_FALSE(m.publishable());
}

// ------------------------------------------------------------ configuration

TEST(GateConfig, DefaultsAndOverrides) {
    json doc = read(kFixtures / "valid" / "explicit_cpu.json");
    doc["preflight"] = {
        {"checks", {"power", "idle", "sustained_load"}},
        {"idle", {{"window_s", 10}, {"max_cpu_ratio", 0.08}}},
        {"sustained_load",
         {{"window_s", 5},
          {"max_drift_ratio", 0.1},
          {"legs",
           {{{"path", "../../../../../engine/tests/fixtures/models/fixed_float32.onnx"},
             {"provider", "cpu"},
             {"precision", "fp32"},
             {"duration_s", 12}}}}}}};
    doc["in_flight"] = {{"load_sample_s", 3}, {"max_wait_s", 60}, {"on_gate_loss", "stop"}};
    doc["retry_flagged"] = 2;
    const su::LoadOutcome o = su::plan_suite(doc, kFixtures / "valid" / "explicit_cpu.json");
    std::string problems;
    for (const su::SuiteConfigError& e : o.errors)
        problems += e.pointer + ": " + e.message + " [" + e.kind + "]\n";
    ASSERT_TRUE(o.plan) << problems;
    const su::SuitePlan& p = *o.plan;
    EXPECT_EQ(p.preflight.checks, (std::vector<std::string>{"power", "idle", "sustained_load"}));
    EXPECT_DOUBLE_EQ(p.preflight.idle_window_s, 10);
    EXPECT_DOUBLE_EQ(p.preflight.idle_max_cpu_ratio, 0.08);
    EXPECT_DOUBLE_EQ(p.preflight.sustained_window_s, 5);
    EXPECT_DOUBLE_EQ(p.in_flight.load_sample_s, 3);
    EXPECT_EQ(p.in_flight.on_gate_loss, "stop");
    EXPECT_EQ(p.retry_flagged, 2);
    ASSERT_EQ(p.sustained.size(), 1u);
    EXPECT_DOUBLE_EQ(p.sustained[0].sustained_duration_s, 12);
    EXPECT_EQ(p.sustained[0].params.trials, 1);
    EXPECT_EQ(p.configurations.size(), 2u);  // the leg is not a planned run

    // The defaults: the full gate and two ResNet-18 legs.
    const su::PreflightSettings d;
    EXPECT_EQ(d.checks, su::gate_check_names());
    EXPECT_DOUBLE_EQ(d.idle_max_cpu_ratio, 0.05);
    ASSERT_EQ(d.sustained_legs.size(), 2u);
    EXPECT_EQ(d.sustained_legs[0].identity.model, "resnet18");
    EXPECT_DOUBLE_EQ(d.sustained_legs[0].duration_s, 180);
    EXPECT_EQ(d.sustained_legs[1].identity.provider, eb::ProviderKind::npu);
    EXPECT_DOUBLE_EQ(d.sustained_legs[1].duration_s, 120);
}

TEST(GateConfig, BadSettingsAreRejected) {
    const fs::path file = kFixtures / "valid" / "explicit_cpu.json";
    for (const json& bad :
         {json{{"preflight", {{"checks", {"thermals"}}}}},
          json{{"preflight", {{"idle", {{"max_cpu_ratio", 2}}}}}},
          json{{"in_flight", {{"on_gate_loss", "ignore"}}}}, json{{"retry_flagged", 9}},
          json{{"preflight",
                {{"sustained_load",
                  {{"legs",
                    {{{"provider", "cpu"}, {"precision", "fp32"}, {"duration_s", 60}}}}}}}}}}) {
        json doc = read(file);
        doc.update(bad);
        EXPECT_FALSE(su::plan_suite(doc, file).plan) << bad.dump();
    }
}
