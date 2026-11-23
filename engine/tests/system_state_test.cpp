// Unit tests for the system-state helpers: power status decoding, power mode
// GUIDs, power-change detection, CPU utilization and the pre-trial idle period.

#include <gtest/gtest.h>

#include "edgebench/benchmark.hpp"
#include "edgebench/doctor.hpp"
#include "edgebench/system_info.hpp"

namespace eb = edgebench;

TEST(PowerStatus, AcLineStatus) {
    EXPECT_EQ(eb::decode_ac_line_status(0), "battery");
    EXPECT_EQ(eb::decode_ac_line_status(1), "ac");
    EXPECT_FALSE(eb::decode_ac_line_status(255).has_value());  // unknown
    EXPECT_FALSE(eb::decode_ac_line_status(7).has_value());
}

TEST(PowerStatus, BatteryPercent) {
    EXPECT_EQ(eb::decode_battery_percent(80, 0), 80);
    EXPECT_EQ(eb::decode_battery_percent(100, 8), 100);              // charging flag
    EXPECT_FALSE(eb::decode_battery_percent(255, 0).has_value());    // unknown percentage
    EXPECT_FALSE(eb::decode_battery_percent(0, 128).has_value());    // no system battery
    EXPECT_FALSE(eb::decode_battery_percent(255, 128).has_value());  // desktop
    EXPECT_EQ(eb::decode_battery_percent(42, 255), 42);  // flag unknown, percentage known
}

TEST(PowerStatus, BatterySaver) {
    EXPECT_FALSE(eb::decode_battery_saver(0));
    EXPECT_TRUE(eb::decode_battery_saver(1));
}

TEST(PowerMode, KnownOverlays) {
    EXPECT_EQ(eb::power_mode_name("961cc777-2547-4f9d-8174-7d86181b8a7a"), "Best power efficiency");
    EXPECT_EQ(eb::power_mode_name("00000000-0000-0000-0000-000000000000"), "Balanced");
    EXPECT_EQ(eb::power_mode_name("3af9b8d9-7c97-431d-ad78-34a8bfea439f"), "Better performance");
    EXPECT_EQ(eb::power_mode_name("ded574b5-45a0-4f42-8737-46345c09c238"), "Best performance");
}

TEST(PowerMode, UnknownOverlays) {
    EXPECT_FALSE(
        eb::power_mode_name("381b4222-f694-41f0-9685-ff5bb260df2e").has_value());  // a plan
    EXPECT_FALSE(
        eb::power_mode_name("DED574B5-45A0-4F42-8737-46345C09C238").has_value());  // not lowercase
    EXPECT_FALSE(eb::power_mode_name("").has_value());
}

TEST(PowerChanges, DetectsEachField) {
    eb::PowerState a;
    a.source = "ac";
    a.power_mode = "Balanced";
    a.power_plan = "Balanced";
    a.power_plan_guid = "381b4222-f694-41f0-9685-ff5bb260df2e";
    a.battery_saver = false;
    a.battery_percent = 80;
    EXPECT_TRUE(eb::power_changes(a, a).empty());

    eb::PowerState b = a;
    b.battery_percent = 79;  // draining or charging is not a change of state
    EXPECT_TRUE(eb::power_changes(a, b).empty());

    b.source = "battery";
    b.power_mode = "Best performance";
    EXPECT_EQ(eb::power_changes(a, b), (std::vector<std::string>{"source", "power_mode"}));

    eb::PowerState c = a;
    c.power_plan_guid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c";
    c.battery_saver = true;
    EXPECT_EQ(eb::power_changes(a, c), (std::vector<std::string>{"power_plan", "battery_saver"}));
}

TEST(CpuUtilization, IdleSystem) {
    // All elapsed time idle: kernel advanced by exactly the idle time.
    EXPECT_EQ(eb::cpu_utilization({100, 200, 50}, {1100, 1200, 50}), 0.0);
}

TEST(CpuUtilization, FullyBusy) {
    // No idle time; kernel and user both advanced.
    EXPECT_EQ(eb::cpu_utilization({0, 0, 0}, {0, 300, 700}), 1.0);
}

TEST(CpuUtilization, HalfBusy) {
    // 1000 units elapsed: 500 idle (counted inside kernel), 200 kernel work, 300 user.
    EXPECT_DOUBLE_EQ(*eb::cpu_utilization({0, 0, 0}, {500, 700, 300}), 0.5);
}

TEST(CpuUtilization, KernelIncludesIdle) {
    // Kernel time = 900 of which 600 is idle; user 100: busy 300 + 100 of 1000.
    EXPECT_DOUBLE_EQ(*eb::cpu_utilization({10, 20, 30}, {610, 920, 130}), 0.4);
}

TEST(CpuUtilization, NoElapsedTimeOrBackwards) {
    EXPECT_FALSE(eb::cpu_utilization({5, 5, 5}, {5, 5, 5}).has_value());
    EXPECT_FALSE(eb::cpu_utilization({10, 10, 10}, {5, 20, 20}).has_value());
}

// Background utilization: system busy time minus the measuring process's own CPU
// time, over the system total ({idle, kernel, user}; kernel includes idle).
// Below, 1000 units elapse: 500 idle, so 500 busy.
TEST(BackgroundCpuUtilization, ZeroOwnTimeIsTheSystemUtilization) {
    EXPECT_DOUBLE_EQ(*eb::background_cpu_utilization({0, 0, 0}, {500, 700, 300}, 40, 40), 0.5);
    EXPECT_DOUBLE_EQ(*eb::background_cpu_utilization({0, 0, 0}, {500, 700, 300}, 40, 40),
                     *eb::cpu_utilization({0, 0, 0}, {500, 700, 300}));
}

TEST(BackgroundCpuUtilization, OwnTimeEqualToAllBusyTimeLeavesNoBackground) {
    EXPECT_EQ(*eb::background_cpu_utilization({0, 0, 0}, {500, 700, 300}, 100, 600), 0.0);
}

TEST(BackgroundCpuUtilization, NormalCaseSubtractsOwnTime) {
    // 500 busy, 200 of them ours: 300 of 1000 from other processes.
    EXPECT_DOUBLE_EQ(*eb::background_cpu_utilization({0, 0, 0}, {500, 700, 300}, 1000, 1200), 0.3);
}

TEST(BackgroundCpuUtilization, NormalizedByTheLogicalProcessorCount) {
    // 1 s on 4 logical processors: the system total is 4 s (40,000,000 units).
    // This process used one full processor (1 s), others 0.5 s: 0.5 / 4.
    const std::uint64_t second = 10'000'000;
    const eb::CpuTimes start{0, 0, 0};
    const eb::CpuTimes end{static_cast<std::uint64_t>(2.5 * second),
                           static_cast<std::uint64_t>(3.0 * second), 1 * second};
    EXPECT_DOUBLE_EQ(*eb::background_cpu_utilization(start, end, 0, second), 0.125);
}

TEST(BackgroundCpuUtilization, OwnTimeAboveBusyTimeClampsToZero) {
    // The counters are updated at different moments: our own time can exceed the
    // system's busy time slightly.
    EXPECT_EQ(*eb::background_cpu_utilization({0, 0, 0}, {500, 700, 300}, 0, 510), 0.0);
}

TEST(BackgroundCpuUtilization, ClockEdgeCases) {
    // No elapsed system time.
    EXPECT_FALSE(eb::background_cpu_utilization({5, 5, 5}, {5, 5, 5}, 0, 0).has_value());
    // System counters going backwards (not two readings of one system).
    EXPECT_FALSE(eb::background_cpu_utilization({10, 10, 10}, {5, 20, 20}, 0, 0).has_value());
    EXPECT_FALSE(eb::background_cpu_utilization({0, 10, 10}, {0, 20, 5}, 0, 0).has_value());
    // Own CPU time going backwards.
    EXPECT_FALSE(eb::background_cpu_utilization({0, 0, 0}, {500, 700, 300}, 10, 5).has_value());
    // Only idle time elapsed, own time zero: no background load.
    EXPECT_EQ(*eb::background_cpu_utilization({0, 0, 0}, {1000, 1000, 0}, 7, 7), 0.0);
}

TEST(BackgroundCpuUtilization, ProcessCpuTimeIsReadable) {
    const std::optional<std::uint64_t> a = eb::query_process_cpu_time();
    ASSERT_TRUE(a.has_value());
    volatile double sink = 0;
    for (int i = 0; i < 2'000'000; ++i)
        sink = sink + 1.0;
    const std::optional<std::uint64_t> b = eb::query_process_cpu_time();
    ASSERT_TRUE(b.has_value());
    EXPECT_GE(*b, *a);
}

TEST(CpuUtilization, LiveSampleIsARatio) {
    const eb::LoadSample s = eb::sample_cpu_load(std::chrono::milliseconds(50));
    EXPECT_GE(s.cpu_utilization_ratio, 0.0);
    EXPECT_LE(s.cpu_utilization_ratio, 1.0);
    EXPECT_GE(s.sample_window_s, 0.045);
}

TEST(IdleBeforeTrial, CooldownOrOneSecondBeforeTheFirstTrial) {
    eb::RunConfig c;
    c.cooldown_ms = 2000;
    EXPECT_EQ(eb::idle_before_trial(c, 0).count(), 2000);
    EXPECT_EQ(eb::idle_before_trial(c, 1).count(), 2000);
    c.cooldown_ms = 0;
    EXPECT_EQ(eb::idle_before_trial(c, 0).count(), 1000);
    EXPECT_EQ(eb::idle_before_trial(c, 1).count(), 0);
    EXPECT_EQ(eb::idle_before_trial(c, 5).count(), 0);
}

TEST(SystemJson, PowerAndLoad) {
    eb::SystemInfo s;
    s.os.name = "Windows 11 Home";
    nlohmann::ordered_json j = eb::to_json(s);
    EXPECT_TRUE(j.at("power").at("source").is_null());  // not queried: every field null
    EXPECT_TRUE(j.at("load").is_null());
    eb::PowerState p;
    p.source = "ac";
    p.power_mode = "Balanced";
    p.battery_percent = 80;
    p.battery_saver = false;
    s.power = p;
    s.load = eb::LoadSample{0.025, 2.0};
    j = eb::to_json(s);
    EXPECT_EQ(j.at("power"), (nlohmann::ordered_json{{"source", "ac"},
                                                     {"power_mode", "Balanced"},
                                                     {"power_plan", nullptr},
                                                     {"battery_percent", 80},
                                                     {"battery_saver", false}}));
    EXPECT_EQ(j.at("load"),
              (nlohmann::ordered_json{{"cpu_utilization_ratio", 0.025}, {"sample_window_s", 2.0}}));
}

// ---------------------------------------------------------------- doctor warnings

namespace {

std::vector<std::string> warning_codes(const eb::SystemInfo& s) {
    std::vector<std::string> codes;
    for (const eb::DoctorWarning& w : eb::system_state_warnings(s))
        codes.push_back(w.code);
    return codes;
}

eb::SystemInfo quiet_system() {
    eb::SystemInfo s;
    s.os.architecture = eb::Architecture::arm64;
    s.process_architecture = eb::Architecture::arm64;
    eb::PowerState p;
    p.source = "ac";
    p.battery_saver = false;
    p.power_mode = "Balanced";
    p.power_mode_known = true;
    s.power = p;
    s.load = eb::LoadSample{0.05, 1.0};
    return s;
}

}  // namespace

TEST(DoctorWarnings, QuietSystemHasNone) {
    EXPECT_TRUE(warning_codes(quiet_system()).empty());
    eb::SystemInfo unknown;  // nothing known: nothing to warn about
    EXPECT_TRUE(warning_codes(unknown).empty());
}

TEST(DoctorWarnings, EachConditionHasItsCode) {
    eb::SystemInfo s = quiet_system();
    s.power->source = "battery";
    EXPECT_EQ(warning_codes(s), std::vector<std::string>{"battery_power"});

    s = quiet_system();
    s.power->battery_saver = true;
    EXPECT_EQ(warning_codes(s), std::vector<std::string>{"battery_saver_on"});

    s = quiet_system();
    s.load->cpu_utilization_ratio = 0.10;  // the limit itself is fine
    EXPECT_TRUE(warning_codes(s).empty());
    s.load->cpu_utilization_ratio = 0.125;
    const std::vector<eb::DoctorWarning> load = eb::system_state_warnings(s);
    ASSERT_EQ(load.size(), 1u);
    EXPECT_EQ(load[0].code, "background_cpu_load_high");
    EXPECT_NE(load[0].message.find("12.5%"), std::string::npos) << load[0].message;
    EXPECT_FALSE(load[0].hint.empty());

    s = quiet_system();
    s.power->power_mode = "01234567-89ab-cdef-0123-456789abcdef";
    s.power->power_mode_known = false;
    EXPECT_EQ(warning_codes(s), std::vector<std::string>{"unknown_power_mode"});

    s = quiet_system();
    s.process_architecture = eb::Architecture::x64;
    const std::vector<eb::DoctorWarning> emu = eb::system_state_warnings(s);
    ASSERT_EQ(emu.size(), 1u);
    EXPECT_EQ(emu[0].code, "process_emulated");
    EXPECT_NE(emu[0].hint.find("arm64"), std::string::npos) << emu[0].hint;
}

TEST(DoctorWarnings, SeveralAtOnceKeepTheirOrder) {
    eb::SystemInfo s = quiet_system();
    s.power->source = "battery";
    s.power->battery_saver = true;
    s.load->cpu_utilization_ratio = 0.5;
    EXPECT_EQ(warning_codes(s), (std::vector<std::string>{"battery_power", "battery_saver_on",
                                                          "background_cpu_load_high"}));
}
