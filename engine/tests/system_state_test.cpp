// Unit tests for the system-state helpers: power status decoding, power mode
// GUIDs, power-change detection, CPU utilization and the pre-trial idle period.

#include <gtest/gtest.h>

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <thread>

#include "edgebench/benchmark.hpp"
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
    // A lower bound only: the sample never ends early, but a loaded machine can make
    // it end late by any amount.
    EXPECT_GE(s.sample_window_s, 0.045);
}

TEST(LoadAttribution, SharesByProcessOwnAndUnattributed) {
    // total 1000 units over the window; busy 600.
    const std::vector<eb::ProcessCpuTime> start = {
        {0, "Idle", 5000},       {4, "System", 100},     {10, "svchost.exe", 50},
        {11, "svchost.exe", 10}, {20, "MsMpEng.exe", 0}, {30, "edgebench.exe", 200},
        {40, "quiet.exe", 7}};
    const std::vector<eb::ProcessCpuTime> end = {
        {0, "Idle", 5400},       {4, "System", 120},       {10, "svchost.exe", 90},
        {11, "svchost.exe", 30}, {20, "MsMpEng.exe", 150}, {30, "edgebench.exe", 400},
        {40, "quiet.exe", 7},    {50, "new.exe", 25}};  // new.exe started in the window
    const eb::LoadAttribution a = eb::attribute_cpu_load(start, end, 600, 1000, 30, 3);
    // Same-named processes are summed; the idle process and this process are not listed.
    ASSERT_EQ(a.top_processes.size(), 3u);
    EXPECT_EQ(a.top_processes[0], (eb::ProcessShare{"MsMpEng.exe", 0.15}));
    EXPECT_EQ(a.top_processes[1], (eb::ProcessShare{"svchost.exe", 0.06}));
    EXPECT_EQ(a.top_processes[2], (eb::ProcessShare{"new.exe", 0.025}));
    EXPECT_DOUBLE_EQ(a.own_share, 0.2);
    // attributed: 20 + 60 + 150 + 200 + 25 = 455 of 600 busy
    EXPECT_DOUBLE_EQ(a.unattributed_share, 0.145);
    EXPECT_TRUE(eb::attribute_cpu_load(start, end, 0, 0, 30, 3).top_processes.empty());
}

TEST(LoadAttribution, ProcessListIsReadableAndHoldsThisProcess) {
    const auto processes = eb::query_all_process_cpu_times();
    ASSERT_TRUE(processes);
    const DWORD own = GetCurrentProcessId();
    EXPECT_TRUE(std::any_of(processes->begin(), processes->end(),
                            [own](const eb::ProcessCpuTime& p) { return p.pid == own; }));
    EXPECT_GT(processes->size(), 10u);
}

// The measuring process's own CPU time is never background load: a thread of this
// process spinning through the whole sample shows up as own_process_share, not in
// the load (other test processes running in parallel may of course appear among
// the top processes under their own names).
TEST(LoadAttribution, OwnWorkIsNotBackgroundLoad) {
    const eb::LoadProbe probe = eb::start_load_probe();
    std::atomic<bool> stop{false};
    std::thread spin([&stop] {
        volatile double sink = 0;
        while (!stop.load())
            sink = sink + 1.0;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    stop = true;
    spin.join();
    const std::optional<eb::LoadSample> s = eb::finish_load_probe(probe);
    ASSERT_TRUE(s);
    ASSERT_TRUE(s->attribution);
    const double one_core = 1.0 / std::thread::hardware_concurrency();
    EXPECT_GE(s->attribution->own_share, 0.5 * one_core);  // the spinning thread
    // The busy time is split into other processes, this process and the rest, so
    // the load can never include the spinning thread.
    EXPECT_LE(s->cpu_utilization_ratio + s->attribution->own_share, 1.0 + 1e-9);
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
