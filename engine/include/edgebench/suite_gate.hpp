// The publish gate of `edgebench suite` (docs/suite.md, "Pre-flight gate"):
// what each check reads from the machine, and how it judges a reading.
//
// Every judgement is a pure function of a reading (evaluate_*), so it can be
// tested with any machine state; the query_* / read_* / scan_* functions read
// the real machine and never change a setting.
#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/system_info.hpp"

namespace edgebench::suite {

enum class GateStatus { pass, warn, fail };
std::string_view to_string(GateStatus status);

/// The outcome of one pre-flight check.
struct GateResult {
    std::string name;
    GateStatus status = GateStatus::fail;
    std::string summary;
    nlohmann::ordered_json details = nlohmann::ordered_json::object();
};

// ------------------------------------------------------------------- power

/// pass: on AC, the effective power mode overlay is Best performance, battery
/// saver off. fail otherwise (also when a value cannot be read).
GateResult evaluate_power(const PowerState& power);

/// The same condition, for the check before every run.
bool power_ready(const PowerState& power);

// ------------------------------------------------------------------- sleep

/// AC timeouts of the active power scheme, in seconds; 0 means never.
struct SleepSettings {
    std::optional<std::uint32_t> standby_ac_s;    // sleep after
    std::optional<std::uint32_t> hibernate_ac_s;  // hibernate after
};

SleepSettings query_sleep_settings();

/// pass: neither timeout can end the suite (never, or longer than `required_s`:
/// the estimate plus the margin). fail otherwise, naming the powercfg command
/// that changes it.
GateResult evaluate_sleep(const SleepSettings& settings, double required_s);

// -------------------------------------------------------------------- idle

/// One process's share of all CPU time during an idle sample.
struct ProcessActivity {
    std::uint32_t pid = 0;
    std::string name;
    double share = 0;
};

struct IdleReading {
    std::optional<double> cpu_ratio;  // other processes' load (own process excluded); null: unread
    double window_s = 0;
    std::vector<ProcessActivity> processes;  // every other process seen, largest share first
    double unattributed_share = 0;           // busy time no process accounts for
    /// Services of interest that are running, with their host process id.
    std::map<std::string, std::uint32_t> running_services;
};

/// Samples the load of other processes for `window` (this process's CPU time is
/// excluded, like the run's own samples).
IdleReading read_idle(std::chrono::milliseconds window);

/// A process (or service) known to add load (docs/suite.md, "Known offenders").
struct Offender {
    std::string_view name;    // image name, or service name for kind "service"
    std::string_view group;   // windows_update, servicing, antivirus, indexer, interactive
    bool service = false;     // `name` is a service (judged on its host process)
    double active_share = 0;  // at or above this share it is busy
    GateStatus when_active = GateStatus::fail;
    GateStatus when_present = GateStatus::pass;  // present but below active_share
    std::string_view why;
};

/// The list, in one place.
std::span<const Offender> known_offenders();

/// fail: the load is above `max_cpu_ratio` (or unknown), or an offender is busy
/// (see known_offenders()); warn: an offender is present but idle, or an
/// interactive application is open; pass otherwise. Details hold the load, the
/// top processes and every offender found.
GateResult evaluate_idle(const IdleReading& reading, double max_cpu_ratio);

// --------------------------------------------------------------- NPU users

struct NpuUser {
    std::uint32_t pid = 0;
    std::string name;
    std::vector<std::string> modules;  // the QNN modules it has loaded
};

struct NpuUserScan {
    std::vector<NpuUser> users;
    int scanned = 0;       // processes whose modules could be listed
    int inaccessible = 0;  // processes this user may not inspect
};

/// Other processes that have QnnHtp.dll or onnxruntime_providers_qnn.dll loaded
/// (best effort: processes that cannot be opened are counted, not inspected).
NpuUserScan scan_npu_users();

/// warn if another process has the QNN backend or EP loaded; pass otherwise.
/// Never fail: the scan cannot see every process, and a loaded DLL does not
/// mean the NPU is busy.
GateResult evaluate_npu_users(const NpuUserScan& scan);

// ---------------------------------------------------------- sustained load

/// The p50 of consecutive windows of `window_ms` of measured time (the running
/// sum of the latencies); a last partial window is dropped.
std::vector<double> windowed_p50(std::span<const double> latencies_ms, double window_ms);

/// last / first - 1 of a series (positive: slower at the end); nullopt with
/// fewer than 2 windows.
std::optional<double> drift_ratio(std::span<const double> series);

/// One continuous run of the sustained-load check.
struct SustainedLeg {
    std::string label;
    double duration_s = 0;
    std::optional<std::string> failure;  // the run could not be done
    std::vector<double> p50_series_ms;   // windowed_p50()
    nlohmann::ordered_json details = nlohmann::ordered_json::object();
};

/// fail if a leg failed, is too short for two windows, or its last window's
/// p50 is slower than the first by more than `max_drift_ratio`.
GateResult evaluate_sustained(const std::vector<SustainedLeg>& legs, double window_s,
                              double max_drift_ratio);

// ---------------------------------------------------------------- in flight

/// Waits between load samples when the load before a run is too high: 30 s,
/// then doubling, cut so that the waits add up to at most `max_wait_s`.
std::vector<double> backoff_schedule(double max_wait_s);

}  // namespace edgebench::suite
