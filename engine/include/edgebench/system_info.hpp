// Hardware, OS and system state (results schema: system.*, memory.*).
//
// Each section has its own query function, so sections can be added without
// touching the others. Values that cannot be determined are empty optionals,
// never guesses. Nothing here changes any system setting.
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace edgebench {

/// Results-schema architecture names: "arm64", "x64", "arm64ec", "x86".
enum class Architecture { unknown, arm64, x64, arm64ec, x86 };
std::string_view to_string(Architecture arch);

struct OsInfo {
    std::string name;                                   // e.g. "Windows 11 Home"
    std::optional<std::string> version;                 // display version, e.g. "25H2"
    std::optional<std::string> build;                   // e.g. "26200.9974"
    Architecture architecture = Architecture::unknown;  // native OS architecture
};

struct CpuInfo {
    std::optional<std::string> name;
    std::optional<int> physical_cores;
    std::optional<int> logical_processors;
};

struct DeviceInfo {
    std::optional<std::string> manufacturer;
    std::optional<std::string> model;
};

/// Power state (results schema: system.power). Never changes any setting.
struct PowerState {
    std::optional<std::string> source;      // "ac" or "battery"; null if unknown
    std::optional<int> battery_percent;     // null without a battery or if unknown
    std::optional<bool> battery_saver;      // null if unknown
    std::optional<std::string> power_plan;  // friendly name of the active scheme
    std::optional<std::string> power_plan_guid;
    /// Windows power mode (Settings > Power): "Best power efficiency", "Balanced",
    /// "Better performance", "Best performance", or the raw overlay GUID if unknown.
    std::optional<std::string> power_mode;
    std::optional<std::string> power_mode_guid;
    bool power_mode_known = false;  // false: power_mode is a raw GUID (or null)
};

/// System-wide CPU utilization over a window (results schema: system.load).
/// One process's share of all CPU time during a load sample.
struct ProcessShare {
    std::string name;      // image name only (e.g. "MsMpEng.exe"); same-named processes summed
    double cpu_share = 0;  // of the window times the logical processor count
    bool operator==(const ProcessShare&) const = default;
};

/// Where the CPU time of a load sample went (attribute_cpu_load()).
struct LoadAttribution {
    std::vector<ProcessShare> top_processes;  // other processes, largest first
    double own_share = 0;                     // the measuring process (not part of the load)
    double unattributed_share = 0;  // busy time no process accounts for (interrupts, DPCs)
};

struct LoadSample {
    /// CPU load of other processes: the system's busy time minus this process's
    /// own CPU time, over the system's total (0..1).
    double cpu_utilization_ratio = 0;
    double sample_window_s = 0;
    /// Per-process breakdown of the same window; null if the process list could
    /// not be read. Not part of system.load.
    std::optional<LoadAttribution> attribution;
};

/// Process memory counters (results schema: memory; process working set only).
struct ProcessMemory {
    std::uint64_t working_set_bytes = 0;
    std::uint64_t peak_working_set_bytes = 0;
    std::uint64_t private_bytes = 0;       // committed private memory
    std::uint64_t peak_private_bytes = 0;  // peak committed memory (peak pagefile usage)
};

struct SystemInfo {
    OsInfo os;
    Architecture process_architecture = Architecture::unknown;
    CpuInfo cpu;
    std::optional<std::uint64_t> memory_total_bytes;  // physical memory visible to the OS
    DeviceInfo device;
    std::optional<PowerState> power;  // filled by the caller (query_power_state())
    std::optional<LoadSample> load;   // filled by the caller (sample_cpu_load())

    /// True if the process runs under emulation (e.g. an x64 build on ARM64 Windows).
    bool emulated() const;
};

// ---- Pure decoding helpers (unit-tested without the Windows API) ----

/// SYSTEM_POWER_STATUS.ACLineStatus: 0 battery, 1 ac, anything else (255) unknown.
std::optional<std::string> decode_ac_line_status(std::uint8_t ac_line_status);
/// SYSTEM_POWER_STATUS.BatteryLifePercent; null for 255 (unknown) or when
/// BatteryFlag has bit 128 (no system battery).
std::optional<int> decode_battery_percent(std::uint8_t life_percent, std::uint8_t battery_flag);
/// SYSTEM_POWER_STATUS.SystemStatusFlag: 1 = battery saver on.
bool decode_battery_saver(std::uint8_t system_status_flag);
/// Name of a Windows power mode overlay GUID (lowercase, hyphenated), or null if
/// the GUID is not one of the known overlays.
std::optional<std::string> power_mode_name(std::string_view overlay_guid);
/// Names of the fields that differ between two power snapshots, from: source,
/// power_mode, power_plan, battery_saver (battery percent is expected to change).
std::vector<std::string> power_changes(const PowerState& before, const PowerState& after);

/// Cumulative system CPU times, as GetSystemTimes reports them (100 ns units).
/// Kernel time includes idle time.
struct CpuTimes {
    std::uint64_t idle = 0;
    std::uint64_t kernel = 0;
    std::uint64_t user = 0;
};
/// Busy fraction between two readings: ((kernel - idle) + user) / (kernel + user)
/// over the deltas, clamped to 0..1; null if no time elapsed.
std::optional<double> cpu_utilization(const CpuTimes& start, const CpuTimes& end);
/// Background (other processes') utilization between two readings: the system's
/// busy time minus `own` CPU time of the measuring process over the same window
/// (kernel + user from GetProcessTimes, 100 ns units), divided by the system's
/// total time. GetSystemTimes sums over all logical processors, so the total is
/// the window times the logical processor count: own time is normalized by the
/// processor count in the same units. Clamped to 0..1 (own time can slightly
/// exceed the system's busy time because the counters are updated at different
/// moments). Null if no time elapsed or any counter went backwards.
std::optional<double> background_cpu_utilization(const CpuTimes& start, const CpuTimes& end,
                                                 std::uint64_t own_start, std::uint64_t own_end);

/// Cumulative CPU time (kernel + user, 100 ns units) of one process.
struct ProcessCpuTime {
    std::uint32_t pid = 0;
    std::string name;  // image name, UTF-8
    std::uint64_t cpu = 0;
};

/// The CPU time of every process between two process lists (matched by process
/// id and name; a process that started in between counts with all of its time):
/// the `top` other processes by share, the share of `own_pid`, and the busy time
/// (`busy`, from CpuTimes) that no listed process accounts for. Shares are of
/// `total` (the CpuTimes total over the window). The idle process (id 0) is not
/// a process here. Pure.
LoadAttribution attribute_cpu_load(const std::vector<ProcessCpuTime>& start,
                                   const std::vector<ProcessCpuTime>& end, std::uint64_t busy,
                                   std::uint64_t total, std::uint32_t own_pid, std::size_t top);

/// Every process with its CPU time (NtQuerySystemInformation,
/// SystemProcessInformation: one call, no process handles); null on failure.
std::optional<std::vector<ProcessCpuTime>> query_all_process_cpu_times();

/// A load measurement in progress: the counters at its start.
struct LoadProbe {
    CpuTimes system;
    std::optional<std::uint64_t> own;
    std::optional<std::vector<ProcessCpuTime>> processes;
    std::chrono::steady_clock::time_point started;
};

/// Reads the counters a LoadSample needs (system, this process, every process).
LoadProbe start_load_probe();
/// The load of other processes since `probe` started (background_cpu_utilization()),
/// with its attribution (top `top` other processes). Null if a counter could not
/// be read or no time elapsed.
std::optional<LoadSample> finish_load_probe(const LoadProbe& probe, std::size_t top = 5);

OsInfo query_os_info();
/// Architecture of the running process (the machine type it was built for).
Architecture query_process_architecture();
CpuInfo query_cpu_info();
std::optional<std::uint64_t> query_memory_total_bytes();
DeviceInfo query_device_info();

/// OS, architecture, CPU, memory and device (power and load are left empty).
SystemInfo query_system_info();

/// Power source, battery, power plan and power mode, read with
/// GetSystemPowerStatus, PowerGetActiveScheme / PowerReadFriendlyName and
/// PowerGetEffectiveOverlayScheme (the same sources as scripts/common/ort_qnn.py).
PowerState query_power_state();

CpuTimes query_cpu_times();

/// CPU time used so far by the current process, all threads (kernel + user from
/// GetProcessTimes, 100 ns units); null if it cannot be read.
std::optional<std::uint64_t> query_process_cpu_time();

/// Sleeps for `window` and returns the CPU load of other processes over it
/// (start_load_probe() / finish_load_probe()).
LoadSample sample_cpu_load(std::chrono::milliseconds window);

/// Memory counters of the current process (GetProcessMemoryInfo).
ProcessMemory query_process_memory();

/// The results schema's `system` block. power and load are included when set
/// (power fields: source, power_mode, power_plan, battery_percent, battery_saver).
nlohmann::ordered_json to_json(const SystemInfo& info);
nlohmann::ordered_json to_json(const PowerState& power);
nlohmann::ordered_json to_json(const LoadSample& load);

}  // namespace edgebench
