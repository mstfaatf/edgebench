// Hardware and OS information (results schema: system.*).
//
// Each section has its own query function, so new sections (power state,
// background load, memory use) can be added without touching the others.
// Values that cannot be determined are empty optionals, never guesses.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

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

struct SystemInfo {
    OsInfo os;
    Architecture process_architecture = Architecture::unknown;
    CpuInfo cpu;
    std::optional<std::uint64_t> memory_total_bytes;  // physical memory visible to the OS
    DeviceInfo device;

    /// True if the process runs under emulation (e.g. an x64 build on ARM64 Windows).
    bool emulated() const;
};

OsInfo query_os_info();
/// Architecture of the running process (the machine type it was built for).
Architecture query_process_architecture();
CpuInfo query_cpu_info();
std::optional<std::uint64_t> query_memory_total_bytes();
DeviceInfo query_device_info();

SystemInfo query_system_info();

/// The results schema's `system` block for the fields above (power is not queried yet).
nlohmann::ordered_json to_json(const SystemInfo& info);

}  // namespace edgebench
