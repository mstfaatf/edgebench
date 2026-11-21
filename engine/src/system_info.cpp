#ifndef _WIN32
#error "system_info.cpp supports Windows only: it reads the registry and Win32 system APIs."
#endif

#include "edgebench/system_info.hpp"

#include <algorithm>
#include <cstdio>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// windows.h first
#include <powrprof.h>
#include <psapi.h>

namespace edgebench {

namespace {

std::string to_utf8(std::wstring_view w) {
    if (w.empty())
        return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0,
                                      nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), n, nullptr,
                        nullptr);
    return out;
}

std::optional<std::string> reg_string(const wchar_t* key, const wchar_t* value) {
    DWORD bytes = 0;
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, value, RRF_RT_REG_SZ, nullptr, nullptr, &bytes) !=
            ERROR_SUCCESS ||
        bytes < sizeof(wchar_t)) {
        return std::nullopt;
    }
    std::wstring buf(bytes / sizeof(wchar_t), L'\0');
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, value, RRF_RT_REG_SZ, nullptr, buf.data(), &bytes) !=
        ERROR_SUCCESS) {
        return std::nullopt;
    }
    buf.resize(bytes / sizeof(wchar_t));
    while (!buf.empty() && (buf.back() == L'\0' || buf.back() == L' '))
        buf.pop_back();
    if (buf.empty())
        return std::nullopt;
    return to_utf8(buf);
}

std::optional<DWORD> reg_dword(const wchar_t* key, const wchar_t* value) {
    DWORD data = 0;
    DWORD bytes = sizeof(data);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, value, RRF_RT_REG_DWORD, nullptr, &data, &bytes) !=
        ERROR_SUCCESS) {
        return std::nullopt;
    }
    return data;
}

Architecture from_machine(USHORT machine) {
    switch (machine) {
        case IMAGE_FILE_MACHINE_ARM64:
            return Architecture::arm64;
        case IMAGE_FILE_MACHINE_AMD64:
            return Architecture::x64;
        case IMAGE_FILE_MACHINE_I386:
            return Architecture::x86;
        default:
            return Architecture::unknown;
    }
}

// The architecture this translation unit was compiled for.
constexpr Architecture compiled_architecture() {
#if defined(_M_ARM64EC)
    return Architecture::arm64ec;
#elif defined(_M_ARM64)
    return Architecture::arm64;
#elif defined(_M_X64)
    return Architecture::x64;
#elif defined(_M_IX86)
    return Architecture::x86;
#else
    return Architecture::unknown;
#endif
}

constexpr const wchar_t* kCurrentVersionKey = L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";

}  // namespace

std::string_view to_string(Architecture arch) {
    switch (arch) {
        case Architecture::arm64:
            return "arm64";
        case Architecture::x64:
            return "x64";
        case Architecture::arm64ec:
            return "arm64ec";
        case Architecture::x86:
            return "x86";
        case Architecture::unknown:
            break;
    }
    return "unknown";
}

bool SystemInfo::emulated() const {
    if (os.architecture == Architecture::unknown || process_architecture == Architecture::unknown)
        return false;
    // ARM64EC code is native ARM64, but it exists to host x64 code: report it too.
    return process_architecture != os.architecture;
}

OsInfo query_os_info() {
    OsInfo os;
    const std::optional<std::string> build = reg_string(kCurrentVersionKey, L"CurrentBuildNumber");
    const std::optional<DWORD> ubr = reg_dword(kCurrentVersionKey, L"UBR");
    if (build)
        os.build = ubr ? *build + "." + std::to_string(*ubr) : *build;
    os.version = reg_string(kCurrentVersionKey, L"DisplayVersion");

    // ProductName still says "Windows 10" on Windows 11; build 22000 and later is Windows 11.
    std::string name = reg_string(kCurrentVersionKey, L"ProductName").value_or("Windows");
    int build_number = 0;
    if (build) {
        try {
            build_number = std::stoi(*build);
        } catch (...) {
            build_number = 0;
        }
    }
    const std::string win10 = "Windows 10";
    if (build_number >= 22000 && name.rfind(win10, 0) == 0)
        name = "Windows 11" + name.substr(win10.size());
    os.name = name;

    USHORT process_machine = IMAGE_FILE_MACHINE_UNKNOWN;
    USHORT native_machine = IMAGE_FILE_MACHINE_UNKNOWN;
    if (IsWow64Process2(GetCurrentProcess(), &process_machine, &native_machine))
        os.architecture = from_machine(native_machine);
    return os;
}

Architecture query_process_architecture() {
    // IsWow64Process2 reports a process machine only for WOW64 (32-bit) processes;
    // x64 and ARM64EC processes on ARM64 are not WOW64, so fall back to the build target.
    USHORT process_machine = IMAGE_FILE_MACHINE_UNKNOWN;
    USHORT native_machine = IMAGE_FILE_MACHINE_UNKNOWN;
    if (IsWow64Process2(GetCurrentProcess(), &process_machine, &native_machine) &&
        process_machine != IMAGE_FILE_MACHINE_UNKNOWN) {
        return from_machine(process_machine);
    }
    return compiled_architecture();
}

CpuInfo query_cpu_info() {
    CpuInfo cpu;
    cpu.name =
        reg_string(L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", L"ProcessorNameString");

    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && bytes > 0) {
        std::vector<unsigned char> buf(bytes);
        auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &bytes)) {
            int cores = 0;
            int logical = 0;
            for (DWORD offset = 0; offset < bytes;) {
                const auto* entry =
                    reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() +
                                                                                     offset);
                if (entry->Relationship == RelationProcessorCore) {
                    ++cores;
                    for (WORD g = 0; g < entry->Processor.GroupCount; ++g) {
                        KAFFINITY mask = entry->Processor.GroupMask[g].Mask;
                        while (mask) {
                            logical += static_cast<int>(mask & 1);
                            mask >>= 1;
                        }
                    }
                }
                offset += entry->Size;
            }
            if (cores > 0)
                cpu.physical_cores = cores;
            if (logical > 0)
                cpu.logical_processors = logical;
        }
    }
    return cpu;
}

std::optional<std::uint64_t> query_memory_total_bytes() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status))
        return std::nullopt;
    return status.ullTotalPhys;
}

DeviceInfo query_device_info() {
    constexpr const wchar_t* kBios = L"HARDWARE\\DESCRIPTION\\System\\BIOS";
    return {reg_string(kBios, L"SystemManufacturer"), reg_string(kBios, L"SystemProductName")};
}

SystemInfo query_system_info() {
    SystemInfo info;
    info.os = query_os_info();
    info.process_architecture = query_process_architecture();
    info.cpu = query_cpu_info();
    info.memory_total_bytes = query_memory_total_bytes();
    info.device = query_device_info();
    return info;
}

nlohmann::ordered_json to_json(const SystemInfo& s) {
    using json = nlohmann::ordered_json;
    auto opt = [](const auto& v) { return v ? json(*v) : json(nullptr); };
    auto arch = [](Architecture a) {
        return a == Architecture::unknown ? json(nullptr) : json(to_string(a));
    };
    return {
        {"os",
         {{"name", s.os.name},
          {"version", opt(s.os.version)},
          {"build", opt(s.os.build)},
          {"architecture", arch(s.os.architecture)}}},
        {"process_architecture", arch(s.process_architecture)},
        {"cpu",
         {{"name", opt(s.cpu.name)},
          {"physical_cores", opt(s.cpu.physical_cores)},
          {"logical_processors", opt(s.cpu.logical_processors)}}},
        {"memory_total_bytes", opt(s.memory_total_bytes)},
        {"device", {{"manufacturer", opt(s.device.manufacturer)}, {"model", opt(s.device.model)}}},
        {"power", s.power ? to_json(*s.power) : to_json(PowerState{})},
        {"load", s.load ? to_json(*s.load) : json(nullptr)},
    };
}

nlohmann::ordered_json to_json(const PowerState& p) {
    using json = nlohmann::ordered_json;
    auto opt = [](const auto& v) { return v ? json(*v) : json(nullptr); };
    return {
        {"source", opt(p.source)},
        {"power_mode", opt(p.power_mode)},
        {"power_plan", opt(p.power_plan)},
        {"battery_percent", opt(p.battery_percent)},
        {"battery_saver", opt(p.battery_saver)},
    };
}

nlohmann::ordered_json to_json(const LoadSample& l) {
    return {{"cpu_utilization_ratio", l.cpu_utilization_ratio},
            {"sample_window_s", l.sample_window_s}};
}

// ----------------------------------------------------------------- pure helpers

std::optional<std::string> decode_ac_line_status(std::uint8_t ac_line_status) {
    switch (ac_line_status) {
        case 0:
            return "battery";
        case 1:
            return "ac";
        default:
            return std::nullopt;
    }
}

std::optional<int> decode_battery_percent(std::uint8_t life_percent, std::uint8_t battery_flag) {
    if (life_percent == 255 || (battery_flag != 255 && (battery_flag & 128)))
        return std::nullopt;
    return std::min<int>(life_percent, 100);
}

bool decode_battery_saver(std::uint8_t system_status_flag) {
    return (system_status_flag & 1) != 0;
}

std::optional<std::string> power_mode_name(std::string_view overlay_guid) {
    // Windows 11 power mode overlays (Settings > System > Power > Power mode).
    if (overlay_guid == "961cc777-2547-4f9d-8174-7d86181b8a7a")
        return "Best power efficiency";
    if (overlay_guid == "00000000-0000-0000-0000-000000000000")
        return "Balanced";
    if (overlay_guid == "3af9b8d9-7c97-431d-ad78-34a8bfea439f")
        return "Better performance";
    if (overlay_guid == "ded574b5-45a0-4f42-8737-46345c09c238")
        return "Best performance";
    return std::nullopt;
}

std::vector<std::string> power_changes(const PowerState& a, const PowerState& b) {
    std::vector<std::string> changed;
    if (a.source != b.source)
        changed.push_back("source");
    if (a.power_mode != b.power_mode)
        changed.push_back("power_mode");
    if (a.power_plan_guid != b.power_plan_guid || a.power_plan != b.power_plan)
        changed.push_back("power_plan");
    if (a.battery_saver != b.battery_saver)
        changed.push_back("battery_saver");
    return changed;
}

std::optional<double> cpu_utilization(const CpuTimes& a, const CpuTimes& b) {
    if (b.idle < a.idle || b.kernel < a.kernel || b.user < a.user)
        return std::nullopt;  // counters went backwards: not two readings of one system
    const std::uint64_t idle = b.idle - a.idle;
    const std::uint64_t kernel = b.kernel - a.kernel;  // includes idle
    const std::uint64_t user = b.user - a.user;
    const std::uint64_t total = kernel + user;
    if (total == 0)
        return std::nullopt;
    const std::uint64_t busy = (kernel >= idle ? kernel - idle : 0) + user;
    return std::clamp(static_cast<double>(busy) / static_cast<double>(total), 0.0, 1.0);
}

// ----------------------------------------------------------------- queries

namespace {

std::string guid_string(const GUID& g) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  static_cast<unsigned long>(g.Data1), g.Data2, g.Data3, g.Data4[0], g.Data4[1],
                  g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7]);
    return buf;
}

std::uint64_t filetime_u64(const FILETIME& f) {
    return (static_cast<std::uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
}

// PowerGetEffectiveOverlayScheme is exported by powrprof.dll but not declared in
// the Windows SDK headers, so it is looked up at run time (from System32 only).
std::optional<GUID> effective_overlay_scheme() {
    using Fn = DWORD(WINAPI*)(GUID*);
    HMODULE powrprof = LoadLibraryExW(L"powrprof.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!powrprof)
        return std::nullopt;
    const auto fn =
        reinterpret_cast<Fn>(GetProcAddress(powrprof, "PowerGetEffectiveOverlayScheme"));
    std::optional<GUID> result;
    GUID g{};
    if (fn && fn(&g) == ERROR_SUCCESS)
        result = g;
    FreeLibrary(powrprof);
    return result;
}

}  // namespace

PowerState query_power_state() {
    PowerState p;
    SYSTEM_POWER_STATUS status{};
    if (GetSystemPowerStatus(&status)) {
        p.source = decode_ac_line_status(status.ACLineStatus);
        p.battery_percent = decode_battery_percent(status.BatteryLifePercent, status.BatteryFlag);
        p.battery_saver = decode_battery_saver(status.SystemStatusFlag);
    }

    GUID* scheme = nullptr;
    if (PowerGetActiveScheme(nullptr, &scheme) == ERROR_SUCCESS && scheme) {
        p.power_plan_guid = guid_string(*scheme);
        DWORD bytes = 0;
        if (PowerReadFriendlyName(nullptr, scheme, nullptr, nullptr, nullptr, &bytes) ==
                ERROR_SUCCESS &&
            bytes >= sizeof(wchar_t)) {
            std::wstring name(bytes / sizeof(wchar_t), L'\0');
            if (PowerReadFriendlyName(nullptr, scheme, nullptr, nullptr,
                                      reinterpret_cast<UCHAR*>(name.data()),
                                      &bytes) == ERROR_SUCCESS) {
                while (!name.empty() && name.back() == L'\0')
                    name.pop_back();
                if (!name.empty())
                    p.power_plan = to_utf8(name);
            }
        }
        LocalFree(scheme);
    }

    if (const std::optional<GUID> overlay = effective_overlay_scheme()) {
        p.power_mode_guid = guid_string(*overlay);
        if (const auto name = power_mode_name(*p.power_mode_guid)) {
            p.power_mode = *name;
            p.power_mode_known = true;
        } else {
            p.power_mode = p.power_mode_guid;  // unknown overlay: record the raw GUID
        }
    }
    return p;
}

CpuTimes query_cpu_times() {
    FILETIME idle{}, kernel{}, user{};
    if (!GetSystemTimes(&idle, &kernel, &user))
        return {};
    return {filetime_u64(idle), filetime_u64(kernel), filetime_u64(user)};
}

LoadSample sample_cpu_load(std::chrono::milliseconds window) {
    using Clock = std::chrono::steady_clock;
    const CpuTimes start = query_cpu_times();
    const Clock::time_point t0 = Clock::now();
    std::this_thread::sleep_for(window);
    const CpuTimes end = query_cpu_times();
    const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    return {cpu_utilization(start, end).value_or(0.0), seconds};
}

ProcessMemory query_process_memory() {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    ProcessMemory m;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(),
                                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                sizeof(counters))) {
        m.working_set_bytes = counters.WorkingSetSize;
        m.peak_working_set_bytes = counters.PeakWorkingSetSize;
        m.private_bytes = counters.PrivateUsage;
        m.peak_private_bytes = counters.PeakPagefileUsage;
    }
    return m;
}

}  // namespace edgebench
