#ifndef _WIN32
#error "system_info.cpp supports Windows only: it reads the registry and Win32 system APIs."
#endif

#include "edgebench/system_info.hpp"

#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

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
    };
}

}  // namespace edgebench
