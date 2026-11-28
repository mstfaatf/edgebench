#include "edgebench/suite_gate.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <set>
#include <thread>

#include "edgebench/ort_loader.hpp"
#include "edgebench/stats.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// windows.h first
#include <powrprof.h>
#include <psapi.h>

namespace edgebench::suite {

namespace {

using ojson = nlohmann::ordered_json;

std::string lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::string percent(double ratio, int digits = 1) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f%%", digits, ratio * 100.0);
    return buf;
}

std::string wide_to_utf8(const wchar_t* w, int n = -1) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, w, n, nullptr, 0, nullptr, nullptr);
    if (size <= 0)
        return {};
    std::string out(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, n, out.data(), size, nullptr, nullptr);
    if (n < 0 && !out.empty())
        out.pop_back();
    return out;
}

// The known offenders: where background load usually comes from on this kind of
// machine. Shares are of all CPU time (every logical processor) during the sample.
constexpr std::array<Offender, 17> kOffenders = {{
    {"MoUsoCoreWorker.exe", "windows_update", false, 0.005, GateStatus::fail, GateStatus::warn,
     "Windows Update orchestrator: scans, downloads and installs updates"},
    {"BackgroundDownload.exe", "windows_update", false, 0.005, GateStatus::fail, GateStatus::warn,
     "Windows Update / Store background download"},
    {"TiWorker.exe", "windows_update", false, 0.005, GateStatus::fail, GateStatus::warn,
     "Windows Modules Installer worker: installs updates and components"},
    {"TrustedInstaller.exe", "windows_update", false, 0.005, GateStatus::fail, GateStatus::warn,
     "Windows Modules Installer: servicing in progress"},
    {"wuauserv", "windows_update", true, 0.005, GateStatus::fail, GateStatus::warn,
     "Windows Update service (judged on the CPU time of the process hosting it)"},
    {"OfficeClickToRun.exe", "windows_update", false, 0.005, GateStatus::fail, GateStatus::pass,
     "Microsoft 365 updates (always running; busy while it updates Office)"},
    {"mscorsvw.exe", "servicing", false, 0.005, GateStatus::fail, GateStatus::warn,
     ".NET native image generation, queued after .NET or Windows updates"},
    {"ngen.exe", "servicing", false, 0.005, GateStatus::fail, GateStatus::warn,
     ".NET native image generation"},
    {"ngentask.exe", "servicing", false, 0.005, GateStatus::fail, GateStatus::warn,
     ".NET native image generation task"},
    {"MsMpEng.exe", "antivirus", false, 0.02, GateStatus::fail, GateStatus::pass,
     "Microsoft Defender antimalware engine (always running; busy during a scan)"},
    {"SearchIndexer.exe", "indexer", false, 0.01, GateStatus::warn, GateStatus::pass,
     "Windows Search indexer (always running; busy while indexing)"},
    {"SearchProtocolHost.exe", "indexer", false, 0.01, GateStatus::warn, GateStatus::pass,
     "Windows Search indexer worker"},
    {"Code.exe", "interactive", false, 0.0, GateStatus::warn, GateStatus::warn,
     "Visual Studio Code: editors, language servers and extensions run in the background"},
    {"chrome.exe", "interactive", false, 0.0, GateStatus::warn, GateStatus::warn,
     "Google Chrome: tabs keep running scripts"},
    {"msedge.exe", "interactive", false, 0.0, GateStatus::warn, GateStatus::warn,
     "Microsoft Edge: tabs keep running scripts (also started in the background by Windows)"},
    {"Teams.exe", "interactive", false, 0.0, GateStatus::warn, GateStatus::warn,
     "Microsoft Teams (classic)"},
    {"ms-teams.exe", "interactive", false, 0.0, GateStatus::warn, GateStatus::warn,
     "Microsoft Teams"},
}};

GateStatus worse(GateStatus a, GateStatus b) {
    return static_cast<int>(a) >= static_cast<int>(b) ? a : b;
}

// GUIDs of the sleep settings (powrprof; defined here so no initguid is needed).
constexpr GUID kSleepSubgroup = {
    0x238C9FA8, 0x0AAD, 0x41ED, {0x83, 0xF4, 0x97, 0xBE, 0x24, 0x2C, 0x8F, 0x20}};
constexpr GUID kStandbyTimeout = {
    0x29F6C1DB, 0x86DA, 0x48C5, {0x9F, 0xDB, 0xF2, 0xB6, 0x7B, 0x1F, 0x44, 0xDA}};
constexpr GUID kHibernateTimeout = {
    0x9D7815A6, 0x7EE4, 0x497E, {0x88, 0x88, 0x51, 0x5A, 0x05, 0xF0, 0x23, 0x64}};

}  // namespace

std::string_view to_string(GateStatus s) {
    return s == GateStatus::pass ? "pass" : s == GateStatus::warn ? "warn" : "fail";
}

// ------------------------------------------------------------------- power

bool power_ready(const PowerState& p) {
    return p.source == "ac" && p.power_mode == "Best performance" && p.battery_saver == false;
}

GateResult evaluate_power(const PowerState& p) {
    GateResult g;
    g.name = "power";
    g.details = {
        {"source", p.source ? ojson(*p.source) : ojson(nullptr)},
        {"power_mode", p.power_mode ? ojson(*p.power_mode) : ojson(nullptr)},
        {"power_mode_overlay_guid", p.power_mode_guid ? ojson(*p.power_mode_guid) : ojson(nullptr)},
        {"power_plan", p.power_plan ? ojson(*p.power_plan) : ojson(nullptr)},
        {"power_plan_guid", p.power_plan_guid ? ojson(*p.power_plan_guid) : ojson(nullptr)},
        {"battery_saver", p.battery_saver ? ojson(*p.battery_saver) : ojson(nullptr)},
        {"battery_percent", p.battery_percent ? ojson(*p.battery_percent) : ojson(nullptr)},
        {"required", "AC power, power mode Best performance, battery saver off"}};
    std::vector<std::string> problems;
    if (p.source != "ac")
        problems.push_back("not on AC power (" + p.source.value_or("unknown") + ")");
    if (p.power_mode != "Best performance")
        problems.push_back("power mode is " + p.power_mode.value_or("unknown") +
                           ", not Best performance (Settings > System > Power)");
    if (p.battery_saver != false)
        problems.push_back(p.battery_saver ? "battery saver is on" : "battery saver state unknown");
    g.status = problems.empty() ? GateStatus::pass : GateStatus::fail;
    if (problems.empty()) {
        g.summary = "AC, Best performance, battery saver off (plan " +
                    p.power_plan.value_or("unknown") + ")";
    } else {
        for (const std::string& s : problems)
            g.summary += (g.summary.empty() ? "" : "; ") + s;
    }
    return g;
}

// ------------------------------------------------------------------- sleep

SleepSettings query_sleep_settings() {
    SleepSettings s;
    GUID* scheme = nullptr;
    if (PowerGetActiveScheme(nullptr, &scheme) != ERROR_SUCCESS)
        return s;
    DWORD value = 0;
    if (PowerReadACValueIndex(nullptr, scheme, &kSleepSubgroup, &kStandbyTimeout, &value) ==
        ERROR_SUCCESS)
        s.standby_ac_s = value;
    if (PowerReadACValueIndex(nullptr, scheme, &kSleepSubgroup, &kHibernateTimeout, &value) ==
        ERROR_SUCCESS)
        s.hibernate_ac_s = value;
    LocalFree(scheme);
    return s;
}

GateResult evaluate_sleep(const SleepSettings& s, double required_s) {
    GateResult g;
    g.name = "sleep";
    auto value = [](const std::optional<std::uint32_t>& v) {
        return v ? ojson(*v) : ojson(nullptr);
    };
    g.details = {{"standby_timeout_ac_s", value(s.standby_ac_s)},
                 {"hibernate_timeout_ac_s", value(s.hibernate_ac_s)},
                 {"required_s", required_s},
                 {"note", "0 means never. Closing the lid can still put the machine to sleep."}};
    std::vector<std::string> problems;
    auto judge = [&](const std::optional<std::uint32_t>& v, const char* what, const char* option) {
        if (!v) {
            problems.push_back(std::string("the AC ") + what + " timeout cannot be read");
            return;
        }
        if (*v != 0 && *v < required_s) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "the AC %s timeout is %u s, shorter than the suite needs (%.0f s); run "
                          "'powercfg /change %s 0' (never)",
                          what, *v, required_s, option);
            problems.push_back(buf);
        }
    };
    judge(s.standby_ac_s, "sleep", "standby-timeout-ac");
    judge(s.hibernate_ac_s, "hibernate", "hibernate-timeout-ac");
    g.status = problems.empty() ? GateStatus::pass : GateStatus::fail;
    if (problems.empty()) {
        auto text = [](const std::optional<std::uint32_t>& v) {
            return *v == 0 ? std::string("never") : std::to_string(*v) + " s";
        };
        g.summary = "AC sleep " + text(s.standby_ac_s) + ", hibernate " + text(s.hibernate_ac_s);
    } else {
        for (const std::string& p : problems)
            g.summary += (g.summary.empty() ? "" : "; ") + p;
    }
    return g;
}

// -------------------------------------------------------------------- idle

std::span<const Offender> known_offenders() {
    return kOffenders;
}

IdleReading read_idle(std::chrono::milliseconds window) {
    IdleReading r;
    // The services an offender may name, with their host process.
    std::set<std::string> services;
    for (const Offender& o : kOffenders) {
        if (o.service)
            services.insert(std::string(o.name));
    }
    if (SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT)) {
        for (const std::string& name : services) {
            const std::wstring w(name.begin(), name.end());
            if (SC_HANDLE svc = OpenServiceW(scm, w.c_str(), SERVICE_QUERY_STATUS)) {
                SERVICE_STATUS_PROCESS st{};
                DWORD needed = 0;
                if (QueryServiceStatusEx(svc, SC_STATUS_PROCESS_INFO, reinterpret_cast<LPBYTE>(&st),
                                         sizeof(st), &needed) &&
                    st.dwCurrentState == SERVICE_RUNNING)
                    r.running_services[name] = st.dwProcessId;
                CloseServiceHandle(svc);
            }
        }
        CloseServiceHandle(scm);
    }

    const DWORD own_pid = GetCurrentProcessId();
    const CpuTimes sys0 = query_cpu_times();
    const std::optional<std::uint64_t> own0 = query_process_cpu_time();
    const std::optional<std::vector<ProcessCpuTime>> p0 = query_all_process_cpu_times();
    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(window);
    const CpuTimes sys1 = query_cpu_times();
    const std::optional<std::uint64_t> own1 = query_process_cpu_time();
    const std::optional<std::vector<ProcessCpuTime>> p1 = query_all_process_cpu_times();
    r.window_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (own0 && own1)
        r.cpu_ratio = background_cpu_utilization(sys0, sys1, *own0, *own1);
    const std::uint64_t total = (sys1.kernel - sys0.kernel) + (sys1.user - sys0.user);
    const std::uint64_t busy = total - (sys1.idle - sys0.idle);
    if (p0 && p1 && total > 0) {
        std::uint64_t attributed = 0;
        for (const ProcessCpuTime& e : *p1) {
            if (e.pid == 0)
                continue;
            std::uint64_t start = 0;
            for (const ProcessCpuTime& s : *p0) {
                if (s.pid == e.pid && s.name == e.name) {
                    start = s.cpu;
                    break;
                }
            }
            const std::uint64_t delta = e.cpu >= start ? e.cpu - start : 0;
            attributed += delta;
            if (e.pid == own_pid)
                continue;
            r.processes.push_back(
                {e.pid, e.name, static_cast<double>(delta) / static_cast<double>(total)});
        }
        r.unattributed_share =
            busy > attributed ? static_cast<double>(busy - attributed) / static_cast<double>(total)
                              : 0.0;
        std::sort(
            r.processes.begin(), r.processes.end(),
            [](const ProcessActivity& a, const ProcessActivity& b) { return a.share > b.share; });
    }
    return r;
}

GateResult evaluate_idle(const IdleReading& r, double max_cpu_ratio) {
    GateResult g;
    g.name = "idle";
    ojson top = ojson::array();
    for (std::size_t i = 0; i < r.processes.size() && i < 8; ++i)
        top.push_back({{"name", r.processes[i].name},
                       {"pid", r.processes[i].pid},
                       {"cpu_share", r.processes[i].share}});
    ojson offenders = ojson::array();
    GateStatus status = GateStatus::pass;
    std::vector<std::string> notes;
    for (const Offender& o : kOffenders) {
        double share = 0;
        bool present = false;
        if (o.service) {
            const auto it = r.running_services.find(std::string(o.name));
            if (it != r.running_services.end()) {
                present = true;
                for (const ProcessActivity& p : r.processes) {
                    if (p.pid == it->second)
                        share = p.share;
                }
            }
        } else {
            const std::string want = lower(o.name);
            for (const ProcessActivity& p : r.processes) {
                if (lower(p.name) == want) {
                    present = true;
                    share += p.share;
                }
            }
        }
        if (!present)
            continue;
        const bool active = share >= o.active_share && (o.active_share > 0 || share > 0);
        const GateStatus s = active ? o.when_active : o.when_present;
        offenders.push_back({{"name", o.name},
                             {"group", o.group},
                             {"cpu_share", share},
                             {"active", active},
                             {"status", to_string(s)},
                             {"why", o.why}});
        if (s != GateStatus::pass) {
            status = worse(status, s);
            notes.push_back(std::string(o.name) + " " +
                            (active ? "busy " + percent(share) : std::string("present")) + " (" +
                            std::string(to_string(s)) + ")");
        }
    }
    g.details = {{"cpu_utilization_ratio", r.cpu_ratio ? ojson(*r.cpu_ratio) : ojson(nullptr)},
                 {"window_s", r.window_s},
                 {"max_cpu_ratio", max_cpu_ratio},
                 {"unattributed_share", r.unattributed_share},
                 {"top_processes", top},
                 {"offenders", offenders}};
    std::string load_text;
    if (!r.cpu_ratio) {
        status = GateStatus::fail;
        load_text = "the CPU load could not be read";
    } else {
        load_text = "background CPU load " + percent(*r.cpu_ratio) + " over " +
                    std::to_string(static_cast<int>(std::lround(r.window_s))) + " s (limit " +
                    percent(max_cpu_ratio) + ")";
        if (*r.cpu_ratio > max_cpu_ratio)
            status = GateStatus::fail;
    }
    g.status = status;
    g.summary = load_text;
    for (const std::string& n : notes)
        g.summary += "; " + n;
    return g;
}

// --------------------------------------------------------------- NPU users

NpuUserScan scan_npu_users() {
    NpuUserScan scan;
    std::vector<DWORD> pids(4096);
    DWORD bytes = 0;
    if (!K32EnumProcesses(pids.data(), static_cast<DWORD>(pids.size() * sizeof(DWORD)), &bytes))
        return scan;
    pids.resize(bytes / sizeof(DWORD));
    const DWORD own = GetCurrentProcessId();
    for (DWORD pid : pids) {
        if (pid == 0 || pid == own)
            continue;
        HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, FALSE, pid);
        if (h == nullptr) {
            ++scan.inaccessible;
            continue;
        }
        std::vector<HMODULE> modules(1024);
        DWORD needed = 0;
        if (!K32EnumProcessModulesEx(h, modules.data(),
                                     static_cast<DWORD>(modules.size() * sizeof(HMODULE)), &needed,
                                     LIST_MODULES_ALL)) {
            ++scan.inaccessible;
            CloseHandle(h);
            continue;
        }
        ++scan.scanned;
        modules.resize(std::min<std::size_t>(modules.size(), needed / sizeof(HMODULE)));
        NpuUser user;
        for (HMODULE m : modules) {
            wchar_t name[MAX_PATH];
            const DWORD n = K32GetModuleBaseNameW(h, m, name, MAX_PATH);
            if (n == 0)
                continue;
            const std::string base = lower(wide_to_utf8(name, static_cast<int>(n)));
            if (base.rfind("qnnhtp", 0) == 0 || base == "onnxruntime_providers_qnn.dll")
                user.modules.push_back(base);
        }
        if (!user.modules.empty()) {
            user.pid = pid;
            wchar_t image[MAX_PATH];
            DWORD size = MAX_PATH;
            if (QueryFullProcessImageNameW(h, 0, image, &size)) {
                const std::string full = wide_to_utf8(image, static_cast<int>(size));
                const std::size_t slash = full.find_last_of("\\/");
                user.name = slash == std::string::npos ? full : full.substr(slash + 1);
            }
            scan.users.push_back(std::move(user));
        }
        CloseHandle(h);
    }
    return scan;
}

GateResult evaluate_npu_users(const NpuUserScan& scan) {
    GateResult g;
    g.name = "npu_users";
    ojson users = ojson::array();
    for (const NpuUser& u : scan.users)
        users.push_back({{"pid", u.pid}, {"name", u.name}, {"modules", u.modules}});
    g.details = {{"users", users},
                 {"processes_scanned", scan.scanned},
                 {"processes_not_inspectable", scan.inaccessible},
                 {"note",
                  "Best effort: processes of other users or protected processes cannot be "
                  "inspected without elevation."}};
    if (scan.users.empty()) {
        g.status = GateStatus::pass;
        g.summary = "no other process has the QNN backend or EP loaded (" +
                    std::to_string(scan.scanned) + " processes scanned, " +
                    std::to_string(scan.inaccessible) + " not inspectable)";
    } else {
        g.status = GateStatus::warn;
        g.summary = "other processes have QNN modules loaded:";
        for (const NpuUser& u : scan.users)
            g.summary += " " + u.name + " (pid " + std::to_string(u.pid) + ")";
    }
    return g;
}

// ---------------------------------------------------------- sustained load

std::vector<double> windowed_p50(std::span<const double> latencies_ms, double window_ms) {
    std::vector<double> series;
    if (window_ms <= 0)
        return series;
    std::vector<double> current;
    double elapsed = 0;
    for (double x : latencies_ms) {
        current.push_back(x);
        elapsed += x;
        if (elapsed >= window_ms) {
            std::sort(current.begin(), current.end());
            series.push_back(nearest_rank(current, 50));
            current.clear();
            elapsed = 0;
        }
    }
    return series;  // a last partial window is dropped
}

std::optional<double> drift_ratio(std::span<const double> series) {
    if (series.size() < 2 || series.front() <= 0)
        return std::nullopt;
    return series.back() / series.front() - 1.0;
}

GateResult evaluate_sustained(const std::vector<SustainedLeg>& legs, double window_s,
                              double max_drift_ratio) {
    GateResult g;
    g.name = "sustained_load";
    g.status = legs.empty() ? GateStatus::fail : GateStatus::pass;
    ojson out = ojson::array();
    for (const SustainedLeg& leg : legs) {
        ojson d = leg.details;
        d["label"] = leg.label;
        d["duration_s"] = leg.duration_s;
        d["window_s"] = window_s;
        d["p50_series_ms"] = leg.p50_series_ms;
        const std::optional<double> drift = drift_ratio(leg.p50_series_ms);
        d["first_window_p50_ms"] =
            leg.p50_series_ms.empty() ? ojson(nullptr) : ojson(leg.p50_series_ms.front());
        d["last_window_p50_ms"] =
            leg.p50_series_ms.empty() ? ojson(nullptr) : ojson(leg.p50_series_ms.back());
        d["drift_ratio"] = drift ? ojson(*drift) : ojson(nullptr);
        std::string text;
        if (leg.failure) {
            g.status = GateStatus::fail;
            text = leg.label + ": " + *leg.failure;
            d["failure"] = *leg.failure;
        } else if (!drift) {
            g.status = GateStatus::fail;
            text = leg.label + ": fewer than two " + std::to_string(static_cast<int>(window_s)) +
                   " s windows";
        } else {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "%s: last window p50 %+.1f%% vs first (%.4g -> %.4g ms)",
                          leg.label.c_str(), *drift * 100.0, leg.p50_series_ms.front(),
                          leg.p50_series_ms.back());
            text = buf;
            if (*drift > max_drift_ratio)
                g.status = GateStatus::fail;
        }
        g.summary += (g.summary.empty() ? "" : "; ") + text;
        out.push_back(d);
    }
    if (legs.empty())
        g.summary = "no sustained-load legs configured";
    g.details = {{"max_drift_ratio", max_drift_ratio}, {"window_s", window_s}, {"legs", out}};
    return g;
}

// ---------------------------------------------------------------- in flight

std::vector<double> backoff_schedule(double max_wait_s) {
    std::vector<double> waits;
    double next = 30, total = 0;
    while (total < max_wait_s - 1e-9) {
        const double w = std::min(next, max_wait_s - total);
        waits.push_back(w);
        total += w;
        next *= 2;
    }
    return waits;
}

}  // namespace edgebench::suite
