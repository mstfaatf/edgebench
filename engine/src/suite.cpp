#include "edgebench/suite.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <system_error>
#include <thread>

#include "edgebench/diagnostics.hpp"
#include "edgebench/ep_context_cache.hpp"
#include "edgebench/exit_codes.hpp"
#include "edgebench/interrupt.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/result_files.hpp"
#include "edgebench/run_identity.hpp"
#include "edgebench/sha256.hpp"
#include "edgebench/stats.hpp"
#include "edgebench/system_info.hpp"
#include "edgebench/version.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace fs = std::filesystem;

namespace edgebench::suite {

namespace {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

/// How long an interrupted run may take to stop by itself before it is terminated.
constexpr auto kStopGrace = std::chrono::seconds(30);

std::string now_utc() {
    return utc_timestamp(std::chrono::system_clock::now());
}

fs::path from_utf8(const std::string& s) {
    return fs::path(std::u8string(s.begin(), s.end()));
}

std::string rel(const fs::path& path, const fs::path& base) {
    std::error_code ec;
    const fs::path r = fs::relative(path, base, ec);
    return path_to_utf8(ec || r.empty() ? path : fs::path(r.generic_wstring()));
}

// --------------------------------------------------------------------- processes

std::wstring widen(const std::string& s) {
    if (s.empty())
        return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

// One argument quoted for CommandLineToArgvW / the MSVC runtime.
std::wstring quote(const std::wstring& arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos)
        return arg;
    std::wstring out = L"\"";
    for (auto it = arg.begin();; ++it) {
        std::size_t backslashes = 0;
        while (it != arg.end() && *it == L'\\') {
            ++it;
            ++backslashes;
        }
        if (it == arg.end()) {
            out.append(backslashes * 2, L'\\');
            break;
        }
        if (*it == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(L'"');
        } else {
            out.append(backslashes, L'\\');
            out.push_back(*it);
        }
    }
    out.push_back(L'"');
    return out;
}

// Every child is in this job: if the suite process dies, its run dies with it.
HANDLE suite_job() {
    static HANDLE job = [] {
        HANDLE j = CreateJobObjectW(nullptr, nullptr);
        if (j != nullptr) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(j, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
        }
        return j;
    }();
    return job;
}

HANDLE open_for_child(const fs::path& path, bool append) {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    return CreateFileW(path.c_str(), append ? FILE_APPEND_DATA : GENERIC_WRITE,
                       FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                       append ? OPEN_ALWAYS : CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}

void write_handle(HANDLE h, const std::string& text) {
    DWORD written = 0;
    WriteFile(h, text.data(), static_cast<DWORD>(text.size()), &written, nullptr);
}

struct ChildOutcome {
    bool started = false;
    std::string start_error;
    int exit_code = -1;
    bool interrupted = false;  // a stop was requested while it ran
    double seconds = 0;
};

// Runs `exe args` with stdout to `stdout_file` (or the log) and stderr to `log`,
// appending a header and a footer to the log. A stop request is forwarded to the
// child as Ctrl+Break; if it has not exited kStopGrace later, it is terminated.
ChildOutcome run_child(const fs::path& exe, const std::vector<std::string>& args,
                       const fs::path& log, const std::optional<fs::path>& stdout_file,
                       const std::function<void()>& started = {}) {
    ChildOutcome out;
    std::error_code ec;
    fs::create_directories(log.parent_path(), ec);
    HANDLE log_handle = open_for_child(log, true);
    if (log_handle == INVALID_HANDLE_VALUE) {
        out.start_error = "cannot open the log file " + path_to_utf8(log);
        return out;
    }
    HANDLE out_handle = log_handle;
    if (stdout_file) {
        out_handle = open_for_child(*stdout_file, false);
        if (out_handle == INVALID_HANDLE_VALUE) {
            CloseHandle(log_handle);
            out.start_error = "cannot create " + path_to_utf8(*stdout_file);
            return out;
        }
    }
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE null_in = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                                 OPEN_EXISTING, 0, nullptr);

    std::string shown = path_to_utf8(exe.filename());
    std::wstring command = quote(exe.native());
    for (const std::string& a : args) {
        command += L" " + quote(widen(a));
        shown += " " + (a.find(' ') == std::string::npos ? a : "\"" + a + "\"");
    }
    write_handle(log_handle, "$ " + shown + "\n--- started " + now_utc() + "\n");

    // Only these handles are inherited.
    HANDLE inherit[3] = {null_in, out_handle, log_handle};
    const DWORD inherit_count = out_handle == log_handle ? 2 : 3;
    if (out_handle == log_handle)
        inherit[1] = log_handle;
    SIZE_T attr_size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
    std::vector<unsigned char> attr_buffer(attr_size);
    auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_buffer.data());
    InitializeProcThreadAttributeList(attrs, 1, 0, &attr_size);
    UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit,
                              inherit_count * sizeof(HANDLE), nullptr, nullptr);

    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof(si);
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = null_in;
    si.StartupInfo.hStdOutput = out_handle;
    si.StartupInfo.hStdError = log_handle;
    si.lpAttributeList = attrs;
    PROCESS_INFORMATION pi{};
    DWORD flags = CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED |
                  EXTENDED_STARTUPINFO_PRESENT;
    if (GetConsoleWindow() == nullptr)
        flags |= CREATE_NO_WINDOW;
    const auto t0 = Clock::now();
    const BOOL ok = CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, TRUE, flags,
                                   nullptr, nullptr, &si.StartupInfo, &pi);
    const DWORD create_error = GetLastError();
    DeleteProcThreadAttributeList(attrs);
    if (!ok) {
        out.start_error = "cannot start " + path_to_utf8(exe) + " (Windows error " +
                          std::to_string(create_error) + ")";
        write_handle(log_handle, "--- " + out.start_error + "\n");
    } else {
        if (HANDLE job = suite_job(); job != nullptr)
            AssignProcessToJobObject(job, pi.hProcess);
        ResumeThread(pi.hThread);
        CloseHandle(pi.hThread);
        out.started = true;
        if (started)
            started();
        std::optional<Clock::time_point> deadline;
        while (WaitForSingleObject(pi.hProcess, 100) == WAIT_TIMEOUT) {
            if (stop_requested() && !deadline) {
                out.interrupted = true;
                // Without a shared console the child cannot be asked: terminate it.
                deadline = GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pi.dwProcessId)
                               ? Clock::now() + kStopGrace
                               : Clock::now();
            }
            if (deadline && Clock::now() > *deadline) {
                TerminateProcess(pi.hProcess, static_cast<UINT>(to_int(ExitCode::interrupted)));
                WaitForSingleObject(pi.hProcess, 5000);
                break;
            }
        }
        DWORD code = 0;
        GetExitCodeProcess(pi.hProcess, &code);
        CloseHandle(pi.hProcess);
        out.exit_code = static_cast<int>(code);
        out.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        if (stop_requested())
            out.interrupted = true;
        char footer[96];
        std::snprintf(footer, sizeof(footer), "--- exit %d in %.1f s\n\n", out.exit_code,
                      out.seconds);
        write_handle(log_handle, footer);
    }
    if (null_in != INVALID_HANDLE_VALUE)
        CloseHandle(null_in);
    if (out_handle != log_handle)
        CloseHandle(out_handle);
    CloseHandle(log_handle);
    return out;
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::optional<json> read_json(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::nullopt;
    try {
        json j = json::parse(in);
        return j;
    } catch (const json::exception&) {
        return std::nullopt;
    }
}

std::string exit_name(int code) {
    for (ExitCode c : kAllExitCodes) {
        if (to_int(c) == code)
            return std::string(name(c));
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%08X", static_cast<unsigned>(code));
    return std::string("unknown (") + buf + ")";
}

// ------------------------------------------------------------------- doctor gate

class DoctorCheck : public PreflightCheck {
public:
    std::string name() const override { return "doctor"; }

    GateResult run(const PreflightContext& ctx) override {
        GateResult g;
        g.name = name();
        const fs::path dir = ctx.suite_dir / "preflight";
        const std::string stem = "doctor-" + std::to_string(ctx.attempt);
        const fs::path report = dir / (stem + ".json");
        const fs::path log = dir / (stem + ".log");
        std::error_code ec;
        fs::create_directories(dir, ec);
        const ChildOutcome child = run_child(ctx.exe, {"doctor", "--json", "--quick"}, log, report);
        g.details["command"] = "edgebench doctor --json --quick";
        g.details["report_file"] = rel(report, ctx.suite_dir);
        g.details["log_file"] = rel(log, ctx.suite_dir);
        if (!child.started) {
            g.status = GateStatus::fail;
            g.summary = child.start_error;
            g.details["exit_code"] = nullptr;
            return g;
        }
        g.details["exit_code"] = child.exit_code;
        const std::optional<json> doc = read_json(report);
        if (!doc || !doc->contains("checks") || !doc->contains("overall")) {
            g.status = GateStatus::fail;
            g.summary = "doctor exited " + std::to_string(child.exit_code) +
                        " without a report (see " + rel(log, ctx.suite_dir) + ")";
            return g;
        }
        ojson checks = ojson::object();
        std::string not_passing;
        for (const json& c : (*doc)["checks"]) {
            const std::string status = c.value("status", "");
            checks[c.value("name", "?")] = status;
            if (status == "warn" || status == "fail")
                not_passing +=
                    (not_passing.empty() ? "" : ", ") + c.value("name", "?") + " " + status;
        }
        g.details["overall"] = (*doc)["overall"].value("status", "");
        g.details["checks"] = checks;
        g.details["versions"] = ojson::parse(doc->value("versions", json(nullptr)).dump());
        const std::string overall = (*doc)["overall"].value("status", "");
        if (child.exit_code != 0 || overall == "fail") {
            g.status = GateStatus::fail;
        } else if (overall == "warn") {
            g.status = GateStatus::warn;
        } else {
            g.status = GateStatus::pass;
        }
        g.summary = "doctor exit " + std::to_string(child.exit_code) + ", " + overall +
                    (not_passing.empty() ? "" : " (" + not_passing + ")");
        return g;
    }
};

// Reads a result's per-iteration latencies (all trials in order).
std::vector<double> read_latencies(const fs::path& result) {
    std::vector<double> out;
    const std::optional<json> doc = read_json(result);
    if (!doc || !doc->contains("raw") || !(*doc)["raw"].is_object())
        return out;
    std::ifstream in(result.parent_path() / from_utf8((*doc)["raw"].value("latencies_file", "")),
                     std::ios::binary);
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
        const std::size_t comma = line.rfind(',');
        if (comma != std::string::npos)
            out.push_back(std::strtod(line.c_str() + comma + 1, nullptr));
    }
    return out;
}

class FunctionCheck : public PreflightCheck {
public:
    FunctionCheck(std::string name, std::function<GateResult(const PreflightContext&)> fn)
        : name_(std::move(name)), fn_(std::move(fn)) {}
    std::string name() const override { return name_; }
    GateResult run(const PreflightContext& ctx) override { return fn_(ctx); }

private:
    std::string name_;
    std::function<GateResult(const PreflightContext&)> fn_;
};

// One continuous run per leg: a short calibration run gives the mean latency,
// then one trial long enough for the leg's duration.
GateResult sustained_load_check(const PreflightContext& ctx) {
    const SuitePlan& plan = *ctx.plan;
    std::vector<SustainedLeg> legs;
    const fs::path dir = ctx.suite_dir / "preflight" / ("sustained-" + std::to_string(ctx.attempt));
    std::error_code ec;
    fs::create_directories(dir, ec);
    for (std::size_t i = 0; i < plan.sustained.size() && !stop_requested(); ++i) {
        const ResolvedConfiguration& c = plan.sustained[i];
        SustainedLeg leg;
        leg.label = c.label;
        leg.duration_s = c.sustained_duration_s;
        const fs::path log = dir / (c.label + ".log");
        RunConfig cal =
            to_run_config(c, dir / (c.label + "-calibration"), "calibration", std::nullopt);
        cal.iterations = 200;
        std::fprintf(stderr, "suite: pre-flight sustained_load: %s, %.0f s\n", c.label.c_str(),
                     c.sustained_duration_s);
        const ChildOutcome a = run_child(ctx.exe, run_arguments(cal), log, std::nullopt);
        const std::optional<json> cal_doc =
            read_json(dir / (c.label + "-calibration") / "calibration_latest.json");
        double mean_ms = 0;
        if (cal_doc && cal_doc->contains("timing") && (*cal_doc)["timing"].is_object())
            mean_ms = (*cal_doc)["timing"]["aggregate"]["stats"].value("mean_ms", 0.0);
        if (!a.started || a.exit_code != 0 || mean_ms <= 0) {
            leg.failure = "the calibration run failed (exit " + std::to_string(a.exit_code) +
                          "; see " + rel(log, ctx.suite_dir) + ")";
            legs.push_back(std::move(leg));
            continue;
        }
        const double wanted = std::ceil(c.sustained_duration_s * 1000.0 / mean_ms);
        RunConfig run = to_run_config(c, dir / c.label, "sustained", std::nullopt);
        run.iterations =
            static_cast<int>(std::clamp(wanted, 100.0, static_cast<double>(kMaxIterations)));
        const ChildOutcome b = run_child(ctx.exe, run_arguments(run), log, std::nullopt);
        const fs::path result = dir / c.label / "sustained_latest.json";
        leg.details = {{"calibration_mean_ms", mean_ms},
                       {"iterations", run.iterations},
                       {"seconds", b.seconds},
                       {"result_file", rel(result, ctx.suite_dir)},
                       {"log_file", rel(log, ctx.suite_dir)}};
        if (!b.started || b.exit_code != 0) {
            leg.failure = "the sustained run failed (exit " + std::to_string(b.exit_code) +
                          "; see " + rel(log, ctx.suite_dir) + ")";
        } else {
            leg.p50_series_ms =
                windowed_p50(read_latencies(result), plan.preflight.sustained_window_s * 1000.0);
        }
        legs.push_back(std::move(leg));
    }
    return evaluate_sustained(legs, plan.preflight.sustained_window_s,
                              plan.preflight.sustained_max_drift_ratio);
}

// --------------------------------------------------------------------- manifest

ojson gate_json(const GateResult& g) {
    return {{"name", g.name},
            {"status", to_string(g.status)},
            {"summary", g.summary},
            {"details", g.details}};
}

template <typename T>
ojson opt(const std::optional<T>& v) {
    return v ? ojson(*v) : ojson(nullptr);
}

template <typename T>
std::optional<T> get_opt(const json& j, const char* key) {
    if (!j.contains(key) || j[key].is_null())
        return std::nullopt;
    return j[key].get<T>();
}

}  // namespace

// ------------------------------------------------------------------------ public

std::string_view to_string(RunStatus s) {
    switch (s) {
        case RunStatus::pending:
            return "pending";
        case RunStatus::running:
            return "running";
        case RunStatus::done:
            return "done";
        case RunStatus::failed:
            return "failed";
        case RunStatus::skipped:
            return "skipped";
    }
    return "pending";
}

std::optional<RunStatus> parse_run_status(std::string_view t) {
    for (RunStatus s : {RunStatus::pending, RunStatus::running, RunStatus::done, RunStatus::failed,
                        RunStatus::skipped}) {
        if (to_string(s) == t)
            return s;
    }
    return std::nullopt;
}

std::unique_ptr<PreflightCheck> make_doctor_check() {
    return std::make_unique<DoctorCheck>();
}

std::vector<std::unique_ptr<PreflightCheck>> make_preflight_checks(
    const std::vector<std::string>& names) {
    std::vector<std::unique_ptr<PreflightCheck>> checks;
    for (const std::string& n : names) {
        if (n == "doctor") {
            checks.push_back(make_doctor_check());
        } else if (n == "power") {
            checks.push_back(std::make_unique<FunctionCheck>(
                n, [](const PreflightContext&) { return evaluate_power(query_power_state()); }));
        } else if (n == "sleep") {
            checks.push_back(std::make_unique<FunctionCheck>(n, [](const PreflightContext& c) {
                return evaluate_sleep(query_sleep_settings(),
                                      c.estimate_s + c.plan->preflight.sleep_margin_s);
            }));
        } else if (n == "idle") {
            checks.push_back(std::make_unique<FunctionCheck>(n, [](const PreflightContext& c) {
                const auto window = std::chrono::milliseconds(
                    std::lround(c.plan->preflight.idle_window_s * 1000.0));
                return evaluate_idle(read_idle(window), c.plan->preflight.idle_max_cpu_ratio);
            }));
        } else if (n == "npu_users") {
            checks.push_back(std::make_unique<FunctionCheck>(
                n, [](const PreflightContext&) { return evaluate_npu_users(scan_npu_users()); }));
        } else if (n == "sustained_load") {
            checks.push_back(std::make_unique<FunctionCheck>(n, sustained_load_check));
        }
    }
    return checks;
}

const std::vector<std::string>& publish_gate_checks() {
    return gate_check_names();
}

bool retry_worthy(const RunRecord& r) {
    return r.status == RunStatus::done &&
           std::any_of(r.warnings.begin(), r.warnings.end(), [](const std::string& w) {
               return w == "background_cpu_load_high" || w == "trial_spread_high";
           });
}

bool SuiteManifest::gate_passed() const {
    for (auto it = preflight.rbegin(); it != preflight.rend(); ++it) {
        if (it->skipped)
            continue;
        for (const std::string& required : publish_gate_checks()) {
            const bool ok =
                std::any_of(it->checks.begin(), it->checks.end(), [&](const GateResult& g) {
                    return g.name == required && g.status == GateStatus::pass;
                });
            if (!ok)
                return false;
        }
        return true;
    }
    return false;
}

bool gate_passed(const std::vector<GateResult>& results, bool fail_on_warn) {
    return std::none_of(results.begin(), results.end(), [&](const GateResult& g) {
        return g.status == GateStatus::fail || (fail_on_warn && g.status == GateStatus::warn);
    });
}

bool SuiteManifest::publishable() const {
    return not_publishable_reasons().empty();
}

std::vector<std::string> SuiteManifest::not_publishable_reasons() const {
    std::vector<std::string> reasons;
    if (preflight_ever_skipped)
        reasons.push_back("the pre-flight check was skipped (--skip-preflight)");
    if (!preflight.empty() && !preflight.back().skipped) {
        for (const std::string& required : publish_gate_checks()) {
            const auto& checks = preflight.back().checks;
            const auto it = std::find_if(checks.begin(), checks.end(),
                                         [&](const GateResult& g) { return g.name == required; });
            if (it == checks.end())
                reasons.push_back("the pre-flight gate did not include '" + required + "'");
            else if (it->status != GateStatus::pass)
                reasons.push_back("pre-flight check '" + required + "': " +
                                  std::string(to_string(it->status)) + " (" + it->summary + ")");
        }
    }
    if (gate_lost)
        reasons.push_back("the gate was lost during the suite: " + *gate_lost);
    if (tool.is_object() && tool.contains("git_dirty") && tool["git_dirty"].is_boolean() &&
        tool["git_dirty"].get<bool>())
        reasons.push_back("built from a working tree with uncommitted changes");
    return reasons;
}

ojson to_json(const SuiteManifest& m) {
    ojson runs = ojson::array();
    std::map<std::string, int> counts;
    for (RunStatus s : {RunStatus::pending, RunStatus::running, RunStatus::done, RunStatus::failed,
                        RunStatus::skipped})
        counts[std::string(to_string(s))] = 0;
    for (const RunRecord& r : m.runs) {
        ++counts[std::string(to_string(r.status))];
        ojson issues = ojson::array();
        for (const std::string& i : r.validation_issues)
            issues.push_back(i);
        runs.push_back({
            {"run_id", r.run_id},
            {"short_id", r.run_id.substr(0, kShortIdLength)},
            {"configuration_id", r.configuration_id},
            {"label", r.label},
            {"order_index", r.order_index},
            {"repetition", r.repetition},
            {"configuration", r.configuration},
            {"status", to_string(r.status)},
            {"attempts", r.attempts},
            {"exit_code", opt(r.exit_code)},
            {"exit_code_name", opt(r.exit_code_name)},
            {"started_utc", opt(r.started_utc)},
            {"ended_utc", opt(r.ended_utc)},
            {"duration_s", opt(r.duration_s)},
            {"cooldown_s_before", opt(r.cooldown_s_before)},
            {"estimate_s", r.estimate_s},
            {"output_dir", opt(r.output_dir)},
            {"result_file", opt(r.result_file)},
            {"result_sha256", opt(r.result_sha256)},
            {"log_file", opt(r.log_file)},
            {"failure", opt(r.failure)},
            {"validation", opt(r.validation)},
            {"validation_issues", issues},
            {"warnings", r.warnings},
            {"cache_build", r.cache_build},
            {"discarded", r.discarded},
            {"in_flight", r.in_flight},
            {"retries", r.retries},
            {"history", r.history},
        });
    }
    ojson preflight = ojson::array();
    for (const PreflightAttempt& a : m.preflight) {
        ojson checks = ojson::array();
        for (const GateResult& g : a.checks)
            checks.push_back(gate_json(g));
        preflight.push_back({{"started_utc", a.started_utc},
                             {"skipped", a.skipped},
                             {"passed", a.passed},
                             {"settings", a.settings},
                             {"checks", checks}});
    }
    ojson invocations = ojson::array();
    for (const Invocation& i : m.invocations) {
        invocations.push_back({{"started_utc", i.started_utc},
                               {"ended_utc", opt(i.ended_utc)},
                               {"resumed", i.resumed},
                               {"fail_fast", i.fail_fast},
                               {"skip_preflight", i.skip_preflight},
                               {"exit_code", opt(i.exit_code)}});
    }
    ojson count_json = ojson::object();
    for (const auto& [k, v] : counts)
        count_json[k] = v;
    return {
        {"suite_manifest_version", kManifestVersion},
        {"suite",
         {{"name", m.suite_name},
          {"config_file", m.config_file},
          {"config_sha256", m.config_sha256},
          {"config", ojson::parse(m.config.dump())}}},
        {"created_utc", m.created_utc},
        {"updated_utc", m.updated_utc},
        {"status", m.status},
        {"tool", m.tool},
        {"versions", m.versions},
        {"rotation", m.rotation},
        {"repetitions", m.repetitions},
        {"validator", {{"python", m.python}, {"script", m.validator}}},
        {"publishable", m.publishable()},
        {"not_publishable_reasons", m.not_publishable_reasons()},
        {"gate_passed", m.gate_passed()},
        {"gate_lost", opt(m.gate_lost)},
        {"in_flight_wait_s", m.in_flight_wait_s},
        {"preflight_ever_skipped", m.preflight_ever_skipped},
        {"preflight", preflight},
        {"invocations", invocations},
        {"estimate",
         {{"total_s", m.estimate.total_s},
          {"gate_s", m.estimate.gate_s},
          {"configurations_from_prior_results", m.estimate.from_prior},
          {"configurations_from_heuristic", m.estimate.from_heuristic}}},
        {"counts", count_json},
        {"runs", runs},
    };
}

Result<SuiteManifest> manifest_from_json(const json& j) {
    auto bad = [](const std::string& what) {
        return Error{ExitCode::usage_error, "suite_manifest_invalid",
                     "not a suite manifest this version can resume: " + what};
    };
    try {
        if (j.value("suite_manifest_version", "") != kManifestVersion)
            return bad("suite_manifest_version is not " + std::string(kManifestVersion));
        SuiteManifest m;
        const json& s = j.at("suite");
        m.suite_name = s.at("name").get<std::string>();
        m.config_file = s.at("config_file").get<std::string>();
        m.config_sha256 = s.at("config_sha256").get<std::string>();
        m.config = s.at("config");
        m.created_utc = j.at("created_utc").get<std::string>();
        m.updated_utc = j.at("updated_utc").get<std::string>();
        m.status = j.at("status").get<std::string>();
        m.tool = ojson::parse(j.at("tool").dump());
        m.versions = ojson::parse(j.at("versions").dump());
        m.rotation = j.at("rotation").get<std::string>();
        m.repetitions = j.at("repetitions").get<int>();
        m.python = j.at("validator").at("python").get<std::string>();
        m.validator = j.at("validator").at("script").get<std::string>();
        m.preflight_ever_skipped = j.at("preflight_ever_skipped").get<bool>();
        for (const json& a : j.at("preflight")) {
            PreflightAttempt p;
            p.started_utc = a.at("started_utc").get<std::string>();
            p.skipped = a.at("skipped").get<bool>();
            p.passed = a.at("passed").get<bool>();
            p.settings = ojson::parse(a.at("settings").dump());
            for (const json& c : a.at("checks")) {
                GateResult g;
                g.name = c.at("name").get<std::string>();
                const std::string st = c.at("status").get<std::string>();
                g.status = st == "pass"   ? GateStatus::pass
                           : st == "warn" ? GateStatus::warn
                                          : GateStatus::fail;
                g.summary = c.at("summary").get<std::string>();
                g.details = ojson::parse(c.at("details").dump());
                p.checks.push_back(std::move(g));
            }
            m.preflight.push_back(std::move(p));
        }
        for (const json& i : j.at("invocations")) {
            Invocation v;
            v.started_utc = i.at("started_utc").get<std::string>();
            v.ended_utc = get_opt<std::string>(i, "ended_utc");
            v.resumed = i.at("resumed").get<bool>();
            v.fail_fast = i.at("fail_fast").get<bool>();
            v.skip_preflight = i.at("skip_preflight").get<bool>();
            v.exit_code = get_opt<int>(i, "exit_code");
            m.invocations.push_back(v);
        }
        m.estimate.total_s = j.at("estimate").at("total_s").get<double>();
        m.estimate.gate_s = j.at("estimate").at("gate_s").get<double>();
        m.gate_lost = get_opt<std::string>(j, "gate_lost");
        m.in_flight_wait_s = j.at("in_flight_wait_s").get<double>();
        m.estimate.from_prior = j.at("estimate").at("configurations_from_prior_results").get<int>();
        m.estimate.from_heuristic = j.at("estimate").at("configurations_from_heuristic").get<int>();
        for (const json& r : j.at("runs")) {
            RunRecord x;
            x.run_id = r.at("run_id").get<std::string>();
            x.configuration_id = r.at("configuration_id").get<std::string>();
            x.label = r.at("label").get<std::string>();
            x.order_index = r.at("order_index").get<int>();
            x.repetition = r.at("repetition").get<int>();
            x.configuration = ojson::parse(r.at("configuration").dump());
            const std::optional<RunStatus> st = parse_run_status(r.at("status").get<std::string>());
            if (!st)
                return bad("run " + x.run_id + " has an unknown status");
            x.status = *st;
            x.attempts = r.at("attempts").get<int>();
            x.exit_code = get_opt<int>(r, "exit_code");
            x.exit_code_name = get_opt<std::string>(r, "exit_code_name");
            x.started_utc = get_opt<std::string>(r, "started_utc");
            x.ended_utc = get_opt<std::string>(r, "ended_utc");
            x.duration_s = get_opt<double>(r, "duration_s");
            x.cooldown_s_before = get_opt<double>(r, "cooldown_s_before");
            x.estimate_s = r.at("estimate_s").get<double>();
            x.output_dir = get_opt<std::string>(r, "output_dir");
            x.result_file = get_opt<std::string>(r, "result_file");
            x.result_sha256 = get_opt<std::string>(r, "result_sha256");
            x.log_file = get_opt<std::string>(r, "log_file");
            x.failure = get_opt<std::string>(r, "failure");
            x.validation = get_opt<std::string>(r, "validation");
            x.validation_issues = r.at("validation_issues").get<std::vector<std::string>>();
            x.warnings = r.at("warnings").get<std::vector<std::string>>();
            x.cache_build = ojson::parse(r.at("cache_build").dump());
            x.discarded = r.at("discarded").get<std::vector<std::string>>();
            x.in_flight = ojson::parse(r.at("in_flight").dump());
            x.retries = r.at("retries").get<int>();
            x.history = ojson::parse(r.at("history").dump());
            m.runs.push_back(std::move(x));
        }
        return m;
    } catch (const json::exception& e) {
        return bad(e.what());
    }
}

ResumeAction resume_action(const RunRecord& run,
                           const std::function<bool(const RunRecord&)>& result_valid) {
    switch (run.status) {
        case RunStatus::done:
            return result_valid(run) ? ResumeAction::keep : ResumeAction::rerun;
        case RunStatus::running:
            return ResumeAction::discard;
        case RunStatus::failed:
            return ResumeAction::rerun;
        case RunStatus::pending:
        case RunStatus::skipped:
            return ResumeAction::run;
    }
    return ResumeAction::run;
}

// --------------------------------------------------------------------- estimate

namespace {

std::string prior_key(const std::string& sha, std::string_view provider, const std::string& compute,
                      std::int64_t batch) {
    return sha + "|" + std::string(provider) + "|" + compute + "|" + std::to_string(batch);
}

double median_ms(const json& summary) {
    if (summary.is_object() && summary.contains("median_ms") && summary["median_ms"].is_number())
        return summary["median_ms"].get<double>();
    return 0;
}

}  // namespace

void PriorResults::add(const json& doc, const std::string& source) {
    try {
        if (!doc.is_object() || doc.value("status", "") != "success" || !doc.contains("timing") ||
            !doc.at("timing").is_object() || !doc.contains("model") ||
            !doc.at("model").is_object() || !doc.contains("execution") ||
            !doc.at("execution").is_object())
            return;
        const json& timing = doc.at("timing");
        const json& config = doc.at("config");
        const json ep = doc.at("execution").value("effective_precision", json(nullptr));
        const std::string provider = config.at("requested").value("execution_provider", "");
        if (!ep.is_object() || provider.empty() || !config.at("batch_size").is_number_integer())
            return;
        // Results of suite cache-building runs (1 iteration) say little about latency.
        if (config.value("measured_iterations", 0) < 10)
            return;
        TimingBasis b;
        b.from_prior = true;
        b.source = source;
        b.iteration_ms = timing.at("aggregate").at("stats").value("mean_ms", 0.0);
        for (const json& t : timing.at("trials"))
            b.iteration_ms = std::max(b.iteration_ms, t.at("stats").value("mean_ms", 0.0));
        const json& init = timing.at("session_init");
        double cold = 0;
        if (init.contains("cold_ms") && init.at("cold_ms").is_array()) {
            for (const json& v : init.at("cold_ms"))
                if (v.is_number())
                    cold = std::max(cold, v.get<double>());
        }
        double cached = 0, build = 0;
        if (init.contains("summary") && init.at("summary").is_object()) {
            const json& s = init.at("summary");
            cold = std::max(cold, median_ms(s.value("cold_ms", json(nullptr))));
            cached = median_ms(s.value("cached_init_ms", json(nullptr)));
            build = median_ms(s.value("context_build_ms", json(nullptr)));
        }
        if (init.contains("session_create_ms") && init.at("session_create_ms").is_number())
            cold = std::max(cold, init.at("session_create_ms").get<double>());
        b.cold_init_s = cold / 1000.0;
        b.cached_init_s = cached / 1000.0;
        b.build_init_s = build / 1000.0;
        entries_.push_back(
            {prior_key(doc.at("model").value("sha256", ""), provider, ep.value("compute", ""),
                       config.at("batch_size").get<std::int64_t>()),
             b});
    } catch (const json::exception&) {
    }
}

PriorResults PriorResults::scan(const std::vector<fs::path>& roots) {
    PriorResults prior;
    std::error_code ec;
    for (const fs::path& root : roots) {
        if (!fs::is_directory(root, ec))
            continue;
        for (auto it = fs::recursive_directory_iterator(
                 root, fs::directory_options::skip_permission_denied, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec)
                break;
            const fs::path& p = it->path();
            if (!it->is_regular_file(ec) || p.extension() != L".json" ||
                p.filename().wstring().ends_with(L"_latest.json") ||
                it->file_size(ec) > 4 * 1024 * 1024)
                continue;
            const std::optional<json> doc = read_json(p);
            if (doc && doc->is_object() && doc->contains("schema_version"))
                prior.add(*doc, path_to_utf8(p));
        }
    }
    return prior;
}

std::optional<TimingBasis> PriorResults::find(const ResolvedConfiguration& c) const {
    const std::string key = prior_key(c.model_sha256, execution_provider_name(c.identity.provider),
                                      c.compute_precision, c.identity.batch);
    std::optional<TimingBasis> best;
    for (const Entry& e : entries_) {
        if (e.key != key)
            continue;
        if (!best) {
            best = e.basis;
            continue;
        }
        // The slowest of the matching results: the estimate errs toward too long.
        best->iteration_ms = std::max(best->iteration_ms, e.basis.iteration_ms);
        best->cold_init_s = std::max(best->cold_init_s, e.basis.cold_init_s);
        best->cached_init_s = std::max(best->cached_init_s, e.basis.cached_init_s);
        best->build_init_s = std::max(best->build_init_s, e.basis.build_init_s);
    }
    return best;
}

TimingBasis heuristic_basis(const ResolvedConfiguration& c) {
    const double mb = static_cast<double>(c.model_size_bytes) / 1e6;
    const double batch = c.identity.batch;
    TimingBasis b;
    b.source = "heuristic";
    if (c.identity.provider == ProviderKind::npu) {
        b.iteration_ms = 1.0 + 0.2 * mb * batch;
        b.cold_init_s = 3.0 + 0.1 * mb * batch;
        b.build_init_s = 1.2 * b.cold_init_s;
        b.cached_init_s = 1.0;
    } else {
        b.iteration_ms = 1.0 + 1.0 * mb * batch;
        b.cold_init_s = 0.5 + 0.05 * mb;
        b.build_init_s = b.cold_init_s;
        b.cached_init_s = b.cold_init_s;
    }
    return b;
}

double estimate_run_s(const ResolvedConfiguration& c, const TimingBasis& basis) {
    const RunParams& p = c.params;
    const bool npu = c.identity.provider == ProviderKind::npu;
    // An earlier result may lack a value (e.g. no cached sessions): use the heuristic's.
    const TimingBasis h = heuristic_basis(c);
    const double cold = basis.cold_init_s > 0 ? basis.cold_init_s : h.cold_init_s;
    const double cached = basis.cached_init_s > 0 ? basis.cached_init_s : h.cached_init_s;
    const double build =
        basis.build_init_s > 0 ? basis.build_init_s : std::max(cold, h.build_init_s);
    const double it_s = (basis.iteration_ms > 0 ? basis.iteration_ms : h.iteration_ms) / 1000.0;
    const double overhead = npu ? kNpuProcessOverheadS : kProcessOverheadS;
    const double trial_idle = p.trial_cooldown_ms / 1000.0;

    const double timed_session = p.init == InitMode::cold ? cold : cached;
    double trial = trial_idle + timed_session + (p.warmup + p.iterations) * it_s;
    if (p.init_breakdown)
        trial += 3 * trial_idle + cold + build + cached;
    double total = overhead + p.trials * trial;
    if (p.trial_cooldown_ms == 0)
        total += p.load_sample_ms / 1000.0;
    total += cold;  // placement session
    if (p.init != InitMode::cold)
        total += cold + cached;  // compiled vs cached outputs, before the trials
    if (p.init == InitMode::build)
        total += build;
    if (p.init == InitMode::cached)  // the cache-building run before it
        total += kNpuProcessOverheadS + build + cold + cached + cold + kValidationS;
    if (p.profile_accelerator) {
        const int n = p.profile_iterations.value_or(kDefaultProfileIterations);
        const double profiled_s = std::max(3 * it_s, it_s + 0.015);
        total += cold + (p.warmup + n) * profiled_s;
    }
    total += kValidationS;
    return (total + p.cooldown_s) * kEstimateMargin;
}

SuiteEstimate estimate_suite(const SuitePlan& plan, const PriorResults& prior) {
    SuiteEstimate e;
    std::vector<double> per_config;
    for (const ResolvedConfiguration& c : plan.configurations) {
        const std::optional<TimingBasis> found = prior.find(c);
        (found ? e.from_prior : e.from_heuristic) += 1;
        per_config.push_back(estimate_run_s(c, found ? *found : heuristic_basis(c)));
    }
    // The gate: doctor, the idle sample, each sustained-load leg (a calibration
    // run of 220 inferences, then the leg) and a second for the quick checks.
    const auto has = [&](const char* name) {
        return std::find(plan.preflight.checks.begin(), plan.preflight.checks.end(), name) !=
               plan.preflight.checks.end();
    };
    double gate = has("doctor") ? kPreflightS : 0.0;
    gate += 1.0;
    if (has("idle"))
        gate += plan.preflight.idle_window_s;
    if (has("sustained_load")) {
        for (const ResolvedConfiguration& c : plan.sustained) {
            const std::optional<TimingBasis> found = prior.find(c);
            const TimingBasis b = found ? *found : heuristic_basis(c);
            const bool npu = c.identity.provider == ProviderKind::npu;
            const double overhead = npu ? kNpuProcessOverheadS : kProcessOverheadS;
            const double cold = b.cold_init_s > 0 ? b.cold_init_s : heuristic_basis(c).cold_init_s;
            const double it_s =
                (b.iteration_ms > 0 ? b.iteration_ms : heuristic_basis(c).iteration_ms) / 1000.0;
            gate += 2 * (overhead + 2 * cold) + 240 * it_s + c.sustained_duration_s;
        }
    }
    e.gate_s = gate * kEstimateMargin;
    e.total_s = e.gate_s;
    // Before every run: the load sample when it is longer than the cooldown.
    const bool sampled = has("idle");
    for (const PlannedRun& r : plan.runs) {
        double s = per_config[r.configuration];
        const double cooldown = plan.configurations[r.configuration].params.cooldown_s;
        if (sampled && plan.in_flight.load_sample_s > cooldown)
            s += (plan.in_flight.load_sample_s - cooldown) * kEstimateMargin;
        e.run_s.push_back(s);
        e.total_s += s;
    }
    return e;
}

// ------------------------------------------------------------------- formatting

namespace {

std::string duration_text(double s) {
    const long long total = static_cast<long long>(std::ceil(s));
    char buf[64];
    if (total >= 3600)
        std::snprintf(buf, sizeof(buf), "%lldh %02lldm", total / 3600, (total % 3600 + 59) / 60);
    else if (total >= 60)
        std::snprintf(buf, sizeof(buf), "%lldm %02llds", total / 60, total % 60);
    else
        std::snprintf(buf, sizeof(buf), "%llds", total);
    return buf;
}

std::string pad(std::string s, std::size_t width) {
    if (s.size() < width)
        s.append(width - s.size(), ' ');
    return s;
}

}  // namespace

std::string format_plan(const SuitePlan& plan, const SuiteEstimate& est) {
    std::ostringstream o;
    o << "Suite       " << plan.name << " (" << path_to_utf8(plan.config_file) << ")\n";
    o << "Config hash " << plan.config_sha256 << "\n";
    auto count = [](std::size_t n, const char* word) {
        return std::to_string(n) + " " + word + (n == 1 ? "" : "s");
    };
    o << "Plan        " << count(plan.configurations.size(), "configuration") << " x "
      << count(static_cast<std::size_t>(plan.repetitions), "repetition") << " = "
      << count(plan.runs.size(), "run") << ", rotation " << to_string(plan.rotation) << "\n\n";
    std::size_t model_w = 5;
    for (const ResolvedConfiguration& c : plan.configurations)
        model_w = std::max(model_w, c.identity.model.size());
    o << pad("order", 6) << pad("run id", 14) << pad("model", model_w + 2) << pad("provider", 10)
      << pad("precision", 14) << pad("batch", 7) << pad("rep", 5) << "estimate\n";
    for (std::size_t i = 0; i < plan.runs.size(); ++i) {
        const PlannedRun& r = plan.runs[i];
        const ResolvedConfiguration& c = plan.configurations[r.configuration];
        std::string precision = c.identity.precision;
        if (c.compute_precision != c.identity.precision)
            precision += "->" + c.compute_precision;
        std::string extra;
        if (c.identity.htp_graph_finalization_mode)
            extra += " finalization " + *c.identity.htp_graph_finalization_mode;
        if (c.params.init != InitMode::cold)
            extra += " init " + std::string(to_string(c.params.init));
        if (c.variant && c.variant->find('_') != std::string::npos)
            extra += " " + *c.variant;
        o << pad(std::to_string(r.order_index), 6) << pad(r.short_id(), 14)
          << pad(c.identity.model, model_w + 2)
          << pad(std::string(cli_name(c.identity.provider)), 10) << pad(precision, 14)
          << pad(std::to_string(c.identity.batch), 7) << pad(std::to_string(r.repetition), 5)
          << duration_text(est.run_s[i]) << extra << "\n";
    }
    o << "\nEstimate    " << duration_text(est.total_s) << " for " << plan.runs.size() << " runs ("
      << est.from_prior << " of " << plan.configurations.size()
      << " configurations estimated from earlier results, " << est.from_heuristic
      << " from a heuristic; cooldowns, process starts, the pre-flight gate ("
      << duration_text(est.gate_s) << ") and a "
      << static_cast<int>(std::lround((kEstimateMargin - 1) * 100)) << "% margin included)\n";
    return o.str();
}

std::string format_summary(const SuiteManifest& m, double wall_s) {
    std::ostringstream o;
    std::map<std::string, int> counts;
    for (const RunRecord& r : m.runs)
        ++counts[std::string(to_string(r.status))];
    o << "\nSuite " << m.suite_name << ": " << m.status << "\n";
    o << "Runs        " << counts["done"] << " done, " << counts["failed"] << " failed, "
      << counts["skipped"] << " skipped, " << counts["pending"] << " pending, " << counts["running"]
      << " running (" << m.runs.size() << " planned)\n";
    o << "Wall time   " << duration_text(wall_s) << " (estimate "
      << duration_text(m.estimate.total_s) << ")\n";
    if (!m.preflight.empty()) {
        const PreflightAttempt& a = m.preflight.back();
        o << "Gate        ";
        if (a.skipped) {
            o << "skipped";
        } else {
            o << (m.gate_passed() ? "passed" : a.passed ? "not passed for publishing (a check warned or is missing)" : "failed");
            for (const GateResult& g : a.checks)
                o << "; " << g.name << " " << to_string(g.status);
        }
        o << "\n";
    }
    int retries = 0;
    for (const RunRecord& r : m.runs)
        retries += r.retries;
    char wait_text[96];
    std::snprintf(wait_text, sizeof(wait_text),
                  "%.0f s before runs (power or background load), %d retries", m.in_flight_wait_s,
                  retries);
    o << "Waiting     " << wait_text << "\n";
    o << "Publishable " << (m.publishable() ? "yes" : "no");
    for (const std::string& r : m.not_publishable_reasons())
        o << "; " << r;
    o << "\n";
    bool header = false;
    for (const RunRecord& r : m.runs) {
        if (r.status != RunStatus::failed)
            continue;
        if (!header)
            o << "Failed runs\n";
        header = true;
        o << "  " << r.order_index << " " << r.run_id.substr(0, kShortIdLength) << " " << r.label
          << ": "
          << (r.exit_code ? "exit " + std::to_string(*r.exit_code) + " (" +
                                r.exit_code_name.value_or("") + ")"
                          : std::string("no exit code"))
          << (r.failure ? ", " + *r.failure : "") << "; log " << r.log_file.value_or("-") << "\n";
    }
    header = false;
    for (const RunRecord& r : m.runs) {
        if (r.warnings.empty())
            continue;
        if (!header)
            o << "Runs with warnings\n";
        header = true;
        std::string codes;
        for (const std::string& w : r.warnings)
            codes += (codes.empty() ? "" : ", ") + w;
        o << "  " << r.order_index << " " << r.run_id.substr(0, kShortIdLength) << " " << r.label
          << ": " << codes << "\n";
    }
    return o.str();
}

std::optional<fs::path> find_repository(const std::vector<fs::path>& starts) {
    std::error_code ec;
    for (const fs::path& start : starts) {
        for (fs::path dir = fs::absolute(start, ec); !dir.empty(); dir = dir.parent_path()) {
            if (fs::is_regular_file(dir / "scripts" / "validate_results.py", ec))
                return dir;
            if (dir == dir.parent_path())
                break;
        }
    }
    return std::nullopt;
}

// ----------------------------------------------------------------------- running

namespace {

enum class Attempt { ran, interrupted, gate_lost };

// Keeps the system from sleeping while the suite runs (the display may still turn
// off). Released on destruction; Windows releases it when the process ends.
struct KeepAwake {
    KeepAwake() { SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED); }
    ~KeepAwake() { SetThreadExecutionState(ES_CONTINUOUS); }
    KeepAwake(const KeepAwake&) = delete;
    KeepAwake& operator=(const KeepAwake&) = delete;
};

struct Runner {
    const SuiteOptions& options;
    const SuiteHooks& hooks;
    SuitePlan plan;
    SuiteManifest manifest;
    fs::path dir;
    fs::path python;
    fs::path validator;
    fs::path repo;

    Result<void> save() {
        manifest.updated_utc = now_utc();
        return replace_file_atomically(dir / std::string(kManifestFileName),
                                       to_json(manifest).dump(2) + "\n");
    }

    int save_or_fail() {
        if (Result<void> s = save(); !s) {
            print_error(s.error().kind, "cannot write the suite manifest: " + s.error().message);
            return to_int(ExitCode::output_error);
        }
        return 0;
    }

    fs::path run_dir(const RunRecord& r) const {
        return dir / "runs" / r.run_id.substr(0, kShortIdLength);
    }

    // Moves a run's output aside (an interrupted or failed attempt), so the next
    // attempt starts in an empty directory.
    void discard_output(RunRecord& r, const std::string& reason) {
        std::error_code ec;
        const fs::path out = run_dir(r);
        ojson entry = {{"attempt", r.attempts},
                       {"status", to_string(r.status)},
                       {"exit_code", opt(r.exit_code)},
                       {"started_utc", opt(r.started_utc)},
                       {"ended_utc", opt(r.ended_utc)},
                       {"warnings", r.warnings},
                       {"in_flight", r.in_flight},
                       {"reason", reason},
                       {"moved_to", nullptr}};
        if (fs::exists(out, ec)) {
            fs::create_directories(dir / "discarded", ec);
            for (int n = 1;; ++n) {
                const fs::path target =
                    dir / "discarded" /
                    (r.run_id.substr(0, kShortIdLength) + "-" + std::to_string(n));
                if (fs::exists(target, ec))
                    continue;
                fs::rename(out, target, ec);
                if (!ec) {
                    r.discarded.push_back(rel(target, dir));
                    entry["moved_to"] = rel(target, dir);
                }
                break;
            }
        }
        if (r.attempts > 0)
            r.history.push_back(entry);
        r.result_file.reset();
        r.result_sha256.reset();
        r.validation.reset();
        r.validation_issues.clear();
        r.warnings.clear();
        r.cache_build = nullptr;
        r.log_file.reset();
        r.output_dir.reset();
    }

    // validate_results.py on a result: passed, with any issue lines.
    std::pair<bool, std::vector<std::string>> validate(const fs::path& result,
                                                       const fs::path& out) {
        const fs::path report = out / "validation.txt";
        const ChildOutcome v = run_child(python, {path_to_utf8(validator), path_to_utf8(result)},
                                         out / "validation.log", report);
        std::vector<std::string> issues;
        std::istringstream lines(read_text(report));
        for (std::string line; std::getline(lines, line);) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line.rfind("  ", 0) == 0)
                issues.push_back(line.substr(2));
        }
        if (!v.started)
            issues.push_back(v.start_error);
        return {v.started && v.exit_code == 0, issues};
    }

    std::optional<fs::path> find_result(const fs::path& out, const std::string& name) {
        std::error_code ec;
        std::optional<fs::path> found;
        for (const auto& e : fs::directory_iterator(out, ec)) {
            const std::string f = path_to_utf8(e.path().filename());
            if (e.is_regular_file(ec) && f.rfind(name + "_", 0) == 0 && f.ends_with(".json") &&
                !f.ends_with("_latest.json"))
                found = e.path();
        }
        return found;
    }

    bool done_result_valid(const RunRecord& r) {
        if (!r.result_file || !r.result_sha256)
            return false;
        const fs::path file = dir / from_utf8(*r.result_file);
        const Result<std::string> sha = sha256_file(file);
        if (!sha || sha.value() != *r.result_sha256)
            return false;
        return validate(file, file.parent_path()).first;
    }

    static std::vector<std::string> warning_codes(const json& doc) {
        std::vector<std::string> codes;
        for (const json& w : doc.value("warnings", json::array())) {
            const std::string c = w.value("code", "");
            if (c != "value_unavailable" && std::find(codes.begin(), codes.end(), c) == codes.end())
                codes.push_back(c);
        }
        return codes;
    }

    SuiteLink link(const RunRecord& r, double cooldown) const {
        SuiteLink l;
        l.suite_name = plan.name;
        l.config_hash = plan.config_sha256;
        l.run_id = r.run_id;
        l.repetition = r.repetition;
        l.order_index = r.order_index;
        l.cooldown_s_before = cooldown;
        l.preflight_skipped = manifest.preflight_ever_skipped;
        l.attempt = r.attempts;
        l.gate_passed = manifest.gate_passed();
        l.publishable = manifest.publishable();
        l.in_flight_wait_s = run_wait_s;
        return l;
    }

    bool wait_s(double seconds) {  // true if interrupted
        return seconds > 0 && wait_or_stop(std::chrono::milliseconds(std::lround(seconds * 1000)));
    }

    bool guarded(const char* check) const {
        return !options.skip_preflight &&
               std::find(plan.preflight.checks.begin(), plan.preflight.checks.end(), check) !=
                   plan.preflight.checks.end();
    }

    // Before a run: AC + Best performance (waiting for it if lost), the cooldown,
    // and a load sample (waiting with backoff while it is too high). Fills
    // r.in_flight; returns ran (go ahead), interrupted or gate_lost.
    Attempt in_flight_check(RunRecord& r, const ResolvedConfiguration& c) {
        const InFlightSettings& f = plan.in_flight;
        ojson d = {{"checked_utc", now_utc()}};
        run_wait_s = 0;
        if (guarded("power")) {
            PowerState ps = query_power_state();
            d["power_ready"] = power_ready(ps);
            double waited = 0;
            if (!power_ready(ps)) {
                const std::string what = evaluate_power(ps).summary;
                print_warning("gate_lost",
                              "before run " + std::to_string(r.order_index + 1) + ": " + what);
                if (f.on_gate_loss == "stop") {
                    manifest.gate_lost = what;
                    r.in_flight = d;
                    return Attempt::gate_lost;
                }
                bool restored = false;
                while (waited < f.power_max_wait_s) {
                    const double step = std::min(15.0, f.power_max_wait_s - waited);
                    if (wait_s(step))
                        return Attempt::interrupted;
                    waited += step;
                    if (power_ready(query_power_state())) {
                        restored = true;
                        break;
                    }
                }
                d["power_wait_s"] = waited;
                run_wait_s += waited;
                if (!restored) {
                    manifest.gate_lost = what + " (not restored within " +
                                         format_seconds(f.power_max_wait_s) + " s)";
                    r.in_flight = d;
                    return Attempt::gate_lost;
                }
                std::fprintf(stderr, "suite: power restored after %.0f s\n", waited);
            }
            d["power_wait_s"] = waited;
        }
        const double cooldown = c.params.cooldown_s;
        const bool sample = guarded("idle");
        const double first = sample ? std::max(cooldown, f.load_sample_s) : cooldown;
        if (first > 0)
            std::fprintf(stderr, "suite: cooldown %s s before run %d of %zu (%s)\n",
                         format_seconds(first).c_str(), r.order_index + 1, manifest.runs.size(),
                         r.label.c_str());
        LoadProbe probe = start_load_probe();
        if (wait_s(first))
            return Attempt::interrupted;
        if (sample) {
            const double limit = plan.preflight.idle_max_cpu_ratio;
            ojson samples = ojson::array();
            auto take = [&](const LoadProbe& p) {
                const std::optional<LoadSample> l = finish_load_probe(p);
                const double ratio = l ? l->cpu_utilization_ratio : 1.0;
                ojson top = ojson::array();
                if (l && l->attribution)
                    for (const ProcessShare& s : l->attribution->top_processes)
                        top.push_back({{"name", s.name}, {"cpu_share", s.cpu_share}});
                samples.push_back({{"cpu_utilization_ratio", l ? ojson(ratio) : ojson(nullptr)},
                                   {"window_s", l ? l->sample_window_s : 0.0},
                                   {"top_processes", top}});
                return ratio;
            };
            double ratio = take(probe);
            double waited = 0;
            for (double w : backoff_schedule(ratio > limit ? f.max_wait_s : 0)) {
                std::fprintf(stderr,
                             "suite: background load %.1f%% (limit %.1f%%); waiting %.0f s before "
                             "run %d\n",
                             ratio * 100, limit * 100, w, r.order_index + 1);
                if (wait_s(w))
                    return Attempt::interrupted;
                waited += w;
                LoadProbe again = start_load_probe();
                if (wait_s(f.load_sample_s))
                    return Attempt::interrupted;
                waited += f.load_sample_s;
                ratio = take(again);
                if (ratio <= limit)
                    break;
            }
            d["max_cpu_ratio"] = limit;
            d["load_samples"] = samples;
            d["load_wait_s"] = waited;
            d["ran_under_load"] = ratio > limit;
            run_wait_s += waited;
            if (ratio > limit)
                print_warning("background_cpu_load_high",
                              "run " + std::to_string(r.order_index + 1) + " starts with " +
                                  format_seconds(std::round(ratio * 1000) / 10) +
                                  "% background load after waiting " + format_seconds(waited) +
                                  " s");
        }
        d["wait_s"] = run_wait_s;
        manifest.in_flight_wait_s += run_wait_s;
        r.in_flight = d;
        return Attempt::ran;
    }

    // One attempt of one run.
    Attempt execute(RunRecord& r, const ResolvedConfiguration& c) {
        std::error_code ec;
        if (const Attempt a = in_flight_check(r, c); a != Attempt::ran)
            return a;
        const double cooldown = c.params.cooldown_s;
        const fs::path out = run_dir(r);
        fs::create_directories(out, ec);
        r.status = RunStatus::running;
        ++r.attempts;
        r.started_utc = now_utc();
        r.ended_utc.reset();
        r.duration_s.reset();
        r.exit_code.reset();
        r.exit_code_name.reset();
        r.failure.reset();
        r.cooldown_s_before = cooldown;
        r.output_dir = rel(out, dir);
        r.log_file = rel(out / "run.log", dir);
        save_or_fail();
        std::fprintf(stderr, "suite: run %d of %zu: %s (repetition %d, %s)\n", r.order_index + 1,
                     manifest.runs.size(), r.label.c_str(), r.repetition,
                     r.run_id.substr(0, kShortIdLength).c_str());
        const auto t0 = Clock::now();
        const std::string name = r.label + "_rep" + std::to_string(r.repetition);
        const fs::path cache_dir = dir / "qnn-context";

        if (c.params.init == InitMode::cached) {
            const fs::path build_out = out / "cache_build";
            RunConfig build = cache_build_config(c, build_out, name, cache_dir);
            const ChildOutcome b =
                run_child(options.exe, run_arguments(build), out / "run.log", std::nullopt);
            const std::optional<fs::path> build_result = find_result(build_out, name);
            r.cache_build = {
                {"exit_code", b.started ? ojson(b.exit_code) : ojson(nullptr)},
                {"exit_code_name", b.started ? ojson(exit_name(b.exit_code)) : ojson(nullptr)},
                {"seconds", b.seconds},
                {"result_file", build_result ? ojson(rel(*build_result, dir)) : ojson(nullptr)}};
            if (b.interrupted)
                return Attempt::interrupted;
            if (!b.started || b.exit_code != 0) {
                r.status = RunStatus::failed;
                r.exit_code = b.started ? std::optional<int>(b.exit_code) : std::nullopt;
                r.exit_code_name =
                    b.started ? std::optional<std::string>(exit_name(b.exit_code)) : std::nullopt;
                r.failure = b.started ? "the cache-building run failed" : b.start_error;
                r.ended_utc = now_utc();
                r.duration_s = std::chrono::duration<double>(Clock::now() - t0).count();
                return Attempt::ran;
            }
        }

        RunConfig run = to_run_config(c, out, name, cache_dir);
        run.suite = link(r, cooldown);
        const ChildOutcome child =
            run_child(options.exe, run_arguments(run), out / "run.log", std::nullopt, [&] {
                if (hooks.run_started)
                    hooks.run_started(r);
            });
        r.duration_s = std::chrono::duration<double>(Clock::now() - t0).count();
        r.ended_utc = now_utc();
        if (child.interrupted)
            return Attempt::interrupted;
        if (!child.started) {
            r.status = RunStatus::failed;
            r.failure = child.start_error;
            return Attempt::ran;
        }
        r.exit_code = child.exit_code;
        r.exit_code_name = exit_name(child.exit_code);
        const std::optional<fs::path> result = find_result(out, name);
        if (result) {
            r.result_file = rel(*result, dir);
            if (const Result<std::string> sha = sha256_file(*result))
                r.result_sha256 = sha.value();
            if (const std::optional<json> doc = read_json(*result))
                r.warnings = warning_codes(*doc);
        }
        if (child.exit_code != 0) {
            r.status = RunStatus::failed;
            r.failure = "edgebench run exited " + std::to_string(child.exit_code);
            return Attempt::ran;
        }
        if (!result) {
            r.status = RunStatus::failed;
            r.failure = "no result file";
            return Attempt::ran;
        }
        const auto [passed, issues] = validate(*result, out);
        r.validation = passed ? "passed" : "failed";
        r.validation_issues = issues;
        if (!passed) {
            r.status = RunStatus::failed;
            r.failure = "the result does not pass scripts/validate_results.py";
            return Attempt::ran;
        }
        r.status = RunStatus::done;
        return Attempt::ran;
    }

    int preflight() {
        PreflightAttempt attempt;
        attempt.started_utc = now_utc();
        if (options.skip_preflight) {
            attempt.skipped = true;
            attempt.passed = false;
            manifest.preflight_ever_skipped = true;
            manifest.preflight.push_back(attempt);
            print_warning("preflight_skipped",
                          "the pre-flight check is skipped: nothing from this suite is "
                          "publishable");
            return save_or_fail();
        }
        PreflightContext ctx{options.exe, dir, static_cast<int>(manifest.preflight.size()) + 1,
                             &plan, estimate_s};
        const PreflightSettings& ps = plan.preflight;
        ojson legs = ojson::array();
        for (const ResolvedConfiguration& c : plan.sustained)
            legs.push_back({{"label", c.label},
                            {"model_file", c.model_file_display},
                            {"duration_s", c.sustained_duration_s}});
        attempt.settings = {
            {"checks", ps.checks},
            {"fail_on_warn", ps.fail_on_warn},
            {"idle", {{"window_s", ps.idle_window_s}, {"max_cpu_ratio", ps.idle_max_cpu_ratio}}},
            {"sleep", {{"margin_s", ps.sleep_margin_s}, {"estimate_s", estimate_s}}},
            {"sustained_load",
             {{"window_s", ps.sustained_window_s},
              {"max_drift_ratio", ps.sustained_max_drift_ratio},
              {"legs", legs}}},
            {"in_flight",
             {{"load_sample_s", plan.in_flight.load_sample_s},
              {"max_wait_s", plan.in_flight.max_wait_s},
              {"power_max_wait_s", plan.in_flight.power_max_wait_s},
              {"on_gate_loss", plan.in_flight.on_gate_loss}}},
            {"retry_flagged", plan.retry_flagged}};
        for (std::unique_ptr<PreflightCheck>& check :
             make_preflight_checks(plan.preflight.checks)) {
            std::fprintf(stderr, "suite: pre-flight %s\n", check->name().c_str());
            GateResult g = check->run(ctx);
            std::fprintf(stderr, "suite: pre-flight %s: %s, %s\n", g.name.c_str(),
                         std::string(to_string(g.status)).c_str(), g.summary.c_str());
            if (g.name == "doctor" && g.details.contains("versions"))
                attempt_versions = g.details["versions"];
            attempt.checks.push_back(std::move(g));
            if (stop_requested())
                break;
        }
        attempt.passed =
            gate_passed(attempt.checks, plan.preflight.fail_on_warn) && !stop_requested();
        manifest.preflight.push_back(attempt);
        if (stop_requested())
            return to_int(ExitCode::interrupted);
        if (!attempt.passed) {
            manifest.status = "aborted";
            if (int code = save_or_fail(); code != 0)
                return code;
            print_error(
                "preflight_failed",
                "the pre-flight gate failed; no run was started (see " +
                    path_to_utf8(dir / std::string(kManifestFileName)) + ")",
                "Fix what the failed check reports, then resume with 'edgebench suite --resume " +
                    path_to_utf8(dir) + "'.");
            return to_int(ExitCode::internal_error);
        }
        return save_or_fail();
    }

    ojson attempt_versions = nullptr;
    double estimate_s = 0;
    double run_wait_s = 0;  // waits before the current run
};

ojson tool_json() {
    const BuildInfo& b = build_info();
    return {{"name", "edgebench"},
            {"version", std::string(version())},
            {"git_commit", b.git_commit ? ojson(*b.git_commit) : ojson(nullptr)},
            {"git_dirty", b.git_dirty ? ojson(*b.git_dirty) : ojson(nullptr)},
            {"build_type", b.build_type},
            {"results_schema_version", std::string(results_schema_version())}};
}

}  // namespace

int run_suite(const SuiteOptions& options, const SuiteHooks& hooks) {
    install_interrupt_handler(InterruptMode::stop_gracefully);
    const auto wall_start = Clock::now();
    std::error_code ec;

    // The configuration, and for --resume the manifest it must match.
    std::optional<SuiteManifest> previous;
    fs::path suite_dir;
    fs::path config_file;
    if (options.resume) {
        suite_dir = fs::absolute(*options.resume, ec);
        const std::optional<json> doc = read_json(suite_dir / std::string(kManifestFileName));
        if (!doc) {
            print_error("suite_manifest_missing", "no readable " + std::string(kManifestFileName) +
                                                      " in " + path_to_utf8(suite_dir));
            return to_int(ExitCode::usage_error);
        }
        Result<SuiteManifest> m = manifest_from_json(*doc);
        if (!m) {
            print_error(m.error().kind, m.error().message);
            return to_int(ExitCode::usage_error);
        }
        previous = std::move(m.value());
        config_file = options.config ? *options.config : from_utf8(previous->config_file);
    } else {
        config_file = *options.config;
    }
    Result<SuitePlan> loaded = load_suite(config_file, options.output_dir);
    if (!loaded) {
        print_error(loaded.error().kind, loaded.error().message,
                    "See docs/suite.md for the configuration format.");
        return to_int(loaded.error().exit_code);
    }
    SuitePlan plan = std::move(loaded.value());

    if (previous) {
        if (plan.config_sha256 != previous->config_sha256) {
            print_error("suite_config_changed",
                        "the configuration " + path_to_utf8(plan.config_file) +
                            " is not the one this suite was started with (SHA256 " +
                            plan.config_sha256.substr(0, 12) + "..., the suite's " +
                            previous->config_sha256.substr(0, 12) + "...)",
                        "Resume with the original configuration, or start a new suite.");
            return to_int(ExitCode::usage_error);
        }
        bool same = plan.runs.size() == previous->runs.size();
        for (std::size_t i = 0; same && i < plan.runs.size(); ++i)
            same = plan.runs[i].run_id == previous->runs[i].run_id;
        if (!same) {
            print_error("suite_plan_changed",
                        "the configuration is unchanged but its runs are not (a model file "
                        "changed since the suite started)",
                        "Restore the model files, or start a new suite.");
            return to_int(ExitCode::usage_error);
        }
        if (previous->tool.value("version", "") != std::string(version()) ||
            previous->tool.value("git_commit", ojson(nullptr)) !=
                tool_json().value("git_commit", ojson(nullptr))) {
            print_error("suite_tool_changed",
                        "this edgebench build (" + tool_json().dump() +
                            ") is not the one the suite was started with (" +
                            previous->tool.dump() + ")",
                        "Resume with the original build, or start a new suite.");
            return to_int(ExitCode::usage_error);
        }
    }

    // Repository: the validator and earlier results for the estimate.
    std::error_code ec2;
    const std::optional<fs::path> repo = find_repository(
        {plan.config_file.parent_path(), fs::current_path(ec2), options.exe.parent_path()});
    std::vector<fs::path> prior_roots;
    if (repo)
        prior_roots.push_back(*repo / "benchmarks" / "results");
    const PriorResults prior = PriorResults::scan(prior_roots);
    const SuiteEstimate estimate = estimate_suite(plan, prior);

    if (options.dry_run) {
        std::fputs(format_plan(plan, estimate).c_str(), stdout);
        return to_int(ExitCode::success);
    }

    const KeepAwake awake;
    Runner runner{options, hooks, std::move(plan), {}, {}, {}, {}, {}};
    runner.estimate_s = estimate.total_s;
    SuitePlan& p = runner.plan;
    if (options.python)
        runner.python = fs::absolute(*options.python, ec);
    else if (repo)
        runner.python = *repo / ".venv" / "Scripts" / "python.exe";
    if (repo)
        runner.validator = *repo / "scripts" / "validate_results.py";
    if (!repo || !fs::is_regular_file(runner.python, ec)) {
        print_error("validator_unavailable",
                    !repo ? std::string("scripts/validate_results.py was not found above the "
                                        "configuration file or the executable")
                          : "the Python interpreter " + path_to_utf8(runner.python) +
                                " was not found; every result is validated with "
                                "scripts/validate_results.py",
                    "Pass --python PATH (an interpreter with jsonschema installed).");
        return to_int(ExitCode::usage_error);
    }

    if (previous) {
        runner.dir = suite_dir;
        runner.manifest = std::move(*previous);
    } else {
        // A new directory: <output root>/<name>_<stamp>[-n].
        fs::create_directories(p.output_root, ec);
        const std::string stamp = file_stamp(std::chrono::system_clock::now());
        for (int n = 1;; ++n) {
            const fs::path candidate =
                p.output_root /
                from_utf8(p.name + "_" + stamp + (n == 1 ? "" : "-" + std::to_string(n)));
            if (CreateDirectoryW(candidate.c_str(), nullptr)) {
                runner.dir = candidate;
                break;
            }
            if (GetLastError() != ERROR_ALREADY_EXISTS || n > 100) {
                print_error("suite_dir_not_writable",
                            "cannot create the suite directory " + path_to_utf8(candidate),
                            "Choose another output directory (--output-dir).");
                return to_int(ExitCode::output_error);
            }
        }
        if (Result<void> copy = create_file_atomically(runner.dir / std::string(kConfigCopyName),
                                                       read_text(p.config_file));
            !copy) {
            print_error(copy.error().kind, copy.error().message);
            return to_int(ExitCode::output_error);
        }
        SuiteManifest& m = runner.manifest;
        m.suite_name = p.name;
        m.config_file = path_to_utf8(p.config_file);
        m.config_sha256 = p.config_sha256;
        m.config = p.config;
        m.created_utc = now_utc();
        m.tool = tool_json();
        m.rotation = std::string(to_string(p.rotation));
        m.repetitions = p.repetitions;
        for (const PlannedRun& pr : p.runs) {
            const ResolvedConfiguration& c = p.configurations[pr.configuration];
            RunRecord r;
            r.run_id = pr.run_id;
            r.configuration_id = c.configuration_id;
            r.label = c.label;
            r.order_index = pr.order_index;
            r.repetition = pr.repetition;
            r.configuration = c.resolved;
            r.configuration["model_file_path"] = path_to_utf8(c.model_file);
            r.configuration["source"] = c.source;
            r.configuration["label"] = c.label;
            m.runs.push_back(std::move(r));
        }
    }
    SuiteManifest& m = runner.manifest;
    m.python = path_to_utf8(runner.python);
    m.validator = path_to_utf8(runner.validator);
    m.estimate = estimate;
    for (std::size_t i = 0; i < m.runs.size(); ++i)
        m.runs[i].estimate_s = estimate.run_s[i];
    m.status = "running";
    Invocation inv;
    inv.started_utc = now_utc();
    inv.resumed = previous.has_value() || options.resume.has_value();
    inv.fail_fast = options.fail_fast;
    inv.skip_preflight = options.skip_preflight;
    m.invocations.push_back(inv);
    if (int code = runner.save_or_fail(); code != 0)
        return code;

    std::fputs(format_plan(p, estimate).c_str(), stdout);
    std::printf("Directory   %s\n\n", path_to_utf8(runner.dir).c_str());
    std::fflush(stdout);

    auto finish = [&](int code) {
        m.invocations.back().ended_utc = now_utc();
        m.invocations.back().exit_code = code;
        const int write = runner.save_or_fail();
        std::fputs(
            format_summary(m, std::chrono::duration<double>(Clock::now() - wall_start).count())
                .c_str(),
            stdout);
        std::printf("Manifest    %s\n",
                    path_to_utf8(runner.dir / std::string(kManifestFileName)).c_str());
        std::fflush(stdout);
        notify_shutdown_complete();
        return write != 0 ? write : code;
    };

    // The pre-flight gate, again on every resume.
    const ojson earlier_versions = m.versions;
    if (int code = runner.preflight(); code != 0) {
        if (code == to_int(ExitCode::interrupted))
            m.status = "interrupted";
        return finish(code);
    }
    if (!runner.attempt_versions.is_null()) {
        if (!earlier_versions.is_null() && earlier_versions != runner.attempt_versions) {
            m.status = "aborted";
            print_error("suite_versions_changed",
                        "ONNX Runtime, QNN EP or QAIRT SDK versions differ from the suite's "
                        "earlier runs: " +
                            runner.attempt_versions.dump() + " vs " + earlier_versions.dump(),
                        "Start a new suite.");
            return finish(to_int(ExitCode::usage_error));
        }
        m.versions = runner.attempt_versions;
    }

    // Resume: decide per run.
    if (previous || options.resume) {
        for (RunRecord& r : m.runs) {
            const ResumeAction a =
                resume_action(r, [&](const RunRecord& x) { return runner.done_result_valid(x); });
            if (a == ResumeAction::keep)
                continue;
            if (a == ResumeAction::discard || a == ResumeAction::rerun)
                runner.discard_output(r, a == ResumeAction::discard ? "resume: interrupted mid-run"
                                         : r.status == RunStatus::failed
                                             ? "resume: failed"
                                             : "resume: result missing or no longer valid");
            r.status = RunStatus::pending;
            r.failure.reset();
        }
        if (int code = runner.save_or_fail(); code != 0)
            return finish(code);
    }

    bool interrupted = false, aborted = false, any_failed = false, gate_lost = false;
    // One attempt of a run; false: stop the suite (interrupted or gate lost).
    auto attempt = [&](RunRecord& r) {
        const auto conf = std::find_if(p.configurations.begin(), p.configurations.end(),
                                       [&](const ResolvedConfiguration& c) {
                                           return c.configuration_id == r.configuration_id;
                                       });
        const Attempt a = runner.execute(r, *conf);
        if (a != Attempt::ran) {
            // The partial attempt (if any) is set aside and the run stays pending.
            if (r.status == RunStatus::running)
                runner.discard_output(r, a == Attempt::interrupted ? "interrupted" : "gate lost");
            r.status = RunStatus::pending;
            r.failure.reset();
            r.exit_code.reset();
            r.exit_code_name.reset();
            (a == Attempt::interrupted ? interrupted : gate_lost) = true;
            runner.save_or_fail();
            return false;
        }
        if (r.status == RunStatus::failed) {
            any_failed = true;
            std::fprintf(stderr, "suite: run %d failed: %s (log %s)\n", r.order_index + 1,
                         r.failure.value_or("").c_str(), r.log_file.value_or("-").c_str());
            if (options.fail_fast)
                aborted = true;
        }
        runner.save_or_fail();
        if (hooks.run_finished)
            hooks.run_finished(r);
        return true;
    };
    for (RunRecord& r : m.runs) {
        if (r.status == RunStatus::done)
            continue;
        if (aborted) {
            r.status = RunStatus::skipped;
            r.failure = "not run: an earlier run failed and --fail-fast was given";
            continue;
        }
        if (stop_requested()) {
            interrupted = true;
            break;
        }
        if (!attempt(r))
            break;
    }
    // Runs flagged for background load or trial spread run again (retry_flagged).
    for (int pass = 0; pass < p.retry_flagged && !interrupted && !gate_lost && !aborted; ++pass) {
        for (RunRecord& r : m.runs) {
            if (!retry_worthy(r) || r.retries >= p.retry_flagged || interrupted || gate_lost)
                continue;
            std::string codes;
            for (const std::string& w : r.warnings)
                if (w == "background_cpu_load_high" || w == "trial_spread_high")
                    codes += (codes.empty() ? "" : ", ") + w;
            std::fprintf(stderr, "suite: run %d again (%s)\n", r.order_index + 1, codes.c_str());
            runner.discard_output(r, "retry_flagged: " + codes);
            ++r.retries;
            r.status = RunStatus::pending;
            if (stop_requested()) {
                interrupted = true;
                break;
            }
            attempt(r);
        }
    }
    if (gate_lost) {
        m.status = "aborted";
        print_error("gate_lost", "the suite stopped: " + m.gate_lost.value_or("power state lost"),
                    "Restore AC power and Best performance, then resume with 'edgebench suite "
                    "--resume " +
                        path_to_utf8(runner.dir) + "'.");
        return finish(to_int(ExitCode::internal_error));
    }
    if (interrupted) {
        m.status = "interrupted";
        std::fprintf(stderr, "suite: interrupted; resume with: edgebench suite --resume %s\n",
                     path_to_utf8(runner.dir).c_str());
        return finish(to_int(ExitCode::interrupted));
    }
    m.status = aborted ? "aborted" : "completed";
    any_failed = any_failed || std::any_of(m.runs.begin(), m.runs.end(), [](const RunRecord& r) {
                     return r.status != RunStatus::done;
                 });
    return finish(any_failed ? to_int(ExitCode::internal_error) : to_int(ExitCode::success));
}

}  // namespace edgebench::suite
