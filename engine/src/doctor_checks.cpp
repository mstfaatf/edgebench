#include "edgebench/doctor_checks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "edgebench/npu_selection.hpp"

namespace edgebench {

namespace {

using json = nlohmann::ordered_json;

json opt(const std::optional<std::string>& v) {
    return v ? json(*v) : json(nullptr);
}

std::string fmt(double v, int decimals) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

std::string join(const std::vector<std::string>& items, const char* separator = ", ") {
    std::string out;
    for (const std::string& s : items)
        out += (out.empty() ? "" : separator) + s;
    return out;
}

DoctorCheck make(std::string_view name) {
    DoctorCheck c;
    c.name = std::string(name);
    return c;
}

void fail(DoctorCheck& c, std::string code, std::string summary, ExitCode exit_code,
          std::string hint = {}) {
    c.status = CheckStatus::fail;
    c.code = std::move(code);
    c.summary = std::move(summary);
    c.exit_code = exit_code;
    c.hint = std::move(hint);
}

void fail(DoctorCheck& c, const Error& e, const std::string& context, std::string hint = {}) {
    fail(c, e.kind, context + ": " + e.message, e.exit_code, std::move(hint));
}

// Keeps the first warning's code and hint; later ones add to the summary.
void warn(DoctorCheck& c, std::string code, std::string summary, std::string hint = {}) {
    if (c.status == CheckStatus::fail)
        return;
    if (c.status != CheckStatus::warn) {
        c.status = CheckStatus::warn;
        c.code = std::move(code);
        c.summary = std::move(summary);
        c.hint = std::move(hint);
    } else {
        c.summary += "; " + summary;
        if (c.hint.empty())
            c.hint = std::move(hint);
    }
}

void not_applicable(DoctorCheck& c, const std::string& reason) {
    c.status = CheckStatus::not_applicable;
    c.summary = reason;
}

json file_json(const BundleFile& f, const FilePin* pin) {
    json j = {{"name", f.name},
              {"path", path_to_utf8(f.path)},
              {"present", f.present},
              {"size_bytes", f.size_bytes ? json(*f.size_bytes) : json(nullptr)},
              {"sha256", opt(f.sha256)},
              {"product_version", opt(f.product_version)},
              {"file_version", opt(f.file_version)}};
    if (pin) {
        j["pinned_sha256"] = pin->sha256;
        j["pinned_size_bytes"] = pin->size_bytes;
        j["matches_pin"] = f.present && f.sha256 == pin->sha256 && f.size_bytes == pin->size_bytes;
    }
    return j;
}

const BundleFile* find_file(const std::vector<BundleFile>& files, const std::string& name) {
    for (const BundleFile& f : files)
        if (f.name == name)
            return &f;
    return nullptr;
}

json files_json(const std::vector<BundleFile>& files, const std::vector<FilePin>& pins) {
    json out = json::array();
    for (const FilePin& p : pins) {
        if (const BundleFile* f = find_file(files, p.name))
            out.push_back(file_json(*f, &p));
        else
            out.push_back(file_json(BundleFile{p.name, {}, false, {}, {}, {}, {}}, &p));
    }
    return out;
}

json modules_json(const std::vector<LoadedModule>& modules) {
    json out = json::array();
    for (const LoadedModule& m : modules)
        out.push_back({{"name", m.name}, {"path", path_to_utf8(m.path)}});
    return out;
}

std::string describe_modules(const std::vector<LoadedModule>& modules) {
    std::vector<std::string> items;
    for (const LoadedModule& m : modules)
        items.push_back(m.name + " (" + path_to_utf8(m.path) + ")");
    return join(items);
}

json error_json(const std::optional<Error>& e) {
    if (!e)
        return nullptr;
    return {{"kind", e->kind},
            {"message", e->message},
            {"exit_code", to_int(e->exit_code)},
            {"ort_error", e->ort_error_name ? json(*e->ort_error_name) : json(nullptr)}};
}

std::string mib(std::uint64_t bytes) {
    return fmt(static_cast<double>(bytes) / (1024.0 * 1024.0), 1) + " MiB";
}

}  // namespace

std::string_view to_string(CheckStatus status) {
    switch (status) {
        case CheckStatus::pass:
            return "pass";
        case CheckStatus::warn:
            return "warn";
        case CheckStatus::fail:
            return "fail";
        case CheckStatus::not_applicable:
            return "not_applicable";
    }
    return "unknown";
}

std::vector<std::string> bundle_file_problems(const std::vector<BundleFile>& files,
                                              const std::vector<FilePin>& pins) {
    std::vector<std::string> problems;
    for (const FilePin& p : pins) {
        const BundleFile* f = find_file(files, p.name);
        if (!f || !f->present) {
            problems.push_back(p.name + ": missing");
        } else if (f->size_bytes != p.size_bytes || f->sha256 != p.sha256) {
            problems.push_back(p.name + ": SHA256 " + f->sha256.value_or("(unreadable)") +
                               " is not the pinned " + p.sha256);
        }
    }
    return problems;
}

DoctorCheck check_ort_bundle(const OrtLoadResult& load, const std::vector<BundleFile>& files,
                             const std::vector<FilePin>& pins,
                             const std::vector<LoadedModule>& modules,
                             const std::vector<LoadedModule>& outside, bool emulated) {
    DoctorCheck c = make("ort_bundle");
    c.details = {{"library_path", path_to_utf8(load.dll_path)},
                 {"load_check", load.ok() ? json(load.load_check()) : json(nullptr)},
                 {"version", load.ort_version.empty() ? json(nullptr) : json(load.ort_version)},
                 {"api_version", load.ok() ? json(load.api_version) : json(nullptr)},
                 {"required_version", expected_onnxruntime_version()},
                 {"required_api_version", expected_onnxruntime_api_version()},
                 {"files", files_json(files, pins)},
                 {"loaded_modules", modules_json(modules)},
                 {"modules_outside_bundle", modules_json(outside)},
                 {"process_emulated", emulated}};
    if (const BundleFile* dll = find_file(files, "onnxruntime.dll"); dll && dll->product_version)
        c.lines.push_back("file version " + *dll->product_version + ", " + path_to_utf8(dll->path));
    if (!load.ok()) {
        fail(c, std::string(to_string(load.error_kind)),
             "cannot load ONNX Runtime: " + load.message, ExitCode::ort_load_failure,
             std::string(load_error_hint(load.error_kind)));
        return c;
    }
    if (const std::vector<std::string> bad = bundle_file_problems(files, pins); !bad.empty()) {
        fail(c, "bundle_file_mismatch",
             "ONNX Runtime files do not match cmake/bundle-pins.json: " + join(bad, "; "),
             ExitCode::ort_load_failure,
             "Rebuild, or run scripts/fetch_deps.ps1 -Force and rebuild, so the bundle holds the "
             "pinned files.");
        return c;
    }
    if (!outside.empty()) {
        fail(c, "module_outside_bundle",
             "ONNX Runtime / QNN modules are loaded from outside the bundle directory: " +
                 describe_modules(outside),
             ExitCode::ort_load_failure,
             std::string(load_error_hint(OrtLoadError::foreign_module_loaded)));
        return c;
    }
    c.summary = "onnxruntime " + load.ort_version + " (C API " + std::to_string(load.api_version) +
                ") from the bundle, " + std::to_string(pins.size()) +
                " files match their SHA256 pins, " + std::to_string(modules.size()) +
                " ONNX Runtime / QNN modules loaded, all from the bundle";
    if (emulated) {
        const std::string summary = c.summary;
        warn(c, "process_emulated",
             summary +
                 "; this build runs under emulation, so timings would not represent "
                 "native performance",
             "Use the native build for this machine's architecture.");
    }
    return c;
}

DoctorCheck check_qnn_bundle(const std::vector<BundleFile>& files, const std::vector<FilePin>& pins,
                             const QnnRegistration& registration) {
    DoctorCheck c = make("qnn_bundle");
    c.details = {{"files", files_json(files, pins)},
                 {"registration", !registration.attempted ? "not_attempted"
                                                          : (registration.error ? "failed" : "ok")},
                 {"error", error_json(registration.error)}};
    if (pins.empty()) {
        if (registration.error) {  // a plugin file next to a build without the QNN EP
            fail(c, *registration.error, "registering the QNN execution provider");
            return c;
        }
        not_applicable(c,
                       "this build has no QNN execution provider (the NPU needs an ARM64 build)");
        return c;
    }
    if (const std::vector<std::string> bad = bundle_file_problems(files, pins); !bad.empty()) {
        fail(c, "qnn_bundle_mismatch",
             "QNN files do not match cmake/bundle-pins.json: " + join(bad, "; "),
             ExitCode::ort_load_failure,
             "Rebuild, or run scripts/fetch_deps.ps1 -Force and rebuild, so the bundle holds the "
             "pinned QNN execution provider files.");
        return c;
    }
    if (registration.error) {
        fail(c, *registration.error, "registering the QNN execution provider");
        return c;
    }
    if (!registration.attempted) {
        not_applicable(c,
                       "the QNN execution provider was not registered: ONNX Runtime is not "
                       "loaded");
        return c;
    }
    std::vector<std::string> names;
    for (const FilePin& p : pins)
        names.push_back(p.name);
    c.summary = std::to_string(pins.size()) +
                " QNN files match their SHA256 pins; the QNN "
                "execution provider registers";
    c.lines.push_back(join(names));
    if (const BundleFile* htp = find_file(files, "QnnHtp.dll"); htp && htp->product_version)
        c.lines.push_back("QnnHtp.dll version " + *htp->product_version);
    return c;
}

DoctorCheck check_backend_versions(const BackendVersionFacts& f) {
    DoctorCheck c = make("backend_versions");
    c.details = {{"backend_build", opt(f.backend_build)},
                 {"sdk_version", opt(f.sdk_version)},
                 {"source", opt(f.source)},
                 {"pinned_sdk_version", f.pinned_sdk_version},
                 {"backend_file_product_version", opt(f.backend_file_product_version)}};
    if (f.not_applicable) {
        not_applicable(c, *f.not_applicable);
        return c;
    }
    if (!f.backend_build) {
        warn(c, "backend_build_unavailable",
             "the QNN execution provider did not report its backend's build id",
             "NPU results will record the SDK version as unavailable.");
        return c;
    }
    c.summary = "backend build " + *f.backend_build + ", SDK " + f.sdk_version.value_or("?") +
                " (reported by the QNN EP before any session)";
    if (f.sdk_version != f.pinned_sdk_version) {
        warn(c, "qnn_sdk_version_disagreement",
             "the backend reports SDK " + f.sdk_version.value_or("(none)") +
                 ", the QNN EP release declares " + f.pinned_sdk_version,
             "Check that the bundled QNN files come from the pinned release.");
    }
    const std::string build =
        f.backend_build->front() == 'v' ? f.backend_build->substr(1) : *f.backend_build;
    if (f.backend_file_product_version && build != *f.backend_file_product_version) {
        warn(c, "qnn_backend_version_disagreement",
             "the backend reports build " + *f.backend_build +
                 ", QnnHtp.dll's version resource "
                 "says " +
                 *f.backend_file_product_version,
             "Check that the bundled QNN files come from the pinned release.");
    }
    if (c.status == CheckStatus::pass) {
        c.lines.push_back("agrees with the release's SDK " + f.pinned_sdk_version +
                          " and QnnHtp.dll " + f.backend_file_product_version.value_or("(?)"));
    }
    return c;
}

DoctorCheck check_devices(const DeviceFacts& f) {
    DoctorCheck c = make("devices");
    json devices = nullptr;
    if (f.devices) {
        devices = json::array();
        for (const EpDeviceInfo& d : *f.devices) {
            devices.push_back(to_json(d));
            c.lines.push_back(describe_device(d));
        }
    }
    json selection = nullptr;
    if (f.selection) {
        const NpuSelection& s = *f.selection;
        selection = {{"status", s.ok() ? "selected" : "none"},
                     {"method", "by_device_type"},
                     {"candidate_count", s.candidate_count},
                     {"device_index", s.index ? json(*s.index) : json(nullptr)},
                     {"failure", s.failure ? json(to_string(*s.failure)) : json(nullptr)},
                     {"message", s.ok() ? json(nullptr) : json(s.message)}};
    }
    c.details = {{"ep_devices", devices}, {"npu_selection", selection}};
    if (f.enumeration_error) {
        fail(c, *f.enumeration_error, "enumerating the EP devices");
        return c;
    }
    const std::size_t n = f.devices ? f.devices->size() : 0;
    if (!f.qnn_registered || !f.selection) {
        c.summary = std::to_string(n) + " EP device(s)" +
                    (f.qnn_in_build ? "; no NPU selection: the QNN EP is not registered"
                                    : "; CPU only (this build has no QNN execution provider)");
        return c;
    }
    const NpuSelection& s = *f.selection;
    if (s.ok()) {
        c.summary = std::to_string(n) +
                    " EP devices; NPU for runs: " + describe_device((*f.devices)[*s.index]);
        return c;
    }
    if (s.failure == NpuSelectionFailure::multiple_npu_devices) {
        fail(c, std::string(to_string(*s.failure)), s.message, ExitCode::ort_runtime_error,
             std::string(hint(*s.failure)));
        return c;
    }
    warn(c, std::string(to_string(*s.failure)),
         "no NPU for runs (CPU runs still work): " + s.reason, std::string(hint(*s.failure)));
    return c;
}

void compare_probe_output(ProbeOutcome& o, std::span<const std::byte> actual,
                          std::span<const std::byte> expected) {
    o.compared = true;
    if (actual.size() != expected.size()) {
        o.elements = expected.size() / sizeof(float);
        o.differing = o.elements;
        o.max_abs_diff.reset();
        return;
    }
    o.elements = actual.size() / sizeof(float);
    o.differing = 0;
    double max_diff = 0;
    for (std::size_t i = 0; i < o.elements; ++i) {
        float a, e;
        std::memcpy(&a, actual.data() + i * sizeof(float), sizeof(float));
        std::memcpy(&e, expected.data() + i * sizeof(float), sizeof(float));
        if (std::memcmp(&a, &e, sizeof(float)) != 0) {
            ++o.differing;
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(a) - e));
        }
    }
    o.max_abs_diff = max_diff;
}

DoctorCheck check_npu_probe(const ProbeFacts& f) {
    DoctorCheck c = make("npu_probe");
    json probes = json::array();
    for (const ProbeOutcome& p : f.probes) {
        probes.push_back({{"model", p.model},
                          {"compute_precision", p.compute},
                          {"strict", true},
                          {"session_create_ms",
                           p.session_create_ms ? json(*p.session_create_ms) : json(nullptr)},
                          {"elements", p.elements},
                          {"differing_elements", p.compared ? json(p.differing) : json(nullptr)},
                          {"max_abs_diff", p.max_abs_diff ? json(*p.max_abs_diff) : json(nullptr)},
                          {"equal", p.compared ? json(p.differing == 0) : json(nullptr)},
                          {"error", error_json(p.error)}});
    }
    c.details = {{"probes", probes}, {"modules_outside_bundle", modules_json(f.outside)}};
    if (f.not_applicable) {
        not_applicable(c, *f.not_applicable);
        return c;
    }
    for (const ProbeOutcome& p : f.probes) {
        if (p.error) {
            fail(c, *p.error, p.model + " (" + p.compute + ", strict)");
            return c;
        }
    }
    if (!f.outside.empty()) {
        fail(c, "module_outside_bundle",
             "modules from outside the bundle were loaded with a probe session: " +
                 describe_modules(f.outside),
             ExitCode::ort_load_failure,
             std::string(load_error_hint(OrtLoadError::foreign_module_loaded)));
        return c;
    }
    std::vector<std::string> parts;
    for (const ProbeOutcome& p : f.probes) {
        if (p.differing > 0 || !p.compared) {
            fail(c, "npu_probe_mismatch",
                 p.model + " (" + p.compute + "): " + std::to_string(p.differing) + " of " +
                     std::to_string(p.elements) +
                     " output values differ from the expected NPU output (max abs diff " +
                     (p.max_abs_diff ? fmt(*p.max_abs_diff, 6) : std::string("n/a")) + ")",
                 ExitCode::internal_error,
                 "The NPU computes something else than the reference: check the QNN files and "
                 "the NPU driver.");
            return c;
        }
        parts.push_back(p.model + " (" + p.compute + ")");
        c.lines.push_back(p.model + ": " + std::to_string(p.elements) +
                          " values bit-identical, session " +
                          (p.session_create_ms ? fmt(*p.session_create_ms, 0) + " ms" : "?"));
    }
    c.summary = "strict NPU sessions, outputs bit-identical to the reference: " + join(parts) +
                "; every module from the bundle";
    return c;
}

DoctorCheck check_npu_context(const ContextFacts& f) {
    DoctorCheck c = make("npu_context");
    c.details = {{"build_ms", f.build_ms ? json(*f.build_ms) : json(nullptr)},
                 {"load_ms", f.load_ms ? json(*f.load_ms) : json(nullptr)},
                 {"outputs_equal", f.outputs_equal ? json(*f.outputs_equal) : json(nullptr)},
                 {"compatibility_info", opt(f.compatibility_info)},
                 {"htp_arch", opt(f.htp_arch)},
                 {"verdict", opt(f.verdict)},
                 {"verdict_note", f.verdict_note.empty() ? json(nullptr) : json(f.verdict_note)},
                 {"error", error_json(f.error)},
                 {"cleanup_error", opt(f.cleanup_error)}};
    if (f.not_applicable) {
        not_applicable(c, *f.not_applicable);
        return c;
    }
    if (f.error) {
        fail(c, *f.error, "EP context of the INT8 probe");
        return c;
    }
    if (f.outputs_equal != true) {
        fail(c, "ep_context_outputs_differ",
             "a session from the EP context gave other outputs than the compiled session",
             ExitCode::internal_error, "Do not use --ep-cache on this machine until this passes.");
        return c;
    }
    c.summary = "built in " + (f.build_ms ? fmt(*f.build_ms, 0) : std::string("?")) +
                " ms, loaded in " + (f.load_ms ? fmt(*f.load_ms, 0) : std::string("?")) +
                " ms, outputs equal; HTP arch " + f.htp_arch.value_or("(not reported)");
    if (f.compatibility_info)
        c.lines.push_back("compatibility info " + *f.compatibility_info);
    c.lines.push_back("verdict " + f.verdict.value_or("(not rated)") +
                      (f.verdict_note.empty() ? std::string() : ": " + f.verdict_note));
    if (!f.htp_arch) {
        warn(c, "htp_arch_unavailable",
             "the context's compatibility info names no HTP architecture",
             "NPU results will record the HTP architecture as unavailable.");
    }
    if (f.cleanup_error) {
        warn(c, "temporary_directory_left",
             "the temporary directory could not be removed: " + *f.cleanup_error);
    }
    return c;
}

DoctorCheck check_ep_cache_dir(const CacheDirFacts& f) {
    DoctorCheck c = make("ep_cache_dir");
    json entries = json::array();
    std::size_t stale = 0, pruned = 0;
    for (const CacheEntryFacts& e : f.entries) {
        entries.push_back({{"key_id", e.key_id},
                           {"model_sha256", opt(e.model_sha256)},
                           {"context_model", opt(e.context_model)},
                           {"size_bytes", e.size_bytes},
                           {"stale_fields", e.stale_fields},
                           {"pruned", e.pruned}});
        if (!e.stale_fields.empty())
            ++stale;
        if (e.pruned)
            ++pruned;
    }
    json foreign = json::array();
    for (const std::filesystem::path& p : f.foreign)
        foreign.push_back(path_to_utf8(p));
    c.details = {{"path", path_to_utf8(f.dir)},
                 {"exists", f.exists},
                 {"writable", f.writable ? json(*f.writable) : json(nullptr)},
                 {"entry_count", f.entries.size()},
                 {"stale_count", stale},
                 {"pruned_count", pruned},
                 {"total_bytes", f.total_bytes},
                 {"environment_known", f.environment_known},
                 {"entries", entries},
                 {"directories_without_manifest", foreign},
                 {"prune_requested", f.prune_requested},
                 {"error", error_json(f.dir_error ? f.dir_error : f.prune_error)}};
    if (f.dir_error) {
        fail(c, *f.dir_error, "the EP context cache directory " + path_to_utf8(f.dir),
             "Pass a writable directory with --ep-cache-dir.");
        return c;
    }
    if (!f.exists) {
        c.summary = path_to_utf8(f.dir) +
                    " does not exist yet (the first --ep-cache run creates "
                    "it)";
        return c;
    }
    c.summary = path_to_utf8(f.dir) + ": " + std::to_string(f.entries.size()) + " entries, " +
                mib(f.total_bytes);
    for (const CacheEntryFacts& e : f.entries) {
        c.lines.push_back(e.key_id.substr(0, 12) + "  " + e.context_model.value_or("?") + "  " +
                          mib(e.size_bytes) +
                          (e.stale_fields.empty()
                               ? std::string()
                               : (e.pruned ? "  STALE (" : "  stale (") + join(e.stale_fields) +
                                     (e.pruned ? "), deleted" : ")")));
    }
    if (f.prune_error) {
        fail(c, *f.prune_error, "deleting stale EP context cache entries");
        return c;
    }
    if (!f.environment_known && !f.entries.empty())
        c.lines.push_back("stale entries cannot be told apart: the NPU environment is unknown");
    if (stale > pruned) {
        warn(c, "stale_cache_entries",
             std::to_string(stale - pruned) + " stale entr" + (stale - pruned == 1 ? "y" : "ies") +
                 " (built with other versions or another device; never used again)",
             "Delete them with edgebench doctor --prune-cache.");
    }
    if (!f.foreign.empty()) {
        warn(c, "cache_directories_without_manifest",
             std::to_string(f.foreign.size()) + " director" +
                 (f.foreign.size() == 1 ? "y" : "ies") +
                 " without an edgebench manifest (never touched by edgebench)",
             "Remove them yourself if you do not need them.");
    }
    return c;
}

DoctorCheck check_power(const std::optional<PowerState>& power) {
    DoctorCheck c = make("power");
    const PowerState p = power.value_or(PowerState{});
    c.details = {{"source", opt(p.source)},
                 {"battery_percent", p.battery_percent ? json(*p.battery_percent) : json(nullptr)},
                 {"battery_saver", p.battery_saver ? json(*p.battery_saver) : json(nullptr)},
                 {"power_plan", opt(p.power_plan)},
                 {"power_mode", opt(p.power_mode)},
                 {"power_mode_known", p.power_mode_known}};
    const std::string note =
        "Published latency runs require AC power and the Windows power mode Best performance.";
    c.summary = "on " + p.source.value_or("unknown power") + ", power mode " +
                p.power_mode.value_or("(unknown)");
    if (p.source != "ac") {
        warn(c, p.source == "battery" ? "battery_power" : "power_source_unknown",
             p.source == "battery" ? "running on battery, which can throttle the CPU and NPU"
                                   : "the power source could not be determined",
             "Plug in. " + note);
    }
    if (p.battery_saver.value_or(false))
        warn(c, "battery_saver_on", "battery saver is on", "Turn battery saver off. " + note);
    if (p.power_mode != "Best performance") {
        warn(c, "power_mode_not_best_performance",
             "power mode is " + p.power_mode.value_or("unknown") + ", not Best performance",
             "Set the power mode to Best performance (Settings > System > Power). " + note);
    }
    return c;
}

DoctorCheck check_background_load(const std::optional<LoadSample>& load) {
    DoctorCheck c = make("background_load");
    if (!load) {
        not_applicable(c, "no load sample (--load-sample-ms 0)");
        return c;
    }
    c.details = {{"cpu_utilization_ratio", load->cpu_utilization_ratio},
                 {"sample_window_s", load->sample_window_s},
                 {"threshold_ratio", kDoctorLoadThreshold}};
    const std::string pct = fmt(load->cpu_utilization_ratio * 100.0, 1) + "%";
    c.summary = "system-wide CPU load " + pct + " over " + fmt(load->sample_window_s, 1) + " s";
    if (load->cpu_utilization_ratio > kDoctorLoadThreshold) {
        warn(c, "background_cpu_load_high",
             "background CPU load is " + pct + " (over 10%), which distorts timings",
             "Close background apps before benchmarking.");
    }
    return c;
}

CheckStatus overall_status(const std::vector<DoctorCheck>& checks) {
    bool warned = false;
    for (const DoctorCheck& c : checks) {
        if (c.status == CheckStatus::fail)
            return CheckStatus::fail;
        warned |= c.status == CheckStatus::warn;
    }
    return warned ? CheckStatus::warn : CheckStatus::pass;
}

ExitCode overall_exit_code(const std::vector<DoctorCheck>& checks) {
    for (const DoctorCheck& c : checks) {
        if (c.status == CheckStatus::fail)
            return c.exit_code == ExitCode::success ? ExitCode::internal_error : c.exit_code;
    }
    return ExitCode::success;
}

}  // namespace edgebench
