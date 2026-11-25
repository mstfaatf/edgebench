#include "edgebench/doctor.hpp"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <sstream>
#include <system_error>

#include "edgebench/bundle_modules.hpp"
#include "edgebench/diagnostics.hpp"
#include "edgebench/qnn_ep.hpp"
#include "edgebench/runtime.hpp"

namespace edgebench {

namespace {

std::string or_unknown(const std::optional<std::string>& v) {
    return v ? *v : "(unknown)";
}

template <typename T>
std::string or_unknown(const std::optional<T>& v) {
    return v ? std::to_string(*v) : "(unknown)";
}

std::string hex(std::uint32_t v) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%x", v);
    return buf;
}

std::string join(const std::map<std::string, std::string>& kv) {
    std::string s;
    for (const auto& [k, v] : kv)
        s += (s.empty() ? "" : ", ") + k + "=" + v;
    return s;
}

std::string gib(std::uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GiB",
                  static_cast<double>(bytes) / (1024.0 * 1024 * 1024));
    return buf;
}

void row(std::ostringstream& out, const char* label, const std::string& value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "  %-18s ", label);
    out << buf << value << "\n";
}

void list_modules(DoctorReport& r) {
    const BundleModuleCheck check = check_bundle_modules(r.bundle_dir);
    for (const LoadedModule& m : check.modules) {
        bool inside = true;
        for (const LoadedModule& o : check.outside)
            inside &= o.path != m.path;
        r.modules.push_back({m, inside});
        if (!inside) {
            r.problems.push_back(
                {"module_outside_bundle",
                 m.name + " is loaded from " + path_to_utf8(m.path) +
                     ", outside the bundle directory " + path_to_utf8(r.bundle_dir),
                 std::string(load_error_hint(OrtLoadError::foreign_module_loaded))});
        }
    }
}

// Registers the bundled QNN EP (when the build has one), enumerates every EP
// device, and selects the NPU the way NPU runs do.
void check_qnn_and_devices(DoctorReport& r, Runtime& runtime) {
    QnnEpCheck& q = r.qnn;
    if (!q.bundled && !qnn_ep_in_build()) {
        q.registration = QnnEpCheck::Registration::not_bundled;
    } else if (Result<EpLibrary*> registered = register_qnn_ep(runtime); registered) {
        q.registration = QnnEpCheck::Registration::ok;
    } else {
        const Error& e = registered.error();
        q.registration = QnnEpCheck::Registration::failed;
        q.error_kind = e.kind;
        q.error_message = e.message;
        r.problems.push_back({e.kind, e.message, std::string(qnn_ep_hint(e.kind)), e.exit_code});
    }

    try {
        r.ep_devices = runtime.ep_devices();
    } catch (const Ort::Exception& e) {
        r.problems.push_back({"ep_enumeration_failed", e.what(), "", ExitCode::ort_runtime_error});
        return;
    }
    if (q.registration != QnnEpCheck::Registration::ok)
        return;
    for (const EpDeviceInfo& d : *r.ep_devices) {
        if (d.ep_name == kQnnEpName && d.ep_metadata.contains("version")) {
            q.ep_version = d.ep_metadata.at("version");
            break;
        }
    }
    q.npu = select_qnn_npu(*r.ep_devices);
    if (q.npu->ok()) {
        q.npu_device = (*r.ep_devices)[*q.npu->index];
    } else {
        // CPU runs still work, so this is not a failed check.
        const NpuSelectionFailure f = *q.npu->failure;
        r.warnings.push_back({std::string(to_string(f)),
                              "no device would be selected for NPU runs: " + q.npu->reason,
                              std::string(hint(f))});
    }
}

}  // namespace

std::string_view to_string(QnnEpCheck::Registration registration) {
    switch (registration) {
        case QnnEpCheck::Registration::not_attempted:
            return "not_attempted";
        case QnnEpCheck::Registration::not_bundled:
            return "not_bundled";
        case QnnEpCheck::Registration::ok:
            return "ok";
        case QnnEpCheck::Registration::failed:
            return "failed";
    }
    return "unknown";
}

std::vector<DoctorWarning> system_state_warnings(const SystemInfo& system) {
    std::vector<DoctorWarning> warnings;
    const PowerState power = system.power.value_or(PowerState{});
    if (power.source == "battery") {
        warnings.push_back({"battery_power",
                            "running on battery power, which can throttle the CPU and NPU",
                            "Plug in before benchmarking."});
    }
    if (power.battery_saver.value_or(false)) {
        warnings.push_back({"battery_saver_on", "battery saver is on, which limits performance",
                            "Turn battery saver off before benchmarking."});
    }
    if (system.load && system.load->cpu_utilization_ratio > 0.10) {
        char pct[16];
        std::snprintf(pct, sizeof(pct), "%.1f%%", system.load->cpu_utilization_ratio * 100.0);
        warnings.push_back({"background_cpu_load_high",
                            std::string("background CPU load is ") + pct +
                                " (over 10%), which distorts CPU timings",
                            "Close background apps before benchmarking."});
    }
    if (power.power_mode && !power.power_mode_known) {
        warnings.push_back(
            {"unknown_power_mode",
             "the power mode overlay " + *power.power_mode + " is not a known Windows power mode",
             ""});
    }
    if (system.emulated()) {
        warnings.push_back(
            {"process_emulated",
             "this is a " + std::string(to_string(system.process_architecture)) +
                 " build running under emulation on " +
                 std::string(to_string(system.os.architecture)) +
                 " Windows; timings would not represent native performance",
             "Use the native " + std::string(to_string(system.os.architecture)) + " build."});
    }
    return warnings;
}

DoctorReport run_doctor(std::chrono::milliseconds load_sample) {
    DoctorReport r;
    r.build = build_info();
    r.system = query_system_info();
    r.system.power = query_power_state();
    if (load_sample.count() > 0)
        r.system.load = sample_cpu_load(load_sample);
    r.warnings = system_state_warnings(r.system);
    r.bundle_dir = executable_dir();

    r.onnxruntime = load_onnxruntime(r.bundle_dir);
    r.qnn.library_path = qnn_ep_library_path(r.bundle_dir);
    std::error_code ec;
    r.qnn.bundled = std::filesystem::is_regular_file(r.qnn.library_path, ec);
    if (!r.onnxruntime.ok()) {
        r.problems.push_back({std::string(to_string(r.onnxruntime.error_kind)),
                              r.onnxruntime.message,
                              std::string(load_error_hint(r.onnxruntime.error_kind))});
        list_modules(r);
    } else {
        Result<std::unique_ptr<Runtime>> runtime = Runtime::create(r.onnxruntime);
        if (!runtime) {
            const Error& e = runtime.error();
            r.problems.push_back({e.kind, e.message, "", e.exit_code});
        } else {
            check_qnn_and_devices(r, *runtime.value());
        }
        // While the runtime (and any EP library it registered) is alive, so every
        // module ONNX Runtime pulled in is listed.
        list_modules(r);
    }

    // Load failures (exit code 4) take precedence over runtime errors (3).
    r.exit_code = ExitCode::success;
    for (const DoctorProblem& p : r.problems) {
        if (p.exit_code == ExitCode::ort_load_failure || r.exit_code == ExitCode::success)
            r.exit_code = p.exit_code;
    }
    return r;
}

std::string format_text(const DoctorReport& r) {
    std::ostringstream out;
    const SystemInfo& s = r.system;
    out << "Build\n";
    row(out, "edgebench", std::string(version()));
    row(out, "git commit",
        r.build.git_commit
            ? *r.build.git_commit +
                  (r.build.git_dirty ? (*r.build.git_dirty ? " (dirty)" : " (clean)") : "")
            : "(unknown)");
    row(out, "build type", r.build.build_type.empty() ? "(unknown)" : r.build.build_type);
    row(out, "compiler", r.build.compiler);
    row(out, "results schema", std::string(results_schema_version()));

    out << "\nSystem\n";
    row(out, "OS", s.os.name);
    row(out, "OS version", or_unknown(s.os.version) + " (build " + or_unknown(s.os.build) + ")");
    row(out, "OS architecture", std::string(to_string(s.os.architecture)));
    row(out, "process",
        std::string(to_string(s.process_architecture)) +
            (s.emulated() ? "  EMULATED" : " (native)"));
    row(out, "CPU", or_unknown(s.cpu.name));
    row(out, "cores",
        or_unknown(s.cpu.physical_cores) + " physical, " + or_unknown(s.cpu.logical_processors) +
            " logical");
    row(out, "memory",
        s.memory_total_bytes
            ? gib(*s.memory_total_bytes) + " (" + std::to_string(*s.memory_total_bytes) + " bytes)"
            : "(unknown)");
    row(out, "device", or_unknown(s.device.manufacturer) + " / " + or_unknown(s.device.model));

    out << "\nPower\n";
    if (s.power) {
        const PowerState& p = *s.power;
        row(out, "source", p.source.value_or("(unknown)"));
        row(out, "battery",
            p.battery_percent ? std::to_string(*p.battery_percent) + "%"
                              : std::string("(no battery or unknown)"));
        row(out, "battery saver",
            p.battery_saver ? (*p.battery_saver ? "on" : "off") : "(unknown)");
        row(out, "power plan",
            p.power_plan.value_or("(unknown)") +
                (p.power_plan_guid ? " (" + *p.power_plan_guid + ")" : std::string()));
        row(out, "power mode",
            p.power_mode ? *p.power_mode + (p.power_mode_known && p.power_mode_guid
                                                ? " (overlay " + *p.power_mode_guid + ")"
                                                : std::string(" (unknown overlay)"))
                         : std::string("(unknown)"));
    }
    if (s.load) {
        char load[64];
        std::snprintf(load, sizeof(load), "%.1f%% over %.1f s",
                      s.load->cpu_utilization_ratio * 100.0, s.load->sample_window_s);
        row(out, "background CPU", load);
    }

    out << "\nONNX Runtime\n";
    const OrtLoadResult& ort = r.onnxruntime;
    row(out, "required",
        std::string(expected_onnxruntime_version()) + " (C API " +
            std::to_string(expected_onnxruntime_api_version()) + ")");
    row(out, "library", path_to_utf8(ort.dll_path));
    if (ort.ok()) {
        row(out, "version", ort.ort_version + " (C API " + std::to_string(ort.api_version) + ")");
        row(out, "load check", std::string(ort.load_check()));
    } else {
        row(out, "load check", "FAILED (" + std::string(to_string(ort.error_kind)) + ")");
        if (!ort.ort_version.empty())
            row(out, "found version", ort.ort_version);
    }

    out << "\nQNN execution provider\n";
    const QnnEpCheck& q = r.qnn;
    row(out, "library",
        path_to_utf8(q.library_path) + (q.bundled ? std::string() : std::string("  (missing)")));
    switch (q.registration) {
        case QnnEpCheck::Registration::not_attempted:
            row(out, "registration", "(not attempted: ONNX Runtime is not loaded)");
            break;
        case QnnEpCheck::Registration::not_bundled:
            row(out, "registration", "not bundled (CPU-only build)");
            break;
        case QnnEpCheck::Registration::ok:
            row(out, "registration", "ok (as " + std::string(kQnnEpName) + ")");
            break;
        case QnnEpCheck::Registration::failed:
            row(out, "registration", "FAILED (" + q.error_kind.value_or("unknown") + ")");
            break;
    }
    if (q.ep_version)
        row(out, "version", *q.ep_version);
    if (q.npu_device) {
        row(out, "NPU for runs",
            describe_device(*q.npu_device) + ", " + std::to_string(q.npu->candidate_count) +
                " candidate");
    } else if (q.npu) {
        row(out, "NPU for runs",
            "none (" + std::string(to_string(*q.npu->failure)) + "): " + q.npu->reason);
    } else {
        row(out, "NPU for runs", "none (the QNN execution provider is not registered)");
    }

    out << "\nLoaded ONNX Runtime / QNN modules (bundle: " << path_to_utf8(r.bundle_dir) << ")\n";
    if (r.modules.empty())
        out << "  (none)\n";
    for (const ModuleCheck& m : r.modules) {
        out << "  " << path_to_utf8(m.module.path)
            << (m.inside_bundle ? "" : "   <-- OUTSIDE the bundle directory") << "\n";
    }

    out << "\nEP devices ("
        << (q.registration == QnnEpCheck::Registration::ok ? "CPU EP and QNN EP"
                                                           : "no plugin EP registered")
        << ")\n";
    if (!r.ep_devices) {
        out << "  (not available: ONNX Runtime is not loaded)\n";
    } else if (r.ep_devices->empty()) {
        out << "  (none)\n";
    } else {
        for (std::size_t i = 0; i < r.ep_devices->size(); ++i) {
            const EpDeviceInfo& d = (*r.ep_devices)[i];
            const bool selected = q.npu && q.npu->index == i;
            out << "  " << d.ep_name << "  type " << d.device_type << "  ep vendor "
                << (d.ep_vendor.empty() ? "(unset)" : d.ep_vendor) << "  vendor "
                << (d.vendor.empty() ? "(unset)" : d.vendor) << "  vendor_id " << hex(d.vendor_id)
                << "  device_id " << hex(d.device_id)
                << (selected ? "   <-- selected for NPU runs" : "") << "\n";
            out << "      device metadata: {" << join(d.metadata) << "}\n";
            out << "      ep metadata:     {" << join(d.ep_metadata) << "}\n";
        }
    }

    out << "\n";
    for (const DoctorWarning& w : r.warnings)
        out << format_warning(w.code, w.message, w.hint);
    for (const DoctorProblem& p : r.problems)
        out << "problem [" << p.code << "]: " << p.message << "\n";
    out << "Result: "
        << (r.exit_code == ExitCode::success
                ? std::string("OK")
                : "FAILED (exit " + std::to_string(to_int(r.exit_code)) + ", " +
                      std::string(name(r.exit_code)) + ")")
        << "\n";
    return out.str();
}

nlohmann::ordered_json to_json(const DoctorReport& r) {
    using json = nlohmann::ordered_json;
    const OrtLoadResult& ort = r.onnxruntime;
    json modules = json::array();
    for (const ModuleCheck& m : r.modules) {
        modules.push_back({{"name", m.module.name},
                           {"path", path_to_utf8(m.module.path)},
                           {"inside_bundle", m.inside_bundle}});
    }
    json devices = nullptr;
    if (r.ep_devices) {
        devices = json::array();
        for (const EpDeviceInfo& d : *r.ep_devices)
            devices.push_back(to_json(d));
    }
    json problems = json::array();
    for (const DoctorProblem& p : r.problems)
        problems.push_back({{"code", p.code}, {"message", p.message}, {"hint", p.hint}});

    const QnnEpCheck& q = r.qnn;
    json npu = nullptr;
    if (q.npu) {
        npu = {{"status", q.npu->ok() ? "selected" : "none"},
               {"selection", "by_device_type"},
               {"candidate_count", q.npu->candidate_count},
               {"device_index", q.npu->index ? json(*q.npu->index) : json(nullptr)},
               {"device", q.npu_device ? to_json(*q.npu_device) : json(nullptr)},
               {"failure", q.npu->failure ? json(to_string(*q.npu->failure)) : json(nullptr)},
               {"message", q.npu->ok() ? json(nullptr) : json(q.npu->message)}};
    }
    const json qnn_ep = {
        {"library_path", path_to_utf8(q.library_path)},
        {"in_build", qnn_ep_in_build()},
        {"bundled", q.bundled},
        {"registration", to_string(q.registration)},
        {"error_kind", q.error_kind ? json(*q.error_kind) : json(nullptr)},
        {"message", q.error_kind ? json(q.error_message) : json(nullptr)},
        {"ep_version", q.ep_version ? json(*q.ep_version) : json(nullptr)},
        {"npu_selection", npu},
    };

    return {
        {"tool",
         {{"name", "edgebench"},
          {"version", version()},
          {"git_commit", r.build.git_commit ? json(*r.build.git_commit) : json(nullptr)},
          {"git_dirty", r.build.git_dirty ? json(*r.build.git_dirty) : json(nullptr)},
          {"build_type", r.build.build_type.empty() ? json(nullptr) : json(r.build.build_type)},
          {"compiler", r.build.compiler},
          {"implementation", "cpp"}}},
        {"results_schema_version", results_schema_version()},
        {"system", to_json(r.system)},
        {"process_emulated", r.system.emulated()},
        {"bundle_dir", path_to_utf8(r.bundle_dir)},
        {"onnxruntime_load",
         {{"status", ort.ok() ? "ok" : "failed"},
          {"error_kind", ort.ok() ? json(nullptr) : json(to_string(ort.error_kind))},
          {"message", ort.ok() ? json(nullptr) : json(ort.message)},
          {"required_version", expected_onnxruntime_version()},
          {"required_api_version", expected_onnxruntime_api_version()},
          {"found_version", ort.ort_version.empty() ? json(nullptr) : json(ort.ort_version)}}},
        {"runtime",
         {{"onnxruntime", ort.ok() ? json{{"version", ort.ort_version},
                                          {"library_path", path_to_utf8(ort.dll_path)},
                                          {"api_version", ort.api_version},
                                          {"load_check", ort.load_check()}}
                                   : json(nullptr)},
          {"loaded_modules", modules},
          {"ep_devices", devices}}},
        {"qnn_ep", qnn_ep},
        {"warnings",
         [&r] {
             json warnings = json::array();
             for (const DoctorWarning& w : r.warnings)
                 warnings.push_back({{"code", w.code}, {"message", w.message}, {"hint", w.hint}});
             return warnings;
         }()},
        {"problems", problems},
        {"result", r.exit_code == ExitCode::success ? "ok" : "failed"},
        {"exit_code", to_int(r.exit_code)},
    };
}

}  // namespace edgebench
