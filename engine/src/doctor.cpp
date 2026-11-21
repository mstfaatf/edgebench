#include "edgebench/doctor.hpp"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <sstream>

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

}  // namespace

DoctorReport run_doctor() {
    DoctorReport r;
    r.build = build_info();
    r.system = query_system_info();
    r.system.power = query_power_state();
    r.system.load = sample_cpu_load(std::chrono::seconds(1));
    const PowerState& power = *r.system.power;
    if (power.source == "battery") {
        r.warnings.push_back(
            "running on battery power, which can throttle the CPU and NPU. "
            "Plug in before benchmarking.");
    }
    if (power.battery_saver.value_or(false)) {
        r.warnings.push_back(
            "battery saver is on, which limits performance. Turn it off "
            "before benchmarking.");
    }
    if (r.system.load->cpu_utilization_ratio > 0.10) {
        char pct[16];
        std::snprintf(pct, sizeof(pct), "%.1f%%", r.system.load->cpu_utilization_ratio * 100.0);
        r.warnings.push_back(std::string("background CPU load is ") + pct +
                             " (over 10%), which distorts CPU timings. Close background apps "
                             "before benchmarking.");
    }
    if (power.power_mode && !power.power_mode_known) {
        r.warnings.push_back("the power mode overlay " + *power.power_mode +
                             " is not a known Windows power mode.");
    }
    r.bundle_dir = executable_dir();
    if (r.system.emulated()) {
        r.warnings.push_back(
            "this is a " + std::string(to_string(r.system.process_architecture)) +
            " build running under emulation on " +
            std::string(to_string(r.system.os.architecture)) +
            " Windows; timings would not represent native performance. Use the native " +
            std::string(to_string(r.system.os.architecture)) + " build.");
    }

    r.onnxruntime = load_onnxruntime(r.bundle_dir);
    if (!r.onnxruntime.ok()) {
        r.problems.push_back(
            {std::string(to_string(r.onnxruntime.error_kind)), r.onnxruntime.message});
    } else {
        try {
            Result<std::unique_ptr<Runtime>> runtime = Runtime::create(r.onnxruntime);
            if (!runtime)
                throw std::runtime_error(runtime.error().message);
            r.ep_devices = runtime.value()->ep_devices();
        } catch (const Ort::Exception& e) {
            r.problems.push_back({"ep_enumeration_failed", e.what()});
        } catch (const std::exception& e) {
            r.problems.push_back({"ep_enumeration_failed", e.what()});
        }
    }

    // After EP enumeration, so everything ONNX Runtime pulled in is listed.
    const std::vector<LoadedModule> modules = list_ort_modules();
    const std::vector<LoadedModule> outside = find_modules_outside(modules, r.bundle_dir);
    for (const LoadedModule& m : modules) {
        bool inside = true;
        for (const LoadedModule& o : outside)
            inside &= o.path != m.path;
        r.modules.push_back({m, inside});
        if (!inside) {
            r.problems.push_back({"module_outside_bundle", m.name + " is loaded from " +
                                                               path_to_utf8(m.path) +
                                                               ", outside the bundle directory " +
                                                               path_to_utf8(r.bundle_dir)});
        }
    }

    r.exit_code = ExitCode::success;
    for (const DoctorProblem& p : r.problems) {
        if (p.code == "ep_enumeration_failed") {
            if (r.exit_code == ExitCode::success)
                r.exit_code = ExitCode::ort_runtime_error;
        } else {
            r.exit_code = ExitCode::ort_load_failure;  // loader errors and foreign modules
        }
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

    out << "\nLoaded ONNX Runtime / QNN modules (bundle: " << path_to_utf8(r.bundle_dir) << ")\n";
    if (r.modules.empty())
        out << "  (none)\n";
    for (const ModuleCheck& m : r.modules) {
        out << "  " << path_to_utf8(m.module.path)
            << (m.inside_bundle ? "" : "   <-- OUTSIDE the bundle directory") << "\n";
    }

    out << "\nEP devices (no plugin EP registered)\n";
    if (!r.ep_devices) {
        out << "  (not available: ONNX Runtime is not loaded)\n";
    } else if (r.ep_devices->empty()) {
        out << "  (none)\n";
    } else {
        for (const EpDeviceInfo& d : *r.ep_devices) {
            out << "  " << d.ep_name << "  type " << d.device_type << "  ep vendor "
                << (d.ep_vendor.empty() ? "(unset)" : d.ep_vendor) << "  vendor "
                << (d.vendor.empty() ? "(unset)" : d.vendor) << "  vendor_id " << hex(d.vendor_id)
                << "  device_id " << hex(d.device_id) << "\n";
            out << "      device metadata: {" << join(d.metadata) << "}\n";
            out << "      ep metadata:     {" << join(d.ep_metadata) << "}\n";
        }
    }

    out << "\n";
    for (const std::string& w : r.warnings)
        out << "warning: " << w << "\n";
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
        problems.push_back({{"code", p.code}, {"message", p.message}});

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
        {"warnings", r.warnings},
        {"problems", problems},
        {"result", r.exit_code == ExitCode::success ? "ok" : "failed"},
        {"exit_code", to_int(r.exit_code)},
    };
}

}  // namespace edgebench
