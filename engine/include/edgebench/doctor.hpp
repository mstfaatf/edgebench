// `edgebench doctor`: checks the build, the system and the ONNX Runtime installation.
#pragma once

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "edgebench/ep_devices.hpp"
#include "edgebench/exit_codes.hpp"
#include "edgebench/npu_selection.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/system_info.hpp"
#include "edgebench/version.hpp"

namespace edgebench {

/// Something that makes the installation unusable; sets a nonzero exit code.
struct DoctorProblem {
    std::string code;  // e.g. "dll_not_found", "module_outside_bundle"
    std::string message;
    std::string hint;  // how to fix it; may be empty
    ExitCode exit_code = ExitCode::ort_load_failure;
};

/// Something that distorts benchmarks but does not fail the check.
struct DoctorWarning {
    std::string code;  // e.g. "battery_power", "background_cpu_load_high"
    std::string message;
    std::string hint;
};

struct ModuleCheck {
    LoadedModule module;
    bool inside_bundle = false;
};

/// The QNN execution provider: the bundled plugin library, its registration, and
/// the device that NPU runs would use.
struct QnnEpCheck {
    enum class Registration {
        not_attempted,  // ONNX Runtime could not be loaded
        not_bundled,    // CPU-only build: the plugin is not part of it
        ok,
        failed,  // see error_kind and error_message
    };
    std::filesystem::path library_path;
    bool bundled = false;  // the library file exists
    Registration registration = Registration::not_attempted;
    std::optional<std::string> error_kind;
    std::string error_message;
    std::optional<std::string> ep_version;  // "version" in the EP metadata of its devices
    std::optional<NpuSelection> npu;        // set when the registration succeeded
    std::optional<EpDeviceInfo> npu_device;
};

std::string_view to_string(QnnEpCheck::Registration registration);

struct DoctorReport {
    BuildInfo build;
    SystemInfo system;
    std::filesystem::path bundle_dir;  // directory of the executable
    OrtLoadResult onnxruntime;
    std::vector<ModuleCheck> modules;  // every ORT/QNN module in the process
    QnnEpCheck qnn;
    std::optional<std::vector<EpDeviceInfo>> ep_devices;  // empty if ORT is not loaded
    std::vector<DoctorProblem> problems;
    std::vector<DoctorWarning> warnings;  // noteworthy but not failures (e.g. emulation)
    ExitCode exit_code = ExitCode::success;
};

/// Warnings about the system state that distorts timings: battery power, battery
/// saver, background CPU load over 10%, an unknown power mode, emulation.
std::vector<DoctorWarning> system_state_warnings(const SystemInfo& system);

/// Runs every check. Sections that do not need ONNX Runtime are filled in even
/// when it cannot be loaded. In builds that bundle the QNN execution provider, a
/// missing or unregistrable plugin is a problem (exit code 4); no selectable NPU
/// is a warning only (CPU runs still work), coded with the selection failure kind.
DoctorReport run_doctor(std::chrono::milliseconds load_sample = std::chrono::seconds(1));

/// Human-readable report (no color codes).
std::string format_text(const DoctorReport& report);

/// Machine-readable report. `tool`, `system` and `runtime` use the results
/// schema's field names where they overlap.
nlohmann::ordered_json to_json(const DoctorReport& report);

}  // namespace edgebench
