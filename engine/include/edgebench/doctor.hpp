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
#include "edgebench/ort_loader.hpp"
#include "edgebench/system_info.hpp"
#include "edgebench/version.hpp"

namespace edgebench {

/// Something that makes the installation unusable; sets a nonzero exit code.
struct DoctorProblem {
    std::string code;  // e.g. "dll_not_found", "module_outside_bundle"
    std::string message;
};

struct ModuleCheck {
    LoadedModule module;
    bool inside_bundle = false;
};

struct DoctorReport {
    BuildInfo build;
    SystemInfo system;
    std::filesystem::path bundle_dir;  // directory of the executable
    OrtLoadResult onnxruntime;
    std::vector<ModuleCheck> modules;                     // every ORT/QNN module in the process
    std::optional<std::vector<EpDeviceInfo>> ep_devices;  // empty if ORT is not loaded
    std::vector<DoctorProblem> problems;
    std::vector<std::string> warnings;  // noteworthy but not failures (e.g. emulation)
    ExitCode exit_code = ExitCode::success;
};

/// Runs every check. Sections that do not need ONNX Runtime are filled in even
/// when it cannot be loaded.
DoctorReport run_doctor(std::chrono::milliseconds load_sample = std::chrono::seconds(1));

/// Human-readable report (no color codes).
std::string format_text(const DoctorReport& report);

/// Machine-readable report. `tool`, `system` and `runtime` use the results
/// schema's field names where they overlap.
nlohmann::ordered_json to_json(const DoctorReport& report);

}  // namespace edgebench
