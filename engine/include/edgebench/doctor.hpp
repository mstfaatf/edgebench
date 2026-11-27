// `edgebench doctor`: every check a user needs before trusting a run, each with a
// pass / warn / fail / not_applicable verdict (doctor_checks.hpp), as text or as a
// JSON document (schemas/doctor-report.v1.0.0.schema.json).
//
// The checks run in an order that respects what the QNN execution provider
// allows: everything that asks the EP about compatibility info (the backend build
// id query inside QnnProvider creation) happens before the first QNN session in
// the process (see qnn_provider.hpp); the probe sessions come after that.
#pragma once

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "edgebench/doctor_checks.hpp"
#include "edgebench/exit_codes.hpp"
#include "edgebench/system_info.hpp"
#include "edgebench/version.hpp"

namespace edgebench {

/// Version of the doctor report document (schemas/doctor-report.v<version>.schema.json).
inline constexpr std::string_view kDoctorSchemaVersion = "1.0.0";

struct DoctorConfig {
    std::chrono::milliseconds load_sample{1000};        // 0: no background load sample
    std::optional<std::filesystem::path> ep_cache_dir;  // default: default_ep_cache_dir()
    bool prune_cache = false;  // delete stale cache entries (manifest-only deletion)
    /// Skip what needs the NPU backend (backend_versions, npu_probe, npu_context:
    /// reported not_applicable): about 1 s instead of about 8 s on an NPU machine.
    bool quick = false;
};

/// The summary of a check that --quick skipped.
inline constexpr std::string_view kSkippedByQuick = "skipped (--quick)";

/// Versions the report names at the top: pinned (cmake/bundle-pins.json) and as
/// found at run time (null if not found).
struct DoctorVersions {
    std::string onnxruntime_pinned;
    std::optional<std::string> onnxruntime_loaded;
    std::string qnn_ep_pinned;
    std::optional<std::string> qnn_ep_reported;  // EP device metadata
    std::string qairt_sdk_pinned;
    std::optional<std::string> qairt_sdk_reported;  // backend build id
    bool qnn_in_build = false;
};

struct DoctorReport {
    BuildInfo build;
    SystemInfo system;
    std::filesystem::path bundle_dir;  // directory of the executable
    DoctorVersions versions;
    std::vector<DoctorCheck> checks;  // in kDoctorCheckNames order
    CheckStatus status = CheckStatus::pass;
    ExitCode exit_code = ExitCode::success;
    double duration_ms = 0;
};

/// Runs every check (doctor_checks.hpp describes each one).
DoctorReport run_doctor(const DoctorConfig& config);

/// Human-readable report: one line per check, then its details and hints.
std::string format_text(const DoctorReport& report);

/// The doctor report document.
nlohmann::ordered_json to_json(const DoctorReport& report);

}  // namespace edgebench
