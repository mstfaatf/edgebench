// The checks of `edgebench doctor` as pure functions: each one turns the facts
// doctor.cpp gathered (files, versions, devices, probe outcomes, system state)
// into a pass / warn / fail / not_applicable verdict with a summary, details for
// the JSON report, and for failures the exit code `edgebench run` gives for the
// same problem. No I/O here, so every classification is unit-tested with fakes.
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/ep_devices.hpp"
#include "edgebench/exit_codes.hpp"
#include "edgebench/module_paths.hpp"
#include "edgebench/npu_selection.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/result.hpp"
#include "edgebench/system_info.hpp"
#include "edgebench/version.hpp"

namespace edgebench {

enum class CheckStatus { pass, warn, fail, not_applicable };
std::string_view to_string(CheckStatus status);

/// One check of the report (doctor report schema: checks[]).
struct DoctorCheck {
    std::string name;
    CheckStatus status = CheckStatus::pass;
    std::string summary;             // one line
    std::vector<std::string> lines;  // more lines for the text report
    nlohmann::ordered_json details = nlohmann::ordered_json::object();
    std::optional<std::string> code;         // what is wrong (warn and fail)
    std::string hint;                        // what to do about it (warn and fail)
    ExitCode exit_code = ExitCode::success;  // fail only: the code `run` gives
    double duration_ms = 0;
};

/// The check names, in report order.
inline constexpr std::string_view kDoctorCheckNames[] = {
    "ort_bundle",   "qnn_bundle", "backend_versions", "devices", "npu_probe", "npu_context",
    "ep_cache_dir", "power",      "background_load"};

/// A file the bundle must contain, as pinned in cmake/bundle-pins.json.
struct FilePin {
    std::string name;
    std::uint64_t size_bytes = 0;
    std::string sha256;
};

/// What doctor found for one pinned file in the bundle directory.
struct BundleFile {
    std::string name;
    std::filesystem::path path;  // long form
    bool present = false;
    std::optional<std::uint64_t> size_bytes;
    std::optional<std::string> sha256;
    std::optional<std::string> product_version;  // version resource
    std::optional<std::string> file_version;
};

/// Files of `pins` that are missing or differ from their pin, as "name: reason".
std::vector<std::string> bundle_file_problems(const std::vector<BundleFile>& files,
                                              const std::vector<FilePin>& pins);

/// ort_bundle: the loaded onnxruntime.dll and the pinned ONNX Runtime files, and
/// every ONNX Runtime / QNN module in the process coming from the bundle. A load
/// failure, a file that does not match its pin and a module from outside the
/// bundle fail with exit code 4; emulation is a warning.
DoctorCheck check_ort_bundle(const OrtLoadResult& load, const std::vector<BundleFile>& files,
                             const std::vector<FilePin>& pins,
                             const std::vector<LoadedModule>& modules,
                             const std::vector<LoadedModule>& outside, bool emulated);

/// The QNN execution provider registration, as tried by doctor.
struct QnnRegistration {
    bool attempted = false;
    std::optional<Error> error;  // set if it failed
};

/// qnn_bundle: the QNN EP plugin and backend files against their pins, and the
/// registration of the plugin. not_applicable in builds without the QNN EP
/// (no QNN pins: x64). A missing or mismatched file or a failed registration
/// fails with exit code 4.
DoctorCheck check_qnn_bundle(const std::vector<BundleFile>& files, const std::vector<FilePin>& pins,
                             const QnnRegistration& registration);

/// What the QNN EP reported about its backend before any session.
struct BackendVersionFacts {
    std::optional<std::string> not_applicable;                // reason, e.g. no NPU device
    std::optional<std::string> backend_build;                 // "v2.52.0.261104115025"
    std::optional<std::string> sdk_version;                   // "2.52.0"
    std::optional<std::string> source;                        // value source
    std::string pinned_sdk_version;                           // what the QNN EP release declares
    std::optional<std::string> backend_file_product_version;  // QnnHtp.dll ProductVersion
};

/// backend_versions: the backend build id and SDK version from the session-free
/// query, and whether they agree with the release's SDK version and the backend
/// file's version. Never fails: a value that is not reported or a disagreement
/// warns.
DoctorCheck check_backend_versions(const BackendVersionFacts& facts);

struct DeviceFacts {
    std::optional<std::vector<EpDeviceInfo>> devices;  // null: not enumerated
    std::optional<Error> enumeration_error;
    bool qnn_registered = false;
    bool qnn_in_build = false;
    std::optional<NpuSelection> selection;  // set when the QNN EP is registered
};

/// devices: every EP device and the NPU an NPU run would select. Enumeration
/// failures and more than one NPU fail (exit code 3, as for `run`); no NPU warns
/// (CPU runs still work).
DoctorCheck check_devices(const DeviceFacts& facts);

/// One probe model run once on the NPU.
struct ProbeOutcome {
    std::string model;    // file name
    std::string compute;  // "int8" or "fp16"
    std::optional<Error> error;
    std::optional<double> session_create_ms;
    std::size_t elements = 0;
    std::size_t differing = 0;  // float elements whose bytes differ from the expected output
    std::optional<double> max_abs_diff;
    bool compared = false;
};

/// Compares float32 output bytes with the expected bytes (same length): fills
/// elements, differing, max_abs_diff and compared.
void compare_probe_output(ProbeOutcome& outcome, std::span<const std::byte> actual,
                          std::span<const std::byte> expected);

struct ProbeFacts {
    std::optional<std::string> not_applicable;  // reason
    std::vector<ProbeOutcome> probes;
    std::vector<LoadedModule>
        outside;  // modules from outside the bundle while a probe session existed
};

/// npu_probe: the INT8 and FP16 probe models in strict mode, outputs byte for
/// byte equal to the committed expected outputs. A session failure keeps its
/// exit code (backend 4, strict refusal 5, other ONNX Runtime errors 3); a
/// mismatch is exit code 1; a module from outside the bundle 4.
DoctorCheck check_npu_probe(const ProbeFacts& facts);

struct ContextFacts {
    std::optional<std::string> not_applicable;  // reason
    std::optional<Error> error;
    std::optional<double> build_ms;
    std::optional<double> load_ms;
    std::optional<bool> outputs_equal;  // context session vs compiled session
    std::optional<std::string> compatibility_info;
    std::optional<std::string> htp_arch;
    std::optional<std::string> verdict;
    std::string verdict_note;  // why the verdict is null, if it is
    std::optional<std::string> cleanup_error;
};

/// npu_context: an EP context of the INT8 probe built in a temporary directory,
/// loaded, and its outputs compared with the compiled session's. A failure keeps
/// its exit code (a context that cannot be loaded: 3); different outputs are
/// exit code 1 (as ep_context_outputs_differ). No HTP arch in the compatibility
/// info, or a temporary directory that could not be removed, warns.
DoctorCheck check_npu_context(const ContextFacts& facts);

/// The fields of an EP context cache key that describe the environment (as
/// opposed to the model and its options): a cache entry whose values differ from
/// the current ones can never be used again.
struct CurrentCacheEnvironment {
    std::optional<std::string> ort_version;
    std::optional<std::string> ep_version;
    std::optional<std::string> qnn_sdk_version;
    std::optional<std::string> qnn_backend_file_version;
    std::optional<nlohmann::json> device;  // {"device_type", "vendor_id", "device_id"}
};

/// The environment key fields in which `key_fields` (a cache manifest's key)
/// differs from `current`, e.g. {"qnn_sdk_version"}. Fields unknown in `current`
/// are not compared.
std::vector<std::string> stale_key_fields(const nlohmann::json& key_fields,
                                          const CurrentCacheEnvironment& current);

struct CacheEntryFacts {
    std::filesystem::path dir;
    std::string key_id;
    std::optional<std::string> model_sha256;
    std::optional<std::string> context_model;
    std::uint64_t size_bytes = 0;
    std::vector<std::string> stale_fields;  // stale_key_fields()
    bool pruned = false;
};

struct CacheDirFacts {
    std::filesystem::path dir;
    std::optional<Error> dir_error;  // the directory cannot be determined or written
    bool exists = false;
    std::optional<bool> writable;
    bool environment_known = false;  // CurrentCacheEnvironment could be read
    std::vector<CacheEntryFacts> entries;
    std::vector<std::filesystem::path> foreign;  // directories without a manifest
    std::uint64_t total_bytes = 0;
    bool prune_requested = false;
    std::optional<Error> prune_error;
};

/// ep_cache_dir: the EP context cache directory: exists (a missing one is fine:
/// the first --ep-cache run creates it), writable (else exit code 6), size, entries,
/// stale entries (warn, unless --prune-cache deleted them) and directories without
/// a manifest (warn: never touched).
DoctorCheck check_ep_cache_dir(const CacheDirFacts& facts);

/// power: AC power and the Windows power mode "Best performance" (with battery
/// saver off) are what published latency runs require; anything else warns.
DoctorCheck check_power(const std::optional<PowerState>& power);

/// The background load above which doctor (and a run) warns.
inline constexpr double kDoctorLoadThreshold = 0.10;

/// background_load: a short system-wide CPU load sample. Above
/// kDoctorLoadThreshold warns; no sample (--load-sample-ms 0) is not_applicable.
DoctorCheck check_background_load(const std::optional<LoadSample>& load);

/// fail if any check failed, else warn if any warned, else pass.
CheckStatus overall_status(const std::vector<DoctorCheck>& checks);

/// The exit code of the first failing check in report order (the failure a run
/// would hit first), or success.
ExitCode overall_exit_code(const std::vector<DoctorCheck>& checks);

}  // namespace edgebench
