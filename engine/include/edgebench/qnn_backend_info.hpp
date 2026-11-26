// What the QNN execution provider reports about its backend and the NPU at run
// time (results schema: runtime.qnn.sdk_version, backend_build, htp_arch,
// soc_model and value_sources; config.ep_context_cache.compatibility).
//
// Two sources, both produced by the QNN EP itself, never derived from file names
// or package versions:
//
// - The backend build id. While it sets up its HTP backend, the QNN EP logs (at
//   verbose severity) "Backend build version: v2.52.0.261104115025", the string
//   QnnBackend_getBuildId() returned. The EP sets up a backend without a session
//   when ONNX Runtime asks it to rate a compiled model's compatibility info
//   (GetModelCompatibilityForEpDevices), so the provider asks it about
//   kQnnBackendProbeInfo, a well-formed string with every field 0, and reads the
//   log of that call. The SDK version is the build id's major.minor.patch, as the
//   EP itself derives it.
//
// - The compatibility info of an EP context model. When the QNN EP compiles a
//   context it stores "v2:<backend id>:<sdk>:<backend api>:<htp archs>:<soc
//   models>:<vtcm MBs>:<user driver>" in the model's metadata
//   (ep_compatibility_info.QNNExecutionProvider). On ARM64 the HTP arch comes from
//   the backend's device platform info. The SoC model is only the EP's soc_model
//   option (0 when not set): the QNN EP does not read the SoC from the device.
//
// Pure text processing; no ONNX Runtime calls.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace edgebench {

/// The compatibility info the provider asks the QNN EP to rate so that it sets up
/// its backend: version 2 with every field 0 (backend id 0 matches no backend, so
/// the EP rates it unsupported). It carries no version or device value.
inline constexpr std::string_view kQnnBackendProbeInfo = "v2:0:0.0.0:0.0.0:0:0:0:0";

/// value_sources entry for the backend build id and the SDK version derived from it.
inline constexpr std::string_view kSourceBackendBuildIdLog = "qnn_ep_log_backend_build_id";
/// value_sources entry for values read from an EP context model's compatibility info.
inline constexpr std::string_view kSourceEpCompatibilityInfo = "ep_compatibility_info";

/// The build id from a "Backend build version: <id>" message, e.g.
/// "v2.52.0.261104115025"; the last one if there are several. Null if none.
std::optional<std::string> find_backend_build_id(const std::vector<std::string>& messages);

/// "2.52.0" for "v2.52.0.261104115025": the QNN EP's own reading of the build id
/// (vMajor.Minor.Patch.BuildId, all numeric). Null for anything else.
std::optional<std::string> sdk_version_from_build_id(std::string_view build_id);

/// The QNN EP's compatibility info, version 2 (the only version it writes).
struct QnnCompatibilityInfo {
    std::uint32_t backend_id = 0;     // 6: HTP
    std::string sdk_version;          // "2.52.0"
    std::string backend_api_version;  // "5.52.0"
    std::vector<std::uint32_t> htp_archs;
    std::vector<std::uint32_t> soc_models;  // the soc_model option; 0: not set
    std::vector<std::uint32_t> vtcm_mbs;    // the vtcm_mb option; 0: not set
    bool htp_user_driver = false;
    bool operator==(const QnnCompatibilityInfo&) const = default;
};

/// Parses version-2 compatibility info; null if it is not exactly that format
/// (8 ':'-separated fields, versions major.minor.patch, equal-length lists).
std::optional<QnnCompatibilityInfo> parse_qnn_compatibility_info(std::string_view info);

/// "v73" for a context compiled for exactly one HTP arch 73; null if the info
/// lists none, several, or arch 0 (unknown).
std::optional<std::string> htp_arch_name(const QnnCompatibilityInfo& info);

/// Schema spelling of ONNX Runtime's OrtCompiledModelCompatibility value:
/// 0 not_applicable, 1 optimal, 2 prefer_recompilation, 3 unsupported.
std::optional<std::string> compatibility_verdict_name(int value);

/// One sentence on why the QNN EP prefers recompiling a context with `info`, if
/// the reason can be read from it: the EP rates a context optimal only if its
/// stored VTCM size equals the device's, and a stored 0 (vtcm_mb not set) never
/// does. Null if no stored VTCM size is 0.
std::optional<std::string> recompilation_reason(const QnnCompatibilityInfo& info);

}  // namespace edgebench
