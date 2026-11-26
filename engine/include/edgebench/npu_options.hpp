// The settings of an NPU run that do not need ONNX Runtime: the QNN EP provider
// options and session config entries edgebench passes, the accepted HTP
// performance modes, and the rule that decides the compute precision.
//
// Everything passed to ONNX Runtime comes from these functions and is recorded
// in the result exactly as passed (config.ep_options, config.session_config);
// every other QNN EP option is left at the EP's default.
#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include "edgebench/onnx_inspect.hpp"
#include "edgebench/result.hpp"
#include "edgebench/run_config.hpp"

namespace edgebench {

/// The values the QNN EP accepts for its htp_performance_mode provider option.
inline constexpr std::array<std::string_view, 10> kHtpPerformanceModes = {
    "burst",
    "balanced",
    "default",
    "high_performance",
    "high_power_saver",
    "low_balanced",
    "low_power_saver",
    "power_saver",
    "extreme_power_saver",
    "sustained_high_performance",
};
inline constexpr std::string_view kDefaultHtpPerformanceMode = "burst";

bool is_htp_performance_mode(std::string_view mode);

/// Comma-separated list of kHtpPerformanceModes, for help and error messages.
std::string htp_performance_modes_list();

/// --htp-performance-mode, or the default (burst).
std::string effective_htp_performance_mode(const RunConfig& config);

/// QNN EP provider options for `config`: backend_path (absolute path of the HTP
/// backend, never a bare name), htp_performance_mode, and
/// enable_htp_fp16_precision = 1 only with --npu-fp16.
std::map<std::string, std::string> qnn_ep_options(const RunConfig& config,
                                                  const std::filesystem::path& backend_path);

/// Session config entries of an NPU session: session.disable_cpu_ep_fallback (1 in
/// strict mode, 0 with --no-strict), ep.context_enable = 0 (the session writes no
/// EP context) and, for runs that use EP contexts (--ep-cache build|use,
/// --init-breakdown), ep.context_embed_mode = kEpContextEmbedMode.
std::map<std::string, std::string> qnn_session_config(const RunConfig& config);

/// The entries of a session that writes the EP context model `context_model`:
/// qnn_session_config() with ep.context_enable = 1, ep.context_file_path and
/// ep.context_embed_mode.
std::map<std::string, std::string> qnn_context_build_config(
    const RunConfig& config, const std::filesystem::path& context_model);

/// The compute precision a run of a model stored as `stored` asks the device for
/// (results schema: config.requested.compute_precision).
///
/// CPU: nullopt (the CPU EP computes in the stored precision; nothing is requested).
/// NPU: "int8" for an INT8 QDQ model (static quantization), which runs as stored
/// and must not be given --npu-fp16; "fp16" for an FP32 model, which requires
/// --npu-fp16 because the NPU has no FP32 compute. Every other combination is a
/// usage error (exit code 2): npu_fp32_needs_fp16, npu_fp16_needs_fp32_model or
/// npu_precision_unsupported.
Result<std::optional<std::string>> requested_compute_precision(const RunConfig& config,
                                                               const ModelPrecision& stored);

/// model.precision_notes for a run with `compute_precision`, or nullopt.
std::optional<std::string> precision_notes(const ModelPrecision& stored,
                                           const std::optional<std::string>& compute_precision);

/// results schema: execution.effective_precision.
struct EffectivePrecision {
    std::string stored;   // model.stored_precision
    std::string compute;  // fp32 | fp16 | int8
    std::string source;   // model | enable_htp_fp16_precision

    bool operator==(const EffectivePrecision&) const = default;
};

/// What the device computes in. CPU: the stored precision (nullopt for a mixed
/// model, whose compute precision is not one value). NPU: `compute_precision`
/// (requested_compute_precision()), from the model unless an FP32 model is
/// computed in FP16; nullopt if it is unknown.
std::optional<EffectivePrecision> effective_precision(
    ProviderKind provider, const ModelPrecision& stored,
    const std::optional<std::string>& compute_precision);

}  // namespace edgebench
