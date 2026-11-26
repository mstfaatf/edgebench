#include "edgebench/npu_options.hpp"

#include <algorithm>

#include "edgebench/ort_loader.hpp"

namespace edgebench {

bool is_htp_performance_mode(std::string_view mode) {
    return std::find(kHtpPerformanceModes.begin(), kHtpPerformanceModes.end(), mode) !=
           kHtpPerformanceModes.end();
}

std::string htp_performance_modes_list() {
    std::string out;
    for (std::string_view mode : kHtpPerformanceModes) {
        if (!out.empty())
            out += ", ";
        out += mode;
    }
    return out;
}

std::string effective_htp_performance_mode(const RunConfig& config) {
    return config.htp_performance_mode.value_or(std::string(kDefaultHtpPerformanceMode));
}

std::map<std::string, std::string> qnn_ep_options(const RunConfig& config,
                                                  const std::filesystem::path& backend_path) {
    std::map<std::string, std::string> options = {
        {"backend_path", path_to_utf8(backend_path)},
        {"htp_performance_mode", effective_htp_performance_mode(config)},
    };
    if (config.npu_fp16)
        options["enable_htp_fp16_precision"] = "1";
    return options;
}

std::map<std::string, std::string> qnn_session_config(const RunConfig& config) {
    return {
        {"session.disable_cpu_ep_fallback", config.strict ? "1" : "0"},
        {"ep.context_enable", "0"},
    };
}

Result<std::optional<std::string>> requested_compute_precision(const RunConfig& config,
                                                               const ModelPrecision& stored) {
    if (config.provider != ProviderKind::npu)
        return std::optional<std::string>();
    const std::string file = path_to_utf8(config.model.filename());
    const bool int8_qdq = stored.stored_precision == "int8" &&
                          stored.quantization_method == "static" &&
                          stored.quantization_format == "qdq";
    const bool fp32 = stored.stored_precision == "fp32" && stored.quantization_method == "none";
    if (int8_qdq) {
        if (config.npu_fp16) {
            return Error{ExitCode::usage_error, "npu_fp16_needs_fp32_model",
                         "--npu-fp16 applies to FP32 models only; " + file +
                             " is an INT8 QDQ model and runs on the NPU as INT8"};
        }
        return std::optional<std::string>("int8");
    }
    if (fp32) {
        if (!config.npu_fp16) {
            return Error{ExitCode::usage_error, "npu_fp32_needs_fp16",
                         "the NPU has no FP32 compute: " + file +
                             " is an FP32 model; pass --npu-fp16 to run it with FP16 compute "
                             "(recorded as compute precision fp16)"};
        }
        return std::optional<std::string>("fp16");
    }
    std::string what = "stored precision " + stored.stored_precision;
    if (stored.quantization_method != "none") {
        what += ", " + stored.quantization_method + " quantization";
        if (stored.quantization_format)
            what += " (" + *stored.quantization_format + ")";
    }
    return Error{ExitCode::usage_error, "npu_precision_unsupported",
                 file + " (" + what +
                     ") cannot run on the NPU: supported are INT8 QDQ models with static "
                     "quantization, and FP32 models with --npu-fp16"};
}

std::optional<std::string> precision_notes(const ModelPrecision& stored,
                                           const std::optional<std::string>& compute_precision) {
    if (compute_precision == "fp16" && stored.stored_precision == "fp32") {
        return "Stored as FP32; the QNN HTP backend computes in FP16 "
               "(enable_htp_fp16_precision = 1). The NPU has no FP32 compute.";
    }
    return std::nullopt;
}

std::optional<EffectivePrecision> effective_precision(
    ProviderKind provider, const ModelPrecision& stored,
    const std::optional<std::string>& compute_precision) {
    const std::string& s = stored.stored_precision;
    if (provider == ProviderKind::cpu) {
        if (s != "fp32" && s != "fp16" && s != "int8")
            return std::nullopt;
        return EffectivePrecision{s, s, "model"};
    }
    if (!compute_precision)
        return std::nullopt;
    const bool fp16_of_fp32 = *compute_precision == "fp16" && s == "fp32";
    return EffectivePrecision{s, *compute_precision,
                              fp16_of_fp32 ? "enable_htp_fp16_precision" : "model"};
}

}  // namespace edgebench
