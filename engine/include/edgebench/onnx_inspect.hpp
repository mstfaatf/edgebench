// Read-only inspection of an ONNX model file (results schema: model.opset,
// ir_version, stored_precision, quantization, source.export_tool).
//
// ONNX Runtime does not expose these, so the file's protobuf is decoded
// directly. Only the few fields needed are read; tensor data is never touched
// (external data files are not opened).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/result.hpp"
#include "edgebench/tensor_info.hpp"

namespace edgebench {

struct OnnxGraphSummary {
    std::optional<std::int64_t> ir_version;
    std::optional<std::int64_t> opset;  // of the default ("" / "ai.onnx") domain
    std::string producer_name, producer_version;
    std::map<std::string, int> op_counts;   // op_type -> nodes (top-level graph)
    std::map<int, int> initializer_dtypes;  // ONNX element type -> initializers
    std::optional<int> weight_dtype;        // dtype of initializers fed to DequantizeLinear
    std::optional<int> activation_dtype;    // zero-point dtype of QuantizeLinear
    /// Graph inputs (initializers excluded) and outputs as declared in the file,
    /// so the model's tensors are known even when no session can be created.
    std::vector<TensorInfo> inputs, outputs;
    /// The model's metadata_props (key -> value; the last entry wins for a repeated key).
    std::map<std::string, std::string> metadata;
};

/// What the file says about precision and quantization, in results-schema terms.
struct ModelPrecision {
    std::string stored_precision;                    // "fp32", "fp16", "int8" or "mixed"
    std::string quantization_method;                 // "none", "static" or "dynamic"
    std::optional<std::string> quantization_format;  // "qdq" or "qoperator"
    std::optional<std::string> activation_dtype;     // e.g. "uint8"
    std::optional<std::string> weight_dtype;         // e.g. "int8"
};

/// Decodes a serialized ModelProto. usage_error (kind "invalid_onnx") if it is
/// not one.
Result<OnnxGraphSummary> inspect_onnx(std::span<const std::byte> model_bytes);
Result<OnnxGraphSummary> inspect_onnx_file(const std::filesystem::path& path);

/// Metadata key EdgeBench's export pipeline uses to say where input normalization
/// happens (results schema model.normalization).
inline constexpr std::string_view kNormalizationMetadataKey = "edgebench.normalization";

/// model.normalization from the file's metadata: "external", "in_graph" or
/// "folded"; nullopt if the key is absent or holds another value.
std::optional<std::string> normalization_from_metadata(const OnnxGraphSummary& graph);

/// What EdgeBench's quantization pipeline recorded in the file's metadata
/// (edgebench.quant.*): results schema model.quantization.calibration and tool.
struct QuantizationRecord {
    std::optional<std::string> calibration_dataset;
    std::optional<std::int64_t> calibration_count;
    std::optional<std::string> tool;
};
QuantizationRecord quantization_from_metadata(const OnnxGraphSummary& graph);

/// Precision and quantization inferred from the graph: QuantizeLinear /
/// DequantizeLinear nodes make a QDQ model (static, or dynamic with
/// DynamicQuantizeLinear), QLinear* and *Integer ops a QOperator model. An
/// 8-bit quantized model is "int8"; other bit widths, or float16 next to float32
/// weights, are "mixed".
ModelPrecision infer_precision(const OnnxGraphSummary& graph);

}  // namespace edgebench
