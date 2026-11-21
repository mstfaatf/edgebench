// Model input/output metadata and the checks every provider applies to it.
#pragma once

#include <onnxruntime_c_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/result.hpp"

namespace edgebench {

/// One model input or output as declared in the model.
struct TensorInfo {
    std::string name;
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<std::int64_t> shape;      // -1 for a dynamic dimension
    std::vector<std::string> dim_params;  // symbolic name per dimension ("" if none)

    /// Product of the dimensions (only meaningful for fully fixed shapes).
    std::size_t element_count() const;
    std::size_t byte_size() const { return element_count() * dtype_size(dtype); }

    /// Bytes per element; 0 for types without a fixed size (string, undefined).
    static std::size_t dtype_size(ONNXTensorElementDataType dtype);
};

/// Results-schema dtype name ("float32", "int64", ...), or "unknown".
std::string_view dtype_name(ONNXTensorElementDataType dtype);

/// Usage error (kind "dynamic_shape") naming the first input or output with a
/// symbolic or dynamic dimension, e.g. "input 'x' has a dynamic dimension 0 ('batch')".
Result<void> check_fixed_shapes(const std::vector<TensorInfo>& inputs,
                                const std::vector<TensorInfo>& outputs);

/// Usage error (kind "unsupported_input_dtype") for any input that is not float32,
/// and (kind "unsupported_output_dtype") for outputs without a fixed element size.
Result<void> check_supported_dtypes(const std::vector<TensorInfo>& inputs,
                                    const std::vector<TensorInfo>& outputs);

/// Read-only view of a tensor's data.
struct TensorView {
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::span<const std::byte> bytes;

    bool empty() const { return bytes.empty(); }
    /// The data as float32 (empty unless dtype is float32).
    std::span<const float> as_float() const;
};

/// Index of the first NaN or infinity in a floating-point view (float32,
/// float64, float16, bfloat16); nullopt if all are finite or the type is not floating point.
std::optional<std::size_t> first_non_finite(const TensorView& view);

}  // namespace edgebench
