#include "edgebench/tensor_info.hpp"

#include <cmath>
#include <cstring>

namespace edgebench {

std::size_t TensorInfo::element_count() const {
    std::size_t n = 1;
    for (std::int64_t d : shape)
        n *= d > 0 ? static_cast<std::size_t>(d) : 0;
    return n;
}

std::size_t TensorInfo::dtype_size(ONNXTensorElementDataType dtype) {
    switch (dtype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return 1;
        default:
            return 0;
    }
}

std::string_view dtype_name(ONNXTensorElementDataType dtype) {
    switch (dtype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return "bfloat16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
            return "int4";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
            return "uint4";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return "string";
        default:
            return "unknown";
    }
}

namespace {

Result<void> check_shapes(const std::vector<TensorInfo>& tensors, const char* role) {
    for (const TensorInfo& t : tensors) {
        for (std::size_t d = 0; d < t.shape.size(); ++d) {
            const std::string param = d < t.dim_params.size() ? t.dim_params[d] : "";
            if (t.shape[d] > 0 && param.empty())
                continue;
            std::string dim = std::to_string(d);
            if (!param.empty())
                dim += " ('" + param + "')";
            return Error{ExitCode::usage_error, "dynamic_shape",
                         std::string(role) + " '" + t.name + "' has a dynamic dimension " + dim +
                             "; only models with fixed input and output shapes are supported"};
        }
    }
    return {};
}

}  // namespace

Result<void> check_fixed_shapes(const std::vector<TensorInfo>& inputs,
                                const std::vector<TensorInfo>& outputs) {
    if (Result<void> r = check_shapes(inputs, "input"); !r)
        return r;
    return check_shapes(outputs, "output");
}

Result<void> check_supported_dtypes(const std::vector<TensorInfo>& inputs,
                                    const std::vector<TensorInfo>& outputs) {
    for (const TensorInfo& t : inputs) {
        if (t.dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return Error{ExitCode::usage_error, "unsupported_input_dtype",
                         "input '" + t.name + "' has dtype " + std::string(dtype_name(t.dtype)) +
                             "; only float32 inputs are supported"};
        }
    }
    for (const TensorInfo& t : outputs) {
        if (TensorInfo::dtype_size(t.dtype) == 0) {
            return Error{ExitCode::usage_error, "unsupported_output_dtype",
                         "output '" + t.name + "' has dtype " + std::string(dtype_name(t.dtype)) +
                             ", which has no fixed element size"};
        }
    }
    return {};
}

std::span<const float> TensorView::as_float() const {
    if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        return {};
    return {reinterpret_cast<const float*>(bytes.data()), bytes.size() / sizeof(float)};
}

std::optional<std::size_t> first_non_finite(const TensorView& view) {
    const std::byte* p = view.bytes.data();
    switch (view.dtype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            const std::size_t n = view.bytes.size() / sizeof(float);
            for (std::size_t i = 0; i < n; ++i) {
                float v;
                std::memcpy(&v, p + i * sizeof(float), sizeof(v));
                if (!std::isfinite(v))
                    return i;
            }
            return std::nullopt;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            const std::size_t n = view.bytes.size() / sizeof(double);
            for (std::size_t i = 0; i < n; ++i) {
                double v;
                std::memcpy(&v, p + i * sizeof(double), sizeof(v));
                if (!std::isfinite(v))
                    return i;
            }
            return std::nullopt;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: {
            // Non-finite when every exponent bit is set.
            const std::uint16_t mask =
                view.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? 0x7C00 : 0x7F80;
            const std::size_t n = view.bytes.size() / 2;
            for (std::size_t i = 0; i < n; ++i) {
                std::uint16_t v;
                std::memcpy(&v, p + i * 2, sizeof(v));
                if ((v & mask) == mask)
                    return i;
            }
            return std::nullopt;
        }
        default:
            return std::nullopt;
    }
}

}  // namespace edgebench
