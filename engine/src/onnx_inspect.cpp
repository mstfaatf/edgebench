#include "edgebench/onnx_inspect.hpp"

#include <onnxruntime_c_api.h>

#include <algorithm>
#include <charconv>
#include <fstream>
#include <set>
#include <string_view>
#include <vector>

#include "edgebench/ort_loader.hpp"
#include "edgebench/tensor_info.hpp"

namespace edgebench {

namespace {

// Minimal protobuf wire-format reader (varint, 64-bit, length-delimited, 32-bit).
class Reader {
public:
    explicit Reader(std::span<const std::byte> data)
        : p_(reinterpret_cast<const std::uint8_t*>(data.data())), end_(p_ + data.size()) {}

    bool done() const { return p_ >= end_; }
    bool ok() const { return ok_; }

    std::uint64_t varint() {
        std::uint64_t v = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (p_ >= end_)
                return fail();
            const std::uint8_t b = *p_++;
            v |= static_cast<std::uint64_t>(b & 0x7F) << shift;
            if (!(b & 0x80))
                return v;
        }
        return fail();
    }

    // Next field: returns false at the end or on malformed input.
    bool next(std::uint32_t& field, std::uint32_t& wire) {
        if (done() || !ok_)
            return false;
        const std::uint64_t key = varint();
        field = static_cast<std::uint32_t>(key >> 3);
        wire = static_cast<std::uint32_t>(key & 7);
        return ok_ && field != 0;
    }

    std::span<const std::byte> bytes() {
        const std::uint64_t n = varint();
        if (!ok_ || n > static_cast<std::uint64_t>(end_ - p_)) {
            fail();
            return {};
        }
        const auto* start = reinterpret_cast<const std::byte*>(p_);
        p_ += n;
        return {start, static_cast<std::size_t>(n)};
    }

    std::string string() {
        const auto b = bytes();
        return {reinterpret_cast<const char*>(b.data()), b.size()};
    }

    void skip(std::uint32_t wire) {
        switch (wire) {
            case 0:
                varint();
                break;
            case 1:
                advance(8);
                break;
            case 2:
                bytes();
                break;
            case 5:
                advance(4);
                break;
            default:
                fail();  // groups (3, 4) are not used by ONNX
        }
    }

private:
    std::uint64_t fail() {
        ok_ = false;
        p_ = end_;
        return 0;
    }
    void advance(std::size_t n) {
        if (static_cast<std::size_t>(end_ - p_) < n)
            fail();
        else
            p_ += n;
    }
    const std::uint8_t* p_;
    const std::uint8_t* end_;
    bool ok_ = true;
};

struct Node {
    std::string op_type, domain;
    std::vector<std::string> inputs;
};

Node read_node(std::span<const std::byte> data) {
    Node n;
    Reader r(data);
    std::uint32_t field = 0, wire = 0;
    while (r.next(field, wire)) {
        if (field == 1 && wire == 2)
            n.inputs.push_back(r.string());
        else if (field == 4 && wire == 2)
            n.op_type = r.string();
        else if (field == 7 && wire == 2)
            n.domain = r.string();
        else
            r.skip(wire);
    }
    return n;
}

// TensorProto: data_type (2), name (8).
std::pair<std::string, int> read_initializer(std::span<const std::byte> data) {
    std::string name;
    int dtype = 0;
    Reader r(data);
    std::uint32_t field = 0, wire = 0;
    while (r.next(field, wire)) {
        if (field == 2 && wire == 0)
            dtype = static_cast<int>(r.varint());
        else if (field == 8 && wire == 2)
            name = r.string();
        else
            r.skip(wire);
    }
    return {name, dtype};
}

// ValueInfoProto: name (1), type (2) -> TypeProto.tensor_type (1) -> elem_type (1),
// shape (2) -> TensorShapeProto.dim (1) -> dim_value (1) | dim_param (2).
TensorInfo read_value_info(std::span<const std::byte> data) {
    TensorInfo t;
    Reader r(data);
    std::uint32_t field = 0, wire = 0;
    while (r.next(field, wire)) {
        if (field == 1 && wire == 2) {
            t.name = r.string();
        } else if (field == 2 && wire == 2) {
            Reader type(r.bytes());
            std::uint32_t f = 0, w = 0;
            while (type.next(f, w)) {
                if (f != 1 || w != 2) {  // only tensor types
                    type.skip(w);
                    continue;
                }
                Reader tensor(type.bytes());
                std::uint32_t tf = 0, tw = 0;
                while (tensor.next(tf, tw)) {
                    if (tf == 1 && tw == 0) {
                        t.dtype = static_cast<ONNXTensorElementDataType>(tensor.varint());
                    } else if (tf == 2 && tw == 2) {
                        Reader shape(tensor.bytes());
                        std::uint32_t sf = 0, sw = 0;
                        while (shape.next(sf, sw)) {
                            if (sf != 1 || sw != 2) {
                                shape.skip(sw);
                                continue;
                            }
                            std::int64_t value = -1;
                            std::string param;
                            Reader dim(shape.bytes());
                            std::uint32_t df = 0, dw = 0;
                            while (dim.next(df, dw)) {
                                if (df == 1 && dw == 0)
                                    value = static_cast<std::int64_t>(dim.varint());
                                else if (df == 2 && dw == 2)
                                    param = dim.string();
                                else
                                    dim.skip(dw);
                            }
                            t.shape.push_back(param.empty() ? value : -1);
                            t.dim_params.push_back(param);
                        }
                    } else {
                        tensor.skip(tw);
                    }
                }
            }
        } else {
            r.skip(wire);
        }
    }
    return t;
}

int most_common(const std::map<int, int>& counts) {
    return std::max_element(counts.begin(), counts.end(),
                            [](const auto& a, const auto& b) { return a.second < b.second; })
        ->first;
}

bool is_default_domain(std::string_view domain) {
    return domain.empty() || domain == "ai.onnx";
}

}  // namespace

Result<OnnxGraphSummary> inspect_onnx(std::span<const std::byte> model_bytes) {
    OnnxGraphSummary s;
    std::span<const std::byte> graph;
    Reader model(model_bytes);
    std::uint32_t field = 0, wire = 0;
    while (model.next(field, wire)) {
        if (field == 1 && wire == 0) {
            s.ir_version = static_cast<std::int64_t>(model.varint());
        } else if (field == 2 && wire == 2) {
            s.producer_name = model.string();
        } else if (field == 3 && wire == 2) {
            s.producer_version = model.string();
        } else if (field == 7 && wire == 2) {
            graph = model.bytes();
        } else if (field == 14 && wire == 2) {  // StringStringEntryProto: key (1), value (2)
            Reader entry(model.bytes());
            std::string key, value;
            std::uint32_t f = 0, w = 0;
            while (entry.next(f, w)) {
                if (f == 1 && w == 2)
                    key = entry.string();
                else if (f == 2 && w == 2)
                    value = entry.string();
                else
                    entry.skip(w);
            }
            s.metadata[key] = value;
        } else if (field == 8 && wire == 2) {  // OperatorSetIdProto: domain (1), version (2)
            Reader opset(model.bytes());
            std::string domain;
            std::optional<std::int64_t> version;
            std::uint32_t f = 0, w = 0;
            while (opset.next(f, w)) {
                if (f == 1 && w == 2)
                    domain = opset.string();
                else if (f == 2 && w == 0)
                    version = static_cast<std::int64_t>(opset.varint());
                else
                    opset.skip(w);
            }
            if (is_default_domain(domain) && version)
                s.opset = version;
        } else {
            model.skip(wire);
        }
    }
    if (!model.ok() || !s.ir_version || graph.empty())
        return Error{ExitCode::usage_error, "invalid_onnx", "not a valid ONNX model file"};

    std::vector<Node> nodes;
    std::map<std::string, int> initializer_dtype;
    std::vector<TensorInfo> graph_inputs;
    Reader g(graph);
    while (g.next(field, wire)) {
        if (field == 1 && wire == 2) {
            nodes.push_back(read_node(g.bytes()));
        } else if (field == 5 && wire == 2) {
            auto [name, dtype] = read_initializer(g.bytes());
            ++s.initializer_dtypes[dtype];
            initializer_dtype[name] = dtype;
        } else if (field == 11 && wire == 2) {
            graph_inputs.push_back(read_value_info(g.bytes()));
        } else if (field == 12 && wire == 2) {
            s.outputs.push_back(read_value_info(g.bytes()));
        } else {
            g.skip(wire);
        }
    }
    if (!g.ok())
        return Error{ExitCode::usage_error, "invalid_onnx", "the model's graph cannot be decoded"};
    // Older models also list initializers as graph inputs; those are not model inputs.
    for (TensorInfo& t : graph_inputs) {
        if (!initializer_dtype.contains(t.name))
            s.inputs.push_back(std::move(t));
    }

    std::map<int, int> weight_dtypes, activation_dtypes;
    for (const Node& n : nodes) {
        if (!is_default_domain(n.domain))
            continue;
        ++s.op_counts[n.op_type];
        if (n.op_type == "DequantizeLinear" && !n.inputs.empty()) {
            if (auto it = initializer_dtype.find(n.inputs[0]); it != initializer_dtype.end())
                ++weight_dtypes[it->second];
        } else if (n.op_type == "QuantizeLinear") {
            // Without a zero point, QuantizeLinear produces uint8 (ONNX spec).
            int dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
            if (n.inputs.size() > 2 && !n.inputs[2].empty()) {
                const auto it = initializer_dtype.find(n.inputs[2]);
                dtype = it != initializer_dtype.end() ? it->second : 0;
            }
            if (dtype != 0)
                ++activation_dtypes[dtype];
        }
    }
    if (!weight_dtypes.empty())
        s.weight_dtype = most_common(weight_dtypes);
    if (!activation_dtypes.empty())
        s.activation_dtype = most_common(activation_dtypes);
    return s;
}

Result<OnnxGraphSummary> inspect_onnx_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return Error{ExitCode::usage_error, "file_read_failed",
                     "cannot open " + path_to_utf8(path) + " for reading"};
    }
    std::vector<std::byte> data(static_cast<std::size_t>(in.tellg()));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!in) {
        return Error{ExitCode::usage_error, "file_read_failed",
                     "error while reading " + path_to_utf8(path)};
    }
    return inspect_onnx(data);
}

std::optional<std::string> normalization_from_metadata(const OnnxGraphSummary& graph) {
    const auto it = graph.metadata.find(std::string(kNormalizationMetadataKey));
    if (it == graph.metadata.end())
        return std::nullopt;
    for (std::string_view known : {"external", "in_graph", "folded"}) {
        if (it->second == known)
            return it->second;
    }
    return std::nullopt;
}

QuantizationRecord quantization_from_metadata(const OnnxGraphSummary& graph) {
    auto get = [&graph](const char* key) -> std::optional<std::string> {
        const auto it = graph.metadata.find(key);
        if (it == graph.metadata.end() || it->second.empty())
            return std::nullopt;
        return it->second;
    };
    QuantizationRecord q;
    q.calibration_dataset = get("edgebench.quant.calibration_dataset");
    q.tool = get("edgebench.quant.tool");
    if (const auto count = get("edgebench.quant.calibration_count")) {
        std::int64_t value = 0;
        const auto [end, ec] = std::from_chars(count->data(), count->data() + count->size(), value);
        if (ec == std::errc{} && end == count->data() + count->size() && value > 0)
            q.calibration_count = value;
    }
    return q;
}

ModelPrecision infer_precision(const OnnxGraphSummary& g) {
    auto has = [&g](std::string_view op) { return g.op_counts.contains(std::string(op)); };
    bool qoperator = false;
    for (const auto& [op, count] : g.op_counts) {
        if (op.rfind("QLinear", 0) == 0 || op == "MatMulInteger" || op == "ConvInteger")
            qoperator = true;
    }
    const bool qdq = has("QuantizeLinear") || has("DequantizeLinear");
    const bool dynamic = has("DynamicQuantizeLinear");

    ModelPrecision p;
    auto name = [](std::optional<int> dtype) -> std::optional<std::string> {
        if (!dtype)
            return std::nullopt;
        return std::string(dtype_name(static_cast<ONNXTensorElementDataType>(*dtype)));
    };
    if (qdq || qoperator || dynamic) {
        p.quantization_method = dynamic ? "dynamic" : "static";
        p.quantization_format = qoperator ? "qoperator" : "qdq";
        p.activation_dtype = name(g.activation_dtype);
        p.weight_dtype = name(g.weight_dtype);
        const std::set<std::string> eight_bit = {"int8", "uint8"};
        const bool all_eight_bit =
            (!p.activation_dtype || eight_bit.contains(*p.activation_dtype)) &&
            (!p.weight_dtype || eight_bit.contains(*p.weight_dtype));
        p.stored_precision = all_eight_bit ? "int8" : "mixed";
        return p;
    }
    p.quantization_method = "none";
    auto count = [&g](int dtype) {
        const auto it = g.initializer_dtypes.find(dtype);
        return it == g.initializer_dtypes.end() ? 0 : it->second;
    };
    const int f32 = count(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const int f16 = count(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) +
                    count(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    if (f16 > 0)
        p.stored_precision = f32 > 0 ? "mixed" : "fp16";
    else
        p.stored_precision = "fp32";
    return p;
}

}  // namespace edgebench
