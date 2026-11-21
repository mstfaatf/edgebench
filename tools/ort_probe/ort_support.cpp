#include "ort_support.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// windows.h first
#include <psapi.h>
#endif

namespace fs = std::filesystem;

namespace probe {

// --------------------------------------------------------------------------- DLLs

#ifdef _WIN32
namespace {

fs::path ModuleFileName(HMODULE module) {
    std::wstring buf(MAX_PATH, L'\0');
    DWORD n = 0;
    while ((n = GetModuleFileNameW(module, buf.data(), static_cast<DWORD>(buf.size()))) ==
           buf.size()) {
        buf.resize(buf.size() * 2);
    }
    buf.resize(n);
    return fs::path(buf);
}

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

fs::path ExecutableDir() {
    return ModuleFileName(nullptr).parent_path();
}

std::string LoadBundledOnnxRuntime(const fs::path& exe_dir) {
    const fs::path dll = exe_dir / "onnxruntime.dll";
    if (!fs::is_regular_file(dll))
        return dll.string() + " not found (the build copies it next to the executable)";
    // Full path, and resolve the DLL's own dependencies from its directory first.
    HMODULE module = LoadLibraryExW(
        dll.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module)
        return "LoadLibraryExW(" + dll.string() + ") failed, error " +
               std::to_string(GetLastError());
    using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
    auto get_api_base = reinterpret_cast<GetApiBaseFn>(GetProcAddress(module, "OrtGetApiBase"));
    if (!get_api_base)
        return dll.string() + " does not export OrtGetApiBase";
    const OrtApiBase* base = get_api_base();
    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        return dll.string() + " is ONNX Runtime " + base->GetVersionString() +
               ", which does not provide C API version " + std::to_string(ORT_API_VERSION);
    }
    Ort::InitApi(api);
    return {};
}

fs::path LoadedModulePath(const char* module_name) {
    HMODULE module = GetModuleHandleA(module_name);
    return module ? ModuleFileName(module) : fs::path();
}

std::vector<fs::path> LoadedOrtAndQnnModules() {
    HANDLE process = GetCurrentProcess();
    std::vector<HMODULE> modules(1024);
    DWORD needed = 0;
    while (true) {
        const auto bytes = static_cast<DWORD>(modules.size() * sizeof(HMODULE));
        if (!K32EnumProcessModules(process, modules.data(), bytes, &needed))
            return {};
        if (needed <= bytes)
            break;
        modules.resize(needed / sizeof(HMODULE));
    }
    modules.resize(needed / sizeof(HMODULE));
    std::vector<fs::path> out;
    for (HMODULE m : modules) {
        fs::path p = ModuleFileName(m);
        const std::string name = Lower(p.filename().string());
        if (name.find("onnxruntime") != std::string::npos || name.find("qnn") != std::string::npos)
            out.push_back(p);
    }
    std::sort(out.begin(), out.end());
    return out;
}
#else
fs::path ExecutableDir() {
    return fs::canonical("/proc/self/exe").parent_path();
}
std::string LoadBundledOnnxRuntime(const fs::path&) {
    Ort::InitApi();
    return {};
}
fs::path LoadedModulePath(const char*) {
    return {};
}
std::vector<fs::path> LoadedOrtAndQnnModules() {
    return {};
}
#endif

bool SameFile(const fs::path& a, const fs::path& b) {
    std::error_code ec;
    return !a.empty() && !b.empty() && fs::equivalent(a, b, ec) && !ec;
}

// --------------------------------------------------------------------------- devices

const char* DeviceTypeName(OrtHardwareDeviceType type) {
    switch (type) {
        case OrtHardwareDeviceType_CPU:
            return "CPU";
        case OrtHardwareDeviceType_GPU:
            return "GPU";
        case OrtHardwareDeviceType_NPU:
            return "NPU";
    }
    return "UNKNOWN";
}

namespace {

std::string Join(const std::unordered_map<std::string, std::string>& kv) {
    std::map<std::string, std::string> sorted(kv.begin(), kv.end());
    std::string s;
    for (const auto& [k, v] : sorted)
        s += (s.empty() ? "" : ", ") + k + "=" + v;
    return s;
}

const char* OrUnset(const char* s) {
    return (s && *s) ? s : "(unset)";
}

}  // namespace

void PrintDevices(const std::vector<Ort::ConstEpDevice>& devices) {
    if (devices.empty())
        std::printf("  (none)\n");
    for (const Ort::ConstEpDevice& d : devices) {
        Ort::ConstHardwareDevice hw = d.Device();
        std::printf(
            "  ep=%-22s ep_vendor=%-10s type=%-3s vendor=%s vendor_id=0x%x device_id=0x%x\n",
            d.EpName(), d.EpVendor(), DeviceTypeName(hw.Type()), OrUnset(hw.Vendor()),
            hw.VendorId(), hw.DeviceId());
        std::printf("      device metadata: {%s}\n      ep metadata: {%s}\n",
                    Join(hw.Metadata().GetKeyValuePairs()).c_str(),
                    Join(d.EpMetadata().GetKeyValuePairs()).c_str());
    }
}

void WriteDevices(JsonWriter& json, const std::vector<Ort::ConstEpDevice>& devices) {
    json.BeginArray();
    for (const Ort::ConstEpDevice& d : devices) {
        Ort::ConstHardwareDevice hw = d.Device();
        json.BeginObject()
            .Key("ep_name")
            .String(d.EpName())
            .Key("ep_vendor")
            .String(d.EpVendor())
            .Key("device_type")
            .String(DeviceTypeName(hw.Type()))
            .Key("device_vendor")
            .String(hw.Vendor() ? hw.Vendor() : "")
            .Key("device_vendor_id")
            .Int(hw.VendorId())
            .Key("device_id")
            .Int(hw.DeviceId());
        auto write_map = [&json](const std::unordered_map<std::string, std::string>& kv) {
            json.BeginObject();
            for (const auto& [k, v] : std::map<std::string, std::string>(kv.begin(), kv.end()))
                json.Key(k).String(v);
            json.EndObject();
        };
        json.Key("device_metadata");
        write_map(hw.Metadata().GetKeyValuePairs());
        json.Key("ep_metadata");
        write_map(d.EpMetadata().GetKeyValuePairs());
        json.EndObject();
    }
    json.EndArray();
}

Ort::ConstEpDevice SelectQnnNpu(const std::vector<Ort::ConstEpDevice>& devices) {
    std::vector<Ort::ConstEpDevice> npus;
    for (const Ort::ConstEpDevice& d : devices) {
        if (std::strcmp(d.EpName(), kQnnEpName) == 0 &&
            d.Device().Type() == OrtHardwareDeviceType_NPU) {
            npus.push_back(d);
        }
    }
    if (npus.size() != 1) {
        throw DeviceSelectionError("expected exactly one " + std::string(kQnnEpName) +
                                   " device of type NPU, found " + std::to_string(npus.size()));
    }
    return npus.front();
}

// --------------------------------------------------------------------------- sessions

std::map<std::string, std::string> QnnProviderOptions(const Options& o, const fs::path& exe_dir) {
    std::map<std::string, std::string> opts = {
        {"backend_path", (exe_dir / kQnnHtpBackend).string()},
        {"htp_performance_mode", "burst"},
    };
    if (o.fp16)
        opts["enable_htp_fp16_precision"] = "1";
    if (o.qnn_profiling != QnnProfiling::kOff) {
        opts["profiling_level"] = ToString(o.qnn_profiling);
        opts["profiling_file_path"] = fs::absolute(o.qnn_profile_file).string();
    }
    return opts;
}

std::map<std::string, std::string> SessionConfig(const Options& o, bool generate_context) {
    std::map<std::string, std::string> cfg;
    if (o.target == Target::kNpu && o.strict)
        cfg["session.disable_cpu_ep_fallback"] = "1";
    if (generate_context) {
        cfg["ep.context_enable"] = "1";
        cfg["ep.context_file_path"] = fs::absolute(o.context_file).string();
    }
    return cfg;
}

Ort::SessionOptions CpuSessionOptions(int threads) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(threads);
    return so;
}

Ort::SessionOptions NpuSessionOptions(Ort::Env& env, const Ort::ConstEpDevice& npu,
                                      const Options& o, const fs::path& exe_dir,
                                      bool generate_context) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(o.threads);
    for (const auto& [k, v] : SessionConfig(o, generate_context))
        so.AddConfigEntry(k.c_str(), v.c_str());
    const std::map<std::string, std::string> qnn = QnnProviderOptions(o, exe_dir);
    const std::unordered_map<std::string, std::string> ep_options(qnn.begin(), qnn.end());
    so.AppendExecutionProvider_V2(env, {npu}, ep_options);
    return so;
}

// --------------------------------------------------------------------------- runner

namespace {

size_t StaticCount(const std::vector<int64_t>& shape, const char* what) {
    size_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0)
            throw std::runtime_error(std::string(what) + " shape must be fully static");
        n *= static_cast<size_t>(d);
    }
    return n;
}

}  // namespace

ModelRunner::ModelRunner(Ort::Session session)
    : session_(std::move(session)),
      cpu_memory_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    if (session_.GetInputCount() != 1)
        throw std::runtime_error("expected a model with exactly one input");
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();

    // GetTensorTypeAndShapeInfo() returns a non-owning view into the TypeInfo, so the
    // TypeInfo objects must outlive it (a temporary TypeInfo leaves it dangling).
    const Ort::TypeInfo in_type = session_.GetInputTypeInfo(0);
    const Ort::TypeInfo out_type = session_.GetOutputTypeInfo(0);
    auto in_info = in_type.GetTensorTypeAndShapeInfo();
    auto out_info = out_type.GetTensorTypeAndShapeInfo();
    if (in_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        out_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("input and first output must be float32");
    }
    input_shape_ = in_info.GetShape();
    output_shape_ = out_info.GetShape();
    input_count_ = StaticCount(input_shape_, "input");
    output_count_ = StaticCount(output_shape_, "output");
}

std::vector<float> ModelRunner::Run(std::span<const float> input) {
    if (input.size() != input_count_)
        throw std::runtime_error("input has the wrong number of elements");
    // CreateTensor does not copy; the const_cast is safe because ORT only reads inputs.
    Ort::Value value =
        Ort::Value::CreateTensor<float>(cpu_memory_, const_cast<float*>(input.data()), input.size(),
                                        input_shape_.data(), input_shape_.size());
    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    std::vector<Ort::Value> outputs = session_.Run(run_options_, in_names, &value, 1, out_names, 1);
    const float* data = outputs.front().GetTensorData<float>();
    return std::vector<float>(data, data + output_count_);
}

void ModelRunner::PrepareBinding(std::span<const float> input) {
    if (input.size() != input_count_)
        throw std::runtime_error("input has the wrong number of elements");
    if (!binding_) {
        input_buffer_.resize(input_count_);
        output_buffer_.resize(output_count_);
        bound_input_ =
            Ort::Value::CreateTensor<float>(cpu_memory_, input_buffer_.data(), input_buffer_.size(),
                                            input_shape_.data(), input_shape_.size());
        bound_output_ = Ort::Value::CreateTensor<float>(cpu_memory_, output_buffer_.data(),
                                                        output_buffer_.size(), output_shape_.data(),
                                                        output_shape_.size());
        binding_ = Ort::IoBinding(session_);
        binding_.BindInput(input_name_.c_str(), bound_input_);
        binding_.BindOutput(output_name_.c_str(), bound_output_);
    }
    std::copy(input.begin(), input.end(), input_buffer_.begin());
}

void ModelRunner::RunBound() {
    session_.Run(run_options_, binding_);
}

std::vector<float> ModelRunner::RunWith(RunMode mode, std::span<const float> input) {
    if (mode == RunMode::kRun)
        return Run(input);
    PrepareBinding(input);
    RunBound();
    return BoundOutput();
}

const char* OrtErrorCodeName(OrtErrorCode code) {
    switch (code) {
        case ORT_OK:
            return "ORT_OK";
        case ORT_FAIL:
            return "ORT_FAIL";
        case ORT_INVALID_ARGUMENT:
            return "ORT_INVALID_ARGUMENT";
        case ORT_NO_SUCHFILE:
            return "ORT_NO_SUCHFILE";
        case ORT_NO_MODEL:
            return "ORT_NO_MODEL";
        case ORT_ENGINE_ERROR:
            return "ORT_ENGINE_ERROR";
        case ORT_RUNTIME_EXCEPTION:
            return "ORT_RUNTIME_EXCEPTION";
        case ORT_INVALID_PROTOBUF:
            return "ORT_INVALID_PROTOBUF";
        case ORT_MODEL_LOADED:
            return "ORT_MODEL_LOADED";
        case ORT_NOT_IMPLEMENTED:
            return "ORT_NOT_IMPLEMENTED";
        case ORT_INVALID_GRAPH:
            return "ORT_INVALID_GRAPH";
        case ORT_EP_FAIL:
            return "ORT_EP_FAIL";
        default:
            return "ORT_UNKNOWN_ERROR_CODE";
    }
}

}  // namespace probe
