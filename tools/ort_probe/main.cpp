// ort_probe: verify that ONNX Runtime and the QNN execution provider work from
// native C++ on Windows on Snapdragon.
//
// Steps (each reports a clear error instead of aborting the whole run):
//   1. Print the ONNX Runtime version and where onnxruntime.dll was loaded from.
//   2. List the built-in execution providers.
//   3. Register the QNN plugin EP library and enumerate EP devices.
//   4. Run the model on the CPU EP.
//   5. If an NPU device exists, run the model on it with CPU fallback disabled.
//   6. Compare the CPU and NPU outputs.
//
// The model input is filled with 0.5 so results are deterministic and can be
// compared against tools/ort_probe/reference.py.
//
// Usage: ort_probe <model.onnx>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr const char* kQnnEpName = "QNNExecutionProvider";
constexpr float kInputValue = 0.5f;

#ifdef _WIN32
constexpr const char* kQnnEpLibrary = "onnxruntime_providers_qnn.dll";
constexpr const char* kQnnHtpBackend = "QnnHtp.dll";
#else
constexpr const char* kQnnEpLibrary = "libonnxruntime_providers_qnn.so";
constexpr const char* kQnnHtpBackend = "libQnnHtp.so";
#endif

// Directory containing this executable; runtime DLLs are copied next to it.
fs::path ExecutableDir() {
#ifdef _WIN32
  std::wstring buf(MAX_PATH, L'\0');
  DWORD n = 0;
  while ((n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()))) == buf.size()) {
    buf.resize(buf.size() * 2);
  }
  buf.resize(n);
  return fs::path(buf).parent_path();
#else
  return fs::canonical("/proc/self/exe").parent_path();
#endif
}

std::string LoadedModulePath(const char* name) {
#ifdef _WIN32
  HMODULE mod = GetModuleHandleA(name);
  if (!mod) return "(not loaded)";
  std::wstring buf(MAX_PATH, L'\0');
  DWORD n = GetModuleFileNameW(mod, buf.data(), static_cast<DWORD>(buf.size()));
  buf.resize(n);
  return fs::path(buf).string();
#else
  (void)name;
  return "(unknown)";
#endif
}

const char* DeviceTypeName(OrtHardwareDeviceType type) {
  switch (type) {
    case OrtHardwareDeviceType_CPU: return "CPU";
    case OrtHardwareDeviceType_GPU: return "GPU";
    case OrtHardwareDeviceType_NPU: return "NPU";
  }
  return "UNKNOWN";
}

void Section(const char* title) { std::printf("\n== %s ==\n", title); }

void Fail(const char* step, const std::exception& e) {
  std::printf("  FAILED (%s): %s\n", step, e.what());
}

// Run one inference with every input filled with kInputValue. Requires a
// single float input with a fully static shape and returns the first output.
std::vector<float> RunOnce(Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  if (session.GetInputCount() != 1) {
    throw std::runtime_error("expected a model with exactly one input, got " +
                             std::to_string(session.GetInputCount()));
  }
  Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
  Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);

  auto type_info = session.GetInputTypeInfo(0);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("input must be float32");
  }
  std::vector<int64_t> shape = tensor_info.GetShape();
  size_t count = 1;
  for (int64_t d : shape) {
    if (d <= 0) throw std::runtime_error("input shape must be fully static (export with a fixed shape)");
    count *= static_cast<size_t>(d);
  }

  std::vector<float> input(count, kInputValue);
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape.data(), shape.size());

  const char* input_names[] = {input_name.get()};
  const char* output_names[] = {output_name.get()};
  std::vector<Ort::Value> outputs =
      session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

  const Ort::Value& out = outputs.front();
  auto out_info = out.GetTensorTypeAndShapeInfo();
  if (out_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("first output must be float32");
  }
  const float* data = out.GetTensorData<float>();
  return std::vector<float>(data, data + out_info.GetElementCount());
}

void PrintOutput(const char* label, const std::vector<float>& out) {
  auto top = std::max_element(out.begin(), out.end());
  std::printf("  %s top-1: %d (%.6f)\n", label, static_cast<int>(top - out.begin()), *top);
  std::printf("  %s first 5:", label);
  for (size_t i = 0; i < std::min<size_t>(5, out.size()); ++i) std::printf(" %.6f", out[i]);
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.onnx>\n", argv[0]);
    return 2;
  }
  const fs::path model_path = fs::absolute(argv[1]);
  if (!fs::is_regular_file(model_path)) {
    std::fprintf(stderr, "model not found: %s\n", model_path.string().c_str());
    return 2;
  }
  const fs::path exe_dir = ExecutableDir();

  int failures = 0;
  std::optional<Ort::Env> env;

  Section("ONNX Runtime");
  try {
    env.emplace(ORT_LOGGING_LEVEL_ERROR, "ort_probe");
    std::printf("  version: %s\n", Ort::GetVersionString().c_str());
    std::printf("  loaded from: %s\n", LoadedModulePath("onnxruntime.dll").c_str());
  } catch (const std::exception& e) {
    Fail("create environment", e);
    return 1;  // nothing else can run without an environment
  }

  Section("Built-in execution providers");
  try {
    for (const std::string& p : Ort::GetAvailableProviders()) std::printf("  %s\n", p.c_str());
  } catch (const std::exception& e) {
    Fail("list providers", e);
    ++failures;
  }

  Section("QNN plugin EP registration");
  bool qnn_registered = false;
  const fs::path qnn_lib = exe_dir / kQnnEpLibrary;
  try {
    if (!fs::is_regular_file(qnn_lib)) {
      throw std::runtime_error(qnn_lib.string() + " not found (did the build copy the QNN DLLs?)");
    }
    env->RegisterExecutionProviderLibrary(kQnnEpName, qnn_lib.native());
    qnn_registered = true;
    std::printf("  registered %s from %s\n", kQnnEpName, qnn_lib.string().c_str());
  } catch (const std::exception& e) {
    Fail("register QNN EP library", e);
    ++failures;
  }

  Section("EP devices");
  std::optional<Ort::ConstEpDevice> npu_device;
  try {
    std::vector<Ort::ConstEpDevice> devices = env->GetEpDevices();
    if (devices.empty()) std::printf("  (none)\n");
    for (const Ort::ConstEpDevice& d : devices) {
      Ort::ConstHardwareDevice hw = d.Device();
      const char* hw_vendor = hw.Vendor();
      std::printf("  ep=%-22s ep_vendor=%-10s device=%-3s device_vendor=%s\n", d.EpName(), d.EpVendor(),
                  DeviceTypeName(hw.Type()), (hw_vendor && *hw_vendor) ? hw_vendor : "(unset)");
      if (!npu_device && std::strcmp(d.EpName(), kQnnEpName) == 0 && hw.Type() == OrtHardwareDeviceType_NPU) {
        npu_device = d;
      }
    }
  } catch (const std::exception& e) {
    Fail("enumerate EP devices", e);
    ++failures;
  }

  std::printf("\nmodel: %s\ninput: every value = %.1f\n", model_path.string().c_str(), kInputValue);

  Section("CPU EP");
  std::optional<std::vector<float>> cpu_out;
  try {
    Ort::SessionOptions so;
    Ort::Session session(*env, model_path.c_str(), so);
    cpu_out = RunOnce(session);
    PrintOutput("cpu", *cpu_out);
  } catch (const std::exception& e) {
    Fail("CPU session", e);
    ++failures;
  }

  Section("QNN EP on NPU (CPU fallback disabled)");
  std::optional<std::vector<float>> npu_out;
  if (!qnn_registered) {
    std::printf("  skipped: QNN EP not registered\n");
  } else if (!npu_device) {
    std::printf("  skipped: no QNN NPU device found\n");
  } else {
    try {
      Ort::SessionOptions so;
      // Fail session creation instead of silently running unsupported nodes on CPU.
      so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      const std::unordered_map<std::string, std::string> ep_options = {
          {"backend_path", (exe_dir / kQnnHtpBackend).string()},
      };
      so.AppendExecutionProvider_V2(*env, {*npu_device}, ep_options);
      Ort::Session session(*env, model_path.c_str(), so);
      npu_out = RunOnce(session);
      PrintOutput("npu", *npu_out);
    } catch (const std::exception& e) {
      Fail("NPU session", e);
      ++failures;
    }
  }

  if (cpu_out && npu_out) {
    Section("CPU vs NPU");
    if (cpu_out->size() != npu_out->size()) {
      std::printf("  output sizes differ: %zu vs %zu\n", cpu_out->size(), npu_out->size());
      ++failures;
    } else {
      float max_diff = 0.f;
      for (size_t i = 0; i < cpu_out->size(); ++i) {
        max_diff = std::max(max_diff, std::fabs((*cpu_out)[i] - (*npu_out)[i]));
      }
      auto top = [](const std::vector<float>& v) { return std::max_element(v.begin(), v.end()) - v.begin(); };
      const bool same_top1 = top(*cpu_out) == top(*npu_out);
      std::printf("  max abs diff: %.6g\n  top-1 match: %s\n", max_diff, same_top1 ? "yes" : "NO");
      if (!same_top1) ++failures;
    }
  }

  if (qnn_registered) {
    try {
      env->UnregisterExecutionProviderLibrary(kQnnEpName);  // all sessions are released by now
    } catch (const std::exception& e) {
      Fail("unregister QNN EP library", e);
    }
  }

  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "OK", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
