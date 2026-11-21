// ort_probe: run an ONNX model with ONNX Runtime from native C++ on the CPU EP
// or on the Snapdragon NPU through the QNN plugin EP, and check the result.
//
//   1. Verify that onnxruntime.dll was loaded from the executable's directory
//      (not the unrelated copy in C:\Windows\System32) and list the loaded
//      ONNX Runtime / QNN DLLs.
//   2. Register the QNN plugin EP library, list all EP devices, and select the
//      single QNN device whose hardware type is NPU.
//   3. Create the session (timed, --init-reps times): CPU EP with a fixed
//      intra-op thread count, or the NPU with CPU fallback disabled (strict),
//      optionally generating and then loading an EP context model.
//   4. Run the inputs (reference tensors written by make_reference.py, or a
//      generated fill), compare the outputs with Python's, optionally with the
//      CPU EP, and optionally measure steady-state latency (plain Run or
//      IoBinding with preallocated buffers).
//
// onnxruntime.dll is delay-loaded and the ORT C++ API is initialized manually
// (ORT_API_MANUAL_INIT), so the first thing main() does is load the copy next to
// the executable by full path and verify it; see edgebench::load_onnxruntime().
//
// Run with --help for the options. Exit codes: 0 ok, 1 check failed, 2 usage,
// 3 ONNX Runtime error.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "edgebench/ort_loader.hpp"
#include "ort_support.hpp"
#include "probe_util.hpp"

namespace fs = std::filesystem;
using namespace probe;

namespace {

constexpr int kExitOk = 0;
constexpr int kExitCheckFailed = 1;
constexpr int kExitUsage = 2;
constexpr int kExitOrtError = 3;

// AllClose bounds for NPU-vs-CPU checks: the HTP backend computes float
// graphs at reduced precision, so exact agreement with the CPU EP is not expected.
constexpr double kCpuCompareRtol = 1e-2;
constexpr double kCpuCompareAtol = 1e-3;

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void Section(const char* title) {
    std::printf("\n== %s ==\n", title);
}

struct Input {
    std::string id;
    std::vector<float> data;
};

struct ImageComparison {
    std::string id;
    Comparison cmp;
};

struct Report {
    JsonWriter json;
    bool failed = false;
};

void WriteModuleList(JsonWriter& j, const std::vector<edgebench::LoadedModule>& modules) {
    j.BeginArray();
    for (const edgebench::LoadedModule& m : modules)
        j.String(m.path.string());
    j.EndArray();
}

bool SameFile(const fs::path& a, const fs::path& b) {
    return !a.empty() && !b.empty() &&
           edgebench::is_same_path(edgebench::resolve_path(a).native(),
                                   edgebench::resolve_path(b).native());
}

void WriteStats(JsonWriter& j, const LatencyStats& s) {
    j.BeginObject()
        .Key("count")
        .Int(static_cast<int64_t>(s.count))
        .Key("mean_ms")
        .Number(s.mean)
        .Key("p50_ms")
        .Number(s.p50)
        .Key("p95_ms")
        .Number(s.p95)
        .Key("min_ms")
        .Number(s.min)
        .Key("max_ms")
        .Number(s.max)
        .Key("stddev_ms")
        .Number(s.stddev)
        .EndObject();
}

void WriteNumbers(JsonWriter& j, const std::vector<double>& v) {
    j.BeginArray();
    for (double x : v)
        j.Number(x);
    j.EndArray();
}

// Removes an EP context model and the context binaries written next to it
// ("<stem>_qnn.bin" etc.), so generation starts from a clean state.
void RemoveContextFiles(const fs::path& ctx) {
    const fs::path dir = ctx.parent_path().empty() ? fs::path(".") : ctx.parent_path();
    if (!fs::is_directory(dir))
        return;
    const std::string stem = ctx.stem().string();
    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name == ctx.filename().string() || name.rfind(stem + "_", 0) == 0)
            fs::remove(entry.path());
    }
}

std::vector<Input> LoadReferenceInputs(const Options& o, const std::vector<TensorRecord>& index) {
    std::vector<Input> inputs;
    for (const TensorRecord& r : SelectRecords(index, "input", "normalized"))
        inputs.push_back({r.image_id, ReadFloat32(o.reference_dir / r.file, r.ElementCount())});
    if (inputs.empty())
        throw std::runtime_error("no 'input normalized' records in the reference index");
    return inputs;
}

std::vector<std::vector<float>> RunInputs(ModelRunner& runner, RunMode mode,
                                          const std::vector<Input>& inputs) {
    std::vector<std::vector<float>> outputs;
    for (const Input& in : inputs)
        outputs.push_back(runner.RunWith(mode, in.data));
    return outputs;
}

int RunProbe(const Options& o) {
    Report report;
    JsonWriter& j = report.json;
    const fs::path exe_dir = edgebench::executable_dir();
    const fs::path model = fs::absolute(o.model);
    if (!fs::is_regular_file(model)) {
        std::fprintf(stderr, "model not found: %s\n", model.string().c_str());
        return kExitUsage;
    }

    j.BeginObject();
    j.Key("onnxruntime_version").String(Ort::GetVersionString());
    j.Key("options")
        .BeginObject()
        .Key("model")
        .String(model.string())
        .Key("target")
        .String(ToString(o.target))
        .Key("strict")
        .Bool(o.strict)
        .Key("fp16")
        .Bool(o.fp16)
        .Key("threads")
        .Int(o.threads)
        .Key("run_mode")
        .String(ToString(o.run_mode))
        .Key("ep_context")
        .Bool(o.ep_context)
        .Key("context_file")
        .String(o.context_file.empty() ? "" : fs::absolute(o.context_file).string())
        .Key("reuse_context")
        .Bool(o.reuse_context)
        .Key("init_reps")
        .Int(o.init_reps)
        .Key("warmup")
        .Int(o.warmup)
        .Key("runs")
        .Int(o.runs)
        .Key("qnn_profiling")
        .String(ToString(o.qnn_profiling))
        .EndObject();

    std::printf(
        "model:   %s\ntarget:  %s  strict=%s  fp16=%s  threads=%d  run-mode=%s  "
        "ep-context=%s\n",
        model.string().c_str(), ToString(o.target), o.strict ? "on" : "off", o.fp16 ? "on" : "off",
        o.threads, ToString(o.run_mode), o.ep_context ? (o.reuse_context ? "reuse" : "on") : "off");

    Ort::Env env(o.verbose_log ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_ERROR, "ort_probe");

    // ---------------------------------------------------------------- DLL check
    Section("ONNX Runtime DLL");
    const fs::path ort_dll = edgebench::loaded_module_path(L"onnxruntime.dll");
    const fs::path bundled = exe_dir / "onnxruntime.dll";
    const bool ort_dll_ok = SameFile(ort_dll, bundled);
    std::printf("  version:  %s\n  loaded:   %s\n  expected: %s  -> %s\n",
                Ort::GetVersionString().c_str(), ort_dll.string().c_str(), bundled.string().c_str(),
                ort_dll_ok ? "OK" : "MISMATCH");
    j.Key("dll_check")
        .BeginObject()
        .Key("onnxruntime_dll")
        .String(ort_dll.string())
        .Key("expected")
        .String(bundled.string())
        .Key("ok")
        .Bool(ort_dll_ok)
        .Key("working_directory")
        .String(fs::current_path().string());
    if (!ort_dll_ok) {
        j.EndObject().EndObject();
        std::fprintf(stderr, "onnxruntime.dll was not loaded from %s\n", exe_dir.string().c_str());
        return kExitCheckFailed;
    }

    // ---------------------------------------------------------------- devices
    Section("EP devices");
    const fs::path qnn_lib = exe_dir / kQnnEpLibrary;
    env.RegisterExecutionProviderLibrary(kQnnEpName, qnn_lib.native());
    std::printf("  registered %s from %s\n", kQnnEpName, qnn_lib.string().c_str());
    const std::vector<Ort::ConstEpDevice> devices = env.GetEpDevices();
    PrintDevices(devices);
    std::optional<Ort::ConstEpDevice> npu;
    if (o.target == Target::kNpu) {
        try {
            npu = SelectQnnNpu(devices);
        } catch (const DeviceSelectionError& e) {
            std::fprintf(stderr, "device selection failed: %s\n", e.what());
            j.EndObject().EndObject();
            return kExitCheckFailed;
        }
        std::printf("  selected: %s / NPU (device_id=0x%x)\n", npu->EpName(),
                    npu->Device().DeviceId());
    }

    auto make_session = [&](const fs::path& path, bool generate_context) {
        if (o.target == Target::kCpu)
            return Ort::Session(env, path.c_str(), CpuSessionOptions(o.threads));
        Ort::SessionOptions so = NpuSessionOptions(env, *npu, o, exe_dir, generate_context);
        return Ort::Session(env, path.c_str(), so);
    };

    // ---------------------------------------------------------------- inputs
    std::vector<TensorRecord> index;
    std::vector<Input> inputs;
    if (!o.reference_dir.empty()) {
        index = ReadIndex(o.reference_dir / "index.tsv");
        inputs = LoadReferenceInputs(o, index);
    }

    // ---------------------------------------------------------------- sessions
    Section("Sessions");
    std::vector<double> cold_ms, cached_ms;
    std::optional<double> generate_ms;
    std::optional<ModelRunner> runner;
    std::optional<std::vector<std::vector<float>>> uncached_outputs;

    if (!o.reuse_context) {
        for (int rep = 0; rep < o.init_reps; ++rep) {
            runner.reset();
            const auto t0 = Clock::now();
            Ort::Session s = make_session(model, false);
            cold_ms.push_back(MsSince(t0));
            runner.emplace(std::move(s));
        }
        std::printf("  cold session creation (ms):");
        for (double ms : cold_ms)
            std::printf(" %.1f", ms);
        std::printf("  median %.1f\n", Median(cold_ms));
    }
    if (inputs.empty()) {  // generated input, sized from the model
        if (!runner)
            runner.emplace(make_session(o.ep_context ? o.context_file : model, false));
        const size_t n = runner->input_count();
        inputs.push_back({o.fill, o.fill == "pattern" ? PatternFill(n)
                                                      : std::vector<float>(n, std::stof(o.fill))});
    }

    if (o.ep_context) {
        if (!o.reuse_context) {
            uncached_outputs = RunInputs(*runner, o.run_mode, inputs);
            runner.reset();
            fs::create_directories(fs::absolute(o.context_file).parent_path());
            RemoveContextFiles(fs::absolute(o.context_file));
            const auto t0 = Clock::now();
            { Ort::Session gen = make_session(model, true); }
            generate_ms = MsSince(t0);
            std::printf("  EP context generation (ms): %.1f -> %s\n", *generate_ms,
                        o.context_file.string().c_str());
        } else if (!fs::is_regular_file(o.context_file)) {
            std::fprintf(stderr, "context file not found: %s\n", o.context_file.string().c_str());
            return kExitUsage;
        }
        for (int rep = 0; rep < o.init_reps; ++rep) {
            runner.reset();
            const auto t0 = Clock::now();
            Ort::Session s = make_session(o.context_file, false);
            cached_ms.push_back(MsSince(t0));
            runner.emplace(std::move(s));
        }
        std::printf("  cached session creation (ms):");
        for (double ms : cached_ms)
            std::printf(" %.1f", ms);
        std::printf("  median %.1f\n", Median(cached_ms));
    }

    // Snapshot while a session is alive: the HTP backend DLLs are unloaded with it.
    const std::vector<edgebench::LoadedModule> modules = edgebench::list_ort_modules();
    const std::vector<edgebench::LoadedModule> outside =
        edgebench::find_modules_outside(modules, exe_dir);
    const bool modules_bundled = outside.empty();
    std::printf("  loaded ONNX Runtime / QNN modules:\n");
    for (const edgebench::LoadedModule& m : modules) {
        const bool here = std::none_of(outside.begin(), outside.end(),
                                       [&m](const auto& x) { return x.path == m.path; });
        std::printf("    %s%s\n", m.path.string().c_str(),
                    here ? "" : "   <-- NOT next to the exe");
    }
    j.Key("loaded_modules");
    WriteModuleList(j, modules);
    j.Key("all_modules_next_to_exe").Bool(modules_bundled).EndObject();  // dll_check
    report.failed |= !modules_bundled;

    j.Key("devices");
    WriteDevices(j, devices);
    if (npu) {
        j.Key("selected_device_id").Int(npu->Device().DeviceId());
        j.Key("provider_options").BeginObject();
        for (const auto& [k, v] : QnnProviderOptions(o, exe_dir))
            j.Key(k).String(v);
        j.EndObject();
    }
    j.Key("session_config").BeginObject();
    for (const auto& [k, v] : SessionConfig(o, false))
        j.Key(k).String(v);
    j.EndObject();
    j.Key("init").BeginObject().Key("cold_ms");
    WriteNumbers(j, cold_ms);
    j.Key("context_generation_ms");
    generate_ms ? j.Number(*generate_ms) : j.Null();
    j.Key("cached_ms");
    WriteNumbers(j, cached_ms);
    j.EndObject();

    // ---------------------------------------------------------------- outputs
    Section("Outputs");
    const std::vector<std::vector<float>> outputs = RunInputs(*runner, o.run_mode, inputs);
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::printf("  [%zu] %s: top-1 %zu (%.6f)\n", i, inputs[i].id.c_str(), ArgMax(outputs[i]),
                    outputs[i][ArgMax(outputs[i])]);
    }
    if (!o.save_outputs.empty()) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            char name[32];
            std::snprintf(name, sizeof(name), "%02zu.f32", i);
            WriteFloat32(o.save_outputs / name, outputs[i]);
        }
    }

    if (uncached_outputs) {
        double max_diff = 0;
        for (size_t i = 0; i < outputs.size(); ++i)
            max_diff =
                std::max(max_diff, Compare(outputs[i], (*uncached_outputs)[i], 0).max_abs_diff);
        std::printf("  cached vs uncached session: max abs diff %.3g -> %s\n", max_diff,
                    max_diff == 0 ? "identical" : "DIFFERENT");
        j.Key("cached_vs_uncached")
            .BeginObject()
            .Key("max_abs_diff")
            .Number(max_diff)
            .Key("identical")
            .Bool(max_diff == 0)
            .EndObject();
        report.failed |= max_diff != 0;
    }

    if (!o.reference_config.empty()) {
        const double tol = o.tolerance >= 0 ? o.tolerance : DefaultTolerance(o.target);
        const std::vector<TensorRecord> refs = SelectRecords(index, "output", o.reference_config);
        if (refs.size() != outputs.size()) {
            throw std::runtime_error("reference config '" + o.reference_config + "' has " +
                                     std::to_string(refs.size()) + " outputs, expected " +
                                     std::to_string(outputs.size()));
        }
        Section("Comparison with Python reference");
        std::printf("  config %s, tolerance %.3g (max abs diff)\n", o.reference_config.c_str(),
                    tol);
        double worst = 0;
        bool all_ok = true, all_top1 = true;
        j.Key("comparison")
            .BeginObject()
            .Key("reference_config")
            .String(o.reference_config)
            .Key("tolerance")
            .Number(tol)
            .Key("images")
            .BeginArray();
        for (size_t i = 0; i < refs.size(); ++i) {
            const std::vector<float> ref =
                ReadFloat32(o.reference_dir / refs[i].file, refs[i].ElementCount());
            const Comparison c = Compare(outputs[i], ref, tol);
            worst = std::max(worst, c.max_abs_diff);
            all_ok &= c.ok();
            all_top1 &= c.top1_match;
            std::printf("  [%zu] %-45s max abs diff %-10.3g top-1 %zu vs %zu %s\n", i,
                        refs[i].image_id.c_str(), c.max_abs_diff, c.argmax_actual,
                        c.argmax_expected, c.ok() ? "OK" : "FAIL");
            j.BeginObject()
                .Key("image_id")
                .String(refs[i].image_id)
                .Key("max_abs_diff")
                .Number(c.max_abs_diff)
                .Key("top1_cpp")
                .Int(static_cast<int64_t>(c.argmax_actual))
                .Key("top1_python")
                .Int(static_cast<int64_t>(c.argmax_expected))
                .Key("ok")
                .Bool(c.ok())
                .EndObject();
        }
        j.EndArray()
            .Key("max_abs_diff")
            .Number(worst)
            .Key("all_top1_match")
            .Bool(all_top1)
            .Key("pass")
            .Bool(all_ok)
            .EndObject();
        std::printf("  overall: max abs diff %.3g, top-1 %s -> %s\n", worst,
                    all_top1 ? "all match" : "MISMATCH", all_ok ? "PASS" : "FAIL");
        report.failed |= !all_ok;
    }

    if (o.compare_cpu) {
        Section("Comparison with the CPU EP");
        ModelRunner cpu(Ort::Session(env, model.c_str(), CpuSessionOptions(o.threads)));
        const std::vector<std::vector<float>> cpu_out = RunInputs(cpu, RunMode::kRun, inputs);
        bool all_close = true;
        double worst = 0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            const Comparison c = Compare(outputs[i], cpu_out[i], INFINITY);
            const bool close = AllClose(outputs[i], cpu_out[i], kCpuCompareRtol, kCpuCompareAtol);
            worst = std::max(worst, c.max_abs_diff);
            all_close &= close;
        }
        std::printf("  max abs diff %.3g; allclose(rtol=%g, atol=%g): %s\n", worst, kCpuCompareRtol,
                    kCpuCompareAtol, all_close ? "yes" : "NO");
        j.Key("cpu_comparison")
            .BeginObject()
            .Key("max_abs_diff")
            .Number(worst)
            .Key("rtol")
            .Number(kCpuCompareRtol)
            .Key("atol")
            .Number(kCpuCompareAtol)
            .Key("allclose")
            .Bool(all_close)
            .EndObject();
        report.failed |= !all_close;
    }

    // ---------------------------------------------------------------- latency
    if (o.runs > 0) {
        Section("Steady-state latency");
        const std::vector<float>& x = inputs.front().data;
        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(o.runs));
        if (o.run_mode == RunMode::kIoBinding) {
            runner->PrepareBinding(x);
            for (int i = 0; i < o.warmup; ++i)
                runner->RunBound();
            for (int i = 0; i < o.runs; ++i) {
                const auto t0 = Clock::now();
                runner->RunBound();
                samples.push_back(MsSince(t0));
            }
        } else {
            for (int i = 0; i < o.warmup; ++i)
                runner->Run(x);
            for (int i = 0; i < o.runs; ++i) {
                const auto t0 = Clock::now();
                runner->Run(x);
                samples.push_back(MsSince(t0));
            }
        }
        const LatencyStats s = ComputeStats(samples);
        std::printf("  %s, %d warmup + %d runs: mean %.4f  p50 %.4f  p95 %.4f  min %.4f ms\n",
                    ToString(o.run_mode), o.warmup, o.runs, s.mean, s.p50, s.p95, s.min);
        j.Key("latency");
        WriteStats(j, s);
    }

    runner.reset();  // release sessions before unregistering the EP library
    env.UnregisterExecutionProviderLibrary(kQnnEpName);

    j.Key("result").String(report.failed ? "failed" : "ok").EndObject();
    if (!o.json.empty()) {
        if (o.json.has_parent_path())
            fs::create_directories(o.json.parent_path());
        std::ofstream(o.json) << j.str() << "\n";
    }
    std::printf("\n%s\n", report.failed ? "FAILED" : "OK");
    return report.failed ? kExitCheckFailed : kExitOk;
}

void WriteErrorJson(const Options& o, const std::string& code_name, int code,
                    const std::string& message) {
    if (o.json.empty())
        return;
    JsonWriter j;
    j.BeginObject()
        .Key("result")
        .String("error")
        .Key("error")
        .BeginObject()
        .Key("code")
        .Int(code)
        .Key("code_name")
        .String(code_name)
        .Key("message")
        .String(message)
        .EndObject()
        .EndObject();
    if (o.json.has_parent_path())
        fs::create_directories(o.json.parent_path());
    std::ofstream(o.json) << j.str() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    const ParseResult parsed = ParseArgs(args);
    if (parsed.options.help) {
        std::printf("%s", Usage(argv[0]).c_str());
        return kExitOk;
    }
    if (!parsed.error.empty()) {
        std::fprintf(stderr, "error: %s\n\n%s", parsed.error.c_str(), Usage(argv[0]).c_str());
        return kExitUsage;
    }
    const Options& o = parsed.options;
    // Load onnxruntime.dll from next to the executable before any ONNX Runtime call.
    const edgebench::OrtLoadResult load = edgebench::load_onnxruntime();
    if (!load.ok()) {
        const std::string kind(edgebench::to_string(load.error_kind));
        std::fprintf(stderr, "cannot use the bundled ONNX Runtime (%s): %s\n", kind.c_str(),
                     load.message.c_str());
        WriteErrorJson(o, "dll_load", -1, kind + ": " + load.message);
        return kExitCheckFailed;
    }
    try {
        return RunProbe(o);
    } catch (const Ort::Exception& e) {
        const OrtErrorCode code = e.GetOrtErrorCode();
        std::fprintf(stderr, "\nONNX Runtime error %d (%s): %s\n", static_cast<int>(code),
                     OrtErrorCodeName(code), e.what());
        WriteErrorJson(o, OrtErrorCodeName(code), static_cast<int>(code), e.what());
        return kExitOrtError;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        WriteErrorJson(o, "std::exception", -1, e.what());
        return kExitCheckFailed;
    }
}
