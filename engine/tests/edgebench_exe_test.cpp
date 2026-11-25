// Runs edgebench.exe (and ort_probe.exe) as child processes: the command-line
// interface, and `doctor` with a correct, missing, or wrong onnxruntime.dll. Every
// case must end with a documented exit code and an explanation, never a crash.

#include <gtest/gtest.h>

#include <onnxruntime_c_api.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <regex>

#include "edgebench/cli.hpp"
#include "edgebench/exit_codes.hpp"
#include "edgebench/npu_selection.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/qnn_ep.hpp"
#include "edgebench/sha256.hpp"
#include "edgebench/system_info.hpp"
#include "edgebench/version.hpp"
#include "process_util.hpp"
#include "test_runtime.hpp"
#include "validator.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using eb::ExitCode;
using eb::test::ProcessResult;
using eb::test::run_process;
using eb::test::TempDir;
using json = nlohmann::ordered_json;

namespace {

constexpr uint32_t kAccessViolation = 0xC0000005;
constexpr uint32_t kDelayLoadModuleNotFound = 0xC06D007E;
constexpr uint32_t kDelayLoadProcNotFound = 0xC06D007F;

const fs::path kEdgebenchExe = fs::path(EDGEBENCH_EXE_PATH).make_preferred();
const fs::path kPreloadHelper = fs::path(EDGEBENCH_PRELOAD_HELPER_PATH).make_preferred();
const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();
const fs::path kFp32Model = kSourceDir / "models" / "mobilenetv2" / "mobilenetv2_fp32_b1.onnx";
const fs::path kSystemOrtDll = L"C:\\Windows\\System32\\onnxruntime.dll";
const fs::path kSystemNonOrtDll = L"C:\\Windows\\System32\\version.dll";

uint32_t code(ExitCode c) {
    return static_cast<uint32_t>(eb::to_int(c));
}

std::string describe(const ProcessResult& r) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08X", r.exit_code);
    return std::string("exit code ") + buf + (r.timed_out ? " (timed out)" : "") +
           "\n--- stdout:\n" + r.out + "\n--- stderr:\n" + r.err;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

void expect_no_crash(const ProcessResult& r) {
    EXPECT_FALSE(r.timed_out);
    EXPECT_NE(r.exit_code, kAccessViolation) << describe(r);
    EXPECT_NE(r.exit_code, kDelayLoadModuleNotFound) << describe(r);
    EXPECT_NE(r.exit_code, kDelayLoadProcNotFound) << describe(r);
}

// A copy of edgebench.exe alone in a fresh temporary directory (no ONNX Runtime).
struct IsolatedExe {
    TempDir dir;
    fs::path exe = dir.path() / "edgebench.exe";
    IsolatedExe() { fs::copy_file(kEdgebenchExe, exe); }
    // A file that passes --model validation (an .onnx regular file).
    fs::path dummy_model() const {
        const fs::path p = dir.path() / "dummy.onnx";
        std::ofstream(p) << "x";
        return p;
    }
};

// `doctor` could not load ONNX Runtime: exit 4, the error kind on stderr, and the
// sections that do not need ONNX Runtime still on stdout.
void expect_doctor_load_failure(const ProcessResult& r, std::string_view kind) {
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [" + std::string(kind) + "]")) << describe(r);
    EXPECT_TRUE(contains(r.out, "FAILED (" + std::string(kind) + ")")) << describe(r);
    EXPECT_TRUE(contains(r.out, "\nSystem\n")) << describe(r);
    EXPECT_TRUE(contains(r.out, "Result: FAILED (exit 4, ort_load_failure)")) << describe(r);
}

}  // namespace

// --------------------------------------------------------------------------- doctor

TEST(Doctor, HappyPathReportsBundledOnnxRuntime) {
    const ProcessResult r = run_process(kEdgebenchExe, {L"doctor", L"--load-sample-ms", L"0"},
                                        kEdgebenchExe.parent_path());
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    // Warnings (e.g. process_emulated for an x64 build on ARM64 Windows) appear once,
    // in the report on stdout; stderr carries only problems, and there are none.
    EXPECT_TRUE(r.err.empty()) << describe(r);
    if (eb::query_system_info().emulated())
        EXPECT_TRUE(contains(r.out, "warning: [process_emulated]")) << describe(r);
    const std::string expected(eb::expected_onnxruntime_version());
    EXPECT_TRUE(contains(r.out, expected + " (C API " + std::to_string(ORT_API_VERSION) + ")"))
        << r.out;
    EXPECT_TRUE(contains(r.out, "full_path_verified")) << r.out;
    EXPECT_TRUE(contains(r.out, eb::path_to_utf8(kEdgebenchExe.parent_path() / "onnxruntime.dll")))
        << r.out;
    EXPECT_TRUE(contains(r.out, "CPUExecutionProvider")) << r.out;
    EXPECT_TRUE(contains(r.out, "Result: OK")) << r.out;
    EXPECT_FALSE(contains(r.out, "OUTSIDE")) << r.out;
}

TEST(Doctor, JsonOutputParses) {
    const ProcessResult r =
        run_process(kEdgebenchExe, {L"doctor", L"--json", L"--load-sample-ms", L"100"},
                    kEdgebenchExe.parent_path());
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const json j = json::parse(r.out);
    EXPECT_EQ(j.at("result"), "ok");
    EXPECT_EQ(j.at("exit_code"), 0);
    EXPECT_EQ(j.at("tool").at("version").get<std::string>(), eb::version());
    EXPECT_EQ(j.at("tool").at("implementation"), "cpp");
    EXPECT_EQ(j.at("results_schema_version").get<std::string>(), eb::results_schema_version());
    const json& ort = j.at("runtime").at("onnxruntime");
    EXPECT_EQ(ort.at("version").get<std::string>(), eb::expected_onnxruntime_version());
    EXPECT_EQ(ort.at("api_version"), ORT_API_VERSION);
    EXPECT_EQ(ort.at("load_check"), "full_path_verified");
    for (const json& m : j.at("runtime").at("loaded_modules"))
        EXPECT_TRUE(m.at("inside_bundle").get<bool>()) << m.dump();
    bool cpu_ep = false;
    for (const json& d : j.at("runtime").at("ep_devices"))
        cpu_ep |= d.at("ep_name") == "CPUExecutionProvider" && d.at("device_type") == "CPU";
    EXPECT_TRUE(cpu_ep) << j.at("runtime").at("ep_devices").dump();
    const bool emulated = eb::query_system_info().emulated();
    EXPECT_EQ(
        j.at("system").at("process_architecture") == j.at("system").at("os").at("architecture"),
        !emulated);
    EXPECT_EQ(j.at("process_emulated"), emulated);
    const json& power = j.at("system").at("power");
    for (const char* key :
         {"source", "power_mode", "power_plan", "battery_percent", "battery_saver"})
        EXPECT_TRUE(power.contains(key)) << key;
    EXPECT_FALSE(power.at("power_mode").is_null());
    EXPECT_FALSE(power.at("power_plan").is_null());
    const json& load = j.at("system").at("load");
    EXPECT_NEAR(load.at("sample_window_s").get<double>(), 0.1, 0.05);
    EXPECT_GE(load.at("cpu_utilization_ratio").get<double>(), 0.0);
    EXPECT_FALSE(j.at("system").at("os").at("name").get<std::string>().empty());
}

TEST(Doctor, FromAnotherWorkingDirectory) {
    TempDir cwd;
    const ProcessResult r =
        run_process(kEdgebenchExe, {L"doctor", L"--load-sample-ms", L"0"}, cwd.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, 0u) << describe(r);
}

TEST(Doctor, MissingDllIsReported) {
    IsolatedExe iso;
    const ProcessResult r =
        run_process(iso.exe, {L"doctor", L"--load-sample-ms", L"0"}, iso.dir.path());
    expect_doctor_load_failure(r, "dll_not_found");
    // The executable reports its own directory in long form, whatever form it was started with.
    const std::string dll = eb::path_to_utf8(eb::resolve_path(iso.dir.path()) / "onnxruntime.dll");
    EXPECT_TRUE(contains(r.err, dll + " not found")) << describe(r);
}

TEST(Doctor, MissingDllJson) {
    IsolatedExe iso;
    const ProcessResult r =
        run_process(iso.exe, {L"doctor", L"--json", L"--load-sample-ms", L"0"}, iso.dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    const json j = json::parse(r.out);
    EXPECT_EQ(j.at("result"), "failed");
    EXPECT_EQ(j.at("onnxruntime_load").at("error_kind"), "dll_not_found");
    EXPECT_TRUE(j.at("runtime").at("onnxruntime").is_null());
    EXPECT_TRUE(j.at("runtime").at("ep_devices").is_null());
    EXPECT_FALSE(j.at("system").at("cpu").at("name").is_null());  // still reported
}

TEST(Doctor, MissingDllIgnoresWorkingDirectoryAndPath) {
    // The bundle directory holds a valid onnxruntime.dll; it must not be picked up
    // through the current directory or PATH.
    IsolatedExe iso;
    const fs::path bundle = kEdgebenchExe.parent_path();
    const ProcessResult r =
        run_process(iso.exe, {L"doctor", L"--load-sample-ms", L"0"}, bundle, bundle);
    expect_doctor_load_failure(r, "dll_not_found");
}

TEST(Doctor, WrongDllVersionIsReported) {
    if (!fs::is_regular_file(kSystemOrtDll))
        GTEST_SKIP() << kSystemOrtDll << " is not present";
    IsolatedExe iso;
    fs::copy_file(kSystemOrtDll, iso.dir.path() / "onnxruntime.dll");
    const ProcessResult r =
        run_process(iso.exe, {L"doctor", L"--load-sample-ms", L"0"}, iso.dir.path());
    expect_doctor_load_failure(r, "version_mismatch");

    // The message names the version that was found and the one required.
    std::smatch m;
    ASSERT_TRUE(std::regex_search(r.err, m, std::regex(R"(is ONNX Runtime (\d+\.\d+\.\d+), )")))
        << describe(r);
    RecordProperty("found_version", m[1].str());
    EXPECT_NE(m[1].str(), std::string(eb::expected_onnxruntime_version()));
    EXPECT_TRUE(contains(r.out, "found version      " + m[1].str())) << r.out;
    EXPECT_TRUE(
        contains(r.err, "requires ONNX Runtime " + std::string(eb::expected_onnxruntime_version())))
        << describe(r);
}

TEST(Doctor, DllWithoutOrtEntryPointIsReported) {
    if (!fs::is_regular_file(kSystemNonOrtDll))
        GTEST_SKIP() << kSystemNonOrtDll << " is not present";
    IsolatedExe iso;
    fs::copy_file(kSystemNonOrtDll, iso.dir.path() / "onnxruntime.dll");
    const ProcessResult r =
        run_process(iso.exe, {L"doctor", L"--load-sample-ms", L"0"}, iso.dir.path());
    expect_doctor_load_failure(r, "entry_point_missing");
    EXPECT_TRUE(contains(r.err, "does not export OrtGetApiBase")) << describe(r);
}

TEST(Doctor, CorruptDllIsReported) {
    IsolatedExe iso;
    std::ofstream(iso.dir.path() / "onnxruntime.dll") << "this is not a DLL";
    const ProcessResult r =
        run_process(iso.exe, {L"doctor", L"--load-sample-ms", L"0"}, iso.dir.path());
    expect_doctor_load_failure(r, "load_failed");
    EXPECT_TRUE(contains(r.err, "LoadLibraryExW(")) << describe(r);
}

// --------------------------------------------------------------------------- doctor: QNN EP

namespace {

// A copy of an executable with the ONNX Runtime DLLs but without the QNN EP, in a
// fresh temporary directory; add_qnn_ep_file() puts something in the plugin's place.
struct CpuOnlyBundle {
    TempDir dir;
    fs::path exe;
    explicit CpuOnlyBundle(const fs::path& built) {
        for (const fs::path name : {built.filename(), fs::path(L"onnxruntime.dll"),
                                    fs::path(L"onnxruntime_providers_shared.dll")})
            fs::copy_file(built.parent_path() / name, dir.path() / name);
        exe = dir.path() / built.filename();
    }
    void add_qnn_ep_file(const fs::path& source) const {
        fs::copy_file(source, dir.path() / "onnxruntime_providers_qnn.dll");
    }
};

}  // namespace

TEST(Doctor, ReportsTheQnnExecutionProviderAndTheNpuChoice) {
    const ProcessResult r =
        run_process(kEdgebenchExe, {L"doctor", L"--json", L"--load-sample-ms", L"0"},
                    kEdgebenchExe.parent_path());
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const json j = json::parse(r.out);
    const json& q = j.at("qnn_ep");
    EXPECT_EQ(q.at("library_path").get<std::string>(),
              eb::path_to_utf8(kEdgebenchExe.parent_path() / "onnxruntime_providers_qnn.dll"));
    EXPECT_EQ(q.at("in_build"), eb::qnn_ep_in_build());
    if (!eb::qnn_ep_in_build()) {
        EXPECT_EQ(q.at("registration"), "not_bundled");
        EXPECT_TRUE(q.at("npu_selection").is_null());
        return;
    }
    EXPECT_EQ(q.at("bundled"), true);
    ASSERT_EQ(q.at("registration"), "ok") << q.dump(2);
    EXPECT_TRUE(q.at("error_kind").is_null());
    // The plugin is listed among the loaded modules, inside the bundle.
    bool plugin_listed = false;
    for (const json& m : j.at("runtime").at("loaded_modules")) {
        EXPECT_TRUE(m.at("inside_bundle").get<bool>()) << m.dump();
        plugin_listed |= m.at("name") == "onnxruntime_providers_qnn.dll";
    }
    EXPECT_TRUE(plugin_listed) << j.at("runtime").at("loaded_modules").dump();
    // Either an NPU is chosen, or the reason is a warning (not a problem).
    const json& npu = q.at("npu_selection");
    ASSERT_TRUE(npu.is_object()) << q.dump(2);
    if (npu.at("status") == "selected") {
        EXPECT_EQ(npu.at("candidate_count"), 1);
        EXPECT_EQ(npu.at("device").at("ep_name"), "QNNExecutionProvider");
        EXPECT_EQ(npu.at("device").at("device_type"), "NPU");
        const json& listed = j.at("runtime").at("ep_devices").at(npu.at("device_index").get<int>());
        EXPECT_EQ(listed, npu.at("device"));
    } else {
        bool warned = false;
        for (const json& w : j.at("warnings"))
            warned |= w.at("code") == npu.at("failure");
        EXPECT_TRUE(warned) << j.at("warnings").dump();
        EXPECT_TRUE(j.at("problems").empty()) << j.at("problems").dump();
    }

    const ProcessResult text = run_process(kEdgebenchExe, {L"doctor", L"--load-sample-ms", L"0"},
                                           kEdgebenchExe.parent_path());
    EXPECT_TRUE(contains(text.out, "\nQNN execution provider\n")) << text.out;
    EXPECT_TRUE(contains(text.out, "registration       ok (as QNNExecutionProvider)")) << text.out;
    EXPECT_TRUE(contains(text.out, "NPU for runs")) << text.out;
    EXPECT_TRUE(contains(text.out, "EP devices (CPU EP and QNN EP)")) << text.out;
}

TEST(Doctor, MissingQnnEpFailsOnlyBuildsThatBundleIt) {
    const CpuOnlyBundle bundle(kEdgebenchExe);
    const ProcessResult r =
        run_process(bundle.exe, {L"doctor", L"--load-sample-ms", L"0"}, bundle.dir.path());
    expect_no_crash(r);
    if (!eb::qnn_ep_in_build()) {
        EXPECT_EQ(r.exit_code, 0u) << describe(r);
        EXPECT_TRUE(contains(r.out, "not bundled (CPU-only build)")) << describe(r);
        return;
    }
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [qnn_ep_not_bundled] the QNN execution provider "))
        << describe(r);
    EXPECT_TRUE(contains(r.err, "hint: ")) << describe(r);
    EXPECT_TRUE(contains(r.out, "registration       FAILED (qnn_ep_not_bundled)")) << describe(r);
    EXPECT_TRUE(contains(r.out, "Result: FAILED (exit 4, ort_load_failure)")) << describe(r);
    // The CPU EP is still enumerated.
    EXPECT_TRUE(contains(r.out, "CPUExecutionProvider")) << describe(r);
}

TEST(Doctor, QnnEpThatIsNotAnEpLibraryIsReported) {
    const CpuOnlyBundle bundle(kEdgebenchExe);
    bundle.add_qnn_ep_file(kSystemNonOrtDll);
    const ProcessResult r =
        run_process(bundle.exe, {L"doctor", L"--load-sample-ms", L"0"}, bundle.dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err,
                         "error: [ep_registration_failed] registering the execution "
                         "provider library "))
        << describe(r);
    EXPECT_TRUE(contains(r.out, "registration       FAILED (ep_registration_failed)"))
        << describe(r);
}

// --------------------------------------------------------------------------- CLI

TEST(Cli, HelpAndVersionWorkWithoutOnnxRuntime) {
    IsolatedExe iso;  // no DLL next to the copy
    const std::vector<std::vector<std::wstring>> cases = {
        {L"--help"}, {L"--version"}, {L"run", L"--help"}, {L"doctor", L"--help"}};
    for (const auto& args : cases) {
        const ProcessResult r = run_process(iso.exe, args, iso.dir.path());
        expect_no_crash(r);
        EXPECT_EQ(r.exit_code, 0u) << describe(r);
        EXPECT_FALSE(r.out.empty()) << describe(r);
        EXPECT_TRUE(r.err.empty()) << describe(r);
    }
    const ProcessResult v = run_process(iso.exe, {L"--version"}, iso.dir.path());
    EXPECT_EQ(v.out, eb::version_text());
}

TEST(Cli, UsageErrorsExit2OnStderr) {
    IsolatedExe iso;
    const std::vector<std::vector<std::wstring>> cases = {
        {},
        {L"bench"},
        {L"--bogus"},
        {L"run"},
        {L"run", L"--model", L"missing.onnx"},
        {L"run", L"--model", iso.dummy_model().native(), L"--provider", L"npu"},
        {L"run", L"--model", iso.dummy_model().native(), L"--threads", L"0"},
        {L"run", L"--model", iso.dummy_model().native(), L"--input", L"constant", L"--seed", L"1"},
        {L"doctor", L"--bogus"},
    };
    for (const auto& args : cases) {
        const ProcessResult r = run_process(iso.exe, args, iso.dir.path());
        expect_no_crash(r);
        EXPECT_EQ(r.exit_code, code(ExitCode::usage_error)) << describe(r);
        EXPECT_TRUE(r.out.empty()) << describe(r);
        EXPECT_FALSE(r.err.empty()) << describe(r);
    }
    const ProcessResult none = run_process(iso.exe, {}, iso.dir.path());
    EXPECT_TRUE(contains(none.err, "doctor")) << describe(none);  // help is printed
    const ProcessResult npu = run_process(
        iso.exe, {L"run", L"--model", iso.dummy_model().native(), L"--provider", L"npu"},
        iso.dir.path());
    EXPECT_TRUE(contains(npu.err, "supported providers: cpu")) << describe(npu);
}

TEST(CliModels, RunDryRunPrintsResolvedConfig) {
    if (!fs::is_regular_file(kFp32Model))
        GTEST_SKIP() << kFp32Model << " is not present (see docs/setup.md)";
    const ProcessResult r = run_process(
        kEdgebenchExe, {L"run", L"--model", kFp32Model.native(), L"--dry-run"}, kSourceDir);
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    EXPECT_TRUE(r.err.empty()) << describe(r);
    const json j = json::parse(r.out);
    EXPECT_EQ(j.at("dry_run"), true);
    EXPECT_EQ(j.at("model").at("file_name"), "mobilenetv2_fp32_b1.onnx");
    EXPECT_EQ(j.at("model").at("size_bytes"), fs::file_size(kFp32Model));
    EXPECT_EQ(j.at("cooldown_ms"), 2000);
    EXPECT_EQ(fs::path(j.at("output_dir").get<std::string>()),
              kSourceDir / "benchmarks" / "results" / "local");
    const json& c = j.at("config");
    EXPECT_EQ(c.at("requested").at("execution_provider"), "CPUExecutionProvider");
    EXPECT_EQ(c.at("intra_op_threads"), 4);
    EXPECT_EQ(c.at("run_binding"), "iobinding_preallocated");
    EXPECT_EQ(c.at("warmup_iterations"), 20);
    EXPECT_EQ(c.at("measured_iterations"), 200);
    EXPECT_EQ(c.at("trials"), 3);
    EXPECT_EQ(
        c.at("input"),
        (json{{"kind", "random"}, {"constant_value", nullptr}, {"seed", 0}, {"dataset", nullptr}}));
    EXPECT_EQ(c.at("graph_optimization_level"), "enable_all");
    EXPECT_EQ(c.at("batch_size"), 1);
    EXPECT_EQ(c.at("session_config"), (json{{"session.intra_op.allow_spinning", "1"}}));
    EXPECT_EQ(j.at("runtime").at("onnxruntime").at("load_check"), "full_path_verified");

    // Model facts: the digest equals an independent SHA-256 of the file.
    EXPECT_EQ(j.at("model").at("sha256").get<std::string>(), eb::sha256_file(kFp32Model).value());
    EXPECT_EQ(
        j.at("model").at("inputs"),
        (json::array({{{"name", "input"}, {"dtype", "float32"}, {"shape", {1, 3, 224, 224}}}})));
    EXPECT_EQ(j.at("model").at("outputs"),
              (json::array({{{"name", "logits"}, {"dtype", "float32"}, {"shape", {1, 1000}}}})));

    // Provider and placement: every node on the CPU EP.
    EXPECT_EQ(j.at("runtime").at("execution_provider").at("name"), "CPUExecutionProvider");
    EXPECT_EQ(j.at("runtime").at("ep_device").at("selection").at("candidate_count"), 1);
    const json& ex = j.at("execution");
    EXPECT_EQ(ex.at("full_device_execution_verified"), true);
    EXPECT_EQ(ex.at("verification_method"), "node_assignment");
    const json& na = ex.at("node_assignment");
    ASSERT_EQ(na.at("nodes_by_ep").size(), 1u) << na.dump();
    EXPECT_GT(na.at("total_nodes").get<int>(), 0);
    EXPECT_EQ(na.at("nodes_by_ep").at("CPUExecutionProvider"), na.at("total_nodes"));
    EXPECT_EQ(j.at("dry_run_check").at("outputs_finite"), true);
    EXPECT_TRUE(j.at("warnings").empty());
}

TEST(Cli, RunDryRunRejectsDynamicAndInt64Models) {
    const fs::path fixtures = kSourceDir / "engine" / "tests" / "fixtures" / "models";
    const std::pair<const wchar_t*, const char*> cases[] = {
        {L"dynamic_batch.onnx", "input 'x' has a dynamic dimension 0 ('batch')"},
        {L"int64_input.onnx", "input 'ids' has dtype int64"},
    };
    for (const auto& [file, message] : cases) {
        const ProcessResult r =
            run_process(kEdgebenchExe,
                        {L"run", L"--model", (fixtures / file).native(), L"--dry-run"}, kSourceDir);
        expect_no_crash(r);
        EXPECT_EQ(r.exit_code, code(ExitCode::usage_error)) << describe(r);
        EXPECT_TRUE(contains(r.err, message)) << describe(r);
        EXPECT_TRUE(r.out.empty()) << describe(r);
    }
}

TEST(Cli, RunDryRunInvalidModelExits3WithOrtCode) {
    TempDir dir;
    const fs::path bad = dir.path() / "garbage.onnx";
    std::ofstream(bad) << "this is not an ONNX model";
    const ProcessResult r =
        run_process(kEdgebenchExe, {L"run", L"--model", bad.native(), L"--dry-run"}, dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_runtime_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "ONNX Runtime error ORT_")) << describe(r);
}

TEST(Cli, RunDryRunNeedsOnnxRuntime) {
    IsolatedExe iso;
    const ProcessResult r = run_process(
        iso.exe, {L"run", L"--model", iso.dummy_model().native(), L"--dry-run"}, iso.dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [dll_not_found] cannot load ONNX Runtime: "))
        << describe(r);
}

// A short run of the fixture model into `out` (no cooldown, no load sample).
std::vector<std::wstring> short_run_args(const fs::path& model, const fs::path& out) {
    return {L"run", L"--model",      model.native(), L"--trials",      L"2", L"--warmup",
            L"3",   L"--iterations", L"10",          L"--cooldown-ms", L"0", L"--load-sample-ms",
            L"0",   L"--output-dir", out.native()};
}

const fs::path kFixtureDir = kSourceDir / "engine" / "tests" / "fixtures" / "models";

TEST(Cli, RunBenchmarksTheFixture) {
    TempDir out;
    std::vector<std::wstring> args = short_run_args(kFixtureDir / "fixed_float32.onnx", out.path());
    const ProcessResult r = run_process(kEdgebenchExe, args, kSourceDir);
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    // Summary on stdout, progress on stderr.
    EXPECT_TRUE(contains(r.out, "\naggregate ")) << describe(r);
    EXPECT_TRUE(contains(r.out, "\ntrial 1 ")) << describe(r);
    EXPECT_TRUE(contains(r.out, "\ntrial 2 ")) << describe(r);
    EXPECT_TRUE(contains(r.out, "2 x 10 measured iterations")) << describe(r);
    EXPECT_TRUE(contains(r.out, "CPUExecutionProvider")) << describe(r);
    EXPECT_FALSE(contains(r.out, "trial 1 of 2: creating session")) << describe(r);
    EXPECT_TRUE(contains(r.err, "trial 1 of 2")) << describe(r);
    EXPECT_TRUE(contains(r.err, "trial 2 of 2")) << describe(r);
    EXPECT_TRUE(contains(r.err, "recording node placement")) << describe(r);
    EXPECT_FALSE(contains(r.err, "idle ")) << describe(r);  // --load-sample-ms 0, cooldown 0
    EXPECT_TRUE(contains(r.out, "\nPower       ")) << describe(r);
    EXPECT_TRUE(contains(r.out, "\nMemory      working set ")) << describe(r);

    // The last summary line names the result file, which exists with its companions.
    const std::string last = r.out.substr(r.out.rfind("\nResults     ") + 13);
    const fs::path result = fs::path(std::u8string(last.begin(), last.end() - 1));
    EXPECT_TRUE(fs::is_regular_file(result)) << last;
    EXPECT_EQ(result.parent_path(), out.path());
    const std::string stem = result.stem().string();
    EXPECT_EQ(stem.rfind("fixed_float32_cpu_iobinding_t4_", 0), 0u) << stem;
    EXPECT_TRUE(fs::is_regular_file(out.path() / stem / "latencies.csv"));
    EXPECT_TRUE(fs::is_regular_file(out.path() / stem / "summary.csv"));
    EXPECT_TRUE(fs::is_regular_file(out.path() / "fixed_float32_cpu_iobinding_t4_latest.json"));
}

TEST(Cli, RunFormatJsonPrintsTheDocument) {
    TempDir out;
    std::vector<std::wstring> args = short_run_args(kFixtureDir / "fixed_float32.onnx", out.path());
    args.insert(args.end(), {L"--format", L"json", L"--name", L"as-json"});
    const ProcessResult r = run_process(kEdgebenchExe, args, kSourceDir);
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const json doc = json::parse(r.out);  // stdout is exactly the document
    EXPECT_EQ(doc.at("status"), "success");
    EXPECT_EQ(doc.at("schema_version"), "1.1.0");
    EXPECT_TRUE(contains(r.err, "result written to ")) << describe(r);
    EXPECT_TRUE(fs::is_regular_file(out.path() / "as-json_latest.json"));
    EXPECT_EQ(json::parse(std::ifstream(out.path() / "as-json_latest.json")), doc);
}

TEST(Cli, RunStopsOnNonFiniteOutputAndWritesAFailedResult) {
    TempDir out;
    std::vector<std::wstring> args = short_run_args(kFixtureDir / "nan_output.onnx", out.path());
    const ProcessResult r = run_process(kEdgebenchExe, args, kSourceDir);
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::internal_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "trial 1 of 2: output 'y' has a non-finite value")) << describe(r);
    EXPECT_TRUE(contains(r.err, "failed result written to ")) << describe(r);
    EXPECT_TRUE(r.out.empty()) << describe(r);
    const fs::path latest = out.path() / "nan_output_cpu_iobinding_t4_latest.json";
    ASSERT_TRUE(fs::is_regular_file(latest)) << describe(r);
    const json doc = json::parse(std::ifstream(latest));
    EXPECT_EQ(doc.at("status"), "failed");
    EXPECT_EQ(doc.at("error").at("stage"), "measurement");
    eb::test::validate_or_skip(latest);
}

TEST(Cli, RunUsageErrorFromTheModelWritesNothing) {
    TempDir out;
    std::vector<std::wstring> args = short_run_args(kFixtureDir / "dynamic_batch.onnx", out.path());
    const ProcessResult r = run_process(kEdgebenchExe, args, kSourceDir);
    EXPECT_EQ(r.exit_code, code(ExitCode::usage_error)) << describe(r);
    EXPECT_TRUE(fs::is_empty(out.path())) << describe(r);
}

TEST(Cli, RunWithAnUnwritableOutputDirectoryExits6AfterTheSummary) {
    TempDir dir;
    std::ofstream(dir.path() / "a_file") << "x";  // a file where a directory is needed
    std::vector<std::wstring> args =
        short_run_args(kFixtureDir / "fixed_float32.onnx", dir.path() / "a_file" / "results");
    const ProcessResult r = run_process(kEdgebenchExe, args, kSourceDir);
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::output_error)) << describe(r);
    EXPECT_TRUE(contains(r.out, "\naggregate ")) << describe(r);  // measurements not lost
    EXPECT_FALSE(contains(r.out, "Results     ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [write_failed] cannot write the result: ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "a_file")) << describe(r);
}

TEST(Cli, RunLoadFailureWritesAFailedResult) {
    IsolatedExe iso;  // no ONNX Runtime next to the copy
    const fs::path out = iso.dir.path() / "results";
    const fs::path model = iso.dir.path() / "fixture.onnx";
    fs::copy_file(kFixtureDir / "fixed_float32.onnx", model);
    std::vector<std::wstring> args = short_run_args(model, out);
    const ProcessResult r = run_process(iso.exe, args, iso.dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [dll_not_found] cannot load ONNX Runtime: "))
        << describe(r);
    const json doc = json::parse(std::ifstream(out / "fixture_cpu_iobinding_t4_latest.json"));
    EXPECT_EQ(doc.at("status"), "failed");
    EXPECT_EQ(doc.at("error").at("stage"), "environment");
    EXPECT_EQ(doc.at("error").at("code"), "dll_not_found");
    EXPECT_TRUE(doc.at("runtime").is_null());
}

#ifdef EDGEBENCH_ORT_PROBE_PATH
TEST(OrtProbe, LoadFailureExits4) {
    TempDir dir;
    const fs::path probe = dir.path() / "ort_probe.exe";
    fs::copy_file(fs::path(EDGEBENCH_ORT_PROBE_PATH).make_preferred(), probe);
    const ProcessResult r = run_process(probe, {L"--model", L"any.onnx"}, dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [dll_not_found] cannot load ONNX Runtime: "))
        << describe(r);
}

TEST(OrtProbe, JsonRefusesToOverwrite) {
    TempDir dir;
    const fs::path existing = dir.path() / "summary.json";
    std::ofstream(existing) << "earlier";
    const ProcessResult r = run_process(fs::path(EDGEBENCH_ORT_PROBE_PATH).make_preferred(),
                                        {L"--model", (kFixtureDir / "fixed_float32.onnx").native(),
                                         L"--target", L"cpu", L"--json", existing.native()},
                                        dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::output_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "already exists; refusing to overwrite")) << describe(r);
    std::ifstream in(existing);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(content, "earlier");
}

TEST(OrtProbe, JsonWritesANewFile) {
    TempDir dir;
    const fs::path target = dir.path() / "new.json";
    const ProcessResult r = run_process(fs::path(EDGEBENCH_ORT_PROBE_PATH).make_preferred(),
                                        {L"--model", (kFixtureDir / "fixed_float32.onnx").native(),
                                         L"--target", L"cpu", L"--json", target.native()},
                                        dir.path());
    EXPECT_EQ(r.exit_code, 0u) << describe(r);
    const json doc = json::parse(std::ifstream(target));
    EXPECT_EQ(doc.at("result"), "ok");
}

// A bundle without the QNN execution provider (as in x64 builds): the CPU target
// works, the NPU target fails with an explanation instead of an ONNX Runtime error.
TEST(OrtProbe, WithoutTheQnnEpOnlyTheCpuTargetWorks) {
    TempDir dir;
    const fs::path built = fs::path(EDGEBENCH_ORT_PROBE_PATH).make_preferred();
    for (const wchar_t* name :
         {L"ort_probe.exe", L"onnxruntime.dll", L"onnxruntime_providers_shared.dll"})
        fs::copy_file(built.parent_path() / name, dir.path() / name);
    const fs::path probe = dir.path() / "ort_probe.exe";
    const std::wstring model = (kFixtureDir / "fixed_float32.onnx").native();

    const ProcessResult cpu =
        run_process(probe, {L"--model", model, L"--target", L"cpu"}, dir.path());
    expect_no_crash(cpu);
    EXPECT_EQ(cpu.exit_code, 0u) << describe(cpu);
    EXPECT_TRUE(contains(cpu.out, "not bundled (CPU-only build)")) << describe(cpu);

    const ProcessResult npu =
        run_process(probe, {L"--model", model, L"--target", L"npu"}, dir.path());
    expect_no_crash(npu);
    EXPECT_EQ(npu.exit_code, code(ExitCode::ort_load_failure)) << describe(npu);
    EXPECT_TRUE(contains(npu.err,
                         "error: [qnn_ep_not_bundled] --target npu needs the QNN "
                         "execution provider"))
        << describe(npu);
    EXPECT_TRUE(contains(npu.err, "hint: The QNN execution provider is bundled with ARM64 builds"))
        << describe(npu);
}

TEST(OrtProbe, QnnEpThatIsNotAnEpLibraryExits4) {
    const CpuOnlyBundle bundle(fs::path(EDGEBENCH_ORT_PROBE_PATH).make_preferred());
    bundle.add_qnn_ep_file(kSystemNonOrtDll);
    const ProcessResult r = run_process(
        bundle.exe,
        {L"--model", (kFixtureDir / "fixed_float32.onnx").native(), L"--target", L"cpu"},
        bundle.dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [ep_registration_failed] ")) << describe(r);
}

// Device selection failures exit with 3 and list the devices. Only reachable on a
// machine whose QNN EP offers no single NPU (e.g. an ARM64 CI runner); elsewhere
// the selection rule is covered by NpuSelection.* on synthetic device lists.
TEST(OrtProbe, NpuTargetWithoutASingleNpuExits3) {
    if (!eb::qnn_ep_in_build())
        GTEST_SKIP() << "this build does not include the QNN execution provider";
    const eb::test::QnnNpuProbe probe = eb::test::probe_qnn_npu();
    ASSERT_FALSE(probe.registration_error) << probe.registration_error->message;
    ASSERT_TRUE(probe.selection);
    if (probe.selection->ok()) {
        GTEST_SKIP() << "this machine has exactly one QNN NPU ("
                     << eb::describe_device(probe.devices[*probe.selection->index]) << ")";
    }
    const std::string kind(eb::to_string(*probe.selection->failure));
    const ProcessResult r = run_process(
        fs::path(EDGEBENCH_ORT_PROBE_PATH).make_preferred(),
        {L"--model", (kFixtureDir / "fixed_float32.onnx").native(), L"--target", L"npu"},
        kEdgebenchExe.parent_path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_runtime_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [" + kind + "] ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "enumerated devices: ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "hint: ")) << describe(r);
}
#endif

// --------------------------------------------------------------------------- doctor on the NPU

TEST(DoctorNpu, SelectsTheQnnNpu) {
    const eb::test::QnnNpuProbe probe = eb::test::probe_qnn_npu();
    if (const std::optional<std::string> reason = eb::test::npu_skip_reason(probe))
        GTEST_SKIP() << *reason;
    const ProcessResult r =
        run_process(kEdgebenchExe, {L"doctor", L"--json", L"--load-sample-ms", L"0"},
                    kEdgebenchExe.parent_path());
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const json j = json::parse(r.out);
    const json& npu = j.at("qnn_ep").at("npu_selection");
    ASSERT_TRUE(npu.is_object()) << j.at("qnn_ep").dump(2);
    ASSERT_EQ(npu.at("status"), "selected") << npu.dump(2);
    EXPECT_EQ(npu.at("device").at("device_type"), "NPU");
    EXPECT_EQ(npu.at("device").at("ep_name"), "QNNExecutionProvider");
    EXPECT_EQ(npu.at("device").at("device_id").get<std::uint32_t>(),
              probe.devices[*probe.selection->index].device_id);
    for (const json& w : j.at("warnings")) {
        for (const char* kind : {"no_qnn_ep_devices", "no_npu_device", "multiple_npu_devices"})
            EXPECT_NE(w.at("code"), kind) << w.dump();
    }
}

// --------------------------------------------------------------------------- preload

TEST(PreloadHelper, ForeignOnnxRuntimeLoadedFirstIsReported) {
    if (!fs::is_regular_file(kSystemOrtDll))
        GTEST_SKIP() << kSystemOrtDll << " is not present";
    const ProcessResult r =
        run_process(kPreloadHelper, {kSystemOrtDll.native()}, kPreloadHelper.parent_path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.output, "load_check: foreign_module_loaded")) << describe(r);
    EXPECT_TRUE(contains(r.output, "already loaded")) << describe(r);
}

TEST(PreloadHelper, BundledOnnxRuntimeLoadedFirstIsAccepted) {
    const fs::path bundled = kPreloadHelper.parent_path() / "onnxruntime.dll";
    const ProcessResult r = run_process(kPreloadHelper, {bundled.native()}, bundled.parent_path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, 0u) << describe(r);
    EXPECT_TRUE(contains(r.output, "load_check: full_path_verified")) << describe(r);
}
