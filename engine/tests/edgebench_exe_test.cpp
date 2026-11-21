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
#include "edgebench/ort_loader.hpp"
#include "edgebench/sha256.hpp"
#include "edgebench/version.hpp"
#include "process_util.hpp"

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
    const ProcessResult r = run_process(kEdgebenchExe, {L"doctor"}, kEdgebenchExe.parent_path());
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    EXPECT_TRUE(r.err.empty()) << describe(r);
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
        run_process(kEdgebenchExe, {L"doctor", L"--json"}, kEdgebenchExe.parent_path());
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
    EXPECT_EQ(j.at("system").at("process_architecture"),
              j.at("system").at("os").at("architecture"));
    EXPECT_EQ(j.at("process_emulated"), false);
    EXPECT_FALSE(j.at("system").at("os").at("name").get<std::string>().empty());
}

TEST(Doctor, FromAnotherWorkingDirectory) {
    TempDir cwd;
    const ProcessResult r = run_process(kEdgebenchExe, {L"doctor"}, cwd.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, 0u) << describe(r);
}

TEST(Doctor, MissingDllIsReported) {
    IsolatedExe iso;
    const ProcessResult r = run_process(iso.exe, {L"doctor"}, iso.dir.path());
    expect_doctor_load_failure(r, "dll_not_found");
    const std::string dll = eb::path_to_utf8(iso.dir.path() / "onnxruntime.dll");
    EXPECT_TRUE(contains(r.err, dll + " not found")) << describe(r);
}

TEST(Doctor, MissingDllJson) {
    IsolatedExe iso;
    const ProcessResult r = run_process(iso.exe, {L"doctor", L"--json"}, iso.dir.path());
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
    const ProcessResult r = run_process(iso.exe, {L"doctor"}, bundle, bundle);
    expect_doctor_load_failure(r, "dll_not_found");
}

TEST(Doctor, WrongDllVersionIsReported) {
    if (!fs::is_regular_file(kSystemOrtDll))
        GTEST_SKIP() << kSystemOrtDll << " is not present";
    IsolatedExe iso;
    fs::copy_file(kSystemOrtDll, iso.dir.path() / "onnxruntime.dll");
    const ProcessResult r = run_process(iso.exe, {L"doctor"}, iso.dir.path());
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
    const ProcessResult r = run_process(iso.exe, {L"doctor"}, iso.dir.path());
    expect_doctor_load_failure(r, "entry_point_missing");
    EXPECT_TRUE(contains(r.err, "does not export OrtGetApiBase")) << describe(r);
}

TEST(Doctor, CorruptDllIsReported) {
    IsolatedExe iso;
    std::ofstream(iso.dir.path() / "onnxruntime.dll") << "this is not a DLL";
    const ProcessResult r = run_process(iso.exe, {L"doctor"}, iso.dir.path());
    expect_doctor_load_failure(r, "load_failed");
    EXPECT_TRUE(contains(r.err, "LoadLibraryExW(")) << describe(r);
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

TEST(Cli, RunDryRunPrintsResolvedConfig) {
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
    EXPECT_TRUE(contains(r.err, "cannot load ONNX Runtime (dll_not_found)")) << describe(r);
}

TEST(Cli, RunWithoutDryRunIsNotAvailableYet) {
    IsolatedExe iso;
    const ProcessResult r =
        run_process(iso.exe, {L"run", L"--model", iso.dummy_model().native()}, iso.dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::internal_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "benchmark execution is not available in this build"))
        << describe(r);
    EXPECT_TRUE(r.out.empty()) << describe(r);
}

#ifdef EDGEBENCH_ORT_PROBE_PATH
TEST(OrtProbe, LoadFailureExits4) {
    TempDir dir;
    const fs::path probe = dir.path() / "ort_probe.exe";
    fs::copy_file(fs::path(EDGEBENCH_ORT_PROBE_PATH).make_preferred(), probe);
    const ProcessResult r = run_process(probe, {L"--model", L"any.onnx"}, dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "(dll_not_found)")) << describe(r);
}
#endif

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
