// Runs edgebench.exe (and ort_probe.exe) as child processes: the command-line
// interface, and `doctor` with a correct, missing, or wrong onnxruntime.dll. Every
// case must end with a documented exit code and an explanation, never a crash.

#include <gtest/gtest.h>

#include <onnxruntime_c_api.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "edgebench/cli.hpp"
#include "edgebench/doctor.hpp"
#include "edgebench/ep_context_cache.hpp"
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

// `doctor` could not load ONNX Runtime: exit 4, the check and the error kind on
// stderr, and the report still on stdout.
void expect_doctor_load_failure(const ProcessResult& r, std::string_view kind) {
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [" + std::string(kind) + "] ort_bundle: ")) << describe(r);
    EXPECT_TRUE(contains(r.out, "FAIL  ort_bundle")) << describe(r);
    EXPECT_TRUE(contains(r.out, "[" + std::string(kind) + "] exit code 4")) << describe(r);
    EXPECT_TRUE(contains(r.out, "Result: FAILED (exit 4, ort_load_failure)")) << describe(r);
}

// The check named `name` of a doctor --json report.
const json& doctor_check(const json& report, std::string_view name) {
    for (const json& c : report.at("checks"))
        if (c.at("name").get<std::string>() == name)
            return c;
    throw std::runtime_error("no check " + std::string(name));
}

// stderr of a doctor run that failed nothing: no error lines (the QNN EP may write
// its own diagnostics there).
void expect_no_doctor_errors(const ProcessResult& r) {
    EXPECT_FALSE(contains(r.err, "error: [")) << describe(r);
}

}  // namespace

// --------------------------------------------------------------------------- doctor
//
// These run `doctor --quick`: the checks that use the NPU backend run only in the
// DoctorNpu suite below, which holds the NPU test lock.

TEST(Doctor, HappyPathReportsBundledOnnxRuntime) {
    const ProcessResult r =
        run_process(kEdgebenchExe, {L"doctor", L"--quick", L"--load-sample-ms", L"0"},
                    kEdgebenchExe.parent_path());
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    expect_no_doctor_errors(r);
    const std::string expected(eb::expected_onnxruntime_version());
    EXPECT_TRUE(contains(r.out, "ONNX Runtime " + expected + " (pinned " + expected + ")"))
        << r.out;
    // PASS natively; WARN (process_emulated) for an x64 build on ARM64 Windows.
    const std::string status = eb::query_system_info().emulated() ? "WARN" : "PASS";
    EXPECT_TRUE(contains(r.out, status + "  ort_bundle        onnxruntime " + expected +
                                    " (C API " + std::to_string(ORT_API_VERSION) +
                                    ") from the bundle"))
        << r.out;
    EXPECT_TRUE(contains(r.out, "match their SHA256 pins")) << r.out;
    if (eb::query_system_info().emulated())
        EXPECT_TRUE(contains(r.out, "[process_emulated]")) << r.out;
    for (const char* name : {"qnn_bundle", "backend_versions", "devices", "npu_probe",
                             "npu_context", "ep_cache_dir", "power", "background_load"})
        EXPECT_TRUE(contains(r.out, std::string(name) + " ")) << name << "\n" << r.out;
    EXPECT_TRUE(contains(r.out, "Result: OK")) << r.out;
}

TEST(Doctor, JsonReportFollowsTheSchema) {
    TempDir out;
    const ProcessResult r =
        run_process(kEdgebenchExe, {L"doctor", L"--json", L"--quick", L"--load-sample-ms", L"100"},
                    kEdgebenchExe.parent_path());
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const json j = json::parse(r.out);
    EXPECT_EQ(j.at("doctor_schema_version"), "1.0.0");
    EXPECT_EQ(j.at("tool").at("version").get<std::string>(), eb::version());
    EXPECT_EQ(j.at("versions").at("onnxruntime").at("loaded").get<std::string>(),
              eb::expected_onnxruntime_version());
    std::vector<std::string> names;
    for (const json& c : j.at("checks"))
        names.push_back(c.at("name"));
    EXPECT_EQ(names, (std::vector<std::string>{"ort_bundle", "qnn_bundle", "backend_versions",
                                               "devices", "npu_probe", "npu_context",
                                               "ep_cache_dir", "power", "background_load"}));
    const json& ort = doctor_check(j, "ort_bundle");
    EXPECT_NE(ort.at("status"), "fail") << ort.dump(2);
    for (const json& f : ort.at("details").at("files"))
        EXPECT_TRUE(f.at("matches_pin").get<bool>()) << f.dump();
    EXPECT_TRUE(ort.at("details").at("modules_outside_bundle").empty());
    EXPECT_EQ(doctor_check(j, "qnn_bundle").at("status"),
              eb::qnn_ep_in_build() ? "pass" : "not_applicable");
    // --quick: the checks that need the NPU are skipped, and say so.
    for (const char* name : {"backend_versions", "npu_probe", "npu_context"}) {
        const json& c = doctor_check(j, name);
        EXPECT_EQ(c.at("status"), "not_applicable") << c.dump(2);
        if (eb::qnn_ep_in_build() && doctor_check(j, "devices").at("status") == "pass")
            EXPECT_EQ(c.at("summary"), std::string(eb::kSkippedByQuick)) << c.dump(2);
    }
    const json& load = doctor_check(j, "background_load");
    ASSERT_TRUE(load.at("details").at("cpu_utilization_ratio").is_number()) << load.dump();
    EXPECT_EQ(j.at("overall").at("exit_code"), 0);
    std::ofstream(out.path() / "doctor.json") << r.out;
    eb::test::validate_doctor_report_or_skip(out.path() / "doctor.json");
}

TEST(Doctor, FromAnotherWorkingDirectory) {
    TempDir cwd;
    const ProcessResult r =
        run_process(kEdgebenchExe, {L"doctor", L"--quick", L"--load-sample-ms", L"0"}, cwd.path());
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
    EXPECT_EQ(j.at("overall").at("status"), "fail");
    EXPECT_EQ(j.at("overall").at("exit_code"), 4);
    const json& ort = doctor_check(j, "ort_bundle");
    EXPECT_EQ(ort.at("status"), "fail");
    EXPECT_EQ(ort.at("code"), "dll_not_found");
    EXPECT_EQ(ort.at("exit_code"), 4);
    EXPECT_TRUE(j.at("versions").at("onnxruntime").at("loaded").is_null());
    EXPECT_FALSE(j.at("system").at("cpu").at("name").is_null());  // still reported
    TempDir out;
    std::ofstream(out.path() / "doctor.json") << r.out;
    eb::test::validate_doctor_report_or_skip(out.path() / "doctor.json");
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

// --------------------------------------------------------------------------- doctor: bundle pins

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

TEST(Doctor, AModifiedBundleFileFailsItsPin) {
    const CpuOnlyBundle bundle(kEdgebenchExe);
    // Still a loadable DLL, but not the pinned file.
    std::ofstream(bundle.dir.path() / "onnxruntime_providers_shared.dll",
                  std::ios::binary | std::ios::app)
        << "extra bytes";
    const ProcessResult r =
        run_process(bundle.exe, {L"doctor", L"--json", L"--quick", L"--load-sample-ms", L"0"},
                    bundle.dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    const json j = json::parse(r.out);
    const json& ort = doctor_check(j, "ort_bundle");
    EXPECT_EQ(ort.at("code"), "bundle_file_mismatch") << ort.dump(2);
    EXPECT_TRUE(contains(ort.at("summary").get<std::string>(), "onnxruntime_providers_shared.dll"))
        << ort.dump(2);
    EXPECT_TRUE(contains(r.err, "error: [bundle_file_mismatch] ort_bundle: ")) << describe(r);
}

TEST(Doctor, MissingQnnFilesFailOnlyBuildsThatBundleThem) {
    const CpuOnlyBundle bundle(kEdgebenchExe);
    const ProcessResult r =
        run_process(bundle.exe, {L"doctor", L"--json", L"--quick", L"--load-sample-ms", L"0"},
                    bundle.dir.path());
    expect_no_crash(r);
    const json j = json::parse(r.out);
    const json& qnn = doctor_check(j, "qnn_bundle");
    if (!eb::qnn_ep_in_build()) {
        EXPECT_EQ(r.exit_code, 0u) << describe(r);
        EXPECT_EQ(qnn.at("status"), "not_applicable") << qnn.dump(2);
        return;
    }
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_EQ(qnn.at("code"), "qnn_bundle_mismatch") << qnn.dump(2);
    EXPECT_TRUE(contains(qnn.at("summary").get<std::string>(), "QnnHtp.dll: missing"))
        << qnn.dump(2);
    EXPECT_TRUE(contains(r.err, "error: [qnn_bundle_mismatch] qnn_bundle: ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "hint: ")) << describe(r);
}

TEST(Doctor, QnnEpThatIsNotAnEpLibraryIsReported) {
    const CpuOnlyBundle bundle(kEdgebenchExe);
    bundle.add_qnn_ep_file(kSystemNonOrtDll);
    const ProcessResult r =
        run_process(bundle.exe, {L"doctor", L"--json", L"--quick", L"--load-sample-ms", L"0"},
                    bundle.dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    const json j = json::parse(r.out);
    const json& qnn = doctor_check(j, "qnn_bundle");
    EXPECT_EQ(qnn.at("status"), "fail") << qnn.dump(2);
    EXPECT_EQ(qnn.at("details").at("registration"), "failed") << qnn.dump(2);
    // ARM64: the plugin is not the pinned file; x64: no pins, the registration fails.
    EXPECT_EQ(qnn.at("code"),
              eb::qnn_ep_in_build() ? "qnn_bundle_mismatch" : "ep_registration_failed")
        << qnn.dump(2);
}

TEST(Doctor, CacheDirectoryIsReportedAndNeverChangedWithoutPrune) {
    TempDir cache;
    fs::create_directory(cache.path() / "not_ours");
    std::ofstream(cache.path() / "not_ours" / "keep.txt") << "x";
    const ProcessResult r = run_process(kEdgebenchExe,
                                        {L"doctor", L"--json", L"--quick", L"--load-sample-ms",
                                         L"0", L"--ep-cache-dir", cache.path().native()},
                                        kEdgebenchExe.parent_path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, 0u) << describe(r);
    const json j = json::parse(r.out);
    const json& c = doctor_check(j, "ep_cache_dir");
    EXPECT_EQ(c.at("status"), "warn") << c.dump(2);
    EXPECT_EQ(c.at("code"), "cache_directories_without_manifest") << c.dump(2);
    EXPECT_EQ(c.at("details").at("writable"), true);
    EXPECT_EQ(c.at("details").at("entry_count"), 0);
    EXPECT_TRUE(fs::is_regular_file(cache.path() / "not_ours" / "keep.txt"));
}

TEST(Doctor, CacheDirectoryThatCannotBeCreatedExits6) {
    TempDir dir;
    std::ofstream(dir.path() / "a_file") << "x";
    const ProcessResult r =
        run_process(kEdgebenchExe,
                    {L"doctor", L"--json", L"--quick", L"--load-sample-ms", L"0", L"--ep-cache-dir",
                     (dir.path() / "a_file" / "cache").native()},
                    kEdgebenchExe.parent_path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::output_error)) << describe(r);
    const json j = json::parse(r.out);
    const json& c = doctor_check(j, "ep_cache_dir");
    EXPECT_EQ(c.at("code"), "ep_cache_dir_not_writable") << c.dump(2);
    EXPECT_EQ(c.at("exit_code"), 6);
    EXPECT_FALSE(fs::exists(dir.path() / "a_file" / "cache"));
}

TEST(Doctor, EmptyCacheDirectoryIsAUsageError) {
    const ProcessResult r = run_process(kEdgebenchExe, {L"doctor", L"--ep-cache-dir", L""},
                                        kEdgebenchExe.parent_path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::usage_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "--ep-cache-dir: must not be empty")) << describe(r);
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
        {L"run", L"--model", iso.dummy_model().native(), L"--provider", L"gpu"},
        {L"run", L"--model", iso.dummy_model().native(), L"--npu-fp16"},
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
    const ProcessResult gpu = run_process(
        iso.exe, {L"run", L"--model", iso.dummy_model().native(), L"--provider", L"gpu"},
        iso.dir.path());
    EXPECT_TRUE(contains(gpu.err, "supported providers: cpu, npu")) << describe(gpu);
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
    EXPECT_EQ(doc.at("schema_version"), "1.2.0");
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

// `run --provider npu` without the QNN execution provider in the bundle: exit 4
// and a failed result (stage ep_registration). Runs everywhere.
TEST(Cli, RunNpuWithoutTheQnnEpExits4AndWritesAFailedResult) {
    const CpuOnlyBundle bundle(kEdgebenchExe);
    TempDir out;
    const ProcessResult r =
        run_process(bundle.exe,
                    {L"run", L"--model", (kFixtureDir / "fixed_float32.onnx").native(),
                     L"--provider", L"npu", L"--npu-fp16", L"--trials", L"1", L"--cooldown-ms",
                     L"0", L"--load-sample-ms", L"0", L"--output-dir", out.path().native()},
                    bundle.dir.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [qnn_ep_not_bundled] ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "hint: ")) << describe(r);
    const fs::path latest = out.path() / "fixed_float32_npu_iobinding_t4_latest.json";
    ASSERT_TRUE(fs::is_regular_file(latest)) << describe(r);
    const json doc = json::parse(std::ifstream(latest));
    EXPECT_EQ(doc.at("error").at("stage"), "ep_registration");
    EXPECT_EQ(doc.at("config").at("strict_mode"), true);
    eb::test::validate_or_skip(latest);
}

// The precision rule is checked before the QNN EP is registered, so these usage
// errors need no NPU and write nothing.
TEST(Cli, RunNpuPrecisionRulesAreUsageErrors) {
    TempDir out;
    const std::vector<std::pair<std::vector<std::wstring>, std::string>> cases = {
        {{L"--model", (kFixtureDir / "fixed_float32.onnx").native()}, "npu_fp32_needs_fp16"},
        {{L"--model", (kFixtureDir / "nan_output.onnx").native(), L"--no-strict"},
         "npu_fp32_needs_fp16"},
    };
    for (const auto& [extra, kind] : cases) {
        std::vector<std::wstring> args = {L"run", L"--provider", L"npu", L"--output-dir",
                                          out.path().native()};
        args.insert(args.end(), extra.begin(), extra.end());
        const ProcessResult r = run_process(kEdgebenchExe, args, kSourceDir);
        expect_no_crash(r);
        EXPECT_EQ(r.exit_code, code(ExitCode::usage_error)) << describe(r);
        EXPECT_TRUE(contains(r.err, "error: [" + kind + "] the NPU has no FP32 compute"))
            << describe(r);
    }
    EXPECT_TRUE(fs::is_empty(out.path()));
}

// The CPU provider has no accelerator to profile: refused before ONNX Runtime loads.
TEST(Cli, ProfileAcceleratorWithTheCpuProviderExits2) {
    TempDir out;
    const ProcessResult r = run_process(
        kEdgebenchExe,
        {L"run", L"--model", (kFixtureDir / "fixed_float32.onnx").native(), L"--provider", L"cpu",
         L"--profile-accelerator", L"--output-dir", out.path().native()},
        kSourceDir);
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::usage_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "--profile-accelerator: is only allowed with --provider npu"))
        << describe(r);
    EXPECT_TRUE(fs::is_empty(out.path()));
}

// Like the probe test below: only reachable where the QNN EP offers no single NPU.
TEST(Cli, RunNpuWithoutASingleNpuExits3) {
    if (!eb::qnn_ep_in_build())
        GTEST_SKIP() << "this build does not include the QNN execution provider";
    const eb::test::QnnNpuProbe probe = eb::test::probe_qnn_npu();
    ASSERT_FALSE(probe.registration_error) << probe.registration_error->message;
    ASSERT_TRUE(probe.selection);
    if (probe.selection->ok()) {
        GTEST_SKIP() << "this machine has exactly one QNN NPU ("
                     << eb::describe_device(probe.devices[*probe.selection->index]) << ")";
    }
    TempDir out;
    const ProcessResult r =
        run_process(kEdgebenchExe,
                    {L"run", L"--model", (kFixtureDir / "fixed_float32.onnx").native(),
                     L"--provider", L"npu", L"--npu-fp16", L"--cooldown-ms", L"0",
                     L"--load-sample-ms", L"0", L"--output-dir", out.path().native()},
                    kSourceDir);
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_runtime_error)) << describe(r);
    EXPECT_TRUE(
        contains(r.err, "error: [" + std::string(eb::to_string(*probe.selection->failure)) + "] "))
        << describe(r);
    const fs::path latest = out.path() / "fixed_float32_npu_iobinding_t4_latest.json";
    ASSERT_TRUE(fs::is_regular_file(latest)) << describe(r);
    eb::test::validate_or_skip(latest);
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

TEST(DoctorNpu, FullReportPassesTheNpuChecks) {
    const eb::test::QnnNpuProbe probe = eb::test::probe_qnn_npu();
    if (const std::optional<std::string> reason = eb::test::npu_skip_reason(probe))
        GTEST_SKIP() << *reason;
    TempDir cache;
    const ProcessResult r = run_process(
        kEdgebenchExe,
        {L"doctor", L"--json", L"--load-sample-ms", L"0", L"--ep-cache-dir", cache.path().native()},
        kEdgebenchExe.parent_path());
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const json j = json::parse(r.out);
    for (const char* name : {"ort_bundle", "qnn_bundle", "backend_versions", "devices", "npu_probe",
                             "npu_context", "ep_cache_dir"})
        EXPECT_EQ(doctor_check(j, name).at("status"), "pass") << doctor_check(j, name).dump(2);

    const json& devices = doctor_check(j, "devices").at("details").at("npu_selection");
    EXPECT_EQ(devices.at("status"), "selected");
    EXPECT_EQ(devices.at("candidate_count"), 1);
    const json& backend = doctor_check(j, "backend_versions").at("details");
    EXPECT_EQ(backend.at("sdk_version"), j.at("versions").at("qairt_sdk").at("reported"));
    EXPECT_EQ(backend.at("source"), "qnn_ep_log_backend_build_id");
    const json& probes = doctor_check(j, "npu_probe").at("details").at("probes");
    ASSERT_EQ(probes.size(), 2u);
    for (const json& p : probes) {
        EXPECT_EQ(p.at("equal"), true) << p.dump(2);
        EXPECT_EQ(p.at("differing_elements"), 0);
        EXPECT_EQ(p.at("elements"), 256);
    }
    EXPECT_EQ(probes[0].at("compute_precision"), "int8");
    EXPECT_EQ(probes[1].at("compute_precision"), "fp16");
    const json& ctx = doctor_check(j, "npu_context").at("details");
    EXPECT_EQ(ctx.at("outputs_equal"), true);
    EXPECT_TRUE(std::regex_match(ctx.at("htp_arch").get<std::string>(), std::regex(R"(v\d+)")))
        << ctx.dump(2);
    EXPECT_TRUE(ctx.at("verdict").is_null());  // not askable after a session (see qnn_provider.hpp)
    EXPECT_FALSE(ctx.at("verdict_note").is_null());
    EXPECT_TRUE(ctx.at("cleanup_error").is_null());
    // Nothing was written to the cache directory given (the context lives in %TEMP%).
    EXPECT_TRUE(fs::is_empty(cache.path()));
    std::ofstream(cache.path().parent_path() / (cache.path().filename().wstring() + L".json"))
        << r.out;
    eb::test::validate_doctor_report_or_skip(cache.path().parent_path() /
                                             (cache.path().filename().wstring() + L".json"));
}

// --prune-cache deletes the stale entry (another QNN SDK version in its key) and
// keeps the current one; without it doctor only warns.
TEST(DoctorNpu, PruneCacheDeletesOnlyStaleEntries) {
    const eb::test::QnnNpuProbe probe = eb::test::probe_qnn_npu();
    if (const std::optional<std::string> reason = eb::test::npu_skip_reason(probe))
        GTEST_SKIP() << *reason;
    TempDir work;
    const fs::path cache = work.path() / "cache";
    const ProcessResult built = run_process(kEdgebenchExe,
                                            {L"run",
                                             L"--model",
                                             (kFixtureDir / "fixed_float32.onnx").native(),
                                             L"--provider",
                                             L"npu",
                                             L"--npu-fp16",
                                             L"--trials",
                                             L"1",
                                             L"--warmup",
                                             L"1",
                                             L"--iterations",
                                             L"2",
                                             L"--cooldown-ms",
                                             L"0",
                                             L"--load-sample-ms",
                                             L"0",
                                             L"--ep-cache",
                                             L"build",
                                             L"--ep-cache-dir",
                                             cache.native(),
                                             L"--output-dir",
                                             (work.path() / "out").native()},
                                            kSourceDir);
    ASSERT_EQ(built.exit_code, 0u) << describe(built);
    std::vector<fs::path> entries;
    for (const fs::directory_entry& e : fs::directory_iterator(cache))
        entries.push_back(e.path());
    ASSERT_EQ(entries.size(), 1u);
    const fs::path current = entries.front();
    const std::optional<eb::CacheManifest> m = eb::read_cache_manifest(current);
    ASSERT_TRUE(m);
    // The same entry under a key with another SDK version: stale.
    std::optional<eb::CacheKeyFields> fields = eb::cache_key_from_json(m->key_fields);
    ASSERT_TRUE(fields);
    fields->qnn_sdk_version = "2.0.0";
    const std::string stale_id = eb::cache_key_id(*fields);
    const fs::path stale = cache / stale_id;
    fs::create_directory(stale);
    for (const eb::CacheFile& f : m->files)
        fs::copy_file(current / f.name, stale / f.name);
    ASSERT_TRUE(eb::write_cache_manifest(stale, stale_id, eb::cache_key_json(*fields),
                                         m->context_model, "2026-01-01T00:00:00.000Z"));

    const std::vector<std::wstring> args = {L"doctor", L"--json",         L"--load-sample-ms",
                                            L"0",      L"--ep-cache-dir", cache.native()};
    const ProcessResult warned = run_process(kEdgebenchExe, args, kEdgebenchExe.parent_path());
    ASSERT_EQ(warned.exit_code, 0u) << describe(warned);
    const json warned_report = json::parse(warned.out);
    const json& w = doctor_check(warned_report, "ep_cache_dir");
    EXPECT_EQ(w.at("code"), "stale_cache_entries") << w.dump(2);
    EXPECT_EQ(w.at("details").at("stale_count"), 1);
    EXPECT_TRUE(fs::exists(stale));  // never deleted without --prune-cache

    std::vector<std::wstring> prune = args;
    prune.push_back(L"--prune-cache");
    const ProcessResult pruned = run_process(kEdgebenchExe, prune, kEdgebenchExe.parent_path());
    ASSERT_EQ(pruned.exit_code, 0u) << describe(pruned);
    const json pruned_report = json::parse(pruned.out);
    const json& p = doctor_check(pruned_report, "ep_cache_dir");
    EXPECT_EQ(p.at("status"), "pass") << p.dump(2);
    EXPECT_EQ(p.at("details").at("pruned_count"), 1);
    EXPECT_FALSE(fs::exists(stale));
    EXPECT_TRUE(fs::is_regular_file(current / eb::kEpCacheManifestName));
}

// Failures exit with the code `run` gives for the same problem.
TEST(DoctorNpu, ExitCodesFollowTheFirstFailure) {
    const eb::test::QnnNpuProbe probe = eb::test::probe_qnn_npu();
    if (const std::optional<std::string> reason = eb::test::npu_skip_reason(probe))
        GTEST_SKIP() << *reason;
    // A bundle without one HTP stub: qnn_bundle fails first, exit code 4.
    TempDir bundle;
    for (const fs::directory_entry& e : fs::directory_iterator(kEdgebenchExe.parent_path())) {
        const std::wstring name = e.path().filename().wstring();
        if (e.is_regular_file() && name != L"QnnHtpV81Stub.dll" &&
            (name.ends_with(L".dll") || name.ends_with(L".so") || name.ends_with(L".cat") ||
             name == kEdgebenchExe.filename().wstring()))
            fs::copy_file(e.path(), bundle.path() / e.path().filename());
    }
    const ProcessResult r4 =
        run_process(bundle.path() / kEdgebenchExe.filename(),
                    {L"doctor", L"--json", L"--load-sample-ms", L"0"}, bundle.path());
    expect_no_crash(r4);
    EXPECT_EQ(r4.exit_code, code(ExitCode::ort_load_failure)) << describe(r4);
    const json j4 = json::parse(r4.out);
    EXPECT_EQ(doctor_check(j4, "qnn_bundle").at("code"), "qnn_bundle_mismatch");
    // The NPU itself still works without that stub (it is another HTP generation's).
    EXPECT_EQ(doctor_check(j4, "npu_probe").at("status"), "pass")
        << doctor_check(j4, "npu_probe").dump(2);

    // An EP context cache directory that cannot be created: exit code 6.
    TempDir dir;
    std::ofstream(dir.path() / "a_file") << "x";
    const ProcessResult r6 =
        run_process(kEdgebenchExe,
                    {L"doctor", L"--json", L"--load-sample-ms", L"0", L"--ep-cache-dir",
                     (dir.path() / "a_file" / "cache").native()},
                    kEdgebenchExe.parent_path());
    EXPECT_EQ(r6.exit_code, code(ExitCode::output_error)) << describe(r6);
    EXPECT_EQ(json::parse(r6.out).at("overall").at("exit_code"), 6);
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
