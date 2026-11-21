// Runs edgebench.exe as a child process with a correct, missing, or wrong
// onnxruntime.dll and checks that every case ends with a clean exit code and an
// explanation, never a crash.

#include <gtest/gtest.h>

#include <onnxruntime_c_api.h>

#include <fstream>
#include <regex>

#include "edgebench/ort_loader.hpp"
#include "process_util.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using eb::test::ProcessResult;
using eb::test::run_process;
using eb::test::TempDir;

namespace {

constexpr uint32_t kExitOrtLoadFailed = 4;
constexpr uint32_t kAccessViolation = 0xC0000005;
constexpr uint32_t kDelayLoadModuleNotFound = 0xC06D007E;
constexpr uint32_t kDelayLoadProcNotFound = 0xC06D007F;

const fs::path kEdgebenchExe = fs::path(EDGEBENCH_EXE_PATH).make_preferred();
const fs::path kPreloadHelper = fs::path(EDGEBENCH_PRELOAD_HELPER_PATH).make_preferred();
const fs::path kSystemOrtDll = L"C:\\Windows\\System32\\onnxruntime.dll";
const fs::path kSystemNonOrtDll = L"C:\\Windows\\System32\\version.dll";

std::string describe(const ProcessResult& r) {
    char code[16];
    std::snprintf(code, sizeof(code), "0x%08X", r.exit_code);
    return std::string("exit code ") + code + (r.timed_out ? " (timed out)" : "") + ", output:\n" +
           r.output;
}

void expect_no_crash(const ProcessResult& r) {
    EXPECT_FALSE(r.timed_out);
    EXPECT_NE(r.exit_code, kAccessViolation) << describe(r);
    EXPECT_NE(r.exit_code, kDelayLoadModuleNotFound) << describe(r);
    EXPECT_NE(r.exit_code, kDelayLoadProcNotFound) << describe(r);
}

// A copy of edgebench.exe alone in a fresh temporary directory.
struct IsolatedExe {
    TempDir dir;
    fs::path exe = dir.path() / "edgebench.exe";
    IsolatedExe() { fs::copy_file(kEdgebenchExe, exe); }
};

void expect_load_failure(const ProcessResult& r, std::string_view kind) {
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, kExitOrtLoadFailed) << describe(r);
    EXPECT_NE(r.output.find("error: cannot load ONNX Runtime (" + std::string(kind) + ")"),
              std::string::npos)
        << describe(r);
}

}  // namespace

TEST(EdgebenchExe, HappyPathReportsBundledOnnxRuntime) {
    const ProcessResult r = run_process(kEdgebenchExe, {}, kEdgebenchExe.parent_path());
    expect_no_crash(r);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    const std::string expected(eb::expected_onnxruntime_version());
    EXPECT_NE(r.output.find("onnxruntime " + expected + "\n"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("C API version: " + std::to_string(ORT_API_VERSION) + "\n"),
              std::string::npos)
        << r.output;
    EXPECT_NE(r.output.find("load check:    full_path_verified"), std::string::npos) << r.output;
    const std::string dll = eb::path_to_utf8(kEdgebenchExe.parent_path() / "onnxruntime.dll");
    EXPECT_NE(r.output.find("library:       " + dll + "\n"), std::string::npos) << r.output;
}

TEST(EdgebenchExe, HappyPathFromAnotherWorkingDirectory) {
    TempDir cwd;
    const ProcessResult r = run_process(kEdgebenchExe, {L"--version"}, cwd.path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, 0u) << describe(r);
}

TEST(EdgebenchExe, MissingDllIsReported) {
    IsolatedExe iso;
    const ProcessResult r = run_process(iso.exe, {}, iso.dir.path());
    expect_load_failure(r, "dll_not_found");
    const std::string dll = eb::path_to_utf8(iso.dir.path() / "onnxruntime.dll");
    EXPECT_NE(r.output.find(dll + " not found"), std::string::npos) << r.output;
}

TEST(EdgebenchExe, MissingDllIgnoresWorkingDirectoryAndPath) {
    // The bundle directory holds a valid onnxruntime.dll; it must not be picked up
    // through the current directory or PATH.
    IsolatedExe iso;
    const fs::path bundle = kEdgebenchExe.parent_path();
    const ProcessResult r = run_process(iso.exe, {}, bundle, bundle);
    expect_load_failure(r, "dll_not_found");
}

TEST(EdgebenchExe, WrongDllVersionIsReported) {
    if (!fs::is_regular_file(kSystemOrtDll))
        GTEST_SKIP() << kSystemOrtDll << " is not present";
    IsolatedExe iso;
    fs::copy_file(kSystemOrtDll, iso.dir.path() / "onnxruntime.dll");
    const ProcessResult r = run_process(iso.exe, {}, iso.dir.path());
    expect_load_failure(r, "version_mismatch");

    // The message names the version that was found and the one required.
    std::smatch m;
    ASSERT_TRUE(std::regex_search(r.output, m, std::regex(R"(is ONNX Runtime (\d+\.\d+\.\d+), )")))
        << r.output;
    RecordProperty("found_version", m[1].str());
    EXPECT_NE(m[1].str(), std::string(eb::expected_onnxruntime_version()));
    EXPECT_NE(
        r.output.find("requires ONNX Runtime " + std::string(eb::expected_onnxruntime_version())),
        std::string::npos)
        << r.output;
}

TEST(EdgebenchExe, DllWithoutOrtEntryPointIsReported) {
    if (!fs::is_regular_file(kSystemNonOrtDll))
        GTEST_SKIP() << kSystemNonOrtDll << " is not present";
    IsolatedExe iso;
    fs::copy_file(kSystemNonOrtDll, iso.dir.path() / "onnxruntime.dll");
    const ProcessResult r = run_process(iso.exe, {}, iso.dir.path());
    expect_load_failure(r, "entry_point_missing");
    EXPECT_NE(r.output.find("does not export OrtGetApiBase"), std::string::npos) << r.output;
}

TEST(EdgebenchExe, CorruptDllIsReported) {
    IsolatedExe iso;
    std::ofstream(iso.dir.path() / "onnxruntime.dll") << "this is not a DLL";
    const ProcessResult r = run_process(iso.exe, {}, iso.dir.path());
    expect_load_failure(r, "load_failed");
    EXPECT_NE(r.output.find("LoadLibraryExW("), std::string::npos) << r.output;
}

TEST(EdgebenchExe, UsageErrorsDoNotLoadOnnxRuntime) {
    IsolatedExe iso;  // no DLL: --help and usage errors must still work
    const ProcessResult help = run_process(iso.exe, {L"--help"}, iso.dir.path());
    expect_no_crash(help);
    EXPECT_EQ(help.exit_code, 0u) << describe(help);
    const ProcessResult bad = run_process(iso.exe, {L"--bogus"}, iso.dir.path());
    expect_no_crash(bad);
    EXPECT_EQ(bad.exit_code, 2u) << describe(bad);
}

TEST(PreloadHelper, ForeignOnnxRuntimeLoadedFirstIsReported) {
    if (!fs::is_regular_file(kSystemOrtDll))
        GTEST_SKIP() << kSystemOrtDll << " is not present";
    const ProcessResult r =
        run_process(kPreloadHelper, {kSystemOrtDll.native()}, kPreloadHelper.parent_path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, kExitOrtLoadFailed) << describe(r);
    EXPECT_NE(r.output.find("load_check: foreign_module_loaded"), std::string::npos) << r.output;
    EXPECT_NE(r.output.find("already loaded"), std::string::npos) << r.output;
}

TEST(PreloadHelper, BundledOnnxRuntimeLoadedFirstIsAccepted) {
    const fs::path bundled = kPreloadHelper.parent_path() / "onnxruntime.dll";
    const ProcessResult r = run_process(kPreloadHelper, {bundled.native()}, bundled.parent_path());
    expect_no_crash(r);
    EXPECT_EQ(r.exit_code, 0u) << describe(r);
    EXPECT_NE(r.output.find("load_check: full_path_verified"), std::string::npos) << r.output;
}
