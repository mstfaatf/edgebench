// The exit-path table of docs/cli.md, row by row, as child processes: console
// interrupts, crashes, unreadable inputs, unusable output directories, failing
// runs, and paths with spaces and non-ASCII characters.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <thread>

#include "edgebench/exit_codes.hpp"
#include "process_util.hpp"
#include "validator.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace fs = std::filesystem;
namespace eb = edgebench;
using eb::ExitCode;
using eb::test::ChildProcess;
using eb::test::ProcessResult;
using eb::test::run_process;
using eb::test::TempDir;
using json = nlohmann::ordered_json;

namespace {

const fs::path kEdgebenchExe = fs::path(EDGEBENCH_EXE_PATH).make_preferred();
const fs::path kSignalHelper = fs::path(EDGEBENCH_SIGNAL_HELPER_PATH).make_preferred();
const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();
const fs::path kFixtures = kSourceDir / "engine" / "tests" / "fixtures" / "models";

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

std::vector<std::wstring> short_run_args(const fs::path& model, const fs::path& out) {
    return {L"run", L"--model",      model.native(), L"--trials",      L"2", L"--warmup",
            L"3",   L"--iterations", L"10",          L"--cooldown-ms", L"0", L"--load-sample-ms",
            L"0",   L"--output-dir", out.native()};
}

// The single <name>_latest.json in `dir`.
fs::path latest_in(const fs::path& dir) {
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::wstring n = e.path().filename().native();
        if (n.size() > 12 && n.ends_with(L"_latest.json"))
            return e.path();
    }
    return {};
}

void expect_valid(const fs::path& result) {
    eb::test::validate_or_skip(result);
}

#define SKIP_WITHOUT_CONSOLE()              \
    if (!ChildProcess::console_available()) \
    GTEST_SKIP() << "no console to share: console control events cannot be sent"

}  // namespace

// --------------------------------------------------------------------------- interrupts

TEST(ExitPaths, CtrlBreakDuringARunWritesAnInterruptedResult) {
    SKIP_WITHOUT_CONSOLE();
    TempDir out;
    std::vector<std::wstring> args = short_run_args(kFixtures / "fixed_float32.onnx", out.path());
    args[8] = L"1000000";  // --iterations: far longer than the test waits
    ChildProcess child(kEdgebenchExe, args, kSourceDir);
    ASSERT_TRUE(child.wait_for_stderr("trial 1 of 2: creating session"));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    ASSERT_TRUE(child.send_ctrl_break());
    const ProcessResult r = child.wait(30000);
    ASSERT_FALSE(r.timed_out) << describe(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::interrupted)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [interrupted] interrupted by Ctrl+Break (trial 1 of 2, "))
        << describe(r);
    EXPECT_TRUE(contains(r.err, "warning: [interrupted_trial_discarded] trial 1 of 2 was "))
        << describe(r);
    EXPECT_TRUE(contains(r.err, "failed result written to ")) << describe(r);
    EXPECT_FALSE(contains(r.err, "recording node placement")) << describe(r);
    const fs::path latest = latest_in(out.path());
    ASSERT_FALSE(latest.empty()) << describe(r);
    const json doc = json::parse(std::ifstream(latest));
    EXPECT_EQ(doc.at("status"), "failed");
    EXPECT_EQ(doc.at("error").at("code"), "interrupted");
    EXPECT_EQ(doc.at("error").at("stage"), "measurement");
    expect_valid(latest);
}

TEST(ExitPaths, SecondCtrlBreakExitsAtOnce) {
    SKIP_WITHOUT_CONSOLE();
    ChildProcess child(kSignalHelper, {L"interrupt"}, kSourceDir);
    // The graceful handler keeps the helper alive after the first break ...
    ASSERT_TRUE(child.wait_for_stderr("ready"));
    ASSERT_TRUE(child.send_ctrl_break());
    ASSERT_TRUE(child.wait_for_stderr("stop requested"));
    // ... and a second one ends it immediately with 130.
    ASSERT_TRUE(child.send_ctrl_break());
    const ProcessResult r = child.wait(10000);
    EXPECT_FALSE(r.timed_out) << describe(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::interrupted)) << describe(r);
}

#ifdef EDGEBENCH_ORT_PROBE_PATH
TEST(ExitPaths, OrtProbeCtrlBreakExits130WithoutWriting) {
    SKIP_WITHOUT_CONSOLE();
    TempDir dir;
    const fs::path json_path = dir.path() / "probe.json";
    ChildProcess child(fs::path(EDGEBENCH_ORT_PROBE_PATH).make_preferred(),
                       {L"--model", (kFixtures / "fixed_float32.onnx").native(), L"--target",
                        L"cpu", L"--runs", L"100000000", L"--json", json_path.native()},
                       dir.path());
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    ASSERT_TRUE(child.send_ctrl_break());
    const ProcessResult r = child.wait(30000);
    EXPECT_FALSE(r.timed_out) << describe(r);
    EXPECT_EQ(r.exit_code, code(ExitCode::interrupted)) << describe(r);
    EXPECT_FALSE(fs::exists(json_path));
}
#endif

// --------------------------------------------------------------------------- crashes

TEST(ExitPaths, UnhandledExceptionIsReportedAndExits1) {
    const ProcessResult r = run_process(kSignalHelper, {L"crash"}, kSourceDir);
    EXPECT_FALSE(r.timed_out);
    EXPECT_EQ(r.exit_code, code(ExitCode::internal_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [internal_error] unhandled exception 0xC0000005 at "))
        << describe(r);
}

// --------------------------------------------------------------------------- inputs

TEST(ExitPaths, LockedModelExits2WithTheOsError) {
    TempDir dir;
    const fs::path model = dir.path() / "locked.onnx";
    fs::copy_file(kFixtures / "fixed_float32.onnx", model);
    HANDLE lock = CreateFileW(model.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    ASSERT_NE(lock, INVALID_HANDLE_VALUE);
    const ProcessResult r =
        run_process(kEdgebenchExe, short_run_args(model, dir.path() / "out"), kSourceDir);
    CloseHandle(lock);
    EXPECT_EQ(r.exit_code, code(ExitCode::usage_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [model_unreadable] --model: cannot open ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "being used by another process")) << describe(r);
    EXPECT_FALSE(fs::exists(dir.path() / "out"));
}

TEST(ExitPaths, OutputDirThatIsAFileExits6BeforeMeasuring) {
    TempDir dir;
    const fs::path file = dir.path() / "results.txt";
    std::ofstream(file) << "x";
    const ProcessResult r = run_process(
        kEdgebenchExe, short_run_args(kFixtures / "fixed_float32.onnx", file), kSourceDir);
    EXPECT_EQ(r.exit_code, code(ExitCode::output_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [output_dir_not_a_directory] ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "hint: ")) << describe(r);
    EXPECT_FALSE(contains(r.err, "trial 1")) << describe(r);  // nothing was measured
    EXPECT_TRUE(r.out.empty()) << describe(r);
}

TEST(ExitPaths, RunFailureExits3AndWritesAFailedResult) {
    TempDir out;
    const ProcessResult r = run_process(
        kEdgebenchExe, short_run_args(kFixtures / "runtime_error.onnx", out.path()), kSourceDir);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_runtime_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [inference_failed] ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "trial 1 of 2, warm-up iteration 1")) << describe(r);
    const fs::path latest = latest_in(out.path());
    ASSERT_FALSE(latest.empty()) << describe(r);
    const json doc = json::parse(std::ifstream(latest));
    EXPECT_EQ(doc.at("error").at("stage"), "warmup");
    EXPECT_EQ(doc.at("error").at("code"), "ORT_INVALID_ARGUMENT");
    expect_valid(latest);
}

TEST(ExitPaths, ThreadsAboveLogicalProcessorsWarnEverywhere) {
    TempDir out;
    std::vector<std::wstring> args = short_run_args(kFixtures / "fixed_float32.onnx", out.path());
    args[4] = L"1";  // --trials
    args.insert(args.end(), {L"--threads", L"256"});
    const ProcessResult r = run_process(kEdgebenchExe, args, kSourceDir);
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    EXPECT_TRUE(contains(r.err, "warning: [threads_exceed_logical_processors] 256 intra-op"))
        << describe(r);
    EXPECT_TRUE(contains(r.out, "[threads_exceed_logical_processors]")) << describe(r);
    const json doc = json::parse(std::ifstream(latest_in(out.path())));
    bool found = false;
    for (const json& w : doc.at("warnings"))
        found = found || w.at("code") == "threads_exceed_logical_processors";
    EXPECT_TRUE(found);
}

TEST(ExitPaths, PathsWithSpacesAndNonAsciiCharacters) {
    TempDir dir(L"edgebench é 测试");
    const fs::path model = dir.path() / L"modèle 模型.onnx";
    fs::copy_file(kFixtures / "fixed_float32.onnx", model);
    const fs::path out = dir.path() / L"résultats 结果";
    std::vector<std::wstring> args = short_run_args(model, out);
    const ProcessResult r = run_process(kEdgebenchExe, args, dir.path());
    ASSERT_EQ(r.exit_code, 0u) << describe(r);
    // The summary names the file in UTF-8.
    EXPECT_TRUE(contains(r.out, "r\xc3\xa9sultats \xe7\xbb\x93\xe6\x9e\x9c")) << describe(r);
    const fs::path latest = latest_in(out);
    ASSERT_FALSE(latest.empty());
    const json doc = json::parse(std::ifstream(latest));
    const std::string file_name = "mod\xc3\xa8le \xe6\xa8\xa1\xe5\x9e\x8b.onnx";
    EXPECT_EQ(doc.at("model").at("file_name"), file_name);
    // The CSVs exist under the non-ASCII directory, and the result validates there.
    const std::string raw = doc.at("raw").at("latencies_file").get<std::string>();
    EXPECT_TRUE(
        fs::is_regular_file(latest.parent_path() / fs::path(std::u8string(raw.begin(), raw.end()))))
        << raw;
    expect_valid(latest);
    // The dry run reports the same absolute path in UTF-8.
    const ProcessResult d =
        run_process(kEdgebenchExe, {L"run", L"--model", model.native(), L"--dry-run"}, dir.path());
    ASSERT_EQ(d.exit_code, 0u) << describe(d);
    const json dry = json::parse(d.out);
    EXPECT_EQ(dry.at("model").at("file_name"), file_name);
    const std::u8string expected = model.u8string();
    EXPECT_EQ(dry.at("model").at("path").get<std::string>(),
              std::string(expected.begin(), expected.end()));
}

TEST(ExitPaths, InvalidModelExits3AndWritesAFailedResult) {
    TempDir dir;
    const fs::path model = dir.path() / "garbage.onnx";
    std::ofstream(model, std::ios::binary) << "this is not a protobuf";
    const fs::path out = dir.path() / "out";
    const ProcessResult r = run_process(kEdgebenchExe, short_run_args(model, out), kSourceDir);
    EXPECT_EQ(r.exit_code, code(ExitCode::ort_runtime_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [session_creation_failed] ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "ORT_INVALID_PROTOBUF")) << describe(r);
    const fs::path latest = latest_in(out);
    ASSERT_FALSE(latest.empty()) << describe(r);
    const json doc = json::parse(std::ifstream(latest));
    EXPECT_EQ(doc.at("error").at("stage"), "session_creation");
    EXPECT_TRUE(doc.at("model").is_null());                    // the file could not be decoded
    EXPECT_TRUE(doc.at("config").at("batch_size").is_null());  // and is not made up
    expect_valid(latest);
}

#ifdef EDGEBENCH_ORT_PROBE_PATH
TEST(ExitPaths, OrtProbeUsageErrorExits2) {
    const ProcessResult r = run_process(fs::path(EDGEBENCH_ORT_PROBE_PATH).make_preferred(),
                                        {L"--target", L"gpu"}, kSourceDir);
    EXPECT_EQ(r.exit_code, code(ExitCode::usage_error)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [usage_error] --target must be cpu or npu")) << describe(r);
    EXPECT_TRUE(contains(r.err, "hint: Run ort_probe --help for the options.")) << describe(r);
}
#endif
