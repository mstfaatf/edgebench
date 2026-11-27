// NPU runs through QnnProvider: strict success, strict refusal, CPU fallback
// without strict mode, a missing HTP backend, and the module check after QNN
// session creation. Every suite ends in "Npu" (label `npu`): the tests need the
// QNN EP, its backend and an NPU, and skip where there is none (see
// npu_skip_reason()). They use the committed fixture models only.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "edgebench/benchmark.hpp"
#include "edgebench/qnn_ep.hpp"
#include "edgebench/result_files.hpp"
#include "edgebench/result_json.hpp"
#include "process_util.hpp"
#include "test_runtime.hpp"
#include "validator.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using eb::ExitCode;
using eb::test::ProcessResult;
using eb::test::run_process;
using eb::test::shared_runtime;
using eb::test::TempDir;
using json = nlohmann::json;

namespace {

const fs::path kEdgebenchExe = fs::path(EDGEBENCH_EXE_PATH).make_preferred();
const fs::path kFixtureDir =
    fs::path(EDGEBENCH_SOURCE_DIR).make_preferred() / "engine" / "tests" / "fixtures" / "models";
const fs::path kAllOnNpu = kFixtureDir / "fixed_float32.onnx";          // MatMul, Add, Relu
const fs::path kUnsupported = kFixtureDir / "npu_unsupported_op.onnx";  // Relu, Det, Relu
const fs::path kSystemNonOrtDll = LR"(C:\Windows\System32\version.dll)";

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string describe(const ProcessResult& r) {
    return "exit " + std::to_string(r.exit_code) + "\nstdout:\n" + r.out + "\nstderr:\n" + r.err;
}

bool has_warning(const eb::BenchmarkResult& r, const std::string& code) {
    return std::any_of(r.warnings.begin(), r.warnings.end(),
                       [&code](const eb::ResultWarning& w) { return w.code == code; });
}

// A short NPU run of a fixture model; FP32 fixtures need --npu-fp16.
eb::RunConfig npu_config(const fs::path& model) {
    eb::RunConfig c;
    c.model = model;
    c.provider = eb::ProviderKind::npu;
    c.npu_fp16 = true;
    c.trials = 2;
    c.warmup = 2;
    c.iterations = 5;
    c.cooldown_ms = 0;
    c.load_sample_ms = 0;
    return c;
}

// Writes the result into `dir` and runs the validator on it.
void write_and_validate(const eb::BenchmarkResult& r, const TempDir& dir) {
    const eb::Result<eb::WrittenResult> written = eb::write_result_files(r, dir.path(), "npu_test");
    ASSERT_TRUE(written.ok()) << written.error().message;
    eb::test::validate_or_skip(written.value().result_file);
}

// Skips without a usable NPU; every test ends with the QNN EP unregistered again.
class QnnProviderNpu : public ::testing::Test {
protected:
    void SetUp() override {
        const eb::test::QnnNpuProbe probe = eb::test::probe_qnn_npu();
        if (const std::optional<std::string> reason = eb::test::npu_skip_reason(probe))
            GTEST_SKIP() << *reason;
    }
    void TearDown() override {
        const eb::Result<void> r = shared_runtime().unregister_ep_library(eb::kQnnEpName);
        EXPECT_TRUE(r.ok()) << r.error().message;
    }
};

// A copy of the built bundle (every file next to edgebench.exe) without `excluded`.
struct NpuBundle {
    TempDir dir;
    fs::path exe;
    explicit NpuBundle(const std::vector<std::wstring>& excluded) {
        for (const fs::directory_entry& e : fs::directory_iterator(kEdgebenchExe.parent_path())) {
            const std::wstring name = e.path().filename().wstring();
            const std::wstring ext = e.path().extension().wstring();
            const bool wanted = ext == L".dll" || ext == L".so" || ext == L".cat" ||
                                e.path().filename() == kEdgebenchExe.filename() ||
                                e.path().filename() == "ort_probe.exe";
            const bool skip = std::any_of(excluded.begin(), excluded.end(), [&](const auto& x) {
                return _wcsicmp(x.c_str(), name.c_str()) == 0 ||
                       (x.back() == L'*' && _wcsnicmp(x.c_str(), name.c_str(), x.size() - 1) == 0);
            });
            if (e.is_regular_file() && wanted && !skip)
                fs::copy_file(e.path(), dir.path() / e.path().filename());
        }
        exe = dir.path() / kEdgebenchExe.filename();
    }
};

std::vector<std::wstring> npu_run_args(const fs::path& model, const fs::path& out) {
    return {L"run",           L"--model",    model.native(),      L"--provider",
            L"npu",           L"--npu-fp16", L"--trials",         L"1",
            L"--warmup",      L"1",          L"--iterations",     L"3",
            L"--cooldown-ms", L"0",          L"--load-sample-ms", L"0",
            L"--output-dir",  out.native()};
}

// The single result file written into `dir`.
fs::path written_result(const fs::path& dir) {
    for (const fs::directory_entry& e : fs::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        if (e.path().extension() == ".json" && !contains(name, "_latest"))
            return e.path();
    }
    return {};
}

}  // namespace

#define SKIP_WITHOUT_NPU()                                                                \
    do {                                                                                  \
        const eb::test::QnnNpuProbe probe_ = eb::test::probe_qnn_npu();                   \
        if (const std::optional<std::string> reason_ = eb::test::npu_skip_reason(probe_)) \
            GTEST_SKIP() << *reason_;                                                     \
    } while (false)

// --------------------------------------------------------------------------- in-process

TEST_F(QnnProviderNpu, StrictRunIsAllOnQnnAndRecordsWhatWasPassed) {
    const eb::BenchmarkResult r = eb::run_benchmark(shared_runtime(), npu_config(kAllOnNpu));
    ASSERT_FALSE(r.failure) << r.failure->error.kind << ": " << r.failure->error.message;
    ASSERT_EQ(r.timing.trials.size(), 2u);

    const eb::ProviderDescription& p = r.runtime.provider;
    EXPECT_EQ(p.execution_provider, "QNNExecutionProvider");
    ASSERT_TRUE(p.ep_device);
    EXPECT_EQ(p.ep_device->device.device_type, "NPU");
    EXPECT_EQ(p.ep_device->candidate_count, 1);
    // Exactly the options passed, key by key; the backend by absolute path.
    ASSERT_EQ(p.ep_options.size(), 3u);
    const fs::path backend = fs::path(std::u8string(p.ep_options.at("backend_path").begin(),
                                                    p.ep_options.at("backend_path").end()));
    EXPECT_TRUE(backend.is_absolute());
    EXPECT_EQ(backend, shared_runtime().bundle_dir() / "QnnHtp.dll");
    EXPECT_EQ(p.ep_options.at("htp_performance_mode"), "burst");
    EXPECT_EQ(p.ep_options.at("enable_htp_fp16_precision"), "1");
    EXPECT_EQ(p.session_config,
              (std::map<std::string, std::string>{{"ep.context_enable", "0"},
                                                  {"session.disable_cpu_ep_fallback", "1"}}));
    ASSERT_TRUE(p.qnn);
    EXPECT_EQ(p.qnn->backend, "htp");
    EXPECT_TRUE(p.qnn->unlisted_options_default);
    // The version resource of the QnnHtp.dll that was loaded, from the bundle.
    ASSERT_TRUE(r.runtime.qnn_backend_file_version);
    EXPECT_TRUE(eb::is_same_path(r.runtime.qnn_backend_file_version->module_path.native(),
                                 (shared_runtime().bundle_dir() / "QnnHtp.dll").native()));
    ASSERT_TRUE(r.runtime.qnn_backend_file_version->product_version);
    EXPECT_TRUE(contains(*r.runtime.qnn_backend_file_version->product_version, "."));
    ASSERT_TRUE(r.effective_precision);
    EXPECT_EQ(*r.effective_precision,
              (eb::EffectivePrecision{"fp32", "fp16", "enable_htp_fp16_precision"}));

    // Precision: FP32 stored, FP16 computed.
    EXPECT_EQ(r.model.precision.stored_precision, "fp32");
    EXPECT_EQ(r.compute_precision, "fp16");
    EXPECT_TRUE(r.model.precision_notes);

    // Placement: everything on QNN, validated by the QNN EP.
    const eb::NodeAssignment& na = r.execution.node_assignment;
    EXPECT_TRUE(r.execution.full_device_execution_verified);
    EXPECT_TRUE(na.all_on("QNNExecutionProvider"));
    EXPECT_EQ(na.verification_method, "combined");
    ASSERT_TRUE(na.qnn_validation);
    EXPECT_FALSE(na.qnn_validation->passes.empty());
    EXPECT_EQ(na.qnn_validation->failed_nodes(), 0);
    EXPECT_FALSE(has_warning(r, "cpu_fallback_nodes"));
    EXPECT_FALSE(has_warning(r, "strict_mode_off"));

    // The HTP backend was loaded from the bundle while a session existed.
    const bool backend_recorded =
        std::any_of(r.runtime.loaded_modules.begin(), r.runtime.loaded_modules.end(),
                    [&backend](const eb::LoadedModule& m) {
                        return _wcsicmp(m.path.filename().c_str(), L"QnnHtp.dll") == 0 &&
                               eb::is_same_path(m.path.native(), backend.native());
                    });
    EXPECT_TRUE(backend_recorded);

    TempDir out;
    write_and_validate(r, out);
}

// Batch equivalence depends on the compile-relevant EP options: the image INT8
// MobileNetV2 at batch 8 computes other outputs than at batch 1 with the default
// graph finalization (batch_not_equivalent), the same outputs with
// --htp-graph-finalization-mode 1, and with a mode no entry was made for the
// default-mode answer is not reused (not available, with a note).
TEST_F(QnnProviderNpu, BatchEquivalenceFollowsTheGraphFinalizationMode) {
    const fs::path model = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred() / "models" /
                           "mobilenetv2" / "mobilenetv2_int8_b8.onnx";
    if (!fs::is_regular_file(model))
        GTEST_SKIP() << model.string() << " is not present (see docs/models.md)";
    const auto has_note = [](const eb::BenchmarkResult& r, const std::string& code) {
        return std::any_of(r.notes.begin(), r.notes.end(),
                           [&code](const eb::ResultWarning& n) { return n.code == code; });
    };
    for (const char* mode : {"", "1", "2"}) {
        SCOPED_TRACE(std::string("finalization mode '") + mode + "'");
        eb::RunConfig c = npu_config(model);
        c.npu_fp16 = false;
        c.trials = 1;
        if (*mode)
            c.htp_graph_finalization_mode = mode;
        const eb::BenchmarkResult r = eb::run_benchmark(shared_runtime(), c);
        ASSERT_FALSE(r.failure) << r.failure->error.kind << ": " << r.failure->error.message;
        const auto& options = r.runtime.provider.ep_options;
        EXPECT_EQ(options.contains("htp_graph_finalization_optimization_mode"), *mode != 0);
        if (*mode)
            EXPECT_EQ(options.at("htp_graph_finalization_optimization_mode"), mode);
        ASSERT_TRUE(r.batch_equivalence);
        const eb::BatchEquivalence& be = *r.batch_equivalence;
        const std::string m = mode;
        if (m.empty()) {
            EXPECT_EQ(be.bitwise_equal_to_b1, false) << be.reason;
            EXPECT_EQ(be.predictions_equal_to_b1, false);
            EXPECT_TRUE(has_warning(r, "batch_not_equivalent"));
        } else if (m == "1") {
            EXPECT_EQ(be.bitwise_basis, "reference_outputs") << be.reason;
            EXPECT_EQ(be.bitwise_equal_to_b1, true);
            EXPECT_EQ(be.predictions_equal_to_b1, true);
            EXPECT_EQ(be.predictions_basis, "bitwise");
            EXPECT_FALSE(has_warning(r, "batch_not_equivalent"));
            EXPECT_FALSE(has_note(r, "batch_equivalence_options_not_covered"));
        } else {
            EXPECT_EQ(be.bitwise_basis, "not_available");
            EXPECT_FALSE(be.predictions_equal_to_b1);
            EXPECT_TRUE(be.options_not_covered) << be.reason;
            EXPECT_FALSE(has_warning(r, "batch_not_equivalent"));
            EXPECT_TRUE(has_note(r, "batch_equivalence_options_not_covered"));
        }
        TempDir out;
        write_and_validate(r, out);
    }
}

TEST_F(QnnProviderNpu, WithoutStrictModeUnsupportedNodesRunOnTheCpuWithAWarning) {
    eb::RunConfig c = npu_config(kUnsupported);
    c.strict = false;
    const eb::BenchmarkResult r = eb::run_benchmark(shared_runtime(), c);
    ASSERT_FALSE(r.failure) << r.failure->error.kind << ": " << r.failure->error.message;
    EXPECT_EQ(r.runtime.provider.session_config.at("session.disable_cpu_ep_fallback"), "0");
    const eb::NodeAssignment& na = r.execution.node_assignment;
    EXPECT_EQ(na.nodes_by_ep.at("CPUExecutionProvider"), 1);  // Det
    EXPECT_EQ(na.nodes_by_ep.at("QNNExecutionProvider"), 2);  // one partition per Relu
    EXPECT_FALSE(r.execution.full_device_execution_verified);
    ASSERT_TRUE(na.qnn_validation);
    ASSERT_EQ(na.qnn_validation->rejected_nodes.size(), 1u);
    EXPECT_EQ(na.qnn_validation->rejected_nodes.front().op_type, "Det");
    EXPECT_TRUE(has_warning(r, "cpu_fallback_nodes"));
    EXPECT_TRUE(has_warning(r, "strict_mode_off"));
    EXPECT_FALSE(has_warning(r, "nodes_not_on_requested_ep"));

    TempDir out;
    write_and_validate(r, out);
}

TEST_F(QnnProviderNpu, StrictRefusalIsExit5WithTheOrtError) {
    const eb::BenchmarkResult r = eb::run_benchmark(shared_runtime(), npu_config(kUnsupported));
    ASSERT_TRUE(r.failure);
    const eb::Error& e = r.failure->error;
    EXPECT_EQ(e.exit_code, ExitCode::strict_mode_refusal) << e.kind << ": " << e.message;
    EXPECT_EQ(e.kind, "strict_mode_refusal");
    EXPECT_EQ(e.ort_error_name, "ORT_FAIL");
    EXPECT_EQ(r.failure->stage, "session_creation");
    EXPECT_TRUE(contains(e.message, "Det node 'unsupported_det'")) << e.message;
    ASSERT_TRUE(e.unsupported_nodes);
    EXPECT_EQ(*e.unsupported_nodes, (std::vector<eb::NamedNode>{{"unsupported_det", "Det"}}));
    EXPECT_TRUE(r.timing.trials.empty());
    const auto doc = eb::to_result_json(r, std::nullopt);
    EXPECT_EQ(doc.at("error").at("kind"), "strict_mode_refusal");
    EXPECT_EQ(doc.at("error").at("unsupported_nodes").size(), 1u);

    TempDir out;
    write_and_validate(r, out);
}

TEST_F(QnnProviderNpu, ModulesAreCheckedAfterEveryQnnSessionCreation) {
    // A QNN-named module from outside the bundle appears after the provider was
    // created (the check after registration passed), just before the session of
    // trial 2: the check after that session's creation must stop the run.
    TempDir dir(L"edgebench_foreign_qnn");
    const fs::path foreign = dir.path() / "QnnForeignTest.dll";
    fs::copy_file(kSystemNonOrtDll, foreign);
    HMODULE module = nullptr;
    const eb::StopHook load_before_trial_2 = [&](eb::StopPoint point, int trial, int) {
        if (point == eb::StopPoint::before_session && trial == 1 && !module)
            module = LoadLibraryExW(foreign.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    };
    const eb::BenchmarkResult r =
        eb::run_benchmark(shared_runtime(), npu_config(kAllOnNpu), {}, load_before_trial_2);
    ASSERT_NE(module, nullptr);
    FreeLibrary(module);
    ASSERT_TRUE(r.failure);
    EXPECT_EQ(r.failure->error.exit_code, ExitCode::ort_load_failure);
    EXPECT_EQ(r.failure->error.kind, "module_outside_bundle");
    EXPECT_EQ(r.failure->stage, "session_creation");
    EXPECT_TRUE(contains(r.failure->error.message, "trial 2 of 2")) << r.failure->error.message;
    EXPECT_TRUE(contains(r.failure->error.message, "QnnForeignTest.dll"));
    EXPECT_EQ(r.timing.trials.size(), 1u);
    const bool recorded =
        std::any_of(r.runtime.loaded_modules.begin(), r.runtime.loaded_modules.end(),
                    [](const eb::LoadedModule& m) { return m.name == "QnnForeignTest.dll"; });
    EXPECT_TRUE(recorded);

    TempDir out;
    write_and_validate(r, out);
}

// --------------------------------------------------------------------------- edgebench.exe

TEST(RunNpu, StrictRefusalExits5AndWritesAFailedResult) {
    SKIP_WITHOUT_NPU();
    TempDir out;
    const ProcessResult r =
        run_process(kEdgebenchExe, npu_run_args(kUnsupported, out.path()), kFixtureDir);
    EXPECT_EQ(r.exit_code, static_cast<uint32_t>(ExitCode::strict_mode_refusal)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [strict_mode_refusal] ")) << describe(r);
    EXPECT_TRUE(contains(r.err, "ORT_FAIL")) << describe(r);
    EXPECT_TRUE(contains(r.err, "hint: ")) << describe(r);
    const fs::path result = written_result(out.path());
    ASSERT_FALSE(result.empty()) << describe(r);
    const json doc = json::parse(std::ifstream(result));
    EXPECT_EQ(doc.at("status"), "failed");
    EXPECT_EQ(doc.at("error").at("code"), "ORT_FAIL");
    EXPECT_EQ(doc.at("config").at("strict_mode"), true);
    EXPECT_EQ(doc.at("config").at("requested").at("compute_precision"), "fp16");
    eb::test::validate_or_skip(result);
}

TEST(RunNpu, MissingHtpBackendExits4NamingTheFile) {
    SKIP_WITHOUT_NPU();
    const NpuBundle bundle({L"QnnHtp.dll"});
    TempDir out;
    const ProcessResult r =
        run_process(bundle.exe, npu_run_args(kAllOnNpu, out.path()), bundle.dir.path());
    EXPECT_EQ(r.exit_code, static_cast<uint32_t>(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [qnn_backend_not_bundled] ")) << describe(r);
    EXPECT_TRUE(contains(r.err, eb::path_to_utf8(eb::resolve_path(bundle.dir.path())) +
                                    "\\QnnHtp.dll does not exist"))
        << describe(r);
    const fs::path result = written_result(out.path());
    ASSERT_FALSE(result.empty()) << describe(r);
    EXPECT_EQ(json::parse(std::ifstream(result)).at("error").at("code"), "qnn_backend_not_bundled");
    eb::test::validate_or_skip(result);
}

TEST(RunNpu, BackendStubFromOutsideTheBundleExits4) {
    SKIP_WITHOUT_NPU();
    // Without its architecture stub (QnnHtpV73Stub.dll, ...) the HTP backend loads
    // the copies installed with the display driver: modules outside the bundle,
    // which only the check after session creation can see.
    const NpuBundle bundle({L"QnnHtpV*"});
    TempDir out;
    const ProcessResult r =
        run_process(bundle.exe, npu_run_args(kAllOnNpu, out.path()), bundle.dir.path());
    EXPECT_EQ(r.exit_code, static_cast<uint32_t>(ExitCode::ort_load_failure)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [module_outside_bundle] trial 1 of 1: ")) << describe(r);
    const fs::path result = written_result(out.path());
    ASSERT_FALSE(result.empty()) << describe(r);
    const json doc = json::parse(std::ifstream(result));
    EXPECT_EQ(doc.at("error").at("stage"), "session_creation");
    eb::test::validate_or_skip(result);
}

#ifdef EDGEBENCH_ORT_PROBE_PATH
TEST(OrtProbeNpu, StrictRefusalExits5) {
    SKIP_WITHOUT_NPU();
    const fs::path probe = fs::path(EDGEBENCH_ORT_PROBE_PATH).make_preferred();
    const ProcessResult r = run_process(
        probe, {L"--model", kUnsupported.native(), L"--target", L"npu", L"--fp16", L"on"},
        probe.parent_path());
    EXPECT_EQ(r.exit_code, static_cast<uint32_t>(ExitCode::strict_mode_refusal)) << describe(r);
    EXPECT_TRUE(contains(r.err, "error: [strict_mode_refusal] ")) << describe(r);
}
#endif
