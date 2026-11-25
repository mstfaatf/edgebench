// NPU settings without ONNX Runtime: the QNN EP options and session config
// entries edgebench passes, the precision rules, the NPU-only CLI options, and
// the parsing and classification of QNN EP log messages (from real logs in
// scripts/tests/fixtures/qnn_logs/).

#include <gtest/gtest.h>

#include <onnxruntime_c_api.h>

#include <fstream>
#include <regex>
#include <string>
#include <vector>

#include "edgebench/cli.hpp"
#include "edgebench/npu_options.hpp"
#include "edgebench/qnn_ep.hpp"
#include "edgebench/qnn_log.hpp"
#include "edgebench/run_config.hpp"
#include "test_files.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using eb::ExitCode;

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

eb::RunConfig npu_config() {
    eb::RunConfig c;
    c.model = "model.onnx";
    c.provider = eb::ProviderKind::npu;
    return c;
}

eb::ModelPrecision int8_qdq() {
    return {"int8", "static", "qdq", "uint8", "int8"};
}

eb::ModelPrecision fp32() {
    return {"fp32", "none", std::nullopt, std::nullopt, std::nullopt};
}

// The bare messages of a verbose ONNX Runtime log as the Runtime's logging
// callback receives them: the prefix "<time> [V:onnxruntime:, <source>] " and the
// console colour codes removed, continuation lines appended to their message.
std::vector<std::string> log_messages(const std::string& fixture) {
    const fs::path path =
        fs::path(EDGEBENCH_SOURCE_DIR) / "scripts" / "tests" / "fixtures" / "qnn_logs" / fixture;
    std::ifstream in(path);
    EXPECT_TRUE(in) << path;
    static const std::regex kAnsi("\x1b\\[[0-9;]*m");
    static const std::regex kPrefix(R"(\[[VIWEF]:onnxruntime:[^,]*, [^\]]*\] (.*))");
    std::vector<std::string> messages;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        line = std::regex_replace(line, kAnsi, "");
        std::smatch m;
        if (std::regex_search(line, m, kPrefix))
            messages.push_back(m[1].str());
        else if (!messages.empty())
            messages.back() += "\n" + line;
    }
    return messages;
}

eb::Error ort_fail(const std::string& message) {
    return eb::ort_error("session_creation_failed", "creating a session for model.onnx", ORT_FAIL,
                         message);
}

const std::string kStrictRefusal =
    "This session contains graph nodes that are assigned to the default CPU EP, but fallback "
    "to CPU EP has been explicitly disabled by the user.";

eb::ParseResult parse(std::vector<std::string> args, const eb::test::TempModel& model) {
    args.insert(args.begin(), {"run", "--model", model.path().string()});
    return eb::parse_command_line(args);
}

}  // namespace

// --------------------------------------------------------------------------- options

TEST(QnnEpOptions, AreExactlyTheBackendAndThePerformanceModeByDefault) {
    const fs::path backend = fs::path(LR"(C:\bundle\QnnHtp.dll)");
    const auto options = eb::qnn_ep_options(npu_config(), backend);
    const std::map<std::string, std::string> expected = {
        {"backend_path", R"(C:\bundle\QnnHtp.dll)"}, {"htp_performance_mode", "burst"}};
    EXPECT_EQ(options, expected);
}

TEST(QnnEpOptions, Fp16PrecisionOnlyWithTheFlag) {
    eb::RunConfig c = npu_config();
    c.npu_fp16 = true;
    c.htp_performance_mode = "balanced";
    const auto options = eb::qnn_ep_options(c, LR"(C:\bundle\QnnHtp.dll)");
    EXPECT_EQ(options.size(), 3u);
    EXPECT_EQ(options.at("enable_htp_fp16_precision"), "1");
    EXPECT_EQ(options.at("htp_performance_mode"), "balanced");
}

TEST(QnnSessionConfig, StrictByDefaultAndNoContextCache) {
    eb::RunConfig c = npu_config();
    EXPECT_EQ(eb::qnn_session_config(c),
              (std::map<std::string, std::string>{{"ep.context_enable", "0"},
                                                  {"session.disable_cpu_ep_fallback", "1"}}));
    EXPECT_TRUE(eb::strict_mode(c));
    c.strict = false;
    EXPECT_EQ(eb::qnn_session_config(c).at("session.disable_cpu_ep_fallback"), "0");
    EXPECT_FALSE(eb::strict_mode(c));
    EXPECT_FALSE(eb::strict_mode(eb::RunConfig{}));  // the CPU provider has no fallback
}

TEST(HtpPerformanceMode, AcceptsTheModesTheEpDocuments) {
    for (std::string_view mode : eb::kHtpPerformanceModes)
        EXPECT_TRUE(eb::is_htp_performance_mode(mode)) << mode;
    for (const char* mode : {"", "turbo", "Burst", "burst "})
        EXPECT_FALSE(eb::is_htp_performance_mode(mode)) << mode;
    EXPECT_EQ(eb::effective_htp_performance_mode(npu_config()), "burst");
    EXPECT_TRUE(contains(eb::htp_performance_modes_list(), "burst, balanced, default, "));
}

TEST(QnnHtpBackend, MissingFilesAreALoadFailureNamingThem) {
    const eb::test::TempModel marker("marker.txt");  // a fresh temporary directory
    const fs::path dir = marker.path().parent_path();
    eb::Result<fs::path> none = eb::check_qnn_htp_backend(dir);
    ASSERT_FALSE(none.ok());
    EXPECT_EQ(none.error().exit_code, ExitCode::ort_load_failure);
    EXPECT_EQ(none.error().kind, "qnn_backend_not_bundled");
    EXPECT_TRUE(contains(none.error().message, "QnnHtp.dll")) << none.error().message;
    EXPECT_TRUE(contains(none.error().message, "QnnHtpPrepare.dll")) << none.error().message;

    std::ofstream(dir / "QnnHtp.dll") << "x";
    eb::Result<fs::path> one = eb::check_qnn_htp_backend(dir);
    ASSERT_FALSE(one.ok());
    EXPECT_TRUE(contains(one.error().message, "file ")) << one.error().message;
    EXPECT_FALSE(contains(one.error().message, "QnnHtp.dll,")) << one.error().message;

    std::ofstream(dir / "QnnHtpPrepare.dll") << "x";
    eb::Result<fs::path> ok = eb::check_qnn_htp_backend(dir);
    ASSERT_TRUE(ok.ok()) << ok.error().message;
    EXPECT_TRUE(ok.value().is_absolute());
    EXPECT_EQ(ok.value().filename(), "QnnHtp.dll");
    EXPECT_FALSE(eb::qnn_ep_hint("qnn_backend_not_bundled").empty());
    EXPECT_FALSE(eb::qnn_ep_hint("strict_mode_refusal").empty());
}

// --------------------------------------------------------------------------- precision

TEST(ComputePrecision, CpuRunsRequestNothing) {
    eb::RunConfig cpu;
    const auto p = eb::requested_compute_precision(cpu, fp32());
    ASSERT_TRUE(p.ok());
    EXPECT_FALSE(p.value().has_value());
}

TEST(ComputePrecision, Int8QdqRunsAsInt8) {
    const auto p = eb::requested_compute_precision(npu_config(), int8_qdq());
    ASSERT_TRUE(p.ok()) << p.error().message;
    EXPECT_EQ(p.value(), "int8");
    EXPECT_FALSE(eb::precision_notes(int8_qdq(), p.value()));
}

TEST(ComputePrecision, Fp32WithoutTheFlagIsRefused) {
    const auto p = eb::requested_compute_precision(npu_config(), fp32());
    ASSERT_FALSE(p.ok());
    EXPECT_EQ(p.error().exit_code, ExitCode::usage_error);
    EXPECT_EQ(p.error().kind, "npu_fp32_needs_fp16");
    EXPECT_TRUE(contains(p.error().message, "the NPU has no FP32 compute")) << p.error().message;
    EXPECT_TRUE(contains(p.error().message, "--npu-fp16")) << p.error().message;
}

TEST(ComputePrecision, Fp32WithTheFlagIsFp16Compute) {
    eb::RunConfig c = npu_config();
    c.npu_fp16 = true;
    const auto p = eb::requested_compute_precision(c, fp32());
    ASSERT_TRUE(p.ok()) << p.error().message;
    EXPECT_EQ(p.value(), "fp16");
    const auto notes = eb::precision_notes(fp32(), p.value());
    ASSERT_TRUE(notes);
    EXPECT_TRUE(contains(*notes, "FP16")) << *notes;
}

TEST(ComputePrecision, TheFlagOnAnInt8ModelIsAUsageError) {
    eb::RunConfig c = npu_config();
    c.npu_fp16 = true;
    const auto p = eb::requested_compute_precision(c, int8_qdq());
    ASSERT_FALSE(p.ok());
    EXPECT_EQ(p.error().exit_code, ExitCode::usage_error);
    EXPECT_EQ(p.error().kind, "npu_fp16_needs_fp32_model");
}

TEST(ComputePrecision, OtherStoredPrecisionsAreNotSupported) {
    const std::vector<eb::ModelPrecision> cases = {
        {"fp16", "none", std::nullopt, std::nullopt, std::nullopt},
        {"mixed", "none", std::nullopt, std::nullopt, std::nullopt},
        {"int8", "dynamic", "qdq", "uint8", "int8"},
        {"int8", "static", "qoperator", "uint8", "int8"},
        {"mixed", "static", "qdq", "uint16", "int8"},
    };
    for (bool flag : {false, true}) {
        eb::RunConfig c = npu_config();
        c.npu_fp16 = flag;
        for (const eb::ModelPrecision& stored : cases) {
            const auto p = eb::requested_compute_precision(c, stored);
            ASSERT_FALSE(p.ok()) << stored.stored_precision;
            EXPECT_EQ(p.error().exit_code, ExitCode::usage_error);
            EXPECT_EQ(p.error().kind, "npu_precision_unsupported") << stored.stored_precision;
        }
    }
}

// --------------------------------------------------------------------------- CLI

TEST(NpuCli, OptionsAreParsed) {
    const eb::test::TempModel model("m.onnx");
    const eb::ParseResult r = parse({"--provider", "npu", "--npu-fp16", "--no-strict",
                                     "--htp-performance-mode", "high_performance"},
                                    model);
    ASSERT_FALSE(r.exit) << r.err;
    EXPECT_EQ(r.run.provider, eb::ProviderKind::npu);
    EXPECT_TRUE(r.run.npu_fp16);
    EXPECT_FALSE(r.run.strict);
    EXPECT_EQ(r.run.htp_performance_mode, "high_performance");
    EXPECT_EQ(eb::default_result_name(r.run), "m_npu_iobinding_t4");
}

TEST(NpuCli, DefaultsAreStrictAndBurst) {
    const eb::test::TempModel model("m.onnx");
    const eb::ParseResult r = parse({"--provider", "npu"}, model);
    ASSERT_FALSE(r.exit) << r.err;
    EXPECT_TRUE(r.run.strict);
    EXPECT_FALSE(r.run.npu_fp16);
    EXPECT_FALSE(r.run.htp_performance_mode);  // given explicitly only
    EXPECT_EQ(eb::effective_htp_performance_mode(r.run), "burst");
    const nlohmann::ordered_json j = eb::resolved_config_json(r.run);
    EXPECT_EQ(j.at("config").at("strict_mode"), true);
    EXPECT_EQ(j.at("config").at("requested").at("execution_provider"), "QNNExecutionProvider");
    EXPECT_EQ(j.at("config").at("requested").at("device_type"), "NPU");
    EXPECT_EQ(j.at("config").at("ep_context_cache").at("state"), "disabled");
    EXPECT_EQ(j.at("config").at("ep_context_cache").at("enabled"), false);
}

TEST(NpuCli, UnknownPerformanceModeIsAUsageError) {
    const eb::test::TempModel model("m.onnx");
    const eb::ParseResult r =
        parse({"--provider", "npu", "--htp-performance-mode", "turbo"}, model);
    ASSERT_TRUE(r.exit);
    EXPECT_EQ(*r.exit, ExitCode::usage_error);
    EXPECT_TRUE(contains(r.err, "unsupported HTP performance mode 'turbo'")) << r.err;
    EXPECT_TRUE(contains(r.err, "sustained_high_performance")) << r.err;
}

TEST(NpuCli, NpuOptionsWithTheCpuProviderAreUsageErrors) {
    const eb::test::TempModel model("m.onnx");
    for (const std::vector<std::string>& args :
         {std::vector<std::string>{"--npu-fp16"}, std::vector<std::string>{"--no-strict"},
          std::vector<std::string>{"--htp-performance-mode", "burst"},
          std::vector<std::string>{"--provider", "cpu", "--npu-fp16"}}) {
        const eb::ParseResult r = parse(args, model);
        ASSERT_TRUE(r.exit) << args.front();
        EXPECT_EQ(*r.exit, ExitCode::usage_error);
        EXPECT_TRUE(contains(r.err, "is only allowed with --provider npu")) << r.err;
    }
}

// --------------------------------------------------------------------------- QNN EP logs

TEST(QnnLog, AllOnQnnValidationOfARealModel) {
    const auto v = eb::parse_qnn_validation(log_messages("s8w_all_on_qnn.log"));
    ASSERT_TRUE(v);
    ASSERT_EQ(v->passes.size(), 2u);
    EXPECT_EQ(v->passes[0], (eb::QnnValidationPass{313, 0}));
    EXPECT_EQ(v->passes[1], (eb::QnnValidationPass{319, 0}));
    EXPECT_TRUE(v->rejected_nodes.empty());
    EXPECT_EQ(v->failed_nodes(), 0);
    EXPECT_TRUE(eb::parse_qnn_unsupported_nodes(log_messages("s8w_all_on_qnn.log")).empty());
}

TEST(QnnLog, RejectedNodeWithItsReason) {
    for (const char* fixture : {"det_strict.log", "det_non_strict.log"}) {
        const auto v = eb::parse_qnn_validation(log_messages(fixture));
        ASSERT_TRUE(v) << fixture;
        ASSERT_EQ(v->passes.size(), 2u);
        EXPECT_EQ(v->passes[0], (eb::QnnValidationPass{2, 1}));
        EXPECT_EQ(v->passes[1], (eb::QnnValidationPass{2, 1}));
        ASSERT_EQ(v->rejected_nodes.size(), 1u) << fixture;  // the same node in both passes
        const eb::QnnRejectedNode& n = v->rejected_nodes.front();
        EXPECT_EQ(n.op_type, "Det");
        EXPECT_EQ(n.node_name, "unsupported_det");
        EXPECT_EQ(n.node_index, 1);
        ASSERT_TRUE(n.reason);
        EXPECT_TRUE(contains(*n.reason, "are not supported by QNN EP")) << *n.reason;
    }
    const auto unsupported = eb::parse_qnn_unsupported_nodes(log_messages("det_strict.log"));
    ASSERT_EQ(unsupported.size(), 1u);  // logged once per pass
    EXPECT_EQ(unsupported.front().op_type, "Det");
    EXPECT_EQ(unsupported.front().node_name, "unsupported_det");
    EXPECT_EQ(eb::describe_nodes(unsupported), "Det node 'unsupported_det'");
}

TEST(QnnLog, NoValidationMessagesIsNotAValidation) {
    EXPECT_FALSE(eb::parse_qnn_validation({}));
    EXPECT_FALSE(eb::parse_qnn_validation({"Node placements", "unrelated"}));
}

TEST(QnnSessionFailure, StrictRefusalIsExit5WithTheUnsupportedNodes) {
    std::vector<std::string> logged;
    for (const std::string& m : log_messages("det_strict.log")) {
        if (contains(m, "Unsupported nodes") || contains(m, "does not support"))
            logged.push_back(m);  // what reaches the callback without verbose logging
    }
    const eb::Error e = eb::classify_qnn_session_failure(ort_fail(kStrictRefusal), true, logged);
    EXPECT_EQ(e.exit_code, ExitCode::strict_mode_refusal);
    EXPECT_EQ(e.kind, "strict_mode_refusal");
    EXPECT_EQ(e.ort_error_name, "ORT_FAIL");
    EXPECT_TRUE(contains(e.message, kStrictRefusal)) << e.message;
    EXPECT_TRUE(contains(e.message, "Det node 'unsupported_det'")) << e.message;
}

TEST(QnnSessionFailure, OnlyStrictModeFailuresAreRefusals) {
    // Without strict mode, the same text cannot be a refusal of ours.
    EXPECT_EQ(eb::classify_qnn_session_failure(ort_fail(kStrictRefusal), false, {}).exit_code,
              ExitCode::ort_runtime_error);
    // Another ORT_FAIL cause, seen when QnnHtpPrepare.dll is missing.
    const eb::Error layout = eb::classify_qnn_session_failure(
        ort_fail("Node 'Conv_token_1' OpType:Conv with domain:com.ms.internal.nhwc was inserted "
                 "using the NHWC format as requested by QNNExecutionProvider, but was not "
                 "selected by that EP."),
        true, {});
    EXPECT_EQ(layout.exit_code, ExitCode::ort_runtime_error);
    EXPECT_EQ(layout.kind, "session_creation_failed");
    // The refusal text under another error code.
    const eb::Error other = eb::classify_qnn_session_failure(
        eb::ort_error("session_creation_failed", "ctx", ORT_INVALID_ARGUMENT, kStrictRefusal), true,
        {});
    EXPECT_EQ(other.exit_code, ExitCode::ort_runtime_error);
}

TEST(QnnSessionFailure, BackendLoadFailureIsExit4EvenThoughOrtReportsARefusal) {
    // Logged by the QNN EP when QnnHtp.dll cannot be loaded: it then claims no node,
    // and ONNX Runtime reports the strict refusal for the whole graph.
    const std::vector<std::string> logged = {
        "QNN SetupBackend failed Unable to load backend, error: load library failed",
        "OrtEp::GetCapability() for QNNExecutionProvider failed with error: [ONNXRuntimeError] "
        ": 11 : EP_FAIL : QNN SetupBackend failed Unable to load backend, error: load library "
        "failed"};
    for (bool strict : {true, false}) {
        const eb::Error e =
            eb::classify_qnn_session_failure(ort_fail(kStrictRefusal), strict, logged);
        EXPECT_EQ(e.exit_code, ExitCode::ort_load_failure);
        EXPECT_EQ(e.kind, "qnn_backend_load_failed");
        EXPECT_EQ(e.ort_error_name, "ORT_FAIL");
        EXPECT_TRUE(contains(e.message, "Unable to load backend")) << e.message;
    }
    EXPECT_TRUE(eb::mentions_qnn_backend_failure(logged));
    EXPECT_FALSE(eb::mentions_qnn_backend_failure(log_messages("det_strict.log")));
}
