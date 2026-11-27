// The doctor's pass / warn / fail classification of every check, on fakes (no
// ONNX Runtime, no files): edgebench/doctor_checks.hpp.

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "edgebench/doctor_checks.hpp"

namespace eb = edgebench;
using eb::CheckStatus;
using eb::ExitCode;

namespace {

const std::string kSha(64, 'a');
const std::string kOtherSha(64, 'b');

std::vector<eb::FilePin> pins(std::initializer_list<const char*> names) {
    std::vector<eb::FilePin> out;
    for (const char* n : names)
        out.push_back({n, 100, kSha});
    return out;
}

std::vector<eb::BundleFile> matching(const std::vector<eb::FilePin>& p) {
    std::vector<eb::BundleFile> out;
    for (const eb::FilePin& pin : p)
        out.push_back({pin.name, "C:/bundle/" + pin.name, true, 100, kSha, "1.0", "1.0.0.0"});
    return out;
}

eb::OrtLoadResult loaded() {
    eb::OrtLoadResult r;
    r.status = eb::OrtLoadStatus::ok;
    r.ort_version = "1.32.0";
    r.api_version = 32;
    r.dll_path = "C:/bundle/onnxruntime.dll";
    return r;
}

eb::EpDeviceInfo device(const char* ep, const char* type, std::uint32_t vendor_id = 0) {
    eb::EpDeviceInfo d;
    d.ep_name = ep;
    d.device_type = type;
    d.vendor_id = vendor_id;
    return d;
}

eb::Error error(ExitCode code, const char* kind) {
    return eb::Error{code, kind, std::string(kind) + " happened"};
}

}  // namespace

TEST(DoctorOrtBundle, PassWarnAndFail) {
    const auto p = pins({"onnxruntime.dll", "onnxruntime_providers_shared.dll"});
    eb::DoctorCheck c = eb::check_ort_bundle(loaded(), matching(p), p, {}, {}, false);
    EXPECT_EQ(c.status, CheckStatus::pass) << c.summary;
    EXPECT_EQ(c.details.at("files").size(), 2u);
    EXPECT_TRUE(c.details.at("files")[0].at("matches_pin").get<bool>());

    // Emulation: a warning.
    c = eb::check_ort_bundle(loaded(), matching(p), p, {}, {}, true);
    EXPECT_EQ(c.status, CheckStatus::warn);
    EXPECT_EQ(c.code, "process_emulated");
    EXPECT_NE(c.summary.find("onnxruntime 1.32.0"), std::string::npos) << c.summary;

    // A file that differs from its pin, or is missing: exit code 4.
    std::vector<eb::BundleFile> files = matching(p);
    files[1].sha256 = kOtherSha;
    c = eb::check_ort_bundle(loaded(), files, p, {}, {}, false);
    EXPECT_EQ(c.status, CheckStatus::fail);
    EXPECT_EQ(c.code, "bundle_file_mismatch");
    EXPECT_EQ(c.exit_code, ExitCode::ort_load_failure);
    EXPECT_NE(c.summary.find("onnxruntime_providers_shared.dll"), std::string::npos) << c.summary;
    files = matching(p);
    files[0].present = false;
    EXPECT_EQ(eb::check_ort_bundle(loaded(), files, p, {}, {}, false).code, "bundle_file_mismatch");

    // A module from outside the bundle: exit code 4.
    const std::vector<eb::LoadedModule> foreign = {{"QnnHtp.dll", "C:/elsewhere/QnnHtp.dll"}};
    c = eb::check_ort_bundle(loaded(), matching(p), p, foreign, foreign, false);
    EXPECT_EQ(c.code, "module_outside_bundle");
    EXPECT_EQ(c.exit_code, ExitCode::ort_load_failure);

    // ONNX Runtime not loaded: the loader's kind, exit code 4.
    eb::OrtLoadResult failed;
    failed.status = eb::OrtLoadStatus::failed;
    failed.error_kind = eb::OrtLoadError::version_mismatch;
    failed.message = "found 1.17.1";
    c = eb::check_ort_bundle(failed, matching(p), p, {}, {}, false);
    EXPECT_EQ(c.status, CheckStatus::fail);
    EXPECT_EQ(c.code, "version_mismatch");
    EXPECT_EQ(c.exit_code, ExitCode::ort_load_failure);
}

TEST(DoctorQnnBundle, NotApplicableWithoutQnnPins) {
    const eb::DoctorCheck c = eb::check_qnn_bundle({}, {}, {});
    EXPECT_EQ(c.status, CheckStatus::not_applicable);
    EXPECT_EQ(c.details.at("registration"), "not_attempted");
    // A plugin file next to a build without the QNN EP that does not register: a failure.
    const eb::DoctorCheck bad = eb::check_qnn_bundle(
        {}, {}, {true, eb::Error{ExitCode::ort_load_failure, "ep_registration_failed", "no"}});
    EXPECT_EQ(bad.status, CheckStatus::fail);
    EXPECT_EQ(bad.details.at("registration"), "failed");
}

TEST(DoctorQnnBundle, FilesAndRegistration) {
    const auto p = pins({"onnxruntime_providers_qnn.dll", "QnnHtp.dll", "QnnSystem.dll"});
    eb::QnnRegistration ok{true, std::nullopt};
    EXPECT_EQ(eb::check_qnn_bundle(matching(p), p, ok).status, CheckStatus::pass);

    std::vector<eb::BundleFile> files = matching(p);
    files[1].present = false;  // no QnnHtp.dll
    eb::DoctorCheck c = eb::check_qnn_bundle(files, p, ok);
    EXPECT_EQ(c.status, CheckStatus::fail);
    EXPECT_EQ(c.code, "qnn_bundle_mismatch");
    EXPECT_EQ(c.exit_code, ExitCode::ort_load_failure);
    EXPECT_NE(c.summary.find("QnnHtp.dll: missing"), std::string::npos) << c.summary;

    eb::QnnRegistration refused{true, error(ExitCode::ort_load_failure, "ep_registration_failed")};
    c = eb::check_qnn_bundle(matching(p), p, refused);
    EXPECT_EQ(c.code, "ep_registration_failed");
    EXPECT_EQ(c.exit_code, ExitCode::ort_load_failure);
}

TEST(DoctorBackendVersions, AgreementAndDisagreement) {
    eb::BackendVersionFacts f;
    f.backend_build = "v2.52.0.261104115025";
    f.sdk_version = "2.52.0";
    f.source = "qnn_ep_log_backend_build_id";
    f.pinned_sdk_version = "2.52.0";
    f.backend_file_product_version = "2.52.0.261104115025";
    EXPECT_EQ(eb::check_backend_versions(f).status, CheckStatus::pass);

    eb::BackendVersionFacts other = f;
    other.pinned_sdk_version = "2.51.0";
    eb::DoctorCheck c = eb::check_backend_versions(other);
    EXPECT_EQ(c.status, CheckStatus::warn);  // never a failure
    EXPECT_EQ(c.code, "qnn_sdk_version_disagreement");

    other = f;
    other.backend_file_product_version = "2.52.0.1";
    EXPECT_EQ(eb::check_backend_versions(other).code, "qnn_backend_version_disagreement");

    other = f;
    other.backend_build.reset();
    other.sdk_version.reset();
    EXPECT_EQ(eb::check_backend_versions(other).code, "backend_build_unavailable");

    other = f;
    other.not_applicable = "no NPU to run on";
    EXPECT_EQ(eb::check_backend_versions(other).status, CheckStatus::not_applicable);
}

TEST(DoctorDevices, NpuSelectionOutcomes) {
    const eb::EpDeviceInfo cpu = device("CPUExecutionProvider", "CPU");
    const eb::EpDeviceInfo npu = device("QNNExecutionProvider", "NPU", 0x4D4F4351);
    eb::DeviceFacts f;
    f.qnn_in_build = true;
    f.qnn_registered = true;

    f.devices = std::vector<eb::EpDeviceInfo>{cpu, npu};
    f.selection = eb::select_qnn_npu(*f.devices);
    EXPECT_EQ(eb::check_devices(f).status, CheckStatus::pass);

    f.devices = std::vector<eb::EpDeviceInfo>{cpu};  // e.g. the ARM64 CI runner
    f.selection = eb::select_qnn_npu(*f.devices);
    eb::DoctorCheck c = eb::check_devices(f);
    EXPECT_EQ(c.status, CheckStatus::warn);
    EXPECT_EQ(c.code, "no_qnn_ep_devices");

    f.devices = std::vector<eb::EpDeviceInfo>{cpu, npu, npu};
    f.selection = eb::select_qnn_npu(*f.devices);
    c = eb::check_devices(f);
    EXPECT_EQ(c.status, CheckStatus::fail);
    EXPECT_EQ(c.code, "multiple_npu_devices");
    EXPECT_EQ(c.exit_code, ExitCode::ort_runtime_error);

    eb::DeviceFacts x64;  // CPU-only build
    x64.devices = std::vector<eb::EpDeviceInfo>{cpu};
    EXPECT_EQ(eb::check_devices(x64).status, CheckStatus::pass);

    eb::DeviceFacts broken;
    broken.enumeration_error = error(ExitCode::ort_runtime_error, "ep_enumeration_failed");
    EXPECT_EQ(eb::check_devices(broken).exit_code, ExitCode::ort_runtime_error);
}

TEST(DoctorNpuProbe, ByteComparison) {
    const std::vector<float> expected = {0.0f, 0.5f, 1.0f};
    std::vector<float> actual = expected;
    const auto bytes = [](const std::vector<float>& v) {
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(v.data()),
                                          v.size() * sizeof(float));
    };
    eb::ProbeOutcome o;
    eb::compare_probe_output(o, bytes(actual), bytes(expected));
    EXPECT_TRUE(o.compared);
    EXPECT_EQ(o.elements, 3u);
    EXPECT_EQ(o.differing, 0u);
    actual[2] = 1.05f;
    eb::compare_probe_output(o, bytes(actual), bytes(expected));
    EXPECT_EQ(o.differing, 1u);
    EXPECT_NEAR(*o.max_abs_diff, 0.05, 1e-6);
    // -0.0 and 0.0 compare equal as numbers but not byte for byte.
    actual = expected;
    actual[0] = -0.0f;
    eb::compare_probe_output(o, bytes(actual), bytes(expected));
    EXPECT_EQ(o.differing, 1u);
    // Another output size: every value counts as different.
    actual.push_back(2.0f);
    eb::compare_probe_output(o, bytes(actual), bytes(expected));
    EXPECT_EQ(o.differing, 3u);
}

TEST(DoctorNpuProbe, Outcomes) {
    eb::ProbeOutcome good;
    good.model = "npu_probe_int8.onnx";
    good.compute = "int8";
    good.compared = true;
    good.elements = 256;
    eb::ProbeFacts f;
    f.probes = {good, good};
    EXPECT_EQ(eb::check_npu_probe(f).status, CheckStatus::pass);

    eb::ProbeFacts mismatch = f;
    mismatch.probes[1].differing = 3;
    mismatch.probes[1].max_abs_diff = 0.05;
    eb::DoctorCheck c = eb::check_npu_probe(mismatch);
    EXPECT_EQ(c.code, "npu_probe_mismatch");
    EXPECT_EQ(c.exit_code, ExitCode::internal_error);

    // Session failures keep their exit code: strict refusal 5, backend 4.
    eb::ProbeFacts refused = f;
    refused.probes[0].error = error(ExitCode::strict_mode_refusal, "strict_mode_refusal");
    EXPECT_EQ(eb::check_npu_probe(refused).exit_code, ExitCode::strict_mode_refusal);
    eb::ProbeFacts backend = f;
    backend.probes[0].error = error(ExitCode::ort_load_failure, "qnn_backend_load_failed");
    EXPECT_EQ(eb::check_npu_probe(backend).exit_code, ExitCode::ort_load_failure);

    eb::ProbeFacts foreign = f;
    foreign.outside = {{"QnnHtpV73StubDrv.dll", "C:/Windows/System32/DriverStore/x.dll"}};
    EXPECT_EQ(eb::check_npu_probe(foreign).code, "module_outside_bundle");

    eb::ProbeFacts none;
    none.not_applicable = "no NPU to run on";
    EXPECT_EQ(eb::check_npu_probe(none).status, CheckStatus::not_applicable);
}

TEST(DoctorNpuContext, Outcomes) {
    eb::ContextFacts f;
    f.build_ms = 2000;
    f.load_ms = 300;
    f.outputs_equal = true;
    f.compatibility_info = "v2:6:2.52.0:5.52.0:73:0:0:0";
    f.htp_arch = "v73";
    EXPECT_EQ(eb::check_npu_context(f).status, CheckStatus::pass);

    eb::ContextFacts differ = f;
    differ.outputs_equal = false;
    eb::DoctorCheck c = eb::check_npu_context(differ);
    EXPECT_EQ(c.code, "ep_context_outputs_differ");
    EXPECT_EQ(c.exit_code, ExitCode::internal_error);

    eb::ContextFacts unloadable = f;
    unloadable.error = error(ExitCode::ort_runtime_error, "ep_context_load_failed");
    EXPECT_EQ(eb::check_npu_context(unloadable).exit_code, ExitCode::ort_runtime_error);

    eb::ContextFacts no_arch = f;
    no_arch.htp_arch.reset();
    EXPECT_EQ(eb::check_npu_context(no_arch).code, "htp_arch_unavailable");

    eb::ContextFacts left = f;
    left.cleanup_error = "access denied";
    EXPECT_EQ(eb::check_npu_context(left).code, "temporary_directory_left");
}

TEST(DoctorEpCacheDir, StaleFields) {
    eb::CurrentCacheEnvironment cur;
    cur.ort_version = "1.32.0";
    cur.ep_version = "2.8.0";
    cur.qnn_sdk_version = "2.52.0";
    cur.qnn_backend_file_version = "2.52.0.261104115025";
    cur.device = nlohmann::json{{"device_id", 7}, {"device_type", "NPU"}, {"vendor_id", 9}};
    nlohmann::json key = {
        {"ort_version", "1.32.0"},     {"ep_version", "2.8.0"},
        {"qnn_sdk_version", "2.52.0"}, {"qnn_backend_file_version", "2.52.0.261104115025"},
        {"device", *cur.device},       {"model_sha256", kSha}};
    EXPECT_TRUE(eb::stale_key_fields(key, cur).empty());
    key["qnn_sdk_version"] = nullptr;  // built before the SDK version was a key field
    key["device"]["device_id"] = 8;
    EXPECT_EQ(eb::stale_key_fields(key, cur),
              (std::vector<std::string>{"qnn_sdk_version", "device"}));
    eb::CurrentCacheEnvironment unknown;  // nothing known: nothing compared
    EXPECT_TRUE(eb::stale_key_fields(key, unknown).empty());
}

TEST(DoctorEpCacheDir, Outcomes) {
    eb::CacheDirFacts f;
    f.dir = "C:/cache";
    f.exists = false;
    EXPECT_EQ(eb::check_ep_cache_dir(f).status, CheckStatus::pass);  // created by the first run

    f.exists = true;
    f.writable = true;
    f.environment_known = true;
    f.entries.push_back({"C:/cache/aa", std::string(64, 'a'), kSha, "m_ctx.onnx", 1000, {}, false});
    EXPECT_EQ(eb::check_ep_cache_dir(f).status, CheckStatus::pass);

    eb::CacheDirFacts stale = f;
    stale.entries.push_back({"C:/cache/bb",
                             std::string(64, 'b'),
                             kSha,
                             "m_ctx.onnx",
                             1000,
                             {"qnn_sdk_version"},
                             false});
    eb::DoctorCheck c = eb::check_ep_cache_dir(stale);
    EXPECT_EQ(c.status, CheckStatus::warn);
    EXPECT_EQ(c.code, "stale_cache_entries");
    EXPECT_NE(c.hint.find("--prune-cache"), std::string::npos);
    EXPECT_EQ(c.details.at("stale_count"), 1);

    stale.entries.back().pruned = true;  // doctor --prune-cache
    c = eb::check_ep_cache_dir(stale);
    EXPECT_EQ(c.status, CheckStatus::pass);
    EXPECT_EQ(c.details.at("pruned_count"), 1);

    eb::CacheDirFacts foreign = f;
    foreign.foreign = {"C:/cache/mine"};
    EXPECT_EQ(eb::check_ep_cache_dir(foreign).code, "cache_directories_without_manifest");

    eb::CacheDirFacts unwritable = f;
    unwritable.dir_error =
        eb::Error{ExitCode::output_error, "ep_cache_dir_not_writable", "access denied"};
    c = eb::check_ep_cache_dir(unwritable);
    EXPECT_EQ(c.status, CheckStatus::fail);
    EXPECT_EQ(c.exit_code, ExitCode::output_error);
}

TEST(DoctorPower, OnlyAcAndBestPerformancePass) {
    eb::PowerState p;
    p.source = "ac";
    p.power_mode = "Best performance";
    p.power_mode_known = true;
    p.battery_saver = false;
    EXPECT_EQ(eb::check_power(p).status, CheckStatus::pass);

    eb::PowerState balanced = p;
    balanced.power_mode = "Balanced";
    eb::DoctorCheck c = eb::check_power(balanced);
    EXPECT_EQ(c.status, CheckStatus::warn);
    EXPECT_EQ(c.code, "power_mode_not_best_performance");
    EXPECT_NE(c.hint.find("Published latency runs require AC power"), std::string::npos);

    eb::PowerState battery = p;
    battery.source = "battery";
    EXPECT_EQ(eb::check_power(battery).code, "battery_power");
    eb::PowerState saver = p;
    saver.battery_saver = true;
    EXPECT_EQ(eb::check_power(saver).code, "battery_saver_on");
    EXPECT_EQ(eb::check_power(std::nullopt).code, "power_source_unknown");
}

TEST(DoctorBackgroundLoad, ThresholdAndNoSample) {
    EXPECT_EQ(eb::check_background_load(eb::LoadSample{0.10, 1.0}).status, CheckStatus::pass);
    const eb::DoctorCheck c = eb::check_background_load(eb::LoadSample{0.125, 1.0});
    EXPECT_EQ(c.status, CheckStatus::warn);
    EXPECT_EQ(c.code, "background_cpu_load_high");
    EXPECT_NE(c.summary.find("12.5%"), std::string::npos) << c.summary;
    EXPECT_EQ(eb::check_background_load(std::nullopt).status, CheckStatus::not_applicable);
}

TEST(DoctorOverall, StatusAndExitCodeOfTheFirstFailure) {
    std::vector<eb::DoctorCheck> checks(3);
    EXPECT_EQ(eb::overall_status(checks), CheckStatus::pass);
    EXPECT_EQ(eb::overall_exit_code(checks), ExitCode::success);
    checks[2].status = CheckStatus::warn;
    checks[1].status = CheckStatus::not_applicable;
    EXPECT_EQ(eb::overall_status(checks), CheckStatus::warn);
    EXPECT_EQ(eb::overall_exit_code(checks), ExitCode::success);  // warnings allowed
    checks[2].status = CheckStatus::fail;
    checks[2].exit_code = ExitCode::internal_error;
    checks[0].status = CheckStatus::fail;
    checks[0].exit_code = ExitCode::ort_load_failure;
    EXPECT_EQ(eb::overall_status(checks), CheckStatus::fail);
    EXPECT_EQ(eb::overall_exit_code(checks), ExitCode::ort_load_failure);  // the first one
}
