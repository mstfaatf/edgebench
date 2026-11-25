// select_qnn_npu(): the pure NPU selection rule (EP name QNNExecutionProvider,
// hardware type NPU, exactly one match), on hand-built device lists.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "edgebench/npu_selection.hpp"
#include "edgebench/qnn_ep.hpp"

namespace eb = edgebench;
using eb::EpDeviceInfo;
using eb::NpuSelectionFailure;

namespace {

EpDeviceInfo device(std::string ep, std::string type, std::uint32_t device_id,
                    std::string ep_vendor = "Qualcomm") {
    EpDeviceInfo d;
    d.ep_name = std::move(ep);
    d.ep_vendor = std::move(ep_vendor);
    d.device_type = std::move(type);
    d.vendor_id = 0x4d4f4351;
    d.device_id = device_id;
    d.ep_metadata = {{"version", "2.8.0"}};
    return d;
}

EpDeviceInfo cpu() {
    EpDeviceInfo d = device("CPUExecutionProvider", "CPU", 0x7, "Microsoft");
    d.vendor = "Qualcomm";
    return d;
}
EpDeviceInfo qnn_npu(std::uint32_t id = 0x41304430) {
    return device("QNNExecutionProvider", "NPU", id);
}
EpDeviceInfo qnn_gpu() {
    return device("QNNExecutionProvider", "GPU", 0x36334330);
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// A failed selection: no index, the given failure, and an exit-code-3 error whose
// message names every enumerated device.
void expect_failure(const eb::NpuSelection& s, NpuSelectionFailure failure,
                    const std::vector<EpDeviceInfo>& devices) {
    EXPECT_FALSE(s.ok());
    EXPECT_FALSE(s.index.has_value());
    ASSERT_TRUE(s.failure.has_value());
    EXPECT_EQ(*s.failure, failure);
    const eb::Error e = s.error();
    EXPECT_EQ(e.exit_code, eb::ExitCode::ort_runtime_error);
    EXPECT_EQ(e.kind, eb::to_string(failure));
    EXPECT_FALSE(s.reason.empty());
    EXPECT_TRUE(contains(e.message, s.reason)) << e.message;
    EXPECT_TRUE(contains(e.message, "enumerated devices: ")) << e.message;
    for (const EpDeviceInfo& d : devices)
        EXPECT_TRUE(contains(e.message, eb::describe_device(d))) << e.message;
    if (devices.empty())
        EXPECT_TRUE(contains(e.message, "(none)")) << e.message;
    EXPECT_FALSE(eb::hint(failure).empty());
    EXPECT_EQ(eb::qnn_ep_hint(eb::to_string(failure)), eb::hint(failure));
}

}  // namespace

TEST(NpuSelection, NoDevicesAtAll) {
    const std::vector<EpDeviceInfo> devices;
    const eb::NpuSelection s = eb::select_qnn_npu(devices);
    expect_failure(s, NpuSelectionFailure::no_qnn_ep_devices, devices);
    EXPECT_EQ(s.candidate_count, 0);
}

TEST(NpuSelection, CpuOnlyMeansNoQnnDevices) {
    const std::vector<EpDeviceInfo> devices = {cpu()};
    const eb::NpuSelection s = eb::select_qnn_npu(devices);
    expect_failure(s, NpuSelectionFailure::no_qnn_ep_devices, devices);
    EXPECT_EQ(s.reason, "QNNExecutionProvider reported no devices");
}

TEST(NpuSelection, GpuOnlyMeansNoNpu) {
    const std::vector<EpDeviceInfo> devices = {cpu(), qnn_gpu()};
    const eb::NpuSelection s = eb::select_qnn_npu(devices);
    expect_failure(s, NpuSelectionFailure::no_npu_device, devices);
    EXPECT_EQ(s.reason, "QNNExecutionProvider reported 1 device(s), none of type NPU");
    EXPECT_EQ(s.candidate_count, 0);
}

TEST(NpuSelection, OneNpuNextToTheGpuIsSelected) {
    // Order as ONNX Runtime reports it here: CPU, QNN NPU, QNN GPU; and reversed.
    for (const std::vector<EpDeviceInfo>& devices :
         {std::vector<EpDeviceInfo>{cpu(), qnn_npu(), qnn_gpu()},
          std::vector<EpDeviceInfo>{qnn_gpu(), cpu(), qnn_npu()}}) {
        const eb::NpuSelection s = eb::select_qnn_npu(devices);
        ASSERT_TRUE(s.ok()) << s.message;
        EXPECT_FALSE(s.failure.has_value());
        EXPECT_EQ(s.candidate_count, 1);
        EXPECT_EQ(devices[*s.index], qnn_npu());
        EXPECT_TRUE(s.message.empty());
        EXPECT_THROW((void)s.error(), std::logic_error);
    }
}

TEST(NpuSelection, TwoNpusAreRefused) {
    const std::vector<EpDeviceInfo> devices = {cpu(), qnn_npu(0x1), qnn_gpu(), qnn_npu(0x2)};
    const eb::NpuSelection s = eb::select_qnn_npu(devices);
    expect_failure(s, NpuSelectionFailure::multiple_npu_devices, devices);
    EXPECT_EQ(s.candidate_count, 2);
    EXPECT_EQ(s.reason,
              "QNNExecutionProvider reported 2 devices of type NPU, expected exactly one");
}

TEST(NpuSelection, NpuOfAnotherEpIsNotSelected) {
    const EpDeviceInfo other = device("OpenVINOExecutionProvider", "NPU", 0x1234, "Intel");
    {
        const std::vector<EpDeviceInfo> devices = {cpu(), other};
        expect_failure(eb::select_qnn_npu(devices), NpuSelectionFailure::no_qnn_ep_devices,
                       devices);
    }
    {
        const std::vector<EpDeviceInfo> devices = {cpu(), other, qnn_gpu()};
        expect_failure(eb::select_qnn_npu(devices), NpuSelectionFailure::no_npu_device, devices);
    }
    {
        // Another EP's NPU does not count as a second candidate.
        const std::vector<EpDeviceInfo> devices = {other, qnn_npu(), cpu()};
        const eb::NpuSelection s = eb::select_qnn_npu(devices);
        ASSERT_TRUE(s.ok()) << s.message;
        EXPECT_EQ(*s.index, 1u);
        EXPECT_EQ(s.candidate_count, 1);
    }
}

TEST(NpuSelection, EpNameMustMatchExactly) {
    const std::vector<EpDeviceInfo> devices = {device("qnnexecutionprovider", "NPU", 1),
                                               device("QNNExecutionProvider2", "NPU", 2)};
    expect_failure(eb::select_qnn_npu(devices), NpuSelectionFailure::no_qnn_ep_devices, devices);
}

TEST(NpuSelection, DeviceDescriptions) {
    EXPECT_EQ(eb::describe_device(qnn_npu()),
              "QNNExecutionProvider NPU (ep vendor Qualcomm, vendor (unset), vendor_id "
              "0x4d4f4351, device_id 0x41304430)");
    EXPECT_EQ(eb::describe_devices({}), "(none)");
    EXPECT_EQ(eb::describe_devices({cpu(), qnn_gpu()}),
              "1. CPUExecutionProvider CPU (ep vendor Microsoft, vendor Qualcomm, vendor_id "
              "0x4d4f4351, device_id 0x7); 2. QNNExecutionProvider GPU (ep vendor Qualcomm, "
              "vendor (unset), vendor_id 0x4d4f4351, device_id 0x36334330)");
}

TEST(NpuSelection, FailureKindsAreStable) {
    EXPECT_EQ(eb::to_string(NpuSelectionFailure::no_qnn_ep_devices), "no_qnn_ep_devices");
    EXPECT_EQ(eb::to_string(NpuSelectionFailure::no_npu_device), "no_npu_device");
    EXPECT_EQ(eb::to_string(NpuSelectionFailure::multiple_npu_devices), "multiple_npu_devices");
}

TEST(QnnEp, RegistrationErrorsHaveHints) {
    for (const char* kind :
         {"qnn_ep_not_bundled", "ep_registration_failed", "module_outside_bundle"})
        EXPECT_FALSE(eb::qnn_ep_hint(kind).empty()) << kind;
    EXPECT_TRUE(eb::qnn_ep_hint("inference_failed").empty());
    EXPECT_EQ(eb::qnn_ep_library_path(LR"(C:\bundle)"),
              std::filesystem::path(LR"(C:\bundle\onnxruntime_providers_qnn.dll)"));
}
