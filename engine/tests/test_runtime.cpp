#include "test_runtime.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "edgebench/qnn_ep.hpp"

namespace edgebench::test {

Runtime& shared_runtime() {
    static std::unique_ptr<Runtime> runtime = [] {
        const OrtLoadResult load = load_onnxruntime();
        if (!load.ok())
            throw std::runtime_error(load.message);
        Result<std::unique_ptr<Runtime>> created = Runtime::create(load);
        if (!created)
            throw std::runtime_error(created.error().message);
        return std::move(created).value();
    }();
    return *runtime;
}

QnnNpuProbe probe_qnn_npu() {
    Runtime& runtime = shared_runtime();
    QnnNpuProbe probe;
    Result<EpLibrary*> registered = register_qnn_ep(runtime);
    if (!registered) {
        probe.registration_error = registered.error();
        return probe;
    }
    probe.devices = runtime.ep_devices();
    probe.selection = select_qnn_npu(probe.devices);
    if (Result<void> r = runtime.unregister_ep_library(kQnnEpName); !r)
        throw std::runtime_error(r.error().message);
    return probe;
}

std::optional<std::string> npu_skip_reason(const QnnNpuProbe& probe) {
    if (!qnn_ep_in_build())
        return "this build does not include the QNN execution provider (x64 builds are CPU-only)";
    if (probe.registration_error || !probe.selection || probe.selection->ok())
        return std::nullopt;
    if (probe.selection->failure == NpuSelectionFailure::multiple_npu_devices)
        return std::nullopt;
    // ACPI vendor id "QCOM", as ONNX Runtime reports it for Qualcomm CPUs, GPUs and NPUs.
    constexpr std::uint32_t kQualcommVendorId = 0x4D4F4351;
    const bool qualcomm =
        std::any_of(probe.devices.begin(), probe.devices.end(),
                    [](const EpDeviceInfo& d) { return d.vendor_id == kQualcommVendorId; });
    if (qualcomm)
        return std::nullopt;
    return "no Qualcomm hardware on this machine, so no QNN NPU (" + probe.selection->message + ")";
}

}  // namespace edgebench::test
