#include "edgebench/npu_selection.hpp"

#include <cstdio>
#include <stdexcept>

namespace edgebench {

namespace {

std::string hex(std::uint32_t v) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%x", v);
    return buf;
}

std::string or_unset(const std::string& s) {
    return s.empty() ? "(unset)" : s;
}

}  // namespace

std::string_view to_string(NpuSelectionFailure failure) {
    switch (failure) {
        case NpuSelectionFailure::no_qnn_ep_devices:
            return "no_qnn_ep_devices";
        case NpuSelectionFailure::no_npu_device:
            return "no_npu_device";
        case NpuSelectionFailure::multiple_npu_devices:
            return "multiple_npu_devices";
    }
    return "unknown";
}

std::string_view hint(NpuSelectionFailure failure) {
    switch (failure) {
        case NpuSelectionFailure::no_qnn_ep_devices:
            return "The QNN execution provider finds no supported Qualcomm hardware on this "
                   "machine; NPU runs need a Snapdragon device with its NPU driver installed.";
        case NpuSelectionFailure::no_npu_device:
            return "Check in Device Manager that the Qualcomm Hexagon NPU driver is installed and "
                   "the device is enabled.";
        case NpuSelectionFailure::multiple_npu_devices:
            return "EdgeBench does not guess between NPUs; please report the device list above.";
    }
    return {};
}

Error NpuSelection::error() const {
    if (!failure)
        throw std::logic_error("NpuSelection::error() on a successful selection");
    return Error{ExitCode::ort_runtime_error, std::string(to_string(*failure)), message};
}

std::string describe_device(const EpDeviceInfo& d) {
    return d.ep_name + " " + d.device_type + " (ep vendor " + or_unset(d.ep_vendor) + ", vendor " +
           or_unset(d.vendor) + ", vendor_id " + hex(d.vendor_id) + ", device_id " +
           hex(d.device_id) + ")";
}

std::string describe_devices(const std::vector<EpDeviceInfo>& devices) {
    if (devices.empty())
        return "(none)";
    std::string s;
    for (std::size_t i = 0; i < devices.size(); ++i)
        s += (i ? "; " : "") + std::to_string(i + 1) + ". " + describe_device(devices[i]);
    return s;
}

NpuSelection select_qnn_npu(const std::vector<EpDeviceInfo>& devices) {
    NpuSelection s;
    int qnn_devices = 0;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (devices[i].ep_name != kQnnEpName)
            continue;
        ++qnn_devices;
        if (devices[i].device_type == "NPU") {
            ++s.candidate_count;
            if (!s.index)
                s.index = i;
        }
    }
    if (s.candidate_count == 1)
        return s;

    const std::string ep(kQnnEpName);
    if (qnn_devices == 0) {
        s.failure = NpuSelectionFailure::no_qnn_ep_devices;
        s.reason = ep + " reported no devices";
    } else if (s.candidate_count == 0) {
        s.failure = NpuSelectionFailure::no_npu_device;
        s.reason = ep + " reported " + std::to_string(qnn_devices) + " device(s), none of type NPU";
    } else {
        s.failure = NpuSelectionFailure::multiple_npu_devices;
        s.reason = ep + " reported " + std::to_string(s.candidate_count) +
                   " devices of type NPU, expected exactly one";
    }
    s.message = s.reason + "; enumerated devices: " + describe_devices(devices);
    s.index.reset();
    return s;
}

}  // namespace edgebench
