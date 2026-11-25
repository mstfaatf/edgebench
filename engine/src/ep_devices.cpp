#include "edgebench/ep_devices.hpp"

#include <onnxruntime_cxx_api.h>

namespace edgebench {

namespace {

const char* device_type(OrtHardwareDeviceType type) {
    switch (type) {
        case OrtHardwareDeviceType_CPU:
            return "CPU";
        case OrtHardwareDeviceType_GPU:
            return "GPU";
        case OrtHardwareDeviceType_NPU:
            return "NPU";
    }
    return "UNKNOWN";
}

std::map<std::string, std::string> to_map(Ort::ConstKeyValuePairs kv) {
    const auto pairs = kv.GetKeyValuePairs();
    return {pairs.begin(), pairs.end()};
}

EpDeviceInfo to_info(const Ort::ConstEpDevice& d) {
    const Ort::ConstHardwareDevice hw = d.Device();
    EpDeviceInfo info;
    info.ep_name = d.EpName() ? d.EpName() : "";
    info.ep_vendor = d.EpVendor() ? d.EpVendor() : "";
    info.device_type = device_type(hw.Type());
    info.vendor = hw.Vendor() ? hw.Vendor() : "";
    info.vendor_id = hw.VendorId();
    info.device_id = hw.DeviceId();
    info.metadata = to_map(hw.Metadata());
    info.ep_metadata = to_map(d.EpMetadata());
    return info;
}

}  // namespace

std::vector<EpDeviceInfo> list_ep_devices(Ort::Env& env) {
    std::vector<EpDeviceInfo> out;
    for (const Ort::ConstEpDevice& d : env.GetEpDevices())
        out.push_back(to_info(d));
    return out;
}

Result<Ort::ConstEpDevice> ep_device_handle(Ort::Env& env, std::size_t index,
                                            const EpDeviceInfo& expected) {
    std::vector<Ort::ConstEpDevice> devices;
    try {
        devices = env.GetEpDevices();
    } catch (const Ort::Exception& e) {
        return ort_error("ep_enumeration_failed", "enumerating EP devices", e.GetOrtErrorCode(),
                         e.what());
    }
    if (index >= devices.size() || !(to_info(devices[index]) == expected)) {
        return Error{ExitCode::ort_runtime_error, "ep_devices_changed",
                     "the EP device list changed after the device was selected (expected " +
                         expected.ep_name + " " + expected.device_type + " at position " +
                         std::to_string(index + 1) + ")"};
    }
    return devices[index];
}

nlohmann::ordered_json to_json(const EpDeviceInfo& d) {
    return {
        {"ep_name", d.ep_name},   {"ep_vendor", d.ep_vendor},     {"device_type", d.device_type},
        {"vendor", d.vendor},     {"vendor_id", d.vendor_id},     {"device_id", d.device_id},
        {"metadata", d.metadata}, {"ep_metadata", d.ep_metadata},
    };
}

}  // namespace edgebench
