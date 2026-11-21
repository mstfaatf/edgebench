// Execution-provider devices enumerated by ONNX Runtime (results schema:
// runtime.ep_devices[]). Requires a successful load_onnxruntime().
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

namespace edgebench {

struct EpDeviceInfo {
    std::string ep_name;
    std::string ep_vendor;
    std::string device_type;  // "CPU", "GPU", "NPU"
    std::string vendor;       // may be empty (the QNN EP leaves it unset)
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::map<std::string, std::string> metadata;     // hardware device metadata
    std::map<std::string, std::string> ep_metadata;  // EP metadata for the device
};

/// Devices ONNX Runtime reports in `env` for the EPs registered so far (without
/// plugin EP libraries: built-in EPs only, e.g. the CPU EP). Throws Ort::Exception
/// on ORT errors. Usually called through Runtime::ep_devices().
std::vector<EpDeviceInfo> list_ep_devices(Ort::Env& env);

nlohmann::ordered_json to_json(const EpDeviceInfo& device);

}  // namespace edgebench
