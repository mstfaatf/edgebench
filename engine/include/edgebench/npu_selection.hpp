// Selection of the NPU device for QNN runs from the enumerated EP devices.
//
// Pure: works on the plain device list (list_ep_devices()), makes no ONNX
// Runtime call, and does not assume a particular machine. The rule is the EP
// name QNNExecutionProvider plus hardware device type NPU, and exactly one
// device must match; anything else is an error that lists every enumerated
// device, so the report shows what the machine offered instead.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/ep_devices.hpp"
#include "edgebench/result.hpp"

namespace edgebench {

inline constexpr std::string_view kQnnEpName = "QNNExecutionProvider";

enum class NpuSelectionFailure {
    /// The QNN EP reported no device at all (not registered, or no supported hardware).
    no_qnn_ep_devices,
    /// The QNN EP reported devices, but none of hardware type NPU (e.g. only the GPU).
    no_npu_device,
    /// The QNN EP reported more than one NPU device; picking one would be a guess.
    multiple_npu_devices,
};

/// Stable identifier, used as the error kind: "no_qnn_ep_devices", ...
std::string_view to_string(NpuSelectionFailure failure);

/// One-line advice for the failure.
std::string_view hint(NpuSelectionFailure failure);

struct NpuSelection {
    std::optional<std::size_t> index;            // position in the device list, if selected
    std::optional<NpuSelectionFailure> failure;  // set if nothing was selected
    std::string reason;                          // why nothing was selected, one clause
    std::string message;      // reason plus every enumerated device (describe_devices())
    int candidate_count = 0;  // QNN EP devices of type NPU

    bool ok() const { return index.has_value(); }
    /// The failure as an Error: exit code 3 (ort_runtime_error), kind to_string(failure).
    /// Must not be called on a successful selection.
    Error error() const;
};

/// The single QNN EP device of hardware type NPU in `devices`.
NpuSelection select_qnn_npu(const std::vector<EpDeviceInfo>& devices);

/// One line per device, e.g.
/// "QNNExecutionProvider NPU (ep vendor Qualcomm, vendor (unset), vendor_id 0x4d4f4351,
/// device_id 0x41304430)".
std::string describe_device(const EpDeviceInfo& device);

/// Every device, "1. <device>; 2. <device>", or "(none)".
std::string describe_devices(const std::vector<EpDeviceInfo>& devices);

}  // namespace edgebench
