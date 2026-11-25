// The one Runtime shared by every test in a test program.
//
// ONNX Runtime keeps a single environment per process: a second Ort::Env reuses
// the first one, including its logging callback, so two Runtime objects would
// make the second one's log capture (node placement) receive nothing. All tests
// therefore use this accessor.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "edgebench/ep_devices.hpp"
#include "edgebench/npu_selection.hpp"
#include "edgebench/result.hpp"
#include "edgebench/runtime.hpp"

namespace edgebench::test {

/// Loads the bundled ONNX Runtime and creates the Runtime on first use; throws
/// std::runtime_error if either fails. Released by a global test environment
/// when the test program finishes (before main() returns).
Runtime& shared_runtime();

/// What the QNN EP offers on this machine: registered on the shared runtime,
/// devices enumerated and the NPU selected, then unregistered again.
struct QnnNpuProbe {
    std::optional<Error> registration_error;  // register_qnn_ep() failed
    std::vector<EpDeviceInfo> devices;
    std::optional<NpuSelection> selection;  // set if the registration succeeded
};
QnnNpuProbe probe_qnn_npu();

/// Why tests that need a QNN NPU cannot run on this machine, or nullopt if they
/// must run (and fail if the NPU is not usable). They are skipped only in builds
/// without the QNN EP (x64) and on machines without any Qualcomm hardware (e.g.
/// the hosted ARM64 CI runners): a Snapdragon machine whose NPU cannot be
/// selected, or whose QNN EP cannot be registered, fails them instead.
std::optional<std::string> npu_skip_reason(const QnnNpuProbe& probe);

}  // namespace edgebench::test
