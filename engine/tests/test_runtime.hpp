// The one Runtime shared by every test in a test program.
//
// ONNX Runtime keeps a single environment per process: a second Ort::Env reuses
// the first one, including its logging callback, so two Runtime objects would
// make the second one's log capture (node placement) receive nothing. All tests
// therefore use this accessor.
#pragma once

#include "edgebench/runtime.hpp"

namespace edgebench::test {

/// Loads the bundled ONNX Runtime and creates the Runtime on first use; throws
/// std::runtime_error if either fails.
Runtime& shared_runtime();

}  // namespace edgebench::test
