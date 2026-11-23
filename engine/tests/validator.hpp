// Running scripts/validate_results.py from the C++ tests.
//
// The validator needs the Python interpreter configured as EDGEBENCH_PYTHON with
// jsonschema installed. When it is not available, tests that would validate a
// result file report themselves as skipped (with the reason) instead of passing.
#pragma once

#include <filesystem>
#include <string>

#include "process_util.hpp"

namespace edgebench::test {

/// Why the validator cannot run here (interpreter missing, or jsonschema not
/// importable); empty if it can. Checked once per test program.
const std::string& validator_unavailable_reason();

/// validate_results.py --strict on `result`, run from the source directory.
ProcessResult run_validator(const std::filesystem::path& result);

/// Validates `result` and expects it to pass; if the validator cannot run,
/// records the current test as skipped with the reason. Assertions made before
/// or after still count: a failure elsewhere still fails the test.
void validate_or_skip(const std::filesystem::path& result);

}  // namespace edgebench::test
