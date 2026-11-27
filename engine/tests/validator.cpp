#include "validator.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "edgebench/ort_loader.hpp"

namespace fs = std::filesystem;

namespace edgebench::test {

namespace {

const fs::path kPython = fs::path(EDGEBENCH_PYTHON).make_preferred();
const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();

std::string find_reason() {
    if (!fs::is_regular_file(kPython))
        return "the Python interpreter " + path_to_utf8(kPython) +
               " (EDGEBENCH_PYTHON) was not found: result validation skipped";
    const ProcessResult r = run_process(kPython, {L"-c", L"import jsonschema"}, kSourceDir);
    if (r.exit_code != 0)
        return "jsonschema cannot be imported by " + path_to_utf8(kPython) +
               ": result validation skipped";
    return {};
}

}  // namespace

const std::string& validator_unavailable_reason() {
    static const std::string reason = find_reason();
    return reason;
}

ProcessResult run_validator(const fs::path& result) {
    return run_process(
        kPython,
        {(kSourceDir / "scripts" / "validate_results.py").native(), L"--strict", result.native()},
        kSourceDir);
}

void validate_or_skip(const fs::path& result) {
    if (const std::string& reason = validator_unavailable_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    const ProcessResult v = run_validator(result);
    EXPECT_EQ(v.exit_code, 0u) << v.output;
}

void validate_doctor_report_or_skip(const fs::path& report, bool runner) {
    if (const std::string& reason = validator_unavailable_reason(); !reason.empty())
        GTEST_SKIP() << reason;
    std::vector<std::wstring> args = {(kSourceDir / "scripts" / "validate_doctor.py").native()};
    if (runner)
        args.push_back(L"--runner");
    args.push_back(report.native());
    const ProcessResult v = run_process(kPython, args, kSourceDir);
    EXPECT_EQ(v.exit_code, 0u) << v.output;
}

}  // namespace edgebench::test
