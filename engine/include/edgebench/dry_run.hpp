// `edgebench run --dry-run`: create the real session, run one inference, and
// report the resolved configuration.
#pragma once

#include <nlohmann/json.hpp>

#include "edgebench/result.hpp"
#include "edgebench/run_config.hpp"

namespace edgebench {

class Runtime;

/// Hashes the model, creates the provider and session (with node placement),
/// runs one inference and checks that every floating-point output value is
/// finite (internal_error naming the output otherwise). Returns the resolved
/// configuration with results-schema field names (see docs/cli.md).
Result<nlohmann::ordered_json> dry_run(Runtime& runtime, const RunConfig& config);

}  // namespace edgebench
