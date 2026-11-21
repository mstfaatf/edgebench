// Identity of a run (results schema: run_id, timestamp_utc).
#pragma once

#include <chrono>
#include <string>

namespace edgebench {

/// A random (version 4) UUID, lowercase and hyphenated, from the OS CSPRNG.
std::string new_run_id();

/// RFC 3339 UTC timestamp with millisecond precision and a Z suffix,
/// e.g. "2026-11-21T14:03:27.053Z".
std::string utc_timestamp(std::chrono::system_clock::time_point t);

}  // namespace edgebench
