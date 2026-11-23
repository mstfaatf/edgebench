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

/// UTC stamp used in result file names: YYYYmmddTHHMMSS.mmmZ, e.g.
/// "20261121T235210.123Z". Milliseconds are truncated, never rounded, so .999
/// stays in the same second. scripts/common/results_io.py run_stamp() produces
/// the same string.
std::string file_stamp(std::chrono::system_clock::time_point t);

}  // namespace edgebench
