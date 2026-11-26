// Serialization of a BenchmarkResult: the results schema 1.2.0 JSON document and
// the two CSV files that go with it (docs/results-schema.md, docs/cli.md).
#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>

#include "edgebench/benchmark.hpp"

namespace edgebench {

/// The document's raw section: where the per-iteration latencies are.
struct RawLatencyLink {
    std::string latencies_file;  // relative to the result file, '/' separators
    std::string sha256;          // of the CSV file's bytes
};

/// The results schema 1.2.0 document for `result`, in schema key order, with
/// every key present (null when unknown). `raw` is null if no latency file was
/// written. Numbers use nlohmann::json's shortest round-trip form.
nlohmann::ordered_json to_result_json(const BenchmarkResult& result,
                                      const std::optional<RawLatencyLink>& raw);

/// latencies.csv: header trial_index,iteration_index,latency_ms, then one row per
/// measured iteration in trial and iteration order.
std::string latencies_csv(const BenchmarkResult& result);

/// summary.csv: one row per trial and an aggregate row with the schema's latency
/// statistics, plus each trial's session creation time (the median for the
/// aggregate row).
std::string summary_csv(const BenchmarkResult& result);

/// A number as written to JSON and CSV: the shortest string that reads back as
/// exactly the same double, '.' as decimal separator, independent of locale.
std::string format_number(double value);

/// One CSV field (RFC 4180): quoted if it contains a comma, quote, CR or LF,
/// with quotes doubled.
std::string csv_field(std::string_view text);

}  // namespace edgebench
