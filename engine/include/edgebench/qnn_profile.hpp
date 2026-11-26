// The QNN execution provider's profiling CSV (EP options profiling_level and
// profiling_file_path), as read for the profiled accelerator pass
// (results schema: timing.profiled_pass).
//
// The EP writes the file when the profiled session is released. Header:
//
//   Msg Timestamp,Message,Time,Unit of Measurement,Timing Source,Event Level,Event Identifier
//
// With profiling_level basic, graph finalization (during session creation) adds
// rows with "(finalize)" identifiers, and every inference adds one row per
// execute event, in order, ending with Message EXECUTE ("QNN (execute) time"):
//
//   0,BACKEND,34,US,BACKEND,ROOT,Accelerator (execute) time
//   0,EXECUTE,112,US,BACKEND,ROOT,QNN (execute) time
//
// "Accelerator (execute) time" is the time the HTP spent on one inference; the
// others add the host side and the RPC to the NPU. Pure text processing.
#pragma once

#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "edgebench/result.hpp"

namespace edgebench {

inline constexpr std::string_view kQnnProfileHeader =
    "Msg Timestamp,Message,Time,Unit of Measurement,Timing Source,Event Level,Event Identifier";

/// The execute event that is the accelerator's time per inference.
inline constexpr std::string_view kAcceleratorExecuteEvent = "Accelerator (execute) time";

struct QnnProfileEvent {
    std::string message;     // BACKEND, EXECUTE, FINALIZE, NODE, ...
    double time = 0;         // in `unit`
    std::string unit;        // US, CYCLES, COUNT, ...
    std::string level;       // ROOT, SUB-EVENT, ...
    std::string identifier;  // e.g. "Accelerator (execute) time"
};

/// Parses the CSV. Error (internal_error, qnn_profile_unreadable) if the header
/// differs or a row has fewer than 7 fields or a non-numeric time. An identifier
/// containing commas keeps them.
Result<std::vector<QnnProfileEvent>> parse_qnn_profile_csv(std::string_view text);

/// The execute and initialization times of a profile, in milliseconds (only
/// events reported in microseconds, at level ROOT).
struct QnnProfileSummary {
    /// Inferences in the profile: rows with Message EXECUTE.
    int inferences = 0;
    /// Per execute event identifier, one value per inference in order, the first
    /// `skip` inferences removed. Contains kAcceleratorExecuteEvent if reported.
    std::map<std::string, std::vector<double>> execute_ms;
    /// Events outside the inferences (graph finalization), in file order.
    std::vector<std::pair<std::string, double>> init_ms;
};

/// Groups `events` into inferences (each ends with its EXECUTE row) and drops the
/// first `skip` of them (the warm-up iterations of the profiled session).
QnnProfileSummary summarize_qnn_profile(const std::vector<QnnProfileEvent>& events, int skip);

}  // namespace edgebench
