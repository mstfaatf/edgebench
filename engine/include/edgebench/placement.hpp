// Node placement: which execution provider each node of a session runs on
// (results schema: execution.node_assignment).
//
// ONNX Runtime logs a "Node placements" block at verbose severity when it
// creates a session:
//
//   Node placements
//    All nodes placed on [CPUExecutionProvider]. Number of nodes: 65
//
// or, with more than one EP, one " Node(s) placed on [<EP>]. Number of nodes: <n>"
// line per EP followed by the node names. The counts are after graph optimization
// (fused nodes count once).
#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/qnn_log.hpp"
#include "edgebench/result.hpp"

namespace edgebench {

class Runtime;

struct NodeAssignment {
    int total_nodes = 0;
    std::map<std::string, int> nodes_by_ep;
    /// Results schema execution.verification_method: counts come from ORT's node
    /// placement log, so "node_assignment".
    std::string verification_method = "node_assignment";
    /// The QNN EP's node validation from the same session log (QNN sessions only;
    /// results schema: execution.qnn_validation).
    std::optional<QnnValidation> qnn_validation;
    /// True if the provider checked, while the placement session was alive, that every
    /// ONNX Runtime / QNN module came from the bundle (a failed check fails the call).
    bool modules_checked_while_alive = false;

    /// True if every node is on `ep` (and there is at least one node).
    bool all_on(std::string_view ep) const;
    /// Nodes on any EP other than `ep`.
    int nodes_not_on(std::string_view ep) const;
};

/// Extracts the node counts from ORT log messages (bare messages, without the
/// timestamp/severity prefix). nullopt if the messages contain no placement block.
std::optional<NodeAssignment> parse_node_placements(const std::vector<std::string>& messages);

/// Creates a separate session with verbose logging (never the session that is
/// timed: verbose logging slows session creation down), captures ORT's log while
/// it is created, and parses the node placements. `options` are the provider's
/// session options; their log severity is set to verbose here.
Result<NodeAssignment> record_node_placement(Runtime& runtime, const std::filesystem::path& model,
                                             Ort::SessionOptions& options);

/// The verbose placement session's parsed node placements and every message it logged.
struct PlacementCapture {
    NodeAssignment assignment;
    std::vector<std::string> messages;
};

/// record_node_placement() that also returns the captured messages. `while_alive`,
/// if given, runs while the placement session exists (e.g. to check the modules
/// it loaded); its error fails the call. ONNX Runtime failures creating the
/// session are session_creation_failed with the ORT error; `messages` of the
/// failed attempt are then lost, so callers that classify failures read
/// Runtime::warnings_since() instead.
Result<PlacementCapture> capture_node_placement(
    Runtime& runtime, const std::filesystem::path& model, Ort::SessionOptions& options,
    const std::function<Result<void>()>& while_alive = {});

}  // namespace edgebench
