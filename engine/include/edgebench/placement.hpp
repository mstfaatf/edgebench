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
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/result.hpp"

namespace edgebench {

class Runtime;

struct NodeAssignment {
    int total_nodes = 0;
    std::map<std::string, int> nodes_by_ep;
    /// Results schema execution.verification_method: counts come from ORT's node
    /// placement log, so "node_assignment".
    std::string verification_method = "node_assignment";

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

}  // namespace edgebench
