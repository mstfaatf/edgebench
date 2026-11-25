#include "edgebench/placement.hpp"

#include <onnxruntime_cxx_api.h>

#include <regex>

#include "edgebench/ort_loader.hpp"
#include "edgebench/runtime.hpp"

namespace edgebench {

bool NodeAssignment::all_on(std::string_view ep) const {
    return total_nodes > 0 && nodes_not_on(ep) == 0;
}

int NodeAssignment::nodes_not_on(std::string_view ep) const {
    int n = 0;
    for (const auto& [name, count] : nodes_by_ep) {
        if (name != ep)
            n += count;
    }
    return n;
}

std::optional<NodeAssignment> parse_node_placements(const std::vector<std::string>& messages) {
    static const std::regex kPlaced(
        R"((?:All nodes|Node\(s\)) placed on \[([\w.\-]+)\]\. Number of nodes: (\d+))");
    bool in_block = false;
    std::optional<NodeAssignment> result;
    for (const std::string& message : messages) {
        if (message.find("Node placements") != std::string::npos) {
            in_block = true;
            result = NodeAssignment{};  // a later block (e.g. a re-created session) wins
            continue;
        }
        if (!in_block)
            continue;
        std::smatch m;
        if (std::regex_search(message, m, kPlaced)) {
            const int n = std::stoi(m[2].str());
            result->nodes_by_ep[m[1].str()] += n;
            result->total_nodes += n;
        }
    }
    if (result && result->nodes_by_ep.empty())
        return std::nullopt;
    return result;
}

Result<PlacementCapture> capture_node_placement(Runtime& runtime,
                                                const std::filesystem::path& model,
                                                Ort::SessionOptions& options,
                                                const std::function<Result<void>()>& while_alive) {
    std::vector<std::string> messages;
    std::optional<Error> alive_error;
    try {
        options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
        Runtime::LogCapture capture(runtime);
        {
            Ort::Session session(runtime.env(), model.c_str(), options);
            if (while_alive) {
                if (Result<void> r = while_alive(); !r)
                    alive_error = r.error();
            }
        }
        messages = capture.messages();
    } catch (const Ort::Exception& e) {
        return ort_error("session_creation_failed",
                         "creating the node placement session for " + path_to_utf8(model),
                         e.GetOrtErrorCode(), e.what());
    }
    if (alive_error)
        return *alive_error;
    std::optional<NodeAssignment> placement = parse_node_placements(messages);
    if (!placement) {
        return Error{ExitCode::internal_error, "placement_unavailable",
                     "ONNX Runtime did not log node placements while creating a verbose session "
                     "for " +
                         path_to_utf8(model) + " (" + std::to_string(messages.size()) +
                         " log messages captured)"};
    }
    return PlacementCapture{std::move(*placement), std::move(messages)};
}

Result<NodeAssignment> record_node_placement(Runtime& runtime, const std::filesystem::path& model,
                                             Ort::SessionOptions& options) {
    Result<PlacementCapture> captured = capture_node_placement(runtime, model, options);
    if (!captured)
        return captured.error();
    return std::move(captured).value().assignment;
}

}  // namespace edgebench
