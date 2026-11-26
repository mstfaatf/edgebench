#include "edgebench/qnn_log.hpp"

#include <onnxruntime_c_api.h>

#include <regex>
#include <sstream>

namespace edgebench {

namespace {

// Every line of every message, in order (continuation lines included).
std::vector<std::string> lines_of(const std::vector<std::string>& messages) {
    std::vector<std::string> out;
    for (const std::string& message : messages) {
        std::istringstream in(message);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            out.push_back(line);
        }
    }
    return out;
}

std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t");
    if (first == std::string::npos)
        return {};
    const auto last = s.find_last_not_of(" \t");
    return s.substr(first, last - first + 1);
}

}  // namespace

int QnnValidation::failed_nodes() const {
    int n = 0;
    for (const QnnValidationPass& p : passes)
        n += p.failed_nodes;
    return n;
}

std::optional<QnnValidation> parse_qnn_validation(const std::vector<std::string>& messages) {
    static const std::regex kPass(R"(GetCapability pass (\d+))");
    static const std::regex kValidation(
        R"(Validation (PASSED|FAILED) for (\d+) nodes in OrtNodeUnit \((\w+)\))");
    static const std::regex kFailedNode(
        R"(Operator type: (\S+) Node name: (.*?) Node index: (\d+))");
    static const std::regex kReason(R"(REASON\s*:\s*(.*))");

    QnnValidation v;
    bool any = false;
    std::optional<QnnRejectedNode> failed;  // rejected node whose details are being read
    auto finish_failed = [&v, &failed]() {
        if (failed && !failed->op_type.empty()) {
            bool known = false;
            for (const QnnRejectedNode& r : v.rejected_nodes) {
                if (r.op_type == failed->op_type && r.node_name == failed->node_name &&
                    r.node_index == failed->node_index)
                    known = true;  // the node is rejected again in a later pass
            }
            if (!known)
                v.rejected_nodes.push_back(*failed);
        }
        failed.reset();
    };

    for (const std::string& line : lines_of(messages)) {
        std::smatch m;
        if (std::regex_search(line, m, kPass)) {
            finish_failed();
            v.passes.push_back({});
            any = true;
            continue;
        }
        if (std::regex_search(line, m, kValidation)) {
            finish_failed();
            if (v.passes.empty())
                v.passes.push_back({});  // a log without pass markers
            any = true;
            const int n = std::stoi(m[2].str());
            if (m[1].str() == "PASSED") {
                v.passes.back().passed_nodes += n;
                ++v.passes.back().passed_units_by_op[m[3].str()];
            } else {
                v.passes.back().failed_nodes += n;
                ++v.passes.back().failed_units_by_op[m[3].str()];
                failed = QnnRejectedNode{m[3].str(), "", std::nullopt, std::nullopt};
            }
            continue;
        }
        if (!failed)
            continue;
        if (std::regex_search(line, m, kFailedNode)) {
            failed->op_type = m[1].str();
            failed->node_name = trim(m[2].str());
            failed->node_index = std::stoi(m[3].str());
        } else if (std::regex_search(line, m, kReason)) {
            failed->reason = trim(m[1].str());
            finish_failed();
        }
    }
    finish_failed();
    if (!any)
        return std::nullopt;
    return v;
}

std::vector<QnnRejectedNode> parse_qnn_unsupported_nodes(const std::vector<std::string>& messages) {
    static const std::regex kUnsupported(R"(Unsupported nodes in QNN EP: (.*))");
    static const std::regex kItem(R"(\{ name: ([^,]*), type: ([^ }]*) \})");
    std::vector<QnnRejectedNode> out;
    for (const std::string& line : lines_of(messages)) {
        std::smatch m;
        if (!std::regex_search(line, m, kUnsupported))
            continue;
        const std::string list = m[1].str();
        for (auto it = std::sregex_iterator(list.begin(), list.end(), kItem);
             it != std::sregex_iterator(); ++it) {
            QnnRejectedNode node{(*it)[2].str(), trim((*it)[1].str()), std::nullopt, std::nullopt};
            bool known = false;
            for (const QnnRejectedNode& n : out)
                known = known || n == node;  // logged once per capability pass
            if (!known)
                out.push_back(std::move(node));
        }
    }
    return out;
}

bool mentions_qnn_backend_failure(const std::vector<std::string>& messages) {
    for (const std::string& message : messages) {
        if (message.find("SetupBackend failed") != std::string::npos ||
            message.find("Unable to load backend") != std::string::npos)
            return true;
    }
    return false;
}

std::string describe_nodes(const std::vector<QnnRejectedNode>& nodes) {
    std::string out;
    for (const QnnRejectedNode& n : nodes) {
        if (!out.empty())
            out += ", ";
        out += n.op_type + " node '" + n.node_name + "'";
    }
    return out;
}

Error classify_qnn_session_failure(Error error, bool strict,
                                   const std::vector<std::string>& logged) {
    if (mentions_qnn_backend_failure(logged)) {
        error.exit_code = ExitCode::ort_load_failure;
        error.kind = "qnn_backend_load_failed";
        for (const std::string& message : logged) {
            if (message.find("SetupBackend failed") != std::string::npos ||
                message.find("Unable to load backend") != std::string::npos) {
                error.message += " (the QNN execution provider reported: " + message + ")";
                break;
            }
        }
        return error;
    }
    if (strict && error.ort_error_code == static_cast<int>(ORT_FAIL) &&
        error.message.find(kCpuFallbackDisabledText) != std::string::npos) {
        error.exit_code = ExitCode::strict_mode_refusal;
        error.kind = "strict_mode_refusal";
        const std::vector<QnnRejectedNode> nodes = parse_qnn_unsupported_nodes(logged);
        error.unsupported_nodes.emplace();
        for (const QnnRejectedNode& n : nodes)
            error.unsupported_nodes->push_back({n.node_name, n.op_type});
        if (!nodes.empty())
            error.message += " Nodes the QNN EP does not support: " + describe_nodes(nodes) + ".";
    }
    return error;
}

}  // namespace edgebench
