// What the QNN execution provider and ONNX Runtime log about a QNN session:
// the QNN EP's per-node validation (results schema: execution.qnn_validation),
// its strict-mode summary of unsupported nodes, backend load failures, and the
// classification of a failed session creation.
//
// Pure text processing on bare ONNX Runtime log messages (as delivered to the
// Runtime's logging callback, without the timestamp/severity prefix). A message
// can span several lines: the QNN EP writes the rejected node and the reason on
// continuation lines of its "Validation FAILED" message, e.g.
//
//   Validation FAILED for 1 nodes in OrtNodeUnit (Det) :
//   	Operator type: Det Node name: unsupported_det Node index: 1
//   	REASON : Operators of type `Det` are not supported by QNN EP. ...
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/result.hpp"

namespace edgebench {

struct QnnRejectedNode {
    std::string op_type;
    std::string node_name;
    std::optional<int> node_index;
    std::optional<std::string> reason;  // null if the log gave none

    bool operator==(const QnnRejectedNode&) const = default;
};

/// One GetCapability pass of the QNN EP (it validates the graph twice: before and
/// after ONNX Runtime's layout transformation).
struct QnnValidationPass {
    int passed_nodes = 0;
    int failed_nodes = 0;

    bool operator==(const QnnValidationPass&) const = default;
};

struct QnnValidation {
    std::vector<QnnValidationPass> passes;        // in order
    std::vector<QnnRejectedNode> rejected_nodes;  // distinct nodes, in order of first rejection

    /// Failed nodes over all passes.
    int failed_nodes() const;
};

/// The QNN EP's node validation from the messages of one session creation, or
/// nullopt if the messages contain none (e.g. a session without verbose logging).
std::optional<QnnValidation> parse_qnn_validation(const std::vector<std::string>& messages);

/// The nodes named by the QNN EP's error "Unsupported nodes in QNN EP: { name: X, type: Y }"
/// (logged in strict mode; at error severity, so also without verbose logging).
/// Distinct, in order; node_index and reason are not part of that message.
std::vector<QnnRejectedNode> parse_qnn_unsupported_nodes(const std::vector<std::string>& messages);

/// True if a message reports that the QNN EP could not load or set up its backend
/// ("QNN SetupBackend failed Unable to load backend ..."). Logged at error
/// severity, so it is seen without verbose logging.
bool mentions_qnn_backend_failure(const std::vector<std::string>& messages);

/// The sentence ONNX Runtime (inference_session.cc) puts in the error when a
/// session would need the CPU EP but session.disable_cpu_ep_fallback is set.
inline constexpr std::string_view kCpuFallbackDisabledText =
    "fallback to CPU EP has been explicitly disabled";

/// Classifies the error of a failed QNN session creation, given the warning and
/// error messages ONNX Runtime logged while the session was being created:
///
/// - the QNN EP logged a backend failure: qnn_backend_load_failed, exit code 4.
///   Checked first, because a backend that does not load makes the QNN EP claim
///   no node, so ONNX Runtime then reports a strict-mode refusal for the whole graph;
/// - strict mode, ORT_FAIL, and the error carries kCpuFallbackDisabledText:
///   strict_mode_refusal, exit code 5 (the unsupported nodes are appended if logged);
/// - anything else: unchanged (ort_runtime_error, exit code 3).
///
/// The ORT error code and name are kept in every case.
Error classify_qnn_session_failure(Error error, bool strict,
                                   const std::vector<std::string>& logged);

/// "Det node 'unsupported_det', ..." for an error message or warning.
std::string describe_nodes(const std::vector<QnnRejectedNode>& nodes);

}  // namespace edgebench
