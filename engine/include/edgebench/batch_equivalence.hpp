// Batch equivalence: whether a model file at batch size N computes, for every
// batch item, exactly what the batch-1 file of the same model variant computes
// (results schema: execution.batch_equivalence).
//
// The answer comes from the committed reference data, not from a check at run
// time: scripts/make_references.py compares the stored reference outputs of
// each file with its batch-1 sibling, per provider, on the 8 reference images or
// windows, and writes models/batch_equivalence.json. The engine looks the model
// up there by its SHA256, the provider ("cpu" or "npu") and the compute
// precision. A file that is not listed gets basis "not_available".
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace edgebench {

inline constexpr std::string_view kBatchEquivalenceFileName = "batch_equivalence.json";

struct BatchEquivalence {
    std::optional<bool> equivalent_to_b1;  // null: not available
    std::string basis = "not_available";   // batch_size_1 | reference_outputs | not_available
    std::string reason;                    // why it is not available (for a warning)
};

/// The summary file for `model`: kBatchEquivalenceFileName in the directory of the
/// model file or the nearest parent directory that has one (models/ for the files
/// of the model pipeline). nullopt if there is none.
std::optional<std::filesystem::path> find_batch_equivalence_file(
    const std::filesystem::path& model);

/// Looks up one model file in a parsed summary. Pure. Batch size 1 is equivalent by
/// definition (basis batch_size_1) and needs no entry. An entry whose batch size
/// differs from `batch_size` is ignored (not_available).
BatchEquivalence batch_equivalence_from(const nlohmann::json& summary, std::string_view sha256,
                                        std::string_view provider,
                                        std::string_view compute_precision,
                                        std::int64_t batch_size);

/// find_batch_equivalence_file() + batch_equivalence_from(). A missing or
/// unreadable summary gives basis not_available with the reason.
BatchEquivalence lookup_batch_equivalence(const std::filesystem::path& model,
                                          std::string_view sha256, std::string_view provider,
                                          std::string_view compute_precision,
                                          std::int64_t batch_size);

}  // namespace edgebench
