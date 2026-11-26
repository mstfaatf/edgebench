#include "edgebench/batch_equivalence.hpp"

#include <fstream>

#include "edgebench/ort_loader.hpp"

namespace fs = std::filesystem;

namespace edgebench {

std::optional<fs::path> find_batch_equivalence_file(const fs::path& model) {
    std::error_code ec;
    fs::path dir = fs::absolute(model, ec).parent_path();
    if (ec)
        return std::nullopt;
    while (true) {
        const fs::path candidate = dir / kBatchEquivalenceFileName;
        if (fs::is_regular_file(candidate, ec))
            return candidate;
        const fs::path parent = dir.parent_path();
        if (parent == dir)
            return std::nullopt;
        dir = parent;
    }
}

BatchEquivalence batch_equivalence_from(const nlohmann::json& summary, std::string_view sha256,
                                        std::string_view provider,
                                        std::string_view compute_precision,
                                        std::int64_t batch_size) {
    if (batch_size == 1)
        return {true, "batch_size_1", {}};
    BatchEquivalence out;
    const auto entries = summary.find("entries");
    if (entries == summary.end() || !entries->is_array()) {
        out.reason = "the batch-equivalence summary has no entries list";
        return out;
    }
    for (const nlohmann::json& e : *entries) {
        if (!e.is_object() || e.value("sha256", "") != sha256 ||
            e.value("provider", "") != provider ||
            e.value("compute_precision", "") != compute_precision)
            continue;
        const auto equivalent = e.find("equivalent_to_b1");
        if (e.value("batch_size", std::int64_t{0}) != batch_size || equivalent == e.end() ||
            !equivalent->is_boolean()) {
            out.reason = "the batch-equivalence entry for this model file is inconsistent";
            return out;
        }
        return {equivalent->get<bool>(), "reference_outputs", {}};
    }
    out.reason = "this model file (SHA256 " + std::string(sha256.substr(0, 12)) + ", " +
                 std::string(provider) + ", " + std::string(compute_precision) +
                 ") is not in the batch-equivalence summary";
    return out;
}

BatchEquivalence lookup_batch_equivalence(const fs::path& model, std::string_view sha256,
                                          std::string_view provider,
                                          std::string_view compute_precision,
                                          std::int64_t batch_size) {
    if (batch_size == 1)
        return batch_equivalence_from(nlohmann::json::object(), sha256, provider, compute_precision,
                                      batch_size);
    const std::optional<fs::path> file = find_batch_equivalence_file(model);
    if (!file) {
        BatchEquivalence out;
        out.reason = "no " + std::string(kBatchEquivalenceFileName) +
                     " in the model's directory or any parent directory";
        return out;
    }
    std::ifstream in(*file, std::ios::binary);
    const nlohmann::json summary = nlohmann::json::parse(in, nullptr, /*allow_exceptions=*/false);
    if (summary.is_discarded() || !summary.is_object()) {
        BatchEquivalence out;
        out.reason = path_to_utf8(*file) + " is not a valid JSON object";
        return out;
    }
    return batch_equivalence_from(summary, sha256, provider, compute_precision, batch_size);
}

}  // namespace edgebench
