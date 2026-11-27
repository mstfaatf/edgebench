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

namespace {

std::optional<bool> optional_bool(const nlohmann::json& e, const char* key, bool& ok) {
    const auto it = e.find(key);
    if (it == e.end() || !(it->is_boolean() || it->is_null())) {
        ok = false;
        return std::nullopt;
    }
    return it->is_null() ? std::nullopt : std::optional<bool>(it->get<bool>());
}

std::optional<std::int64_t> optional_count(const nlohmann::json& e, const char* key, bool& ok) {
    const auto it = e.find(key);
    if (it == e.end() || it->is_null()) {
        ok = ok && it != e.end();
        return std::nullopt;
    }
    if (!it->is_number_integer() || it->get<std::int64_t>() < 0) {
        ok = false;
        return std::nullopt;
    }
    return it->get<std::int64_t>();
}

// The rules of the results schema for the fields of one entry.
bool consistent(const BatchEquivalence& b) {
    if (!b.bitwise_equal_to_b1)
        return false;  // every entry of the summary has a bitwise answer
    const bool counted =
        b.predictions_basis == "eval_quick_top1" || b.predictions_basis == "test_split_flags";
    if (*b.bitwise_equal_to_b1)
        return b.predictions_basis == "bitwise" && b.predictions_equal_to_b1 == true &&
               !b.predictions_compared && !b.predictions_differing;
    if (b.predictions_basis == "not_available")
        return !b.predictions_equal_to_b1 && !b.predictions_compared && !b.predictions_differing;
    return counted && b.predictions_equal_to_b1 && b.predictions_compared &&
           b.predictions_differing && *b.predictions_differing <= *b.predictions_compared &&
           *b.predictions_equal_to_b1 == (*b.predictions_differing == 0);
}

// The entry's ep_compile_options as a map; nullopt if missing or not an object of
// strings.
std::optional<EpCompileOptions> entry_options(const nlohmann::json& e) {
    const auto it = e.find("ep_compile_options");
    if (it == e.end() || !it->is_object())
        return std::nullopt;
    EpCompileOptions out;
    for (const auto& [key, value] : it->items()) {
        if (!value.is_string())
            return std::nullopt;
        out.emplace(key, value.get<std::string>());
    }
    return out;
}

}  // namespace

std::string describe_compile_options(const EpCompileOptions& options) {
    if (options.empty())
        return "the EP defaults";
    std::string out;
    for (const auto& [key, value] : options) {
        if (!out.empty())
            out += ", ";
        out += key + "=" + value;
    }
    return out;
}

BatchEquivalence batch_equivalence_from(const nlohmann::json& summary, std::string_view sha256,
                                        std::string_view provider,
                                        std::string_view compute_precision,
                                        const EpCompileOptions& compile_options,
                                        std::int64_t batch_size) {
    if (batch_size == 1) {
        BatchEquivalence b1;
        b1.bitwise_equal_to_b1 = true;
        b1.bitwise_basis = "batch_size_1";
        b1.predictions_equal_to_b1 = true;
        b1.predictions_basis = "bitwise";
        return b1;
    }
    BatchEquivalence out;
    if (summary.value("batch_equivalence_version", "") != kBatchEquivalenceVersion) {
        out.reason =
            "the batch-equivalence summary is not version " + std::string(kBatchEquivalenceVersion);
        return out;
    }
    const auto entries = summary.find("entries");
    if (entries == summary.end() || !entries->is_array()) {
        out.reason = "the batch-equivalence summary has no entries list";
        return out;
    }
    bool listed_with_other_options = false;
    for (const nlohmann::json& e : *entries) {
        if (!e.is_object() || e.value("sha256", "") != sha256 ||
            e.value("provider", "") != provider ||
            e.value("compute_precision", "") != compute_precision)
            continue;
        const std::optional<EpCompileOptions> options = entry_options(e);
        if (!options) {
            out.reason = "the batch-equivalence entry for this model file is inconsistent";
            return out;
        }
        if (*options != compile_options) {
            listed_with_other_options = true;
            continue;
        }
        bool ok = e.value("batch_size", std::int64_t{0}) == batch_size;
        BatchEquivalence found;
        found.bitwise_equal_to_b1 = optional_bool(e, "bitwise_equal_to_b1", ok);
        found.bitwise_basis = "reference_outputs";
        found.predictions_equal_to_b1 = optional_bool(e, "predictions_equal_to_b1", ok);
        found.predictions_basis = e.value("predictions_basis", "");
        found.predictions_compared = optional_count(e, "predictions_compared", ok);
        found.predictions_differing = optional_count(e, "predictions_differing", ok);
        if (!ok || !consistent(found)) {
            out.reason = "the batch-equivalence entry for this model file is inconsistent";
            return out;
        }
        if (found.predictions_basis == "not_available")
            found.reason =
                "the batch-equivalence summary has no prediction comparison for this "
                "model file";
        return found;
    }
    const std::string file = "this model file (SHA256 " + std::string(sha256.substr(0, 12)) +
                             ", " + std::string(provider) + ", " +
                             std::string(compute_precision) + ")";
    if (listed_with_other_options) {
        out.options_not_covered = true;
        out.reason = "the batch-equivalence summary lists " + file +
                     " only for other compile-relevant EP options than this run's (" +
                     describe_compile_options(compile_options) +
                     "); entries measured with other options are not reused";
        return out;
    }
    out.reason = file + " is not in the batch-equivalence summary";
    if (!compile_options.empty())
        out.reason += " (compile-relevant EP options " + describe_compile_options(compile_options) +
                      ")";
    return out;
}

BatchEquivalence lookup_batch_equivalence(const fs::path& model, std::string_view sha256,
                                          std::string_view provider,
                                          std::string_view compute_precision,
                                          const EpCompileOptions& compile_options,
                                          std::int64_t batch_size) {
    if (batch_size == 1)
        return batch_equivalence_from(nlohmann::json::object(), sha256, provider, compute_precision,
                                      compile_options, batch_size);
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
    return batch_equivalence_from(summary, sha256, provider, compute_precision, compile_options,
                                  batch_size);
}

}  // namespace edgebench
