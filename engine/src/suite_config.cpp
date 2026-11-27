#include "edgebench/suite_config.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <system_error>

#include "edgebench/ep_context_cache.hpp"
#include "edgebench/model_info.hpp"
#include "edgebench/npu_options.hpp"
#include "edgebench/onnx_inspect.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/sha256.hpp"

namespace fs = std::filesystem;

namespace edgebench::suite {

namespace {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

constexpr std::string_view kModelsManifest = "manifest.json";

// Keys of a configuration that say what is measured, and the run parameters.
const std::vector<std::string> kIdentityKeys = {"model",
                                                "provider",
                                                "precision",
                                                "batch",
                                                "variant",
                                                "npu_fp16",
                                                "htp_performance_mode",
                                                "htp_graph_finalization_mode",
                                                "strict"};
const std::vector<std::string> kParamKeys = {"warmup",
                                             "iterations",
                                             "trials",
                                             "trial_cooldown_ms",
                                             "load_sample_ms",
                                             "threads",
                                             "binding",
                                             "init",
                                             "init_breakdown",
                                             "profile_accelerator",
                                             "profile_iterations",
                                             "cooldown_s"};

bool contains(const std::vector<std::string>& keys, const std::string& key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

std::string escape_pointer_token(const std::string& token) {
    std::string out;
    for (char c : token) {
        if (c == '~')
            out += "~0";
        else if (c == '/')
            out += "~1";
        else
            out += c;
    }
    return out;
}

std::string ptr(const std::string& base, const std::string& key) {
    return base + "/" + escape_pointer_token(key);
}
std::string ptr(const std::string& base, std::size_t index) {
    return base + "/" + std::to_string(index);
}

std::string type_name(const json& v) {
    return std::string(v.type_name());
}

// Collects problems while reading the document.
struct Reader {
    std::vector<SuiteConfigError> errors;

    void add(std::string pointer, std::string message, std::string kind = "suite_config_invalid") {
        errors.push_back({std::move(pointer), std::move(message), std::move(kind)});
    }

    // Unknown keys of `obj` (allowed: `keys`).
    void only_keys(const json& obj, const std::string& at, const std::vector<std::string>& keys) {
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            if (!contains(keys, it.key()))
                add(ptr(at, it.key()), "unknown key '" + it.key() + "'", "unknown_key");
        }
    }

    bool is_object(const json& v, const std::string& at) {
        if (v.is_object())
            return true;
        add(at, "must be an object (got " + type_name(v) + ")");
        return false;
    }
    bool is_array(const json& v, const std::string& at, bool non_empty) {
        if (!v.is_array()) {
            add(at, "must be an array (got " + type_name(v) + ")");
            return false;
        }
        if (non_empty && v.empty()) {
            add(at, "must not be empty");
            return false;
        }
        return true;
    }
    std::optional<std::string> string(const json& v, const std::string& at) {
        if (v.is_string())
            return v.get<std::string>();
        add(at, "must be a string (got " + type_name(v) + ")");
        return std::nullopt;
    }
    std::optional<bool> boolean(const json& v, const std::string& at) {
        if (v.is_boolean())
            return v.get<bool>();
        add(at, "must be true or false (got " + type_name(v) + ")");
        return std::nullopt;
    }
    std::optional<int> integer(const json& v, const std::string& at) {
        if (v.is_number_integer()) {
            const auto x = v.get<std::int64_t>();
            if (x >= -2147483647LL && x <= 2147483647LL)
                return static_cast<int>(x);
        }
        add(at, "must be an integer (got " + (v.is_number() ? v.dump() : type_name(v)) + ")");
        return std::nullopt;
    }
    std::optional<std::string> choice(const json& v, const std::string& at,
                                      const std::vector<std::string>& allowed) {
        const std::optional<std::string> s = string(v, at);
        if (!s)
            return std::nullopt;
        if (contains(allowed, *s))
            return s;
        std::string list;
        for (const std::string& a : allowed)
            list += (list.empty() ? "" : ", ") + a;
        add(at, "unsupported value '" + *s + "'; supported: " + list);
        return std::nullopt;
    }
};

// A model the configuration may use.
struct ModelFileEntry {
    fs::path path;        // as configured (relative to the config directory) or manifest-relative
    std::string display;  // what the manifest and messages show
    std::string precision;
    int batch = 1;
    std::optional<std::string> variant;
    std::optional<std::string> manifest_sha256;  // manifest models only
};
struct ModelEntry {
    std::string name;
    bool from_manifest = false;
    std::vector<ModelFileEntry> files;
};

// --------------------------------------------------------------------- parsing pieces

void read_identity_key(Reader& rd, const json& v, const std::string& at, const std::string& key,
                       Identity& id) {
    if (key == "model") {
        if (auto s = rd.string(v, at))
            id.model = *s;
    } else if (key == "provider") {
        if (auto s = rd.choice(v, at, {"cpu", "npu"}))
            id.provider = *parse_provider(*s);
    } else if (key == "precision") {
        if (auto s = rd.choice(v, at, {"fp32", "int8"}))
            id.precision = *s;
    } else if (key == "batch") {
        if (auto n = rd.integer(v, at)) {
            if (*n < 1)
                rd.add(at, "must be at least 1 (got " + std::to_string(*n) + ")");
            id.batch = *n;
        }
    } else if (key == "variant") {
        if (auto s = rd.string(v, at))
            id.variant = *s;
    } else if (key == "npu_fp16") {
        if (auto b = rd.boolean(v, at))
            id.npu_fp16 = *b;
    } else if (key == "htp_performance_mode") {
        if (v.is_null())
            id.htp_performance_mode.reset();
        else if (auto s = rd.string(v, at))
            id.htp_performance_mode = *s;
    } else if (key == "htp_graph_finalization_mode") {
        if (v.is_null())
            id.htp_graph_finalization_mode.reset();
        else if (auto s = rd.string(v, at))
            id.htp_graph_finalization_mode = *s;
    } else if (key == "strict") {
        if (auto b = rd.boolean(v, at))
            id.strict = *b;
    }
}

void read_param_key(Reader& rd, const json& v, const std::string& at, const std::string& key,
                    RunParams& p) {
    auto int_into = [&](int& target) {
        if (auto n = rd.integer(v, at))
            target = *n;
    };
    if (key == "warmup") {
        int_into(p.warmup);
    } else if (key == "iterations") {
        int_into(p.iterations);
    } else if (key == "trials") {
        int_into(p.trials);
    } else if (key == "trial_cooldown_ms") {
        int_into(p.trial_cooldown_ms);
    } else if (key == "load_sample_ms") {
        int_into(p.load_sample_ms);
    } else if (key == "threads") {
        int_into(p.threads);
    } else if (key == "binding") {
        if (auto s = rd.choice(v, at, {"run", "iobinding"}))
            p.binding = *parse_binding(*s);
    } else if (key == "init") {
        if (auto s = rd.choice(v, at, {"cold", "build", "cached"}))
            p.init = *s == "cold"    ? InitMode::cold
                     : *s == "build" ? InitMode::build
                                     : InitMode::cached;
    } else if (key == "init_breakdown") {
        if (auto b = rd.boolean(v, at))
            p.init_breakdown = *b;
    } else if (key == "profile_accelerator") {
        if (auto b = rd.boolean(v, at))
            p.profile_accelerator = *b;
    } else if (key == "profile_iterations") {
        if (auto n = rd.integer(v, at))
            p.profile_iterations = *n;
    } else if (key == "cooldown_s") {
        if (!v.is_number()) {
            rd.add(at, "must be a number of seconds (got " + type_name(v) + ")");
        } else {
            const double s = v.get<double>();
            if (!std::isfinite(s) || s < 0 || s > kMaxCooldownS)
                rd.add(at, "must be between 0 and " +
                               std::to_string(static_cast<int>(kMaxCooldownS)) + " seconds (got " +
                               v.dump() + ")");
            else
                p.cooldown_s = s;
        }
    }
}

void read_params(Reader& rd, const json& obj, const std::string& at, RunParams& p) {
    if (!rd.is_object(obj, at))
        return;
    rd.only_keys(obj, at, kParamKeys);
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        if (contains(kParamKeys, it.key()))
            read_param_key(rd, it.value(), ptr(at, it.key()), it.key(), p);
    }
}

// A match object: a subset of the identity keys with the values to compare.
struct Match {
    json keys;  // object
    std::string pointer;
    int matched = 0;
};

std::optional<Match> read_match(Reader& rd, const json& obj, const std::string& at,
                                bool allow_empty) {
    if (!rd.is_object(obj, at))
        return std::nullopt;
    rd.only_keys(obj, at, kIdentityKeys);
    if (obj.empty() && !allow_empty) {
        rd.add(at, "must name at least one key to match");
        return std::nullopt;
    }
    // Type-check the values by reading them into a throwaway identity.
    Identity scratch;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        if (contains(kIdentityKeys, it.key()))
            read_identity_key(rd, it.value(), ptr(at, it.key()), it.key(), scratch);
    }
    return Match{obj, at, 0};
}

json identity_json(const Identity& id) {
    return json{
        {"model", id.model},
        {"provider", cli_name(id.provider)},
        {"precision", id.precision},
        {"batch", id.batch},
        {"variant", id.variant ? json(*id.variant) : json(nullptr)},
        {"npu_fp16", id.npu_fp16},
        {"htp_performance_mode",
         id.htp_performance_mode ? json(*id.htp_performance_mode) : json(nullptr)},
        {"htp_graph_finalization_mode",
         id.htp_graph_finalization_mode ? json(*id.htp_graph_finalization_mode) : json(nullptr)},
        {"strict", id.strict},
    };
}

bool matches(const Match& m, const Identity& id) {
    const json values = identity_json(id);
    for (auto it = m.keys.begin(); it != m.keys.end(); ++it) {
        if (values.at(it.key()) != it.value())
            return false;
    }
    return true;
}

// A configuration as written: identity keys (some required) and run parameters.
struct RawConfiguration {
    Identity identity;
    json param_keys = json::object();  // the run parameters written in the entry itself
    std::string source;
    std::string pointer;
};

std::optional<RawConfiguration> read_configuration(Reader& rd, const json& obj,
                                                   const std::string& at, std::string source) {
    if (!rd.is_object(obj, at))
        return std::nullopt;
    std::vector<std::string> allowed = kIdentityKeys;
    allowed.insert(allowed.end(), kParamKeys.begin(), kParamKeys.end());
    rd.only_keys(obj, at, allowed);
    bool complete = true;
    for (const char* key : {"model", "provider", "precision", "batch"}) {
        if (!obj.contains(key)) {
            rd.add(at, std::string("missing required key '") + key + "'");
            complete = false;
        }
    }
    RawConfiguration c;
    c.source = std::move(source);
    c.pointer = at;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        if (contains(kIdentityKeys, it.key()))
            read_identity_key(rd, it.value(), ptr(at, it.key()), it.key(), c.identity);
        else if (contains(kParamKeys, it.key()))
            c.param_keys[it.key()] = it.value();
    }
    if (!complete)
        return std::nullopt;
    return c;
}

// ------------------------------------------------------------------------- models

std::optional<fs::path> find_models_dir(const fs::path& start) {
    std::error_code ec;
    for (fs::path dir = start; !dir.empty(); dir = dir.parent_path()) {
        if (fs::is_regular_file(dir / "models" / std::string(kModelsManifest), ec))
            return dir / "models";
        if (dir == dir.parent_path())
            break;
    }
    return std::nullopt;
}

fs::path from_utf8(const std::string& s) {
    return fs::path(std::u8string(s.begin(), s.end()));
}

std::string variant_default(const std::string& precision) {
    return precision == "int8" ? "int8" : "fp32";
}

// The label part for a non-default variant: "int8_folded" -> "_folded".
std::string variant_suffix(const std::optional<std::string>& variant,
                           const std::string& precision) {
    if (!variant || *variant == variant_default(precision))
        return {};
    const std::string prefix = precision + "_";
    if (variant->rfind(prefix, 0) == 0)
        return "_" + variant->substr(prefix.size());
    return "_" + *variant;
}

}  // namespace

// ------------------------------------------------------------------------- public

std::string_view to_string(InitMode mode) {
    switch (mode) {
        case InitMode::cold:
            return "cold";
        case InitMode::build:
            return "build";
        case InitMode::cached:
            return "cached";
    }
    return "cold";
}

std::string_view to_string(Rotation rotation) {
    return rotation == Rotation::blocked ? "blocked" : "rotated";
}

bool is_valid_suite_name(std::string_view name) {
    return is_valid_result_name(name);
}

std::string config_hash(const json& doc) {
    return sha256_hex(canonical_json(doc));
}

std::string run_id(const ojson& resolved, int repetition) {
    json j = json::parse(resolved.dump());
    j["repetition"] = repetition;
    return sha256_hex(canonical_json(j));
}

std::vector<std::pair<std::size_t, int>> run_order(std::size_t n, int repetitions,
                                                   Rotation rotation) {
    std::vector<std::pair<std::size_t, int>> order;
    if (n == 0 || repetitions <= 0)
        return order;
    if (rotation == Rotation::blocked) {
        for (std::size_t c = 0; c < n; ++c) {
            for (int r = 0; r < repetitions; ++r)
                order.emplace_back(c, r);
        }
        return order;
    }
    // First row of a Williams design: 0, 1, n-1, 2, n-2, ...; row r adds r (mod n).
    std::vector<std::size_t> base = {0};
    for (std::size_t k = 1, up = 1, down = n - 1; base.size() < n; ++k) {
        if (k % 2 == 1)
            base.push_back(up++);
        else
            base.push_back(down--);
    }
    const std::size_t period = n % 2 == 0 ? n : 2 * n;
    for (int r = 0; r < repetitions; ++r) {
        const std::size_t row = static_cast<std::size_t>(r) % period;
        std::vector<std::size_t> line(n);
        for (std::size_t k = 0; k < n; ++k)
            line[k] = (base[k] + row % n) % n;
        if (row >= n)  // odd n: the second n rows are the first n reversed
            std::reverse(line.begin(), line.end());
        for (std::size_t c : line)
            order.emplace_back(c, r);
    }
    return order;
}

Error to_error(const fs::path& config_file, const std::vector<SuiteConfigError>& errors) {
    std::string message = path_to_utf8(config_file) + ": " + std::to_string(errors.size()) +
                          (errors.size() == 1 ? " problem" : " problems");
    for (const SuiteConfigError& e : errors)
        message += "\n  " + (e.pointer.empty() ? std::string("(file)") : e.pointer) + ": " +
                   e.message + " [" + e.kind + "]";
    return Error{ExitCode::usage_error,
                 errors.empty() ? "suite_config_invalid" : errors.front().kind, message};
}

RunConfig to_run_config(const ResolvedConfiguration& c, const fs::path& output_dir,
                        const std::string& name, const std::optional<fs::path>& ep_cache_dir) {
    RunConfig r;
    r.model = c.model_file;
    r.provider = c.identity.provider;
    r.threads = c.params.threads;
    r.binding = c.params.binding;
    r.warmup = c.params.warmup;
    r.iterations = c.params.iterations;
    r.trials = c.params.trials;
    r.cooldown_ms = c.params.trial_cooldown_ms;
    r.load_sample_ms = c.params.load_sample_ms;
    r.output_dir = output_dir;
    r.name = name;
    if (!c.label.empty())
        r.label = c.label;
    r.strict = c.identity.strict;
    r.npu_fp16 = c.identity.npu_fp16;
    r.htp_performance_mode = c.identity.htp_performance_mode;
    r.htp_graph_finalization_mode = c.identity.htp_graph_finalization_mode;
    switch (c.params.init) {
        case InitMode::cold:
            r.ep_cache = EpCacheMode::off;
            break;
        case InitMode::build:
            r.ep_cache = EpCacheMode::build;
            break;
        case InitMode::cached:
            r.ep_cache = EpCacheMode::use;
            break;
    }
    r.init_breakdown = c.params.init_breakdown;
    r.profile_accelerator = c.params.profile_accelerator;
    r.profile_iterations = c.params.profile_iterations;
    if (r.uses_ep_context())
        r.ep_cache_dir = ep_cache_dir;
    return r;
}

RunConfig cache_build_config(const ResolvedConfiguration& c, const fs::path& output_dir,
                             const std::string& name, const fs::path& ep_cache_dir) {
    RunConfig r = to_run_config(c, output_dir, name, ep_cache_dir);
    r.trials = 1;
    r.iterations = 1;
    r.warmup = 0;
    r.cooldown_ms = 0;
    r.load_sample_ms = 0;
    r.ep_cache = EpCacheMode::build;
    r.ep_cache_dir = ep_cache_dir;
    r.init_breakdown = false;
    r.profile_accelerator = false;
    r.profile_iterations.reset();
    r.label = c.label + " (cache build)";
    return r;
}

std::vector<std::string> run_arguments(const RunConfig& c) {
    std::vector<std::string> a = {"run",
                                  "--model",
                                  path_to_utf8(c.model),
                                  "--provider",
                                  std::string(cli_name(c.provider)),
                                  "--threads",
                                  std::to_string(c.threads),
                                  "--binding",
                                  std::string(cli_name(c.binding)),
                                  "--warmup",
                                  std::to_string(c.warmup),
                                  "--iterations",
                                  std::to_string(c.iterations),
                                  "--trials",
                                  std::to_string(c.trials),
                                  "--cooldown-ms",
                                  std::to_string(c.cooldown_ms),
                                  "--load-sample-ms",
                                  std::to_string(c.load_sample_ms),
                                  "--output-dir",
                                  path_to_utf8(c.output_dir)};
    if (c.name)
        a.insert(a.end(), {"--name", *c.name});
    if (c.label)
        a.insert(a.end(), {"--label", *c.label});
    if (c.provider == ProviderKind::npu) {
        if (!c.strict)
            a.push_back("--no-strict");
        if (c.npu_fp16)
            a.push_back("--npu-fp16");
        if (c.htp_performance_mode)
            a.insert(a.end(), {"--htp-performance-mode", *c.htp_performance_mode});
        if (c.htp_graph_finalization_mode)
            a.insert(a.end(), {"--htp-graph-finalization-mode", *c.htp_graph_finalization_mode});
        a.insert(a.end(), {"--ep-cache", std::string(schema_name(c.ep_cache))});
        if (c.ep_cache_dir)
            a.insert(a.end(), {"--ep-cache-dir", path_to_utf8(*c.ep_cache_dir)});
        if (c.init_breakdown)
            a.push_back("--init-breakdown");
        if (c.profile_accelerator)
            a.push_back("--profile-accelerator");
        if (c.profile_iterations)
            a.insert(a.end(), {"--profile-iterations", std::to_string(*c.profile_iterations)});
    }
    if (c.suite) {
        const SuiteLink& s = *c.suite;
        a.insert(a.end(),
                 {"--suite-name", s.suite_name, "--suite-config-hash", s.config_hash,
                  "--suite-run-id", s.run_id, "--suite-repetition", std::to_string(s.repetition),
                  "--suite-order-index", std::to_string(s.order_index), "--suite-cooldown-s-before",
                  format_seconds(s.cooldown_s_before)});
        if (s.preflight_skipped)
            a.push_back("--suite-preflight-skipped");
    }
    return a;
}

LoadOutcome plan_suite(const json& doc, const fs::path& config_file,
                       const std::optional<fs::path>& output_dir) {
    Reader rd;
    LoadOutcome out;
    std::error_code ec;
    const fs::path config_abs = fs::absolute(config_file, ec);
    const fs::path base_dir = config_abs.parent_path();

    if (!doc.is_object()) {
        rd.add("", "the file must hold a JSON object");
        out.errors = rd.errors;
        return out;
    }
    rd.only_keys(doc, "",
                 {"suite_config_version", "name", "description", "output_dir", "models_dir",
                  "models", "configurations", "matrix", "defaults", "overrides", "repetitions",
                  "rotation", "cooldown_s", "preflight"});
    for (const char* key : {"suite_config_version", "name", "output_dir", "models"}) {
        if (!doc.contains(key))
            rd.add("", std::string("missing required key '") + key + "'");
    }
    if (doc.contains("suite_config_version")) {
        const auto v = rd.string(doc["suite_config_version"], "/suite_config_version");
        if (v && *v != kConfigVersion)
            rd.add("/suite_config_version",
                   "unsupported version '" + *v + "'; supported: " + std::string(kConfigVersion),
                   "suite_config_version_unsupported");
    }

    SuitePlan plan;
    plan.config_file = config_abs;
    plan.config = doc;
    plan.config_sha256 = config_hash(doc);
    if (doc.contains("name")) {
        if (auto s = rd.string(doc["name"], "/name")) {
            if (!is_valid_suite_name(*s))
                rd.add("/name",
                       "may only contain letters, digits, '-', '_' and '.', and must "
                       "not start with '.' (got '" +
                           *s + "')");
            plan.name = *s;
        }
    }
    if (doc.contains("description"))
        rd.string(doc["description"], "/description");
    if (output_dir) {
        plan.output_root = fs::absolute(*output_dir, ec);
    } else if (doc.contains("output_dir")) {
        if (auto s = rd.string(doc["output_dir"], "/output_dir")) {
            if (s->empty())
                rd.add("/output_dir", "must not be empty");
            else
                plan.output_root = (base_dir / from_utf8(*s)).lexically_normal();
        }
    }
    if (doc.contains("repetitions")) {
        if (auto n = rd.integer(doc["repetitions"], "/repetitions")) {
            if (*n < 1 || *n > kMaxRepetitions)
                rd.add("/repetitions", "must be between 1 and " + std::to_string(kMaxRepetitions) +
                                           " (got " + std::to_string(*n) + ")");
            plan.repetitions = *n;
        }
    }
    if (doc.contains("rotation")) {
        if (auto s = rd.choice(doc["rotation"], "/rotation", {"rotated", "blocked"}))
            plan.rotation = *s == "blocked" ? Rotation::blocked : Rotation::rotated;
    }
    if (doc.contains("preflight") && rd.is_object(doc["preflight"], "/preflight")) {
        const json& p = doc["preflight"];
        rd.only_keys(p, "/preflight", {"checks", "fail_on_warn"});
        if (p.contains("checks") && rd.is_array(p["checks"], "/preflight/checks", false)) {
            plan.preflight.checks.clear();
            for (std::size_t i = 0; i < p["checks"].size(); ++i) {
                if (auto s = rd.choice(p["checks"][i], ptr("/preflight/checks", i), {"doctor"})) {
                    if (contains(plan.preflight.checks, *s))
                        rd.add(ptr("/preflight/checks", i), "'" + *s + "' is listed twice");
                    plan.preflight.checks.push_back(*s);
                }
            }
        }
        if (p.contains("fail_on_warn")) {
            if (auto b = rd.boolean(p["fail_on_warn"], "/preflight/fail_on_warn"))
                plan.preflight.fail_on_warn = *b;
        }
    }

    // Defaults for every configuration: built-in, then cooldown_s, then defaults.
    RunParams defaults;
    if (doc.contains("cooldown_s"))
        read_param_key(rd, doc["cooldown_s"], "/cooldown_s", "cooldown_s", defaults);
    if (doc.contains("defaults"))
        read_params(rd, doc["defaults"], "/defaults", defaults);

    // Models.
    std::vector<ModelEntry> models;
    bool need_manifest = false;
    if (doc.contains("models") && rd.is_array(doc["models"], "/models", true)) {
        std::set<std::string> names;
        for (std::size_t i = 0; i < doc["models"].size(); ++i) {
            const json& m = doc["models"][i];
            const std::string at = ptr("/models", i);
            ModelEntry entry;
            if (m.is_string()) {
                entry.name = m.get<std::string>();
                entry.from_manifest = true;
                need_manifest = true;
            } else if (rd.is_object(m, at)) {
                rd.only_keys(m, at, {"name", "files"});
                if (!m.contains("name") || !m.contains("files")) {
                    rd.add(at, "a model object needs 'name' and 'files'");
                    continue;
                }
                if (auto s = rd.string(m["name"], ptr(at, "name")))
                    entry.name = *s;
                if (rd.is_array(m["files"], ptr(at, "files"), true)) {
                    for (std::size_t k = 0; k < m["files"].size(); ++k) {
                        const json& f = m["files"][k];
                        const std::string fat = ptr(ptr(at, "files"), k);
                        if (!rd.is_object(f, fat))
                            continue;
                        rd.only_keys(f, fat, {"path", "precision", "batch", "variant"});
                        ModelFileEntry file;
                        bool ok = true;
                        for (const char* key : {"path", "precision", "batch"}) {
                            if (!f.contains(key)) {
                                rd.add(fat, std::string("missing required key '") + key + "'");
                                ok = false;
                            }
                        }
                        if (!ok)
                            continue;
                        if (auto s = rd.string(f["path"], ptr(fat, "path"))) {
                            file.path = (base_dir / from_utf8(*s)).lexically_normal();
                            file.display = *s;
                        }
                        if (auto s =
                                rd.choice(f["precision"], ptr(fat, "precision"), {"fp32", "int8"}))
                            file.precision = *s;
                        if (auto n = rd.integer(f["batch"], ptr(fat, "batch")))
                            file.batch = *n;
                        if (f.contains("variant")) {
                            if (auto s = rd.string(f["variant"], ptr(fat, "variant")))
                                file.variant = *s;
                        }
                        entry.files.push_back(file);
                    }
                }
            } else {
                continue;
            }
            if (!entry.name.empty() && !is_valid_result_name(entry.name))
                rd.add(at, "model name '" + entry.name +
                               "' may only contain letters, digits, '-', '_' and '.'");
            if (!names.insert(entry.name).second)
                rd.add(at, "model '" + entry.name + "' is listed twice");
            models.push_back(std::move(entry));
        }
    }

    // models/manifest.json for manifest models.
    if (doc.contains("models_dir")) {
        if (auto s = rd.string(doc["models_dir"], "/models_dir"))
            plan.models_dir = (base_dir / from_utf8(*s)).lexically_normal();
    } else if (need_manifest) {
        plan.models_dir = find_models_dir(base_dir);
    }
    json manifest;
    if (need_manifest) {
        if (!plan.models_dir) {
            rd.add("/models",
                   "models named by their manifest name need models/manifest.json; "
                   "none was found above " +
                       path_to_utf8(base_dir) + " (set models_dir)",
                   "models_manifest_missing");
        } else {
            std::ifstream in(*plan.models_dir / std::string(kModelsManifest), std::ios::binary);
            try {
                manifest = json::parse(in);
                if (!manifest.contains("files") || !manifest["files"].is_array())
                    throw std::runtime_error("no files list");
            } catch (const std::exception& e) {
                rd.add("/models",
                       "cannot read " +
                           path_to_utf8(*plan.models_dir / std::string(kModelsManifest)) + ": " +
                           e.what(),
                       "models_manifest_missing");
                manifest = json();
            }
        }
        if (manifest.is_object()) {
            for (ModelEntry& m : models) {
                if (!m.from_manifest)
                    continue;
                for (const json& f : manifest["files"]) {
                    if (f.value("model", "") != m.name)
                        continue;
                    ModelFileEntry file;
                    file.display = f.value("file", "");
                    file.path = (*plan.models_dir / from_utf8(file.display)).lexically_normal();
                    file.precision = f.value("precision", "");
                    file.batch = f.value("batch_size", 0);
                    file.variant = f.value("variant", "");
                    file.manifest_sha256 = f.value("sha256", "");
                    m.files.push_back(file);
                }
                if (m.files.empty()) {
                    for (std::size_t i = 0; i < doc["models"].size(); ++i) {
                        if (doc["models"][i] == m.name)
                            rd.add(ptr("/models", i),
                                   "unknown model '" + m.name +
                                       "': models/manifest.json has no "
                                       "file of it",
                                   "unknown_model");
                    }
                }
            }
        }
    }

    // Configurations: an explicit list or a matrix.
    std::vector<RawConfiguration> raw;
    const bool has_list = doc.contains("configurations");
    const bool has_matrix = doc.contains("matrix");
    std::vector<Match> excludes;
    if (has_list == has_matrix) {
        rd.add("", "give exactly one of 'configurations' and 'matrix'");
    } else if (has_list) {
        const json& list = doc["configurations"];
        if (rd.is_array(list, "/configurations", true)) {
            for (std::size_t i = 0; i < list.size(); ++i) {
                if (auto c = read_configuration(rd, list[i], ptr("/configurations", i),
                                                "configurations[" + std::to_string(i) + "]"))
                    raw.push_back(std::move(*c));
            }
        }
    } else if (rd.is_object(doc["matrix"], "/matrix")) {
        const json& mx = doc["matrix"];
        rd.only_keys(mx, "/matrix",
                     {"models", "providers", "precisions", "batches", "npu", "include", "exclude"});
        for (const char* key : {"providers", "precisions", "batches"}) {
            if (!mx.contains(key))
                rd.add("/matrix", std::string("missing required key '") + key + "'");
        }
        auto axis = [&](const char* key, std::vector<json>& values) {
            const std::string at = ptr("/matrix", key);
            if (!mx.contains(key) || !rd.is_array(mx[key], at, true))
                return;
            for (std::size_t i = 0; i < mx[key].size(); ++i) {
                const json& v = mx[key][i];
                if (std::find(values.begin(), values.end(), v) != values.end())
                    rd.add(ptr(at, i), "value " + v.dump() + " is listed twice");
                values.push_back(v);
            }
        };
        std::vector<json> model_axis, providers, precisions, batches;
        if (mx.contains("models"))
            axis("models", model_axis);
        else
            for (const ModelEntry& m : models)
                model_axis.push_back(m.name);
        axis("providers", providers);
        axis("precisions", precisions);
        axis("batches", batches);
        std::map<std::string, std::vector<json>> npu_axes = {
            {"npu_fp16", {false}},
            {"htp_performance_mode", {nullptr}},
            {"htp_graph_finalization_mode", {nullptr}},
            {"strict", {true}}};
        if (mx.contains("npu") && rd.is_object(mx["npu"], "/matrix/npu")) {
            const json& npu = mx["npu"];
            rd.only_keys(
                npu, "/matrix/npu",
                {"npu_fp16", "htp_performance_mode", "htp_graph_finalization_mode", "strict"});
            for (auto& [key, values] : npu_axes) {
                if (!npu.contains(key))
                    continue;
                values.clear();
                const std::string at = ptr("/matrix/npu", key);
                if (!rd.is_array(npu[key], at, true))
                    continue;
                for (std::size_t i = 0; i < npu[key].size(); ++i)
                    values.push_back(npu[key][i]);
            }
        }
        // Type-check the axis values once.
        Identity scratch;
        for (std::size_t i = 0; i < model_axis.size(); ++i)
            read_identity_key(rd, model_axis[i], ptr("/matrix/models", i), "model", scratch);
        for (std::size_t i = 0; i < providers.size(); ++i)
            read_identity_key(rd, providers[i], ptr("/matrix/providers", i), "provider", scratch);
        for (std::size_t i = 0; i < precisions.size(); ++i)
            read_identity_key(rd, precisions[i], ptr("/matrix/precisions", i), "precision",
                              scratch);
        for (std::size_t i = 0; i < batches.size(); ++i)
            read_identity_key(rd, batches[i], ptr("/matrix/batches", i), "batch", scratch);
        for (auto& [key, values] : npu_axes) {
            for (std::size_t i = 0; i < values.size(); ++i)
                read_identity_key(rd, values[i], ptr(ptr("/matrix/npu", key), i), key, scratch);
        }
        const std::size_t before_product = rd.errors.size();
        if (before_product == 0) {
            std::vector<json> npu_combos = {json::object()};
            for (const auto& [key, values] : npu_axes) {
                std::vector<json> next;
                for (const json& partial : npu_combos) {
                    for (const json& v : values) {
                        json c = partial;
                        c[key] = v;
                        next.push_back(c);
                    }
                }
                npu_combos = std::move(next);
            }
            for (const json& m : model_axis)
                for (const json& p : providers)
                    for (const json& pr : precisions)
                        for (const json& b : batches) {
                            const bool npu = p == "npu";
                            for (const json& extra :
                                 npu ? npu_combos : std::vector<json>{json::object()}) {
                                RawConfiguration c;
                                c.source = "matrix";
                                c.pointer = "/matrix";
                                Identity& id = c.identity;
                                id.model = m.get<std::string>();
                                id.provider = *parse_provider(p.get<std::string>());
                                id.precision = pr.get<std::string>();
                                id.batch = b.get<int>();
                                for (auto it = extra.begin(); it != extra.end(); ++it)
                                    read_identity_key(rd, it.value(), "/matrix/npu", it.key(), id);
                                raw.push_back(std::move(c));
                            }
                        }
        }
        if (mx.contains("exclude") && rd.is_array(mx["exclude"], "/matrix/exclude", false)) {
            for (std::size_t i = 0; i < mx["exclude"].size(); ++i) {
                if (auto m = read_match(rd, mx["exclude"][i], ptr("/matrix/exclude", i), false))
                    excludes.push_back(std::move(*m));
            }
        }
        std::vector<RawConfiguration> kept;
        for (RawConfiguration& c : raw) {
            bool drop = false;
            for (Match& m : excludes) {
                if (matches(m, c.identity)) {
                    ++m.matched;
                    drop = true;
                }
            }
            if (!drop)
                kept.push_back(std::move(c));
        }
        raw = std::move(kept);
        for (const Match& m : excludes) {
            if (m.matched == 0 && before_product == 0 && rd.errors.empty())
                rd.add(m.pointer, "matches no configuration of the matrix", "unmatched_exclude");
        }
        if (mx.contains("include") && rd.is_array(mx["include"], "/matrix/include", false)) {
            for (std::size_t i = 0; i < mx["include"].size(); ++i) {
                if (auto c = read_configuration(rd, mx["include"][i], ptr("/matrix/include", i),
                                                "matrix.include[" + std::to_string(i) + "]"))
                    raw.push_back(std::move(*c));
            }
        }
    }

    // Overrides: run parameters for the configurations a match selects.
    struct Override {
        Match match;
        json set;
    };
    std::vector<Override> overrides;
    if (doc.contains("overrides") && rd.is_array(doc["overrides"], "/overrides", false)) {
        for (std::size_t i = 0; i < doc["overrides"].size(); ++i) {
            const json& o = doc["overrides"][i];
            const std::string at = ptr("/overrides", i);
            if (!rd.is_object(o, at))
                continue;
            rd.only_keys(o, at, {"match", "set"});
            if (!o.contains("match") || !o.contains("set")) {
                rd.add(at, "an override needs 'match' and 'set'");
                continue;
            }
            auto m = read_match(rd, o["match"], ptr(at, "match"), true);
            RunParams scratch;
            read_params(rd, o["set"], ptr(at, "set"), scratch);
            if (o["set"].is_object() && o["set"].empty())
                rd.add(ptr(at, "set"), "must set at least one run parameter");
            if (m)
                overrides.push_back({std::move(*m), o["set"]});
        }
    }

    if (!rd.errors.empty()) {
        out.errors = rd.errors;
        return out;
    }
    if (raw.empty()) {
        rd.add(has_matrix ? "/matrix" : "/configurations", "no configuration is left to run",
               "no_configurations");
        out.errors = rd.errors;
        return out;
    }

    // Resolve every configuration.
    std::map<std::string, std::string> seen_ids;  // configuration id -> source
    std::map<fs::path, std::string> file_hashes;
    for (std::size_t index = 0; index < raw.size(); ++index) {
        RawConfiguration& rc = raw[index];
        ResolvedConfiguration c;
        c.identity = rc.identity;
        c.source = rc.source;
        c.source_index = index;
        c.params = defaults;
        for (Override& o : overrides) {
            if (matches(o.match, c.identity)) {
                ++o.match.matched;
                read_params(rd, o.set,
                            ptr(o.match.pointer.substr(0, o.match.pointer.rfind('/')), "set"),
                            c.params);
            }
        }
        for (auto it = rc.param_keys.begin(); it != rc.param_keys.end(); ++it)
            read_param_key(rd, it.value(), ptr(rc.pointer, it.key()), it.key(), c.params);

        const Identity& id = c.identity;
        const std::string what = id.model + " " + std::string(cli_name(id.provider)) + " " +
                                 id.precision + " b" + std::to_string(id.batch) + " (" + c.source +
                                 ")";
        const std::string at = rc.pointer;

        // The model file.
        const auto model_it = std::find_if(models.begin(), models.end(),
                                           [&](const ModelEntry& m) { return m.name == id.model; });
        if (model_it == models.end()) {
            rd.add(at, what + ": unknown model '" + id.model + "' (not in 'models')",
                   "unknown_model");
            continue;
        }
        const std::string wanted_variant = id.variant ? *id.variant : variant_default(id.precision);
        const ModelFileEntry* file = nullptr;
        for (const ModelFileEntry& f : model_it->files) {
            const bool variant_ok = model_it->from_manifest
                                        ? f.variant == wanted_variant
                                        : (!id.variant || f.variant == id.variant);
            if (f.batch == id.batch && f.precision == id.precision && variant_ok) {
                if (file != nullptr && !model_it->from_manifest) {
                    rd.add(at,
                           what + ": more than one file of model '" + id.model +
                               "' matches; give each a variant",
                           "ambiguous_model_file");
                }
                file = &f;
            }
        }
        if (file == nullptr) {
            rd.add(at,
                   what + ": model '" + id.model + "' has no " + id.precision + " file at batch " +
                       std::to_string(id.batch) +
                       (model_it->from_manifest ? " (variant " + wanted_variant + ")" : "") +
                       "; exclude this configuration or add the file",
                   "no_model_file");
            continue;
        }
        c.model_file = fs::absolute(file->path, ec);
        c.model_file_display = file->display;
        c.variant = file->variant;
        if (!fs::is_regular_file(c.model_file, ec)) {
            rd.add(at, what + ": model file not found: " + path_to_utf8(c.model_file),
                   "model_file_missing");
            continue;
        }
        if (!file_hashes.contains(c.model_file)) {
            const Result<std::string> sha = sha256_file(c.model_file);
            file_hashes[c.model_file] = sha ? sha.value() : std::string();
        }
        c.model_sha256 = file_hashes[c.model_file];
        c.model_size_bytes = fs::file_size(c.model_file, ec);
        if (c.model_sha256.empty()) {
            rd.add(at, what + ": cannot read " + path_to_utf8(c.model_file), "model_file_missing");
            continue;
        }
        if (file->manifest_sha256 && *file->manifest_sha256 != c.model_sha256) {
            rd.add(at,
                   what + ": " + file->display + " differs from models/manifest.json (SHA256 " +
                       c.model_sha256.substr(0, 12) + "..., manifest " +
                       file->manifest_sha256->substr(0, 12) + "...); regenerate it",
                   "model_file_mismatch");
            continue;
        }
        const Result<OnnxGraphSummary> graph = inspect_onnx_file(c.model_file);
        if (!graph) {
            rd.add(at, what + ": " + graph.error().message, "model_file_mismatch");
            continue;
        }
        const ModelPrecision stored = infer_precision(graph.value());
        if (stored.stored_precision != id.precision) {
            rd.add(at,
                   what + ": " + c.model_file_display + " stores " + stored.stored_precision +
                       ", not " + id.precision,
                   "model_file_mismatch");
            continue;
        }
        if (!graph.value().inputs.empty() && !graph.value().inputs.front().shape.empty()) {
            const std::int64_t dim0 = graph.value().inputs.front().shape.front();
            if (dim0 > 0 && dim0 != id.batch) {
                rd.add(at,
                       what + ": " + c.model_file_display + " has batch size " +
                           std::to_string(dim0) + ", not " + std::to_string(id.batch),
                       "model_file_mismatch");
                continue;
            }
        }

        // The run it stands for: the same rules as `edgebench run`.
        const RunConfig run = to_run_config(c, "suite-output", "suite-run", std::nullopt);
        static const std::map<std::string, std::string> kSuiteKey = {
            {"model", "model"},
            {"threads", "threads"},
            {"warmup", "warmup"},
            {"iterations", "iterations"},
            {"trials", "trials"},
            {"cooldown-ms", "trial_cooldown_ms"},
            {"load-sample-ms", "load_sample_ms"},
            {"npu-fp16", "npu_fp16"},
            {"no-strict", "strict"},
            {"htp-performance-mode", "htp_performance_mode"},
            {"htp-graph-finalization-mode", "htp_graph_finalization_mode"},
            {"ep-cache", "init"},
            {"ep-cache-dir", "init"},
            {"init-breakdown", "init_breakdown"},
            {"profile-accelerator", "profile_accelerator"},
            {"profile-iterations", "profile_iterations"}};
        bool valid = true;
        for (const ConfigError& e : validate(run)) {
            const auto key = kSuiteKey.find(e.field);
            std::string message = e.message;
            // validate() names the run options; the suite's keys are what the user wrote.
            for (const auto& [option, suite_key] : kSuiteKey) {
                const std::string flag = "--" + option;
                for (std::size_t p = message.find(flag); p != std::string::npos;
                     p = message.find(flag, p + suite_key.size()))
                    message.replace(p, flag.size(), suite_key);
            }
            if (message.find("provider npu") != std::string::npos ||
                message.find("--provider") != std::string::npos) {
                for (std::size_t p = message.find("--provider"); p != std::string::npos;
                     p = message.find("--provider", p))
                    message.replace(p, 10, "provider");
            }
            rd.add(at,
                   what + ": " + (key != kSuiteKey.end() ? key->second : e.field) + ": " + message,
                   e.kind == "usage_error" ? "invalid_combination" : e.kind);
            valid = false;
        }
        if (!valid)
            continue;
        const Result<std::optional<std::string>> compute = requested_compute_precision(run, stored);
        if (!compute) {
            std::string message = compute.error().message;
            for (std::size_t p = message.find("--npu-fp16"); p != std::string::npos;
                 p = message.find("--npu-fp16", p))
                message.replace(p, 10, "npu_fp16");
            rd.add(at, what + ": " + message, "invalid_combination");
            continue;
        }
        c.compute_precision = compute.value() ? *compute.value() : id.precision;
        if (id.provider == ProviderKind::npu) {
            c.ep_compile_options = compile_relevant_ep_options(qnn_ep_options(run, "QnnHtp.dll"));
        }

        // Label: <model>_b<batch>_<provider>_<compute>[_<variant>][_finalization<m>][_htp-<mode>]
        c.label = id.model + "_b" + std::to_string(id.batch) + "_" +
                  std::string(cli_name(id.provider)) + "_" + c.compute_precision +
                  variant_suffix(c.variant, id.precision);
        if (id.htp_graph_finalization_mode)
            c.label += "_finalization" + *id.htp_graph_finalization_mode;
        if (id.htp_performance_mode && *id.htp_performance_mode != kDefaultHtpPerformanceMode)
            c.label += "_htp-" + *id.htp_performance_mode;
        if (!id.strict)
            c.label += "_nonstrict";

        const bool npu = id.provider == ProviderKind::npu;
        auto opt_str = [](const std::optional<std::string>& s) {
            return s ? ojson(*s) : ojson(nullptr);
        };
        ojson compile_options = ojson::object();
        for (const auto& [k, v] : c.ep_compile_options)
            compile_options[k] = v;
        c.resolved = ojson{
            {"run_id_version", kRunIdVersion},
            {"model", id.model},
            {"model_file", c.model_file_display},
            {"model_sha256", c.model_sha256},
            {"variant", opt_str(c.variant)},
            {"provider", cli_name(id.provider)},
            {"precision", id.precision},
            {"compute_precision", c.compute_precision},
            {"batch", id.batch},
            {"strict", strict_mode(run)},
            {"npu_fp16", id.npu_fp16},
            {"htp_performance_mode",
             npu ? ojson(effective_htp_performance_mode(run)) : ojson(nullptr)},
            {"htp_graph_finalization_mode", opt_str(id.htp_graph_finalization_mode)},
            {"ep_compile_options", npu ? compile_options : ojson(nullptr)},
            {"warmup", c.params.warmup},
            {"iterations", c.params.iterations},
            {"trials", c.params.trials},
            {"trial_cooldown_ms", c.params.trial_cooldown_ms},
            {"load_sample_ms", c.params.load_sample_ms},
            {"threads", c.params.threads},
            {"binding", cli_name(c.params.binding)},
            {"input", "random"},
            {"seed", 0},
            {"graph_optimization_level", schema_name(GraphOptLevel::enable_all)},
            {"init", to_string(c.params.init)},
            {"init_breakdown", c.params.init_breakdown},
            {"profile_accelerator", c.params.profile_accelerator},
            {"profile_iterations", c.params.profile_accelerator
                                       ? ojson(run.effective_profile_iterations())
                                       : ojson(nullptr)},
            {"cooldown_s", c.params.cooldown_s},
        };
        c.configuration_id = sha256_hex(canonical_json(json::parse(c.resolved.dump())));
        if (const auto dup = seen_ids.find(c.configuration_id); dup != seen_ids.end()) {
            rd.add(at, what + ": the same configuration as " + dup->second,
                   "duplicate_configuration");
            continue;
        }
        seen_ids[c.configuration_id] = c.source;
        plan.configurations.push_back(std::move(c));
    }
    for (const Override& o : overrides) {
        if (o.match.matched == 0)
            rd.add(o.match.pointer, "matches no configuration", "unmatched_override");
    }
    if (!rd.errors.empty()) {
        out.errors = rd.errors;
        return out;
    }

    for (const auto& [config, repetition] :
         run_order(plan.configurations.size(), plan.repetitions, plan.rotation)) {
        PlannedRun r;
        r.configuration = config;
        r.repetition = repetition;
        r.order_index = static_cast<int>(plan.runs.size());
        r.run_id = run_id(plan.configurations[config].resolved, repetition);
        plan.runs.push_back(r);
    }
    out.plan = std::move(plan);
    return out;
}

Result<SuitePlan> load_suite(const fs::path& config_file,
                             const std::optional<fs::path>& output_dir) {
    std::error_code ec;
    if (!fs::is_regular_file(config_file, ec)) {
        return Error{ExitCode::usage_error, "suite_config_not_found",
                     "suite configuration not found: " + path_to_utf8(config_file)};
    }
    std::ifstream in(config_file, std::ios::binary);
    json doc;
    try {
        doc = json::parse(in);
    } catch (const json::parse_error& e) {
        return Error{ExitCode::usage_error, "suite_config_not_json",
                     path_to_utf8(config_file) + ": not valid JSON: " + e.what()};
    }
    LoadOutcome outcome = plan_suite(doc, config_file, output_dir);
    if (!outcome.plan)
        return to_error(config_file, outcome.errors);
    return std::move(*outcome.plan);
}

}  // namespace edgebench::suite
