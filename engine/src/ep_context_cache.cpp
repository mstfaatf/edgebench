#include "edgebench/ep_context_cache.hpp"

#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "edgebench/npu_options.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/result_files.hpp"
#include "edgebench/sha256.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace edgebench {

namespace {

Error not_writable(const fs::path& dir, const std::string& why) {
    return Error{
        ExitCode::output_error, "ep_cache_dir_not_writable",
        "the EP context cache directory " + path_to_utf8(dir) + " is not writable: " + why};
}

Error delete_failed(const fs::path& path, const std::string& why) {
    return Error{ExitCode::output_error, "ep_cache_delete_failed",
                 "cannot delete " + path_to_utf8(path) + " from the EP context cache: " + why};
}

std::string last_error_text() {
    const DWORD code = GetLastError();
    wchar_t* buffer = nullptr;
    const DWORD n = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
    std::wstring text = n ? std::wstring(buffer, n) : L"error " + std::to_wstring(code);
    if (buffer)
        LocalFree(buffer);
    while (!text.empty() && (text.back() == L'\n' || text.back() == L'\r' || text.back() == L'.'))
        text.pop_back();
    return path_to_utf8(fs::path(text));
}

// A name of a file directly inside an entry directory: no separators, no "..".
bool is_plain_file_name(const std::string& name) {
    return !name.empty() && name != "." && name != ".." &&
           name.find_first_of("/\\:") == std::string::npos;
}

fs::path utf8_path(const std::string& s) {
    return fs::path(std::u8string(s.begin(), s.end()));
}

bool same_path(const fs::path& a, const fs::path& b) {
    std::error_code ec1, ec2;
    const fs::path x = fs::weakly_canonical(a, ec1), y = fs::weakly_canonical(b, ec2);
    if (ec1 || ec2)
        return false;
    return _wcsicmp(x.c_str(), y.c_str()) == 0;
}

}  // namespace

std::map<std::string, std::string> compile_session_config(
    const std::map<std::string, std::string>& session_config) {
    std::map<std::string, std::string> out;
    for (const auto& [key, value] : session_config) {
        if (key != kEpContextEnableKey && key != kEpContextFilePathKey)
            out.emplace(key, value);
    }
    return out;
}

json cache_key_json(const CacheKeyFields& k) {
    auto opt = [](const std::optional<std::string>& v) { return v ? json(*v) : json(nullptr); };
    json device = nullptr;
    if (k.device)
        device = {{"device_type", k.device->device_type},
                  {"vendor_id", k.device->vendor_id},
                  {"device_id", k.device->device_id}};
    // nlohmann::json keeps object keys sorted, so the order here does not matter.
    return {{"model_sha256", k.model_sha256},
            {"ort_version", opt(k.ort_version)},
            {"ep_version", opt(k.ep_version)},
            {"qnn_sdk_version", opt(k.qnn_sdk_version)},
            {"qnn_backend_file_version", opt(k.qnn_backend_file_version)},
            {"ep_options", k.ep_options},
            {"ep_options_sha256", ep_options_sha256(k.ep_options)},
            {"session_config", k.session_config},
            {"graph_optimization_level", k.graph_optimization_level},
            {"device", device}};
}

std::optional<CacheKeyFields> cache_key_from_json(const json& j) {
    try {
        CacheKeyFields k;
        auto opt = [&j](const char* key) -> std::optional<std::string> {
            const json& v = j.at(key);
            return v.is_null() ? std::nullopt : std::optional<std::string>(v.get<std::string>());
        };
        k.model_sha256 = j.at("model_sha256").get<std::string>();
        k.ort_version = opt("ort_version");
        k.ep_version = opt("ep_version");
        k.qnn_sdk_version = opt("qnn_sdk_version");
        k.qnn_backend_file_version = opt("qnn_backend_file_version");
        k.ep_options = j.at("ep_options").get<std::map<std::string, std::string>>();
        k.session_config = j.at("session_config").get<std::map<std::string, std::string>>();
        k.graph_optimization_level = j.at("graph_optimization_level").get<std::string>();
        if (const json& d = j.at("device"); !d.is_null())
            k.device = CacheDevice{d.at("device_type").get<std::string>(),
                                   d.at("vendor_id").get<std::uint32_t>(),
                                   d.at("device_id").get<std::uint32_t>()};
        return k;
    } catch (const json::exception&) {
        return std::nullopt;
    }
}

std::string canonical_json(const json& j) {
    return j.dump(-1, ' ', /*ensure_ascii=*/false, json::error_handler_t::strict);
}

std::string cache_key_id(const CacheKeyFields& key) {
    return sha256_hex(canonical_json(cache_key_json(key)));
}

std::string ep_options_sha256(const std::map<std::string, std::string>& ep_options) {
    return sha256_hex(canonical_json(json(ep_options)));
}

std::string context_model_name(const fs::path& model) {
    return path_to_utf8(model.stem()) + "_ctx.onnx";
}

Result<fs::path> default_ep_cache_dir() {
    wchar_t* value = nullptr;
    std::size_t size = 0;
    if (_wdupenv_s(&value, &size, L"LOCALAPPDATA") != 0 || value == nullptr || *value == L'\0') {
        std::free(value);
        return not_writable(fs::path("%LOCALAPPDATA%") / "EdgeBench" / "qnn-context",
                            "LOCALAPPDATA is not set; pass --ep-cache-dir");
    }
    const fs::path dir = fs::path(value) / "EdgeBench" / "qnn-context";
    std::free(value);
    return dir;
}

Result<fs::path> prepare_ep_cache_dir(const fs::path& dir) {
    std::error_code ec;
    const fs::path absolute = fs::absolute(dir, ec);
    if (ec)
        return not_writable(dir, ec.message());
    fs::create_directories(absolute, ec);
    if (ec || !fs::is_directory(absolute, ec))
        return not_writable(absolute, ec ? ec.message() : "it is not a directory");
    // Create (and at once delete) a file: the only reliable writability test.
    const fs::path probe =
        absolute / (".edgebench-write-test-" + std::to_string(GetCurrentProcessId()));
    HANDLE h = CreateFileW(probe.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                           FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        return not_writable(absolute, last_error_text());
    CloseHandle(h);
    return absolute;
}

std::optional<CacheManifest> read_cache_manifest(const fs::path& entry_dir) {
    std::ifstream in(entry_dir / kEpCacheManifestName, std::ios::binary);
    if (!in)
        return std::nullopt;
    const json j = json::parse(in, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object() ||
        j.value("edgebench_cache_version", "") != kEpCacheManifestVersion)
        return std::nullopt;
    try {
        CacheManifest m;
        m.key_id = j.at("key_id").get<std::string>();
        m.key_fields = j.at("key_fields");
        m.created_utc = j.at("created_utc").get<std::string>();
        m.context_model = j.at("context_model").get<std::string>();
        if (m.key_id.size() != 64 || !m.key_fields.is_object() ||
            !is_plain_file_name(m.context_model))
            return std::nullopt;
        for (const json& f : j.at("files")) {
            CacheFile file{f.at("name").get<std::string>(), f.at("size_bytes").get<std::uint64_t>(),
                           f.at("sha256").get<std::string>()};
            if (!is_plain_file_name(file.name) || file.name == kEpCacheManifestName)
                return std::nullopt;
            m.files.push_back(std::move(file));
        }
        return m;
    } catch (const json::exception&) {
        return std::nullopt;
    }
}

Result<CacheManifest> write_cache_manifest(const fs::path& entry_dir, const std::string& key_id,
                                           const json& key_fields, const std::string& context_model,
                                           const std::string& created_utc) {
    CacheManifest m{key_id, key_fields, created_utc, context_model, {}};
    std::error_code ec;
    for (const fs::directory_entry& e : fs::directory_iterator(entry_dir, ec)) {
        const std::string name = path_to_utf8(e.path().filename());
        if (name == kEpCacheManifestName || !e.is_regular_file())
            continue;
        Result<std::string> sha = sha256_file(e.path());
        if (!sha)
            return sha.error();
        m.files.push_back({name, static_cast<std::uint64_t>(e.file_size()), sha.value()});
    }
    if (ec)
        return not_writable(entry_dir, ec.message());
    std::sort(m.files.begin(), m.files.end(),
              [](const CacheFile& a, const CacheFile& b) { return a.name < b.name; });
    json files = json::array();
    for (const CacheFile& f : m.files)
        files.push_back({{"name", f.name}, {"size_bytes", f.size_bytes}, {"sha256", f.sha256}});
    const nlohmann::ordered_json doc = {
        {"edgebench_cache_version", kEpCacheManifestVersion},
        {"description",
         "EP context cache entry written by edgebench. Only the files listed here are deleted "
         "with the entry."},
        {"key_id", key_id},
        {"key_fields", key_fields},
        {"created_utc", created_utc},
        {"context_model", context_model},
        {"files", files},
    };
    if (Result<void> w =
            replace_file_atomically(entry_dir / kEpCacheManifestName, doc.dump(2) + "\n");
        !w)
        return w.error();
    return m;
}

Result<std::vector<fs::path>> delete_cache_entry(const fs::path& cache_dir,
                                                 const fs::path& entry_dir) {
    std::vector<fs::path> deleted;
    if (!same_path(entry_dir.parent_path(), cache_dir))
        return delete_failed(entry_dir,
                             "it is not a direct subdirectory of " + path_to_utf8(cache_dir));
    const std::optional<CacheManifest> manifest = read_cache_manifest(entry_dir);
    if (!manifest)
        return deleted;  // not an entry the engine wrote: leave it alone
    std::vector<fs::path> targets;
    for (const CacheFile& f : manifest->files)
        targets.push_back(entry_dir / utf8_path(f.name));
    targets.push_back(entry_dir / kEpCacheManifestName);  // last: the list of what to delete
    for (const fs::path& file : targets) {
        std::error_code ec;
        if (!fs::exists(file, ec))
            continue;
        if (!fs::is_regular_file(file, ec))
            return delete_failed(file, "it is not a regular file");
        if (!fs::remove(file, ec) || ec)
            return delete_failed(file, ec ? ec.message() : "remove failed");
        deleted.push_back(file);
    }
    std::error_code ec;
    if (fs::is_empty(entry_dir, ec) && !ec)
        fs::remove(entry_dir, ec);  // only an empty directory; anything else stays
    return deleted;
}

CacheEnvironment cache_environment(const CacheKeyFields& key) {
    CacheEnvironment env{key.ort_version, key.ep_version, key.qnn_sdk_version,
                         key.qnn_backend_file_version, std::nullopt};
    if (key.device)
        env.device = cache_key_json(key).at("device");
    return env;
}

std::vector<std::string> stale_key_fields(const json& k, const CacheEnvironment& cur) {
    std::vector<std::string> stale;
    auto compare = [&](const char* field, const std::optional<std::string>& value) {
        if (!value)
            return;
        const auto it = k.find(field);
        if (it == k.end() || !it->is_string() || it->get<std::string>() != *value)
            stale.emplace_back(field);
    };
    compare("ort_version", cur.ort_version);
    compare("ep_version", cur.ep_version);
    compare("qnn_sdk_version", cur.qnn_sdk_version);
    compare("qnn_backend_file_version", cur.qnn_backend_file_version);
    if (cur.device) {
        const auto it = k.find("device");
        if (it == k.end() || *it != *cur.device)
            stale.emplace_back("device");
    }
    return stale;
}

bool same_model_configuration(const json& a, const json& b) {
    try {
        const auto options = [](const json& k) {
            return compile_relevant_ep_options(
                k.at("ep_options").get<std::map<std::string, std::string>>());
        };
        return a.at("model_sha256") == b.at("model_sha256") && options(a) == options(b) &&
               a.at("session_config") == b.at("session_config") &&
               a.at("graph_optimization_level") == b.at("graph_optimization_level");
    } catch (const json::exception&) {
        return false;  // a malformed key is never treated as this run's
    }
}

Result<std::vector<fs::path>> delete_stale_cache_entries(const fs::path& cache_dir,
                                                         const CacheKeyFields& current) {
    std::vector<fs::path> deleted;
    std::vector<fs::path> stale;
    const json current_key = cache_key_json(current);
    const std::string keep_key_id = cache_key_id(current);
    const CacheEnvironment env = cache_environment(current);
    std::error_code ec;
    for (const fs::directory_entry& e : fs::directory_iterator(cache_dir, ec)) {
        if (!e.is_directory())
            continue;
        const std::optional<CacheManifest> m = read_cache_manifest(e.path());
        if (!m || path_to_utf8(e.path().filename()) != m->key_id || m->key_id == keep_key_id)
            continue;
        if (same_model_configuration(m->key_fields, current_key) &&
            !stale_key_fields(m->key_fields, env).empty())
            stale.push_back(e.path());
    }
    if (ec)
        return delete_failed(cache_dir, ec.message());
    for (const fs::path& dir : stale) {
        Result<std::vector<fs::path>> d = delete_cache_entry(cache_dir, dir);
        if (!d)
            return d.error();
        deleted.insert(deleted.end(), d.value().begin(), d.value().end());
    }
    return deleted;
}

CacheLookup lookup_cache_entry(const fs::path& cache_dir, const std::string& key_id,
                               const json& key_fields) {
    CacheLookup out;
    out.entry_dir = cache_dir / key_id;
    std::error_code ec;
    if (!fs::exists(out.entry_dir, ec)) {
        out.state = CacheEntryState::missing;
        return out;
    }
    out.manifest = read_cache_manifest(out.entry_dir);
    if (!fs::is_directory(out.entry_dir, ec) || !out.manifest) {
        out.state = CacheEntryState::foreign;
        out.manifest.reset();
        out.problem = path_to_utf8(out.entry_dir) + " has no manifest written by edgebench";
        return out;
    }
    const CacheManifest& m = *out.manifest;
    out.state = CacheEntryState::corrupt;
    if (m.key_id != key_id || m.key_fields != key_fields) {
        out.problem = "its manifest records a different key";
        return out;
    }
    const bool has_model = std::any_of(m.files.begin(), m.files.end(), [&m](const CacheFile& f) {
        return f.name == m.context_model;
    });
    if (!has_model) {
        out.problem = "its manifest does not list the context model " + m.context_model;
        return out;
    }
    for (const CacheFile& f : m.files) {
        const fs::path file = out.entry_dir / utf8_path(f.name);
        const auto size = fs::file_size(file, ec);
        if (ec) {
            out.problem = f.name + " is missing";
            return out;
        }
        if (size != f.size_bytes) {
            out.problem = f.name + " has " + std::to_string(size) + " bytes, its manifest " +
                          std::to_string(f.size_bytes);
            return out;
        }
        Result<std::string> sha = sha256_file(file);
        if (!sha || sha.value() != f.sha256) {
            out.problem = f.name + " does not have the SHA256 its manifest records";
            return out;
        }
    }
    out.state = CacheEntryState::valid;
    return out;
}

}  // namespace edgebench
