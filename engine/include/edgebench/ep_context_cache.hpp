// The EP context cache: precompiled QNN contexts on disk, keyed by everything
// they depend on.
//
// Layout: one subdirectory of the cache directory per cache key, named by the
// key id, holding the EP context model (<model stem>_ctx.onnx), its context
// binary (<model stem>_ctx_qnn.bin; embed mode 0, see kEpContextEmbedMode) and a
// manifest the engine writes (kEpCacheManifestName: key fields, key id, creation
// time, and every file with its size and SHA256).
//
// The key id is the SHA256 of the key fields serialized canonically (JSON, keys
// sorted, no whitespace, UTF-8): model SHA256, ONNX Runtime version, QNN EP
// version, QNN SDK version (when known) and the HTP backend file version, every
// EP option, every session config entry that affects compilation, the graph
// optimization level, and the device (type, vendor id, device id). A change to
// any of them gives a new key, so a cache can only ever be used with exactly
// the configuration it was built with.
//
// Deletion is restricted: the engine deletes only files listed in a manifest it
// wrote (plus that manifest), and only in direct subdirectories of the cache
// directory. A directory without a valid manifest is never touched.
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "edgebench/result.hpp"

namespace edgebench {

inline constexpr std::string_view kEpCacheManifestName = "edgebench-cache.json";
inline constexpr std::string_view kEpCacheManifestVersion = "1.0.0";

/// ep.context_embed_mode for every context the engine builds: 0, the context
/// binary in its own file next to the context model. It is the mode whose cached
/// initialization and output equality were measured (docs/feasibility.md), and
/// cached sessions load the binary from its file instead of parsing it out of a
/// protobuf string (a short comparison on MobileNetV2 INT8: median cached
/// initialization 336 ms with mode 0, 368 ms with mode 1).
inline constexpr std::string_view kEpContextEmbedMode = "0";

/// Session config entries that never go into the cache key: whether the session
/// writes a context and where (they differ between the build and the load of the
/// same context, and do not change what is compiled).
inline constexpr std::string_view kEpContextEnableKey = "ep.context_enable";
inline constexpr std::string_view kEpContextFilePathKey = "ep.context_file_path";
inline constexpr std::string_view kEpContextEmbedModeKey = "ep.context_embed_mode";

struct CacheDevice {
    std::string device_type;  // "NPU"
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    bool operator==(const CacheDevice&) const = default;
};

/// Everything an EP context depends on (results schema: config.ep_context_cache.key).
struct CacheKeyFields {
    std::string model_sha256;
    std::optional<std::string> ort_version;
    std::optional<std::string> ep_version;       // QNN EP plugin
    std::optional<std::string> qnn_sdk_version;  // reported by the backend; null until read
    std::optional<std::string> qnn_backend_file_version;  // QnnHtp.dll ProductVersion
    std::map<std::string, std::string> ep_options;        // every EP option passed
    std::map<std::string, std::string> session_config;    // compile_session_config()
    std::string graph_optimization_level;
    std::optional<CacheDevice> device;
    bool operator==(const CacheKeyFields&) const = default;
};

/// The session config entries that affect compilation: all of `session_config`
/// except kEpContextEnableKey and kEpContextFilePathKey.
std::map<std::string, std::string> compile_session_config(
    const std::map<std::string, std::string>& session_config);

/// The key fields as a JSON object (keys sorted; also config.ep_context_cache.key).
nlohmann::json cache_key_json(const CacheKeyFields& key);
/// The key fields read back from cache_key_json(); nullopt if malformed.
std::optional<CacheKeyFields> cache_key_from_json(const nlohmann::json& j);
/// `j` serialized with sorted keys, no whitespace, UTF-8 unescaped (Python:
/// json.dumps(j, sort_keys=True, separators=(",", ":"), ensure_ascii=False)).
std::string canonical_json(const nlohmann::json& j);
/// SHA256 (hex) of canonical_json(cache_key_json(key)).
std::string cache_key_id(const CacheKeyFields& key);
/// SHA256 (hex) of canonical_json(ep_options) (key field ep_options_sha256).
std::string ep_options_sha256(const std::map<std::string, std::string>& ep_options);

struct CacheFile {
    std::string name;  // file name inside the entry directory
    std::uint64_t size_bytes = 0;
    std::string sha256;
    bool operator==(const CacheFile&) const = default;
};

/// The manifest of one cache entry.
struct CacheManifest {
    std::string key_id;
    nlohmann::json key_fields;  // cache_key_json()
    std::string created_utc;
    std::string context_model;     // file name of the EP context model
    std::vector<CacheFile> files;  // every file of the entry except the manifest
};

/// %LOCALAPPDATA%\EdgeBench\qnn-context (absolute). Error if LOCALAPPDATA is unset
/// (exit code 6, ep_cache_dir_not_writable).
Result<std::filesystem::path> default_ep_cache_dir();

/// Creates `dir` if needed and checks that a file can be created in it. Returns
/// the absolute path. Errors: exit code 6, kind ep_cache_dir_not_writable.
Result<std::filesystem::path> prepare_ep_cache_dir(const std::filesystem::path& dir);

/// The manifest in `entry_dir`, if there is one the engine wrote: right version,
/// a key id, and only plain file names. nullopt otherwise.
std::optional<CacheManifest> read_cache_manifest(const std::filesystem::path& entry_dir);

/// Lists every file in `entry_dir` (except the manifest), hashes it, and writes
/// the manifest. The directory must only hold files the engine created.
Result<CacheManifest> write_cache_manifest(const std::filesystem::path& entry_dir,
                                           const std::string& key_id,
                                           const nlohmann::json& key_fields,
                                           const std::string& context_model,
                                           const std::string& created_utc);

/// Deletes the files listed in `entry_dir`'s manifest and the manifest, then the
/// directory if it is empty. `entry_dir` must be a direct subdirectory of
/// `cache_dir`. Without a valid manifest nothing is deleted. Returns the paths
/// deleted. Errors: exit code 6, kind ep_cache_delete_failed.
Result<std::vector<std::filesystem::path>> delete_cache_entry(
    const std::filesystem::path& cache_dir, const std::filesystem::path& entry_dir);

/// Deletes every entry of `cache_dir` (a subdirectory named by its manifest's key
/// id) whose key fields have model SHA256 `model_sha256` and whose key id is not
/// `keep_key_id`: caches of the same model whose key changed. Returns the paths
/// deleted.
Result<std::vector<std::filesystem::path>> delete_stale_cache_entries(
    const std::filesystem::path& cache_dir, std::string_view model_sha256,
    std::string_view keep_key_id);

enum class CacheEntryState {
    missing,  // no directory for the key
    valid,    // manifest for this key; every file present with its size and SHA256
    corrupt,  // manifest for this key, but a file is missing, truncated or changed
    foreign,  // a directory for the key without a manifest the engine wrote
};

struct CacheLookup {
    CacheEntryState state = CacheEntryState::missing;
    std::filesystem::path entry_dir;
    std::optional<CacheManifest> manifest;
    std::string problem;  // what is wrong (corrupt, foreign)
};

/// The entry for `key_id` in `cache_dir`, checked against `key_fields` and its
/// own manifest.
CacheLookup lookup_cache_entry(const std::filesystem::path& cache_dir, const std::string& key_id,
                               const nlohmann::json& key_fields);

/// "<model stem>_ctx.onnx": the context model's file name for `model`.
std::string context_model_name(const std::filesystem::path& model);

}  // namespace edgebench
