// EP context cache (edgebench/ep_context_cache.hpp): key derivation, manifests,
// manifest-only deletion, stale entries and lookups, in temporary directories.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "edgebench/ep_context_cache.hpp"
#include "edgebench/ort_loader.hpp"
#include "test_files.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using json = nlohmann::json;

namespace {

eb::CacheKeyFields base_key() {
    eb::CacheKeyFields k;
    k.model_sha256 = std::string(64, 'a');
    k.ort_version = "1.32.0";
    k.ep_version = "2.8.0";
    k.qnn_sdk_version = std::nullopt;
    k.qnn_backend_file_version = "2.52.0.261104115025";
    k.ep_options = {{"backend_path", "C:\\bundle\\QnnHtp.dll"}, {"htp_performance_mode", "burst"}};
    k.session_config = {{"session.disable_cpu_ep_fallback", "1"}, {"ep.context_embed_mode", "0"}};
    k.graph_optimization_level = "enable_all";
    k.device = eb::CacheDevice{"NPU", 0, 0x41304430};
    return k;
}

void write_file(const fs::path& p, const std::string& content) {
    std::ofstream(p, std::ios::binary | std::ios::trunc) << content;
}

// A cache directory under a fresh temporary directory.
struct TempCache {
    eb::test::TempModel anchor{"anchor.bin"};
    fs::path dir = anchor.path().parent_path() / "cache";
    TempCache() { fs::create_directories(dir); }
};

// An entry like the engine writes: <cache>/<key id>/{m_ctx.onnx, m_ctx_qnn.bin, manifest}.
fs::path make_entry(const fs::path& cache, const eb::CacheKeyFields& key) {
    const std::string id = eb::cache_key_id(key);
    const fs::path entry = cache / id;
    fs::create_directories(entry);
    write_file(entry / "m_ctx.onnx", "context model");
    write_file(entry / "m_ctx_qnn.bin", std::string(1000, 'b'));
    const auto m = eb::write_cache_manifest(entry, id, eb::cache_key_json(key), "m_ctx.onnx",
                                            "2026-11-26T00:00:00.000Z");
    EXPECT_TRUE(m) << (m ? "" : m.error().message);
    return entry;
}

}  // namespace

TEST(EpCacheKey, EveryFieldChangesTheKey) {
    const std::string base = eb::cache_key_id(base_key());
    EXPECT_EQ(base.size(), 64u);
    const std::vector<std::pair<std::string, std::function<void(eb::CacheKeyFields&)>>> changes = {
        {"model_sha256", [](auto& k) { k.model_sha256 = std::string(64, 'b'); }},
        {"ort_version", [](auto& k) { k.ort_version = "1.33.0"; }},
        {"ort_version null", [](auto& k) { k.ort_version.reset(); }},
        {"ep_version", [](auto& k) { k.ep_version = "2.9.0"; }},
        {"qnn_sdk_version", [](auto& k) { k.qnn_sdk_version = "2.52.0"; }},
        {"qnn_backend_file_version", [](auto& k) { k.qnn_backend_file_version = "2.53.0.1"; }},
        {"ep option value", [](auto& k) { k.ep_options["htp_performance_mode"] = "balanced"; }},
        {"ep option added", [](auto& k) { k.ep_options["enable_htp_fp16_precision"] = "1"; }},
        {"ep option removed", [](auto& k) { k.ep_options.erase("htp_performance_mode"); }},
        {"finalization mode",
         [](auto& k) { k.ep_options["htp_graph_finalization_optimization_mode"] = "1"; }},
        {"backend path", [](auto& k) { k.ep_options["backend_path"] = "D:\\QnnHtp.dll"; }},
        {"strict mode", [](auto& k) { k.session_config["session.disable_cpu_ep_fallback"] = "0"; }},
        {"embed mode", [](auto& k) { k.session_config["ep.context_embed_mode"] = "1"; }},
        {"graph optimization", [](auto& k) { k.graph_optimization_level = "enable_basic"; }},
        {"device type", [](auto& k) { k.device->device_type = "GPU"; }},
        {"vendor id", [](auto& k) { k.device->vendor_id = 0x4D4F4351; }},
        {"device id", [](auto& k) { k.device->device_id = 1; }},
        {"no device", [](auto& k) { k.device.reset(); }},
    };
    std::set<std::string> ids = {base};
    for (const auto& [what, change] : changes) {
        eb::CacheKeyFields k = base_key();
        change(k);
        const std::string id = eb::cache_key_id(k);
        EXPECT_NE(id, base) << what;
        ids.insert(id);
    }
    EXPECT_EQ(ids.size(), changes.size() + 1) << "two changes gave the same key";
}

TEST(EpCacheKey, OrderDoesNotMatter) {
    // The same options inserted in other orders, and JSON objects built in other
    // orders, serialize identically.
    eb::CacheKeyFields a = base_key(), b = base_key();
    b.ep_options.clear();
    b.ep_options.emplace("htp_performance_mode", "burst");
    b.ep_options.emplace("backend_path", "C:\\bundle\\QnnHtp.dll");
    EXPECT_EQ(eb::cache_key_id(a), eb::cache_key_id(b));
    const nlohmann::ordered_json x = {{"b", 1}, {"a", {{"d", 2}, {"c", 3}}}};
    const nlohmann::ordered_json y = {{"a", {{"c", 3}, {"d", 2}}}, {"b", 1}};
    EXPECT_EQ(eb::canonical_json(json::parse(x.dump())), eb::canonical_json(json::parse(y.dump())));
    EXPECT_EQ(eb::canonical_json(json::parse(x.dump())), R"({"a":{"c":3,"d":2},"b":1})");
}

TEST(EpCacheKey, ContextWriteEntriesAreNotPartOfTheKey) {
    const std::map<std::string, std::string> build = {
        {"session.disable_cpu_ep_fallback", "1"},
        {"ep.context_enable", "1"},
        {"ep.context_file_path", "C:\\cache\\x\\m_ctx.onnx"},
        {"ep.context_embed_mode", "0"}};
    const std::map<std::string, std::string> load = {{"session.disable_cpu_ep_fallback", "1"},
                                                     {"ep.context_enable", "0"},
                                                     {"ep.context_embed_mode", "0"}};
    EXPECT_EQ(eb::compile_session_config(build), eb::compile_session_config(load));
    EXPECT_EQ(eb::compile_session_config(load).size(), 2u);
}

TEST(EpCacheKey, CanonicalSerializationIsStable) {
    // Pinned, so that any change to the serialization (which would invalidate every
    // cache, and break the validator's recomputation) is deliberate.
    eb::CacheKeyFields k;
    k.model_sha256 = std::string(64, '0');
    k.ep_options = {{"b", "2"}, {"a", "\\x"}};
    k.session_config = {{"s", "1"}};
    k.graph_optimization_level = "enable_all";
    EXPECT_EQ(eb::canonical_json(eb::cache_key_json(k)),
              R"({"device":null,"ep_options":{"a":"\\x","b":"2"},"ep_options_sha256":")" +
                  eb::ep_options_sha256(k.ep_options) +
                  R"(","ep_version":null,"graph_optimization_level":"enable_all",)"
                  R"("model_sha256":")" +
                  std::string(64, '0') +
                  R"(","ort_version":null,"qnn_backend_file_version":null,)"
                  R"("qnn_sdk_version":null,"session_config":{"s":"1"}})");
    EXPECT_EQ(eb::ep_options_sha256({}), eb::ep_options_sha256({}));
    EXPECT_NE(eb::ep_options_sha256({}), eb::ep_options_sha256({{"a", ""}}));
}

TEST(EpCacheKey, KeyFieldsRoundTripThroughJson) {
    const eb::CacheKeyFields k = base_key();
    const auto back = eb::cache_key_from_json(eb::cache_key_json(k));
    ASSERT_TRUE(back);
    EXPECT_EQ(*back, k);
    EXPECT_FALSE(eb::cache_key_from_json(json{{"model_sha256", 1}}));
}

TEST(EpCacheEntries, ManifestListsEveryFileWithItsHash) {
    TempCache cache;
    const fs::path entry = make_entry(cache.dir, base_key());
    const auto m = eb::read_cache_manifest(entry);
    ASSERT_TRUE(m);
    EXPECT_EQ(m->key_id, eb::cache_key_id(base_key()));
    EXPECT_EQ(m->context_model, "m_ctx.onnx");
    ASSERT_EQ(m->files.size(), 2u);
    EXPECT_EQ(m->files[0].name, "m_ctx.onnx");
    EXPECT_EQ(m->files[1].name, "m_ctx_qnn.bin");
    EXPECT_EQ(m->files[1].size_bytes, 1000u);
    EXPECT_EQ(m->files[1].sha256.size(), 64u);
}

TEST(EpCacheEntries, DeletesOnlyTheFilesItsManifestLists) {
    TempCache cache;
    const fs::path entry = make_entry(cache.dir, base_key());
    write_file(entry / "user_notes.txt", "not ours");  // added after the manifest
    const auto deleted = eb::delete_cache_entry(cache.dir, entry);
    ASSERT_TRUE(deleted) << deleted.error().message;
    EXPECT_EQ(deleted.value().size(), 3u);  // the two files and the manifest
    EXPECT_FALSE(fs::exists(entry / "m_ctx.onnx"));
    EXPECT_FALSE(fs::exists(entry / eb::kEpCacheManifestName));
    EXPECT_TRUE(fs::exists(entry / "user_notes.txt"));  // never listed: kept
    EXPECT_TRUE(fs::is_directory(entry));               // not empty: kept

    // Now without a manifest: nothing is deleted.
    const auto again = eb::delete_cache_entry(cache.dir, entry);
    ASSERT_TRUE(again);
    EXPECT_TRUE(again.value().empty());
    EXPECT_TRUE(fs::exists(entry / "user_notes.txt"));
}

TEST(EpCacheEntries, AnEmptiedEntryDirectoryIsRemoved) {
    TempCache cache;
    const fs::path entry = make_entry(cache.dir, base_key());
    ASSERT_TRUE(eb::delete_cache_entry(cache.dir, entry));
    EXPECT_FALSE(fs::exists(entry));
    EXPECT_TRUE(fs::is_directory(cache.dir));
}

TEST(EpCacheEntries, RefusesToDeleteOutsideTheCacheDirectory) {
    TempCache cache;
    const fs::path entry = make_entry(cache.dir, base_key());
    const fs::path other = cache.dir.parent_path() / "other";
    fs::create_directories(other);
    // A manifest that is not a direct child of the cache directory.
    const fs::path nested = entry / "nested";
    fs::create_directories(nested);
    fs::copy_file(entry / eb::kEpCacheManifestName, nested / eb::kEpCacheManifestName);
    for (const fs::path& dir : {other, nested, cache.dir}) {
        const auto r = eb::delete_cache_entry(cache.dir, dir);
        ASSERT_FALSE(r) << eb::path_to_utf8(dir);
        EXPECT_EQ(r.error().kind, "ep_cache_delete_failed");
        EXPECT_EQ(r.error().exit_code, eb::ExitCode::output_error);
    }
    EXPECT_TRUE(fs::exists(entry / "m_ctx_qnn.bin"));
}

TEST(EpCacheEntries, ManifestsWithPathsAreIgnored) {
    TempCache cache;
    const fs::path entry = make_entry(cache.dir, base_key());
    json m = json::parse(std::ifstream(entry / eb::kEpCacheManifestName));
    m["files"].push_back({{"name", "..\\outside.txt"}, {"size_bytes", 1}, {"sha256", ""}});
    write_file(entry / eb::kEpCacheManifestName, m.dump());
    write_file(cache.dir / "outside.txt", "x");
    EXPECT_FALSE(eb::read_cache_manifest(entry));
    const auto r = eb::delete_cache_entry(cache.dir, entry);
    ASSERT_TRUE(r);
    EXPECT_TRUE(r.value().empty());
    EXPECT_TRUE(fs::exists(cache.dir / "outside.txt"));
    EXPECT_TRUE(fs::exists(entry / "m_ctx.onnx"));
}

TEST(EpCacheEntries, StaleEntriesAreThoseOfTheSameModelWithAnotherKey) {
    TempCache cache;
    const eb::CacheKeyFields current = base_key();
    eb::CacheKeyFields older = current;
    older.ep_version = "2.7.0";
    eb::CacheKeyFields other_model = current;
    other_model.model_sha256 = std::string(64, 'c');
    const fs::path keep = make_entry(cache.dir, current);
    const fs::path stale = make_entry(cache.dir, older);
    const fs::path unrelated = make_entry(cache.dir, other_model);
    const fs::path foreign = cache.dir / "something-else";
    fs::create_directories(foreign);
    write_file(foreign / "m_ctx.onnx", "user file");

    const auto deleted =
        eb::delete_stale_cache_entries(cache.dir, current.model_sha256, eb::cache_key_id(current));
    ASSERT_TRUE(deleted) << deleted.error().message;
    EXPECT_EQ(deleted.value().size(), 3u);
    for (const fs::path& p : deleted.value())
        EXPECT_EQ(p.parent_path(), stale) << eb::path_to_utf8(p);
    EXPECT_FALSE(fs::exists(stale));
    EXPECT_TRUE(fs::exists(keep / "m_ctx_qnn.bin"));
    EXPECT_TRUE(fs::exists(unrelated / "m_ctx_qnn.bin"));
    EXPECT_TRUE(fs::exists(foreign / "m_ctx.onnx"));
}

TEST(EpCacheEntries, LookupTellsMissingValidCorruptAndForeign) {
    TempCache cache;
    const eb::CacheKeyFields key = base_key();
    const std::string id = eb::cache_key_id(key);
    const json fields = eb::cache_key_json(key);
    EXPECT_EQ(eb::lookup_cache_entry(cache.dir, id, fields).state, eb::CacheEntryState::missing);

    const fs::path entry = make_entry(cache.dir, key);
    const eb::CacheLookup valid = eb::lookup_cache_entry(cache.dir, id, fields);
    EXPECT_EQ(valid.state, eb::CacheEntryState::valid) << valid.problem;
    EXPECT_EQ(valid.entry_dir, entry);

    write_file(entry / "m_ctx_qnn.bin", std::string(999, 'b'));  // truncated
    eb::CacheLookup corrupt = eb::lookup_cache_entry(cache.dir, id, fields);
    EXPECT_EQ(corrupt.state, eb::CacheEntryState::corrupt);
    EXPECT_NE(corrupt.problem.find("m_ctx_qnn.bin"), std::string::npos) << corrupt.problem;

    write_file(entry / "m_ctx_qnn.bin", std::string(999, 'b') + "c");  // same size, changed
    corrupt = eb::lookup_cache_entry(cache.dir, id, fields);
    EXPECT_EQ(corrupt.state, eb::CacheEntryState::corrupt);
    EXPECT_NE(corrupt.problem.find("SHA256"), std::string::npos) << corrupt.problem;

    fs::remove(entry / "m_ctx_qnn.bin");
    EXPECT_EQ(eb::lookup_cache_entry(cache.dir, id, fields).state, eb::CacheEntryState::corrupt);

    fs::remove(entry / eb::kEpCacheManifestName);
    const eb::CacheLookup foreign = eb::lookup_cache_entry(cache.dir, id, fields);
    EXPECT_EQ(foreign.state, eb::CacheEntryState::foreign);
    EXPECT_FALSE(foreign.problem.empty());
}

TEST(EpCacheEntries, LookupComparesTheKeyFields) {
    TempCache cache;
    const eb::CacheKeyFields key = base_key();
    make_entry(cache.dir, key);
    eb::CacheKeyFields other = key;
    other.graph_optimization_level = "disable_all";
    // Same directory name, different fields (an edited manifest): not usable.
    EXPECT_EQ(
        eb::lookup_cache_entry(cache.dir, eb::cache_key_id(key), eb::cache_key_json(other)).state,
        eb::CacheEntryState::corrupt);
}

TEST(EpCacheDirectory, MustBeWritable) {
    TempCache cache;
    const auto ok = eb::prepare_ep_cache_dir(cache.dir / "new" / "nested");
    ASSERT_TRUE(ok) << ok.error().message;
    EXPECT_TRUE(fs::is_directory(ok.value()));
    EXPECT_TRUE(ok.value().is_absolute());
    EXPECT_TRUE(fs::is_empty(ok.value()));  // the write test leaves nothing behind

    const fs::path file = cache.dir / "a_file";
    write_file(file, "x");
    const auto bad = eb::prepare_ep_cache_dir(file / "cache");
    ASSERT_FALSE(bad);
    EXPECT_EQ(bad.error().kind, "ep_cache_dir_not_writable");
    EXPECT_EQ(bad.error().exit_code, eb::ExitCode::output_error);
}

TEST(EpCacheDirectory, DefaultIsUnderLocalAppData) {
    const auto dir = eb::default_ep_cache_dir();
    ASSERT_TRUE(dir) << dir.error().message;
    EXPECT_TRUE(dir.value().is_absolute());
    EXPECT_EQ(dir.value().filename(), "qnn-context");
    EXPECT_EQ(dir.value().parent_path().filename(), "EdgeBench");
}

TEST(EpCacheEntries, ContextModelIsNamedAfterTheModel) {
    EXPECT_EQ(eb::context_model_name("C:/m/mobilenetv2_int8_b1.onnx"),
              "mobilenetv2_int8_b1_ctx.onnx");
}
