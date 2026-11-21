// In-process tests of the ONNX Runtime loader against the bundled DLLs next to
// this test executable.

#include "edgebench/ort_loader.hpp"

#include <onnxruntime_cxx_api.h>

#include <gtest/gtest.h>

#include <fstream>

#include "process_util.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace fs = std::filesystem;
namespace eb = edgebench;

TEST(OrtLoader, LoadsTheBundledLibrary) {
    const eb::OrtLoadResult r = eb::load_onnxruntime();
    ASSERT_TRUE(r.ok()) << eb::to_string(r.error_kind) << ": " << r.message;
    EXPECT_EQ(r.error_kind, eb::OrtLoadError::none);
    EXPECT_TRUE(r.message.empty());
    EXPECT_EQ(r.load_check(), "full_path_verified");
    EXPECT_EQ(r.ort_version, eb::expected_onnxruntime_version());
    EXPECT_EQ(r.api_version, ORT_API_VERSION);
    EXPECT_EQ(r.dll_path, eb::executable_dir() / "onnxruntime.dll");

    // The C++ API is usable and talks to the same library.
    EXPECT_EQ(Ort::GetVersionString(), r.ort_version);
    EXPECT_TRUE(
        eb::is_same_path(eb::loaded_module_path(L"onnxruntime.dll").native(), r.dll_path.native()));
}

TEST(OrtLoader, IsIdempotent) {
    const eb::OrtLoadResult a = eb::load_onnxruntime();
    const eb::OrtLoadResult b = eb::load_onnxruntime();
    ASSERT_TRUE(a.ok());
    EXPECT_TRUE(b.ok());
    EXPECT_EQ(a.dll_path, b.dll_path);
}

TEST(OrtLoader, ListsLoadedModulesInsideTheBundle) {
    const eb::OrtLoadResult r = eb::load_onnxruntime();
    ASSERT_TRUE(r.ok()) << r.message;
    ASSERT_FALSE(r.loaded_modules.empty());
    bool found = false;
    for (const eb::LoadedModule& m : r.loaded_modules) {
        EXPECT_TRUE(m.path.is_absolute()) << m.path;
        EXPECT_EQ(eb::path_to_utf8(m.path.filename()), m.name);
        found |= m.name == "onnxruntime.dll";
    }
    EXPECT_TRUE(found);
    EXPECT_TRUE(eb::find_modules_outside(eb::list_ort_modules(), eb::executable_dir()).empty());
}

TEST(OrtLoader, FindsModulesOutsideAnotherDirectory) {
    ASSERT_TRUE(eb::load_onnxruntime().ok());
    const auto modules = eb::list_ort_modules();
    const auto outside = eb::find_modules_outside(modules, eb::executable_dir() / "elsewhere");
    EXPECT_EQ(outside.size(), modules.size());
}

TEST(ResolvePath, ExpandsShortNames) {
    eb::test::TempDir tmp;
    const fs::path long_dir = tmp.path() / "a directory with a long name";
    fs::create_directory(long_dir);
    std::ofstream(long_dir / "some_module.dll") << "x";

    std::wstring short_path(MAX_PATH, L'\0');
    const DWORD n = GetShortPathNameW(long_dir.c_str(), short_path.data(), MAX_PATH);
    ASSERT_GT(n, 0u);
    short_path.resize(n);
    if (eb::is_same_path(short_path, long_dir.native()))
        GTEST_SKIP() << "8.3 short names are disabled on this volume";

    // The short spelling is not lexically equal to the long one ...
    const fs::path module = fs::path(short_path) / "some_module.dll";
    EXPECT_FALSE(eb::is_path_within(module.native(), long_dir.native()));
    // ... but the resolved paths are, and the bundle check uses them.
    EXPECT_TRUE(eb::is_same_path(eb::resolve_path(module).native(),
                                 eb::resolve_path(long_dir / "some_module.dll").native()));
    const std::vector<eb::LoadedModule> modules = {{"some_module.dll", module}};
    EXPECT_TRUE(eb::find_modules_outside(modules, long_dir).empty());
    EXPECT_TRUE(eb::find_modules_outside({{"some_module.dll", long_dir / "some_module.dll"}},
                                         fs::path(short_path))
                    .empty());
}

TEST(ResolvePath, StripsExtendedPrefixAndKeepsMissingPaths) {
    const fs::path exe_dir = eb::executable_dir();
    const fs::path resolved = eb::resolve_path(exe_dir);
    EXPECT_FALSE(resolved.native().starts_with(L"\\\\?\\")) << resolved;
    EXPECT_TRUE(eb::is_same_path(resolved.native(), exe_dir.native()));

    const fs::path missing = exe_dir / "does-not-exist" / "x.dll";
    EXPECT_TRUE(eb::is_same_path(eb::resolve_path(missing).native(), missing.native()));
}
