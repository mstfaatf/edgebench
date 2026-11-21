#include "edgebench/module_paths.hpp"

#include <gtest/gtest.h>

#include "edgebench/ort_loader.hpp"

namespace eb = edgebench;

// --------------------------------------------------------------------------- versions

TEST(ParseVersion, AcceptsMajorMinorPatch) {
    const auto v = eb::parse_version("1.32.0");
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, (eb::Version{1, 32, 0}));
    EXPECT_EQ(eb::parse_version("10.0.26100"), (eb::Version{10, 0, 26100}));
}

TEST(ParseVersion, RejectsOtherForms) {
    for (const char* bad : {"", "1", "1.32", "1.32.0.1", "1.32.0-dev", "v1.32.0", "1..0", "1.32.",
                            ".32.0", "1.-2.0", "1.32.x", " 1.32.0", "99999999999.0.0"}) {
        EXPECT_FALSE(eb::parse_version(bad).has_value()) << "'" << bad << "'";
    }
}

TEST(VersionMatches, RequiresSameMajorMinorPatch) {
    EXPECT_TRUE(eb::version_matches("1.32.0", "1.32.0"));
    EXPECT_TRUE(eb::version_matches("01.32.00", "1.32.0"));  // numeric comparison
    EXPECT_FALSE(eb::version_matches("1.17.1", "1.32.0"));
    EXPECT_FALSE(eb::version_matches("1.32.1", "1.32.0"));
    EXPECT_FALSE(eb::version_matches("2.32.0", "1.32.0"));
    EXPECT_FALSE(eb::version_matches("1.32.0-dev", "1.32.0"));
    EXPECT_FALSE(eb::version_matches("", "1.32.0"));
    EXPECT_FALSE(eb::version_matches("1.32.0", ""));
}

// --------------------------------------------------------------------------- paths

TEST(NormalizePath, CaseAndSlashes) {
    EXPECT_EQ(eb::normalize_path(L"C:/Dev/EdgeBench/Build"), L"c:\\dev\\edgebench\\build");
    EXPECT_EQ(eb::normalize_path(L"C:\\dev//edgebench\\\\bin"), L"c:\\dev\\edgebench\\bin");
}

TEST(NormalizePath, ExtendedLengthPrefixes) {
    EXPECT_EQ(eb::normalize_path(L"\\\\?\\C:\\Dev\\bin"), L"c:\\dev\\bin");
    EXPECT_EQ(eb::normalize_path(L"\\\\?\\UNC\\Server\\Share\\bin"), L"\\\\server\\share\\bin");
    EXPECT_EQ(eb::normalize_path(L"\\\\?\\unc\\server\\share"), L"\\\\server\\share");
    EXPECT_EQ(eb::normalize_path(L"//server/share/bin"), L"\\\\server\\share\\bin");
}

TEST(NormalizePath, TrailingSeparators) {
    EXPECT_EQ(eb::normalize_path(L"C:\\dev\\bin\\"), L"c:\\dev\\bin");
    EXPECT_EQ(eb::normalize_path(L"C:\\dev\\bin\\\\//"), L"c:\\dev\\bin");
    EXPECT_EQ(eb::normalize_path(L"C:\\"), L"c:");
    EXPECT_EQ(eb::normalize_path(L"C:"), L"c:");
}

TEST(NormalizePath, DotSegments) {
    EXPECT_EQ(eb::normalize_path(L"C:\\dev\\.\\x\\..\\bin"), L"c:\\dev\\bin");
    EXPECT_EQ(eb::normalize_path(L"C:\\..\\dev"), L"c:\\dev");  // cannot climb above the root
    EXPECT_EQ(eb::normalize_path(L"a\\..\\..\\b"), L"..\\b");   // relative paths may
    EXPECT_EQ(eb::normalize_path(L""), L"");
}

TEST(NormalizePath, KeepsNonAsciiCharacters) {
    EXPECT_EQ(eb::normalize_path(L"C:\\Users\\Jos\u00C9"), L"c:\\users\\jos\u00C9");
}

TEST(IsSamePath, IgnoresSpellingDifferences) {
    EXPECT_TRUE(
        eb::is_same_path(L"C:\\Dev\\bin\\onnxruntime.dll", L"\\\\?\\c:/dev/BIN//ONNXRUNTIME.DLL"));
    EXPECT_FALSE(eb::is_same_path(L"C:\\dev\\bin\\onnxruntime.dll",
                                  L"C:\\Windows\\System32\\onnxruntime.dll"));
    EXPECT_FALSE(eb::is_same_path(L"", L""));
}

TEST(IsPathWithin, DirectoryAndSubdirectories) {
    const wchar_t* dir = L"C:\\dev\\edgebench\\bin\\Release";
    EXPECT_TRUE(eb::is_path_within(L"C:\\dev\\edgebench\\bin\\Release\\onnxruntime.dll", dir));
    EXPECT_TRUE(eb::is_path_within(L"c:/DEV/edgebench/bin/release/qnn/QnnHtp.dll", dir));
    EXPECT_TRUE(eb::is_path_within(L"\\\\?\\C:\\dev\\edgebench\\bin\\Release\\x.dll",
                                   L"C:\\dev\\edgebench\\bin\\Release\\"));
    EXPECT_TRUE(eb::is_path_within(dir, dir));
}

TEST(IsPathWithin, RejectsSiblingsAndOtherDirectories) {
    const wchar_t* dir = L"C:\\dev\\edgebench\\bin\\Release";
    // Same prefix, different directory name.
    EXPECT_FALSE(eb::is_path_within(L"C:\\dev\\edgebench\\bin\\Release2\\onnxruntime.dll", dir));
    EXPECT_FALSE(eb::is_path_within(L"C:\\Windows\\System32\\onnxruntime.dll", dir));
    EXPECT_FALSE(eb::is_path_within(L"C:\\dev\\edgebench\\bin", dir));
    EXPECT_FALSE(eb::is_path_within(L"C:\\dev\\edgebench\\bin\\Release\\..\\x.dll", dir));
    EXPECT_FALSE(eb::is_path_within(L"", dir));
    EXPECT_FALSE(eb::is_path_within(L"C:\\x.dll", L""));
}

TEST(IsPathWithin, DriveRoot) {
    EXPECT_TRUE(eb::is_path_within(L"C:\\x\\y.dll", L"C:\\"));
    EXPECT_FALSE(eb::is_path_within(L"D:\\x\\y.dll", L"C:\\"));
}

// --------------------------------------------------------------------------- module names

TEST(IsOrtOrQnnModule, MatchesOrtAndQnnDlls) {
    for (const wchar_t* name :
         {L"onnxruntime.dll", L"ONNXRUNTIME.DLL", L"onnxruntime_providers_shared.dll",
          L"onnxruntime_providers_qnn.dll", L"QnnHtp.dll", L"QnnHtpPrepare.dll",
          L"QnnHtpV73Stub.dll", L"QnnSystem.dll", L"qnnsaver.dll"}) {
        EXPECT_TRUE(eb::is_ort_or_qnn_module(name)) << name;
    }
}

TEST(IsOrtOrQnnModule, RejectsOthers) {
    for (const wchar_t* name :
         {L"kernel32.dll", L"edgebench.exe", L"onnxruntime.exe", L"libQnnHtpV73Skel.so",
          L"onnxruntime_pybind11_state.pyd", L"myqnn.dll", L"Genie.dll", L"onnxruntime.dll.bak",
          L"C:\\bin\\onnxruntime.dll", L""}) {
        EXPECT_FALSE(eb::is_ort_or_qnn_module(name)) << name;
    }
}

// --------------------------------------------------------------------------- bundle check

TEST(FindModulesOutside, ReportsOnlyModulesOutsideTheBundle) {
    // Paths that do not exist are compared as given (after normalization).
    const std::vector<eb::LoadedModule> modules = {
        {"onnxruntime.dll", L"C:\\eb-test-nonexistent\\Bin\\onnxruntime.dll"},
        {"QnnHtp.dll", L"c:/eb-test-nonexistent/bin/QnnHtp.dll"},
        {"onnxruntime.dll", L"C:\\Windows\\System32\\onnxruntime.dll"},
        {"QnnSystem.dll", L"C:\\eb-test-nonexistent\\bin2\\QnnSystem.dll"},
    };
    const auto outside = eb::find_modules_outside(modules, L"C:\\eb-test-nonexistent\\bin\\");
    ASSERT_EQ(outside.size(), 2u);
    EXPECT_EQ(outside[0].path, modules[2].path);
    EXPECT_EQ(outside[1].path, modules[3].path);
}

// --------------------------------------------------------------------------- result

TEST(OrtLoadResult, LoadCheckNames) {
    eb::OrtLoadResult r;
    r.status = eb::OrtLoadStatus::ok;
    EXPECT_EQ(r.load_check(), "full_path_verified");
    r.status = eb::OrtLoadStatus::failed;
    r.error_kind = eb::OrtLoadError::version_mismatch;
    EXPECT_EQ(r.load_check(), "version_mismatch");
}

TEST(OrtLoadError, StableNames) {
    EXPECT_EQ(eb::to_string(eb::OrtLoadError::none), "none");
    EXPECT_EQ(eb::to_string(eb::OrtLoadError::dll_not_found), "dll_not_found");
    EXPECT_EQ(eb::to_string(eb::OrtLoadError::load_failed), "load_failed");
    EXPECT_EQ(eb::to_string(eb::OrtLoadError::entry_point_missing), "entry_point_missing");
    EXPECT_EQ(eb::to_string(eb::OrtLoadError::version_mismatch), "version_mismatch");
    EXPECT_EQ(eb::to_string(eb::OrtLoadError::api_version_unsupported), "api_version_unsupported");
    EXPECT_EQ(eb::to_string(eb::OrtLoadError::foreign_module_loaded), "foreign_module_loaded");
}
