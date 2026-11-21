// Test helper: optionally loads a DLL first (argument 1, a full path), then runs
// edgebench::load_onnxruntime() and prints the result. Used to check that an
// onnxruntime.dll loaded from somewhere else before the loader is detected.
//
// Output: "load_check: <value>" and "message: <text>". Exit codes: 0 loaded,
// 4 loader failed, 10 the preload itself failed.

#include <cstdio>

#include "edgebench/ort_loader.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

int wmain(int argc, wchar_t** argv) {
    if (argc == 2) {
        if (!LoadLibraryExW(argv[1], nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)) {
            std::printf("preload failed: error %lu\n", GetLastError());
            return 10;
        }
    }
    const edgebench::OrtLoadResult r = edgebench::load_onnxruntime();
    const std::string_view check = r.load_check();
    std::printf("load_check: %.*s\nmessage: %s\n", static_cast<int>(check.size()), check.data(),
                r.message.c_str());
    return r.ok() ? 0 : 4;
}
