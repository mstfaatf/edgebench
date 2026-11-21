// edgebench: CPU vs NPU inference benchmarking for Windows on Snapdragon.
//
// ONNX Runtime is delay-loaded and initialized manually: the first thing main()
// does (after --help) is load the onnxruntime.dll next to the executable by full
// path and verify it, so a missing or wrong DLL is reported instead of crashing.
//
// Exit codes: 0 ok, 2 usage error, 4 ONNX Runtime could not be loaded.

#include <cstdio>
#include <string_view>

#include "edgebench/ort_loader.hpp"
#include "edgebench/version.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 2;
constexpr int kExitOrtLoadFailed = 4;

void print_usage(std::FILE* out) {
    std::fputs(
        "Usage: edgebench [options]\n"
        "\n"
        "Loads the bundled ONNX Runtime and prints its version and location.\n"
        "\n"
        "Options:\n"
        "  --version   Same as no options\n"
        "  -h, --help  Show this help\n"
        "\n"
        "Exit codes: 0 ok, 2 usage error, 4 ONNX Runtime could not be loaded.\n",
        out);
}

int load_and_report() {
    const std::string_view v = edgebench::version();
    std::printf("edgebench %.*s\n", static_cast<int>(v.size()), v.data());
    std::fflush(stdout);

    const edgebench::OrtLoadResult r = edgebench::load_onnxruntime();
    if (!r.ok()) {
        const std::string_view kind = edgebench::to_string(r.error_kind);
        std::fprintf(stderr, "error: cannot load ONNX Runtime (%.*s)\n  %s\n",
                     static_cast<int>(kind.size()), kind.data(), r.message.c_str());
        return kExitOrtLoadFailed;
    }
    const std::string_view check = r.load_check();
    std::printf("onnxruntime %s\n  C API version: %d\n  library:       %s\n  load check:    %.*s\n",
                r.ort_version.c_str(), r.api_version, edgebench::path_to_utf8(r.dll_path).c_str(),
                static_cast<int>(check.size()), check.data());
    return kExitOk;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        print_usage(stderr);
        return kExitUsage;
    }
    if (argc == 2) {
        const std::string_view arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            print_usage(stdout);
            return kExitOk;
        }
        if (arg != "--version") {
            std::fprintf(stderr, "edgebench: unknown option '%s'\n\n", argv[1]);
            print_usage(stderr);
            return kExitUsage;
        }
    }
    return load_and_report();
}
