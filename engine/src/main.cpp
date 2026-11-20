// edgebench: CPU vs NPU inference benchmarking for Windows on Snapdragon.

#include <cstdio>
#include <string_view>

#include "edgebench/version.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 2;

void print_usage(std::FILE* out) {
    std::fputs(
        "Usage: edgebench [options]\n"
        "\n"
        "Options:\n"
        "  --version   Print the EdgeBench and ONNX Runtime versions\n"
        "  -h, --help  Show this help\n",
        out);
}

void print_version() {
    const std::string ort = edgebench::onnxruntime_version();
    std::printf("edgebench %.*s\n", static_cast<int>(edgebench::version().size()),
                edgebench::version().data());
    std::printf("onnxruntime %s\n", ort.empty() ? "(unknown)" : ort.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        print_usage(stderr);
        return kExitUsage;
    }
    const std::string_view arg = argv[1];
    if (arg == "--version") {
        print_version();
        return kExitOk;
    }
    if (arg == "--help" || arg == "-h") {
        print_usage(stdout);
        return kExitOk;
    }
    std::fprintf(stderr, "edgebench: unknown option '%s'\n\n", argv[1]);
    print_usage(stderr);
    return kExitUsage;
}
