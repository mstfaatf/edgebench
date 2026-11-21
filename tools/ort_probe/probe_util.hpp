// Helpers for ort_probe that do not depend on ONNX Runtime: command-line
// parsing, raw tensor files and the reference index, output comparison,
// latency statistics, and a minimal JSON writer. Unit-tested in
// probe_util_test.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace probe {

// --------------------------------------------------------------------------- options

enum class Target { kCpu, kNpu };
enum class RunMode { kRun, kIoBinding };
enum class QnnProfiling { kOff, kBasic, kDetailed };

const char* ToString(Target t);
const char* ToString(RunMode m);
const char* ToString(QnnProfiling p);

struct Options {
    std::filesystem::path model;
    Target target = Target::kNpu;
    bool strict = true;  // session.disable_cpu_ep_fallback=1 (NPU only)
    bool fp16 = false;   // enable_htp_fp16_precision=1 (NPU only)
    int threads = 4;     // intra_op_num_threads for every session
    RunMode run_mode = RunMode::kRun;

    bool ep_context = false;             // generate and use an EP context model (NPU only)
    std::filesystem::path context_file;  // required with ep_context
    bool reuse_context = false;          // load an existing context file, do not regenerate

    std::filesystem::path reference_dir;  // index.tsv + raw tensors from make_reference.py
    std::string reference_config;         // outputs to compare with, e.g. "npu_s8w"
    double tolerance = -1.0;              // max abs diff allowed; < 0 means the target default
    std::string fill = "pattern";         // input when no reference dir: "pattern" or a number
    bool compare_cpu = false;             // also run the inputs on the CPU EP and compare
    std::filesystem::path save_outputs;   // directory for this run's outputs (raw float32)

    int init_reps = 1;  // session creations to time (cold, and cached with ep_context)
    int warmup = 20;
    int runs = 0;  // timed runs on the first input; 0 = no latency measurement

    QnnProfiling qnn_profiling = QnnProfiling::kOff;
    std::filesystem::path qnn_profile_file;  // required unless profiling is off

    std::filesystem::path json;  // write a JSON summary here
    bool verbose_log = false;    // ORT verbose logging to stderr
    bool help = false;
};

struct ParseResult {
    Options options;
    std::string error;  // empty on success
};

// Parses the arguments after argv[0]. Never throws; reports problems in .error.
ParseResult ParseArgs(const std::vector<std::string>& args);
std::string Usage(std::string_view program);

// Default max-abs-diff tolerance for comparisons against Python outputs: CPU
// kernels match to float rounding; the NPU is bit-exact run to run (repeat
// variation of Python NPU sessions is 0), so the same model must match exactly.
double DefaultTolerance(Target t);

// --------------------------------------------------------------------------- tensors

struct TensorRecord {
    std::string kind;    // "input" or "output"
    std::string config;  // input format (e.g. "normalized") or output config (e.g. "npu_s8w")
    int image = 0;       // position in the image list
    std::string image_id;
    std::filesystem::path file;  // relative to the index directory
    std::string dtype;           // "float32"
    std::vector<int64_t> shape;

    size_t ElementCount() const;
};

// Reads index.tsv: a header line, then one tab-separated record per line
// (kind, config, image, image_id, file, dtype, comma-separated shape).
// Throws std::runtime_error on malformed input.
std::vector<TensorRecord> ReadIndex(const std::filesystem::path& index_file);
std::vector<TensorRecord> SelectRecords(const std::vector<TensorRecord>& records,
                                        std::string_view kind, std::string_view config);

// Raw little-endian float32 files. ReadFloat32 throws if the size does not match.
std::vector<float> ReadFloat32(const std::filesystem::path& file, size_t expected_count);
void WriteFloat32(const std::filesystem::path& file, std::span<const float> data);

// Deterministic, non-constant fill: values in [-1, 1) that differ per element.
std::vector<float> PatternFill(size_t count);

// --------------------------------------------------------------------------- comparison

struct Comparison {
    double max_abs_diff = 0.0;
    size_t argmax_actual = 0;
    size_t argmax_expected = 0;
    bool top1_match = false;
    bool within_tolerance = false;
    bool ok() const { return top1_match && within_tolerance; }
};

size_t ArgMax(std::span<const float> values);
// numpy.allclose semantics: |actual - expected| <= atol + rtol * |expected| for every
// element (NaN never matches). Throws std::invalid_argument if the sizes differ.
bool AllClose(std::span<const float> actual, std::span<const float> expected, double rtol,
              double atol);
// Throws std::invalid_argument if the sizes differ or the inputs are empty.
Comparison Compare(std::span<const float> actual, std::span<const float> expected,
                   double tolerance);

// --------------------------------------------------------------------------- statistics

struct LatencyStats {
    size_t count = 0;
    double mean = 0, p50 = 0, p95 = 0, min = 0, max = 0, stddev = 0;
};

// Nearest-rank percentiles (as in docs/results-schema.md); sample stddev.
// Throws std::invalid_argument on an empty input.
LatencyStats ComputeStats(std::vector<double> samples);
double Median(std::vector<double> values);

// --------------------------------------------------------------------------- JSON

std::string JsonEscape(std::string_view s);

// Streaming JSON writer that inserts commas; keys are only valid inside objects.
class JsonWriter {
public:
    JsonWriter& BeginObject();
    JsonWriter& EndObject();
    JsonWriter& BeginArray();
    JsonWriter& EndArray();
    JsonWriter& Key(std::string_view key);
    JsonWriter& String(std::string_view value);
    JsonWriter& Number(double value);
    JsonWriter& Int(int64_t value);
    JsonWriter& Bool(bool value);
    JsonWriter& Null();
    const std::string& str() const { return out_; }

private:
    void Separator();
    std::string out_;
    std::vector<bool> first_;  // per open container: no element written yet
    bool after_key_ = false;
};

}  // namespace probe
