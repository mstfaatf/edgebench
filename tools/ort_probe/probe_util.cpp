#include "probe_util.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace probe {

const char* ToString(Target t) {
    return t == Target::kCpu ? "cpu" : "npu";
}

const char* ToString(RunMode m) {
    return m == RunMode::kRun ? "run" : "iobinding";
}

const char* ToString(QnnProfiling p) {
    switch (p) {
        case QnnProfiling::kOff:
            return "off";
        case QnnProfiling::kBasic:
            return "basic";
        case QnnProfiling::kDetailed:
            return "detailed";
    }
    return "off";
}

double DefaultTolerance(Target t) {
    return t == Target::kCpu ? 1e-5 : 0.0;
}

// --------------------------------------------------------------------------- options

namespace {

bool ParseOnOff(std::string_view v, bool& out) {
    if (v == "on" || v == "1" || v == "true") {
        out = true;
        return true;
    }
    if (v == "off" || v == "0" || v == "false") {
        out = false;
        return true;
    }
    return false;
}

bool ParseInt(std::string_view v, int& out) {
    const char* end = v.data() + v.size();
    auto [ptr, ec] = std::from_chars(v.data(), end, out);
    return ec == std::errc() && ptr == end;
}

bool ParseDouble(std::string_view v, double& out) {
    // std::from_chars for double is available in MSVC; keep a strict full-match check.
    const char* end = v.data() + v.size();
    auto [ptr, ec] = std::from_chars(v.data(), end, out);
    return ec == std::errc() && ptr == end;
}

}  // namespace

ParseResult ParseArgs(const std::vector<std::string>& args) {
    ParseResult r;
    Options& o = r.options;
    auto fail = [&r](std::string msg) {
        r.error = std::move(msg);
        return r;
    };

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "-h" || a == "--help") {
            o.help = true;
            return r;
        }
        if (a == "--verbose-log") {
            o.verbose_log = true;
            continue;
        }
        if (a == "--reuse-context") {
            o.reuse_context = true;
            continue;
        }
        if (a == "--compare-cpu") {
            o.compare_cpu = true;
            continue;
        }
        if (a.rfind("--", 0) != 0)
            return fail("unexpected argument '" + a + "'");
        if (i + 1 >= args.size())
            return fail("missing value for " + a);
        const std::string& v = args[++i];

        if (a == "--model") {
            o.model = v;
        } else if (a == "--target") {
            if (v == "cpu")
                o.target = Target::kCpu;
            else if (v == "npu")
                o.target = Target::kNpu;
            else
                return fail("--target must be cpu or npu");
        } else if (a == "--strict") {
            if (!ParseOnOff(v, o.strict))
                return fail("--strict must be on or off");
        } else if (a == "--fp16") {
            if (!ParseOnOff(v, o.fp16))
                return fail("--fp16 must be on or off");
        } else if (a == "--ep-context") {
            if (!ParseOnOff(v, o.ep_context))
                return fail("--ep-context must be on or off");
        } else if (a == "--context-file") {
            o.context_file = v;
        } else if (a == "--threads") {
            if (!ParseInt(v, o.threads) || o.threads < 1)
                return fail("--threads must be a positive integer");
        } else if (a == "--run-mode") {
            if (v == "run")
                o.run_mode = RunMode::kRun;
            else if (v == "iobinding")
                o.run_mode = RunMode::kIoBinding;
            else
                return fail("--run-mode must be run or iobinding");
        } else if (a == "--reference-dir") {
            o.reference_dir = v;
        } else if (a == "--reference-config") {
            o.reference_config = v;
        } else if (a == "--tolerance") {
            if (!ParseDouble(v, o.tolerance) || o.tolerance < 0)
                return fail("--tolerance must be a non-negative number");
        } else if (a == "--fill") {
            double ignored = 0;
            if (v != "pattern" && !ParseDouble(v, ignored))
                return fail("--fill must be 'pattern' or a number");
            o.fill = v;
        } else if (a == "--save-outputs") {
            o.save_outputs = v;
        } else if (a == "--init-reps") {
            if (!ParseInt(v, o.init_reps) || o.init_reps < 1)
                return fail("--init-reps must be a positive integer");
        } else if (a == "--warmup") {
            if (!ParseInt(v, o.warmup) || o.warmup < 0)
                return fail("--warmup must be a non-negative integer");
        } else if (a == "--runs") {
            if (!ParseInt(v, o.runs) || o.runs < 0)
                return fail("--runs must be a non-negative integer");
        } else if (a == "--qnn-profiling") {
            if (v == "off")
                o.qnn_profiling = QnnProfiling::kOff;
            else if (v == "basic")
                o.qnn_profiling = QnnProfiling::kBasic;
            else if (v == "detailed")
                o.qnn_profiling = QnnProfiling::kDetailed;
            else
                return fail("--qnn-profiling must be off, basic or detailed");
        } else if (a == "--qnn-profile-file") {
            o.qnn_profile_file = v;
        } else if (a == "--json") {
            o.json = v;
        } else {
            return fail("unknown option '" + a + "'");
        }
    }

    if (o.model.empty())
        return fail("--model is required");
    if (o.target == Target::kCpu &&
        (o.fp16 || o.ep_context || o.qnn_profiling != QnnProfiling::kOff)) {
        return fail("--fp16, --ep-context and --qnn-profiling apply to --target npu only");
    }
    if (o.ep_context && o.context_file.empty())
        return fail("--ep-context on requires --context-file");
    if (o.reuse_context && !o.ep_context)
        return fail("--reuse-context requires --ep-context on");
    if (o.qnn_profiling != QnnProfiling::kOff && o.qnn_profile_file.empty())
        return fail("--qnn-profiling requires --qnn-profile-file");
    if (!o.reference_config.empty() && o.reference_dir.empty())
        return fail("--reference-config requires --reference-dir");
    return r;
}

std::string Usage(std::string_view program) {
    std::ostringstream s;
    s << "usage: " << program << " --model PATH [options]\n"
      << R"(
Runs an ONNX model with ONNX Runtime on the CPU EP or on the Snapdragon NPU
through the QNN plugin EP, optionally compares the outputs with reference
outputs from tools/ort_probe/make_reference.py, and measures latency.

  --model PATH             ONNX model (float32 input and output, static shapes)
  --target cpu|npu         execution target (default npu)
  --strict on|off          NPU: disable CPU EP fallback (default on)
  --fp16 on|off            NPU: enable_htp_fp16_precision (default off)
  --threads N              intra-op threads for every session (default 4)
  --run-mode run|iobinding Session::Run with Ort::Value inputs, or IoBinding with
                           preallocated input/output buffers (default run)
  --ep-context on|off      NPU: generate an EP context model and run from it (default off)
  --context-file PATH      EP context model path (required with --ep-context on)
  --reuse-context          load an existing --context-file instead of regenerating it
  --reference-dir DIR      reference tensors (index.tsv): their inputs are used
  --reference-config NAME  reference outputs to compare with, e.g. npu_s8w
  --tolerance X            max abs diff allowed (default 1e-5 for cpu, 0 for npu)
  --fill pattern|VALUE     input when no reference dir is given (default pattern)
  --compare-cpu            also run the inputs on the CPU EP and compare
  --save-outputs DIR       write this run's outputs as raw float32 files
  --init-reps N            session creations to time (default 1)
  --warmup N               untimed runs before timing (default 20)
  --runs N                 timed runs on the first input (default 0: no timing)
  --qnn-profiling off|basic|detailed   QNN EP profiling (default off)
  --qnn-profile-file PATH  QNN profiling CSV output
  --json PATH              write a JSON summary
  --verbose-log            ONNX Runtime verbose logging to stderr

Exit codes: 0 ok, 1 check failed (output comparison, device selection) or
unexpected error, 2 usage error, 3 ONNX Runtime error, 4 ONNX Runtime could not
be loaded or a loaded ONNX Runtime/QNN DLL is not the bundled one.
)";
    return s.str();
}

// --------------------------------------------------------------------------- tensors

size_t TensorRecord::ElementCount() const {
    size_t n = 1;
    for (int64_t d : shape)
        n *= static_cast<size_t>(d);
    return n;
}

namespace {

std::vector<std::string> SplitTabs(const std::string& line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (true) {
        size_t tab = line.find('\t', start);
        fields.push_back(
            line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
        if (tab == std::string::npos)
            break;
        start = tab + 1;
    }
    return fields;
}

std::vector<int64_t> ParseShape(const std::string& s) {
    std::vector<int64_t> shape;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        std::string dim =
            s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        int64_t d = 0;
        auto [ptr, ec] = std::from_chars(dim.data(), dim.data() + dim.size(), d);
        if (ec != std::errc() || ptr != dim.data() + dim.size() || d <= 0)
            throw std::runtime_error("invalid shape '" + s + "'");
        shape.push_back(d);
        if (comma == std::string::npos)
            break;
        start = comma + 1;
    }
    return shape;
}

}  // namespace

std::vector<TensorRecord> ReadIndex(const fs::path& index_file) {
    std::ifstream in(index_file);
    if (!in)
        throw std::runtime_error("cannot open " + index_file.string());
    std::vector<TensorRecord> records;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line_no == 1 || line.empty())
            continue;  // header
        std::vector<std::string> f = SplitTabs(line);
        if (f.size() != 7) {
            throw std::runtime_error(index_file.string() + ":" + std::to_string(line_no) +
                                     ": expected 7 tab-separated fields, got " +
                                     std::to_string(f.size()));
        }
        TensorRecord r;
        r.kind = f[0];
        r.config = f[1];
        auto [ptr, ec] = std::from_chars(f[2].data(), f[2].data() + f[2].size(), r.image);
        if (ec != std::errc() || ptr != f[2].data() + f[2].size())
            throw std::runtime_error(index_file.string() + ":" + std::to_string(line_no) +
                                     ": invalid image index '" + f[2] + "'");
        r.image_id = f[3];
        r.file = f[4];
        r.dtype = f[5];
        if (r.dtype != "float32")
            throw std::runtime_error(index_file.string() + ":" + std::to_string(line_no) +
                                     ": unsupported dtype '" + r.dtype + "'");
        r.shape = ParseShape(f[6]);
        records.push_back(std::move(r));
    }
    return records;
}

std::vector<TensorRecord> SelectRecords(const std::vector<TensorRecord>& records,
                                        std::string_view kind, std::string_view config) {
    std::vector<TensorRecord> out;
    for (const TensorRecord& r : records) {
        if (r.kind == kind && r.config == config)
            out.push_back(r);
    }
    std::sort(out.begin(), out.end(),
              [](const TensorRecord& a, const TensorRecord& b) { return a.image < b.image; });
    return out;
}

std::vector<float> ReadFloat32(const fs::path& file, size_t expected_count) {
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("cannot open " + file.string());
    const auto bytes = static_cast<size_t>(in.tellg());
    if (bytes != expected_count * sizeof(float)) {
        throw std::runtime_error(file.string() + ": expected " +
                                 std::to_string(expected_count * sizeof(float)) + " bytes, got " +
                                 std::to_string(bytes));
    }
    std::vector<float> data(expected_count);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    if (!in)
        throw std::runtime_error("failed to read " + file.string());
    return data;
}

void WriteFloat32(const fs::path& file, std::span<const float> data) {
    if (file.has_parent_path())
        fs::create_directories(file.parent_path());
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size_bytes()));
    if (!out)
        throw std::runtime_error("failed to write " + file.string());
}

std::vector<float> PatternFill(size_t count) {
    std::vector<float> v(count);
    for (size_t i = 0; i < count; ++i)
        v[i] = static_cast<float>((i * 37 + 11) % 64) / 32.0f - 1.0f;
    return v;
}

// --------------------------------------------------------------------------- comparison

size_t ArgMax(std::span<const float> values) {
    return static_cast<size_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

bool AllClose(std::span<const float> actual, std::span<const float> expected, double rtol,
              double atol) {
    if (actual.size() != expected.size())
        throw std::invalid_argument("compared tensors differ in size");
    for (size_t i = 0; i < actual.size(); ++i) {
        const double e = expected[i];
        if (!(std::fabs(static_cast<double>(actual[i]) - e) <= atol + rtol * std::fabs(e)))
            return false;
    }
    return true;
}

Comparison Compare(std::span<const float> actual, std::span<const float> expected,
                   double tolerance) {
    if (actual.size() != expected.size())
        throw std::invalid_argument("compared tensors differ in size");
    if (actual.empty())
        throw std::invalid_argument("compared tensors are empty");
    Comparison c;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = std::fabs(static_cast<double>(actual[i]) - expected[i]);
        if (!(d <= c.max_abs_diff))  // also catches NaN
            c.max_abs_diff = std::isnan(d) ? INFINITY : d;
    }
    c.argmax_actual = ArgMax(actual);
    c.argmax_expected = ArgMax(expected);
    c.top1_match = c.argmax_actual == c.argmax_expected;
    c.within_tolerance = c.max_abs_diff <= tolerance;
    return c;
}

// --------------------------------------------------------------------------- statistics

LatencyStats ComputeStats(std::vector<double> samples) {
    if (samples.empty())
        throw std::invalid_argument("no samples");
    std::sort(samples.begin(), samples.end());
    LatencyStats s;
    s.count = samples.size();
    double sum = 0;
    for (double x : samples)
        sum += x;
    s.mean = sum / static_cast<double>(s.count);
    auto rank = [&](double p) {
        auto r = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(s.count)));
        return samples[std::max<size_t>(r, 1) - 1];
    };
    s.p50 = rank(50);
    s.p95 = rank(95);
    s.min = samples.front();
    s.max = samples.back();
    if (s.count > 1) {
        double ss = 0;
        for (double x : samples)
            ss += (x - s.mean) * (x - s.mean);
        s.stddev = std::sqrt(ss / static_cast<double>(s.count - 1));
    }
    return s;
}

double Median(std::vector<double> values) {
    if (values.empty())
        throw std::invalid_argument("no values");
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

// --------------------------------------------------------------------------- JSON

std::string JsonEscape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += ch;
                }
        }
    }
    return out;
}

void JsonWriter::Separator() {
    if (after_key_) {
        after_key_ = false;
        return;
    }
    if (!first_.empty()) {
        if (!first_.back())
            out_ += ',';
        first_.back() = false;
    }
}

JsonWriter& JsonWriter::BeginObject() {
    Separator();
    out_ += '{';
    first_.push_back(true);
    return *this;
}

JsonWriter& JsonWriter::EndObject() {
    out_ += '}';
    first_.pop_back();
    return *this;
}

JsonWriter& JsonWriter::BeginArray() {
    Separator();
    out_ += '[';
    first_.push_back(true);
    return *this;
}

JsonWriter& JsonWriter::EndArray() {
    out_ += ']';
    first_.pop_back();
    return *this;
}

JsonWriter& JsonWriter::Key(std::string_view key) {
    Separator();
    out_ += '"';
    out_ += JsonEscape(key);
    out_ += "\":";
    after_key_ = true;
    return *this;
}

JsonWriter& JsonWriter::String(std::string_view value) {
    Separator();
    out_ += '"';
    out_ += JsonEscape(value);
    out_ += '"';
    return *this;
}

JsonWriter& JsonWriter::Number(double value) {
    Separator();
    if (!std::isfinite(value)) {
        out_ += "null";
        return *this;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", value);
    out_ += buf;
    return *this;
}

JsonWriter& JsonWriter::Int(int64_t value) {
    Separator();
    out_ += std::to_string(value);
    return *this;
}

JsonWriter& JsonWriter::Bool(bool value) {
    Separator();
    out_ += value ? "true" : "false";
    return *this;
}

JsonWriter& JsonWriter::Null() {
    Separator();
    out_ += "null";
    return *this;
}

}  // namespace probe
