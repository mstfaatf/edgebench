#include "probe_util.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using namespace probe;

namespace {

ParseResult Parse(std::vector<std::string> args) {
    return ParseArgs(args);
}

fs::path TempDir(const char* name) {
    fs::path dir = fs::temp_directory_path() / "ort_probe_tests" / name;
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

}  // namespace

// --------------------------------------------------------------------------- options

TEST(ParseArgs, Defaults) {
    ParseResult r = Parse({"--model", "m.onnx"});
    ASSERT_TRUE(r.error.empty()) << r.error;
    EXPECT_EQ(r.options.model, fs::path("m.onnx"));
    EXPECT_EQ(r.options.target, Target::kNpu);
    EXPECT_TRUE(r.options.strict);
    EXPECT_FALSE(r.options.fp16);
    EXPECT_EQ(r.options.threads, 4);
    EXPECT_EQ(r.options.run_mode, RunMode::kRun);
    EXPECT_EQ(r.options.runs, 0);
}

TEST(ParseArgs, FullNpuCommandLine) {
    ParseResult r = Parse({"--model",
                           "m.onnx",
                           "--target",
                           "npu",
                           "--strict",
                           "off",
                           "--fp16",
                           "on",
                           "--threads",
                           "2",
                           "--run-mode",
                           "iobinding",
                           "--ep-context",
                           "on",
                           "--context-file",
                           "cache/qnn/x_ctx.onnx",
                           "--reuse-context",
                           "--init-reps",
                           "3",
                           "--warmup",
                           "5",
                           "--runs",
                           "10",
                           "--qnn-profiling",
                           "basic",
                           "--qnn-profile-file",
                           "p.csv",
                           "--tolerance",
                           "0.25",
                           "--json",
                           "out.json"});
    ASSERT_TRUE(r.error.empty()) << r.error;
    const Options& o = r.options;
    EXPECT_FALSE(o.strict);
    EXPECT_TRUE(o.fp16);
    EXPECT_EQ(o.threads, 2);
    EXPECT_EQ(o.run_mode, RunMode::kIoBinding);
    EXPECT_TRUE(o.ep_context);
    EXPECT_TRUE(o.reuse_context);
    EXPECT_EQ(o.init_reps, 3);
    EXPECT_EQ(o.warmup, 5);
    EXPECT_EQ(o.runs, 10);
    EXPECT_EQ(o.qnn_profiling, QnnProfiling::kBasic);
    EXPECT_DOUBLE_EQ(o.tolerance, 0.25);
}

TEST(ParseArgs, Errors) {
    EXPECT_FALSE(Parse({}).error.empty());                                      // no model
    EXPECT_FALSE(Parse({"--model"}).error.empty());                             // missing value
    EXPECT_FALSE(Parse({"--model", "m", "--target", "gpu"}).error.empty());     // bad target
    EXPECT_FALSE(Parse({"--model", "m", "--threads", "0"}).error.empty());      // threads >= 1
    EXPECT_FALSE(Parse({"--model", "m", "--threads", "4x"}).error.empty());     // not a number
    EXPECT_FALSE(Parse({"--model", "m", "--ep-context", "on"}).error.empty());  // no context file
    EXPECT_FALSE(Parse({"--model", "m", "--reuse-context"}).error.empty());
    EXPECT_FALSE(Parse({"--model", "m", "--target", "cpu", "--fp16", "on"}).error.empty());
    EXPECT_FALSE(Parse({"--model", "m", "--qnn-profiling", "basic"}).error.empty());
    EXPECT_FALSE(Parse({"--model", "m", "--reference-config", "c"}).error.empty());
    EXPECT_FALSE(Parse({"--model", "m", "--fill", "abc"}).error.empty());
    EXPECT_FALSE(Parse({"--model", "m", "--bogus", "1"}).error.empty());
    EXPECT_FALSE(Parse({"--model", "m", "stray"}).error.empty());
}

TEST(ParseArgs, ReferenceDirWithoutConfigUsesInputsOnly) {
    EXPECT_TRUE(Parse({"--model", "m", "--reference-dir", "d"}).error.empty());
}

TEST(ParseArgs, Help) {
    EXPECT_TRUE(Parse({"--help"}).options.help);
}

// --------------------------------------------------------------------------- tensors

TEST(TensorFiles, RoundTripAndSizeCheck) {
    const fs::path dir = TempDir("roundtrip");
    const std::vector<float> data = {1.5f, -2.0f, 0.0f, 3.25f};
    WriteFloat32(dir / "sub" / "t.f32", data);
    EXPECT_EQ(ReadFloat32(dir / "sub" / "t.f32", 4), data);
    EXPECT_THROW(ReadFloat32(dir / "sub" / "t.f32", 5), std::runtime_error);
    EXPECT_THROW(ReadFloat32(dir / "missing.f32", 4), std::runtime_error);
}

TEST(ReadIndex, ParsesAndSelects) {
    const fs::path dir = TempDir("index");
    std::ofstream(dir / "index.tsv")
        << "kind\tconfig\timage\timage_id\tfile\tdtype\tshape\n"
        << "output\tnpu_s8w\t1\tval/b.JPEG\toutputs/npu_s8w/01.f32\tfloat32\t1,1000\r\n"
        << "input\tnormalized\t0\tval/a.JPEG\tinputs/00.f32\tfloat32\t1,3,224,224\n"
        << "output\tnpu_s8w\t0\tval/a.JPEG\toutputs/npu_s8w/00.f32\tfloat32\t1,1000\n";
    const std::vector<TensorRecord> all = ReadIndex(dir / "index.tsv");
    ASSERT_EQ(all.size(), 3u);
    EXPECT_EQ(all[1].ElementCount(), 3u * 224 * 224);

    const auto outs = SelectRecords(all, "output", "npu_s8w");
    ASSERT_EQ(outs.size(), 2u);
    EXPECT_EQ(outs[0].image, 0);  // sorted by image
    EXPECT_EQ(outs[1].image_id, "val/b.JPEG");
    EXPECT_EQ(outs[1].shape, (std::vector<int64_t>{1, 1000}));
    EXPECT_TRUE(SelectRecords(all, "output", "cpu_fp32").empty());
}

TEST(ReadIndex, RejectsMalformedLines) {
    const fs::path dir = TempDir("bad_index");
    const char* header = "kind\tconfig\timage\timage_id\tfile\tdtype\tshape\n";
    std::ofstream(dir / "fields.tsv") << header << "input\tnormalized\t0\tx\tf\tfloat32\n";
    std::ofstream(dir / "dtype.tsv") << header << "input\tnormalized\t0\tx\tf\tuint8\t1\n";
    std::ofstream(dir / "shape.tsv") << header << "input\tnormalized\t0\tx\tf\tfloat32\t1,0\n";
    std::ofstream(dir / "image.tsv") << header << "input\tnormalized\tz\tx\tf\tfloat32\t1\n";
    for (const char* f : {"fields.tsv", "dtype.tsv", "shape.tsv", "image.tsv"})
        EXPECT_THROW(ReadIndex(dir / f), std::runtime_error) << f;
}

TEST(PatternFill, DeterministicAndInRange) {
    const std::vector<float> a = PatternFill(100), b = PatternFill(100);
    EXPECT_EQ(a, b);
    for (float v : a) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LT(v, 1.0f);
    }
    EXPECT_NE(a[0], a[1]);
}

// --------------------------------------------------------------------------- comparison

TEST(Compare, ExactMatch) {
    const std::vector<float> a = {0.1f, 0.9f, 0.3f};
    const Comparison c = Compare(a, a, 0.0);
    EXPECT_EQ(c.max_abs_diff, 0.0);
    EXPECT_TRUE(c.ok());
    EXPECT_EQ(c.argmax_actual, 1u);
}

TEST(Compare, ToleranceAndTop1AreSeparate) {
    const std::vector<float> expected = {0.0f, 1.0f, 0.5f};
    const std::vector<float> close = {0.0f, 1.0f, 0.5f + 1e-6f};
    EXPECT_TRUE(Compare(close, expected, 1e-5).ok());
    EXPECT_FALSE(Compare(close, expected, 0.0).within_tolerance);
    EXPECT_TRUE(Compare(close, expected, 0.0).top1_match);

    const std::vector<float> swapped = {0.0f, 0.5f, 1.0f};
    const Comparison c = Compare(swapped, expected, 10.0);
    EXPECT_TRUE(c.within_tolerance);
    EXPECT_FALSE(c.top1_match);
    EXPECT_FALSE(c.ok());
}

TEST(Compare, NanIsNeverWithinTolerance) {
    const std::vector<float> expected = {0.0f, 1.0f};
    const std::vector<float> actual = {std::nanf(""), 1.0f};
    EXPECT_FALSE(Compare(actual, expected, 1e9).within_tolerance);
}

TEST(Compare, SizeMismatchThrows) {
    const std::vector<float> a = {1.0f}, b = {1.0f, 2.0f}, empty;
    EXPECT_THROW(Compare(a, b, 0.0), std::invalid_argument);
    EXPECT_THROW(Compare(empty, empty, 0.0), std::invalid_argument);
}

TEST(AllClose, RelativeAndAbsolute) {
    const std::vector<float> expected = {100.0f, 0.0f};
    EXPECT_TRUE(AllClose(std::vector<float>{100.9f, 0.0005f}, expected, 1e-2, 1e-3));
    EXPECT_FALSE(AllClose(std::vector<float>{102.0f, 0.0f}, expected, 1e-2, 1e-3));
    EXPECT_FALSE(AllClose(std::vector<float>{100.0f, 0.01f}, expected, 1e-2, 1e-3));
    EXPECT_FALSE(AllClose(std::vector<float>{std::nanf(""), 0.0f}, expected, 1e-2, 1e-3));
}

// --------------------------------------------------------------------------- statistics

TEST(Stats, NearestRankPercentiles) {
    std::vector<double> samples;
    for (int i = 100; i >= 1; --i)
        samples.push_back(i);  // 1..100, unsorted
    const LatencyStats s = ComputeStats(samples);
    EXPECT_EQ(s.count, 100u);
    EXPECT_DOUBLE_EQ(s.mean, 50.5);
    EXPECT_DOUBLE_EQ(s.p50, 50);
    EXPECT_DOUBLE_EQ(s.p95, 95);
    EXPECT_DOUBLE_EQ(s.min, 1);
    EXPECT_DOUBLE_EQ(s.max, 100);
    EXPECT_NEAR(s.stddev, 29.0115, 1e-3);
    EXPECT_THROW(ComputeStats({}), std::invalid_argument);
}

TEST(Stats, Median) {
    EXPECT_DOUBLE_EQ(Median({3, 1, 2}), 2);
    EXPECT_DOUBLE_EQ(Median({4, 1, 3, 2}), 2.5);
    EXPECT_THROW(Median({}), std::invalid_argument);
}

// --------------------------------------------------------------------------- JSON

TEST(Json, EscapesAndStructure) {
    EXPECT_EQ(JsonEscape("a\"b\\c\n\x01"), "a\\\"b\\\\c\\n\\u0001");
    JsonWriter j;
    j.BeginObject()
        .Key("s")
        .String("C:\\path")
        .Key("n")
        .Number(0.5)
        .Key("i")
        .Int(-3)
        .Key("b")
        .Bool(true)
        .Key("z")
        .Null()
        .Key("inf")
        .Number(INFINITY)
        .Key("a")
        .BeginArray()
        .Int(1)
        .Int(2)
        .BeginObject()
        .EndObject()
        .EndArray()
        .EndObject();
    EXPECT_EQ(j.str(),
              R"({"s":"C:\\path","n":0.5,"i":-3,"b":true,"z":null,"inf":null,"a":[1,2,{}]})");
}
