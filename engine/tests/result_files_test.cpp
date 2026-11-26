// Unit tests for result export without ONNX Runtime: file stamps, result names,
// CSV and number formatting, collision suffixes and atomic file writes.

#include <gtest/gtest.h>

#include <charconv>
#include <clocale>
#include <fstream>
#include <limits>
#include <sstream>

#include "edgebench/result_files.hpp"
#include "edgebench/result_json.hpp"
#include "edgebench/run_identity.hpp"
#include "test_files.hpp"

namespace eb = edgebench;
namespace fs = std::filesystem;
using namespace std::chrono;

namespace {

system_clock::time_point at(int y, unsigned mo, unsigned d, int h, int mi, int s, int us) {
    return system_clock::time_point(sys_days(year{y} / month{mo} / day{d})) + hours(h) +
           minutes(mi) + seconds(s) + microseconds(us);
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// A temporary directory for file tests (removed with its contents).
struct TempDirectory {
    eb::test::TempModel anchor{"anchor.txt"};
    fs::path path() const { return anchor.path().parent_path(); }
};

// A small successful result with known latencies (no ONNX Runtime involved).
eb::BenchmarkResult fake_result() {
    eb::BenchmarkResult r;
    r.schema_version = "1.2.0";
    r.run_id = eb::new_run_id();
    r.started_at = at(2026, 11, 21, 23, 52, 10, 123456);
    r.timestamp_utc = eb::utc_timestamp(r.started_at);
    r.config.model = "model.onnx";
    r.config.trials = 2;
    r.config.iterations = 3;
    r.batch_size = 1;
    const std::vector<std::vector<double>> latencies = {{0.1, 0.2, 0.30000000000000004},
                                                        {1.0, 1e-7, 2.5}};
    for (int i = 0; i < 2; ++i) {
        eb::TrialResult t;
        t.trial_index = i;
        t.sequence = i;
        t.session_create_ms = 10.0 + i;
        t.latencies_ms = latencies[static_cast<std::size_t>(i)];
        t.stats = eb::compute_stats(t.latencies_ms, 1);
        t.session_init.cold_ms = t.session_create_ms;
        r.timing.trials.push_back(t);
        r.timing.session_init.timed_ms.push_back(t.session_create_ms);
        r.timing.session_init.cold_ms.push_back(t.session_create_ms);
    }
    r.timing.session_init.session_create_ms = 10.0;
    r.timing.aggregate = eb::AggregateResult{
        eb::compute_stats(std::vector<double>{0.1, 0.2, 0.30000000000000004, 1.0, 1e-7, 2.5}, 1),
        eb::compute_trial_variation(std::vector<double>{r.timing.trials[0].stats.mean_ms,
                                                        r.timing.trials[1].stats.mean_ms})};
    return r;
}

}  // namespace

// --------------------------------------------------------------------------- stamps

TEST(FileStamp, Format) {
    EXPECT_EQ(eb::file_stamp(at(2026, 11, 21, 23, 52, 10, 123456)), "20261121T235210.123Z");
    EXPECT_EQ(eb::file_stamp(at(2026, 1, 2, 3, 4, 5, 6000)), "20260102T030405.006Z");
    EXPECT_EQ(eb::file_stamp(system_clock::time_point{}), "19700101T000000.000Z");
}

TEST(FileStamp, MillisecondsAreTruncatedNeverRounded) {
    // .999999 s must stay in its second, minute, day and year.
    EXPECT_EQ(eb::file_stamp(at(2026, 12, 31, 23, 59, 59, 999999)), "20261231T235959.999Z");
    EXPECT_EQ(eb::file_stamp(at(2026, 11, 21, 10, 0, 0, 999)), "20261121T100000.000Z");
    EXPECT_EQ(eb::file_stamp(system_clock::time_point{} - microseconds(1)),
              "19691231T235959.999Z");  // before the epoch: floor, not truncation toward 0
}

// --------------------------------------------------------------------------- names

TEST(ResultName, Validation) {
    for (const char* ok : {"run1", "fp32-cpu_t4", "a.b", "A", "x_20261121"})
        EXPECT_TRUE(eb::is_valid_result_name(ok)) << ok;
    for (const char* bad : {"", ".hidden", "has space", "a/b", "a\\b", "x:y", "c*", "\xC3\xA9"})
        EXPECT_FALSE(eb::is_valid_result_name(bad)) << bad;
    EXPECT_FALSE(eb::is_valid_result_name(std::string(129, 'a')));
}

TEST(ResultName, DerivedFromTheConfiguration) {
    eb::RunConfig c;
    c.model = "models/mobilenetv2/mobilenetv2_fp32_b1.onnx";
    EXPECT_EQ(eb::default_result_name(c), "mobilenetv2_fp32_b1_cpu_iobinding_t4");
    c.binding = eb::RunBinding::run;
    c.threads = 8;
    EXPECT_EQ(eb::default_result_name(c), "mobilenetv2_fp32_b1_cpu_run_t8");
    c.model = "dir/my model (v2).onnx";  // characters outside the --name set
    EXPECT_EQ(eb::default_result_name(c), "my_model__v2__cpu_run_t8");
    EXPECT_TRUE(eb::is_valid_result_name(eb::default_result_name(c)));
    c.name = "chosen";
    EXPECT_EQ(eb::result_name(c), "chosen");
}

// --------------------------------------------------------------------------- formatting

TEST(CsvField, Rfc4180Escaping) {
    EXPECT_EQ(eb::csv_field("plain"), "plain");
    EXPECT_EQ(eb::csv_field(""), "");
    EXPECT_EQ(eb::csv_field("a,b"), "\"a,b\"");
    EXPECT_EQ(eb::csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    EXPECT_EQ(eb::csv_field("two\nlines"), "\"two\nlines\"");
    EXPECT_EQ(eb::csv_field("cr\r"), "\"cr\r\"");
}

TEST(FormatNumber, ShortestRoundTrip) {
    for (double v : {0.1, 0.30000000000000004, 1e-7, 123456.789, 2.5, 1.0 / 3.0,
                     std::numeric_limits<double>::denorm_min(), 1.7976931348623157e308}) {
        const std::string s = eb::format_number(v);
        double back = 0;
        const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), back);
        ASSERT_EQ(ec, std::errc()) << s;
        EXPECT_EQ(back, v) << s;  // exact round trip
        EXPECT_EQ(s.find(','), std::string::npos) << s;
    }
    EXPECT_EQ(eb::format_number(0.1), "0.1");
    EXPECT_EQ(eb::format_number(2.0), "2.0");
}

TEST(FormatNumber, IndependentOfTheCLocale) {
    const std::string before = eb::format_number(1234.5);
    if (!std::setlocale(LC_ALL, "de-DE"))
        GTEST_SKIP() << "German locale not available";
    const std::string german = eb::format_number(1234.5);
    std::setlocale(LC_ALL, "C");
    EXPECT_EQ(german, before);
    EXPECT_EQ(german, "1234.5");
}

TEST(Csv, LatenciesAndSummary) {
    const eb::BenchmarkResult r = fake_result();
    EXPECT_EQ(eb::latencies_csv(r),
              "trial_index,iteration_index,latency_ms\r\n"
              "0,0,0.1\r\n0,1,0.2\r\n0,2,0.30000000000000004\r\n"
              "1,0,1.0\r\n1,1,1e-07\r\n1,2,2.5\r\n");
    const std::string summary = eb::summary_csv(r);
    std::istringstream lines(summary);
    std::vector<std::string> rows;
    for (std::string line; std::getline(lines, line);)
        rows.push_back(line);
    ASSERT_EQ(rows.size(), 4u);
    EXPECT_EQ(rows[0],
              "row,trial_index,count,mean_ms,p50_ms,p95_ms,p99_ms,stddev_ms,min_ms,max_ms,"
              "inferences_per_sec,items_per_sec,session_create_ms\r");
    EXPECT_EQ(rows[1].rfind("trial,0,3,", 0), 0u) << rows[1];
    EXPECT_NE(rows[1].find(",,,"), std::string::npos) << rows[1];  // p95 and p99 null (count 3)
    EXPECT_EQ(rows[2].substr(rows[2].size() - 6), ",11.0\r") << rows[2];  // trial 1's create ms
    EXPECT_EQ(rows[3].rfind("aggregate,,6,", 0), 0u) << rows[3];
}

// --------------------------------------------------------------------------- files

TEST(AtomicFiles, CreateRefusesToOverwrite) {
    TempDirectory dir;
    const fs::path p = dir.path() / "a.json";
    ASSERT_TRUE(eb::create_file_atomically(p, "first").ok());
    const eb::Result<void> again = eb::create_file_atomically(p, "second");
    ASSERT_FALSE(again.ok());
    EXPECT_EQ(again.error().exit_code, eb::ExitCode::output_error);
    EXPECT_EQ(again.error().kind, "file_exists");
    EXPECT_NE(again.error().message.find("a.json"), std::string::npos);
    EXPECT_EQ(read_file(p), "first");  // untouched
}

TEST(AtomicFiles, ReplaceSwapsTheWholeFile) {
    TempDirectory dir;
    const fs::path p = dir.path() / "x_latest.json";
    ASSERT_TRUE(eb::replace_file_atomically(p, "old content, longer").ok());
    ASSERT_TRUE(eb::replace_file_atomically(p, "new").ok());
    EXPECT_EQ(read_file(p), "new");
    for (const auto& e : fs::directory_iterator(dir.path()))  // no temporary files left
        EXPECT_EQ(e.path().filename().string().find(".tmp-"), std::string::npos) << e.path();
}

TEST(AtomicFiles, ErrorsNameThePath) {
    TempDirectory dir;
    const fs::path p = dir.path() / "missing_dir" / "a.json";
    const eb::Result<void> r = eb::create_file_atomically(p, "x");
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().exit_code, eb::ExitCode::output_error);
    EXPECT_NE(r.error().message.find("missing_dir"), std::string::npos) << r.error().message;
}

TEST(WriteResultFiles, LayoutAndRawLink) {
    TempDirectory dir;
    const eb::BenchmarkResult r = fake_result();
    const auto w = eb::write_result_files(r, dir.path() / "out", "fake");
    ASSERT_TRUE(w.ok()) << w.error().message;
    EXPECT_EQ(w.value().stamp, "20261121T235210.123Z");
    const fs::path base = dir.path() / "out" / "fake_20261121T235210.123Z";
    EXPECT_EQ(w.value().result_file, fs::path(base.string() + ".json"));
    EXPECT_EQ(read_file(base / "latencies.csv"), eb::latencies_csv(r));
    EXPECT_EQ(read_file(base / "summary.csv"), eb::summary_csv(r));
    EXPECT_EQ(read_file(dir.path() / "out" / "fake_latest.json"), read_file(w.value().result_file));
    EXPECT_EQ(w.value().document.at("raw").at("latencies_file"),
              "fake_20261121T235210.123Z/latencies.csv");
    EXPECT_EQ(w.value().document.at("raw").at("format"), "csv");
}

TEST(WriteResultFiles, SkipsTakenStamps) {
    TempDirectory dir;
    const eb::BenchmarkResult r = fake_result();
    // Pre-create the first two candidates: a result file and a run folder.
    std::ofstream(dir.path() / "fake_20261121T235210.123Z.json") << "earlier run";
    fs::create_directory(dir.path() / "fake_20261121T235210.123Z-2");
    const auto w = eb::write_result_files(r, dir.path(), "fake");
    ASSERT_TRUE(w.ok()) << w.error().message;
    EXPECT_EQ(w.value().stamp, "20261121T235210.123Z-3");
    EXPECT_EQ(read_file(dir.path() / "fake_20261121T235210.123Z.json"), "earlier run");
    EXPECT_TRUE(fs::is_empty(dir.path() / "fake_20261121T235210.123Z-2"));
}

TEST(WriteResultFiles, NoLatenciesNoRunFolder) {
    TempDirectory dir;
    eb::BenchmarkResult r = fake_result();
    r.timing = eb::TimingResult{};
    eb::set_failure(r, eb::Error{eb::ExitCode::ort_load_failure, "dll_not_found", "missing"},
                    "environment");
    const auto w = eb::write_result_files(r, dir.path(), "failed");
    ASSERT_TRUE(w.ok()) << w.error().message;
    EXPECT_FALSE(w.value().run_dir.has_value());
    EXPECT_TRUE(w.value().document.at("raw").is_null());
    EXPECT_EQ(w.value().document.at("status"), "failed");
    EXPECT_FALSE(fs::exists(dir.path() / ("failed_" + w.value().stamp)));
}

TEST(WriteResultFiles, UnusableDirectoryIsAnOutputError) {
    TempDirectory dir;
    std::ofstream(dir.path() / "a_file") << "x";
    const auto w = eb::write_result_files(fake_result(), dir.path() / "a_file" / "sub", "x");
    ASSERT_FALSE(w.ok());
    EXPECT_EQ(w.error().exit_code, eb::ExitCode::output_error);
    EXPECT_NE(w.error().message.find("a_file"), std::string::npos) << w.error().message;
}
