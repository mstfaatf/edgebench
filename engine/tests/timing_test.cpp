// Unit tests for the timing building blocks: statistics, trial scheduling, timer
// resolution, run identity and ONNX model inspection.

#include <gtest/gtest.h>

#include <onnxruntime_c_api.h>

#include <chrono>
#include <cmath>
#include <map>
#include <regex>
#include <set>

#include "edgebench/onnx_inspect.hpp"
#include "edgebench/run_identity.hpp"
#include "edgebench/scheduler.hpp"
#include "edgebench/stats.hpp"
#include "edgebench/timer_info.hpp"

namespace eb = edgebench;
namespace fs = std::filesystem;

// --------------------------------------------------------------------------- statistics

TEST(NearestRank, KnownTable) {
    // Wikipedia's nearest-rank example: {15, 20, 35, 40, 50}.
    const std::vector<double> v = {15, 20, 35, 40, 50};
    EXPECT_EQ(eb::nearest_rank(v, 5), 15);    // rank ceil(0.25) = 1
    EXPECT_EQ(eb::nearest_rank(v, 30), 20);   // rank ceil(1.5) = 2
    EXPECT_EQ(eb::nearest_rank(v, 40), 20);   // rank 2 exactly
    EXPECT_EQ(eb::nearest_rank(v, 50), 35);   // rank ceil(2.5) = 3
    EXPECT_EQ(eb::nearest_rank(v, 100), 50);  // rank 5
}

TEST(NearestRank, ExactRanksDoNotRoundUp) {
    std::vector<double> v(100);
    for (int i = 0; i < 100; ++i)
        v[static_cast<std::size_t>(i)] = i + 1;  // 1..100
    EXPECT_EQ(eb::nearest_rank(v, 50), 50);
    EXPECT_EQ(eb::nearest_rank(v, 95), 95);
    EXPECT_EQ(eb::nearest_rank(v, 99), 99);
    std::vector<double> w(200);
    for (int i = 0; i < 200; ++i)
        w[static_cast<std::size_t>(i)] = i + 1;
    EXPECT_EQ(eb::nearest_rank(w, 50), 100);
    EXPECT_EQ(eb::nearest_rank(w, 95), 190);
    EXPECT_EQ(eb::nearest_rank(w, 99), 198);
}

TEST(Stats, HandComputedSmallArray) {
    // 1, 2, 3, 4: mean 2.5, sample variance 5/3, p50 rank ceil(2) = 2.
    const std::vector<double> v = {4, 1, 3, 2};
    const eb::LatencyStats s = eb::compute_stats(v, 2);
    EXPECT_EQ(s.count, 4u);
    EXPECT_DOUBLE_EQ(s.mean_ms, 2.5);
    EXPECT_EQ(s.p50_ms, 2);
    EXPECT_FALSE(s.p95_ms.has_value());  // count < 20
    EXPECT_FALSE(s.p99_ms.has_value());  // count < 100
    ASSERT_TRUE(s.stddev_ms.has_value());
    EXPECT_NEAR(*s.stddev_ms, std::sqrt(5.0 / 3.0), 1e-12);
    EXPECT_EQ(s.min_ms, 1);
    EXPECT_EQ(s.max_ms, 4);
    EXPECT_DOUBLE_EQ(s.inferences_per_sec, 400.0);  // 1000 / 2.5
    EXPECT_DOUBLE_EQ(s.items_per_sec, 800.0);       // batch size 2
}

TEST(Stats, SingleElement) {
    const eb::LatencyStats s = eb::compute_stats(std::vector<double>{3.0}, 1);
    EXPECT_EQ(s.count, 1u);
    EXPECT_EQ(s.mean_ms, 3.0);
    EXPECT_EQ(s.p50_ms, 3.0);
    EXPECT_EQ(s.min_ms, 3.0);
    EXPECT_EQ(s.max_ms, 3.0);
    EXPECT_FALSE(s.stddev_ms.has_value());  // count < 2
    EXPECT_FALSE(s.p95_ms.has_value());
}

TEST(Stats, TwoElements) {
    const eb::LatencyStats s = eb::compute_stats(std::vector<double>{1.0, 3.0}, 1);
    EXPECT_EQ(s.mean_ms, 2.0);
    EXPECT_EQ(s.p50_ms, 1.0);  // rank ceil(1) = 1: the lower value
    ASSERT_TRUE(s.stddev_ms.has_value());
    EXPECT_NEAR(*s.stddev_ms, std::sqrt(2.0), 1e-12);
}

TEST(Stats, ThresholdsForPercentiles) {
    std::vector<double> v(19, 1.0);
    EXPECT_FALSE(eb::compute_stats(v, 1).p95_ms.has_value());
    v.push_back(2.0);  // 20 values
    const eb::LatencyStats s20 = eb::compute_stats(v, 1);
    ASSERT_TRUE(s20.p95_ms.has_value());
    EXPECT_EQ(*s20.p95_ms, 1.0);  // rank 19
    EXPECT_FALSE(s20.p99_ms.has_value());
    v.resize(100, 1.0);
    EXPECT_TRUE(eb::compute_stats(v, 1).p99_ms.has_value());
}

TEST(Stats, AllEqualValues) {
    const std::vector<double> v(250, 0.5);
    const eb::LatencyStats s = eb::compute_stats(v, 1);
    EXPECT_EQ(s.mean_ms, 0.5);
    EXPECT_EQ(s.p50_ms, 0.5);
    EXPECT_EQ(*s.p95_ms, 0.5);
    EXPECT_EQ(*s.p99_ms, 0.5);
    EXPECT_EQ(*s.stddev_ms, 0.0);
    const eb::TrialVariation tv = eb::compute_trial_variation(std::vector<double>{0.5, 0.5, 0.5});
    EXPECT_EQ(tv.trial_mean_min_ms, 0.5);
    EXPECT_EQ(tv.trial_mean_max_ms, 0.5);
    EXPECT_EQ(*tv.trial_mean_stddev_ms, 0.0);
    EXPECT_EQ(*tv.trial_mean_cv_ratio, 0.0);
}

TEST(TrialVariation, HandComputed) {
    const eb::TrialVariation v = eb::compute_trial_variation(std::vector<double>{2.0, 4.0, 3.0});
    EXPECT_EQ(v.trial_mean_min_ms, 2.0);
    EXPECT_EQ(v.trial_mean_max_ms, 4.0);
    EXPECT_NEAR(*v.trial_mean_stddev_ms, 1.0, 1e-12);       // sample stddev of 2, 3, 4
    EXPECT_NEAR(*v.trial_mean_cv_ratio, 1.0 / 3.0, 1e-12);  // 1 / mean 3
}

TEST(TrialVariation, SingleTrialHasNoSpread) {
    const eb::TrialVariation v = eb::compute_trial_variation(std::vector<double>{2.0});
    EXPECT_EQ(v.trial_mean_min_ms, 2.0);
    EXPECT_FALSE(v.trial_mean_stddev_ms.has_value());
    EXPECT_FALSE(v.trial_mean_cv_ratio.has_value());
}

namespace {

eb::TrialVariation variation(double min_ms, double max_ms, std::optional<double> cv) {
    eb::TrialVariation v;
    v.trial_mean_min_ms = min_ms;
    v.trial_mean_max_ms = max_ms;
    v.trial_mean_cv_ratio = cv;
    v.trial_mean_stddev_ms = cv ? std::optional<double>(0.0) : std::nullopt;
    return v;
}

}  // namespace

TEST(TrialSpread, CoefficientOfVariationThreshold) {
    // Ratio well inside 1.25, so only the CV decides.
    EXPECT_FALSE(eb::trial_spread_high(variation(1.0, 1.1, 0.0999)));
    EXPECT_FALSE(eb::trial_spread_high(variation(1.0, 1.1, 0.10)));  // strictly greater
    EXPECT_TRUE(eb::trial_spread_high(variation(1.0, 1.1, 0.1001)));
}

TEST(TrialSpread, SlowestOverFastestThreshold) {
    // CV well inside 10%, so only the slowest/fastest ratio decides.
    EXPECT_FALSE(eb::trial_spread_high(variation(4.0, 4.0 * 1.2499, 0.01)));
    EXPECT_FALSE(eb::trial_spread_high(variation(4.0, 4.0 * 1.25, 0.01)));  // strictly greater
    EXPECT_TRUE(eb::trial_spread_high(variation(4.0, 4.0 * 1.2501, 0.01)));
}

TEST(TrialSpread, FromComputedVariation) {
    EXPECT_FALSE(eb::trial_spread_high(eb::compute_trial_variation(std::vector<double>{2.0})));
    EXPECT_FALSE(
        eb::trial_spread_high(eb::compute_trial_variation(std::vector<double>{3.9, 4.0, 4.1})));
    // The spread seen with a slow machine state: 3.77 / 10.60 / 10.35 ms.
    EXPECT_TRUE(eb::trial_spread_high(
        eb::compute_trial_variation(std::vector<double>{3.77, 10.60, 10.35})));
}

TEST(Stats, RejectsEmptyInput) {
    EXPECT_THROW(eb::compute_stats(std::vector<double>{}, 1), std::invalid_argument);
    EXPECT_THROW(eb::nearest_rank(std::vector<double>{}, 50), std::invalid_argument);
    EXPECT_THROW(eb::compute_trial_variation(std::vector<double>{}), std::invalid_argument);
}

// --------------------------------------------------------------------------- scheduler

namespace {

std::vector<std::size_t> configs_of(const std::vector<eb::ScheduledTrial>& order) {
    std::vector<std::size_t> out;
    for (const auto& t : order)
        out.push_back(t.config_index);
    return out;
}

}  // namespace

TEST(Scheduler, SingleConfiguration) {
    const auto order = eb::schedule_trials(1, 4);
    EXPECT_EQ(configs_of(order), (std::vector<std::size_t>{0, 0, 0, 0}));
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(order[static_cast<std::size_t>(i)].trial_index, i);
        EXPECT_EQ(order[static_cast<std::size_t>(i)].sequence, i);
    }
}

TEST(Scheduler, RotatesTheStartingConfiguration) {
    EXPECT_EQ(configs_of(eb::schedule_trials(2, 3)), (std::vector<std::size_t>{0, 1, 1, 0, 0, 1}));
    EXPECT_EQ(configs_of(eb::schedule_trials(3, 3)),
              (std::vector<std::size_t>{0, 1, 2, 1, 2, 0, 2, 0, 1}));
    EXPECT_EQ(configs_of(eb::schedule_trials(4, 2)),
              (std::vector<std::size_t>{0, 1, 2, 3, 1, 2, 3, 0}));
    EXPECT_EQ(configs_of(eb::schedule_trials(3, 1)), (std::vector<std::size_t>{0, 1, 2}));
}

TEST(Scheduler, PropertiesForSeveralSizes) {
    for (std::size_t n : {1u, 2u, 3u, 4u}) {
        for (int trials : {1, 2, 3, 4, 5, 8}) {
            const auto order = eb::schedule_trials(n, trials);
            ASSERT_EQ(order.size(), n * static_cast<std::size_t>(trials));
            std::map<std::size_t, std::vector<int>> per_config;
            std::map<std::size_t, int> first_in_round;
            for (std::size_t i = 0; i < order.size(); ++i) {
                const auto& t = order[i];
                EXPECT_EQ(t.sequence, static_cast<int>(i));
                EXPECT_EQ(t.trial_index, static_cast<int>(i / n));  // round number
                per_config[t.config_index].push_back(t.trial_index);
                if (i % n == 0) {
                    EXPECT_EQ(t.config_index, (i / n) % n);  // round r starts at r mod N
                    ++first_in_round[t.config_index];
                }
            }
            // Every configuration gets trials 0..trials-1, in order.
            for (std::size_t c = 0; c < n; ++c) {
                std::vector<int> expected(static_cast<std::size_t>(trials));
                for (int k = 0; k < trials; ++k)
                    expected[static_cast<std::size_t>(k)] = k;
                EXPECT_EQ(per_config[c], expected) << "n=" << n << " trials=" << trials;
            }
            // Over a multiple of N rounds, every configuration starts equally often.
            if (trials % static_cast<int>(n) == 0) {
                for (std::size_t c = 0; c < n; ++c)
                    EXPECT_EQ(first_in_round[c], trials / static_cast<int>(n));
            }
        }
    }
}

TEST(Scheduler, EmptyInputs) {
    EXPECT_TRUE(eb::schedule_trials(0, 3).empty());
    EXPECT_TRUE(eb::schedule_trials(3, 0).empty());
}

// --------------------------------------------------------------------------- timer

TEST(TimerInfo, EffectiveResolution) {
    EXPECT_DOUBLE_EQ(eb::effective_resolution_ns(1, 1'000'000'000, 10'000'000), 100.0);
    EXPECT_DOUBLE_EQ(eb::effective_resolution_ns(1, 1'000'000'000, 0), 1.0);  // counter unknown
    EXPECT_DOUBLE_EQ(eb::effective_resolution_ns(1, 1'000'000, 10'000'000),
                     1000.0);  // coarser nominal
    EXPECT_DOUBLE_EQ(eb::effective_resolution_ns(1, 1'000'000'000, 1'000'000'000), 1.0);
}

TEST(TimerInfo, SteadyClock) {
    const eb::TimerInfo t = eb::steady_clock_info();
    EXPECT_EQ(t.source, "std::chrono::steady_clock");
    ASSERT_TRUE(t.resolution_ns.has_value());
    using period = std::chrono::steady_clock::period;
    EXPECT_GE(*t.resolution_ns, 1e9 * period::num / period::den);
    EXPECT_GT(*t.resolution_ns, 0.0);
    EXPECT_LE(*t.resolution_ns, 1e6);  // a usable benchmark clock is finer than 1 ms
}

// --------------------------------------------------------------------------- identity

TEST(RunIdentity, UuidV4) {
    const std::regex uuid("^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$");
    std::set<std::string> seen;
    for (int i = 0; i < 100; ++i) {
        const std::string id = eb::new_run_id();
        EXPECT_TRUE(std::regex_match(id, uuid)) << id;
        seen.insert(id);
    }
    EXPECT_EQ(seen.size(), 100u);
}

TEST(RunIdentity, UtcTimestamp) {
    using namespace std::chrono;
    const sys_days day = year{2026} / November / 21;
    const auto t =
        system_clock::time_point(day) + hours(14) + minutes(3) + seconds(27) + milliseconds(53);
    EXPECT_EQ(eb::utc_timestamp(t), "2026-11-21T14:03:27.053Z");
    EXPECT_EQ(eb::utc_timestamp(system_clock::time_point(sys_days(year{2000} / January / 1))),
              "2000-01-01T00:00:00.000Z");
}

// --------------------------------------------------------------------------- ONNX inspection

namespace {

const fs::path kFixtures =
    fs::path(EDGEBENCH_SOURCE_DIR) / "engine" / "tests" / "fixtures" / "models";

}  // namespace

TEST(OnnxInspect, FixedFloat32Fixture) {
    const auto g = eb::inspect_onnx_file(kFixtures / "fixed_float32.onnx");
    ASSERT_TRUE(g.ok()) << g.error().message;
    EXPECT_EQ(g.value().ir_version, 8);
    EXPECT_EQ(g.value().opset, 17);
    EXPECT_EQ(g.value().producer_name, "edgebench-test-fixtures");
    EXPECT_EQ(g.value().op_counts,
              (std::map<std::string, int>{{"Add", 1}, {"MatMul", 1}, {"Relu", 1}}));
    EXPECT_EQ(g.value().initializer_dtypes, (std::map<int, int>{{1, 2}}));  // two float tensors
    const eb::ModelPrecision p = eb::infer_precision(g.value());
    EXPECT_EQ(p.stored_precision, "fp32");
    EXPECT_EQ(p.quantization_method, "none");
    EXPECT_FALSE(p.quantization_format.has_value());
}

TEST(OnnxInspect, QdqModelIsInt8) {
    eb::OnnxGraphSummary g;
    g.op_counts = {{"QuantizeLinear", 66}, {"DequantizeLinear", 172}, {"Conv", 52}};
    g.weight_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    g.activation_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    const eb::ModelPrecision p = eb::infer_precision(g);
    EXPECT_EQ(p.stored_precision, "int8");
    EXPECT_EQ(p.quantization_method, "static");
    EXPECT_EQ(p.quantization_format, "qdq");
    EXPECT_EQ(p.weight_dtype, "int8");
    EXPECT_EQ(p.activation_dtype, "uint8");

    g.activation_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;  // 16-bit activations
    EXPECT_EQ(eb::infer_precision(g).stored_precision, "mixed");
    g.op_counts = {{"DynamicQuantizeLinear", 1}, {"MatMulInteger", 1}};
    EXPECT_EQ(eb::infer_precision(g).quantization_method, "dynamic");
    EXPECT_EQ(eb::infer_precision(g).quantization_format, "qoperator");
}

TEST(OnnxInspect, Fp16Weights) {
    eb::OnnxGraphSummary g;
    g.initializer_dtypes = {{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 10}};
    EXPECT_EQ(eb::infer_precision(g).stored_precision, "fp16");
    g.initializer_dtypes[ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT] = 1;
    EXPECT_EQ(eb::infer_precision(g).stored_precision, "mixed");
}

TEST(OnnxInspect, RejectsNonOnnxBytes) {
    const std::string junk = "this is not an ONNX model";
    const auto r = eb::inspect_onnx(std::as_bytes(std::span(junk.data(), junk.size())));
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().kind, "invalid_onnx");
    EXPECT_FALSE(eb::inspect_onnx({}).ok());
}
