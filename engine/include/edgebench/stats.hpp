// Latency statistics with the definitions of the results schema
// (docs/results-schema.md): nearest-rank percentiles, sample standard deviation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace edgebench {

/// timing.trials[].stats and timing.aggregate.stats.
struct LatencyStats {
    std::size_t count = 0;
    double mean_ms = 0;
    double p50_ms = 0;
    std::optional<double> p95_ms;     // null if count < 20
    std::optional<double> p99_ms;     // null if count < 100
    std::optional<double> stddev_ms;  // sample (n - 1); null if count < 2
    double min_ms = 0;
    double max_ms = 0;
    double inferences_per_sec = 0;  // 1000 / mean_ms
    double items_per_sec = 0;       // inferences_per_sec * batch_size
};

/// timing.aggregate.trial_variation: spread of the per-trial means.
struct TrialVariation {
    double trial_mean_min_ms = 0;
    double trial_mean_max_ms = 0;
    std::optional<double> trial_mean_stddev_ms;  // sample (n - 1); null with fewer than 2 trials
    std::optional<double> trial_mean_cv_ratio;   // stddev / mean of the trial means
};

/// Nearest-rank percentile of `sorted` (ascending, non-empty): the value at rank
/// ceil(p / 100 * n), 1-based. Always an observed value. `percent` is 1..100.
double nearest_rank(std::span<const double> sorted, int percent);

/// Sample standard deviation (n - 1 denominator); nullopt for fewer than 2 values.
std::optional<double> sample_stddev(std::span<const double> values);

/// Statistics of `latencies_ms` (non-empty). Thresholds for p95/p99/stddev follow
/// the schema. Throughput assumes sequential runs of `batch_size` items each.
LatencyStats compute_stats(std::span<const double> latencies_ms, std::int64_t batch_size);

/// Spread of one value per trial (non-empty): the trial means
/// (timing.aggregate.trial_variation) or the trial medians (trial_spread). The
/// field names say "mean" for the schema's trial_variation; for the medians they
/// hold the same statistics of the medians.
TrialVariation compute_trial_variation(std::span<const double> trial_values_ms);

/// Trials whose medians (p50) have a coefficient of variation above this are too
/// spread out to be summarized by one number (warning trial_spread_high). The
/// spread is judged on the medians, not the means: one slow iteration moves a
/// trial's mean (a separate effect, visible in max and p99) but not its median, so
/// the medians show whether the trials agree.
inline constexpr double kHighTrialMedianCv = 0.10;
/// The same when the slowest trial median exceeds the fastest by more than this factor.
inline constexpr double kHighTrialMedianRatio = 1.25;
/// What trial_spread_high is judged on (results schema: timing.aggregate.trial_spread.judged_on).
inline constexpr std::string_view kTrialSpreadStatistic = "trial_p50";

/// True if the per-trial medians in `medians` (compute_trial_variation() of the
/// trial p50s) are spread out by either threshold above (strictly greater).
/// False for a single trial.
bool trial_spread_high(const TrialVariation& medians);

}  // namespace edgebench
