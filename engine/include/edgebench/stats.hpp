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

// ---------------------------------------------------------------- spikes
//
// A spike is a measured iteration slower than kSpikeFactor times its trial's
// median, and at least kSpikeFloorMs slower than the median. The floor keeps
// sub-millisecond jitter of very fast models (an interrupt or a cache miss adds
// up to about 0.15 ms to a 0.05 ms inference) from counting as a spike: on the
// local CPU/NPU matrix, no autoencoder trial (medians 0.04 to 0.25 ms) had an
// iteration more than 0.15 ms above its median, while every isolated outlier of
// the image models was 1.3 ms or more above it. The spike statistics only
// report: medians, percentiles and means keep every sample.

inline constexpr double kSpikeFactor = 2.0;
inline constexpr double kSpikeFloorMs = 0.25;
/// latency_spikes_high: the spikes move the mean by more than this fraction ...
inline constexpr double kHighSpikeMeanShift = 0.05;
/// ... or more than this fraction of the iterations are spikes. On the local
/// matrix, trials with real outliers had 1 to 7 % of them; 2 % needs more than
/// two spikes in a 100-iteration trial.
inline constexpr double kHighSpikeRate = 0.02;

/// timing.trials[].spikes (one trial) or timing.aggregate.spikes (pooled).
struct SpikeStats {
    std::size_t iterations = 0;
    std::size_t count = 0;
    std::optional<double> threshold_ms;  // one trial only: max(k x median, median + floor)
    double rate = 0;                     // count / iterations
    double max_to_median_ratio = 0;      // slowest iteration over its trial's median
    double mean_ms = 0;
    double mean_without_spikes_ms = 0;
    double mean_shift_ratio = 0;  // mean / mean_without_spikes - 1
};

/// The spikes of one trial's latencies (non-empty, in any order).
SpikeStats compute_spikes(std::span<const double> latencies_ms);

/// The spikes of several trials pooled, each judged against its own median.
SpikeStats combine_spikes(std::span<const std::vector<double>> trials_latencies_ms);

/// True if `aggregate` exceeds kHighSpikeMeanShift or kHighSpikeRate (strictly).
bool spikes_high(const SpikeStats& aggregate);

}  // namespace edgebench
