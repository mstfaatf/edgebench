#include "edgebench/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace edgebench {

namespace {

constexpr std::size_t kMinCountP95 = 20;
constexpr std::size_t kMinCountP99 = 100;

double mean(std::span<const double> v) {
    double sum = 0;
    for (double x : v)
        sum += x;
    return sum / static_cast<double>(v.size());
}

}  // namespace

double nearest_rank(std::span<const double> sorted, int percent) {
    if (sorted.empty())
        throw std::invalid_argument("nearest_rank: no values");
    if (percent < 1 || percent > 100)
        throw std::invalid_argument("nearest_rank: percent must be 1..100");
    // ceil(percent * n / 100) in integer arithmetic, so no rounding can shift the rank.
    const std::size_t n = sorted.size();
    std::size_t rank = (static_cast<std::size_t>(percent) * n + 99) / 100;
    rank = std::clamp<std::size_t>(rank, 1, n);
    return sorted[rank - 1];
}

std::optional<double> sample_stddev(std::span<const double> values) {
    if (values.size() < 2)
        return std::nullopt;
    const double m = mean(values);
    double ss = 0;  // two-pass: sum of squared deviations from the mean
    for (double x : values)
        ss += (x - m) * (x - m);
    return std::sqrt(ss / static_cast<double>(values.size() - 1));
}

LatencyStats compute_stats(std::span<const double> latencies_ms, std::int64_t batch_size) {
    if (latencies_ms.empty())
        throw std::invalid_argument("compute_stats: no latencies");
    std::vector<double> sorted(latencies_ms.begin(), latencies_ms.end());
    std::sort(sorted.begin(), sorted.end());

    LatencyStats s;
    s.count = sorted.size();
    s.mean_ms = mean(sorted);
    s.p50_ms = nearest_rank(sorted, 50);
    if (s.count >= kMinCountP95)
        s.p95_ms = nearest_rank(sorted, 95);
    if (s.count >= kMinCountP99)
        s.p99_ms = nearest_rank(sorted, 99);
    s.stddev_ms = sample_stddev(sorted);
    s.min_ms = sorted.front();
    s.max_ms = sorted.back();
    s.inferences_per_sec = s.mean_ms > 0 ? 1000.0 / s.mean_ms : 0.0;
    s.items_per_sec = s.inferences_per_sec * static_cast<double>(batch_size);
    return s;
}

TrialVariation compute_trial_variation(std::span<const double> trial_values_ms) {
    if (trial_values_ms.empty())
        throw std::invalid_argument("compute_trial_variation: no trials");
    TrialVariation v;
    const auto [lo, hi] = std::minmax_element(trial_values_ms.begin(), trial_values_ms.end());
    v.trial_mean_min_ms = *lo;
    v.trial_mean_max_ms = *hi;
    v.trial_mean_stddev_ms = sample_stddev(trial_values_ms);
    if (v.trial_mean_stddev_ms) {
        const double m = mean(trial_values_ms);
        v.trial_mean_cv_ratio = m > 0 ? *v.trial_mean_stddev_ms / m : 0.0;
    }
    return v;
}

namespace {

struct SpikeSums {
    std::size_t iterations = 0, count = 0;
    double sum = 0, sum_without = 0, max_ratio = 0;
};

// Adds one trial: the sum over the sorted values (as compute_stats() sums them).
double add_trial(SpikeSums& s, std::span<const double> latencies_ms) {
    if (latencies_ms.empty())
        throw std::invalid_argument("compute_spikes: no latencies");
    std::vector<double> sorted(latencies_ms.begin(), latencies_ms.end());
    std::sort(sorted.begin(), sorted.end());
    const double median = nearest_rank(sorted, 50);
    const double threshold = std::max(kSpikeFactor * median, median + kSpikeFloorMs);
    for (double x : sorted) {
        s.sum += x;
        if (x > threshold)
            ++s.count;
        else
            s.sum_without += x;
    }
    s.iterations += sorted.size();
    if (median > 0)
        s.max_ratio = std::max(s.max_ratio, sorted.back() / median);
    return threshold;
}

SpikeStats finish(const SpikeSums& s) {
    SpikeStats r;
    r.iterations = s.iterations;
    r.count = s.count;
    r.rate =
        s.iterations > 0 ? static_cast<double>(s.count) / static_cast<double>(s.iterations) : 0.0;
    r.max_to_median_ratio = s.max_ratio;
    r.mean_ms = s.sum / static_cast<double>(s.iterations);
    // At least half of a trial is at or below its median, so never empty.
    r.mean_without_spikes_ms = s.sum_without / static_cast<double>(s.iterations - s.count);
    r.mean_shift_ratio =
        r.mean_without_spikes_ms > 0 ? r.mean_ms / r.mean_without_spikes_ms - 1.0 : 0.0;
    return r;
}

}  // namespace

SpikeStats compute_spikes(std::span<const double> latencies_ms) {
    SpikeSums s;
    const double threshold = add_trial(s, latencies_ms);
    SpikeStats r = finish(s);
    r.threshold_ms = threshold;
    return r;
}

SpikeStats combine_spikes(std::span<const std::vector<double>> trials) {
    if (trials.empty())
        throw std::invalid_argument("combine_spikes: no trials");
    SpikeSums s;
    for (const std::vector<double>& t : trials)
        add_trial(s, t);
    return finish(s);
}

bool spikes_high(const SpikeStats& a) {
    return a.mean_shift_ratio > kHighSpikeMeanShift || a.rate > kHighSpikeRate;
}

bool trial_spread_high(const TrialVariation& v) {
    if (v.trial_mean_cv_ratio && *v.trial_mean_cv_ratio > kHighTrialMedianCv)
        return true;
    return v.trial_mean_min_ms > 0 &&
           v.trial_mean_max_ms > kHighTrialMedianRatio * v.trial_mean_min_ms;
}

}  // namespace edgebench
