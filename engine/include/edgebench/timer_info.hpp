// The clock used for latency measurement (results schema: timing.timer).
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace edgebench {

struct TimerInfo {
    std::string source;                   // e.g. "std::chrono::steady_clock"
    std::optional<double> resolution_ns;  // tick period actually delivered
};

/// The effective tick period of a clock that reports a nominal period of
/// `period_num / period_den` seconds and is backed by a counter running at
/// `counter_hz` (0 if unknown): the coarser of the two, in nanoseconds.
double effective_resolution_ns(std::intmax_t period_num, std::intmax_t period_den,
                               std::int64_t counter_hz);

/// std::chrono::steady_clock, with its resolution verified against
/// QueryPerformanceFrequency (MSVC implements steady_clock on the performance
/// counter but reports a nominal 1 ns period).
TimerInfo steady_clock_info();

}  // namespace edgebench
