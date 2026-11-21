#include "edgebench/timer_info.hpp"

#include <algorithm>
#include <chrono>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace edgebench {

double effective_resolution_ns(std::intmax_t period_num, std::intmax_t period_den,
                               std::int64_t counter_hz) {
    const double nominal = 1e9 * static_cast<double>(period_num) / static_cast<double>(period_den);
    const double counter = counter_hz > 0 ? 1e9 / static_cast<double>(counter_hz) : 0.0;
    return std::max(nominal, counter);
}

TimerInfo steady_clock_info() {
    using period = std::chrono::steady_clock::period;
    std::int64_t hz = 0;
#ifdef _WIN32
    LARGE_INTEGER f{};
    if (QueryPerformanceFrequency(&f))
        hz = f.QuadPart;
#endif
    return {"std::chrono::steady_clock", effective_resolution_ns(period::num, period::den, hz)};
}

}  // namespace edgebench
