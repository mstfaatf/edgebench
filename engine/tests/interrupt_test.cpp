// The stop flag and the interruptible wait (no console events: those are tested
// on child processes in exit_paths_test.cpp).

#include "edgebench/interrupt.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace eb = edgebench;
using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

namespace {

struct StopGuard {
    StopGuard() { eb::reset_stop(); }
    ~StopGuard() { eb::reset_stop(); }
};

}  // namespace

TEST(StopFlag, RequestAndReset) {
    StopGuard guard;
    EXPECT_FALSE(eb::stop_requested());
    eb::request_stop("the test");
    EXPECT_TRUE(eb::stop_requested());
    EXPECT_EQ(eb::stop_reason(), "the test");
    eb::reset_stop();
    EXPECT_FALSE(eb::stop_requested());
}

TEST(WaitOrStop, NeverEndsEarly) {
    StopGuard guard;
    for (int i = 0; i < 20; ++i) {
        const auto t0 = Clock::now();
        EXPECT_FALSE(eb::wait_or_stop(milliseconds(20)));
        EXPECT_GE(Clock::now() - t0, milliseconds(20)) << "iteration " << i;
    }
    EXPECT_FALSE(eb::wait_or_stop(milliseconds(0)));
}

TEST(WaitOrStop, EndsAtOnceOnAStopRequest) {
    StopGuard guard;
    std::thread stopper([] {
        std::this_thread::sleep_for(milliseconds(50));
        eb::request_stop("the test");
    });
    const auto t0 = Clock::now();
    EXPECT_TRUE(eb::wait_or_stop(std::chrono::seconds(30)));
    stopper.join();
    EXPECT_LT(Clock::now() - t0, std::chrono::seconds(5));
    // Once stopped, every further wait returns immediately.
    EXPECT_TRUE(eb::wait_or_stop(std::chrono::seconds(30)));
}
