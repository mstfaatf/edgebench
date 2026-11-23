// Test helper for process-level behaviour that cannot be checked in-process.
//
//   signal_helper crash      installs the crash handler, then writes through a null pointer
//   signal_helper interrupt  installs the graceful interrupt handler and keeps running after
//                            the first stop request (prints "stop requested"), so a second
//                            Ctrl+Break must end the process with exit code 130
//
// Prints "ready" on stderr once the handler is installed.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include "edgebench/crash_handler.hpp"
#include "edgebench/interrupt.hpp"

namespace {

void say(const char* text) {
    std::fputs(text, stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "crash") == 0) {
        edgebench::install_crash_handler();
        say("ready");
        volatile int* p = nullptr;
        *p = 1;
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "interrupt") == 0) {
        edgebench::install_interrupt_handler(edgebench::InterruptMode::stop_gracefully);
        say("ready");
        // The stop event stays signalled, so after the first request wait_or_stop()
        // returns at once: keep running on the clock, at most 30 s in total.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        bool reported = false;
        while (std::chrono::steady_clock::now() < deadline) {
            if (!edgebench::wait_or_stop(std::chrono::milliseconds(50)))
                continue;
            if (!reported) {
                say("stop requested");
                reported = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return 0;
    }
    say("usage: signal_helper crash|interrupt");
    return 2;
}
