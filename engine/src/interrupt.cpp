#ifndef _WIN32
#error "interrupt.cpp supports Windows only: it uses SetConsoleCtrlHandler."
#endif

#include "edgebench/interrupt.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "edgebench/exit_codes.hpp"

namespace edgebench {

namespace {

std::atomic<bool> g_stop{false};
std::atomic<DWORD> g_event_type{0xFFFFFFFF};  // console event that raised the flag
std::atomic<InterruptMode> g_mode{InterruptMode::stop_gracefully};
HANDLE g_stop_event = nullptr;  // manual-reset: signalled while a stop is requested
HANDLE g_done_event = nullptr;  // manual-reset: signalled once the result is written
std::once_flag g_init;

void init_events() {
    std::call_once(g_init, [] {
        g_stop_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        g_done_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    });
}

constexpr DWORD kCloseWaitMs = 4000;  // Windows terminates ~5 s after a close event

BOOL WINAPI console_handler(DWORD type) {
    switch (type) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
            if (g_mode.load() == InterruptMode::exit_immediately || g_stop.exchange(true)) {
                // Nothing to save, or a second interrupt: end now, without writing.
                TerminateProcess(GetCurrentProcess(), to_int(ExitCode::interrupted));
            }
            g_event_type.store(type);
            SetEvent(g_stop_event);
            return TRUE;
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            if (g_mode.load() == InterruptMode::exit_immediately)
                TerminateProcess(GetCurrentProcess(), to_int(ExitCode::interrupted));
            if (!g_stop.exchange(true)) {
                g_event_type.store(type);
                SetEvent(g_stop_event);
            }
            // Returning lets Windows terminate the process: give the main thread
            // time to write the partial result first.
            WaitForSingleObject(g_done_event, kCloseWaitMs);
            return TRUE;
        default:
            return FALSE;
    }
}

std::mutex g_reason_mutex;
std::string g_reason;  // for request_stop()

}  // namespace

void install_interrupt_handler(InterruptMode mode) {
    init_events();
    g_mode.store(mode);
    static std::once_flag installed;
    std::call_once(installed, [] { SetConsoleCtrlHandler(&console_handler, TRUE); });
}

bool stop_requested() noexcept {
    return g_stop.load(std::memory_order_relaxed);
}

std::string stop_reason() {
    if (!stop_requested())
        return {};
    switch (g_event_type.load()) {
        case CTRL_C_EVENT:
            return "Ctrl+C";
        case CTRL_BREAK_EVENT:
            return "Ctrl+Break";
        case CTRL_CLOSE_EVENT:
            return "the console window was closed";
        case CTRL_LOGOFF_EVENT:
            return "the user logged off";
        case CTRL_SHUTDOWN_EVENT:
            return "the system is shutting down";
        default: {
            const std::lock_guard<std::mutex> lock(g_reason_mutex);
            return g_reason.empty() ? std::string("a stop request") : g_reason;
        }
    }
}

void request_stop(std::string_view reason) {
    init_events();
    {
        const std::lock_guard<std::mutex> lock(g_reason_mutex);
        g_reason = std::string(reason);
    }
    g_event_type.store(0xFFFFFFFF);
    g_stop.store(true);
    SetEvent(g_stop_event);
}

void reset_stop() noexcept {
    init_events();
    g_stop.store(false);
    g_event_type.store(0xFFFFFFFF);
    ResetEvent(g_stop_event);
    ResetEvent(g_done_event);
}

bool wait_or_stop(std::chrono::milliseconds duration) {
    init_events();
    // A timed wait can end up to a timer tick early: wait again until the
    // deadline, so an idle period is never shorter than requested.
    const auto deadline = std::chrono::steady_clock::now() + duration;
    for (;;) {
        if (stop_requested())
            return true;
        const auto left = std::chrono::ceil<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (left.count() <= 0)
            return false;
        const DWORD ms = static_cast<DWORD>(std::min<long long>(left.count(), 0x7FFFFFFF));
        if (WaitForSingleObject(g_stop_event, ms) == WAIT_OBJECT_0)
            return true;
    }
}

void notify_shutdown_complete() noexcept {
    init_events();
    SetEvent(g_done_event);
}

}  // namespace edgebench
