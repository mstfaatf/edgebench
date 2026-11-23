// Ctrl+C, Ctrl+Break and console-close handling.
//
// The console control handler only sets an atomic stop flag and signals an
// event; the program checks the flag at safe points (between iterations, around
// session creation, during idle periods) and stops there. ONNX Runtime calls in
// progress are not interrupted. A second Ctrl+C or Ctrl+Break ends the process
// immediately with exit code 130. On console close, logoff or shutdown, Windows
// kills the process about 5 s after the handler returns, so the handler waits up
// to about 4 s for notify_shutdown_complete() first.
#pragma once

#include <chrono>
#include <string>
#include <string_view>

namespace edgebench {

enum class InterruptMode {
    /// Set the stop flag and let the program stop and write a partial result.
    stop_gracefully,
    /// Exit with code 130 at once (for tools that have nothing to save).
    exit_immediately,
};

/// Installs the console control handler (once per process).
void install_interrupt_handler(InterruptMode mode);

/// True once Ctrl+C, Ctrl+Break or a close/logoff/shutdown event was received
/// (or request_stop() was called).
bool stop_requested() noexcept;

/// What stopped the run, e.g. "Ctrl+C" or "the console was closed"; empty if none.
std::string stop_reason();

/// Raises the stop flag as if a Ctrl+C had arrived (used by tests).
void request_stop(std::string_view reason = "a stop request");

/// Clears the stop flag (tests only; the handler never clears it).
void reset_stop() noexcept;

/// Waits for `duration` or until a stop is requested, whichever comes first.
/// Returns true if it was cut short by a stop request.
bool wait_or_stop(std::chrono::milliseconds duration);

/// Tells a waiting close/logoff/shutdown handler that the result has been written.
void notify_shutdown_complete() noexcept;

}  // namespace edgebench
