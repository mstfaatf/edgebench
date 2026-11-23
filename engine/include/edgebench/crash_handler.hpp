// Last-resort handling of unhandled structured exceptions (access violations
// and the like): print one error line and exit with code 1, instead of the
// Windows crash dialog. No minidump is written.
#pragma once

namespace edgebench {

/// Installs the unhandled exception filter and turns off the Windows error
/// dialogs for this process. Call once, first thing in main.
void install_crash_handler();

}  // namespace edgebench
