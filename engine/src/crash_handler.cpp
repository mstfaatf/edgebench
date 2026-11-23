#ifndef _WIN32
#error "crash_handler.cpp supports Windows only: it uses SetUnhandledExceptionFilter."
#endif

#include "edgebench/crash_handler.hpp"

#include <cstdio>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "edgebench/exit_codes.hpp"

namespace edgebench {

namespace {

LONG WINAPI unhandled_exception_filter(EXCEPTION_POINTERS* info) {
    // The process state is unknown here: format into a stack buffer and write it
    // with WriteFile, without touching the heap or the C runtime's stdio locks.
    const DWORD code = info && info->ExceptionRecord ? info->ExceptionRecord->ExceptionCode : 0;
    const void* address = info && info->ExceptionRecord ? info->ExceptionRecord->ExceptionAddress
                                                        : nullptr;
    char line[160];
    const int n = std::snprintf(line, sizeof(line),
                                "error: [internal_error] unhandled exception 0x%08lX at %p\n",
                                static_cast<unsigned long>(code), address);
    if (n > 0) {
        DWORD written = 0;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), line, static_cast<DWORD>(n), &written, nullptr);
    }
    TerminateProcess(GetCurrentProcess(), to_int(ExitCode::internal_error));
    return EXCEPTION_EXECUTE_HANDLER;
}

}  // namespace

void install_crash_handler() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    SetUnhandledExceptionFilter(&unhandled_exception_filter);
}

}  // namespace edgebench
