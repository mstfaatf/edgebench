#include "edgebench/diagnostics.hpp"

#include <cstdio>

namespace edgebench {

namespace {

std::string format_line(std::string_view level, std::string_view id, std::string_view message,
                        std::string_view hint) {
    std::string out;
    out.reserve(level.size() + id.size() + message.size() + hint.size() + 16);
    out.append(level).append(": [").append(id.empty() ? std::string_view("unknown") : id);
    out.append("] ").append(message).append("\n");
    if (!hint.empty())
        out.append("hint: ").append(hint).append("\n");
    return out;
}

void write_stderr(const std::string& text) {
    std::fflush(stdout);  // keep stdout and stderr in order on a shared console
    std::fputs(text.c_str(), stderr);
    std::fflush(stderr);
}

}  // namespace

std::string format_error(std::string_view kind, std::string_view message, std::string_view hint) {
    return format_line("error", kind, message, hint);
}

std::string format_warning(std::string_view code, std::string_view message,
                           std::string_view hint) {
    return format_line("warning", code, message, hint);
}

void print_error(std::string_view kind, std::string_view message, std::string_view hint) {
    write_stderr(format_error(kind, message, hint));
}

void print_error(const Error& error, std::string_view hint) {
    print_error(error.kind, error.message, hint);
}

void print_warning(std::string_view code, std::string_view message, std::string_view hint) {
    write_stderr(format_warning(code, message, hint));
}

}  // namespace edgebench
