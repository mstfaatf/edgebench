// The one format for errors and warnings on stderr (docs/cli.md):
//
//   error: [<kind>] <message>
//   hint: <what to do>            (optional, one line)
//   warning: [<code>] <message>
//
// <kind> and <code> are stable identifiers (error kinds, warning codes).
#pragma once

#include <string>
#include <string_view>

#include "edgebench/result.hpp"

namespace edgebench {

/// "error: [kind] message\n", plus "hint: ...\n" if `hint` is not empty.
std::string format_error(std::string_view kind, std::string_view message,
                         std::string_view hint = {});

/// "warning: [code] message\n", plus "hint: ...\n" if `hint` is not empty.
std::string format_warning(std::string_view code, std::string_view message,
                           std::string_view hint = {});

/// Writes format_error() / format_warning() to stderr and flushes it.
void print_error(std::string_view kind, std::string_view message, std::string_view hint = {});
void print_error(const Error& error, std::string_view hint = {});
void print_warning(std::string_view code, std::string_view message, std::string_view hint = {});

}  // namespace edgebench
