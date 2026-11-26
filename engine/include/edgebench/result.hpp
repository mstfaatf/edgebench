// Result<T>: a value or an Error, for operations that can fail in expected ways
// (bad model, ONNX Runtime error, I/O). MSVC's C++20 mode has no std::expected.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "edgebench/exit_codes.hpp"

namespace edgebench {

/// A graph node an execution provider named in an error (results schema:
/// error.unsupported_nodes).
struct NamedNode {
    std::string name;
    std::string op_type;

    bool operator==(const NamedNode&) const = default;
};

struct Error {
    ExitCode exit_code = ExitCode::internal_error;
    std::string kind;     // stable identifier, e.g. "dynamic_shape", "session_creation_failed"
    std::string message;  // human-readable, complete sentence(s)
    std::optional<int> ort_error_code;  // OrtErrorCode value, if ONNX Runtime reported the error
    std::optional<std::string> ort_error_name;  // e.g. "ORT_INVALID_PROTOBUF"
    /// Strict-mode refusals only: the nodes the execution provider said it cannot
    /// run (empty if it named none). Null for every other error.
    std::optional<std::vector<NamedNode>> unsupported_nodes;
};

/// Thrown by Result::value() / error() on the wrong alternative (a programming error).
struct BadResultAccess : std::logic_error {
    using std::logic_error::logic_error;
};

template <typename T>
class [[nodiscard]] Result {
public:
    Result(T value) : data_(std::in_place_index<0>, std::move(value)) {}
    Result(Error error) : data_(std::in_place_index<1>, std::move(error)) {}

    bool ok() const noexcept { return data_.index() == 0; }
    explicit operator bool() const noexcept { return ok(); }

    T& value() & { return check_value(), std::get<0>(data_); }
    const T& value() const& { return check_value(), std::get<0>(data_); }
    T&& value() && { return check_value(), std::get<0>(std::move(data_)); }

    const Error& error() const {
        if (ok())
            throw BadResultAccess("Result::error() on a value");
        return std::get<1>(data_);
    }

private:
    void check_value() const {
        if (!ok())
            throw BadResultAccess("Result::value() on an error: " + std::get<1>(data_).message);
    }
    std::variant<T, Error> data_;
};

template <>
class [[nodiscard]] Result<void> {
public:
    Result() = default;
    Result(Error error) : error_(std::move(error)) {}

    bool ok() const noexcept { return !error_.has_value(); }
    explicit operator bool() const noexcept { return ok(); }

    const Error& error() const {
        if (ok())
            throw BadResultAccess("Result::error() on success");
        return *error_;
    }

private:
    std::optional<Error> error_;
};

/// Name of an OrtErrorCode value, e.g. "ORT_FAIL"; "ORT_UNKNOWN_ERROR_CODE" if unknown.
const char* ort_error_code_name(int code) noexcept;

/// An Error for an ONNX Runtime failure (exit code ort_runtime_error). The message is
/// "<context>: ONNX Runtime error <NAME> (code <n>): <ORT message>".
Error ort_error(std::string kind, const std::string& context, int ort_code,
                const std::string& ort_message);

}  // namespace edgebench
