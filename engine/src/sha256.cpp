#ifndef _WIN32
#error "sha256.cpp supports Windows only: it uses the BCrypt (CNG) API."
#endif

#include "edgebench/sha256.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// windows.h first
#include <bcrypt.h>

#include "edgebench/ort_loader.hpp"

namespace edgebench {

namespace {

// Incremental SHA-256 over the CNG provider; throws std::runtime_error on API failure.
class Sha256 {
public:
    Sha256() {
        check(BCryptOpenAlgorithmProvider(&alg_, BCRYPT_SHA256_ALGORITHM, nullptr, 0), "open");
        check(BCryptCreateHash(alg_, &hash_, nullptr, 0, nullptr, 0, 0), "create");
    }
    ~Sha256() {
        if (hash_)
            BCryptDestroyHash(hash_);
        if (alg_)
            BCryptCloseAlgorithmProvider(alg_, 0);
    }
    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;

    void update(const void* data, size_t size) {
        auto* p = static_cast<PUCHAR>(const_cast<void*>(data));
        while (size > 0) {
            const ULONG n = static_cast<ULONG>(std::min<size_t>(size, 1u << 30));
            check(BCryptHashData(hash_, p, n, 0), "update");
            p += n;
            size -= n;
        }
    }

    std::string hex() {
        std::array<UCHAR, 32> digest{};
        check(BCryptFinishHash(hash_, digest.data(), static_cast<ULONG>(digest.size()), 0),
              "finish");
        static constexpr char kHex[] = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (UCHAR b : digest) {
            out += kHex[b >> 4];
            out += kHex[b & 0xF];
        }
        return out;
    }

private:
    static void check(NTSTATUS status, const char* what) {
        if (!BCRYPT_SUCCESS(status))
            throw std::runtime_error(std::string("SHA-256 (BCrypt) ") + what + " failed");
    }
    BCRYPT_ALG_HANDLE alg_ = nullptr;
    BCRYPT_HASH_HANDLE hash_ = nullptr;
};

}  // namespace

std::string sha256_hex(std::span<const std::byte> data) {
    Sha256 h;
    h.update(data.data(), data.size());
    return h.hex();
}

std::string sha256_hex(std::string_view text) {
    Sha256 h;
    h.update(text.data(), text.size());
    return h.hex();
}

Result<std::string> sha256_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Error{ExitCode::usage_error, "file_read_failed",
                     "cannot open " + path_to_utf8(path) + " for reading"};
    }
    Sha256 h;
    std::vector<char> buf(1 << 20);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize got = in.gcount();
        if (got > 0)
            h.update(buf.data(), static_cast<size_t>(got));
    }
    if (in.bad()) {
        return Error{ExitCode::usage_error, "file_read_failed",
                     "error while reading " + path_to_utf8(path)};
    }
    return h.hex();
}

}  // namespace edgebench
