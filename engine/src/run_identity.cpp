#ifndef _WIN32
#error "run_identity.cpp supports Windows only: it uses BCryptGenRandom."
#endif

#include "edgebench/run_identity.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// windows.h first
#include <bcrypt.h>

namespace edgebench {

std::string new_run_id() {
    std::array<unsigned char, 16> b{};
    if (!BCRYPT_SUCCESS(BCryptGenRandom(nullptr, b.data(), static_cast<ULONG>(b.size()),
                                        BCRYPT_USE_SYSTEM_PREFERRED_RNG))) {
        throw std::runtime_error("BCryptGenRandom failed");
    }
    b[6] = static_cast<unsigned char>((b[6] & 0x0F) | 0x40);  // version 4
    b[8] = static_cast<unsigned char>((b[8] & 0x3F) | 0x80);  // RFC 4122 variant
    char out[37];
    std::snprintf(out, sizeof(out),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0],
                  b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13],
                  b[14], b[15]);
    return out;
}

std::string utc_timestamp(std::chrono::system_clock::time_point t) {
    using namespace std::chrono;
    const auto ms = duration_cast<milliseconds>(t.time_since_epoch());
    const sys_days day = floor<days>(sys_time<milliseconds>(ms));
    const year_month_day ymd(day);
    const hh_mm_ss<milliseconds> tod(ms - day.time_since_epoch());
    char out[32];
    std::snprintf(out, sizeof(out), "%04d-%02u-%02uT%02d:%02d:%02d.%03dZ",
                  static_cast<int>(ymd.year()), static_cast<unsigned>(ymd.month()),
                  static_cast<unsigned>(ymd.day()), static_cast<int>(tod.hours().count()),
                  static_cast<int>(tod.minutes().count()), static_cast<int>(tod.seconds().count()),
                  static_cast<int>(tod.subseconds().count()));
    return out;
}

}  // namespace edgebench
