// Deterministic, platform-independent input data (documented in docs/cli.md).
//
// Random inputs use SplitMix64 and a fixed bit mapping to float32, so the same
// seed gives bit-identical inputs with any compiler, standard library or CPU
// architecture (unlike std::uniform_real_distribution).
#pragma once

#include <cstdint>
#include <span>

namespace edgebench {

/// SplitMix64 (Steele, Lea and Flood, 2014). The state starts at the seed; each
/// call adds 0x9E3779B97F4A7C15 to the state and returns its mixed value.
class SplitMix64 {
public:
    explicit constexpr SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

    constexpr std::uint64_t next() noexcept {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    /// Uniform float32 in [0, 1): the top 24 bits of next() times 2^-24. Every
    /// result is exactly representable, so there is no rounding to differ.
    constexpr float next_float() noexcept {
        return static_cast<float>(next() >> 40) * (1.0f / 16777216.0f);
    }

private:
    std::uint64_t state_;
};

/// Fills `data` with consecutive next_float() values of `rng`, in order.
void fill_random(std::span<float> data, SplitMix64& rng) noexcept;

/// Fills `data` with `value`.
void fill_constant(std::span<float> data, float value) noexcept;

}  // namespace edgebench
