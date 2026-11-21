#include "edgebench/input_fill.hpp"

#include <algorithm>

namespace edgebench {

void fill_random(std::span<float> data, SplitMix64& rng) noexcept {
    for (float& x : data)
        x = rng.next_float();
}

void fill_constant(std::span<float> data, float value) noexcept {
    std::fill(data.begin(), data.end(), value);
}

}  // namespace edgebench
