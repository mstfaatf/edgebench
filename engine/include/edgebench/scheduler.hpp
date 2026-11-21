// Trial order for one or more benchmark configurations.
//
// Trials run in rounds; every round runs each configuration once, and round r
// starts at configuration r mod N. Over N rounds every configuration runs first
// once and follows each other configuration equally often, so warm-up effects and
// thermal drift are not always charged to the same configuration.
//
//   N = 3, 3 trials:  0 1 2 | 1 2 0 | 2 0 1
#pragma once

#include <cstddef>
#include <vector>

namespace edgebench {

struct ScheduledTrial {
    std::size_t config_index = 0;  // which configuration runs
    int trial_index = 0;           // trial number of that configuration (0-based, = round)
    int sequence = 0;              // position in the overall order (0-based)
};

/// The order of `trials_per_config` trials of each of `config_count`
/// configurations; empty if either count is 0.
std::vector<ScheduledTrial> schedule_trials(std::size_t config_count, int trials_per_config);

}  // namespace edgebench
