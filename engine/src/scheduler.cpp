#include "edgebench/scheduler.hpp"

namespace edgebench {

std::vector<ScheduledTrial> schedule_trials(std::size_t config_count, int trials_per_config) {
    std::vector<ScheduledTrial> order;
    if (config_count == 0 || trials_per_config <= 0)
        return order;
    order.reserve(config_count * static_cast<std::size_t>(trials_per_config));
    for (int round = 0; round < trials_per_config; ++round) {
        const std::size_t start = static_cast<std::size_t>(round) % config_count;
        for (std::size_t k = 0; k < config_count; ++k) {
            order.push_back({(start + k) % config_count, round, static_cast<int>(order.size())});
        }
    }
    return order;
}

}  // namespace edgebench
