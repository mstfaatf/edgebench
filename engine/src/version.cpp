#include "edgebench/version.hpp"

#include "version_config.hpp"

namespace edgebench {

std::string_view version() noexcept {
    return EDGEBENCH_VERSION_STRING;
}

std::string_view results_schema_version() noexcept {
    return EDGEBENCH_RESULTS_SCHEMA_VERSION_STRING;
}

}  // namespace edgebench
