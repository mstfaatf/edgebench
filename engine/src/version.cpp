#include "edgebench/version.hpp"

#include "version_config.hpp"

namespace edgebench {

std::string_view version() noexcept {
    return EDGEBENCH_VERSION_STRING;
}

}  // namespace edgebench
