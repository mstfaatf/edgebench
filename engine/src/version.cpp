#include "edgebench/version.hpp"

#include "build_info.hpp"
#include "version_config.hpp"

#ifndef EDGEBENCH_BUILD_TYPE
#define EDGEBENCH_BUILD_TYPE ""
#endif

namespace edgebench {

std::string_view version() noexcept {
    return EDGEBENCH_VERSION_STRING;
}

std::string_view results_schema_version() noexcept {
    return EDGEBENCH_RESULTS_SCHEMA_VERSION_STRING;
}

const BuildInfo& build_info() {
    static const BuildInfo info = [] {
        BuildInfo b;
        const std::string commit = EDGEBENCH_GIT_COMMIT;
        if (commit.size() == 40)
            b.git_commit = commit;
        constexpr int dirty = EDGEBENCH_GIT_DIRTY;
        if constexpr (dirty >= 0)
            b.git_dirty = dirty == 1;
        b.build_type = EDGEBENCH_BUILD_TYPE;
        b.compiler = EDGEBENCH_COMPILER_STRING;
        return b;
    }();
    return info;
}

}  // namespace edgebench
