# Fail configuration unless the C++ compiler targets native ARM64.
#
# ARM64EC and x64 builds would run under emulation on Windows on Arm, which is
# not what EdgeBench measures.

if(NOT CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
  message(FATAL_ERROR
    "EdgeBench must be built for native ARM64, but the compiler targets "
    "'${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}'. Configure with one of the "
    "arm64-* presets (e.g. cmake --preset arm64-release) or from an ARM64 "
    "developer prompt.")
endif()
