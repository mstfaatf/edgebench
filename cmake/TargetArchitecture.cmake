# Sets EDGEBENCH_TARGET_ARCH (arm64 or x64) from the C++ compiler's target and
# fails configuration for anything else.
#
#   arm64  The measurement build: native ARM64, with the QNN execution provider
#          for the NPU.
#   x64    CPU-only build for continuous integration and development on x64
#          machines. It has no QNN execution provider, and on Windows on Arm it
#          runs under emulation, so its timings do not describe the device
#          (`edgebench doctor` warns about emulation).
#
# ARM64EC is rejected: it exists to host x64 code and is neither of the above.

if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
  set(EDGEBENCH_TARGET_ARCH arm64)
elseif(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
  set(EDGEBENCH_TARGET_ARCH x64)
else()
  message(FATAL_ERROR
    "EdgeBench is built for ARM64 or x64, but the compiler targets "
    "'${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}'. Configure with one of the "
    "arm64-* or x64-* presets (e.g. cmake --preset arm64-release). See docs/setup.md.")
endif()
message(STATUS "EdgeBench target architecture: ${EDGEBENCH_TARGET_ARCH}")
