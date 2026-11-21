cmake_minimum_required(VERSION 3.25)

project(edgebench
  VERSION 0.1.0
  DESCRIPTION "CPU vs NPU inference benchmarking for Windows on Snapdragon"
  HOMEPAGE_URL "https://github.com/mstfaatf/edgebench"
  LANGUAGES CXX)

option(EDGEBENCH_BUILD_TESTS "Build the EdgeBench unit tests" ON)
option(EDGEBENCH_BUILD_TOOLS "Build the diagnostic tools in tools/ (ort_probe)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(RequireNativeArm64)
include(EdgeBenchWarnings)

# Put all executables in one directory per configuration so the runtime DLLs
# only need to be copied once.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Must match the onnxruntime version in requirements.txt (and scripts/fetch_deps.ps1),
# so the C++ engine and the Python tooling run the same ONNX Runtime.
find_package(OnnxRuntime 1.32.0 EXACT REQUIRED COMPONENTS QNN)
message(STATUS "ONNX Runtime ${OnnxRuntime_VERSION}: ${OnnxRuntime_ROOT}")

if(EDGEBENCH_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
    URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)
  # Match the project's dynamic CRT (/MD) and keep gmock/install out of the build.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
endif()

add_subdirectory(engine)
if(EDGEBENCH_BUILD_TOOLS)
  add_subdirectory(tools/ort_probe)
endif()
