cmake_minimum_required(VERSION 3.25)

project(edgebench
  VERSION 0.1.0
  DESCRIPTION "CPU vs NPU inference benchmarking for Windows on Snapdragon"
  HOMEPAGE_URL "https://github.com/mstfaatf/edgebench"
  LANGUAGES CXX)

option(EDGEBENCH_BUILD_TESTS "Build the EdgeBench unit tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(RequireNativeArm64)

# Put all executables in one directory per configuration so the runtime DLLs
# only need to be copied once.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(OnnxRuntime 1.32.0 REQUIRED COMPONENTS QNN)
message(STATUS "ONNX Runtime ${OnnxRuntime_VERSION}: ${OnnxRuntime_ROOT}")

if(EDGEBENCH_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(engine)
