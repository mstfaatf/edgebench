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

# The ONNX Runtime version the C++ code is built against and accepts at run time
# (generated into engine/src/version_config.hpp). Must match the onnxruntime
# version in requirements.txt (and scripts/fetch_deps.ps1), so the C++ engine and
# the Python tooling run the same ONNX Runtime.
set(EDGEBENCH_ONNXRUNTIME_VERSION 1.32.0)
find_package(OnnxRuntime ${EDGEBENCH_ONNXRUNTIME_VERSION} EXACT REQUIRED COMPONENTS QNN)
message(STATUS "ONNX Runtime ${OnnxRuntime_VERSION}: ${OnnxRuntime_ROOT}")

# Results schema version the engine writes (generated into engine/src/version_config.hpp).
# Each version has its own schema file, schemas/benchmark-result.v<version>.schema.json.
set(EDGEBENCH_RESULTS_SCHEMA_VERSION 1.1.0)
set(EDGEBENCH_RESULTS_SCHEMA_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/schemas/benchmark-result.v${EDGEBENCH_RESULTS_SCHEMA_VERSION}.schema.json")
if(NOT EXISTS "${EDGEBENCH_RESULTS_SCHEMA_FILE}")
  message(FATAL_ERROR "EDGEBENCH_RESULTS_SCHEMA_VERSION is ${EDGEBENCH_RESULTS_SCHEMA_VERSION}, "
    "but ${EDGEBENCH_RESULTS_SCHEMA_FILE} does not exist")
endif()

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

# Results schema and validator tests (Python, platform-independent; ctest label "schema").
# The same tests run without CMake: python -m pytest scripts/tests/test_validate_results.py
if(EDGEBENCH_BUILD_TESTS)
  set(EDGEBENCH_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe"
    CACHE FILEPATH "Python interpreter with the packages from requirements.txt, for the schema tests")
  if(EXISTS "${EDGEBENCH_PYTHON}")
    add_test(NAME results_schema_tests
      COMMAND "${EDGEBENCH_PYTHON}" -m pytest scripts/tests/test_validate_results.py -q -p no:cacheprovider
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(results_schema_tests PROPERTIES LABELS schema)
  else()
    message(STATUS "EDGEBENCH_PYTHON (${EDGEBENCH_PYTHON}) not found: schema tests not added to CTest")
  endif()
endif()
