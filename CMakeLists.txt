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
include(TargetArchitecture)
include(EdgeBenchWarnings)

# Put all executables in one directory per configuration so the runtime DLLs
# only need to be copied once.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# The ONNX Runtime version the C++ code is built against and accepts at run time
# (generated into engine/src/version_config.hpp) comes from cmake/bundle-pins.json,
# like the versions and SHA256 pins scripts/fetch_deps.ps1 uses. It must match the
# onnxruntime version in requirements.txt, so the C++ engine and the Python
# tooling run the same ONNX Runtime.
include(BundlePins)
# The QNN execution provider (NPU) exists for ARM64 only; x64 builds are CPU-only.
if(EDGEBENCH_TARGET_ARCH STREQUAL "arm64")
  find_package(OnnxRuntime ${EDGEBENCH_ONNXRUNTIME_VERSION} EXACT REQUIRED COMPONENTS QNN)
else()
  find_package(OnnxRuntime ${EDGEBENCH_ONNXRUNTIME_VERSION} EXACT REQUIRED)
endif()
message(STATUS "ONNX Runtime ${OnnxRuntime_VERSION}: ${OnnxRuntime_ROOT}")

# Results schema version the engine writes (generated into engine/src/version_config.hpp).
# Each version has its own schema file, schemas/benchmark-result.v<version>.schema.json.
set(EDGEBENCH_RESULTS_SCHEMA_VERSION 1.2.0)
set(EDGEBENCH_RESULTS_SCHEMA_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/schemas/benchmark-result.v${EDGEBENCH_RESULTS_SCHEMA_VERSION}.schema.json")
if(NOT EXISTS "${EDGEBENCH_RESULTS_SCHEMA_FILE}")
  message(FATAL_ERROR "EDGEBENCH_RESULTS_SCHEMA_VERSION is ${EDGEBENCH_RESULTS_SCHEMA_VERSION}, "
    "but ${EDGEBENCH_RESULTS_SCHEMA_FILE} does not exist")
endif()

# Third-party C++ libraries, pinned by release tag and SHA256 (versions are listed in
# docs/setup.md). SYSTEM makes their headers external, so /W4 /WX does not apply to them.
# Downloaded archives are kept in EDGEBENCH_DOWNLOAD_DIR, outside the build tree,
# so a fresh build directory reuses them (each is re-verified against its hash).
include(FetchContent)
set(EDGEBENCH_DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/_downloads"
  CACHE PATH "Directory for the third-party archives downloaded at configure time")
FetchContent_Declare(cli11
  URL https://github.com/CLIUtils/CLI11/releases/download/v2.7.2/CLI11-2.7.2-Source.tar.gz
  URL_HASH SHA256=0cb0ef44c4c7129ea972505ee81e6d09c71b2e7fe59ca3a13c47d8a994d682b2
  DOWNLOAD_DIR "${EDGEBENCH_DOWNLOAD_DIR}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SYSTEM)
set(CLI11_PRECOMPILED OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(CLI11_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  DOWNLOAD_DIR "${EDGEBENCH_DOWNLOAD_DIR}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SYSTEM)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cli11 nlohmann_json)

# Git commit and dirty flag, refreshed on every build (cmake/BuildInfo.cmake).
find_package(Git QUIET)

if(EDGEBENCH_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
    URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
    DOWNLOAD_DIR "${EDGEBENCH_DOWNLOAD_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)
  # Match the project's dynamic CRT (/MD) and keep gmock/install out of the build.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
  # Python with the packages from requirements.txt: runs the schema tests, and the
  # engine tests use it to validate result documents with scripts/validate_results.py.
  set(EDGEBENCH_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe"
    CACHE FILEPATH "Python interpreter with the packages from requirements.txt, for the schema tests")
endif()

add_subdirectory(engine)
if(EDGEBENCH_BUILD_TOOLS)
  add_subdirectory(tools/ort_probe)
endif()

# Results schema and validator tests (Python, platform-independent; ctest label "schema").
# The same tests run without CMake: python -m pytest scripts/tests/test_validate_results.py
if(EDGEBENCH_BUILD_TESTS)
  if(EXISTS "${EDGEBENCH_PYTHON}")
    add_test(NAME results_schema_tests
      COMMAND "${EDGEBENCH_PYTHON}" -m pytest scripts/tests/test_validate_results.py -q -p no:cacheprovider
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(results_schema_tests PROPERTIES LABELS schema)
  else()
    message(STATUS "EDGEBENCH_PYTHON (${EDGEBENCH_PYTHON}) not found: schema tests not added to CTest")
  endif()
endif()
