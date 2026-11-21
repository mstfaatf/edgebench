# edgebench_set_warnings() comes from cmake/EdgeBenchWarnings.cmake; GoogleTest is
# made available by the root CMakeLists.txt when EDGEBENCH_BUILD_TESTS is on.

# --- Core library -----------------------------------------------------------

configure_file(src/version_config.hpp.in
  "${CMAKE_CURRENT_BINARY_DIR}/generated/version_config.hpp" @ONLY)

add_library(edgebench_core STATIC
  src/module_paths.cpp
  src/ort_loader.cpp
  src/version.cpp)
target_include_directories(edgebench_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
# OnnxRuntime::OnnxRuntime brings ORT_API_MANUAL_INIT and the delay-load of
# onnxruntime.dll to everything linking edgebench_core (cmake/FindOnnxRuntime.cmake).
target_link_libraries(edgebench_core PUBLIC OnnxRuntime::OnnxRuntime)
edgebench_set_warnings(edgebench_core)

# --- CLI --------------------------------------------------------------------

add_executable(edgebench src/main.cpp)
target_link_libraries(edgebench PRIVATE edgebench_core)
edgebench_set_warnings(edgebench)
onnxruntime_copy_runtime_files(edgebench QNN)

# --- Tests ------------------------------------------------------------------
#
# ctest labels:
#   unit        no ONNX Runtime DLLs needed
#   ort_bundle  needs the bundled ONNX Runtime next to the executables (and runs
#               edgebench.exe); exclude with `ctest -LE ort_bundle`

if(EDGEBENCH_BUILD_TESTS)
  add_executable(edgebench_tests
    tests/module_paths_test.cpp
    tests/version_test.cpp)
  target_link_libraries(edgebench_tests PRIVATE edgebench_core GTest::gtest_main)
  target_compile_definitions(edgebench_tests PRIVATE
    EDGEBENCH_EXPECTED_VERSION="${PROJECT_VERSION}"
    EDGEBENCH_EXPECTED_ORT_VERSION="${OnnxRuntime_VERSION}"
    EDGEBENCH_SCHEMA_DIR="${PROJECT_SOURCE_DIR}/schemas")
  edgebench_set_warnings(edgebench_tests)
  gtest_discover_tests(edgebench_tests DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS unit)

  # Loads ONNX Runtime from the given DLL first, then runs the loader; used by the
  # foreign-module tests.
  add_executable(edgebench_preload_helper tests/preload_helper.cpp)
  target_link_libraries(edgebench_preload_helper PRIVATE edgebench_core)
  edgebench_set_warnings(edgebench_preload_helper)

  add_executable(edgebench_bundle_tests
    tests/ort_loader_test.cpp
    tests/edgebench_exe_test.cpp
    tests/process_util.cpp)
  target_link_libraries(edgebench_bundle_tests PRIVATE edgebench_core GTest::gtest_main)
  target_compile_definitions(edgebench_bundle_tests PRIVATE
    EDGEBENCH_EXE_PATH="$<TARGET_FILE:edgebench>"
    EDGEBENCH_PRELOAD_HELPER_PATH="$<TARGET_FILE:edgebench_preload_helper>")
  add_dependencies(edgebench_bundle_tests edgebench edgebench_preload_helper)
  edgebench_set_warnings(edgebench_bundle_tests)
  onnxruntime_copy_runtime_files(edgebench_bundle_tests)
  gtest_discover_tests(edgebench_bundle_tests DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS ort_bundle)
endif()
