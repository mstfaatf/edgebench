# edgebench_set_warnings() comes from cmake/EdgeBenchWarnings.cmake; GoogleTest is
# made available by the root CMakeLists.txt when EDGEBENCH_BUILD_TESTS is on.

# --- Core library -----------------------------------------------------------

configure_file(src/version_config.hpp.in
  "${CMAKE_CURRENT_BINARY_DIR}/generated/version_config.hpp" @ONLY)

add_library(edgebench_core STATIC
  src/version.cpp)
target_include_directories(edgebench_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(edgebench_core PUBLIC OnnxRuntime::OnnxRuntime)
edgebench_set_warnings(edgebench_core)

# --- CLI --------------------------------------------------------------------

add_executable(edgebench src/main.cpp)
target_link_libraries(edgebench PRIVATE edgebench_core)
edgebench_set_warnings(edgebench)
onnxruntime_copy_runtime_files(edgebench QNN)

# --- Tests ------------------------------------------------------------------

if(EDGEBENCH_BUILD_TESTS)
  add_executable(edgebench_tests
    tests/version_test.cpp)
  target_link_libraries(edgebench_tests PRIVATE edgebench_core GTest::gtest_main)
  target_compile_definitions(edgebench_tests PRIVATE
    EDGEBENCH_EXPECTED_VERSION="${PROJECT_VERSION}"
    EDGEBENCH_EXPECTED_ORT_VERSION="${OnnxRuntime_VERSION}")
  edgebench_set_warnings(edgebench_tests)
  onnxruntime_copy_runtime_files(edgebench_tests)

  # Discover tests when CTest runs, after the runtime DLLs have been copied.
  gtest_discover_tests(edgebench_tests DISCOVERY_MODE PRE_TEST)
endif()
