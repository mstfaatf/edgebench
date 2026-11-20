# Warning flags for EdgeBench's own targets. Third-party code (GoogleTest, ONNX
# Runtime headers) is excluded: ONNX Runtime comes in as an imported target and
# GoogleTest is declared SYSTEM, so their headers are treated as external.
function(edgebench_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX /permissive- /utf-8 /external:W0)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

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

  add_executable(edgebench_tests
    tests/version_test.cpp)
  target_link_libraries(edgebench_tests PRIVATE edgebench_core GTest::gtest_main)
  target_compile_definitions(edgebench_tests PRIVATE
    EDGEBENCH_EXPECTED_VERSION="${PROJECT_VERSION}"
    EDGEBENCH_EXPECTED_ORT_VERSION="${OnnxRuntime_VERSION}")
  edgebench_set_warnings(edgebench_tests)
  onnxruntime_copy_runtime_files(edgebench_tests)

  include(GoogleTest)
  # Discover tests when CTest runs, after the runtime DLLs have been copied.
  gtest_discover_tests(edgebench_tests DISCOVERY_MODE PRE_TEST)
endif()
