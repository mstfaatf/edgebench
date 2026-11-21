# edgebench_set_warnings() comes from cmake/EdgeBenchWarnings.cmake; GoogleTest,
# CLI11 and nlohmann/json are made available by the root CMakeLists.txt.

# --- Core library -----------------------------------------------------------

configure_file(src/version_config.hpp.in
  "${CMAKE_CURRENT_BINARY_DIR}/generated/version_config.hpp" @ONLY)

# Git commit and dirty flag, regenerated on every build (the header only changes
# when the values do).
set(EDGEBENCH_BUILD_INFO_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/build_info.hpp")
set(_build_info_command "${CMAKE_COMMAND}" -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
  -DOUTPUT=${EDGEBENCH_BUILD_INFO_HEADER} -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
  -P "${PROJECT_SOURCE_DIR}/cmake/BuildInfo.cmake")
execute_process(COMMAND ${_build_info_command})
add_custom_target(edgebench_build_info
  COMMAND ${_build_info_command}
  BYPRODUCTS "${EDGEBENCH_BUILD_INFO_HEADER}"
  COMMENT "Updating git build information"
  VERBATIM)

add_library(edgebench_core STATIC
  src/benchmark.cpp
  src/cli.cpp
  src/cpu_provider.cpp
  src/doctor.cpp
  src/dry_run.cpp
  src/ep_devices.cpp
  src/exit_codes.cpp
  src/input_fill.cpp
  src/model_info.cpp
  src/module_paths.cpp
  src/onnx_inspect.cpp
  src/ort_bench_session.cpp
  src/ort_loader.cpp
  src/placement.cpp
  src/result.cpp
  src/run_config.cpp
  src/run_identity.cpp
  src/runtime.cpp
  src/scheduler.cpp
  src/sha256.cpp
  src/stats.cpp
  src/system_info.cpp
  src/tensor_info.cpp
  src/timer_info.cpp
  src/version.cpp)
add_dependencies(edgebench_core edgebench_build_info)
target_include_directories(edgebench_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(edgebench_core PRIVATE EDGEBENCH_BUILD_TYPE="$<CONFIG>")
# OnnxRuntime::OnnxRuntime brings ORT_API_MANUAL_INIT and the delay-load of
# onnxruntime.dll to everything linking edgebench_core (cmake/FindOnnxRuntime.cmake).
# nlohmann/json appears in public headers; CLI11 is used only inside cli.cpp.
target_link_libraries(edgebench_core
  PUBLIC OnnxRuntime::OnnxRuntime nlohmann_json::nlohmann_json
  PRIVATE CLI11::CLI11 bcrypt powrprof)
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
    tests/cli_test.cpp
    tests/core_types_test.cpp
    tests/exit_codes_test.cpp
    tests/module_paths_test.cpp
    tests/run_config_test.cpp
    tests/system_state_test.cpp
    tests/timing_test.cpp
    tests/version_test.cpp)
  target_link_libraries(edgebench_tests PRIVATE edgebench_core GTest::gtest_main)
  target_compile_definitions(edgebench_tests PRIVATE
    EDGEBENCH_EXPECTED_VERSION="${PROJECT_VERSION}"
    EDGEBENCH_EXPECTED_ORT_VERSION="${OnnxRuntime_VERSION}"
    EDGEBENCH_SCHEMA_DIR="${PROJECT_SOURCE_DIR}/schemas"
    EDGEBENCH_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  edgebench_set_warnings(edgebench_tests)
  gtest_discover_tests(edgebench_tests DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS unit)

  # Loads ONNX Runtime from the given DLL first, then runs the loader; used by the
  # foreign-module tests.
  add_executable(edgebench_preload_helper tests/preload_helper.cpp)
  target_link_libraries(edgebench_preload_helper PRIVATE edgebench_core)
  edgebench_set_warnings(edgebench_preload_helper)

  add_executable(edgebench_bundle_tests
    tests/benchmark_test.cpp
    tests/ort_loader_test.cpp
    tests/edgebench_exe_test.cpp
    tests/provider_test.cpp
    tests/process_util.cpp)
  target_link_libraries(edgebench_bundle_tests PRIVATE edgebench_core GTest::gtest_main)
  target_compile_definitions(edgebench_bundle_tests PRIVATE
    EDGEBENCH_EXE_PATH="$<TARGET_FILE:edgebench>"
    EDGEBENCH_PRELOAD_HELPER_PATH="$<TARGET_FILE:edgebench_preload_helper>"
    EDGEBENCH_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    EDGEBENCH_PYTHON="${EDGEBENCH_PYTHON}")
  add_dependencies(edgebench_bundle_tests edgebench edgebench_preload_helper)
  if(EDGEBENCH_BUILD_TOOLS)
    target_compile_definitions(edgebench_bundle_tests PRIVATE
      EDGEBENCH_ORT_PROBE_PATH="$<TARGET_FILE:ort_probe>")
    add_dependencies(edgebench_bundle_tests ort_probe)
  endif()
  edgebench_set_warnings(edgebench_bundle_tests)
  onnxruntime_copy_runtime_files(edgebench_bundle_tests)
  gtest_discover_tests(edgebench_bundle_tests DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS ort_bundle)
endif()
