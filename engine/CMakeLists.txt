# edgebench_set_warnings() comes from cmake/EdgeBenchWarnings.cmake; GoogleTest,
# CLI11 and nlohmann/json are made available by the root CMakeLists.txt.

# --- Core library -----------------------------------------------------------

configure_file(src/version_config.hpp.in
  "${CMAKE_CURRENT_BINARY_DIR}/generated/version_config.hpp" @ONLY)

# The pinned bundle files (cmake/bundle-pins.json) and the doctor's NPU probe
# models with their expected outputs (engine/resources/doctor/), built into the
# binary so `edgebench doctor` needs no files besides the bundle.
edgebench_generate_bundle_pins("${CMAKE_CURRENT_BINARY_DIR}/generated/bundle_pins.hpp")
include(EmbedFiles)
set(_doctor_resources "${CMAKE_CURRENT_SOURCE_DIR}/resources/doctor")
edgebench_embed_files("${CMAKE_CURRENT_BINARY_DIR}/generated/doctor_probes.hpp"
  edgebench::doctor_resources
  kNpuProbeInt8Model "${_doctor_resources}/npu_probe_int8.onnx"
  kNpuProbeInt8Expected "${_doctor_resources}/npu_probe_int8.npu.f32"
  kNpuProbeFp32Model "${_doctor_resources}/npu_probe_fp32.onnx"
  kNpuProbeFp32Expected "${_doctor_resources}/npu_probe_fp32.npu_fp16.f32")

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
  src/batch_equivalence.cpp
  src/bundle_modules.cpp
  src/cli.cpp
  src/cpu_provider.cpp
  src/crash_handler.cpp
  src/diagnostics.cpp
  src/doctor.cpp
  src/doctor_checks.cpp
  src/dry_run.cpp
  src/ep_context_cache.cpp
  src/ep_devices.cpp
  src/ep_library.cpp
  src/exit_codes.cpp
  src/input_fill.cpp
  src/interrupt.cpp
  src/model_info.cpp
  src/module_paths.cpp
  src/npu_options.cpp
  src/npu_selection.cpp
  src/onnx_inspect.cpp
  src/ort_bench_session.cpp
  src/ort_loader.cpp
  src/placement.cpp
  src/qnn_ep.cpp
  src/qnn_backend_info.cpp
  src/qnn_log.cpp
  src/qnn_profile.cpp
  src/qnn_provider.cpp
  src/result.cpp
  src/result_files.cpp
  src/result_json.cpp
  src/run_config.cpp
  src/run_identity.cpp
  src/runtime.cpp
  src/scheduler.cpp
  src/sha256.cpp
  src/stats.cpp
  src/suite.cpp
  src/suite_config.cpp
  src/suite_gate.cpp
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
  PRIVATE CLI11::CLI11 bcrypt powrprof version)
edgebench_set_warnings(edgebench_core)

# --- CLI --------------------------------------------------------------------

add_executable(edgebench src/main.cpp)
target_link_libraries(edgebench PRIVATE edgebench_core)
edgebench_set_warnings(edgebench)
onnxruntime_copy_runtime_files(edgebench QNN)

# --- Tests ------------------------------------------------------------------
#
# ctest labels (docs/setup.md, "Tests"):
#   unit    no ONNX Runtime DLL needed
#   ort     the bundled ONNX Runtime for the build's architecture and the committed
#           fixture models only
#   models  exported models or reference data (models/, data/ort_probe_reference/);
#           also needs the bundled ONNX Runtime
#   npu     the QNN execution provider, its backend and an NPU (ARM64 Snapdragon only)
#   schema  the Python results-schema and validator suite (root CMakeLists.txt)
#
# In the ONNX Runtime test programs the label follows the test suite name: suites
# ending in "Models" get `models`, suites ending in "Npu" get `npu`, all others
# `ort`. CI runs `ctest -L "unit|ort"`.

# NPU tests share the one NPU: the RESOURCE_LOCK keeps ctest from running two of
# them at the same time even with -j (other tests still run in parallel). The
# timeout is about 8x the slowest NPU test (~40 s), so a hang shows up as a
# timeout instead of a stuck run.
set(EDGEBENCH_NPU_TEST_TIMEOUT 300)

# Registers the tests of an ONNX Runtime test program under the ort/models/npu labels.
function(edgebench_discover_ort_tests target)
  gtest_discover_tests(${target} DISCOVERY_MODE PRE_TEST
    TEST_FILTER "-*Models.*:*Npu.*" PROPERTIES LABELS ort)
  gtest_discover_tests(${target} DISCOVERY_MODE PRE_TEST
    TEST_FILTER "*Models.*" PROPERTIES LABELS models)
  gtest_discover_tests(${target} DISCOVERY_MODE PRE_TEST
    TEST_FILTER "*Npu.*" PROPERTIES LABELS npu RESOURCE_LOCK npu
    TIMEOUT ${EDGEBENCH_NPU_TEST_TIMEOUT})
endfunction()

if(EDGEBENCH_BUILD_TESTS)
  add_executable(edgebench_tests
    tests/batch_equivalence_test.cpp
    tests/ep_context_cache_test.cpp
    tests/cli_test.cpp
    tests/doctor_checks_test.cpp
    tests/core_types_test.cpp
    tests/exit_codes_test.cpp
    tests/interrupt_test.cpp
    tests/module_paths_test.cpp
    tests/npu_options_test.cpp
    tests/npu_selection_test.cpp
    tests/qnn_backend_info_test.cpp
    tests/result_files_test.cpp
    tests/run_config_test.cpp
    tests/suite_gate_test.cpp
    tests/suite_test.cpp
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

  # Crashes or ignores the first interrupt on request; used by the exit-path tests.
  add_executable(edgebench_signal_helper tests/signal_helper.cpp)
  target_link_libraries(edgebench_signal_helper PRIVATE edgebench_core)
  edgebench_set_warnings(edgebench_signal_helper)

  add_executable(edgebench_bundle_tests
    tests/benchmark_test.cpp
    tests/exit_paths_test.cpp
    tests/failure_paths_test.cpp
    tests/ort_loader_test.cpp
    tests/edgebench_exe_test.cpp
    tests/provider_test.cpp
    tests/qnn_ep_test.cpp
    tests/qnn_provider_test.cpp
    tests/ep_cache_npu_test.cpp
    tests/backend_info_npu_test.cpp
    tests/reference_models_test.cpp
    tests/suite_exe_test.cpp
    tests/test_runtime.cpp
    tests/process_util.cpp
    tests/validator.cpp)
  target_link_libraries(edgebench_bundle_tests PRIVATE edgebench_core GTest::gtest_main)
  target_compile_definitions(edgebench_bundle_tests PRIVATE
    EDGEBENCH_EXE_PATH="$<TARGET_FILE:edgebench>"
    EDGEBENCH_PRELOAD_HELPER_PATH="$<TARGET_FILE:edgebench_preload_helper>"
    EDGEBENCH_SIGNAL_HELPER_PATH="$<TARGET_FILE:edgebench_signal_helper>"
    EDGEBENCH_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    EDGEBENCH_PYTHON="${EDGEBENCH_PYTHON}")
  add_dependencies(edgebench_bundle_tests edgebench edgebench_preload_helper
    edgebench_signal_helper)
  if(EDGEBENCH_BUILD_TOOLS)
    target_compile_definitions(edgebench_bundle_tests PRIVATE
      EDGEBENCH_ORT_PROBE_PATH="$<TARGET_FILE:ort_probe>")
    add_dependencies(edgebench_bundle_tests ort_probe)
  endif()
  edgebench_set_warnings(edgebench_bundle_tests)
  onnxruntime_copy_runtime_files(edgebench_bundle_tests)
  edgebench_discover_ort_tests(edgebench_bundle_tests)
endif()
