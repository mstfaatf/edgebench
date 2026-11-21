# ort_probe: diagnostic for ONNX Runtime + the QNN plugin EP from native C++.
# Built as part of the root project (EDGEBENCH_BUILD_TOOLS); see docs/setup.md.

# Helpers without an ONNX Runtime dependency (unit-tested).
add_library(ort_probe_util STATIC probe_util.cpp)
target_include_directories(ort_probe_util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
edgebench_set_warnings(ort_probe_util)

# ONNX Runtime is loaded explicitly, by full path, through edgebench_core's
# loader (engine/include/edgebench/ort_loader.hpp); ORT_API_MANUAL_INIT and the
# delay-load of onnxruntime.dll come with the OnnxRuntime::OnnxRuntime target.
add_executable(ort_probe main.cpp ort_support.cpp)
target_link_libraries(ort_probe PRIVATE ort_probe_util edgebench_core)
edgebench_set_warnings(ort_probe)
# onnxruntime.dll and the QNN EP/backends must sit next to the executable: the
# loader searches the application directory first, and ort_probe refuses to run
# if onnxruntime.dll was loaded from anywhere else (Windows ships an unrelated
# copy in System32).
onnxruntime_copy_runtime_files(ort_probe QNN)

if(EDGEBENCH_BUILD_TESTS)
  add_executable(ort_probe_tests probe_util_test.cpp)
  target_link_libraries(ort_probe_tests PRIVATE ort_probe_util GTest::gtest_main)
  edgebench_set_warnings(ort_probe_tests)
  gtest_discover_tests(ort_probe_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)
endif()
