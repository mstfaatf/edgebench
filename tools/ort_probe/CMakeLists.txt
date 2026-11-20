cmake_minimum_required(VERSION 3.24)
project(ort_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Native ARM64 only. ARM64EC and x64 builds would run under emulation and are
# not what this tool is meant to measure.
if(NOT CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
  message(FATAL_ERROR
    "ort_probe must be built for native ARM64, but the compiler targets "
    "'${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}'. Use the arm64-release preset "
    "(cmake --preset arm64-release) or an ARM64 developer prompt.")
endif()

# Binaries downloaded by scripts/fetch_deps.ps1.
get_filename_component(EDGEBENCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(ORT_ROOT "${EDGEBENCH_ROOT}/third_party/onnxruntime" CACHE PATH "ONNX Runtime package root")
set(QNN_EP_ROOT "${EDGEBENCH_ROOT}/third_party/onnxruntime-qnn" CACHE PATH "QNN execution provider package root")

foreach(required IN ITEMS
    "${ORT_ROOT}/include/onnxruntime_cxx_api.h"
    "${ORT_ROOT}/lib/onnxruntime.lib"
    "${ORT_ROOT}/lib/onnxruntime.dll"
    "${QNN_EP_ROOT}/onnxruntime_providers_qnn.dll"
    "${QNN_EP_ROOT}/QnnHtp.dll")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR
      "Missing ${required}\n"
      "Run: powershell -ExecutionPolicy Bypass -File scripts/fetch_deps.ps1")
  endif()
endforeach()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime.dll"
  IMPORTED_IMPLIB "${ORT_ROOT}/lib/onnxruntime.lib"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include")

add_executable(ort_probe main.cpp)
target_link_libraries(ort_probe PRIVATE onnxruntime)
if(MSVC)
  target_compile_options(ort_probe PRIVATE /W4 /permissive- /utf-8)
endif()

# Runtime files that must sit next to the executable:
#  - onnxruntime.dll (+ providers_shared). Windows also ships an onnxruntime.dll
#    in System32; the copy next to the exe is found first.
#  - the QNN plugin EP, the QNN backends, and the HTP skel/catalog files that
#    QnnHtp.dll loads from its own directory.
file(GLOB QNN_RUNTIME_FILES
  "${QNN_EP_ROOT}/*.dll"
  "${QNN_EP_ROOT}/*.so"
  "${QNN_EP_ROOT}/*.cat")
set(ORT_RUNTIME_FILES
  "${ORT_ROOT}/lib/onnxruntime.dll"
  "${ORT_ROOT}/lib/onnxruntime_providers_shared.dll")

add_custom_command(TARGET ort_probe POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${ORT_RUNTIME_FILES} ${QNN_RUNTIME_FILES}
          "$<TARGET_FILE_DIR:ort_probe>"
  COMMENT "Copying ONNX Runtime and QNN runtime files next to ort_probe"
  VERBATIM)
