cmake_minimum_required(VERSION 3.25)
project(ort_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared modules from the repository root.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(RequireNativeArm64)

find_package(OnnxRuntime REQUIRED COMPONENTS QNN)

add_executable(ort_probe main.cpp)
target_link_libraries(ort_probe PRIVATE OnnxRuntime::OnnxRuntime)
if(MSVC)
  target_compile_options(ort_probe PRIVATE /W4 /permissive- /utf-8)
endif()

onnxruntime_copy_runtime_files(ort_probe QNN)
