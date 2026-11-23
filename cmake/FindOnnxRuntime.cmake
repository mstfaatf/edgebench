# FindOnnxRuntime
# ---------------
#
# Locates the prebuilt ONNX Runtime package and, optionally, the Qualcomm QNN
# execution provider plugin, as downloaded by scripts/fetch_deps.ps1.
#
#   find_package(OnnxRuntime REQUIRED)
#   find_package(OnnxRuntime REQUIRED COMPONENTS QNN)
#
# Hints (cache variables):
#   OnnxRuntime_ROOT   ONNX Runtime package root   (default: <repo>/third_party/onnxruntime
#                      for ARM64, <repo>/third_party/onnxruntime-x64 for x64; chosen from
#                      EDGEBENCH_TARGET_ARCH, see cmake/TargetArchitecture.cmake)
#   QnnEp_ROOT         QNN EP package root         (default: <repo>/third_party/onnxruntime-qnn;
#                      ARM64 only)
#
# Results:
#   OnnxRuntime::OnnxRuntime     imported shared library target. Everything linking it
#                                is compiled with ORT_API_MANUAL_INIT and, with MSVC,
#                                delay-loads onnxruntime.dll (see below).
#   OnnxRuntime_FOUND, OnnxRuntime_QNN_FOUND
#   OnnxRuntime_VERSION          e.g. 1.32.0 (from the package's .fetched marker, if present)
#   OnnxRuntime_RUNTIME_FILES    DLLs that must sit next to an executable using ONNX Runtime
#   OnnxRuntime_QNN_RUNTIME_FILES  QNN plugin EP, backends, and HTP skel/catalog files
#
# Functions:
#   onnxruntime_copy_runtime_files(<target> [QNN])
#     Adds a post-build step copying the runtime files next to <target>'s output.
#     QNN also copies the QNN files when the QNN component was found (ARM64 builds);
#     on x64 it copies ONNX Runtime alone.

get_filename_component(_ort_repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(EDGEBENCH_TARGET_ARCH STREQUAL "x64")
  set(_ort_default_root "${_ort_repo_root}/third_party/onnxruntime-x64")
else()
  set(_ort_default_root "${_ort_repo_root}/third_party/onnxruntime")
endif()
set(OnnxRuntime_ROOT "${_ort_default_root}" CACHE PATH "ONNX Runtime package root")
set(QnnEp_ROOT "${_ort_repo_root}/third_party/onnxruntime-qnn" CACHE PATH "QNN execution provider package root")

find_path(OnnxRuntime_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS "${OnnxRuntime_ROOT}/include" NO_DEFAULT_PATH)
find_library(OnnxRuntime_IMPLIB onnxruntime
  PATHS "${OnnxRuntime_ROOT}/lib" NO_DEFAULT_PATH)
find_file(OnnxRuntime_DLL onnxruntime.dll
  PATHS "${OnnxRuntime_ROOT}/lib" NO_DEFAULT_PATH)
find_file(OnnxRuntime_PROVIDERS_SHARED_DLL onnxruntime_providers_shared.dll
  PATHS "${OnnxRuntime_ROOT}/lib" NO_DEFAULT_PATH)

# The fetch script writes "<version> <sha256>" to .fetched.
unset(OnnxRuntime_VERSION)
if(EXISTS "${OnnxRuntime_ROOT}/.fetched")
  file(READ "${OnnxRuntime_ROOT}/.fetched" _ort_marker)
  string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" OnnxRuntime_VERSION "${_ort_marker}")
endif()

if("QNN" IN_LIST OnnxRuntime_FIND_COMPONENTS)
  find_file(OnnxRuntime_QNN_EP_DLL onnxruntime_providers_qnn.dll
    PATHS "${QnnEp_ROOT}" NO_DEFAULT_PATH)
  find_file(OnnxRuntime_QNN_HTP_DLL QnnHtp.dll
    PATHS "${QnnEp_ROOT}" NO_DEFAULT_PATH)
  if(OnnxRuntime_QNN_EP_DLL AND OnnxRuntime_QNN_HTP_DLL)
    set(OnnxRuntime_QNN_FOUND TRUE)
    # QnnHtp.dll loads its stub/skel/catalog files from its own directory.
    file(GLOB OnnxRuntime_QNN_RUNTIME_FILES
      "${QnnEp_ROOT}/*.dll" "${QnnEp_ROOT}/*.so" "${QnnEp_ROOT}/*.cat")
  else()
    set(OnnxRuntime_QNN_FOUND FALSE)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime
  REQUIRED_VARS OnnxRuntime_INCLUDE_DIR OnnxRuntime_IMPLIB OnnxRuntime_DLL
  VERSION_VAR OnnxRuntime_VERSION
  HANDLE_COMPONENTS
  REASON_FAILURE_MESSAGE
    "Run: powershell -ExecutionPolicy Bypass -File scripts/fetch_deps.ps1 -Arch ${EDGEBENCH_TARGET_ARCH}")

if(OnnxRuntime_FOUND)
  set(OnnxRuntime_RUNTIME_FILES "${OnnxRuntime_DLL}")
  if(OnnxRuntime_PROVIDERS_SHARED_DLL)
    list(APPEND OnnxRuntime_RUNTIME_FILES "${OnnxRuntime_PROVIDERS_SHARED_DLL}")
  endif()

  if(NOT TARGET OnnxRuntime::OnnxRuntime)
    add_library(OnnxRuntime::OnnxRuntime SHARED IMPORTED)
    # Include directories of imported targets are treated as system headers,
    # so project warning levels (/W4 /WX) do not apply to them.
    set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
      IMPORTED_LOCATION "${OnnxRuntime_DLL}"
      IMPORTED_IMPLIB "${OnnxRuntime_IMPLIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIR}")
    # Explicit loading, for every consumer: the C++ API must not initialize itself
    # during static initialization (ORT_API_MANUAL_INIT), and onnxruntime.dll must
    # not be loaded by the normal DLL search order at process start (/DELAYLOAD).
    # The program loads the bundled copy by full path and verifies it first
    # (edgebench::load_onnxruntime() in engine/src/ort_loader.cpp). Setting this on
    # the imported target keeps every translation unit in a binary consistent.
    set_property(TARGET OnnxRuntime::OnnxRuntime APPEND PROPERTY
      INTERFACE_COMPILE_DEFINITIONS ORT_API_MANUAL_INIT)
    if(MSVC)
      # Binaries that reach ONNX Runtime only through the loader's GetProcAddress
      # have no imports from onnxruntime.dll, so the linker reports LNK4199 for
      # them; the delay-load is still wanted as a safety net for any direct import.
      set_property(TARGET OnnxRuntime::OnnxRuntime APPEND PROPERTY
        INTERFACE_LINK_OPTIONS "/DELAYLOAD:onnxruntime.dll" "/IGNORE:4199")
      set_property(TARGET OnnxRuntime::OnnxRuntime APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES delayimp)
    endif()
  endif()
endif()

mark_as_advanced(OnnxRuntime_INCLUDE_DIR OnnxRuntime_IMPLIB OnnxRuntime_DLL
  OnnxRuntime_PROVIDERS_SHARED_DLL OnnxRuntime_QNN_EP_DLL OnnxRuntime_QNN_HTP_DLL)

# Copy runtime files next to a target after it is built. Windows ships an
# unrelated onnxruntime.dll in System32; a copy next to the executable is found
# first by the DLL search order.
function(onnxruntime_copy_runtime_files target)
  cmake_parse_arguments(PARSE_ARGV 1 _arg "QNN" "" "")
  set(_files ${OnnxRuntime_RUNTIME_FILES})
  if(_arg_QNN AND OnnxRuntime_QNN_FOUND)
    list(APPEND _files ${OnnxRuntime_QNN_RUNTIME_FILES})
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_files} "$<TARGET_FILE_DIR:${target}>"
    COMMENT "Copying ONNX Runtime runtime files next to ${target}"
    VERBATIM)
endfunction()
