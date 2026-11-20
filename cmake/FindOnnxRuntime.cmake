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
#   OnnxRuntime_ROOT   ONNX Runtime package root   (default: <repo>/third_party/onnxruntime)
#   QnnEp_ROOT         QNN EP package root         (default: <repo>/third_party/onnxruntime-qnn)
#
# Results:
#   OnnxRuntime::OnnxRuntime     imported shared library target
#   OnnxRuntime_FOUND, OnnxRuntime_QNN_FOUND
#   OnnxRuntime_VERSION          e.g. 1.32.0 (from the package's .fetched marker, if present)
#   OnnxRuntime_RUNTIME_FILES    DLLs that must sit next to an executable using ONNX Runtime
#   OnnxRuntime_QNN_RUNTIME_FILES  QNN plugin EP, backends, and HTP skel/catalog files
#
# Functions:
#   onnxruntime_copy_runtime_files(<target> [QNN])
#     Adds a post-build step copying the runtime files next to <target>'s output.

get_filename_component(_ort_repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(OnnxRuntime_ROOT "${_ort_repo_root}/third_party/onnxruntime" CACHE PATH "ONNX Runtime package root")
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
    "Run: powershell -ExecutionPolicy Bypass -File scripts/fetch_deps.ps1")

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
  if(_arg_QNN)
    if(NOT OnnxRuntime_QNN_FOUND)
      message(FATAL_ERROR "onnxruntime_copy_runtime_files(${target} QNN): QNN component not found")
    endif()
    list(APPEND _files ${OnnxRuntime_QNN_RUNTIME_FILES})
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_files} "$<TARGET_FILE_DIR:${target}>"
    COMMENT "Copying ONNX Runtime runtime files next to ${target}"
    VERBATIM)
endfunction()
