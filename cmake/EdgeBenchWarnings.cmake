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
