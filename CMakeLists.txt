# Builds the kernel library, the benchmark programs and the tests.
cmake_minimum_required(VERSION 3.20)
project(roofline_kernels CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
add_compile_options(-march=native -Wall -Wextra)

add_library(kernels
  src/gemm_ref.cpp src/gemm_naive.cpp src/gemm_blocked.cpp src/gemm_simd.cpp
  src/gemm_goto.cpp src/registry.cpp
  src/attention_scalar.cpp src/attention.cpp
  src/decode_scalar.cpp src/decode.cpp)
target_include_directories(kernels PUBLIC src)
target_link_libraries(kernels PUBLIC OpenMP::OpenMP_CXX)

# The scalar stages must stay scalar. The auto-vectoriser would hide the step
# that the SIMD stage is meant to measure.
set_source_files_properties(
  src/gemm_ref.cpp src/gemm_naive.cpp src/gemm_blocked.cpp
  src/attention_scalar.cpp src/decode_scalar.cpp PROPERTIES
  COMPILE_OPTIONS "-fno-tree-vectorize;-fno-tree-slp-vectorize")

add_executable(bench_gemm bench/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE kernels ${CMAKE_DL_LIBS})

add_executable(bench_attn bench/bench_attn.cpp)
target_link_libraries(bench_attn PRIVATE kernels)

add_executable(attn_phases bench/attn_phases.cpp)
target_link_libraries(attn_phases PRIVATE kernels)

add_executable(bench_machine bench/bench_machine.cpp)
target_link_libraries(bench_machine PRIVATE kernels)

add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE kernels)

add_executable(test_attn tests/test_attn.cpp)
target_link_libraries(test_attn PRIVATE kernels)

enable_testing()
add_test(NAME gemm COMMAND test_gemm)
add_test(NAME attn COMMAND test_attn)
