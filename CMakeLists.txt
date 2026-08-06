# Builds the kernel library, the benchmark programs and the tests.
cmake_minimum_required(VERSION 3.20)
project(roofline_kernels CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-march=native -Wall -Wextra)

add_library(kernels
  src/gemm_ref.cpp src/gemm_naive.cpp src/gemm_blocked.cpp)
target_include_directories(kernels PUBLIC src)

add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE kernels)

add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE kernels)

enable_testing()
add_test(NAME gemm COMMAND test_gemm)
