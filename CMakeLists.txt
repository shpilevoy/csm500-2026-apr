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
  src/gemm_ref.cpp src/gemm_naive.cpp)
target_include_directories(kernels PUBLIC src)
