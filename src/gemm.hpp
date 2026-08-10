// GEMM kernels.
#pragma once

#include <span>
#include <string_view>

// C (MxN) = A (MxK) * B (KxN). Row-major FP32. C is overwritten.
using GemmFn = void (*)(int M, int N, int K, const float* A, int lda,
                        const float* B, int ldb, float* C, int ldc);

// Reference that every kernel is checked against. Scalar, sums in double.
void gemm_ref(int M, int N, int K, const float* A, int lda, const float* B,
              int ldb, float* C, int ldc);

// ijk triple loop.
void gemm_naive(int M, int N, int K, const float* A, int lda, const float* B,
                int ldb, float* C, int ldc);
// ikj order, so the inner loop has unit stride.
void gemm_reorder(int M, int N, int K, const float* A, int lda, const float* B,
                  int ldb, float* C, int ldc);
// Adds cache blocking. Still scalar.
void gemm_blocked(int M, int N, int K, const float* A, int lda, const float* B,
                  int ldb, float* C, int ldc);
// Adds an AVX2/FMA inner loop.
void gemm_simd(int M, int N, int K, const float* A, int lda, const float* B,
               int ldb, float* C, int ldc);

struct GemmKernel {
  const char* name;
  GemmFn fn;
};

std::span<const GemmKernel> gemm_kernels();
GemmFn find_gemm(std::string_view name);
