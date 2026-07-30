// GEMM kernels.
#pragma once

// C (MxN) = A (MxK) * B (KxN). Row-major FP32. C is overwritten.
using GemmFn = void (*)(int M, int N, int K, const float* A, int lda,
                        const float* B, int ldb, float* C, int ldc);

// Reference that every kernel is checked against. Scalar, sums in double.
void gemm_ref(int M, int N, int K, const float* A, int lda, const float* B,
              int ldb, float* C, int ldc);

// ijk triple loop.
void gemm_naive(int M, int N, int K, const float* A, int lda, const float* B,
                int ldb, float* C, int ldc);
