// The naive ijk GEMM.
#include "gemm.hpp"

void gemm_naive(int M, int N, int K, const float* A, int lda, const float* B,
                int ldb, float* C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int p = 0; p < K; ++p) {
        acc += A[i * lda + p] * B[p * ldb + j];
      }
      C[i * ldc + j] = acc;
    }
  }
}
