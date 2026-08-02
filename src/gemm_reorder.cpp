// GEMM with the ikj loop order.
#include <cstring>

#include "gemm.hpp"

void gemm_reorder(int M, int N, int K, const float* A, int lda, const float* B,
                  int ldb, float* C, int ldc) {
  for (int i = 0; i < M; ++i) {
    float* c = C + i * ldc;
    std::memset(c, 0, N * sizeof(float));
    for (int p = 0; p < K; ++p) {
      const float a = A[i * lda + p];
      const float* b = B + p * ldb;
      for (int j = 0; j < N; ++j) c[j] += a * b[j];
    }
  }
}
