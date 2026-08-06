// Scalar GEMM with a reordered loop nest and with cache blocking.
#include <algorithm>
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

void gemm_blocked(int M, int N, int K, const float* A, int lda, const float* B,
                  int ldb, float* C, int ldc) {
  const int BK = 64, BN = 512;
  for (int i = 0; i < M; ++i) std::memset(C + i * ldc, 0, N * sizeof(float));
  for (int pp = 0; pp < K; pp += BK) {
    const int p_end = std::min(pp + BK, K);
    for (int jj = 0; jj < N; jj += BN) {
      const int j_end = std::min(jj + BN, N);
      for (int i = 0; i < M; ++i) {
        float* c = C + i * ldc;
        for (int p = pp; p < p_end; ++p) {
          const float a = A[i * lda + p];
          const float* b = B + p * ldb;
          for (int j = jj; j < j_end; ++j) c[j] += a * b[j];
        }
      }
    }
  }
}
