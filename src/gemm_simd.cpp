// GEMM with the loop nest of gemm_blocked and an AVX2/FMA inner loop.
#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "gemm.hpp"

void gemm_simd(int M, int N, int K, const float* A, int lda, const float* B,
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
          const __m256 va = _mm256_set1_ps(a);
          const float* b = B + p * ldb;
          int j = jj;
          for (; j + 8 <= j_end; j += 8) {
            const __m256 vc = _mm256_fmadd_ps(va, _mm256_loadu_ps(b + j),
                                              _mm256_loadu_ps(c + j));
            _mm256_storeu_ps(c + j, vc);
          }
          for (; j < j_end; ++j) c[j] += a * b[j];
        }
      }
    }
  }
}
