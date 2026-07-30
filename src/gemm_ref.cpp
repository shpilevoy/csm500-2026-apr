// The reference GEMM that sums in double.
#include "gemm.hpp"

void gemm_ref(int M, int N, int K, const float* A, int lda, const float* B,
              int ldb, float* C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int p = 0; p < K; ++p) {
        acc += static_cast<double>(A[i * lda + p]) * B[p * ldb + j];
      }
      C[i * ldc + j] = static_cast<float>(acc);
    }
  }
}
