// The packed GEMM behind gemm_goto.
#pragma once

#include <cstddef>

struct GemmPackedOpts {
  bool accumulate = false;
  int threads = 1;
};

void gemm_packed(int M, int N, int K, const float* A, int lda, const float* B,
                 int ldb, float* C, int ldc, GemmPackedOpts opts);
