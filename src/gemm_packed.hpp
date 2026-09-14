// The packed GEMM behind gemm_goto, with the extra options that attention needs.
#pragma once

#include <cstddef>

struct GemmPackedOpts {
  // B is given as its transpose, an N x K row-major matrix with row stride ldb.
  bool trans_b = false;
  bool accumulate = false;
  int threads = 1;
  // Non-temporal stores. They help only when C is much larger than the caches
  // and is not read again soon. Ignored when it cannot apply.
  bool stream_c = false;
  // B already packed by gemm_pack_b with the same N, K and block sizes. B and
  // ldb are then ignored.
  const float* packed_b = nullptr;
};

// Size in floats of the packed form of a K x N matrix B.
std::size_t gemm_packed_b_size(int N, int K);
// Packs B once so that many products can use it through packed_b.
void gemm_pack_b(int N, int K, const float* B, int ldb, bool trans, float* out);

void gemm_packed(int M, int N, int K, const float* A, int lda, const float* B,
                 int ldb, float* C, int ldc, GemmPackedOpts opts);
