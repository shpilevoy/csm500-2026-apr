// GEMM kernels, from a naive triple loop to a packed microkernel.
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
// Adds packing and a 6x16 microkernel.
void gemm_goto(int M, int N, int K, const float* A, int lda, const float* B,
               int ldb, float* C, int ldc);

// Block sizes in elements.
struct GemmParams {
  // Used by blocked and simd.
  int bk = 64;
  int bn = 512;
  // Used by goto and goto_mt. Packed A is mc x kc and packed B is kc x nc.
  // The default values are those of the BLIS configuration for AMD Zen in
  // single precision. They come from F. G. Van Zee and others, "BLIS", file
  // config/zen/bli_cntx_init_zen.c. https://github.com/flame/blis
  // (accessed 17 Aug 2026).
  int mc = 144;
  int kc = 256;
  int nc = 4080;
};
GemmParams& gemm_params();
bool gemm_set_param(std::string_view name, int value);

struct GemmKernel {
  const char* name;
  GemmFn fn;
};

std::span<const GemmKernel> gemm_kernels();
GemmFn find_gemm(std::string_view name);
