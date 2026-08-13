// GEMM in the style of Goto and BLIS. Both operands are packed into buffers
// that fit the caches. One microkernel does all the arithmetic.
#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "gemm.hpp"
#include "util.hpp"

namespace {

// The microkernel updates an MR x NR tile of C that is held in registers.
// 6 rows x 2 ymm = 12 accumulators. That leaves 4 of the 16 ymm registers for
// the two B vectors and the broadcast A element.
// BLIS uses the same 6 x 16 tile for single precision on this processor
// family. See the note in gemm.hpp. The kernel itself is written new.
constexpr int MR = 6, NR = 16;

// Packs an mc x kc block of A as slices of MR rows. Each slice is stored column
// by column, so the microkernel reads it with unit stride.
void pack_a(int mc, int kc, const float* A, int lda, float* out) {
  for (int i = 0; i < mc; i += MR) {
    const int rows = std::min(MR, mc - i);
    for (int p = 0; p < kc; ++p) {
      for (int r = 0; r < rows; ++r) *out++ = A[(i + r) * lda + p];
      for (int r = rows; r < MR; ++r) *out++ = 0.0f;
    }
  }
}

// Packs one NR-column slice of a kc-row block of B, row by row.
void pack_b_slice(int kc, int cols, const float* B, int ldb, float* out) {
  for (int p = 0; p < kc; ++p, out += NR) {
    std::memcpy(out, B + p * ldb, cols * sizeof(float));
    for (int c = cols; c < NR; ++c) out[c] = 0.0f;
  }
}

enum class Write { Add, Store, Stream };

// C[0:6, 0:16] (+)= a (6 x kc, packed) * b (kc x 16, packed).
// Stream uses non-temporal stores, so the processor does not read a line
// before it overwrites it. It needs a 32-byte aligned C.
inline void ukernel(int kc, const float* a, const float* b, float* c, int ldc, Write mode) {
  __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00;
  __m256 c20 = c00, c21 = c00, c30 = c00, c31 = c00;
  __m256 c40 = c00, c41 = c00, c50 = c00, c51 = c00;
  for (int p = 0; p < kc; ++p, a += MR, b += NR) {
    const __m256 b0 = _mm256_load_ps(b), b1 = _mm256_load_ps(b + 8);
    __m256 x = _mm256_broadcast_ss(a + 0);
    c00 = _mm256_fmadd_ps(x, b0, c00);
    c01 = _mm256_fmadd_ps(x, b1, c01);
    x = _mm256_broadcast_ss(a + 1);
    c10 = _mm256_fmadd_ps(x, b0, c10);
    c11 = _mm256_fmadd_ps(x, b1, c11);
    x = _mm256_broadcast_ss(a + 2);
    c20 = _mm256_fmadd_ps(x, b0, c20);
    c21 = _mm256_fmadd_ps(x, b1, c21);
    x = _mm256_broadcast_ss(a + 3);
    c30 = _mm256_fmadd_ps(x, b0, c30);
    c31 = _mm256_fmadd_ps(x, b1, c31);
    x = _mm256_broadcast_ss(a + 4);
    c40 = _mm256_fmadd_ps(x, b0, c40);
    c41 = _mm256_fmadd_ps(x, b1, c41);
    x = _mm256_broadcast_ss(a + 5);
    c50 = _mm256_fmadd_ps(x, b0, c50);
    c51 = _mm256_fmadd_ps(x, b1, c51);
  }
  auto add_row = [mode](float* row, __m256 lo, __m256 hi) {
    if (mode == Write::Stream) {
      _mm256_stream_ps(row, lo);
      _mm256_stream_ps(row + 8, hi);
      return;
    }
    if (mode == Write::Add) {
      lo = _mm256_add_ps(_mm256_loadu_ps(row), lo);
      hi = _mm256_add_ps(_mm256_loadu_ps(row + 8), hi);
    }
    _mm256_storeu_ps(row, lo);
    _mm256_storeu_ps(row + 8, hi);
  };
  add_row(c + 0 * ldc, c00, c01);
  add_row(c + 1 * ldc, c10, c11);
  add_row(c + 2 * ldc, c20, c21);
  add_row(c + 3 * ldc, c30, c31);
  add_row(c + 4 * ldc, c40, c41);
  add_row(c + 5 * ldc, c50, c51);
}

}  // namespace
