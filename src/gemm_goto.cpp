// GEMM in the style of Goto and BLIS. Both operands are packed into buffers
// that fit the caches. One microkernel does all the arithmetic.
#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "gemm.hpp"
#include "gemm_packed.hpp"
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

// The same slice when B is given transposed.
void pack_bt_slice(int kc, int cols, const float* Bt, int ldb, float* out) {
  for (int c = 0; c < cols; ++c) {
    const float* col = Bt + c * ldb;
    for (int p = 0; p < kc; ++p) out[p * NR + c] = col[p];
  }
  for (int c = cols; c < NR; ++c) {
    for (int p = 0; p < kc; ++p) out[p * NR + c] = 0.0f;
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

// The block sizes in use. mc and nc are rounded to whole register tiles.
struct Blocks {
  int mc, kc, nc;
};
Blocks blocks() {
  const GemmParams& p = gemm_params();
  return {std::max(MR, p.mc / MR * MR), p.kc, std::max(NR, p.nc / NR * NR)};
}

// Packs the NR-column slice that starts at column jr of block (jc, pc) of B.
void pack_b_at(int nc, int kc, const float* B, int ldb, bool trans, int jc,
               int pc, int jr, float* block) {
  float* slice = block + static_cast<std::size_t>(jr) * kc;
  const int cols = std::min(NR, nc - jr);
  if (trans) {
    pack_bt_slice(kc, cols, B + (jc + jr) * ldb + pc, ldb, slice);
  } else {
    pack_b_slice(kc, cols, B + pc * ldb + jc + jr, ldb, slice);
  }
}

std::size_t packed_block_size(int nc, int kc) {
  return static_cast<std::size_t>((nc + NR - 1) / NR * NR) * kc;
}

}  // namespace

std::size_t gemm_packed_b_size(int N, int K) {
  return packed_block_size(N, K);
}

void gemm_pack_b(int N, int K, const float* B, int ldb, bool trans, float* out) {
  const Blocks blk = blocks();
  for (int jc = 0; jc < N; jc += blk.nc) {
    const int nc = std::min(blk.nc, N - jc);
    for (int pc = 0; pc < K; pc += blk.kc) {
      const int kc = std::min(blk.kc, K - pc);
      for (int jr = 0; jr < nc; jr += NR) pack_b_at(nc, kc, B, ldb, trans, jc, pc, jr, out);
      out += packed_block_size(nc, kc);
    }
  }
}

void gemm_packed(int M, int N, int K, const float* A, int lda, const float* B,
                 int ldb, float* C, int ldc, GemmPackedOpts opts) {
  const Blocks blk = blocks();
  const int MC = blk.mc, KC = blk.kc, NC = blk.nc;

  static thread_local Scratch b_scratch;
  const float* packed = opts.packed_b;
  float* const b_buf = packed ? nullptr : b_scratch.get(packed_block_size(std::min(NC, N), std::min(KC, K)));

  for (int jc = 0; jc < N; jc += NC) {
    const int nc = std::min(NC, N - jc);
    for (int pc = 0; pc < K; pc += KC) {
      const int kc = std::min(KC, K - pc);
      const float* const Bp = packed ? packed : b_buf;
      if (packed) packed += packed_block_size(nc, kc);
      const bool store = !opts.accumulate && pc == 0;
      // Non-temporal stores need an aligned C. They help only when a single block
      // overwrites C.
      const bool stream = store && opts.stream_c && K <= KC &&
                          reinterpret_cast<std::uintptr_t>(C) % 32 == 0 && ldc % 8 == 0 && jc % 8 == 0;
      const Write full_tile = stream ? Write::Stream : store ? Write::Store : Write::Add;
      {
        if (!opts.packed_b) {
          for (int jr = 0; jr < nc; jr += NR) {
            pack_b_at(nc, kc, B, ldb, opts.trans_b, jc, pc, jr, b_buf);
          }
        }

        // Rows are split evenly across threads in whole MR tiles. The split
        // then stays balanced when M is only a few multiples of MC.
        const int tiles = (M + MR - 1) / MR;
        const int t = 0, T = 1;
        const int i_begin = static_cast<int>(static_cast<long>(tiles) * t / T) * MR;
        const int i_end = std::min(M, static_cast<int>(static_cast<long>(tiles) * (t + 1) / T) * MR);

        static thread_local Scratch a_scratch;
        float* const Ap = a_scratch.get(static_cast<std::size_t>(MC) * KC);

        for (int ic = i_begin; ic < i_end; ic += MC) {
          const int mc = std::min(MC, i_end - ic);
          pack_a(mc, kc, A + ic * lda + pc, lda, Ap);
          for (int jr = 0; jr < nc; jr += NR) {
            const float* b = Bp + static_cast<std::size_t>(jr) * kc;
            const int cols = std::min(NR, nc - jr);
            for (int ir = 0; ir < mc; ir += MR) {
              const float* a = Ap + static_cast<std::size_t>(ir) * kc;
              const int rows = std::min(MR, mc - ir);
              float* c = C + (ic + ir) * ldc + jc + jr;
              if (rows == MR && cols == NR) {
                ukernel(kc, a, b, c, ldc, full_tile);
              } else {
                alignas(32) float tile[MR * NR];
                ukernel(kc, a, b, tile, NR, Write::Store);
                for (int r = 0; r < rows; ++r) {
                  for (int q = 0; q < cols; ++q) {
                    c[r * ldc + q] = store ? tile[r * NR + q] : c[r * ldc + q] + tile[r * NR + q];
                  }
                }
              }
            }
          }
        }
        if (stream) _mm_sfence();  // make the streamed lines of this thread visible
      }
    }
  }
}

void gemm_goto(int M, int N, int K, const float* A, int lda, const float* B,
               int ldb, float* C, int ldc) {
  gemm_packed(M, N, K, A, lda, B, ldb, C, ldc, {});
}
