// Attention kernels built on the packed GEMM.
#include <algorithm>
#include <cmath>
#include "attention.hpp"
#include "gemm.hpp"
#include "gemm_packed.hpp"
#include "simd_math.hpp"
#include "util.hpp"

namespace {

// Computes output rows [q0, q0 + rows) of one head. Kp and Vp are its packed
// transposed keys and packed values. S holds the scores.
void attn_rows(int L, int d, const float* Q, const float* Kp, const float* Vp,
               float* O, int q0, int rows, float* S, float* inv_sum, bool stream) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  float* o = O + static_cast<std::size_t>(q0) * d;
  const int n = L;
  gemm_packed(rows, n, d, Q + static_cast<std::size_t>(q0) * d, d, nullptr, 0, S, L,
              {.stream_c = stream, .packed_b = Kp});
  for (int r = 0; r < rows; ++r) {
    float* s = S + static_cast<std::size_t>(r) * L;
    const int valid = n;
    inv_sum[r] = 1.0f / row_exp(s, valid, scale, row_max(s, valid) * scale);
    std::fill(s + valid, s + n, 0.0f);
  }
  gemm_packed(rows, d, n, S, L, nullptr, 0, o, d, {.packed_b = Vp});
  // Scaling the d outputs is cheaper than another pass over the L scores.
  for (int r = 0; r < rows; ++r) row_scale(o + static_cast<std::size_t>(r) * d, d, inv_sum[r]);
}

// Shared by attn_mat, attn_mat_stream and attn_qblock. qb is the number of
// query rows in one block.
void attn_query_blocked(int H, int L, int d, const float* Q, const float* K,
                        const float* V, float* O, int qb, bool stream) {
  const std::size_t head = static_cast<std::size_t>(L) * d;
  const std::size_t k_size = gemm_packed_b_size(L, d), v_size = gemm_packed_b_size(d, L);
  const int blocks = (L + qb - 1) / qb;
  static Scratch packed;
  float* const P = packed.get(H * (k_size + v_size));
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int h = 0; h < H; ++h) {
      gemm_pack_b(L, d, K + h * head, d, true, P + h * (k_size + v_size));
      gemm_pack_b(d, L, V + h * head, d, false, P + h * (k_size + v_size) + k_size);
    }
#pragma omp for collapse(2) schedule(dynamic)
    for (int h = 0; h < H; ++h) {
      for (int b = 0; b < blocks; ++b) {
        static thread_local Scratch scratch;
        const int q0 = b * qb, rows = std::min(qb, L - q0);
        float* const S = scratch.get(static_cast<std::size_t>(qb) * L + qb);
        const float* Kp = P + h * (k_size + v_size);
        attn_rows(L, d, Q + h * head, Kp, Kp + k_size, O + h * head, q0, rows, S,
                  S + static_cast<std::size_t>(qb) * L, stream);
      }
    }
  }
}

}  // namespace

void attn_mat(int H, int L, int d, const float* Q, const float* K,
              const float* V, float* O) {
  attn_query_blocked(H, L, d, Q, K, V, O, L, false);
}

void attn_mat_stream(int H, int L, int d, const float* Q, const float* K,
                     const float* V, float* O) {
  attn_query_blocked(H, L, d, Q, K, V, O, L, true);
}

void attn_qblock(int H, int L, int d, const float* Q, const float* K,
                 const float* V, float* O) {
  int qb = attn_params().qb;
  if (qb <= 0) qb = std::max(6, kScoreTileFloats / L);
  attn_query_blocked(H, L, d, Q, K, V, O, std::min(qb, L), false);
}

void attn_tiled(int H, int L, int d, const float* Q, const float* K,
                const float* V, float* O) {
  const int TQ = std::min(attn_params().tq, L), TK = std::min(attn_params().tk, L);
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  const std::size_t head = static_cast<std::size_t>(L) * d;
  const int blocks = (L + TQ - 1) / TQ, key_tiles = (L + TK - 1) / TK;
  const std::size_t k_size = gemm_packed_b_size(TK, d), v_size = gemm_packed_b_size(d, TK);
  const std::size_t slot = k_size + v_size;
  static Scratch packed;
  float* const P = packed.get(static_cast<std::size_t>(H) * key_tiles * slot);
#pragma omp parallel
  {
#pragma omp for collapse(2) schedule(static)
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < key_tiles; ++t) {
        const int k0 = t * TK, cols = std::min(TK, L - k0);
        float* p = P + (static_cast<std::size_t>(h) * key_tiles + t) * slot;
        gemm_pack_b(cols, d, K + h * head + static_cast<std::size_t>(k0) * d, d, true, p);
        gemm_pack_b(d, cols, V + h * head + static_cast<std::size_t>(k0) * d, d, false, p + k_size);
      }
    }
#pragma omp for collapse(2) schedule(dynamic)
    for (int h = 0; h < H; ++h) {
      for (int b = 0; b < blocks; ++b) {
        static thread_local Scratch scratch;
        float* const S = scratch.get(static_cast<std::size_t>(TQ) * TK + 2 * TQ);
        float* const run_max = S + static_cast<std::size_t>(TQ) * TK;
        float* const run_sum = run_max + TQ;

        const int q0 = b * TQ, rows = std::min(TQ, L - q0);
        const float* q = Q + h * head + static_cast<std::size_t>(q0) * d;
        float* o = O + h * head + static_cast<std::size_t>(q0) * d;
        std::fill(run_max, run_max + rows, -INFINITY);
        std::fill(run_sum, run_sum + rows, 0.0f);

        const int tiles = key_tiles;
        for (int t = 0; t < tiles; ++t) {
          const int k0 = t * TK, cols = std::min(TK, L - k0);
          const float* p = P + (static_cast<std::size_t>(h) * key_tiles + t) * slot;
          gemm_packed(rows, cols, d, q, d, nullptr, 0, S, cols, {.packed_b = p});
          for (int r = 0; r < rows; ++r) {
            float* s = S + static_cast<std::size_t>(r) * cols;
            const int valid = cols;
            std::fill(s + valid, s + cols, 0.0f);
            if (valid == 0) continue;
            const float new_max = std::max(run_max[r], row_max(s, valid) * scale);
            // Online softmax. Rescale the totals so far to the new maximum.
            if (t > 0 && new_max != run_max[r]) {
              const float alpha = std::exp(run_max[r] - new_max);
              run_sum[r] *= alpha;
              row_scale(o + static_cast<std::size_t>(r) * d, d, alpha);
            }
            run_max[r] = new_max;
            run_sum[r] += row_exp(s, valid, scale, new_max);
          }
          gemm_packed(rows, d, cols, S, cols, nullptr, 0, o, d,
                      {.accumulate = t > 0, .packed_b = p + k_size});
        }
        for (int r = 0; r < rows; ++r) {
          row_scale(o + static_cast<std::size_t>(r) * d, d, 1.0f / run_sum[r]);
        }
      }
    }
  }
}

namespace {
constexpr AttnKernel kKernels[] = {
    {"naive", attn_naive},
    {"mat", attn_mat},
    {"mat_stream", attn_mat_stream},
    {"qblock", attn_qblock},
    {"tiled", attn_tiled},
};
}  // namespace

std::span<const AttnKernel> attn_kernels() { return kKernels; }

AttnFn find_attn(std::string_view name) {
  for (const auto& k : kKernels) {
    if (name == k.name) return k.fn;
  }
  return nullptr;
}

AttnParams& attn_params() {
  static AttnParams params;
  return params;
}

bool attn_set_param(std::string_view name, int value) {
  AttnParams& p = attn_params();
  if (value <= 0) return false;
  if (name == "qb") p.qb = value;
  else if (name == "tq") p.tq = value;
  else if (name == "tk") p.tk = value;
  else return false;
  return true;
}
