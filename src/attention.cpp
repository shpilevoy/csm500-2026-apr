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

namespace {
constexpr AttnKernel kKernels[] = {
    {"naive", attn_naive},
    {"mat", attn_mat},
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
